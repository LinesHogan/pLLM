"""CPU adaptive-depth contracts: compact execution against a rowwise masked decoder oracle."""

from collections.abc import Callable
from dataclasses import replace

import torch

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import TokenProgramEffect
from pllm.execution.custom_forward import SerialCustomForward
from pllm.execution.token_program import KV, Segment, TokenProgram
from pllm.execution.token_program_executor import execute_segment, execute_token_program


class _Decoder(SerialCustomForward):
    """Small deterministic causal decoder with distinct layer K/V and explicit slot ownership."""

    def __init__(self, program: TokenProgram, layers: int, hidden: int, dtype: torch.dtype) -> None:
        self.program = program
        self.cache = torch.full((3, layers, 4, 2, hidden), float("nan"), dtype=dtype)
        self.seed = torch.sin(torch.arange(3 * hidden).reshape(3, hidden).float() * 0.17).to(dtype)
        self.batch: tuple[PackedPhaseRow, ...] = ()
        self.indices = torch.arange(3)
        self.offset = 0
        self.output = self.seed
        self.decisions: list[torch.Tensor] = []

    @property
    def rows(self) -> tuple[PackedPhaseRow, ...]:
        return self.batch

    def stage(self, order: tuple[int, ...], position: int) -> None:
        self.offset = position
        self.indices = torch.tensor(order)
        self.batch = tuple(
            PackedPhaseRow(index, request, position, HiddenPhase.DECODE, RequestIdentity(str(request), request, 0))
            for index, request in enumerate(order)
        )
        self.decisions = []

    def begin(self, row: PackedPhaseRow) -> tuple[torch.Tensor, torch.Tensor]:
        self.indices = torch.tensor([row.request_ordinal])
        return torch.tensor([self.offset]), self.seed[row.request_ordinal : row.request_ordinal + 1].clone()

    def project(self, ordinal: int, hidden: torch.Tensor) -> torch.Tensor:
        norm = hidden.float() * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + 1e-5)
        key = norm * (0.13 + ordinal * 0.01) + self.offset * 0.03
        value = torch.roll(norm, 1, -1) * (0.21 + ordinal * 0.02)
        return torch.stack((key, value), dim=1).to(hidden.dtype)

    def batch_layer(
        self, ordinal: int, hidden: torch.Tensor, residual: torch.Tensor | None, kv: KV
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stream = hidden if residual is None else hidden + residual
        projected = self.project(ordinal, stream)
        if kv == "commit":
            self.cache[self.indices, ordinal, self.offset] = projected
        previous = (
            self.cache[self.indices, ordinal, : self.offset].sum(1) if self.offset else torch.zeros_like(projected)
        )
        branch = (projected[:, 0] + projected[:, 1] + 0.1 * previous[:, 1]).tanh()
        return branch, stream.clone()

    def layer(
        self,
        row: PackedPhaseRow,
        ordinal: int,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        kv: KV = "commit",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.batch_layer(ordinal, hidden, residual, kv)

    def propagate(self, row: PackedPhaseRow, ordinal: int, positions: torch.Tensor, hidden: torch.Tensor) -> None:
        self.cache[row.request_ordinal, ordinal, self.offset] = self.project(ordinal, hidden)[0]

    def masked_segment(
        self, row: PackedPhaseRow, positions: torch.Tensor, step: Segment, hidden: torch.Tensor, active: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        self.decisions.append(active.clone())
        return super().masked_segment(row, positions, step, hidden, active)

    def compact(self, step: Segment, hidden: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, ...]:
        self.decisions.append(active.clone())
        output = tuple(hidden.clone() for _ in step.taps)
        count = int(active.sum())  # CPU oracle only; CUDA never uses this executor.
        if count == len(hidden):
            return execute_segment(step, hidden, self.batch_layer)
        for ordinal in step.propagate:
            projected = self.project(ordinal, hidden)
            skipped = ~active.flatten()
            self.cache[self.indices[skipped], ordinal, self.offset] = projected[skipped]
        if count:
            rows = torch.argsort(~active.flatten(), stable=True)[:count]
            indices = self.indices
            self.indices = indices.index_select(0, rows)
            computed = execute_segment(step, hidden.index_select(0, rows), self.batch_layer)
            self.indices = indices
            for destination, value in zip(output, computed, strict=True):
                destination.index_copy_(0, rows, value)
        return output

    def normalize(self, row: PackedPhaseRow, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return hidden + residual

    def commit(self, row: PackedPhaseRow, record: Callable[[], None]) -> None:
        record()

    def finish(self, outputs: tuple[torch.Tensor, ...]) -> None:
        self.output = torch.cat(outputs)

    def run_compacted(self) -> None:
        self.output, _ = execute_token_program(
            self.program,
            self.seed.index_select(0, self.indices),
            torch.full((len(self.indices),), self.offset),
            torch.ones(len(self.indices), 1, dtype=torch.bool),
            (),
            self.batch_layer,
            lambda: None,
            adaptive=self.compact,
        )


def check_depth(program: TokenProgram, layers: int, hidden: int, dtype: torch.dtype) -> tuple[str, ...]:
    """Compare output, decisions and every KV slot across compaction, masking and row schedules."""
    reference = _Decoder(program, layers, hidden, dtype)
    observations: list[tuple[torch.Tensor, tuple[torch.Tensor, ...]]] = []
    for order in ((0, 1, 2), (2, 1, 0), (0,), (0, 2, 1)):
        compact, masked = (_Decoder(program, layers, hidden, dtype) for _ in range(2))
        if order == (0, 2, 1):
            compact.seed[1:] *= -7
            masked.seed[1:] *= -7
        for position in range(4):
            compact.stage(order, position)
            masked.stage(order, position)
            compact.run_compacted()
            masked.run({"decode": program}, TokenProgramEffect(layers, "decode"), lambda index, value: None)
            torch.testing.assert_close(compact.output, masked.output)
            decisions = len(compact.decisions)
            for index, active in enumerate(compact.decisions):
                torch.testing.assert_close(active, torch.cat(masked.decisions[index::decisions]), rtol=0, atol=0)
            selected = torch.tensor(order)
            cache = compact.cache[selected, :, : position + 1]
            missing = torch.isnan(cache).any(-1).any(-1).nonzero()
            if len(missing):
                row, layer, token = (int(value) for value in missing[0])
                raise AssertionError(f"KV hole for request {order[row]}, layer {layer}, token {token}")
            torch.testing.assert_close(cache, masked.cache[selected, :, : position + 1])
            if order == (0, 1, 2):
                reference.cache.copy_(compact.cache)
                observations.append((compact.output.clone(), tuple(value.clone() for value in compact.decisions)))
            else:
                target = selected[:1] if order == (0, 2, 1) else selected
                compared = len(target)
                torch.testing.assert_close(cache[:compared], reference.cache[target, :, : position + 1])
                outputs, decisions = observations[position]
                torch.testing.assert_close(compact.output[:compared], outputs[target])
                for value, expected in zip(compact.decisions, decisions, strict=True):
                    torch.testing.assert_close(value[:compared], expected[target], rtol=0, atol=0)
    return ("adaptive compaction = masked oracle", "adaptive row permutation / isolation", "adaptive KV policy")


def check_never_exit(program: TokenProgram, layers: int, hidden: int, dtype: torch.dtype) -> None:
    """Force every active decision true and compare bitwise with the same slices without active=."""
    plain = replace(
        program,
        steps=tuple(
            replace(step, active=None, propagate=()) if isinstance(step, Segment) else step for step in program.steps
        ),
    )
    compact, vanilla = (_Decoder(item, layers, hidden, dtype) for item in (program, plain))

    def all_active(step: Segment, seed: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return compact.compact(step, seed, torch.ones_like(active))

    for position in range(4):
        compact.stage((0, 1, 2), position)
        vanilla.stage((0, 1, 2), position)
        output, _ = execute_token_program(
            program,
            compact.seed,
            torch.full((3,), position),
            torch.ones(3, 1, dtype=torch.bool),
            (),
            compact.batch_layer,
            lambda: None,
            adaptive=all_active,
        )
        vanilla.run_compacted()
        torch.testing.assert_close(output, vanilla.output, rtol=0, atol=0)
        torch.testing.assert_close(
            compact.cache[:, :, : position + 1], vanilla.cache[:, :, : position + 1], rtol=0, atol=0
        )
