"""Projection geometry checks at probing and the eager Exact projection-task driver."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pllm.backends.vllm.architecture import Decoder, LinearOutput, Mlp, Projection
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch
from pllm.contracts.parameter_overlay import (
    CapturedParameterOperand,
    ParameterOverlayFieldBinding,
    ParameterOverlayOperator,
    ParameterTarget,
    ParameterTargetShape,
)
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.state_ref import StateRef
from pllm.execution.cuda_stream import PinnedStaging
from pllm.execution.overlay_task import ProjectionTransition, RouteOverlay
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.state_execution import DataT, StateTaskContext


def inspect_parameter_targets(
    decoder: Decoder, targets: tuple[ParameterTarget, ...], dtype: torch.dtype
) -> tuple[ParameterTargetShape, ...]:
    specs: list[ParameterTargetShape] = []
    for index, target in enumerate(targets):
        module, offset, size, _ = decoder.projection(target)
        if (
            target in targets[:index]
            or module.tp_size != 1
            or not module.return_bias
            or module.weight.dtype is not dtype
            or offset + size > module.weight.shape[0]
        ):
            raise ValueError("parameter overlay requires unique, exact unsharded projections matching model dtype")
        specs.append(ParameterTargetShape(target, int(module.weight.shape[1]), size))
    return tuple(specs)


@dataclass(frozen=True, slots=True)
class _PreparedRows:
    source: PackedPhaseRow
    ref: StateRef
    rows: torch.Tensor
    binding: ParameterOverlayFieldBinding
    offset: int
    size: int
    prompt_end: torch.Tensor


class _ExactProjection:
    def __init__(self, module: Projection, mlp: Mlp | None) -> None:
        self.module, self.mlp = module, mlp
        self.bindings: list[tuple[ParameterOverlayFieldBinding, int, int]] = []
        self.original_forward = module.forward
        self.original_mlp_forward = None if mlp is None else mlp.forward
        self.prepared: tuple[_PreparedRows, ...] = ()
        self.captures: tuple[CapturedParameterOperand, ...] = ()
        self.mlp_input: torch.Tensor | None = None
        self.total_rows = 0
        self.apply_count = self.updated_apply_count = 0
        self.transition: ProjectionTransition | None = None
        self.routes: tuple[RouteOverlay, ...] = ()

    def captures_binding(self, binding: ParameterOverlayFieldBinding) -> bool:
        return self.original_mlp_forward is not None or binding.operator is ParameterOverlayOperator.HEAD_ADD

    def forward_mlp(self, value: torch.Tensor) -> torch.Tensor:
        assert self.original_mlp_forward is not None
        self.mlp_input = value.detach()
        try:
            return self.original_mlp_forward(value)
        finally:
            self.mlp_input = None

    def forward(self, hidden: torch.Tensor) -> LinearOutput:
        base = self.original_forward(hidden)
        if not self.prepared:
            return base
        assert hidden.shape[0] == self.total_rows and not self.captures
        # The unquantized TP1 projection owns this fresh output, not an input or state page.
        match base:
            case (output, *_):
                pass
            case output:
                pass
        # A replay span can contain both phases of one request. Finish its prompt
        # transition (including prompt-end flush) before reading its decode state.
        phases = tuple(
            phase
            for phase in (HiddenPhase.PREFILL, HiddenPhase.DECODE)
            if any(item.source.phase is phase for item in self.prepared)
        )
        for phase in phases:
            prepared = tuple(item for item in self.prepared if item.source.phase is phase)
            self.forward_segment(hidden, output, prepared, complete=phase is phases[-1])
        return base

    def forward_segment(
        self, hidden: torch.Tensor, output: torch.Tensor, prepared: tuple[_PreparedRows, ...], *, complete: bool
    ) -> None:
        inputs = tuple(hidden.index_select(0, item.rows) for item in prepared)
        captures = tuple(
            self.capture(item, selected)
            for item, selected in zip(prepared, inputs, strict=True)
            if self.captures_binding(item.binding)
        )
        if self.routes:
            residuals = {
                key: value
                for route in self.routes
                for key, value in route.run(
                    tuple(
                        operand
                        for operand in captures
                        if operand.identity == route.data.step.rows.rows[route.row].identity
                    )
                ).items()
            }
        else:
            assert self.transition is not None
            residuals = self.transition.run(captures, complete=complete)
        for item, selected in zip(prepared, inputs, strict=True):
            delta = (residuals[item.source.identity, item.binding.target] * item.binding.scale).to(output.dtype)
            columns = output[:, item.offset : item.offset + item.size]
            columns.index_add_(0, item.rows, delta)
            self.apply_count += selected.shape[0]
            if item.ref.version_id > 0:
                self.updated_apply_count += selected.shape[0]
        if not self.routes:
            self.captures += captures

    def capture(self, item: _PreparedRows, selected: torch.Tensor) -> CapturedParameterOperand:
        if item.binding.operator is ParameterOverlayOperator.HEAD_ADD:
            desired = selected
        else:
            assert self.mlp_input is not None and self.mlp_input.shape[0] == self.total_rows
            desired = self.mlp_input.index_select(0, item.rows)
        return CapturedParameterOperand(
            item.source.identity,
            item.binding.target,
            item.source.phase,
            selected.detach(),
            desired.detach(),
            self.module.weight[item.offset : item.offset + item.size].detach(),
            item.prompt_end,
        )

    def clear(self) -> None:
        self.prepared, self.captures = (), ()
        self.mlp_input = None
        self.total_rows = 0
        self.transition = None
        self.routes = ()


class ExactParameterOverlay:
    """Eager Exact projection-task driver; pinned reads use the shared tensor op."""

    def __init__(self, decoder: Decoder, bindings: tuple[ParameterOverlayFieldBinding, ...]) -> None:
        modules: dict[int, _ExactProjection] = {}
        for binding in bindings:
            module, offset, size, mlp = decoder.projection(binding.target)
            if id(module) not in modules:
                modules[id(module)] = _ExactProjection(module, mlp)
            modules[id(module)].bindings.append((binding, offset, size))
        self._modules = tuple(modules.values())
        self._staging = PinnedStaging()
        for handle in self._modules:
            handle.module.forward = handle.forward
            if handle.mlp is not None:
                handle.mlp.forward = handle.forward_mlp

    @property
    def apply_count(self) -> int:
        return sum(handle.apply_count for handle in self._modules)

    @property
    def updated_apply_count(self) -> int:
        return sum(handle.updated_apply_count for handle in self._modules)

    def prepare(
        self,
        rows: PackedPhaseRowBatch,
        context: StateTaskContext[DataT],
        directory: RequestStateDirectory,
        transition: ProjectionTransition | None = None,
    ) -> None:
        assert transition is not None
        grouped: dict[tuple[RequestIdentity, HiddenPhase], list[PackedPhaseRow]] = {}
        for row in rows.rows:
            grouped.setdefault((row.identity, row.phase), []).append(row)
        table = self._modules[0].module.weight.new_empty((len(rows.rows) + len(grouped),), dtype=torch.long)
        host = self._staging.take(table.numel()).numpy()
        host[: len(rows.rows)] = [row.packed_row_index for source in grouped.values() for row in source]
        host[len(rows.rows) :] = [
            source[-1].token_offset == source[-1].prompt_length - 1 for source in grouped.values()
        ]
        self._staging.copy_to(table)
        indices = torch.split(table[: len(rows.rows)], [len(source) for source in grouped.values()])
        prompt_ends = table[len(rows.rows) :].bool().unbind()
        requests = tuple(
            (source[0], context.selection.ref_for(identity), index, prompt_end)
            for ((identity, _), source), index, prompt_end in zip(grouped.items(), indices, prompt_ends, strict=True)
        )
        for handle in self._modules:
            handle.transition = transition
            handle.prepared = tuple(
                _PreparedRows(source, ref, index, binding, offset, size, prompt_end)
                for source, ref, index, prompt_end in requests
                for binding, offset, size in handle.bindings
            )
            handle.total_rows = rows.total_rows

    def prepare_wave(
        self, rows: tuple[PackedPhaseRow, ...], routes: tuple[RouteOverlay, ...], indices: tuple[torch.Tensor, ...]
    ) -> None:
        """Use one projection call across the wave and request-local update operands."""
        for row, route, index in zip(rows, routes, indices, strict=True):
            route.prompt_end.fill_(row.token_offset + index.numel() == row.prompt_length)
        for handle in self._modules:
            handle.routes = routes
            handle.prepared = tuple(
                _PreparedRows(
                    row,
                    route.batch.after_refs[route.batch.selection.state_indices[route.row]],
                    index,
                    binding,
                    offset,
                    size,
                    route.prompt_end,
                )
                for row, route, index in zip(rows, routes, indices, strict=True)
                for binding, offset, size in handle.bindings
            )
            handle.captures, handle.total_rows = (), sum(index.numel() for index in indices)

    def take_captures(self) -> tuple[CapturedParameterOperand, ...]:
        for handle in self._modules:
            if handle.routes:
                continue
            expected = sum(handle.captures_binding(item.binding) for item in handle.prepared)
            assert len(handle.captures) == expected
            handle.captures = ()
        return ()

    def clear(self) -> None:
        for handle in self._modules:
            handle.clear()

    def close(self) -> None:
        self.clear()
        for handle in self._modules:
            handle.module.forward = handle.original_forward
            if handle.mlp is not None and handle.original_mlp_forward is not None:
                handle.mlp.forward = handle.original_mlp_forward
