"""Execute prepared coefficient windows with invocation-owned prefix storage."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch._higher_order_ops.associative_scan import associative_scan

from pllm.contracts.tensor_policy import ScanForm
from pllm.execution.scan_analysis import ScanAnalysis, stage
from pllm.kernels.additive_scan import launch_prefix as additive_prefix
from pllm.kernels.right_matrix_scan import MatrixPrefixBuffers, launch_prefix as matrix_prefix

Step = Callable[..., tuple[torch.Tensor, torch.Tensor]]


def competitive(width: int, chunk: int, window: int, form: ScanForm) -> bool:
    """Both matrix work guards, independently of measured host or device latency."""
    depth = window if form is ScanForm.ORDERED else 2 * (window.bit_length() - 1) + 1
    return width <= chunk and depth * width <= window * chunk


def diagonal_prefix(gate: torch.Tensor, bias: torch.Tensor, carry: torch.Tensor) -> torch.Tensor:
    """Out-of-place associative doubling, including odd windows and inactive identities."""
    a, b = gate, bias
    offset = 1
    while offset < len(a):
        a, b = (
            torch.cat((a[:offset], a[:-offset] * a[offset:])),
            torch.cat((b[:offset], b[:-offset] * a[offset:] + b[offset:])),
        )
        offset *= 2
    return a * carry + b


def _compose(
    earlier: tuple[torch.Tensor, torch.Tensor], later: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    a, b = earlier
    c, d = later
    return a * c, b * c + d


def associative_prefix(gate: torch.Tensor, bias: torch.Tensor, carry: torch.Tensor) -> torch.Tensor:
    """PyTorch's pointwise affine prefix, compiled during cold CUDA preparation."""
    a, b = associative_scan(_compose, (gate, bias), dim=0, combine_mode="pointwise")
    return a * carry + b


@dataclass(frozen=True)
class PreparedScan:
    """A selected algorithm and its funded envelope, without retained mutable tensors."""

    step: Step
    chunk: int
    analysis: ScanAnalysis | None = None
    window: int = 1
    form: ScanForm = ScanForm.ORDERED
    workspace: int = 0
    pointwise: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = diagonal_prefix

    def __call__(
        self,
        carry: torch.Tensor,
        over: tuple[torch.Tensor, ...],
        invariants: tuple[torch.Tensor, ...],
        valid_rows: torch.Tensor | None,
        *,
        tracing: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if self.analysis is None:
            outputs: list[torch.Tensor] = []
            for start in range(0, len(over[0]), self.chunk):
                updated, output = self.step(carry, *(t[start : start + self.chunk] for t in over), *invariants)
                carry = updated if valid_rows is None else torch.where(start + self.chunk <= valid_rows, updated, carry)
                outputs.append(output)
            return carry, tuple(outputs)
        return self.windows(carry, over, invariants, valid_rows, tracing=tracing)

    def windows(
        self,
        carry: torch.Tensor,
        over: tuple[torch.Tensor, ...],
        invariants: tuple[torch.Tensor, ...],
        valid_rows: torch.Tensor | None,
        *,
        tracing: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        assert self.analysis is not None
        count = len(over[0]) // self.chunk
        outputs: list[torch.Tensor] = []
        for start in range(0, count, self.window):
            size = min(self.window, count - start)
            packed = tuple(
                t[start * self.chunk : (start + size) * self.chunk].reshape(size, self.chunk, *t.shape[1:])
                for t in over
            )
            packed = tuple(stage(t, order, self.window) for t, order in zip(packed, self.analysis.orders, strict=True))
            coefficient, bias, *cuts = self.analysis.coefficients(carry, *packed, *invariants)
            if valid_rows is None:
                active = torch.ones(self.window, device=carry.device, dtype=torch.bool)
                active[size:].fill_(False)
            else:
                index = torch.arange(self.window, device=carry.device) + start
                active = (index < count) & ((index + 1) * self.chunk <= valid_rows)
            states, carry = self.prefix(carry, coefficient, bias, active, tracing=tracing)
            readout = self.analysis.readout(states, *cuts)[:size]
            if self.analysis.copy_readout or size != self.window:
                readout = readout.clone()
            outputs.extend(readout.unbind())
        return carry, tuple(outputs)

    def prefix(
        self,
        carry: torch.Tensor,
        coefficient: torch.Tensor,
        bias: torch.Tensor,
        active: torch.Tensor,
        *,
        tracing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.analysis is not None
        bias = torch.broadcast_to(bias, (self.window, *carry.shape))
        mask = active.reshape(self.window, *(1 for _ in carry.shape))
        if self.analysis.kind == "additive":
            states = carry.new_empty((self.window, *carry.shape))
            final = torch.empty_like(carry)
            matrix = carry.view(-1, carry.shape[-1] if carry.ndim else 1)
            increments = bias.reshape(self.window, *matrix.shape)
            if not tracing:
                additive_prefix(
                    increments,
                    matrix,
                    active,
                    states.view(self.window, *matrix.shape),
                    final.view_as(matrix),
                    self.analysis.scale,
                    self.form,
                )
            return states, final
        bias = torch.where(mask, bias, 0)
        if self.analysis.kind == "diagonal":
            coefficient = coefficient.reshape(
                (self.window,) + (1,) * (carry.ndim + 1 - coefficient.ndim) + coefficient.shape[1:]
            )
            gate = torch.where(mask, coefficient, 1)
            inclusive = self.pointwise(torch.broadcast_to(gate, bias.shape), bias, carry)
            states = torch.cat((carry.unsqueeze(0), inclusive[:-1]))
            return states, torch.where(active.any(), inclusive[-1], carry)
        left = self.analysis.kind == "left"
        state = carry.transpose(-1, -2).contiguous() if left else carry
        buffers = MatrixPrefixBuffers(state, self.window)
        aa = coefficient.transpose(-1, -2) if left else coefficient
        bb = bias.transpose(-1, -2) if left else bias
        eye = torch.eye(state.shape[-1], dtype=state.dtype, device=state.device)
        buffers.a.copy_(torch.where(mask, aa, eye))
        buffers.b.copy_(bb)
        matrix_prefix(state, active, buffers, self.form)
        states, final = buffers.states[:-1], buffers.final
        return (
            (states.transpose(-1, -2).contiguous(), final.transpose(-1, -2).contiguous()) if left else (states, final)
        )
