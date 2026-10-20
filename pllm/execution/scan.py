"""Public chunk recurrences and the execution context that specializes them."""

from collections.abc import Callable
from contextvars import ContextVar
from functools import cached_property, lru_cache
from typing import TypeVar

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import _disable_current_modes  # pyright: ignore[reportPrivateUsage]

from pllm.contracts.region_program import SizeParameter
from pllm.contracts.region_selection import RegionSelectionPolicy
from pllm.execution.exported_cell import TensorCellModule
from pllm.execution.exported_region import import_exported_region
from pllm.execution.memory import MemoryTrace
from pllm.execution.region_streams import RegionStreamBindings
from pllm.execution.tensor_scan import TensorScan

Step = Callable[..., tuple[torch.Tensor, torch.Tensor]]
R = TypeVar("R")


class ScanContext:
    """An operation's region policy, row capacity and CUDA lanes."""

    active: ContextVar["ScanContext | None"] = ContextVar("scan_context", default=None)

    def __init__(self, policy: RegionSelectionPolicy, max_rows: int, device: torch.device) -> None:
        self.policy, self.max_rows, self.device = policy, max_rows, device

    def bind(self, function: Callable[..., R]) -> Callable[..., R]:
        """Use the same scope for memory tracing and execution."""

        def run(*args: object, **kwargs: object) -> R:
            token = self.active.set(self)
            try:
                return function(*args, **kwargs)
            finally:
                self.active.reset(token)

        return run

    @cached_property
    def streams(self) -> tuple[torch.cuda.Stream, ...]:
        return (torch.cuda.Stream(device=self.device),)

    @lru_cache(maxsize=128)
    def specialize(
        self,
        step: Step,
        specs: tuple[tuple[torch.Size, tuple[int, ...], torch.dtype, torch.device], ...],
        invariants: int,
        chunk: int,
    ) -> TensorScan:
        """Export once per callable and guarded layout; all other calls are cache lookups."""

        def ordered(carry: torch.Tensor, *args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return step(carry, *args[invariants:], *args[:invariants])

        # Closure/owner constants belong to the callable: changing them requires a new step object.
        with _disable_current_modes(), FakeTensorMode():
            samples = tuple(
                torch.empty_strided(shape, stride, dtype=dtype, device=device) for shape, stride, dtype, device in specs
            )
            region = import_exported_region(
                torch.export.export(TensorCellModule(ordered), samples, strict=False), carry=1
            )
            scan = TensorScan(
                region.program,
                SizeParameter("chunks", 0, (self.max_rows + chunk - 1) // chunk),
                carry=1,
                invariants=invariants,
                functions=region.functions,
            )
        if samples[0].is_meta:
            return scan
        if self.policy.tuning == "heuristic":
            return scan.prepared(options=self.policy, parallel=True)
        samples = tuple(
            torch.empty_strided(shape, stride, dtype=dtype, device=device).zero_()
            for shape, stride, dtype, device in specs
        )
        return scan.optimized(
            samples[:1],
            samples[1 : 1 + invariants],
            tuple((s,) * min(3, (self.max_rows + chunk - 1) // chunk) for s in samples[1 + invariants :]),
            options=self.policy,
            streams=RegionStreamBindings(torch.cuda.current_stream(self.device), self.streams),
        )[0]


_context = lru_cache(maxsize=128)(ScanContext)


def scan(
    step: Step,
    *,
    carry: torch.Tensor,
    over: tuple[torch.Tensor, ...],
    chunk: int,
    invariants: tuple[torch.Tensor, ...] = (),
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Run ``step(carry, *slices, *invariants) -> (next_carry, output)`` in chunk order.

    Return the final carry and an ordered tuple of outputs (empty for zero rows).
    Inputs are immutable; carry retains its shape, stride, dtype and device.
    Reuse the same callable with immutable closure/owner constants to reuse its
    specialization; pass varying tensors through invariants. CPU, partial chunks
    and differentiable calls execute serially. Bound operations supply their plan policy.

        def step(total, rows, scale):
            next_total = total + rows.sum(0)
            return next_total, rows * scale
        total, outputs = pllm.scan(step, carry=total, over=(rows,), chunk=32, invariants=(scale,))
        result = torch.cat(outputs)
    """
    if chunk < 1 or not over or any(t.ndim == 0 or t.shape[0] != over[0].shape[0] for t in over):
        raise ValueError("scan needs a positive chunk and aligned row tensors")
    context = ScanContext.active.get() or _context(RegionSelectionPolicy(), over[0].shape[0], carry.device)
    if (
        context.device.type == "cuda"
        and over[0].shape[0]
        and over[0].shape[0] % chunk == 0
        and not (torch.is_grad_enabled() and any(t.requires_grad for t in (carry, *over, *invariants)))
    ):
        columns = tuple(tuple(torch.ops.aten.split.Tensor(t, chunk)) for t in over)
        compiled = context.specialize(
            step,
            tuple((t.shape, t.stride(), t.dtype, t.device) for t in (carry, *invariants, *(c[0] for c in columns))),
            len(invariants),
            chunk,
        )
        if carry.is_meta:
            MemoryTrace.reserve(sum(limit.bytes for limit in compiled.payload_limits))
        else:
            result = compiled(
                (carry,),
                invariants,
                columns,
                streams=RegionStreamBindings(torch.cuda.current_stream(context.device), context.streams)
                if compiled.parallel
                else None,
            )
            return result.carry[0], result.columns[0]
    outputs: list[torch.Tensor] = []
    for start in range(0, over[0].shape[0], chunk):
        carry, output = step(carry, *(t[start : start + chunk] for t in over), *invariants)
        outputs.append(output)
    return carry, tuple(outputs)
