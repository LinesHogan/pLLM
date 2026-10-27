"""Public chunk recurrences and the execution context that specializes them."""

from collections.abc import Callable
from contextvars import ContextVar
from functools import cached_property, lru_cache
from typing import TypeVar

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import _disable_current_modes  # pyright: ignore[reportPrivateUsage]

from pllm.contracts.region_program import SizeParameter
from pllm.contracts.tensor_policy import TensorPolicy
from pllm.execution.exported_cell import TensorCellModule
from pllm.execution.exported_region import ExportedRegion, import_exported_region
from pllm.execution.memory import MemoryTrace
from pllm.execution.region_streams import RegionStreamBindings
from pllm.execution.tensor_scan import TensorScan

Step = Callable[..., tuple[torch.Tensor, torch.Tensor]]
R = TypeVar("R")
Specs = tuple[tuple[torch.Size, tuple[int, ...], torch.dtype, torch.device], ...]


@lru_cache(maxsize=128)
def _export_step(step: Step, specs: Specs, invariants: int, counted: bool) -> ExportedRegion:
    """Export one source layout independently of policy and chunk-count buckets."""

    def ordered(carry: torch.Tensor, *args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = args[invariants:-1] if counted else args[invariants:]
        updated, output = step(carry, *sequences, *args[:invariants])
        return (torch.where(args[-1], updated, carry) if counted else updated), output

    with _disable_current_modes(), FakeTensorMode():
        samples = tuple(
            torch.empty_strided(shape, stride, dtype=dtype, device=device) for shape, stride, dtype, device in specs
        )
        return import_exported_region(torch.export.export(TensorCellModule(ordered), samples, strict=False), carry=1)


class ScanContext:
    """An operation's region policy, row capacity and CUDA lanes."""

    active: ContextVar["ScanContext | None"] = ContextVar("scan_context", default=None)

    def __init__(
        self, policy: TensorPolicy, max_rows: int, device: torch.device, payload_bytes: int | None = None
    ) -> None:
        self.policy, self.max_rows, self.device = policy, max_rows, device
        self.payload_bytes = payload_bytes

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
    def _template(self, step: Step, specs: Specs, invariants: int, chunk: int, counted: bool) -> "ScanTemplate":
        return ScanTemplate(self, _export_step(step, specs, invariants, counted), specs, invariants, chunk)

    def specialize(
        self, step: Step, specs: Specs, invariants: int, chunk: int, count: int, counted: bool = False
    ) -> TensorScan:
        """Reuse source export and one prepared plan per bounded count bucket."""
        tracing = specs[0][3].type == "meta"
        layouts = tuple(
            (shape, stride, dtype, self.device if device.type == "meta" else device)
            for shape, stride, dtype, device in specs
        )
        return self._template(step, layouts, invariants, chunk, counted).for_count(count, tracing=tracing)


class ScanTemplate:
    """Execution-owned preparation caches; no invocation tensors or carry state."""

    def __init__(self, context: ScanContext, region: ExportedRegion, specs: Specs, invariants: int, chunk: int) -> None:
        self.context, self.region, self.specs = context, region, specs
        self.invariants = invariants
        self.maximum = (context.max_rows + chunk - 1) // chunk + 1
        self.sources: dict[int, TensorScan] = {}
        self.plans: dict[int, TensorScan] = {}

    def for_count(self, count: int, *, tracing: bool) -> TensorScan:
        """Tune at the first real C in a bucket, then reuse for the whole guarded domain."""
        capacity = min(self.maximum, 1 << (max(1, count) - 1).bit_length())
        if not self.context.policy.allow_numerical_transforms:
            capacity = self.maximum  # Only the reference exists; one plan covers every C.
        if capacity not in self.sources:
            with _disable_current_modes():
                self.sources[capacity] = self._source(capacity)
        scan = self.sources[capacity]
        if tracing:
            return scan
        if capacity not in self.plans:
            self.plans[capacity] = self._prepare(scan, count)
        return self.plans[capacity]

    def _source(self, capacity: int) -> TensorScan:
        scan = TensorScan(
            self.region.program,
            SizeParameter("chunks", 0, capacity),
            carry=1,
            invariants=self.invariants,
            functions=self.region.functions,
        )
        if self.context.policy.allow_numerical_transforms:
            from pllm.execution.affine_scan import payload_budget

            scan.payload_limits = payload_budget(
                scan.program,
                self.region.functions,
                scan.payload_limits,
                self.context.payload_bytes,
            )
        return scan

    def _prepare(self, scan: TensorScan, count: int) -> TensorScan:
        context = self.context
        if context.policy.tuning == "heuristic":
            return scan.prepared(options=context.policy, parallel=True)
        samples = tuple(
            torch.empty_strided(shape, stride, dtype=dtype, device=device).zero_()
            for shape, stride, dtype, device in self.specs[: 1 + self.invariants]
        )
        columns = tuple(
            tuple(
                torch.empty_strided(
                    (count, *shape), (shape[0] * stride[0] if shape else 1, *stride), dtype=dtype, device=device
                )
                .zero_()
                .unbind()
            )
            for shape, stride, dtype, device in self.specs[1 + self.invariants :]
        )
        return scan.optimized(
            samples[:1],
            samples[1 : 1 + self.invariants],
            columns,
            options=context.policy,
            streams=RegionStreamBindings(torch.cuda.current_stream(context.device), context.streams),
        )[0]


_context = lru_cache(maxsize=128)(ScanContext)


def scan(
    step: Step,
    *,
    carry: torch.Tensor,
    over: tuple[torch.Tensor, ...],
    chunk: int,
    invariants: tuple[torch.Tensor, ...] = (),
    valid_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Run ``step(carry, *slices, *invariants) -> (next_carry, output)`` in chunk order.

    Return the final carry and an ordered tuple of outputs (empty for zero rows).
    With ``valid_rows``, shapes describe capacity: only complete chunks within
    the device count advance carry. Padded chunks still return their readouts
    from the unchanged carry; callers select their actual output rows.
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
    if valid_rows is not None and (
        valid_rows.shape != () or valid_rows.dtype != torch.int64 or valid_rows.device != carry.device
    ):
        raise ValueError("scan valid_rows needs an int64 scalar on the carry device")
    context = ScanContext.active.get() or _context(TensorPolicy(), over[0].shape[0], carry.device)
    if (
        context.device.type == "cuda"
        and over[0].shape[0]
        and over[0].shape[0] % chunk == 0
        and not (torch.is_grad_enabled() and any(t.requires_grad for t in (carry, *over, *invariants)))
        and len({t.untyped_storage() for t in (carry, *over, *invariants)}) == 1 + len(over) + len(invariants)
    ):
        columns = tuple(tuple(torch.ops.aten.split.Tensor(t, chunk)) for t in over)
        if valid_rows is not None:
            active = (torch.arange(len(columns[0]), device=carry.device) + 1) * chunk <= valid_rows
            columns += (tuple(active.unbind()),)
        compiled = context.specialize(
            step,
            tuple((t.shape, t.stride(), t.dtype, t.device) for t in (carry, *invariants, *(c[0] for c in columns))),
            len(invariants),
            chunk,
            over[0].shape[0] // chunk,
            valid_rows is not None,
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
        updated, output = step(carry, *(t[start : start + chunk] for t in over), *invariants)
        carry = updated if valid_rows is None else torch.where(start + chunk <= valid_rows, updated, carry)
        outputs.append(output)
    return carry, tuple(outputs)
