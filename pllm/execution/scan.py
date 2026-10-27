"""Public chunk recurrences and execution-owned, budgeted FX preparation."""

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import replace
from functools import lru_cache
from time import perf_counter
from typing import TypeVar

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import _disable_current_modes  # pyright: ignore[reportPrivateUsage]

from pllm.contracts.tensor_policy import ScanForm, TensorPolicy
from pllm.execution.memory import MemoryTrace, measure
from pllm.execution.numerics import warn_numerical_transforms
from pllm.execution.scan_analysis import Specs, analyze
from pllm.execution.scan_execution import PreparedScan, Step, associative_prefix, competitive

R = TypeVar("R")


class ScanContext:
    """An operation's numerical policy, maximum row count and workspace ceiling."""

    active: ContextVar["ScanContext | None"] = ContextVar("scan_context", default=None)

    def __init__(
        self,
        policy: TensorPolicy,
        max_rows: int,
        device: torch.device,
        payload_bytes: int | None = None,
        *,
        replay: bool = True,
    ) -> None:
        self.policy, self.max_rows, self.device = policy, max_rows, device
        self.payload_bytes = payload_bytes
        self.replay = replay

    def bind(self, function: Callable[..., R]) -> Callable[..., R]:
        """Use the same execution-owned scope for memory tracing and execution."""

        def run(*args: object, **kwargs: object) -> R:
            token = self.active.set(self)
            try:
                return function(*args, **kwargs)
            finally:
                self.active.reset(token)

        return run

    @lru_cache(maxsize=128)
    def _template(self, step: Step, specs: Specs, invariants: int, chunk: int, counted: bool) -> "ScanTemplate":
        return ScanTemplate(self, step, specs, invariants, chunk, counted)

    def specialize(
        self, step: Step, specs: Specs, invariants: int, chunk: int, count: int, counted: bool = False
    ) -> PreparedScan:
        """Reuse tracing and preparation across a bounded bucket of actual chunk counts."""
        tracing = specs[0][3].type == "meta"
        layouts = tuple(
            (shape, stride, dtype, self.device if device.type == "meta" else device)
            for shape, stride, dtype, device in specs
        )
        return self._template(step, layouts, invariants, chunk, counted).for_count(count, tracing=tracing)


class ScanTemplate:
    """Cold shape analysis and one selected plan per count bucket; no carry storage."""

    def __init__(
        self, context: ScanContext, step: Step, specs: Specs, invariants: int, chunk: int, counted: bool
    ) -> None:
        self.context, self.step, self.specs = context, step, specs
        self.invariants, self.chunk, self.counted = invariants, chunk, counted
        self.maximum = (context.max_rows + chunk - 1) // chunk + 1
        sequences = specs[1 + invariants : -1] if counted else specs[1 + invariants :]
        self.analysis = (
            analyze(step, (specs[0], *sequences, *specs[1 : 1 + invariants]))
            if context.policy.allow_numerical_transforms
            else None
        )
        self.candidates: dict[int, tuple[PreparedScan, ...]] = {}
        self.plans: dict[int, PreparedScan] = {}

    def samples(
        self, count: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor | None]:
        values = tuple(
            torch.empty_strided(shape, stride, dtype=dtype, device=device).zero_()
            for shape, stride, dtype, device in self.specs
        )
        sequences = values[1 + self.invariants : -1] if self.counted else values[1 + self.invariants :]
        over = tuple(t.new_zeros((count * self.chunk, *t.shape[1:])) for t in sequences)
        return (
            values[0],
            over,
            values[1 : 1 + self.invariants],
            values[-1].new_full((), count * self.chunk) if self.counted else None,
        )

    def funded(self, capacity: int) -> tuple[PreparedScan, ...]:
        source = PreparedScan(self.step, self.chunk)
        choices = [source]
        analysis = self.analysis
        if analysis is not None and self.specs[0][2] in (torch.float32, torch.bfloat16):
            maximum = 64 if self.context.payload_bytes is not None or analysis.kind in ("left", "right") else 8
            for window in (1, 2, 4, 8, 16, 32, 64):
                if window > min(maximum, 1 << (capacity - 1).bit_length()):
                    break
                form = ScanForm.HILLIS_STEELE if window >= 16 or analysis.kind == "diagonal" else ScanForm.ORDERED
                if analysis.kind in ("left", "right"):
                    width = self.specs[0][0][-1 if analysis.kind == "right" else -2]
                    if self.specs[0][2] != torch.float32 or not competitive(width, self.chunk, window, form):
                        continue
                choices.append(PreparedScan(self.step, self.chunk, analysis, window, form))
        funded: list[PreparedScan] = []
        with _disable_current_modes(), torch.inference_mode(False), FakeTensorMode():
            args = self.samples(capacity)
            scratch = torch.backends.cuda.cublas_workspace_size() + torch.backends.cuda.cublaslt_workspace_size()
            for choice in choices:
                if choice.analysis is not None:
                    carry, over, invariants, _ = args
                    packed = tuple(t.new_empty((choice.window, self.chunk, *t.shape[1:])) for t in over)
                    choice = replace(choice, analysis=choice.analysis.batched((carry, *packed, *invariants), len(over)))
                usage = measure(lambda: choice(*args))
                workspace = usage.workspace + scratch
                if choice.analysis is not None and choice.analysis.kind == "diagonal":
                    # Pointwise HOP composition has two products and an addition per
                    # level. Fund every level's full arrays, even when Inductor fuses them.
                    workspace += (
                        4 * choice.window.bit_length() * choice.window * args[0].numel() * args[0].element_size()
                    )
                if self.context.payload_bytes is None or workspace <= self.context.payload_bytes:
                    funded.append(replace(choice, workspace=workspace))
        if not funded:
            raise ValueError("increase the scan execution budget to fund its tensor and BLAS workspace")
        return tuple(funded)

    def for_count(self, count: int, *, tracing: bool) -> PreparedScan:
        """Reserve the entire bucket before tuning at its first actual chunk count."""
        capacity = min(self.maximum, 1 << (max(1, count) - 1).bit_length())
        if capacity not in self.candidates:
            self.candidates[capacity] = self.funded(capacity)
        candidates = self.candidates[capacity]
        if tracing:
            return replace(candidates[-1], workspace=max(c.workspace for c in candidates))
        if capacity not in self.plans:
            self.plans[capacity] = self._prepare(candidates, count)
        return self.plans[capacity]

    def _prepare(self, candidates: tuple[PreparedScan, ...], count: int) -> PreparedScan:
        selected = candidates[-1]
        if self.context.policy.tuning == "measured" and len(candidates) > 1:
            args = self.samples(count)
            selected = min(
                (self.executable(plan, args) for plan in candidates), key=lambda plan: self.latency(plan, args)
            )
        else:
            selected = (
                self.executable(selected, self.samples(count))
                if selected.analysis is not None and selected.analysis.kind == "diagonal"
                else selected
            )
        if selected.analysis is not None:
            warn_numerical_transforms(
                f"{selected.analysis.coefficients.code}:{selected.window}:{selected.form}",
                (f"{selected.analysis.kind} scan: batched coefficients and {selected.form.value} prefix",),
            )
        return selected

    @staticmethod
    def executable(
        plan: PreparedScan,
        args: tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor | None],
    ) -> PreparedScan:
        """Compile and warm the supported pointwise HOP before capture or selection."""
        if plan.analysis is not None and plan.analysis.kind == "diagonal":
            pointwise = torch.compile(  # pyright: ignore[reportUnknownMemberType]
                associative_prefix, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
            )
            plan = replace(plan, pointwise=pointwise)
            plan(*args)
        return plan

    def latency(
        self,
        plan: PreparedScan,
        args: tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor | None],
    ) -> float:
        """Warm CUDA replay timing includes source execution as a real candidate."""
        for _ in range(3):
            plan(*args)
        torch.cuda.synchronize()
        if not self.context.replay:
            started = perf_counter()
            for _ in range(3):
                plan(*args)
                torch.cuda.synchronize()
            return (perf_counter() - started) * 1000 / 3
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = plan(*args)
        for _ in range(3):
            graph.replay()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        del result
        return start.elapsed_time(end) / 10


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
    """Run ``step(carry, *slices, *invariants)`` in chunk order, returning exclusive readouts.

    Counted calls advance carry only for complete chunks within the device row count;
    inactive capacity still evaluates its readout. Strict, CPU, partial, aliased and
    differentiable calls preserve source operation order. Step owners and closures
    must be immutable; pass varying tensor values through explicit arguments.
    """
    if chunk < 1 or not over or any(t.ndim == 0 or t.shape[0] != over[0].shape[0] for t in over):
        raise ValueError("scan needs a positive chunk and aligned row tensors")
    if valid_rows is not None and (
        valid_rows.shape != () or valid_rows.dtype != torch.int64 or valid_rows.device != carry.device
    ):
        raise ValueError("scan valid_rows needs an int64 scalar on the carry device")
    context = ScanContext.active.get() or _context(TensorPolicy(), over[0].shape[0], carry.device)
    source = PreparedScan(step, chunk)
    if (
        context.device.type == "cuda"
        and len(over[0])
        and len(over[0]) % chunk == 0
        and carry.is_contiguous()
        and not (torch.is_grad_enabled() and any(t.requires_grad for t in (carry, *over, *invariants)))
        and len({t.untyped_storage() for t in (carry, *over, *invariants)}) == 1 + len(over) + len(invariants)
    ):
        samples = (carry, *invariants, *(t[:chunk] for t in over), *((valid_rows,) if valid_rows is not None else ()))
        compiled = context.specialize(
            step,
            tuple((t.shape, t.stride(), t.dtype, t.device) for t in samples),
            len(invariants),
            chunk,
            len(over[0]) // chunk,
            valid_rows is not None,
        )
        if carry.is_meta:
            MemoryTrace.reserve(compiled.workspace)
            state, outputs = source(carry, over, invariants, valid_rows)
            if compiled.analysis is not None:
                return state.clone(), tuple(output.clone() for output in outputs)
            return state, outputs
        else:
            return compiled(carry, over, invariants, valid_rows)
    return source(carry, over, invariants, valid_rows)
