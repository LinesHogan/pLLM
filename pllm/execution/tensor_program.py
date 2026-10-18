"""Native tensor functions bound to the original state invocation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Generic, Literal, TypeVar

import torch

from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerFeedback
from pllm.contracts.semantic_graph import ImplementationId, SemanticOutputKind, StateResource, TaskSymbol
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.ptts.page_borrow import BorrowedStateRoot, borrowed_root
from pllm.execution.ptts.page_bundle import PagePin, PagePublication, PageWrite
from pllm.execution.ptts.page_capture import KernelCapture
from pllm.execution.ptts.request_state import RequestStateSelection
from pllm.execution.state_execution import StateTaskContext
from pllm.execution.trace import TracePhase
from pllm.kernels.paged_linear import PagedRows, TensorRows

I = TypeVar("I")
D = TypeVar("D")
R = TypeVar("R")
Roots = tuple[tuple[torch.Tensor, ...], ...]


def _no_pages(fields: tuple[int, ...], rows: tuple[int, ...]) -> tuple[PagedRows, ...]:
    raise RuntimeError("this mathematical batch has no issued page access")


@dataclass(frozen=True, slots=True)
class TensorBatch:
    """Borrowed mathematical operands, never a storage or publication capability."""

    selection: RequestStateSelection
    roots: Roots
    after_refs: tuple[StateRef, ...]
    stream: torch.cuda.Stream | None
    feedback: SamplerFeedback | None = None
    captured: tuple[torch.Tensor, ...] | None = None
    epoch: int = 1
    observing: bool = False
    paged: Callable[[tuple[int, ...], tuple[int, ...]], tuple[PagedRows, ...]] = _no_pages
    results: dict[tuple[TaskSymbol, str], torch.Tensor] = field(
        default_factory=lambda: dict[tuple[TaskSymbol, str], torch.Tensor]()
    )
    borrowed_roots: tuple[BorrowedStateRoot, ...] = ()
    local_pin: PagePin | None = None

    def parameter_root(self, resource: str) -> tuple[torch.Tensor, ...]:
        # No borrowed state: preserve the original sole-root reader (existing workload suite).
        if not self.borrowed_roots:
            return self.roots[0]
        borrowed = borrowed_root(self.borrowed_roots, resource)
        if borrowed is not None:
            assert self.local_pin is not None
            return borrowed.tensors(self.local_pin)
        group = self.selection.for_resource(StateResource(resource))
        (ref,) = group.refs
        return self.roots[self.selection.refs.index(ref)]

    def tensor_rows(self, fields: tuple[int, ...], rows: tuple[int, ...]) -> tuple[TensorRows, ...]:
        """Pack fields once per logical state; share one mapping or an existing equal-row grouping."""
        count = len(self.roots)
        group = len(rows) // count if count else 0
        indices = (
            group
            if group and len(rows) == count * group and rows == tuple(index // group for index in range(len(rows)))
            else torch.tensor(rows, device=self.roots[0][0].device, dtype=torch.long)
        )
        return tuple(
            TensorRows(
                self.roots[0][field].flatten().unsqueeze(0)
                if count == 1
                else torch.stack(tuple(root[field].flatten() for root in self.roots)),
                indices,
            )
            for field in fields
        )


@dataclass(frozen=True, slots=True)
class Observation:
    name: str
    value: torch.Tensor
    row: PackedPhaseRow | None = None
    ref: StateRef | None = None
    step_id: int | None = None
    gradient: bool = False
    identities: tuple[RequestIdentity, ...] = ()
    iteration: int = 0
    export: bool = False


@dataclass(frozen=True, slots=True)
class TaskOutput:
    guidance: PackedSamplerGuidanceBatch | None = None
    gradients: tuple[torch.Tensor, ...] = ()
    loss_count: int = 0
    update_count: int = 0
    trained_rows: int = 0
    changed: tuple[int, ...] = ()
    observations: tuple[Observation, ...] = ()
    gradient_indices: tuple[int, ...] = ()
    loss: torch.Tensor | None = None
    # Per-lineage memory writes, rows aligned to the gradient/update selection.
    field_writes: tuple[tuple[LayoutFieldId, torch.Tensor], ...] = ()


def autograd_loss(
    function: Callable[[D, Mapping[str, torch.Tensor]], torch.Tensor | None],
    names: tuple[str, ...],
    trained_rows: Callable[[D], int],
) -> Callable[[D, TensorBatch], TaskOutput]:
    """Differentiate named Adam fields of one logical instance, without owning its update."""

    def execute(data: D, batch: TensorBatch) -> TaskOutput:
        (root,) = batch.roots
        with torch.enable_grad():
            parameters = {name: root[5 * index].detach().requires_grad_(True) for index, name in enumerate(names)}
            loss = function(data, parameters)
            if loss is None:
                return TaskOutput()
            if loss.ndim != 0:
                raise ValueError("optimizer functions must return a scalar loss")
            gradients = torch.autograd.grad(loss, tuple(parameters.values()))
        count = trained_rows(data)
        if count < 1:
            raise ValueError("a training loss requires positive effective row count; return None to skip")
        return TaskOutput(loss=loss.detach(), gradients=gradients, loss_count=1, update_count=1, trained_rows=count)

    return execute


@dataclass(frozen=True, slots=True)
class InvocationOutput:
    selection: RequestStateSelection
    guidance: PackedSamplerGuidanceBatch | None
    publication: PagePublication | None
    loss: torch.Tensor | None = None
    value: torch.Tensor | None = None


def _read(context: StateTaskContext[D], version: Literal["input", "private", "published"] = "input") -> TensorBatch:
    if version != "input" and (len(context.selection.groups) > 1 or len(context.page_publications) > 1):
        raise ValueError("multiple resources or publications require explicit Region snapshot sources")
    refs = context.pin.refs
    access: tuple[PagePin | PageWrite, ...] = (context.pin,) * len(refs)
    if version == "private" and context.write is not None:
        write = context.write
        positions = {ref: i for i, ref in enumerate(write.before_refs)}
        refs = tuple(write.after_refs[positions[ref]] if ref in positions else ref for ref in refs)
        access = tuple(write if ref in positions else context.pin for ref in context.pin.refs)
    elif version in ("private", "published") and context.publication is not None:
        receipt = context.publication
        pin = context.borrow_published(receipt.after_refs)
        versions = dict(zip(receipt.before_refs, receipt.after_refs, strict=True))
        refs = tuple(versions.get(ref, ref) for ref in refs)
        access = tuple(pin if ref in versions else context.pin for ref in context.pin.refs)
    roots = tuple(
        tuple(
            context.arena.borrow_field(token, ref, LayoutFieldId(i))
            if isinstance(token, PagePin)
            else context.arena.write_field(token, token.after_refs.index(ref), LayoutFieldId(i))
            for i in range(len(context.arena.root(original).pages))
        )
        for token, ref, original in zip(access, refs, context.pin.refs, strict=True)
    )
    return TensorBatch(
        context.selection,
        roots,
        refs,
        context.pin.stream,
        context.sampler_feedback,
        epoch=context.arena.plan.state_physical.plan_epoch,
        observing=context.trace.observing,
        results=context.results,
        borrowed_roots=context.borrowed_roots,
        local_pin=context.pin,
        paged=lambda fields, rows: context.arena.paged_rows(
            access, refs, tuple(LayoutFieldId(field) for field in fields), rows
        ),
    )


@dataclass(frozen=True, slots=True)
class FunctionMath(Generic[I, D, R]):
    initializers: Mapping[str, Callable[[Roots], None]]
    rows: Callable[[I], tuple[PackedPhaseRow, ...]]
    prepare_data: Callable[[I, TensorBatch], D]
    output_data: Callable[[D, InvocationOutput], R]

    def initialize(self, roots: Roots, resources: tuple[str, ...]) -> None:
        if len(roots) != len(resources) or not set(resources) <= self.initializers.keys():
            raise ValueError("state initialization requires a named initializer for every root")
        for resource in dict.fromkeys(resources):
            self.initializers[resource](
                tuple(root for root, name in zip(roots, resources, strict=True) if name == resource)
            )

    def prepare(self, value: I, context: StateTaskContext[D]) -> D:
        with (
            nullcontext() if context.pin.stream is None else torch.cuda.stream(context.pin.stream),
            torch.inference_mode(False),
        ):
            return self.prepare_data(value, _read(context))

    def output(self, context: StateTaskContext[D]) -> R:
        graph = context.arena.plan.semantic_graph
        value = next(
            (
                context.results[graph.tasks[task.task_id].symbol, output.name]
                for task in reversed(context.arena.plan.schedule_physical.tasks)
                for output in graph.tasks[task.task_id].outputs
                if (graph.tasks[task.task_id].symbol, output.name) in context.results
                and output.kind is not SemanticOutputKind.TEMPORARY
            ),
            None,
        )
        return self.output_data(
            context.data(),
            InvocationOutput(
                context.selection,
                context.guidance,
                context.publication,
                context.loss,
                None if value is None else value.detach(),
            ),
        )


@dataclass(frozen=True, slots=True)
class TensorOperation(Generic[D]):
    implementation_id: ImplementationId
    function: Callable[[D, TensorBatch], TaskOutput]
    mode: Literal["read", "transition", "update", "gradient"] = "read"
    select: Callable[[D, RequestStateSelection], tuple[int, ...]] | None = None
    optimizer: Callable[[Roots, tuple[torch.Tensor, ...]], None] | None = None
    capture: KernelCapture | None = None
    capture_inputs: Callable[[D, TensorBatch], tuple[torch.Tensor, ...]] | None = None
    capture_padding: tuple[float, ...] = ()
    publish: bool = False
    gradient_fields: int = 1
    version: Literal["input", "private", "published"] = "input"
    copy_fields: tuple[LayoutFieldId, ...] | None = None

    def execute(self, context: StateTaskContext[D]) -> None:
        data, stream = context.data(), context.pin.stream
        with nullcontext() if stream is None else torch.cuda.stream(stream), torch.inference_mode(False):
            batch = _read(context, self.version)
            selected = tuple(range(len(batch.roots))) if self.select is None else self.select(data, context.selection)
            if len(set(selected)) != len(selected) or any(i < 0 or i >= len(batch.roots) for i in selected):
                raise ValueError("update selection is outside the original state batch")
            if self.mode == "transition" and selected:
                context.write = context.arena.prepare_transition(
                    tuple(context.pin.refs[i] for i in selected), stream, copy_fields=self.copy_fields
                )
                batch = _read(context, "private")
            if self.capture is not None and selected:
                if self.version != "input":
                    raise ValueError("this capture binding requires the invocation input version")
                if self.capture_inputs is None:
                    raise ValueError("captured operation requires explicit tensor inputs")
                view = self.capture.view
                values = self.capture_inputs(data, batch)
                if len(values) != len(view.inputs) or any(
                    value.ndim != buffer.ndim
                    or any(size > bound for size, bound in zip(value.shape, buffer.shape, strict=True))
                    for value, buffer in zip(values, view.inputs, strict=True)
                ):
                    raise ValueError("input differs from its signed captured shapes")
                if self.capture.writable:
                    if context.write is None:
                        raise ValueError("captured mutation requires a private transition")
                    context.arena.select_captured(view, context.write, context.write.after_refs)
                else:
                    context.arena.select_captured(view, context.pin, context.pin.refs)
                for i, (value, buffer) in enumerate(zip(values, view.inputs, strict=True)):
                    if value.shape != buffer.shape:
                        buffer.fill_(self.capture_padding[i] if self.capture_padding else 0)
                    buffer[tuple(slice(0, size) for size in value.shape)].copy_(value)
                context.arena.replay_captured(view)
                batch = replace(batch, captured=tuple(value[: len(selected)].clone() for value in view.results))
            output = self.function(data, batch)
            if output.gradient_indices:
                if (
                    not output.gradients
                    or len(set(output.gradient_indices)) != len(output.gradient_indices)
                    or any(index not in selected for index in output.gradient_indices)
                ):
                    raise ValueError("gradient selection must be a subset of the authorized state selection")
                selected = output.gradient_indices
            if output.gradients:
                if (
                    self.mode not in ("update", "gradient")
                    or self.optimizer is None
                    or len(output.gradients) != len(selected) * self.gradient_fields
                ):
                    raise ValueError("gradients require an explicit matching optimizer update")
                context.write = context.arena.prepare_transition(
                    tuple(context.pin.refs[i] for i in selected), stream, copy_fields=self.copy_fields
                )
                self.optimizer(context.arena.writable_roots(context.write), output.gradients)
            if output.field_writes:
                if context.write is None:
                    raise ValueError("memory writes require the invocation's private transition")
                roots = context.arena.writable_roots(context.write)
                for field_id, values in output.field_writes:
                    if values.shape[0] != len(selected) or values.ndim < 2:
                        raise ValueError("memory writes require one row per selected lineage")
                    for root, value in zip(roots, values.unbind(), strict=True):
                        if root[int(field_id)].shape != value.shape or root[int(field_id)].dtype != value.dtype:
                            raise ValueError("memory write differs from its signed field shape or dtype")
                        root[int(field_id)].copy_(value)
            if self.mode == "update" or self.publish:
                context.publish()
            if self.mode != "read":
                changed = output.changed or (selected if output.gradients else ())
                transition = context.publication if context.publication is not None else context.write
                versions = (
                    {} if transition is None else dict(zip(transition.before_refs, transition.after_refs, strict=True))
                )
                context.updated_refs = tuple(versions.get(batch.after_refs[i], batch.after_refs[i]) for i in changed)
            if output.guidance is not None:
                context.guidance = output.guidance
                context.guidance_ready = CudaStreamReadiness(stream) if stream is not None else None
            if output.loss is not None:
                context.loss = output.loss
            context.loss_count += output.loss_count
            context.update_count += output.update_count
            context.trained_rows += output.trained_rows
            if output.observations:
                task = next(
                    task
                    for task in context.arena.plan.schedule_physical.tasks
                    if task.implementation_id == self.implementation_id
                )
                for observation in output.observations:
                    if observation.export:
                        context.trace.record(
                            TracePhase.TENSOR,
                            context.invocation_id,
                            task_id=task.task_id,
                            identities=observation.identities,
                            step_id=observation.step_id,
                            tensor=observation.value,
                            tensor_name=observation.name,
                        )
                        continue
                    context.trace.observe(
                        context.invocation_id,
                        task.task_id,
                        observation.name,
                        observation.value,
                        row=observation.row,
                        refs=() if observation.ref is None else (observation.ref,),
                        identities=observation.identities,
                        iteration=observation.iteration,
                        step_id=observation.step_id,
                        gradient=observation.gradient,
                    )
