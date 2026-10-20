"""Native tensor functions bound to the original state invocation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Generic, Literal, Protocol, TypeVar

import torch
from torch.optim.sgd import sgd

from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerFeedback
from pllm.contracts.semantic_graph import ImplementationId, SemanticOutputKind, StateResource, TaskSymbol
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.cuda_stream import CudaStreamReadiness, upload
from pllm.execution.memory import MemoryUsage, measure
from pllm.execution.ptts.page_borrow import BorrowedStateRoot, borrowed_root
from pllm.execution.ptts.page_bundle import PagePin, PagePublication, PageWrite, Slots
from pllm.execution.ptts.request_state import RequestStateSelection, StateAddress
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.state_execution import StateTaskContext
from pllm.execution.trace import TracePhase
from pllm.kernels.adamw import AdamWConfig, AdamWFields, apply_adamw, paged_adamw

I = TypeVar("I")
D = TypeVar("D")
R = TypeVar("R")
Roots = tuple[tuple[torch.Tensor, ...], ...]


def _no_slots(field: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    raise RuntimeError("this mathematical batch has no issued page access")


def _no_matrix() -> tuple[str, Slots]:
    raise RuntimeError("this mathematical batch has no issued page access")


@dataclass(frozen=True, slots=True)
class TensorBatch:
    """Borrowed mathematical operands, never a storage or publication capability."""

    selection: RequestStateSelection
    roots: Sequence[tuple[torch.Tensor, ...]]
    after_refs: Sequence[StateRef]
    stream: torch.cuda.Stream | None
    feedback: SamplerFeedback | None = None
    epoch: int = 1
    observing: bool = False
    # Field pool and the slot of every selected state's page, in selection order.
    field_slots: Callable[[int], tuple[torch.Tensor, tuple[int, ...]]] = _no_slots
    slot_matrix: Callable[[], tuple[str, Slots]] = _no_matrix
    results: dict[tuple[TaskSymbol, str], torch.Tensor] = field(
        default_factory=lambda: dict[tuple[TaskSymbol, str], torch.Tensor]()
    )
    borrowed_roots: tuple[BorrowedStateRoot, ...] = ()
    local_pin: PagePin | None = None
    completed: frozenset[TaskSymbol] = frozenset()

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
        positions = write.before_refs.positions(refs)
        refs = refs.replace_versions(write.before_refs, write.after_refs)
        access = tuple(write if index >= 0 else context.pin for index in positions.tolist())
    elif version in ("private", "published") and context.publication is not None:
        receipt = context.publication
        pin = context.borrow_published(receipt.after_refs)
        positions = receipt.before_refs.positions(refs)
        refs = refs.replace_versions(receipt.before_refs, receipt.after_refs)
        access = tuple(pin if index >= 0 else context.pin for index in positions.tolist())
    roots = context.arena.lazy_roots(access, refs)
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
        field_slots=roots.field_slots,
        slot_matrix=roots.slot_matrix,
    )


@dataclass(frozen=True, slots=True)
class FunctionMath(Generic[I, D, R]):
    initializers: Mapping[str, Callable[[Roots, tuple[RequestStateKey | None, ...]], None]]
    rows: Callable[[I], tuple[PackedPhaseRow, ...]]
    prepare_data: Callable[[I, TensorBatch], D]
    output_data: Callable[[D, InvocationOutput], R]

    def initialize(self, roots: Roots, addresses: tuple[StateAddress, ...]) -> None:
        resources = tuple(str(address.resource) for address in addresses)
        if len(roots) != len(addresses) or not set(resources) <= self.initializers.keys():
            raise ValueError("state initialization requires a named initializer for every root")
        for resource in dict.fromkeys(resources):
            selected = tuple(
                (root, address.key)
                for root, address in zip(roots, addresses, strict=True)
                if address.resource == resource
            )
            self.initializers[resource](tuple(root for root, _ in selected), tuple(key for _, key in selected))

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


FieldSlots = Callable[[int], tuple[torch.Tensor, tuple[int, ...]]]


class Optimizer(Protocol):
    @property
    def groups(self) -> int: ...

    def update(
        self, roots: Sequence[tuple[torch.Tensor, ...]], slots: FieldSlots, gradients: tuple[torch.Tensor, ...]
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class PoolSGD:
    """Apply ordinary SGD to private parameter views, shared by batch and region training."""

    lr: float
    groups: int

    @torch.no_grad()
    def update(
        self, roots: Sequence[tuple[torch.Tensor, ...]], slots: FieldSlots, gradients: tuple[torch.Tensor, ...]
    ) -> None:
        """Use Torch's optimizer implementation without carrying momentum state."""
        sgd(
            [parameter for root in roots for parameter in root],
            list(gradients),
            [],
            weight_decay=0,
            momentum=0,
            lr=self.lr,
            dampening=0,
            nesterov=False,
            maximize=False,
            foreach=False,
        )


@dataclass(frozen=True, slots=True)
class PoolAdamW:
    """One update entry point for grouped tensor views and FP32 CUDA page pools."""

    config: AdamWConfig
    groups: int

    def update(
        self, roots: Sequence[tuple[torch.Tensor, ...]], slots: FieldSlots, gradients: tuple[torch.Tensor, ...]
    ) -> None:
        """Update the borrowed parameter views; meta executes the same paged tensor path."""
        parameter = roots[0][0]
        if self.groups != 1 or parameter.device.type == "cpu" or parameter.dtype != torch.float32:
            apply_adamw(
                tuple(root[5 * group : 5 * group + 5] for root in roots for group in range(self.groups)),
                AdamWFields(*(LayoutFieldId(i) for i in range(5))),
                self.config,
                gradients,
            )
            return
        fields = tuple(slots(i) for i in range(5))
        pools = tuple(pool.reshape(len(pool), -1) if index != 2 else pool for index, (pool, _) in enumerate(fields))
        table = upload([rows for _, rows in fields], parameter.device)
        paged_adamw(pools, table, torch.stack(gradients).reshape(len(gradients), -1), self.config)


MemoryDataT = TypeVar("MemoryDataT", contravariant=True)


class MemoryOperation(Protocol[MemoryDataT]):
    """Resource projection of a bound executable operation."""

    def memory(self, data: MemoryDataT, batch: TensorBatch) -> MemoryUsage: ...
    def input_shapes(
        self, rows: int, hidden: int, training_shapes: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[int, ...], ...]: ...
    def resident(self, capture_bytes: int) -> int: ...


class TensorStep(Protocol[D]):
    @property
    def symbol(self) -> TaskSymbol: ...
    @property
    def training(self) -> bool: ...
    def execute(self, context: StateTaskContext[D], batch: TensorBatch) -> TaskOutput: ...
    def memory(
        self, data: D, batch: TensorBatch, continuation: Callable[[D, TensorBatch], TaskOutput]
    ) -> MemoryUsage: ...


@dataclass(frozen=True, slots=True)
class TensorOperation(Generic[D]):
    implementation_id: ImplementationId
    function: Callable[[D, TensorBatch], TaskOutput]
    mode: Literal["read", "transition", "update", "gradient"] = "read"
    select: Callable[[D, RequestStateSelection], tuple[int, ...]] | None = None
    optimizer: Optimizer | None = None
    publish: bool = False
    version: Literal["input", "private", "published"] = "input"
    copy_fields: tuple[LayoutFieldId, ...] | None = None
    step: TensorStep[D] | None = None

    def input_shapes(
        self, rows: int, hidden: int, training_shapes: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[int, ...], ...]:
        return ((rows, hidden),)

    def resident(self, capture_bytes: int) -> int:
        return capture_bytes if self.step is not None else 0

    def memory(self, data: D, batch: TensorBatch) -> MemoryUsage:
        """Execute this operation's tensor path, excluding the already-funded pages."""
        if self.step is not None:
            return self.step.memory(data, batch, self.function)

        def run() -> TaskOutput:
            output = self.function(data, batch)
            if self.optimizer is not None and output.gradients:
                self.optimizer.update(batch.roots, batch.field_slots, output.gradients)
            return output

        return measure(run)

    def execute(self, context: StateTaskContext[D]) -> None:
        data, stream = context.data(), context.pin.stream
        with nullcontext() if stream is None else torch.cuda.stream(stream), torch.inference_mode(False):
            batch = _read(context, self.version)
            selected = tuple(range(len(batch.roots))) if self.select is None else self.select(data, context.selection)
            if len(set(selected)) != len(selected) or any(i < 0 or i >= len(batch.roots) for i in selected):
                raise ValueError("update selection is outside the original state batch")
            if self.mode == "transition" and selected:
                context.write = context.arena.prepare_transition(
                    context.pin.refs.take(selected), stream, copy_fields=self.copy_fields
                )
                batch = _read(context, "private")
            if self.step is None:
                output = self.function(data, batch)
            else:
                output = self.step.execute(context, batch)
                if not self.step.training:
                    output = self.function(data, replace(batch, completed=frozenset((self.step.symbol,))))
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
                    or len(output.gradients) != len(selected) * self.optimizer.groups
                ):
                    raise ValueError("gradients require an explicit matching optimizer update")
                context.write = context.arena.prepare_transition(
                    context.pin.refs.take(selected), stream, copy_fields=self.copy_fields
                )
                roots = context.arena.lazy_roots((context.write,) * len(selected), context.write.after_refs)
                self.optimizer.update(roots, roots.field_slots, output.gradients)
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
                refs = StateRefs.of(batch.after_refs)
                if transition is not None:
                    refs = refs.replace_versions(transition.before_refs, transition.after_refs)
                context.updated_refs = refs.take(changed) if changed else ()
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


class BatchOperation(TensorOperation[D]):
    """A tensor operation whose admitted rows come from the external training batch."""

    def input_shapes(
        self, rows: int, hidden: int, training_shapes: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[int, ...], ...]:
        return (training_shapes[0],)


class BatchPreparation(BatchOperation[D]):
    """Validate both hidden and logit input capacities before training."""

    def input_shapes(
        self, rows: int, hidden: int, training_shapes: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[int, ...], ...]:
        return training_shapes


@dataclass(frozen=True)
class TensorComputation:
    """A producer's tensor program and homogeneous input shapes, runnable outside the worker."""

    function: Callable[..., tuple[torch.Tensor, ...]]
    shapes: tuple[tuple[int, ...], ...]
    dtype: torch.dtype

    def __call__(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.function(*inputs)

    def memory(self) -> MemoryUsage:
        inputs = tuple(torch.empty(shape, device="meta", dtype=self.dtype) for shape in self.shapes)
        return measure(lambda: self(*inputs))
