"""Native tensor functions bound to the original state invocation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Generic, Literal, Never, Protocol, TypeVar

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.optim.sgd import sgd

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.parameter_overlay import FieldView, ParameterTarget
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerFeedback
from pllm.contracts.semantic_graph import ImplementationId, SemanticOutputKind, StateResource, TaskSymbol
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.cuda_stream import CudaStreamReadiness, upload
from pllm.execution.ptts.page_borrow import BorrowedStateRoot, borrowed_root
from pllm.execution.ptts.page_bundle import PagePin, PagePublication, PageWrite
from pllm.execution.ptts.request_state import RequestStateSelection, StateAddress
from pllm.execution.ptts.slot_pools import Slots
from pllm.execution.state_execution import ProjectionResiduals, StateTaskContext
from pllm.execution.step_input import TensorStepInput
from pllm.execution.torch_private import MemoryTrace, MemoryUsage, measure
from pllm.execution.trace import TracePhase
from pllm.kernels.adamw import AdamWConfig, AdamWFields, apply_adamw, paged_adamw
from pllm.kernels.page_copy import copy_pages

I = TypeVar("I", bound=TensorStepInput)
D = TypeVar("D")
R = TypeVar("R")
Roots = tuple[tuple[torch.Tensor, ...], ...]


def _unwired(*_args: int | StateResource) -> Never:
    raise RuntimeError("this mathematical batch has no issued page or resource access")


@dataclass(frozen=True, slots=True)
class TensorBatch:
    """Borrowed mathematical operands, never a storage or publication capability."""

    selection: RequestStateSelection
    roots: Sequence[tuple[torch.Tensor, ...]]
    after_refs: Sequence[StateRef]
    stream: torch.cuda.Stream | None
    feedback: SamplerFeedback | None = None
    observing: bool = False
    # Field pool and the slot of every selected state's page, in selection order.
    field_slots: Callable[[int], tuple[torch.Tensor, tuple[int, ...]]] = _unwired
    slot_matrix: Callable[[], tuple[str, Slots]] = _unwired
    results: dict[tuple[TaskSymbol, str], torch.Tensor] = field(
        default_factory=lambda: dict[tuple[TaskSymbol, str], torch.Tensor]()
    )
    borrowed_roots: tuple[BorrowedStateRoot, ...] = ()
    local_pin: PagePin | None = None
    select_resource: Callable[[StateResource], TensorBatch] = _unwired

    def for_resource(self, resource: StateResource) -> TensorBatch:
        """Use the operation resource without redirecting reads of other states."""
        return (
            self
            if len(self.selection.groups) == 1 and self.selection.groups[0].resource == resource
            else self.select_resource(resource)
        )

    def parameter_root(self, resource: str) -> tuple[torch.Tensor, ...]:
        borrowed = borrowed_root(self.borrowed_roots, resource)
        if borrowed is not None:
            assert self.local_pin is not None
            return borrowed.tensors(self.local_pin)
        batch = self.for_resource(StateResource(resource))
        assert len(batch.selection.refs) == 1
        return batch.roots[0]


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
    export: Literal["drop", "block"] | None = None


@dataclass(frozen=True, slots=True)
class TaskOutput:
    residuals: ProjectionResiduals = field(
        default_factory=lambda: dict[tuple[RequestIdentity, ParameterTarget], torch.Tensor]()
    )
    guidance: PackedSamplerGuidanceBatch | None = None
    gradients: tuple[torch.Tensor, ...] = ()
    loss_count: int = 0
    update_count: int = 0
    trained_rows: int = 0
    changed: tuple[int, ...] = ()
    update_mask: torch.Tensor | None = None
    counters: torch.Tensor | None = None
    observations: tuple[Observation, ...] = ()
    gradient_indices: tuple[int, ...] = ()
    loss: torch.Tensor | None = None
    # Per-lineage memory writes, rows aligned to the gradient/update selection.
    field_writes: tuple[tuple[FieldView, torch.Tensor], ...] = ()


def autograd_loss(
    function: Callable[[D, Mapping[str, torch.Tensor]], torch.Tensor | None],
    names: tuple[str, ...],
    summary: Callable[[D], TaskOutput],
) -> Callable[[D, TensorBatch], TaskOutput]:
    """Compile tensor forward/loss and AOT backward; report and update outside the graph."""
    compiled = torch.compile(  # pyright: ignore[reportUnknownMemberType]
        function, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
    )

    def execute(data: D, batch: TensorBatch) -> TaskOutput:
        (root,) = batch.roots
        with torch.enable_grad():
            parameters = {name: root[5 * index].detach().requires_grad_(True) for index, name in enumerate(names)}
            sample = root[0]
            call = function if sample.device.type != "cuda" or MemoryTrace.active.get() is not None else compiled
            loss = call(data, parameters)
            if loss is None:
                return TaskOutput()
            output = summary(data)
            if output.update_mask is None and output.trained_rows == 0:
                return TaskOutput()
            if loss.ndim != 0 or output.trained_rows < 0:
                raise ValueError(
                    "optimizer functions must return a scalar loss with a positive effective row count (or None to skip)"
                )
            gradients = torch.autograd.grad(loss, tuple(parameters.values()))
        return replace(output, loss=loss.detach(), gradients=gradients)

    return execute


@dataclass(frozen=True, slots=True)
class InvocationOutput:
    selection: RequestStateSelection
    guidance: PackedSamplerGuidanceBatch | None
    publication: PagePublication | None
    loss: torch.Tensor | None = None
    value: torch.Tensor | None = None


def read_batch(
    context: StateTaskContext[D], version: Literal["input", "private", "published"] = "input"
) -> TensorBatch:
    """Borrow the invocation's selected version for tensor operations and layer replay."""
    assert version == "input" or len(context.publications) <= 1
    refs = context.pin.refs
    access: tuple[PagePin | PageWrite, ...] = (context.pin,) * len(refs)
    write = context.sole_write()
    if version == "private" and write is not None:
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

    def scoped(resource: StateResource) -> TensorBatch:
        selection = context.selection.scoped(resource)
        selected = roots.select(tuple(context.selection.refs.index(ref) for ref in selection.refs))
        return replace(
            batch,
            selection=selection,
            roots=selected,
            after_refs=tuple(refs[context.selection.refs.index(ref)] for ref in selection.refs),
            field_slots=selected.field_slots,
            slot_matrix=selected.slot_matrix,
        )

    batch = TensorBatch(
        context.selection,
        roots,
        refs,
        context.pin.stream,
        context.sampler_feedback,
        observing=context.trace.observing,
        results=context.results,
        borrowed_roots=context.borrowed_roots,
        local_pin=context.pin,
        field_slots=roots.field_slots,
        slot_matrix=roots.slot_matrix,
        select_resource=scoped,
    )
    return batch


@dataclass(frozen=True, slots=True)
class FunctionMath(Generic[I, D, R]):
    initializers: Mapping[str, Callable[[Roots, tuple[RequestStateKey | None, ...]], None]]
    rows: Callable[[I], tuple[PackedPhaseRow, ...]]
    prepare_data: Callable[[I, TensorBatch], D]
    output_data: Callable[[D, InvocationOutput], R]
    resume_data: Callable[[D, D], D] | None = None
    parameter_phases: tuple[HiddenPhase, ...] = ()

    def initialize(self, roots: Roots, addresses: tuple[StateAddress, ...]) -> None:
        resources = tuple(str(address.resource) for address in addresses)
        assert len(roots) == len(addresses) and set(resources) <= self.initializers.keys()
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
            value.order_auxiliary()
            data = self.prepare_data(value, read_batch(context))
            return data if context.value is None or self.resume_data is None else self.resume_data(context.value, data)

    def output(self, context: StateTaskContext[D]) -> R:
        graph = context.arena.plan.semantic_graph
        value = next(
            (
                context.results[graph.tasks[task.task_id].symbol, output.name]
                for task in reversed(context.arena.plan.schedule.tasks)
                for output in graph.tasks[task.task_id].outputs
                if (graph.tasks[task.task_id].symbol, output.name) in context.results
                and output.kind not in (SemanticOutputKind.TEMPORARY, SemanticOutputKind.HOST)
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
        self.tensors(tuple(parameter for root in roots for parameter in root), gradients)

    @torch.no_grad()
    def tensors(self, parameters: tuple[torch.Tensor, ...], gradients: tuple[torch.Tensor, ...]) -> None:
        """Update private tensors without requiring pooled field slots."""
        sgd(
            list(parameters),
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


def apply_adamw_or_reserve(
    roots: Sequence[tuple[torch.Tensor, ...]],
    fields: AdamWFields,
    config: AdamWConfig,
    gradients: tuple[torch.Tensor, ...],
) -> None:
    """Apply AdamW, or, on meta pages while planning, reserve the scratch the CUDA kernel would allocate."""
    if not (roots and roots[0][0].is_meta):
        apply_adamw(roots, fields, config, gradients)
        return
    # Torch's capturable implementation requires a CUDA tag; the storage stays meta.
    with FakeTensorMode() as mode:

        def operand(value: torch.Tensor) -> torch.Tensor:
            return FakeTensor(mode, value, torch.device("cuda:0"))

        pages = tuple(tuple(operand(value) for value in root) for root in roots)
        values = tuple(operand(value) for value in gradients)
        scratch = measure(lambda: apply_adamw(pages, fields, config, values))
    MemoryTrace.reserve(scratch.workspace)


@dataclass(frozen=True, slots=True)
class TensorAdamW:
    """Apply one Torch AdamW batch to the caller's already-private five-field groups."""

    config: AdamWConfig
    groups: int

    def update(
        self, roots: Sequence[tuple[torch.Tensor, ...]], slots: FieldSlots, gradients: tuple[torch.Tensor, ...]
    ) -> None:
        """Update in place; the invocation retains publication and page ownership."""
        self.tensors(roots, gradients)

    def tensors(self, roots: Sequence[tuple[torch.Tensor, ...]], gradients: tuple[torch.Tensor, ...]) -> None:
        """Apply the same grouped Torch update to ordinary or gathered private fields."""
        apply_adamw_or_reserve(
            tuple(root[5 * group : 5 * group + 5] for root in roots for group in range(self.groups)),
            AdamWFields(*(LayoutFieldId(i) for i in range(5))),
            self.config,
            gradients,
        )


@dataclass(frozen=True, slots=True)
class PageUpdate(TensorAdamW):
    """One update entry point for grouped tensor views and FP32 CUDA page pools."""

    def update(
        self, roots: Sequence[tuple[torch.Tensor, ...]], slots: FieldSlots, gradients: tuple[torch.Tensor, ...]
    ) -> None:
        """Update the borrowed parameter views; meta executes the same paged tensor path."""
        parameter = roots[0][0]
        if self.groups != 1 or parameter.device.type == "cpu" or parameter.dtype != torch.float32:
            self.tensors(roots, gradients)
            return
        fields = tuple(slots(i) for i in range(5))
        pools = tuple(pool.reshape(len(pool), -1) if index != 2 else pool for index, (pool, _) in enumerate(fields))
        table = upload([rows for _, rows in fields], parameter.device)
        paged_adamw(pools, table, torch.stack(gradients).reshape(len(gradients), -1), self.config)

    @torch.no_grad()
    def staged(
        self,
        pools: tuple[torch.Tensor, ...],
        fields: tuple[torch.Tensor, ...],
        sources: torch.Tensor,
        destinations: torch.Tensor,
        gradients: torch.Tensor,
        captures: Mapping[LayoutFieldId, torch.Tensor],
        last: torch.Tensor,
        *,
        paged: bool,
    ) -> None:
        """Update issued destinations and carry fields inside the shared tensor body."""
        count = gradients.shape[0]
        if paged:
            paged_adamw(pools[:5], destinations[:5], gradients, self.config, sources[:5, :count].contiguous())
        else:
            self.tensors(
                tuple(tuple(value[row] for value in fields[:5]) for row in range(count)), tuple(gradients.unbind())
            )
            rows = torch.arange(count, device=destinations.device)
            for field in range(5):
                copy_pages(fields[field], pools[field], rows, destinations[field])
        for field in set(range(5, len(pools))) - captures.keys():
            copy_pages(pools[field], pools[field], sources[field, :count], destinations[field])
        for field, value in captures.items():
            assert value.shape[1:] == pools[field].shape[1:] and value.dtype == pools[field].dtype
            copy_pages(value.detach(), pools[field], last, destinations[field])


MemoryDataT = TypeVar("MemoryDataT", contravariant=True)


class MemoryOperation(Protocol[MemoryDataT]):
    """Resource projection of a bound executable operation."""

    def memory(self, data: MemoryDataT, batch: TensorBatch) -> MemoryUsage: ...
    def input_shapes(
        self, rows: int, hidden: int, training_shapes: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[int, ...], ...]: ...
    def resident(self, capture_bytes: int) -> int: ...


def native_body(
    function: Callable[[D, TensorBatch], TaskOutput], data: D, batch: TensorBatch, context: StateTaskContext[D] | None
) -> TaskOutput:
    """Adapt native mathematical functions to the operation's single body interface."""
    return function(data, batch)


def _no_resident(capture_bytes: int) -> int:
    return 0


@dataclass(frozen=True, slots=True)
class TensorOperation(Generic[D]):
    """Run one mathematical body and settle its declared effects."""

    implementation_id: ImplementationId
    body: Callable[[D, TensorBatch, StateTaskContext[D] | None], TaskOutput]
    mode: Literal["read", "transition", "update", "gradient"] = "read"
    select: Callable[[D, RequestStateSelection], tuple[int, ...]] | None = None
    optimizer: Optimizer | None = None
    publish: bool = False
    version: Literal["input", "private", "published"] = "input"
    copy_fields: tuple[LayoutFieldId, ...] | None = None
    capture_resident: Callable[[int], int] = _no_resident

    def input_shapes(
        self, rows: int, hidden: int, training_shapes: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[int, ...], ...]:
        return ((rows, hidden),)

    def resident(self, capture_bytes: int) -> int:
        return self.capture_resident(capture_bytes)

    def memory(self, data: D, batch: TensorBatch) -> MemoryUsage:
        """Execute this operation's tensor path, excluding the already-funded pages."""

        def run() -> TaskOutput:
            output = self.body(data, batch, None)
            if self.optimizer is not None and output.gradients:
                self.apply_gradients(batch.roots, batch.field_slots, batch.roots, output)
            return output

        return measure(run)

    def apply_gradients(
        self,
        roots: Sequence[tuple[torch.Tensor, ...]],
        slots: FieldSlots,
        before: Sequence[tuple[torch.Tensor, ...]],
        output: TaskOutput,
    ) -> None:
        """Settle optimizer fields on-device, preserving every field on a skipped update."""
        assert self.optimizer is not None
        self.optimizer.update(roots, slots, output.gradients)
        if output.update_mask is not None:
            for root, source in zip(roots, before, strict=True):
                for value, previous in zip(root[: 5 * self.optimizer.groups], source, strict=False):
                    torch.where(output.update_mask, value, previous, out=value)

    def execute(self, context: StateTaskContext[D]) -> ProjectionResiduals:
        data, stream = context.data(), context.pin.stream
        with nullcontext() if stream is None else torch.cuda.stream(stream), torch.inference_mode(False):
            batch = read_batch(context, self.version)
            selected = tuple(range(len(batch.roots))) if self.select is None else self.select(data, context.selection)
            assert len(set(selected)) == len(selected) and all(0 <= i < len(batch.roots) for i in selected)
            write: PageWrite | None = None
            if self.mode == "transition" and selected:
                if self.version == "input" or context.sole_write() is None:
                    write = context.prepare_owned_write(context.pin.refs.take(selected), copy_fields=self.copy_fields)
                batch = read_batch(context, "private")
            output = self.body(data, batch, context)
            if output.gradient_indices:
                assert (
                    output.gradients
                    and len(set(output.gradient_indices)) == len(output.gradient_indices)
                    and all(index in selected for index in output.gradient_indices)
                )
                selected = output.gradient_indices
            if output.gradients:
                assert (
                    self.mode in ("update", "gradient")
                    and self.optimizer is not None
                    and len(output.gradients) == len(selected) * self.optimizer.groups
                )
                write = context.prepare_owned_write(context.pin.refs.take(selected), copy_fields=self.copy_fields)
                roots = context.arena.lazy_roots((write,) * len(selected), write.after_refs)
                self.apply_gradients(roots, roots.field_slots, tuple(batch.roots[index] for index in selected), output)
            if write is None:
                write = context.sole_write()
            if output.field_writes:
                assert write is not None
                roots = context.arena.lazy_roots((write,) * len(selected), write.after_refs)
                for view, values in output.field_writes:
                    pool, slots = roots.field_slots(view.field)
                    target = view.read(pool, pooled=True)
                    assert values.shape == (len(selected), *view.shape) and values.dtype == target.dtype
                    copy_pages(
                        values.detach().contiguous(),
                        target,
                        upload(tuple(range(len(selected))), pool.device),
                        upload(slots, pool.device),
                    )
            if write is not None and (self.mode == "update" or self.publish):
                context.publish_owned_write(write)
            if output.counters is not None:
                context.overlay_counters = (
                    output.counters if context.overlay_counters is None else context.overlay_counters + output.counters
                )
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
                    for task in context.arena.plan.schedule.tasks
                    if task.implementation_id == self.implementation_id
                )
                for observation in output.observations:
                    if observation.export is not None:
                        context.trace.record(
                            TracePhase.TENSOR,
                            context.invocation_id,
                            task_id=task.task_id,
                            identities=observation.identities,
                            step_id=observation.step_id,
                            tensor=observation.value,
                            tensor_name=observation.name,
                            export=observation.export,
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
            return output.residuals


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
