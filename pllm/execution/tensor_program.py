"""Tensor operations and how their results are written.

Owns the operands a body borrows (TensorBatch, read_batch), what a body returns (TaskOutput), the optimizers
(PoolSGD, PageUpdate) and TensorOperation, which runs one body and turns its output into private
writes, publications, counters and trace records. TensorDeclaration and the other bindings build TensorOperations;
StateExecution runs them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Generic, Literal, Never, Protocol, TypeVar

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.optim.sgd import sgd

from pllm import _core
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedRows
from pllm.contracts.parameter_overlay import Projection
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerFeedback, SamplerStage
from pllm.contracts.semantic_graph import ImplementationId, SemanticOutputKind, StateResource, TaskSymbol
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.cuda_stream import CudaStreamReadiness, upload
from pllm.execution.ptts.page_borrow import BorrowedStateRoot, borrowed_root
from pllm.execution.ptts.page_bundle import PagePin, PagePublication, PageWrite
from pllm.execution.ptts.request_state import RequestStateSelection, StateAddress
from pllm.execution.ptts.slot_pools import Slots
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.state_execution import ProjectionResiduals, StateTaskContext
from pllm.execution.step_input import StepInput
from pllm.execution.torch_private import MemoryTrace, MemoryUsage, measure
from pllm.execution.trace import TraceEventKind
from pllm.kernels.adamw import AdamWConfig, AdamWFields, apply_adamw, paged_adamw
from pllm.kernels.page_copy import copy_pages

if TYPE_CHECKING:
    from pllm.execution.attention_methods import AttentionProgram
    from pllm.execution.model_sites import ResidualRewrite
    from pllm.execution.overlay_task import OverlayTransition

I = TypeVar("I", bound=StepInput)
D = TypeVar("D")
R = TypeVar("R")
Roots = tuple[tuple[torch.Tensor, ...], ...]
ProducedVersions = tuple[tuple[StateResource, Literal["private", "published"]], ...]
# An AdamW state's leading fields, in layout order: one flat parameter field and its optimizer fields.
ADAM_FIELDS = ("parameters", "gradient", "step", "exp_avg", "exp_avg_sq")
ADAM = len(ADAM_FIELDS)


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
    resources: Mapping[StateResource, TensorBatch] = field(default_factory=lambda: dict[StateResource, TensorBatch]())
    retain_gradients: bool = True

    def for_resource(self, resource: StateResource) -> TensorBatch:
        """Use the operation resource without redirecting reads of other states."""
        return (
            self
            if len(self.selection.groups) == 1 and self.selection.groups[0].resource == resource
            else replace(self.resources[resource], resources=self.resources)
        )

    def state_free(self) -> TensorBatch:
        """The view of a task that reads no state, whatever the step selected; other states stay reachable."""
        return replace(self, selection=RequestStateSelection(self.selection.rows, ()), roots=(), after_refs=())

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
    # Each exported row's token position within its request.
    positions: tuple[int, ...] = ()
    iteration: int = 0
    export: Literal["drop", "block"] | None = None
    dropped_rows: int = 0


@dataclass(frozen=True, slots=True)
class TaskOutput:
    residuals: ProjectionResiduals = field(
        default_factory=lambda: dict[tuple[RequestIdentity, Projection], torch.Tensor]()
    )
    guidance: PackedSamplerGuidanceBatch | None = None
    gradients: tuple[torch.Tensor, ...] = ()
    loss_count: int = 0
    # Counts known on the host; `counters` carries device [applied updates, trained rows] known only on the device.
    update_count: int = 0
    trained_rows: int = 0
    changed: tuple[int, ...] = ()
    update_mask: torch.Tensor | None = None
    counters: torch.Tensor | None = None
    observations: tuple[Observation, ...] = ()
    gradient_indices: tuple[int, ...] = ()
    loss: torch.Tensor | None = None


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
            (name,) = names
            parameters = {
                name: root[0].detach().to(torch.promote_types(root[0].dtype, torch.float32)).requires_grad_(True)
            }
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
    guidance: tuple[PackedSamplerGuidanceBatch, ...]
    publication: PagePublication | None
    loss: torch.Tensor | None = None
    value: torch.Tensor | None = None


def _batch_access(
    context: StateTaskContext[D],
    reads: tuple[tuple[StateResource, Literal["input", "private", "published"]], ...],
) -> tuple[StateRefs, tuple[PagePin | PageWrite, ...], Slots]:
    refs = context.pin.refs
    access: list[PagePin | PageWrite] = [context.pin]
    owners = np.zeros(len(refs), dtype=np.int64)
    for resource, origin in reads:
        if origin == "input":
            continue
        write = context.pending_write(resource)
        receipt = context.publication_of(resource)
        if origin == "private" and write is not None:
            before, after = write.before_refs, write.after_refs
            token: PagePin | PageWrite = write
        elif receipt is not None:
            before, after = receipt.before_refs, receipt.after_refs
            token = context.borrow_published(after)
        else:
            continue
        positions = before.positions(refs)
        versions, owners = _core.read_versions(refs.versions, owners, positions, after.versions, len(access))
        refs = StateRefs(refs.layouts, refs.lineages, versions, refs.rows)
        access.append(token)
    return refs, tuple(access), owners


def read_batch(
    context: StateTaskContext[D],
    version: Literal["input", "private", "published"] = "input",
    resource: StateResource = StateResource(""),
    produced: ProducedVersions = (),
) -> TensorBatch:
    """Borrow the invocation's selected version for tensor operations and layer replay: the input version, or the
    pending private write / the publication of `resource` this invocation made. Produced reads are
    selected independently, so a writer can consume another state's new values in the same step."""
    refs, access, owners = _batch_access(context, (*produced, (resource, version)))
    roots = context.arena.lazy_roots(access, owners, refs)

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
        retain_gradients=context.retain_gradients or context.trace.observing,
    )
    if len(context.selection.groups) <= 1:
        return batch
    resources: dict[StateResource, TensorBatch] = {}
    for group in context.selection.groups:
        selection = context.selection.scoped(group.resource)
        indices = context.selection.refs.positions(selection.refs)
        selected = roots.select(indices)
        resources[group.resource] = replace(
            batch,
            selection=selection,
            roots=selected,
            after_refs=refs.take(indices),
            field_slots=selected.field_slots,
            slot_matrix=selected.slot_matrix,
        )
    return replace(batch, resources=resources)


@dataclass(frozen=True, slots=True)
class FunctionMath(Generic[I, D, R]):
    initializers: Mapping[str, Callable[[Roots, tuple[RequestStateKey | None, ...]], None]]
    rows: Callable[[I], PackedRows]
    prepare_data: Callable[[I, TensorBatch], D]
    output_data: Callable[[D, InvocationOutput], R]
    resume_data: Callable[[D, D], D] | None = None
    # The @pllm.custom_residual methods the backend installs on the model.
    model_sites: tuple[ResidualRewrite, ...] = ()
    parameter_overlay: OverlayTransition | None = None
    attention: AttentionProgram | None = None

    @property
    def parameter_phases(self) -> tuple[HiddenPhase, ...]:
        """Derive training phases from the declared overlay rather than retaining another copy."""
        return () if self.parameter_overlay is None else self.parameter_overlay.phases

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


@dataclass(frozen=True, slots=True)
class PoolSGD:
    """Apply ordinary SGD to private parameter views, shared by batch and region training."""

    lr: float

    @torch.no_grad()
    def update(
        self, roots: Sequence[tuple[torch.Tensor, ...]], slots: FieldSlots, gradients: tuple[torch.Tensor, ...]
    ) -> None:
        """Use Torch's optimizer implementation without carrying momentum state."""
        self.tensors(tuple(parameter for root in roots for parameter in root), gradients)

    @torch.no_grad()
    def tensors(self, parameters: tuple[torch.Tensor, ...], gradients: tuple[torch.Tensor, ...]) -> None:
        """Update private tensors without requiring pooled field slots."""
        values = tuple(parameter.to(torch.promote_types(parameter.dtype, torch.float32)) for parameter in parameters)
        sgd(
            list(values),
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

        for parameter, value in zip(parameters, values, strict=True):
            if value is not parameter:
                parameter.copy_(value)


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
class PageUpdate:
    """AdamW over the ADAM_FIELDS of private roots: Torch's update on tensor views, the paged kernel on FP32 CUDA
    page pools. The invocation retains publication and page ownership."""

    config: AdamWConfig

    def tensors(self, roots: Sequence[tuple[torch.Tensor, ...]], gradients: tuple[torch.Tensor, ...]) -> None:
        """Apply Torch's update to ordinary or gathered private fields."""
        apply_adamw_or_reserve(
            tuple(root[:ADAM] for root in roots),
            AdamWFields(*(LayoutFieldId(i) for i in range(ADAM))),
            self.config,
            gradients,
        )

    def update(
        self, roots: Sequence[tuple[torch.Tensor, ...]], slots: FieldSlots, gradients: tuple[torch.Tensor, ...]
    ) -> None:
        """Update the borrowed parameter views; meta executes the same paged tensor path."""
        parameter = roots[0][0]
        if parameter.device.type == "cpu" or parameter.dtype != torch.float32:
            self.tensors(roots, gradients)
            return
        fields = tuple(slots(i) for i in range(ADAM))
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
        carries: Mapping[LayoutFieldId, torch.Tensor],
        updates: torch.Tensor | None,
        last: torch.Tensor | None,
        *,
        paged: bool,
    ) -> None:
        """Write every issued destination inside the shared tensor body, without a host sync.

        An absent update mask means every issued destination takes the optimizer step. Masked updates
        preserve other destinations. Carries read their last trained row when one exists.
        """
        count = gradients.shape[0]
        stepped = destinations if updates is None else torch.where(updates, destinations, -1)
        if paged:
            paged_adamw(pools[:ADAM], stepped[:ADAM], gradients, self.config, sources[:ADAM, :count].contiguous())
        else:
            self.tensors(
                tuple(tuple(value[row] for value in fields[:ADAM]) for row in range(count)), tuple(gradients.unbind())
            )
            rows = torch.arange(count, device=destinations.device)
            for field in range(ADAM):
                copy_pages(fields[field], pools[field], rows, stepped[field])
        carried = None
        if carries:
            assert last is not None
            carried = destinations if updates is None else torch.where(updates & (last >= 0), destinations, -1)
            for field, value in carries.items():
                assert value.shape[1:] == pools[field].shape[1:] and value.dtype == pools[field].dtype
                copy_pages(value.detach(), pools[field], last.clamp_min(0), carried[field])
        for field in range(len(pools)):
            if updates is None and (field < ADAM or field in carries):
                continue
            written = stepped if field < ADAM else carried if field in carries else None
            copy_pages(
                pools[field],
                pools[field],
                sources[field, :count],
                destinations[field] if written is None else torch.where(written[field] >= 0, -1, destinations[field]),
            )


MemoryDataT = TypeVar("MemoryDataT", contravariant=True)


class MemoryOperation(Protocol[MemoryDataT]):
    """Resource projection of a bound executable operation."""

    def memory(self, data: MemoryDataT, batch: TensorBatch) -> MemoryUsage: ...
    def input_shapes(
        self, rows: int, hidden: int, training_shapes: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[int, ...], ...]: ...
    def resident(self, capture_bytes: int) -> int: ...


def native_body(
    function: Callable[[D, TensorBatch], TaskOutput], data: D, batch: TensorBatch, context: StateTaskContext[D]
) -> TaskOutput:
    """Adapt native mathematical functions to the operation's single body interface."""
    return function(data, batch)


def _no_resident(capture_bytes: int) -> int:
    return 0


@dataclass(frozen=True, slots=True)
class TensorOperation(Generic[D]):
    """Run one mathematical body and write its declared effects."""

    implementation_id: ImplementationId
    body: Callable[[D, TensorBatch, StateTaskContext[D]], TaskOutput]
    # The same body for planning: run on meta or example inputs without page writes, so memory is traceable.
    trace_body: Callable[[D, TensorBatch], TaskOutput]
    mode: Literal["read", "transition", "update", "gradient"] = "read"
    select: Callable[[D, RequestStateSelection], tuple[int, ...]] | None = None
    optimizer: PageUpdate | None = None
    publish: bool = False
    version: Literal["input", "private", "published"] = "input"
    produced: ProducedVersions = ()
    copy_fields: tuple[LayoutFieldId, ...] | None = None
    select_fields: (
        Callable[[D, RequestStateSelection, tuple[int, ...]], tuple[tuple[LayoutFieldId, ...], ...]] | None
    ) = None
    capture_resident: Callable[[int], int] = _no_resident
    # The state the operation changes; its body sees that state's selection and roots. Empty for a reader.
    resource: StateResource = StateResource("")

    def input_shapes(
        self, rows: int, hidden: int, training_shapes: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[int, ...], ...]:
        return ((rows, hidden),)

    def resident(self, capture_bytes: int) -> int:
        return self.capture_resident(capture_bytes)

    def memory(self, data: D, batch: TensorBatch) -> MemoryUsage:
        """Execute this operation's tensor path, excluding the already-funded pages."""

        batch = self.scoped(batch)

        def run() -> TaskOutput:
            output = self.trace_body(data, batch)
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
                for value, previous in zip(root[:ADAM], source, strict=False):
                    torch.where(output.update_mask, value, previous, out=value)

    def scoped(self, batch: TensorBatch) -> TensorBatch:
        """The roots the body writes, or every readable state for a read-only operation."""
        return batch.for_resource(self.resource) if self.resource else batch

    def batch(self, context: StateTaskContext[D], version: Literal["input", "private", "published"]) -> TensorBatch:
        return self.scoped(read_batch(context, version, self.resource, self.produced))

    def execute(self, context: StateTaskContext[D]) -> ProjectionResiduals:
        data, stream = context.data(), context.pin.stream
        with nullcontext() if stream is None else torch.cuda.stream(stream), torch.inference_mode(False):
            batch = self.batch(context, self.version)
            selected = tuple(range(len(batch.roots))) if self.select is None else self.select(data, batch.selection)
            write: PageWrite | None = None
            if self.mode == "transition" and selected:
                if self.version == "input" or context.pending_write(self.resource) is None:
                    fields = (
                        None if self.select_fields is None else self.select_fields(data, context.selection, selected)
                    )
                    write = context.prepare_owned_write(
                        batch.selection.refs.take(selected), fields=fields, copy_fields=self.copy_fields
                    )
                batch = self.batch(context, "private")
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
                    and len(output.gradients) == len(selected)
                )
                write = context.prepare_owned_write(batch.selection.refs.take(selected), copy_fields=self.copy_fields)
                roots = context.arena.lazy_roots((write,), np.zeros(len(selected), dtype=np.int64), write.after_refs)
                self.apply_gradients(roots, roots.field_slots, tuple(batch.roots[index] for index in selected), output)
            if write is None:
                write = context.pending_write(self.resource)
            if write is not None and (self.mode == "update" or self.publish):
                context.publish_owned_write(write)
            if output.counters is not None:
                context.training_counters = (
                    output.counters
                    if context.training_counters is None
                    else context.training_counters + output.counters
                )
            if output.guidance is not None:
                context.guidance = tuple(
                    sorted(
                        (*context.guidance, output.guidance),
                        key=lambda item: (item.stage is SamplerStage.PACKED_POST_FILTER_EXACT, item.order),
                    )
                )
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
                            TraceEventKind.TENSOR,
                            context.invocation_id,
                            task_id=task.task_id,
                            identities=observation.identities,
                            positions=observation.positions,
                            step_id=observation.step_id,
                            tensor=observation.value,
                            tensor_name=observation.name,
                            export=observation.export,
                            dropped_rows=observation.dropped_rows,
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
