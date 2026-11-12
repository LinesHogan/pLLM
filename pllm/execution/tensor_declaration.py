"""Tensor declarations bound once to the plan: model effects, row tasks and their execution binding.

TensorDeclaration holds declaration state and binds the declared tasks to the
specialized plan (native implementations, scans, fixed-shape steps, input preparation).
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from math import prod
from typing import TYPE_CHECKING, Literal

import torch

from pllm.config import ExecutionOptions
from pllm.contracts.backend_attachment import (
    ActivationPatchBinding,
    BackendAttachmentSpec,
    MissingPolicy,
    ModelProbe,
    SamplerBinding,
)
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.host_batch import StateImport
from pllm.contracts.member import MemberDeclaration
from pllm.contracts.packed_rows import PackedPhaseRowBatch, PackedRows
from pllm.contracts.parameter_overlay import FieldView, ParameterOverlayFieldBinding, Projection
from pllm.contracts.request_identity import RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerStage
from pllm.contracts.semantic_graph import (
    AnchorSymbol,
    ConditionalInputGroup,
    HiddenInputSource,
    HostPayloadSchemaId,
    HostServiceId,
    HostServiceInputSource,
    HostServicePort,
    InputTokensInputSource,
    MathContract,
    PackedRowsInputSource,
    PackedSamplerRowsInputSource,
    SemanticEffect,
    SemanticGraph,
    SemanticInput,
    SemanticInputSource,
    SemanticOutput,
    SemanticOutputKind,
    SemanticState,
    SemanticTask,
    StateEffectKind,
    StateResource,
    StateScope,
    StateSelectionInputSource,
    TaskImplementationSpec,
    TaskResourceFact,
    TaskResultInputSource,
    TaskSymbol,
    TimingConstraint,
    TokenProgramEffect,
    WorkloadId,
)
from pllm.contracts.state_feed import FeedBinding
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import (
    LayoutFieldId,
    StateTensorLayout,
    TensorDType,
    TensorFieldSpec,
    TensorLayoutRequest,
    TensorLayoutSpec,
)
from pllm.execution.attention_methods import AttentionProgram
from pllm.execution.causal_batch import CausalBinder
from pllm.execution.custom_forward import CustomForwardTask
from pllm.execution.declaration import (
    Call,
    Dimensions,
    FieldShard,
    LayoutField,
    State,
    When,
    native,
    operation_mode,
    planned_implementation,
)
from pllm.execution.model_sites import SiteOwners
from pllm.execution.module_state import copy_initial
from pllm.execution.overlay_batch import OverlayBatch
from pllm.execution.overlay_task import OverlayTransition
from pllm.execution.ptts.page_bundle import PagePublication
from pllm.execution.ptts.request_state import RequestStateSelection
from pllm.execution.scan import ScanContext
from pllm.execution.state_execution import (
    ConditionalProducerTask,
    PublishStateTask,
    TaskGraphImplementation,
)
from pllm.execution.step_capture import StepCapture
from pllm.execution.task_memory import example, producers_first, task_example
from pllm.execution.tensor_executor import FieldWriteExecutor, TensorExecutor, reads_rows
from pllm.execution.tensor_frame import (
    OverlayEntry,
    TensorFrame,
    TensorMath,
    field_values,
    flat_gradient,
)
from pllm.execution.tensor_program import (
    ADAM,
    ADAM_FIELDS,
    FunctionMath,
    InvocationOutput,
    Roots,
    TaskOutput,
    TensorBatch,
    TensorOperation,
    native_body,
)
from pllm.execution.tensor_task import TensorTask
from pllm.execution.token_program import TokenProgram
from pllm.execution.torch_private import HostGather, compile_math
from pllm.kernels.adamw import AdamWConfig

if TYPE_CHECKING:
    from pllm.execution.checkpoint import PretrainedFile
    from pllm.execution.compile_facts import BackendCompileFacts
    from pllm.execution.step_input import StepInput


def _initialize_fields(
    initialize: Callable[[tuple[torch.Tensor, ...], torch.Generator], None],
    shards: Mapping[int, FieldShard],
    tensors: tuple[torch.Tensor, ...],
    generator: torch.Generator,
    rank: int,
) -> None:
    full = tuple(tensor.new_zeros(shards[i].shape) if i in shards else tensor for i, tensor in enumerate(tensors))
    initialize(full, generator)
    for i, shard in shards.items():
        tensors[i].copy_(shard.read(full[i], tuple(tensors[i].shape), rank))


@dataclass(frozen=True, slots=True)
class StepResult:
    value: torch.Tensor | None
    invocation: InvocationOutput
    entries: tuple[OverlayEntry, ...] = ()
    input_gradients: Mapping[tuple[str, str], torch.Tensor] = field(
        default_factory=lambda: dict[tuple[str, str], torch.Tensor]()
    )
    parameter_gradient_fields: Mapping[StateRef, tuple[torch.Tensor, ...]] = field(
        default_factory=lambda: dict[StateRef, tuple[torch.Tensor, ...]]()
    )

    metrics: Mapping[str, torch.Tensor] = field(default_factory=lambda: dict[str, torch.Tensor]())
    # A host training step's state after its update, by field name: owned copies on the step's stream. Empty for
    # engine steps.
    parameters: Mapping[str, torch.Tensor] = field(default_factory=lambda: dict[str, torch.Tensor]())

    @property
    def parameter_gradients(self) -> Mapping[StateRef, torch.Tensor]:
        """Explicit flat inspection; execution retains the original gradient fields."""
        return {ref: flat_gradient(values) for ref, values in self.parameter_gradient_fields.items()}

    @property
    def loss(self) -> torch.Tensor | None:
        return self.invocation.loss

    @property
    def guidance(self) -> tuple[PackedSamplerGuidanceBatch, ...]:
        return self.invocation.guidance

    @property
    def publication(self) -> PagePublication | None:
        return self.invocation.publication


TensorOutput = StepResult | torch.Tensor | None


def _tensor_math(function: TensorMath, **inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
    # Dynamo installs bytecode globals in this module, rather than a frozen
    # workload function's private snapshot of its author's globals.
    return function(**inputs)


def _host_math(
    function: TensorMath, names: tuple[str, ...], **inputs: torch.Tensor
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    tables = tuple(inputs[name] for name in names)
    if not tables[0].is_cuda:
        return function(**inputs)
    with HostGather(tables):
        return function(**inputs)


def _host_table_task(task: TensorTask, resources: set[StateResource]) -> TensorTask:
    """Lower every host field's row reads through the same bounded gather."""
    names = tuple(
        key
        for key, source in task.sources.items()
        if isinstance(source, StateSelectionInputSource) and source.resource in resources
    )
    return replace(task, function=partial(_host_math, task.function, names)) if names else task


def _lookup_math(task: TensorTask, facts: BackendCompileFacts | None) -> TensorMath | None:
    """Let Inductor fuse token lookups, casts and pointwise math into one table read."""
    if (
        facts is None  # Host execution has no backend numerical policy.
        or facts.device.type != "cuda"  # CPU execution supplies the untransformed reference math.
        or not facts.timeline.tensor_policy.allow_numerical_transforms  # Fusion can change rounding.
        or task.differentiable  # Task boundaries must retain the authored graph for differentiation.
        or not any(  # Only token lookups use this table-read fusion.
            isinstance(source, InputTokensInputSource) for source in task.sources.values()
        )
    ):
        return None
    return compile_math(partial(_tensor_math, task.function))


class TensorDeclaration:
    """Own tensor declaration state and bind tasks; results remain invocation-local."""

    def __init__(
        self,
        name: str,
        dimensions: Dimensions,
        *,
        host: bool = False,
    ) -> None:
        self.graph = SemanticGraph(WorkloadId(name), ())
        self.layouts = TensorLayoutRequest(())
        self.effects = BackendAttachmentSpec()
        self.operations: dict[TaskSymbol, TensorOperation] = {}
        self.field_shards: dict[str, dict[int, FieldShard]] = {}
        self.publications: dict[TaskSymbol, PublishStateTask] = {}
        # Fields that start at zero in a fresh page pool, so admission need not write them (AdamW state other than
        # the parameters). Request-scoped pages are recycled and dirty, so only instance states qualify.
        self.zero_start_fields: dict[str, frozenset[int]] = {}
        self.custom_forward_programs: dict[str, TokenProgram] | None = None
        self.overlay_math: OverlayTransition | None = None
        self.custom_forward_updates = False
        self.custom_forward_rows = 0
        self.attention = AttentionProgram()
        # The @pllm.custom_residual methods, by the model places they own.
        self.sites = SiteOwners()
        self.dimensions = dimensions
        self.host = host
        self.tensor_parallel_rank = 0
        self.gather_bytes: dict[tuple[TaskSymbol, str], int] = {}
        self.pretrained: dict[str, PretrainedFile] = {}
        self.members: dict[str, MemberDeclaration] = {}
        self.task_members: dict[TaskSymbol, tuple[str, ...]] = {}
        self.checkpoint = ""
        self.state_sources: dict[str, str] = {}
        self.feeds: tuple[FeedBinding, ...] = ()
        self.tensor_tasks: dict[TaskSymbol, TensorTask] = {}
        self.initializers: dict[str, Callable[[tuple[torch.Tensor, ...], torch.Generator], None]] = {}
        # Module states: their flat initial values, built once at bind from the model dimensions and seed.
        self.initial_values: dict[str, Callable[[Dimensions, int], torch.Tensor]] = {}
        self.causal: CausalBinder | None = None
        self.probe: ModelProbe | None = None
        self.overlay_targets: tuple[Projection, ...] = ()
        self.field_views: dict[tuple[str, str], FieldView] = {}
        self.imports: dict[str, tuple[FieldView, ...]] = {}
        self.position_states: tuple[tuple[str, float], ...] = ()

    @contextmanager
    def member(self, name: str, declaration: MemberDeclaration) -> Generator[None]:
        """Retain author names while lowering a member into one or more physical tasks."""
        self.members[name] = declaration
        symbols = {task.symbol for task in self.graph.tasks}
        yield
        self.task_members.update((task.symbol, (name,)) for task in self.graph.tasks if task.symbol not in symbols)

    @property
    def parameter_phases(self) -> tuple[HiddenPhase, ...]:
        """The overlay phases in the backend effect declaration."""
        return self.effects.overlay_phases

    def retained_fields(self, resource: str) -> dict[int, TensorFieldSpec]:
        """Layout fields whose initial values admission copies; zero-start fields are left alone."""
        layout = next(item.layout for item in self.layouts.layouts if item.resource == resource)
        zero = self.zero_start_fields.get(resource, frozenset())
        return {i: field for i, field in enumerate(layout.fields) if i not in zero}

    @property
    def template_bytes(self) -> int:
        """Device memory held by the retained initial values of every initialized state."""
        return sum(
            prod(field.shape) * field.dtype.itemsize
            for resource in {*self.initializers, *self.initial_values}
            if resource in self.initial_values
            or not any(
                state.resource == resource and state.scope is StateScope.INSTANCE and state.immutable
                for state in self.graph.states
            )
            for field in self.retained_fields(resource).values()
        )

    def initializer(
        self, resource: str, device: torch.device, seed: int
    ) -> Callable[[Roots, tuple[RequestStateKey | None, ...]], None]:
        """Copy one resource's shared initial tensors into each newly admitted root."""
        init = self.initializers.get(resource)
        shards = self.field_shards.get(resource, {})
        if resource in self.initial_values:
            init = partial(copy_initial, self.initial_values[resource](self.dimensions, seed).to(device))
        if init is None:
            return lambda roots, keys: None
        state = next(state for state in self.graph.states if state.resource == resource)
        if state.scope is StateScope.INSTANCE and state.immutable and resource not in self.initial_values:
            initial_device = torch.device("cpu") if state.placement == "host" else device

            def seed_roots(roots: Roots, keys: tuple[RequestStateKey | None, ...]) -> None:
                for root in roots:
                    _initialize_fields(
                        init,
                        shards,
                        root,
                        torch.Generator(device=initial_device).manual_seed(seed),
                        self.tensor_parallel_rank,
                    )

            return seed_roots
        layout = next(item.layout for item in self.layouts.layouts if item.resource == resource)
        tensors = tuple(torch.zeros(field.shape, dtype=field.dtype.torch, device=device) for field in layout.fields)
        # A resource's initial tensors are shared values, independent of admission history.
        _initialize_fields(
            init, shards, tensors, torch.Generator(device=device).manual_seed(seed), self.tensor_parallel_rank
        )
        values = {i: tensors[i] for i in self.retained_fields(resource)}

        def initialize(roots: Roots, keys: tuple[RequestStateKey | None, ...]) -> None:
            for root in roots:
                for index, value in values.items():
                    root[index].copy_(value)

        return initialize

    def select_capture(self, resources: tuple[TaskResourceFact, ...]) -> None:
        """Run each tensor task that was funded resident memory as a captured CUDA graph."""
        captured = {fact.task for fact in resources if fact.resident_bytes}
        self.graph = replace(
            self.graph,
            tasks=tuple(
                replace(
                    task,
                    implementations=(
                        native(f"python.cuda_graph:{self.graph.workload_id}:{task.symbol}@1", "compute/cuda_graph@1"),
                    ),
                )
                if task.symbol in self.tensor_tasks and task.symbol in captured
                else task
                for task in self.graph.tasks
            ),
        )

    def require_resident_side_tasks(self, plan: PlanBundle, operations: Mapping[TaskSymbol, TensorOperation]) -> None:
        """Side-lane tensor tasks run concurrently with the model, so their inputs must be fixed-size."""
        if not plan.timeline.task_resources:
            return
        for task in plan.schedule.tasks:
            symbol = plan.semantic_graph.tasks[task.task_id].symbol
            if task.lane == "side" and symbol in self.tensor_tasks and not operations[symbol].resident(1):
                raise ValueError(
                    "side tensor tasks need fixed row inputs; declare row inputs or move the task before the model"
                )

    def step_row_reader(self) -> Callable[[StepInput], PackedRows]:
        """The rows that open a step's state context, chosen once at bind: a model effect reading request state
        selects every packed row's request; a custom forward over no request state selects none."""
        effects, request = self.effects, any(state.scope is StateScope.REQUEST for state in self.graph.states)
        if (
            effects.activation_bindings
            or self.sites.rewrites
            or (effects.custom_forward is not None and effects.overlay_bindings)
        ):
            return (lambda value: value.rows.rows) if request else (lambda value: value.rows.rows[:0])
        if effects.custom_forward is not None and not request:
            return lambda value: value.rows.rows[:0]
        read = self.row_reader()
        return lambda value: read(value.rows)

    def function_math(self, device: torch.device, seed: int) -> FunctionMath:
        """Bind state initialization, step rows and frame preparation for execution."""
        for weights in self.pretrained.values():
            weights.verify()
        return FunctionMath(
            {str(state.resource): self.initializer(str(state.resource), device, seed) for state in self.graph.states},
            self.step_row_reader(),
            self.bind_prepare(),
            lambda data, invocation: self.result(data, invocation) if self.graph.states else invocation.value,
            lambda previous, incoming: replace(
                previous, step=incoming.step, rows=incoming.rows, plan=None, training=None
            ),
            model_sites=tuple(self.sites.rewrites),
            attention=self.attention,
            parameter_overlay=self.overlay_math,
        )

    @property
    def has_exports(self) -> bool:
        """Whether tensor tasks need a host observation callback."""
        return any(task.export is not None for task in self.tensor_tasks.values())

    def configure_scans(self, facts: BackendCompileFacts) -> None:
        """Scope model-route scans and give a chunked route overlay its eager batched executor; ordinary task
        scopes and executors are built in executables."""
        if self.overlay_math is not None and any(task.custom_forward is not None for task in self.graph.tasks):
            math = replace(
                self.overlay_math,
                function=ScanContext(
                    facts.timeline.tensor_policy,
                    self.custom_forward_rows,
                    facts.device,
                    replay="compute/cuda_graph@1" in facts.timeline.capabilities,
                    blas_workspace_bytes=facts.probe.blas_workspace_bytes,
                ).bind(self.overlay_math.function),
            )
            self.overlay_math = math if math.chunk is None else replace(math, executor=OverlayBatch(math))

    def executables(
        self,
        facts: BackendCompileFacts | None,
        resources: tuple[TaskResourceFact, ...],
        *,
        warmup: bool = False,
        map_host: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> dict[TaskSymbol, TensorOperation]:
        """Build each tensor operation once, with the same scan scope for eager and captured math."""
        capture = facts is not None and "compute/cuda_graph@1" in facts.timeline.capabilities
        limits = {fact.task: fact for fact in resources}
        operations = {
            symbol: self._operation(symbol, capture, limits)
            for symbol in self.operations
            if symbol not in self.tensor_tasks
        }
        host = {state.resource for state in self.graph.states if state.placement == "host"}
        tasks = {symbol: _host_table_task(task, host) for symbol, task in self.tensor_tasks.items()}
        scans = (
            None
            if facts is None
            else ScanContext(
                facts.timeline.tensor_policy,
                facts.probe.max_packed_rows,
                facts.device,
                None
                if facts.timeline.execution_peak_budget_bytes is None
                else facts.timeline.execution_peak_budget_bytes // max(1, facts.max_admitted_lineages),
                replay=capture,
                blas_workspace_bytes=facts.probe.blas_workspace_bytes,
            )
        )
        compiled = {symbol: _lookup_math(task, facts) for symbol, task in tasks.items()}
        warmed = {
            dependency.symbol
            for symbol, function in compiled.items()
            if warmup and function is not None
            for dependency in tasks[symbol].dependencies(tasks)
        }
        sample = example(self, facts, ExecutionOptions()) if warmed and facts is not None else None
        for symbol in producers_first(self.graph, tuple(self.tensor_tasks)):
            task = tasks[symbol]
            values = (
                None if sample is None or symbol not in warmed else (task_example(sample[0], task.phase), sample[1])
            )
            build = partial(
                self._tensor_operation,
                task,
                tasks,
                facts,
                limits if capture else {},
                compiled[symbol],
                values,
                map_host,
            )
            operations[symbol] = build() if scans is None else scans.bind(build)()
        if scans is None:
            return operations
        return {
            symbol: replace(operation, body=scans.bind(operation.body), trace_body=scans.bind(operation.trace_body))
            for symbol, operation in operations.items()
        }

    def _operation(
        self, symbol: TaskSymbol, capture: bool, limits: Mapping[TaskSymbol, TaskResourceFact]
    ) -> TensorOperation:
        """A declared operation; a chunked overlay transition runs through its OverlayBatch."""
        operation, math = self.operations[symbol], self.overlay_math
        if math is None or math.chunk is None or symbol != math.task:
            return operation
        overlay = OverlayBatch(math, limits.get(symbol) if capture else None)
        batched = replace(math, executor=overlay)
        return replace(
            operation, body=partial(native_body, batched), trace_body=batched.trace, capture_resident=overlay.resident
        )

    def _tensor_operation(
        self,
        task: TensorTask,
        tasks: Mapping[TaskSymbol, TensorTask],
        facts: BackendCompileFacts | None,
        limits: Mapping[TaskSymbol, TaskResourceFact],
        compiled: TensorMath | None,
        sample: tuple[TensorFrame, TensorBatch] | None = None,
        map_host: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> TensorOperation:
        """A tensor task's executor: a field writer or a body over masked tables, captured when `limits` funds it."""
        declaration = next(item for item in self.graph.tasks if item.symbol == task.symbol)
        state = task.training
        optimizer = None if state is None else state.page_update()
        step_capture = self._step_capture(task, facts, limits)
        executor: TensorExecutor | FieldWriteExecutor
        if task.writer is None:
            executor = TensorExecutor(
                task,
                tasks,
                optimizer,
                capture=step_capture,
                device=torch.device("cpu") if facts is None else facts.device,
                filter_rows=task.phase is not None and task.phase is not self.row_phase(),
                all_rows=task.symbol in self.model_site_tasks,
                compiled=compiled,
                reuse_results=self.reuses_results(task),
            )
            if sample is not None:
                host_states = {state.resource for state in self.graph.states if state.placement == "host"}
                host_inputs = frozenset(
                    name
                    for name, source in task.sources.items()
                    if isinstance(source, StateSelectionInputSource) and source.resource in host_states
                )
                executor.warmup(*sample, host_inputs=host_inputs, map_host=map_host)
        else:
            executor = FieldWriteExecutor(task.writer, step_capture)
        return TensorOperation(
            declaration.implementations[0].implementation_id,
            executor,
            executor.trace,
            operation_mode(
                state,
                transition=any(
                    effect.kind in (StateEffectKind.TRANSITION, StateEffectKind.TRANSITION_MAYBE)
                    for effect in declaration.effects
                ),
            ),
            optimizer=optimizer,
            capture_resident=executor.resident,
            resource=StateResource("") if state is None else state.declaration.resource,
        )

    def _step_capture(
        self, task: TensorTask, facts: BackendCompileFacts | None, limits: Mapping[TaskSymbol, TaskResourceFact]
    ) -> StepCapture | None:
        """Capture a funded writer or row-reading task; exports run eagerly."""
        if facts is None or task.symbol not in limits or task.export is not None:
            return None
        if self.reuses_results(task) or task.writer is None and not reads_rows(task, self.tensor_tasks):
            return None
        # A task without its own resource reads every state of the invocation.
        request = any(
            task.resource in ("", state.resource) and state.scope is StateScope.REQUEST for state in self.graph.states
        )
        return StepCapture(limits[task.symbol], facts.max_admitted_lineages if request else 1)

    def reuses_results(self, task: TensorTask) -> bool:
        """A sampler can reuse results whose producers used its invocation row domain."""
        return (
            task.sampler_stage is not None
            and not task.differentiable
            and not task.policy.forward_rows
            and task.phase is None
            and bool(task.sources)
            and all(
                isinstance(source, TaskResultInputSource) and source.task not in self.model_site_tasks
                for source in task.sources.values()
            )
        )

    def prepare_import(self, update: StateImport[torch.Tensor], batch: TensorBatch, data: TensorFrame) -> None:
        """Convert validated fields to storage precision and select their private roots."""
        (resource,) = self.imports
        selection = batch.selection
        group = selection.for_resource(StateResource(resource))
        if group.instance:
            data.replacements = ((int(selection.refs.positions(group.refs)[0]), update.values),)
        else:
            selected: dict[int, int] = {}
            for row, index in zip(selection.rows, group.state_indices, strict=True):
                if selection.refs[int(index)].version_id != 0:
                    continue
                source = data.step.import_rows[row.identity.request_id]
                if index in selected and selected[index] != source:
                    raise ValueError("one import batch cannot replace a shared request state with two values")
                selected[index] = source
            data.replacements = tuple(
                (index, tuple(tensor[row] for tensor in update.values)) for index, row in selected.items()
            )
        replacements: list[tuple[int, tuple[torch.Tensor, ...]]] = []
        for index, values in data.replacements:
            root = tuple(view.read(batch.roots[index][view.field]) for view in self.imports[resource])
            replacements.append((index, field_values(values, root, resource=resource, validate=False)))
        data.replacements = tuple(replacements)

    @property
    def model_site_tasks(self) -> frozenset[TaskSymbol]:
        """The method closure that produces values for every scheduled model row."""
        return frozenset(
            task.symbol
            for rewrite in self.sites.rewrites
            for source in rewrite.results
            for task in self.tensor_tasks[source.task].dependencies(self.tensor_tasks)
        )

    @property
    def uses_sampler_rows(self) -> bool:
        """Additive sampler frames include completed prompt endings as well as decode rows."""
        sampler = self.effects.sampler
        return any(item.stage is SamplerStage.PACKED_PRE_FILTER_ADD for item in sampler)

    def row_phase(self) -> HiddenPhase | None:
        """The frame's preselected phase; None keeps every request's final row for model effects or prefill."""
        if (
            self.uses_sampler_rows
            or self.graph.input_token_history >= 0
            or self.imports
            or self.overlay_targets
            or self.effects.activation_bindings
            or any(
                isinstance(port.source, HiddenInputSource) and port.source.phase is not HiddenPhase.DECODE
                for task in self.graph.tasks
                for port in task.inputs
            )
        ):
            return None
        return HiddenPhase.DECODE

    def row_reader(self) -> Callable[[PackedPhaseRowBatch], PackedRows]:
        """Bind the frame's row selection once; tasks filter only when they need a narrower phase."""
        if self.row_phase() is None:
            return lambda rows: rows.producer_rows(None)
        return lambda rows: rows.latest_decode_rows

    def bind_prepare(self) -> Callable[[StepInput, TensorBatch], TensorFrame]:
        """Prepare independent input channels into the same invocation frame."""
        read = self.row_reader()
        # Additive logits read the rows the sampler draws from that complete a prompt or decode, in sampler order.
        sampled = self.uses_sampler_rows

        def prepare(value: StepInput, batch: TensorBatch) -> TensorFrame:
            if sampled:
                labels, packed = value.rows, value.sampler_rows.packed_indices
                ends = (labels.phases[packed] == 0) | (
                    labels.token_offsets[packed] == labels.prompt_lengths[packed] - 1
                )
                rows = PackedRows(labels, packed[ends & (labels.recomputed[packed] == 0)])
            else:
                rows = read(value.rows)
            data = TensorFrame(value, rows)
            match value.batch:
                case StateImport() as update:
                    self.prepare_import(update, batch, data)
                case _:
                    pass
            return data

        return prepare

    @staticmethod
    def result(data: TensorFrame, invocation: InvocationOutput) -> StepResult:
        return StepResult(
            invocation.value if data.training is None else data.training.output.detach(),
            invocation,
            tuple(data.entries),
            data.input_gradients,
            data.parameter_gradient_fields,
            {} if data.training is None else data.training.metrics,
            data.parameters,
        )

    def state(
        self,
        name: str,
        fields: tuple[LayoutField, ...] | None = None,
        *,
        scope: StateScope = StateScope.INSTANCE,
        keys: tuple[RequestKeyField, ...] = (),
        optimizer: AdamWConfig | None = None,
        carry: tuple[LayoutField, ...] = (),
    ) -> State:
        """Declare state and its resolved tensor layout, including optimizer and carried fields."""
        if fields is None:
            assert optimizer is None and not carry, "an optimizer and carried fields need declared parameter fields"
            declaration = SemanticState(StateResource(name), scope, keys)
            self.graph = replace(
                self.graph, states=tuple(sorted((*self.graph.states, declaration), key=lambda state: state.resource))
            )
            return State(declaration, ())
        if optimizer is not None:
            (parameters,) = fields  # declare_parameters packs AdamW parameters into one flat field
            fields = tuple(
                replace(
                    parameters,
                    name=f"{parameters.name}.{name}",
                    shape=() if name == "step" else parameters.shape,
                    dtype=TensorDType.FLOAT32
                    if name == "step"
                    else (
                        parameters.dtype
                        if name == "parameters" or parameters.dtype is TensorDType.FLOAT64
                        else TensorDType.FLOAT32
                    ),
                )
                for name in ADAM_FIELDS
            )
        fields = (*fields, *carry)
        self.field_shards[name] = {i: field.shard for i, field in enumerate(fields) if field.shard is not None}
        if len({field.name for field in fields}) != len(fields):
            raise ValueError(f"{name}: duplicate field name")
        # Ports read parameters and carries; the optimizer reads its own fields from the version it updates.
        declaration = SemanticState(
            StateResource(name), scope, keys, writer_fields=() if optimizer is None else tuple(range(1, ADAM))
        )
        self.graph = replace(
            self.graph, states=tuple(sorted((*self.graph.states, declaration), key=lambda state: state.resource))
        )
        shapes: list[tuple[int, ...]] = []
        for layout_field in fields:
            if callable(layout_field.shape):
                shapes.append(layout_field.shape(self.dimensions))
            else:
                shapes.append(layout_field.shape)

        def dtype(field: LayoutField) -> TensorDType:
            if callable(field.dtype):
                return field.dtype(self.dimensions)
            return field.dtype

        layout = TensorLayoutSpec(
            f"{self.graph.workload_id}.{name}@1",
            tuple(
                TensorFieldSpec(LayoutFieldId(i), shape, dtype(field), field.alignment)
                for i, (field, shape) in enumerate(zip(fields, shapes, strict=True))
            ),
        )
        self.layouts = TensorLayoutRequest(
            tuple(sorted((*self.layouts.layouts, StateTensorLayout(name, layout)), key=lambda entry: entry.resource))
        )
        if optimizer is not None and scope is StateScope.INSTANCE:
            self.zero_start_fields[name] = frozenset(range(1, ADAM))  # every AdamW field after the parameters
        return State(declaration, fields, optimizer)

    @staticmethod
    def input(service: str, port: str, schema: str) -> HostServiceInputSource:
        """Name a typed host-service input port."""
        return HostServiceInputSource(HostServiceId(service), HostServicePort(port), HostPayloadSchemaId(schema))

    rows = staticmethod(PackedRowsInputSource)
    sampler_rows = staticmethod(PackedSamplerRowsInputSource)

    def compute(
        self,
        name: str,
        *,
        inputs: Mapping[str, SemanticInputSource],
        extra_inputs: tuple[SemanticInput, ...] = (),
        outputs: tuple[str, ...] = (),
        implementations: tuple[TaskImplementationSpec, ...] = (),
        when: When,
        reads: tuple[SemanticEffect, ...] = (),
        changes: State | None = None,
        effect: StateEffectKind = StateEffectKind.UPDATE_MAYBE,
        source: Call | None = None,
        result_schema: str = "",
        output_kind: SemanticOutputKind = SemanticOutputKind.TEMPORARY,
        custom_forward: TokenProgramEffect | None = None,
        math: MathContract | None = None,
        function: Callable[[TensorFrame, TensorBatch], TaskOutput] | None = None,
        operation_type: type[TensorOperation] = TensorOperation,
        select: Callable[[TensorFrame, RequestStateSelection], tuple[int, ...]] | None = None,
    ) -> Call:
        """Declare a semantic task and its optional tensor operation."""
        if any(task.symbol == name for task in self.graph.tasks):
            raise ValueError(f"duplicate operation {name}")
        implementations = implementations or (native(f"{self.graph.workload_id}:{name}@1"),)
        effects = (
            reads
            if changes is None
            else (
                *reads,
                SemanticEffect(changes.declaration.resource, effect, None if source is None else source.symbol),
            )
        )
        task = SemanticTask(
            TaskSymbol(name),
            TimingConstraint(*(AnchorSymbol(value) for value in when.value)),
            tuple(SemanticInput(key, value) for key, value in inputs.items()) + extra_inputs,
            tuple(
                SemanticOutput(
                    output,
                    SemanticOutputKind.HOST if result_schema else output_kind,
                    HostPayloadSchemaId(result_schema),
                )
                for output in outputs
            ),
            implementations,
            effects,
            custom_forward,
            math,
        )
        self.graph = replace(self.graph, tasks=tuple(sorted((*self.graph.tasks, task), key=lambda item: item.symbol)))
        if function is not None:
            optimizer = None if changes is None or changes.optimizer is None else changes.page_update()
            self.operations[task.symbol] = operation_type(
                implementations[0].implementation_id,
                partial(native_body, function),
                function,
                operation_mode(
                    changes,
                    transition=effect in (StateEffectKind.TRANSITION, StateEffectKind.TRANSITION_MAYBE),
                ),
                select=select,
                optimizer=optimizer,
                resource=StateResource("") if changes is None else changes.declaration.resource,
            )
        return Call(task.symbol, outputs)

    def update_task(self, symbol: TaskSymbol, change: Callable[[SemanticTask], SemanticTask]) -> None:
        """Replace one declared task by `change(task)`."""
        self.graph = replace(
            self.graph, tasks=tuple(change(task) if task.symbol == symbol else task for task in self.graph.tasks)
        )

    def bind(
        self, plan: PlanBundle | None = None, *, operations: Mapping[TaskSymbol, TensorOperation]
    ) -> tuple[TensorOperation, ...]:
        """Bind declared tensor operations to the implementations selected by the plan."""
        functions: list[TensorOperation] = []
        graph = self.graph if plan is None else plan.semantic_graph
        for task in graph.tasks:
            if task.symbol not in operations:
                continue
            identity = planned_implementation(task, plan)
            versions = tuple(
                effect for effect in task.effects if effect.kind is StateEffectKind.READ and effect.version is not None
            )
            operation = replace(operations[task.symbol], implementation_id=identity)
            produced: list[tuple[StateResource, Literal["private", "published"]]] = []
            for effect in versions:
                producer = next(item for item in graph.tasks if item.symbol == effect.version)
                private = any(
                    effect.kind in (StateEffectKind.TRANSITION, StateEffectKind.TRANSITION_MAYBE)
                    for effect in producer.effects
                )
                produced.append((effect.resource, "private" if private else "published"))
            operation = replace(operation, produced=tuple(produced))
            if task.custom_forward is None:
                functions.append(operation)
        return tuple(functions)

    def bind_all(
        self,
        plan: PlanBundle | None = None,
        *,
        model_probe: ModelProbe | None = None,
        operations: Mapping[TaskSymbol, TensorOperation],
    ) -> tuple[TaskGraphImplementation, ...]:
        """Bind tensor operations, custom forwards and publications with conditional input guards."""
        functions: list[TaskGraphImplementation] = list(self.bind(plan, operations=operations))
        graph = self.graph if plan is None else plan.semantic_graph
        for task in graph.tasks:
            if task.symbol in operations and task.custom_forward is None:
                continue
            identity = planned_implementation(task, plan)
            if task.custom_forward is not None:
                assert model_probe is not None and model_probe.causal_domain is not None
                assert self.custom_forward_programs is not None
                functions.append(
                    CustomForwardTask(
                        task,
                        self.custom_forward_programs,
                        model_probe.causal_domain,
                        identity,
                        operation=None
                        if task.symbol not in operations
                        else replace(operations[task.symbol], implementation_id=identity),
                        overlay_math=self.overlay_math,
                    )
                )
            else:
                functions.append(replace(self.publications[task.symbol], implementation_id=identity))
        guarded = {
            item.implementation_id
            for item in (() if plan is None else plan.schedule.tasks)
            if any(graph.tasks[item.task_id].symbol in group.tasks for group in graph.input_groups)
        }
        return tuple(
            ConditionalProducerTask(function) if function.implementation_id in guarded else function
            for function in functions
        )

    def conditional_input(self, borrowed_resources: tuple[str, ...], publication: Call) -> None:
        """Declare one atomic input closure ending at `publication`, the write of the state an activation patch
        reads, without changing the task/state symbols."""
        tasks = {task.symbol: task for task in self.graph.tasks}
        readers = tuple(task.symbol for task in self.graph.tasks if task.custom_forward is not None)
        resource = tasks[publication.symbol].effects[-1].resource
        # The producers are the publication and every task whose result it transitively reads.
        producers, pending = set[TaskSymbol](), [publication.symbol]
        while pending:
            symbol = pending.pop()
            if symbol not in producers:
                producers.add(symbol)
                pending.extend(
                    port.source.task for port in tasks[symbol].inputs if isinstance(port.source, TaskResultInputSource)
                )
        self.graph = replace(
            self.graph,
            input_groups=(
                ConditionalInputGroup(
                    resource,
                    tuple(StateResource(name) for name in sorted(borrowed_resources)),
                    tuple(sorted(producers)),
                    publication.symbol,
                    readers,
                    all(
                        binding.on_missing is MissingPolicy.HOLD
                        for binding in self.effects.activation_bindings
                        if binding.resource == resource
                    ),
                ),
            ),
        )

    def publish(
        self,
        transition: Call,
        state: State,
        *,
        when: When = When.PUBLISH,
        name: str = "",
        schema: str = "page_publication@1",
        output: str = "publication",
        implementations: tuple[TaskImplementationSpec, ...] = (),
    ) -> Call:
        """Declare publication of a transition into the state's next readable version."""
        name = name or f"{state.declaration.resource}.publish"
        implementations = implementations or (native(f"state:{name}@1", "transition/full_root@1"),)
        call = self.compute(
            name,
            inputs={"transition": transition[transition.outputs[0]]},
            outputs=(output,),
            changes=state,
            effect=StateEffectKind.PUBLISH,
            source=transition,
            result_schema=schema,
            when=when,
            implementations=implementations,
        )
        self.publications[call.symbol] = PublishStateTask(
            implementations[0].implementation_id, state.declaration.resource
        )
        return call

    def sampler(self, prediction: Call, stage: SamplerStage) -> None:
        """Send `prediction`'s result to vLLM's sampler at `stage`."""
        self.effects = replace(self.effects, sampler=(*self.effects.sampler, SamplerBinding(prediction.symbol, stage)))

    def trains(self, transition: Call) -> None:
        """Record a @pllm.train method."""
        self.effects = replace(self.effects, training=(*self.effects.training, transition.symbol))

    def exports(self, call: Call) -> None:
        """Expose a task result through the workload export boundary."""
        self.effects = replace(self.effects, exports=(*self.effects.exports, call.symbol))

    def overlay(
        self,
        bindings: tuple[ParameterOverlayFieldBinding, ...],
        transition: Call | None = None,
        *,
        phases: tuple[HiddenPhase, ...] = (),
    ) -> None:
        """Correct the target projections from the bindings' states; `transition` trains one of them in forward."""
        if self.effects.in_forward is not None or (transition is not None and self.effects.overlay_bindings):
            raise ValueError(
                "a @pllm.train_in_forward method drives every projection hook of its workload, so it does not yet "
                "combine with a pllm.model.Adapter; train the adapter's state in that method or use two workloads"
            )
        self.model_reads(tuple(StateResource(binding.resource) for binding in bindings), at_admission=True)
        self.effects = replace(
            self.effects,
            overlay_bindings=(*self.effects.overlay_bindings, *bindings),
            in_forward=self.effects.in_forward if transition is None else transition.symbol,
            overlay_phases=phases,
        )

    def activation_patch(self, bindings: tuple[ActivationPatchBinding, ...]) -> None:
        """Declare activation patches and the model state reads they need."""
        self.model_reads(tuple(binding.resource for binding in bindings), at_admission=True)
        self.effects = replace(self.effects, activation_bindings=(*self.effects.activation_bindings, *bindings))

    def custom_forward(self, task: TaskSymbol) -> None:
        """Select the task that supplies the model's custom forward."""
        self.effects = replace(self.effects, custom_forward=task)

    def model_reads(self, resources: tuple[StateResource, ...], *, at_admission: bool = False) -> None:
        """Note states the model forward reads, so planning keeps their versions readable. Reads `at_admission` see
        the version each request was admitted with, so planning keeps one version per admitted request."""
        graph = replace(self.graph, model_reads=tuple(sorted({*self.graph.model_reads, *resources})))
        if at_admission:
            graph = replace(graph, admission_reads=tuple(sorted({*graph.admission_reads, *resources})))
        self.graph = graph

    @property
    def affects_model(self) -> bool:
        """Whether a declared effect reaches the model, its sampler, training or exports; guidance with beta=0 does
        not, since it declares no sampler stage."""
        return not self.effects.empty or bool(self.sites.rewrites) or bool(self.attention.methods)

    def finish(self) -> tuple[SemanticGraph, TensorLayoutRequest, BackendAttachmentSpec]:
        """Return the graph, layouts and backend effects of a workload that affects the model."""
        if not self.affects_model:
            raise ValueError(
                "the workload has no effect on the model: add @pllm.train, @pllm.add_logits, @pllm.guide_sampling, "
                "@pllm.export, @pllm.train_in_forward, @pllm.custom_residual, @pllm.custom_attention, a custom_forward or a "
                "pllm.model.ActivationPatch/Adapter"
            )
        return self.graph, self.layouts, self.effects
