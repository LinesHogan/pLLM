"""Build semantic graphs and bind planned implementations for workload declarations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from functools import partial
from typing import Generic, Literal, TypeVar

from pllm.contracts.backend_attachment import (
    ActivationPatchBinding,
    AttachmentKind,
    AttachmentTaskRole,
    BackendAttachmentSpec,
    BackendPlacementProbe,
    MissingPolicy,
)
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import FieldView, ParameterOverlayFieldBinding
from pllm.contracts.request_identity import RequestKeyField
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import (
    AnchorSymbol,
    ConditionalInputGroup,
    ExecutionKind,
    HostPayloadSchemaId,
    HostServiceId,
    HostServiceInputSource,
    HostServicePort,
    ImplementationId,
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
    TaskResultInputSource,
    TaskSymbol,
    TimingConstraint,
    TokenProgramEffect,
    WorkloadId,
)
from pllm.contracts.tensor_layout import (
    LayoutFieldId,
    StateTensorLayout,
    TensorDType,
    TensorFieldSpec,
    TensorLayoutRequest,
    TensorLayoutSpec,
)
from pllm.execution.execution import TaskGraphImplementation
from pllm.execution.overlay_task import OverlayTransition
from pllm.execution.ptts.request_state import RequestStateSelection
from pllm.execution.replay import TokenwiseReplayTask
from pllm.execution.state_execution import (
    ConditionalProducerTask,
    ProjectionResiduals,
    PublishStateTask,
    StateTaskContext,
)
from pllm.execution.tensor_program import (
    Optimizer,
    PageUpdate,
    TaskOutput,
    TensorBatch,
    TensorOperation,
    native_body,
)
from pllm.execution.token_program import TokenProgram
from pllm.kernels.adamw import AdamWConfig

D = TypeVar("D")
OperationMode = Literal["read", "transition", "update", "gradient"]


@dataclass(frozen=True, slots=True)
class Dimensions:
    hidden_size: int
    vocab_size: int
    target_shapes: tuple[tuple[int, int], ...] = ()
    dtype: TensorDType = TensorDType.FLOAT32
    layer_count: int = 0


Shape = tuple[int, ...] | Callable[[Dimensions], tuple[int, ...]]


class When(Enum):
    AUTOMATIC = ("request.prepare", "next.prepare")
    BEFORE_SAMPLE = ("model.execute", "sampler.pre_draw")
    AFTER_SAMPLE = ("token.commit", "next.prepare")
    BEFORE_MODEL = ("request.prepare", "model.execute")
    MODEL_UPDATE = ("model.execute", "state.publish")
    PUBLISH = ("state.publish", "next.prepare")
    HOST_COMPUTE = ("host.input.ready", "host.compute.end")
    HOST_OUTPUT = ("host.compute.end", "host.output.publish")
    HOST_COMMIT = ("host.output.publish", "host.next")
    HOST = ("host.input.ready", "host.next")
    OBSERVE = ("state.publish", "sampler.pre_draw")


@dataclass(frozen=True, slots=True)
class Field:
    name: str
    shape: Shape
    dtype: TensorDType | Callable[[Dimensions], TensorDType] = TensorDType.FLOAT32
    alignment: int = 16


# The fields AdamW keeps per parameter field, in layout order.
ADAM_FIELDS = ("parameters", "gradient", "step", "exp_avg", "exp_avg_sq")


def adam_fields(shape: Shape, *, dtype: TensorDType = TensorDType.FLOAT32, alignment: int = 16) -> tuple[Field, ...]:
    return tuple(
        Field(name, () if name == "step" else shape, TensorDType.FLOAT32 if name == "step" else dtype, alignment)
        for name in ADAM_FIELDS
    )


@dataclass(frozen=True, slots=True)
class State:
    declaration: SemanticState
    fields: tuple[Field, ...]
    optimizer: AdamWConfig | None = None
    # Count of leading optimizer-expanded fields; trailing fields are plain memory.
    optimizer_fields: int = 0

    def read(self, phase: HiddenPhase = HiddenPhase.DECODE) -> StateSelectionInputSource:
        return StateSelectionInputSource(self.declaration.resource, phase)

    def field(self, name: str) -> LayoutFieldId:
        return LayoutFieldId(next(i for i, field in enumerate(self.fields) if field.name == name))

    def view(self, name: str, dimensions: Dimensions | None) -> FieldView:
        """Describe a whole allocated field in the model's resolved layout."""
        index = self.field(name)
        shape = self.fields[index].shape
        return FieldView(
            index,
            0,
            shape(dimensions) if callable(shape) and dimensions is not None else () if callable(shape) else shape,
        )

    def version(self, producer: Call | None = None) -> SemanticEffect:
        return SemanticEffect(
            self.declaration.resource, StateEffectKind.READ, None if producer is None else producer.symbol
        )

    def page_update(self) -> PageUpdate:
        """The AdamW update over each parameter's optimizer fields."""
        assert self.optimizer is not None
        return PageUpdate(self.optimizer, self.optimizer_fields // len(ADAM_FIELDS))

    @property
    def copy_fields(self) -> tuple[LayoutFieldId, ...]:
        """The fields an optimizer update carries into the next version: all but each parameter's gradient."""
        return tuple(
            LayoutFieldId(i)
            for i in range(len(self.fields))
            if i >= self.optimizer_fields or ADAM_FIELDS[i % len(ADAM_FIELDS)] != "gradient"
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """The parameter field names AdamW was expanded from."""
        return tuple(
            field.name.removesuffix(".parameters") for field in self.fields[: self.optimizer_fields : len(ADAM_FIELDS)]
        )


def operation_mode(changes: State | None, *, transition: bool, optimizer: bool) -> OperationMode:
    """How an operation changes its state: not at all, in place, or through a private transition that an optimizer
    applies gradients to or the operation writes itself."""
    if changes is None:
        return "read"
    return ("gradient" if optimizer else "transition") if transition else "update"


def planned_implementation(task: SemanticTask, plan: PlanBundle | None) -> ImplementationId:
    """The implementation the plan selected for `task`, or its declared one when binding without a plan."""
    if plan is None:
        return task.implementations[0].implementation_id
    return next(
        item.implementation_id for item in plan.schedule.tasks if plan.semantic_graph.tasks[item.task_id] is task
    )


@dataclass(frozen=True, slots=True)
class Call:
    symbol: TaskSymbol
    outputs: tuple[str, ...]

    def __getitem__(self, name: str) -> TaskResultInputSource:
        if name not in self.outputs:
            raise ValueError(f"{self.symbol}: unknown output {name}")
        return TaskResultInputSource(self.symbol, name)


def native(identity: str, *capabilities: str) -> TaskImplementationSpec:
    return TaskImplementationSpec(ImplementationId(identity), ExecutionKind.NATIVE_TASK, capabilities)


class Builder(Generic[D]):
    """One original graph; references only name its inputs, states and outputs."""

    def __init__(self, name: str, digest: str, dimensions: Dimensions | None = None) -> None:
        self.graph = SemanticGraph(WorkloadId(name), (), digest)
        self.layouts = TensorLayoutRequest(())
        self.attachment: BackendAttachmentSpec | None = None
        self.dimensions = dimensions
        self.operations: dict[TaskSymbol, TensorOperation[D]] = {}
        self.publications: dict[TaskSymbol, PublishStateTask[D]] = {}
        # Fields that start at zero in a fresh page pool, so admission need not write them (AdamW state other than
        # the parameters). Request-scoped pages are recycled and dirty, so only instance states qualify.
        self.zero_start_fields: dict[str, frozenset[int]] = {}
        self.route_program: dict[str, TokenProgram] | None = None
        self.overlay_math: OverlayTransition | None = None
        self.route_updates = False
        self.route_rows = 0

    def state(
        self,
        name: str,
        fields: tuple[Field, ...] | None = None,
        *,
        namespace: str = "",
        scope: StateScope = StateScope.INSTANCE,
        keys: tuple[RequestKeyField, ...] = (),
        optimizer: AdamWConfig | None = None,
        memory: tuple[Field, ...] = (),
    ) -> State:
        if fields is None:
            assert optimizer is None and not memory, "an optimizer and memory need declared parameter fields"
            declaration = SemanticState(StateResource(name), scope, keys)
            self.graph = replace(
                self.graph, states=tuple(sorted((*self.graph.states, declaration), key=lambda state: state.resource))
            )
            return State(declaration, ())
        optimizer_fields = 0
        if optimizer is not None:
            fields = tuple(
                replace(
                    part,
                    name=f"{field.name}.{part.name}",
                    dtype=TensorDType.FLOAT32 if part.name == "step" else field.dtype,
                )
                for field in fields
                for part in adam_fields(field.shape, alignment=field.alignment)
            )
            optimizer_fields = len(fields)
        fields = (*fields, *memory)
        if len({field.name for field in fields}) != len(fields):
            raise ValueError(f"{name}: duplicate field name")
        declaration = SemanticState(StateResource(name), scope, keys)
        self.graph = replace(
            self.graph, states=tuple(sorted((*self.graph.states, declaration), key=lambda state: state.resource))
        )
        if self.dimensions is not None or all(
            not callable(field.shape) and not callable(field.dtype) for field in fields
        ):
            shapes: list[tuple[int, ...]] = []
            for field in fields:
                if callable(field.shape):
                    assert self.dimensions is not None
                    shapes.append(field.shape(self.dimensions))
                else:
                    shapes.append(field.shape)

            def dtype(field: Field) -> TensorDType:
                if callable(field.dtype):
                    assert self.dimensions is not None
                    return field.dtype(self.dimensions)
                return field.dtype

            layout = TensorLayoutSpec(
                namespace or f"{self.graph.workload_id}.{name}@1",
                tuple(
                    TensorFieldSpec(LayoutFieldId(i), shape, dtype(field), field.alignment)
                    for i, (field, shape) in enumerate(zip(fields, shapes, strict=True))
                ),
            )
            self.layouts = TensorLayoutRequest(
                tuple(
                    sorted((*self.layouts.layouts, StateTensorLayout(name, layout)), key=lambda entry: entry.resource)
                )
            )
        if optimizer is not None and scope is StateScope.INSTANCE:
            self.zero_start_fields[name] = frozenset(
                i for i in range(optimizer_fields) if ADAM_FIELDS[i % len(ADAM_FIELDS)] != "parameters"
            )
        return State(declaration, fields, optimizer, optimizer_fields)

    @staticmethod
    def input(service: str, port: str, schema: str) -> HostServiceInputSource:
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
        replay: TokenProgramEffect | None = None,
        math: MathContract | None = None,
        function: Callable[[D, TensorBatch], TaskOutput] | None = None,
        operation_type: type[TensorOperation[D]] = TensorOperation,
        select: Callable[[D, RequestStateSelection], tuple[int, ...]] | None = None,
        optimizer: Optimizer | None = None,
    ) -> Call:
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
            replay,
            math,
        )
        self.graph = replace(self.graph, tasks=tuple(sorted((*self.graph.tasks, task), key=lambda item: item.symbol)))
        if function is not None:
            copy_fields: tuple[LayoutFieldId, ...] | None = None
            if changes is not None and changes.optimizer is not None and optimizer is None:
                copy_fields, optimizer = changes.copy_fields, changes.page_update()
            self.operations[task.symbol] = operation_type(
                implementations[0].implementation_id,
                partial(native_body, function),
                operation_mode(
                    changes,
                    transition=effect in (StateEffectKind.TRANSITION, StateEffectKind.TRANSITION_MAYBE),
                    optimizer=optimizer is not None,
                ),
                select=select,
                optimizer=optimizer,
                copy_fields=copy_fields,
            )
        return Call(task.symbol, outputs)

    def update_task(self, symbol: TaskSymbol, change: Callable[[SemanticTask], SemanticTask]) -> None:
        """Replace one declared task by `change(task)`."""
        self.graph = replace(
            self.graph, tasks=tuple(change(task) if task.symbol == symbol else task for task in self.graph.tasks)
        )

    def bind(
        self, plan: PlanBundle | None = None, *, operations: Mapping[TaskSymbol, TensorOperation[D]]
    ) -> tuple[TensorOperation[D], ...]:
        functions: list[TensorOperation[D]] = []
        graph = self.graph if plan is None else plan.semantic_graph
        for task in graph.tasks:
            if task.symbol not in operations:
                continue
            identity = planned_implementation(task, plan)
            versions = tuple(
                effect.version
                for effect in task.effects
                if effect.kind is StateEffectKind.READ and effect.version is not None
            )
            version = operations[task.symbol].version
            if versions:
                producer = next(item for item in graph.tasks if item.symbol == versions[0])
                version = (
                    "private"
                    if any(
                        effect.kind in (StateEffectKind.TRANSITION, StateEffectKind.TRANSITION_MAYBE)
                        for effect in producer.effects
                    )
                    else "published"
                )
            if task.replay is None:
                functions.append(replace(operations[task.symbol], implementation_id=identity, version=version))
        return tuple(functions)

    def bind_all(
        self,
        plan: PlanBundle | None = None,
        *,
        model_probe: BackendPlacementProbe | None = None,
        operations: Mapping[TaskSymbol, TensorOperation[D]],
    ) -> tuple[TaskGraphImplementation[StateTaskContext[D], ProjectionResiduals], ...]:
        functions: list[TaskGraphImplementation[StateTaskContext[D], ProjectionResiduals]] = list(
            self.bind(plan, operations=operations)
        )
        graph = self.graph if plan is None else plan.semantic_graph
        for task in graph.tasks:
            if task.symbol in operations and task.replay is None:
                continue
            identity = planned_implementation(task, plan)
            if task.replay is not None:
                assert model_probe is not None and model_probe.causal_domain is not None
                assert self.route_program is not None
                functions.append(
                    TokenwiseReplayTask[D](
                        task,
                        self.route_program,
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

    def conditional_input(self, borrowed_resources: tuple[str, ...]) -> None:
        """Declare one atomic input closure without changing the task/state symbols."""
        if self.attachment is None or not self.attachment.activation_bindings or self.graph.input_groups:
            raise ValueError("conditional input requires one original activation chain")
        publication = self.attachment.tasks[-1][1]
        readers = tuple(task.symbol for task in self.graph.tasks if task.replay is not None)
        self.graph = replace(
            self.graph,
            input_groups=(
                ConditionalInputGroup(
                    self.attachment.state_resource,
                    tuple(StateResource(name) for name in sorted(borrowed_resources)),
                    tuple(task.symbol for task in self.graph.tasks if task.symbol not in readers),
                    publication,
                    readers,
                    all(binding.on_missing is MissingPolicy.HOLD for binding in self.attachment.activation_bindings),
                ),
            ),
        )

    def publish(
        self,
        transition: Call,
        state: State,
        *,
        when: When = When.PUBLISH,
        name: str = "publish",
        schema: str = "canonical.page_publication@1",
        output: str = "publication",
        implementations: tuple[TaskImplementationSpec, ...] = (),
    ) -> Call:
        implementations = implementations or (native("canonical.state:publish@1", "transition/full_root@1"),)
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
        self.publications[call.symbol] = PublishStateTask(implementations[0].implementation_id)
        return call

    def sampler(
        self,
        prediction: Call,
        transition: Call,
        state: State,
        *,
        output: str = "guidance",
        stage: SamplerStage = SamplerStage.PACKED_POST_FILTER_EXACT,
    ) -> None:
        prediction[output]
        # The sampler attachment keeps an earlier pllm.model.Overlay's bindings; an activation patch would need
        # this attachment's state_resource, which names the trained state.
        previous = self.attachment
        if previous is not None and previous.activation_bindings:
            raise ValueError(
                "@pllm.add_logits and @pllm.guide_sampling do not combine with pllm.model.Activation; remove one of them"
            )
        self.attachment = BackendAttachmentSpec(
            AttachmentKind.REQUEST_TRAINING,
            ((AttachmentTaskRole.COMPUTE, prediction.symbol), (AttachmentTaskRole.TRANSITION, transition.symbol)),
            AnchorSymbol("sampler.pre_draw"),
            state.declaration.resource,
            () if previous is None else previous.overlay_bindings,
            sampler_stage=stage,
        )

    def request_training(self, transition: Call, state: State) -> None:
        """Train request state from hidden rows after the token transaction commits."""
        self.attachment = BackendAttachmentSpec(
            AttachmentKind.REQUEST_TRAINING,
            ((AttachmentTaskRole.TRANSITION, transition.symbol),),
            AnchorSymbol("sampler.pre_draw"),
            state.declaration.resource,
        )

    def overlay(
        self,
        state: State,
        producer: tuple[Call, Call] | None,
        bindings: tuple[ParameterOverlayFieldBinding, ...] = (),
        *,
        activations: tuple[ActivationPatchBinding, ...] = (),
    ) -> None:
        self._attach(
            BackendAttachmentSpec(
                AttachmentKind.ACTIVATION_PATCH if activations else AttachmentKind.PARAMETER_OVERLAY,
                ()
                if producer is None
                else (
                    (AttachmentTaskRole.TRANSITION, producer[0].symbol),
                    (AttachmentTaskRole.PUBLISH, producer[1].symbol),
                ),
                AnchorSymbol("sampler.complete"),
                state.declaration.resource,
                bindings,
                activations,
            )
        )

    def finish(self) -> tuple[SemanticGraph, TensorLayoutRequest, BackendAttachmentSpec]:
        if self.attachment is None:
            raise ValueError(
                "the workload has no effect on the model: add @pllm.train, @pllm.add_logits, @pllm.guide_sampling, "
                "@pllm.export, @pllm.overlay, a layer_route or a pllm.model.Activation/Overlay"
            )
        return self.graph, self.layouts, self.attachment

    def model_effect(self, call: Call, kind: AttachmentKind = AttachmentKind.OBSERVATION) -> None:
        self._attach(
            BackendAttachmentSpec(
                kind, ((AttachmentTaskRole.COMPUTE, call.symbol),), AnchorSymbol("sampler.pre_draw"), StateResource("")
            )
        )

    def _attach(self, spec: BackendAttachmentSpec) -> None:
        self.graph = replace(
            self.graph,
            model_reads=tuple(
                sorted(
                    set(self.graph.model_reads)
                    | {StateResource(binding.resource) for binding in spec.overlay_bindings}
                    | ({spec.state_resource} if spec.activation_bindings else set())
                )
            ),
        )
        previous = self.attachment
        if previous is None or previous.kind is spec.kind is AttachmentKind.OBSERVATION:
            self.attachment = spec if previous is None else replace(previous, tasks=previous.tasks + spec.tasks)
            return
        if {previous.kind, spec.kind} != {AttachmentKind.ACTIVATION_PATCH, AttachmentKind.MODEL_REPLAY}:
            raise ValueError("only activation and tokenwise model effects can share an attachment")
        activation, replay = (previous, spec) if previous.kind is AttachmentKind.ACTIVATION_PATCH else (spec, previous)
        if replay.activation_bindings or replay.overlay_bindings or activation.overlay_bindings:
            raise ValueError("model replay requires one activation publication")
        ((_, forward_symbol),) = replay.tasks
        publication_symbol = activation.tasks[-1][1] if activation.tasks else None
        forward = next(task for task in self.graph.tasks if task.symbol == forward_symbol)
        publication = next((task for task in self.graph.tasks if task.symbol == publication_symbol), None)
        if forward.effects or any(port.name == "activation_publication" for port in forward.inputs):
            raise ValueError("model replay already declares state dependencies")
        dependency = (
            ()
            if publication is None
            else (
                SemanticInput(
                    "activation_publication", TaskResultInputSource(publication.symbol, publication.outputs[0].name)
                ),
            )
        )
        # Preserve both the value dependency and the exact state-version read.
        forward = replace(
            forward,
            inputs=(*forward.inputs, *dependency),
            effects=(SemanticEffect(activation.state_resource, StateEffectKind.READ, publication_symbol),),
        )
        self.update_task(forward_symbol, lambda _: forward)
        self.attachment = replace(
            replay,
            tasks=(*replay.tasks, *activation.tasks),
            state_resource=activation.state_resource,
            activation_bindings=activation.activation_bindings,
        )
