"""Local authoring references that construct the original semantic graph directly."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Generic, TypeVar

import torch

from pllm.contracts.backend_attachment import (
    ActivationPatchBinding, AttachmentKind, AttachmentTaskRole, BackendAttachmentSpec, BackendPlacementProbe,
)
from pllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from pllm.contracts.parameter_overlay import ParameterOverlayFieldBinding
from pllm.contracts.request_identity import RequestKeyField
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import (
    AnchorSymbol, ConditionalLayerReplay, ExecutionKind, HiddenInputSource, HostPayloadSchemaId,
    HostServiceId, HostServiceInputSource, HostServicePort, ImplementationId, SemanticEffect,
    SemanticGraph, SemanticInput, SemanticInputSource, SemanticOutput, SemanticOutputKind,
    SemanticState, SemanticTask, StateEffectKind, StateResource, StateScope, StateSelectionInputSource,
    TaskImplementationSpec, TaskResultInputSource, TaskSymbol, TimingConstraint, TokenwiseLayerReplay,
    WorkloadId, PackedRowsInputSource, PackedSamplerRowsInputSource, SamplerFeedbackInputSource, MathContract,
    StepIdInputSource, TaskResourceFact,
)
from pllm.kernels.adamw import AdamWConfig, AdamWFields, apply_adamw
from pllm.contracts.tensor_layout import (
    LayoutFieldId, MutationGranularity, StateTensorLayout, TensorDType, TensorFieldSpec,
    TensorLayoutRequest, TensorLayoutSpec,
)
from pllm.contracts.compilation import PlanBundle
from pllm.execution.tensor_program import TensorBatch, TensorOperation, TaskOutput
from pllm.execution.execution import TaskGraphImplementation
from pllm.execution.state_execution import StateTaskContext, PublishStateTask
from pllm.execution.ptts.request_state import RequestStateSelection
from pllm.execution.replay import ReplayStateTask, TokenwiseReplayTask, REPLAY_IMPLEMENTATION
from pllm.execution.region_replay import ConditionalReplayRegion
from pllm.contracts.region_program import TensorType

D = TypeVar("D")


@dataclass(frozen=True, slots=True)
class Dimensions:
    hidden_size: int
    vocab_size: int
    target_shapes: tuple[tuple[int, int], ...] = ()
    dtype: TensorDType = TensorDType.FLOAT32
    layer_count: int = 0


Shape = tuple[int, ...] | Callable[[Dimensions], tuple[int, ...]]


class When(Enum):
    BEFORE_SAMPLE = ("model.execute", "sampler.pre_draw")
    AFTER_SAMPLE = ("token.commit", "next.prepare")
    BEFORE_MODEL = ("request.prepare", "model.execute")
    MODEL_UPDATE = ("model.execute", "state.publish")
    PUBLISH = ("state.publish", "next.prepare")
    REPLAY = ("state.publish", "sampler.pre_draw")
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
    mutation: MutationGranularity = MutationGranularity.WHOLE_TENSOR


def adam_fields(shape: Shape, *, dtype: TensorDType = TensorDType.FLOAT32,
                alignment: int = 16, mutation: MutationGranularity = MutationGranularity.WHOLE_TENSOR) -> tuple[Field, ...]:
    return tuple(Field(name, () if name == "step" else shape,
        TensorDType.FLOAT32 if name == "step" else dtype, alignment, mutation)
        for name in ("parameters", "gradient", "step", "exp_avg", "exp_avg_sq"))


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

    def version(self, producer: Call | None = None) -> SemanticEffect:
        return SemanticEffect(self.declaration.resource, StateEffectKind.READ,
            None if producer is None else producer.symbol)


@dataclass(frozen=True, slots=True)
class Call:
    symbol: TaskSymbol
    outputs: tuple[str, ...]

    def __getitem__(self, name: str) -> TaskResultInputSource:
        if name not in self.outputs:
            raise ValueError(f"{self.symbol}: unknown output {name}")
        return TaskResultInputSource(self.symbol, name)


def native(identity: str, *capabilities: str, host: bool = False) -> TaskImplementationSpec:
    return TaskImplementationSpec(ImplementationId(identity),
        ExecutionKind.HOST_SERVICE if host else ExecutionKind.NATIVE_TASK, capabilities)


@dataclass(frozen=True, slots=True)
class HostOperation(Generic[D]):
    implementation_id: ImplementationId
    function: Callable[[D], None]

    def execute(self, context: D) -> None:
        self.function(context)


class Builder(Generic[D]):
    """One original graph; references only name its inputs, states and outputs."""

    def __init__(self, name: str, digest: str, dimensions: Dimensions | None = None,
                 resources: tuple[TaskResourceFact, ...] = ()) -> None:
        self.graph = SemanticGraph(WorkloadId(name), (), digest)
        self.layouts = TensorLayoutRequest(())
        self.attachment: BackendAttachmentSpec | None = None
        self.dimensions = dimensions
        self.operations: dict[TaskSymbol, TensorOperation[D]] = {}
        self.host_operations: dict[TaskSymbol, HostOperation[D]] = {}
        self.resources = resources

    def state(self, name: str, fields: tuple[Field, ...] | None = None, *, namespace: str = "",
              scope: StateScope = StateScope.INSTANCE, keys: tuple[RequestKeyField, ...] = (),
              optimizer: AdamWConfig | None = None, memory: tuple[Field, ...] = ()) -> State:
        if fields is None:
            if optimizer is not None or memory:
                raise ValueError("an optimizer requires declared parameter fields")
            declaration = SemanticState(StateResource(name), scope, keys)
            self.graph = replace(self.graph, states=(*self.graph.states, declaration))
            return State(declaration, ())
        if memory and any(part.name in {field.name for field in fields} for part in memory):
            raise ValueError(f"{name}: memory fields duplicate a parameter field")
        optimizer_fields = 0
        if optimizer is not None:
            fields = tuple(replace(part, name=f"{field.name}.{part.name}",
                dtype=TensorDType.FLOAT32 if part.name == "step" else field.dtype)
                for field in fields for part in adam_fields(field.shape,
                    alignment=field.alignment, mutation=field.mutation))
            optimizer_fields = len(fields)
        fields = (*fields, *memory)
        if len({field.name for field in fields}) != len(fields):
            raise ValueError(f"{name}: duplicate field name")
        declaration = SemanticState(StateResource(name), scope, keys)
        self.graph = replace(self.graph, states=(*self.graph.states, declaration))
        if self.dimensions is not None or all(not callable(field.shape) and not callable(field.dtype) for field in fields):
            shapes: list[tuple[int, ...]] = []
            for field in fields:
                if callable(field.shape):
                    if self.dimensions is None:
                        raise ValueError("model dimensions are not specialized")
                    shapes.append(field.shape(self.dimensions))
                else:
                    shapes.append(field.shape)
            def dtype(field: Field) -> TensorDType:
                if callable(field.dtype):
                    if self.dimensions is None:
                        raise ValueError("model dtype is not specialized")
                    return field.dtype(self.dimensions)
                return field.dtype
            layout = TensorLayoutSpec(namespace or f"{self.graph.workload_id}.{name}@1", tuple(TensorFieldSpec(LayoutFieldId(i), shape,
                dtype(field), field.alignment, field.mutation) for i, (field, shape) in enumerate(zip(fields, shapes, strict=True))))
            self.layouts = TensorLayoutRequest((*self.layouts.layouts, StateTensorLayout(name, layout)))
        return State(declaration, fields, optimizer, optimizer_fields)

    @staticmethod
    def hidden(layer: int, *, site: HiddenSite = HiddenSite.BLOCK_OUTPUT,
               phase: HiddenPhase = HiddenPhase.DECODE) -> HiddenInputSource:
        return HiddenInputSource(layer, site, phase)

    @staticmethod
    def input(service: str, port: str, schema: str) -> HostServiceInputSource:
        return HostServiceInputSource(HostServiceId(service), HostServicePort(port), HostPayloadSchemaId(schema))

    rows = staticmethod(PackedRowsInputSource)
    sampler_rows = staticmethod(PackedSamplerRowsInputSource)
    feedback = staticmethod(SamplerFeedbackInputSource)
    step_id = staticmethod(StepIdInputSource)

    def compute(self, name: str, *, inputs: Mapping[str, SemanticInputSource],
                outputs: tuple[str, ...] = (), implementations: tuple[TaskImplementationSpec, ...] = (),
                when: When, reads: tuple[SemanticEffect, ...] = (),
                changes: State | None = None, effect: StateEffectKind = StateEffectKind.UPDATE_MAYBE,
                source: Call | None = None, result_schema: str = "", output_schemas: Mapping[str, str] | None = None,
                replay: ConditionalLayerReplay | TokenwiseLayerReplay | None = None,
                math: MathContract | None = None,
                function: Callable[[D, TensorBatch], TaskOutput] | None = None,
                host_function: Callable[[D], None] | None = None,
                select: Callable[[D, RequestStateSelection], tuple[int, ...]] | None = None,
                optimizer: Callable[[tuple[tuple[torch.Tensor, ...], ...], tuple[torch.Tensor, ...]], None] | None = None) -> Call:
        if any(task.symbol == name for task in self.graph.tasks):
            raise ValueError(f"duplicate operation {name}")
        if function is not None and host_function is not None:
            raise ValueError(f"{name}: choose tensor or host mathematics")
        schemas = dict.fromkeys(outputs, result_schema) if output_schemas is None else output_schemas
        if any(output not in outputs for output in schemas):
            raise ValueError(f"{name}: schema references an unknown output")
        implementations = implementations or (native(f"{self.graph.workload_id}:{name}@1"),)
        effects = reads if changes is None else (*reads, SemanticEffect(changes.declaration.resource,
            effect, None if source is None else source.symbol))
        task = SemanticTask(TaskSymbol(name), TimingConstraint(*(AnchorSymbol(value) for value in when.value)),
            tuple(SemanticInput(key, value) for key, value in inputs.items()),
            tuple(SemanticOutput(output, SemanticOutputKind.HOST if schemas.get(output) else SemanticOutputKind.TEMPORARY,
                HostPayloadSchemaId(schemas.get(output, ""))) for output in outputs), implementations, effects, replay, math)
        self.graph = replace(self.graph, tasks=tuple(sorted((*self.graph.tasks, task), key=lambda item: item.symbol)))
        if function is not None:
            groups = changes.optimizer_fields // 5 if changes is not None and changes.optimizer is not None else 1
            copy_fields: tuple[LayoutFieldId, ...] | None = None
            if changes is not None and changes.optimizer is not None and optimizer is None:
                config, groups = changes.optimizer, changes.optimizer_fields // 5
                copy_fields = tuple(LayoutFieldId(i) for i in range(len(changes.fields))
                    if i >= changes.optimizer_fields or i % 5 != 1)
                def update(roots: tuple[tuple[torch.Tensor, ...], ...], gradients: tuple[torch.Tensor, ...]) -> None:
                    apply_adamw(tuple(root[5 * group:5 * group + 5] for root in roots for group in range(groups)),
                        AdamWFields(*(LayoutFieldId(i) for i in range(5))), config, gradients)
                optimizer = update
            self.operations[task.symbol] = TensorOperation(implementations[0].implementation_id, function,
                "read" if changes is None else ("gradient" if optimizer is not None else "transition") if effect in (StateEffectKind.TRANSITION, StateEffectKind.TRANSITION_MAYBE)
                else "update", select=select, optimizer=optimizer, gradient_fields=groups, copy_fields=copy_fields)
        if host_function is not None:
            if len(implementations) != 1:
                raise ValueError("host function requires one explicit implementation")
            self.host_operations[task.symbol] = HostOperation(implementations[0].implementation_id, host_function)
        return Call(task.symbol, outputs)

    def bind_host(self) -> tuple[HostOperation[D], ...]:
        if set(self.host_operations) != {task.symbol for task in self.graph.tasks}:
            raise ValueError("host graph requires a function for every operation")
        return tuple(self.host_operations[task.symbol] for task in self.graph.tasks)

    def bind(self, plan: PlanBundle | None = None) -> tuple[TensorOperation[D], ...]:
        functions: list[TensorOperation[D]] = []
        graph = self.graph if plan is None else plan.semantic_graph
        for task in graph.tasks:
            if task.symbol not in self.operations:
                continue
            if plan is None and len(task.implementations) != 1:
                raise ValueError("multiple implementations require the original signed plan")
            identity = task.implementations[0].implementation_id if plan is None else next(
                item.implementation_id for item in plan.schedule_physical.tasks if graph.tasks[item.task_id] is task)
            versions = tuple(effect.version for effect in task.effects if effect.kind is StateEffectKind.READ and effect.version is not None)
            version = "input"
            if versions:
                producer = next(item for item in graph.tasks if item.symbol == versions[0])
                version = "private" if any(effect.kind in (StateEffectKind.TRANSITION, StateEffectKind.TRANSITION_MAYBE)
                    for effect in producer.effects) else "published"
            functions.append(replace(self.operations[task.symbol], implementation_id=identity, version=version))
        return tuple(functions)

    def bind_all(self, plan: PlanBundle | None = None, *,
                 model_probe: BackendPlacementProbe | None = None) -> tuple[TaskGraphImplementation[StateTaskContext[D]], ...]:
        functions: list[TaskGraphImplementation[StateTaskContext[D]]] = list(self.bind(plan))
        graph = self.graph if plan is None else plan.semantic_graph
        for task in graph.tasks:
            if task.symbol in self.operations:
                continue
            if plan is None and len(task.implementations) != 1:
                raise ValueError("multiple implementations require the original signed plan")
            identity = task.implementations[0].implementation_id if plan is None else next(
                item.implementation_id for item in plan.schedule_physical.tasks if graph.tasks[item.task_id] is task)
            if isinstance(task.replay, ConditionalLayerReplay):
                if plan is None or self.dimensions is None:
                    raise ValueError("conditional replay requires a specialized backend plan")
                placements = {(r.device_kind.value, r.device_ordinal) for r in plan.state_physical.representations}
                if len(placements) != 1:
                    raise ValueError("conditional replay requires one original state device")
                kind, ordinal = next(iter(placements))
                tensor = TensorType((1, self.dimensions.hidden_size), "torch." + self.dimensions.dtype.value,
                    "cpu" if kind == "cpu" else f"cuda:{ordinal}")
                functions.append(ReplayStateTask[D](task, ConditionalReplayRegion(tensor, task.replay), identity))
            elif isinstance(task.replay, TokenwiseLayerReplay):
                if plan is None or model_probe is None or model_probe.causal_domain is None:
                    raise ValueError("tokenwise replay requires a specialized backend plan and model layout")
                tensor = TensorType((1, model_probe.hidden_size), "torch." + model_probe.tensor_dtype.value,
                    "cpu" if model_probe.device_kind.value == "cpu" else f"cuda:{model_probe.device_ordinal}")
                workspace = next((fact.workspace_bytes for fact in plan.timeline.task_resources if fact.task == task.symbol), None)
                if workspace is None:
                    raise ValueError("tokenwise replay requires its original signed task workspace")
                functions.append(TokenwiseReplayTask[D].bind(task, tensor, model_probe.max_packed_rows,
                    model_probe.causal_domain, identity, plan.timeline.region_policy, workspace,
                    stream_lanes=2 if "execution/region_streams@1" in model_probe.capabilities else 1,
                    source_imports=model_probe.region_imports, function_bodies=model_probe.region_bodies))
            elif any(effect.kind is StateEffectKind.PUBLISH for effect in task.effects):
                functions.append(PublishStateTask[D](identity))
            else:
                raise ValueError(f"{task.symbol}: no explicit native implementation")
        return tuple(functions)

    def publish(self, transition: Call, state: State, *, when: When = When.PUBLISH,
                name: str = "publish", schema: str = "canonical.page_publication@1",
                output: str = "publication", implementations: tuple[TaskImplementationSpec, ...] = ()) -> Call:
        return self.compute(name, inputs={"transition": transition[transition.outputs[0]]}, outputs=(output,),
            changes=state, effect=StateEffectKind.PUBLISH, source=transition, result_schema=schema, when=when,
            implementations=implementations or (native("canonical.state:publish@1", "transition/full_root@1"),))

    def replay(self, publication: Call, state: State, effect: ConditionalLayerReplay) -> Call:
        return self.compute("replay", inputs={"publication": publication[publication.outputs[0]]},
            reads=(state.version(publication),), when=When.REPLAY, replay=effect,
            implementations=(native(REPLAY_IMPLEMENTATION, "model/layer_replay@1"),))

    def sampler(self, prediction: Call, transition: Call, state: State, *,
                output: str = "guidance", stage: SamplerStage = SamplerStage.PACKED_POST_FILTER_EXACT) -> None:
        prediction[output]
        self.graph = replace(self.graph, tasks=tuple(replace(task, outputs=tuple(
            replace(value, kind=SemanticOutputKind.SAMPLER) if value.name == output else value
            for value in task.outputs)) if task.symbol == prediction.symbol else task for task in self.graph.tasks))
        self.attachment = BackendAttachmentSpec(AttachmentKind.SAMPLER_TRAINING,
            ((AttachmentTaskRole.COMPUTE, prediction.symbol), (AttachmentTaskRole.TRANSITION, transition.symbol)),
            AnchorSymbol("sampler.pre_draw"), state.declaration.resource, sampler_stage=stage)

    def overlay(self, state: State, transition: Call, publication: Call,
                bindings: tuple[ParameterOverlayFieldBinding, ...] = (), *,
                activations: tuple[ActivationPatchBinding, ...] = ()) -> None:
        self.attachment = BackendAttachmentSpec(AttachmentKind.ACTIVATION_PATCH if activations else AttachmentKind.PARAMETER_OVERLAY,
            ((AttachmentTaskRole.TRANSITION, transition.symbol), (AttachmentTaskRole.PUBLISH, publication.symbol)),
            AnchorSymbol("sampler.complete"), state.declaration.resource, bindings, activations)

    def finish(self) -> tuple[SemanticGraph, TensorLayoutRequest, BackendAttachmentSpec]:
        if self.attachment is None:
            raise ValueError("model workload has no declared backend effect")
        return self.graph, self.layouts, self.attachment

    def model_effect(self, call: Call, kind: AttachmentKind = AttachmentKind.OBSERVATION) -> None:
        self.attachment = BackendAttachmentSpec(kind, ((AttachmentTaskRole.COMPUTE, call.symbol),),
            AnchorSymbol("sampler.pre_draw"), StateResource(""))
