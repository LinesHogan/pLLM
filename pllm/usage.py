"""Cold Python authoring boundary. Reflection stops at the original typed graph."""

from collections.abc import Callable, Mapping
from collections import Counter
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from dis import get_instructions
from enum import Enum
from hashlib import blake2b, sha256
from inspect import Parameter as SignatureParameter, getclosurevars, signature
from marshal import dumps
import sys
from types import CellType, CodeType, FunctionType, MethodType, ModuleType
from typing import Annotated, ClassVar, Generic, TypeVar, cast, get_args, get_origin

import torch

from pllm.authoring import BackendWorkload, BackendCompileFacts, BackendRequirements, CausalModel, CausalTrainBatch, TensorStepInput
from pllm.config import ExecutionOptions
from pllm.contracts.backend_attachment import BackendAttachmentSpec, ActivationPatchOperator, AttachmentKind
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.tensor_layout import MutationGranularity, TensorDType, TensorLayoutRequest
from pllm.contracts.hidden_locator import HiddenSite, HiddenPhase
from pllm.contracts.parameter_overlay import ParameterTarget, ParameterOverlayFieldBinding, ParameterOverlayOperator
from pllm.contracts.request_identity import RequestKeyField
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import LossReduction, SchedulerTimeline, TaskImplementationSpec, TaskResourceFact
from pllm.contracts.semantic_graph import AnchorSymbol, ConditionalLayerReplay, StateEffectKind
from pllm.contracts.semantic_graph import HiddenInputSource, MathContract, SemanticGraph, SemanticInputSource, StateScope, TaskResultInputSource, TaskSymbol, SamplerFeedbackInputSource
from pllm.execution.codec import boundary_digest
from pllm.execution.declaration import Builder, Call, Dimensions, Field, Shape, State, When, native
from pllm.execution.tensor_binding import Buffer as Buffer, TensorUpdate as TensorUpdate
from pllm.execution.tensor_binding import TensorBinding, TensorFrame, TensorPort, Tensors, TensorOutput as TensorOutput, SGD as SGD
from pllm.execution.tensor_program import FunctionMath, Roots, InvocationOutput, TensorBatch, autograd_loss
from pllm.execution.state_execution import MathBinding, StateExecution
from pllm.execution.host_runtime import (
    HostWorkload, HostRuntimeFacts, TensorHostBinding, HostTaskState,
    HostTransactionService, TransactionHostBinding, CompletionInput,
)
from pllm.execution.replay import Recurrence as Recurrence
from pllm.kernels.adamw import AdamWConfig
from pllm.execution.shared_tensor import tensor_dtype

F = TypeVar("F", bound=Callable[..., torch.Tensor | TensorUpdate])
G = TypeVar("G", bound=Callable[..., object])
C = TypeVar("C")
RequestT = TypeVar("RequestT")
TransactionT = TypeVar("TransactionT")
ContextT = TypeVar("ContextT")
SnapshotT = TypeVar("SnapshotT")
ResultT = TypeVar("ResultT")
InputT = TypeVar("InputT")


def transaction(name: str, facts: HostRuntimeFacts, *, config: object,
                state_factory: Callable[[], HostTaskState[TransactionT, ContextT, SnapshotT, ResultT]],
                service: HostTransactionService[RequestT, TransactionT, ResultT],
                update: Callable[[ContextT], None], query: Callable[[ContextT], None],
                publish: Callable[[ContextT], None], timeline: SchedulerTimeline | None = None) -> TransactionHostBinding[
                    RequestT, TransactionT, ContextT, SnapshotT, ResultT]:
    """Bind native structured math; retain the original state, service and context."""
    b = Builder[ContextT](name, boundary_digest(config, facts))
    resource = b.state("state")
    changed = b.compute("update", host_function=update,
        inputs={"context": b.input(name, "transaction", f"{name}.transaction@1")},
        outputs=("context",), changes=resource, when=When.HOST_COMPUTE,
        implementations=(native(f"{name}:update@1", f"compute/{facts.device.type}@1"),))
    queried = b.compute("query", host_function=query, inputs={"context": changed["context"]},
        outputs=("context",), reads=(resource.version(changed),), when=When.HOST_OUTPUT,
        implementations=(native(f"{name}:query@1", f"compute/{facts.device.type}@1"),))
    b.compute("publish", host_function=publish, inputs={"context": queried["context"]},
        outputs=("result",), result_schema=f"{name}.result@1",
        reads=(resource.version(changed),), when=When.HOST_COMMIT,
        implementations=(native(f"{name}:publish@1", "host/service@1", host=True),))
    return TransactionHostBinding(b.graph, state_factory, b.bind_host(), service, facts, timeline)


class Phase(Enum):
    BEFORE_MODEL = When.BEFORE_MODEL
    OBSERVE = When.OBSERVE
    BEFORE_SAMPLE = When.BEFORE_SAMPLE
    AFTER_SAMPLE = When.AFTER_SAMPLE
    MODEL_UPDATE = When.MODEL_UPDATE


@dataclass(frozen=True)
class Config:
    name: str


@dataclass(frozen=True)
class Hidden:
    layer: int | Config = 0
    site: HiddenSite = HiddenSite.BLOCK_OUTPUT


@dataclass(frozen=True)
class Samples:
    limit: int | Config | None = None


@dataclass(frozen=True)
class Batch:
    field: str


@dataclass(frozen=True)
class Parameter:
    state: str
    field: str
    per_row: bool = False


@dataclass(frozen=True)
class StateIndex:
    state: str


@dataclass(frozen=True)
class StateVersion:
    state: str


@dataclass(frozen=True)
class Result:
    operation: str


@dataclass(frozen=True)
class BaseLogits:
    pass


@dataclass(frozen=True)
class SampledTokens:
    pass


class Operand(Enum):
    INPUT = "input"
    TARGET = "target"
    WEIGHT = "weight"


@dataclass(frozen=True)
class InstanceState:
    fields: Mapping[str, Shape | Buffer] = field(default_factory=lambda: dict[str, Shape | Buffer]())
    optimizer: AdamWConfig | SGD | None = field(default=None, kw_only=True)
    initialize: Callable[[Tensors, torch.Generator], None] | None = None
    namespace: str = ""
    shape: Shape | None = None
    dtype: torch.dtype = torch.float32
    alignment: int = field(default=16, kw_only=True)
    mutation: MutationGranularity = field(default=MutationGranularity.WHOLE_TENSOR, kw_only=True)

    def __post_init__(self) -> None:
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("optimizer state requires a supported floating dtype")
        if self.shape is not None:
            if self.fields:
                raise ValueError("declare a single shape or named fields, not both")
            object.__setattr__(self, "fields", {"value": self.shape})
        if not self.fields:
            raise ValueError("state requires a shape or named fields")


@dataclass(frozen=True)
class RequestState(InstanceState):
    """Per-request parameters, read as tensors aligned to the current operation's rows."""

    keys: tuple[RequestKeyField, ...] = (RequestKeyField.REQUEST_ID,)


class _NativeMath:
    @staticmethod
    def native_methods(functions: Mapping[str, Callable[..., object]]) -> Mapping[str, str]:
        """Freeze code and decorator/source contracts for an explicit native binding."""
        return {name: boundary_digest(_fingerprint(function, structured=True),
            tuple((key, _fingerprint(getattr(function, key, None))) for key in (
                "__pllm_phase__", "__pllm_effect__", "__pllm_gradient__", "__pllm_rows__", "__pllm_hidden__")),
            tuple((key, _fingerprint(get_args(parameter.annotation)[1]))
                for key, parameter in signature(function, eval_str=False).parameters.items()
                if get_origin(parameter.annotation) is Annotated)) for name, function in functions.items()}

    def require_native_methods(self, reference: Mapping[str, str], *,
                               resource_declaration: Callable[..., object] | None = None) -> None:
        """Reject inherited kernel/cost assumptions after ordinary mathematics changes."""
        if resource_declaration is not None:
            declaration = getattr(self, "resource_facts")
            if (declaration.__func__ if isinstance(declaration, MethodType) else declaration) is not resource_declaration:
                return
        current = self.native_methods({name: getattr(self, name) for name in reference})
        for name, fingerprint in reference.items():
            if current[name] != fingerprint:
                contract = "resource_facts declaration" if resource_declaration is not None else "native implementation"
                raise ValueError(f"{name}: customized mathematics require an explicit matching {contract}")


class _LearningMath(ABC, Generic[InputT, ContextT, ResultT]):

    @abstractmethod
    def prepare(self, value: InputT) -> ContextT: ...

    @abstractmethod
    def loss(self, data: ContextT, parameters: Tensors) -> torch.Tensor | None: ...

    @abstractmethod
    def trained_rows(self, data: ContextT) -> int: ...

    @abstractmethod
    def output(self, data: ContextT, result: InvocationOutput) -> ResultT: ...


class CompletionWorkload(_NativeMath, _LearningMath[CompletionInput, ContextT, ResultT],
                         HostWorkload[StateExecution[CompletionInput, ContextT, ResultT]]):
    """Structured completion math with named parameters and one original instance root."""

    name: ClassVar[str] = "completion.learning"

    @property
    @abstractmethod
    def parameters(self) -> InstanceState: ...

    def resource_facts(self, facts: HostRuntimeFacts) -> SchedulerTimeline:
        return facts.timeline()

    def declare_host(self, facts: HostRuntimeFacts) -> TensorHostBinding[CompletionInput, ContextT, ResultT]:
        if facts.hidden_size < 1:
            raise ValueError("completion learning requires a positive hidden size")
        owner = _snapshot(self)
        definition = owner.parameters
        if isinstance(definition, RequestState) or not isinstance(definition.optimizer, AdamWConfig):
            raise ValueError("completion learning requires instance parameters with an optimizer")
        if any(isinstance(shape, Buffer) for shape in definition.fields.values()):
            raise ValueError("completion parameters require tensor shapes, not overlay buffers")
        shapes = cast(Mapping[str, Shape], definition.fields)
        identity = _fingerprint(owner.loss)
        b = Builder[ContextT](self.name, boundary_digest(facts, _fingerprint(_values(owner)),
            _fingerprint(definition), identity, _fingerprint(owner.trained_rows)), Dimensions(facts.hidden_size, 0))
        state = b.state("parameters", tuple(Field(name, shape,
            TensorDType(str(definition.dtype).removeprefix("torch.")), definition.alignment, definition.mutation) for name, shape in shapes.items()),
            namespace=definition.namespace, optimizer=definition.optimizer)
        update = b.compute("train", function=autograd_loss(_freeze(owner.loss), tuple(shapes), _freeze(owner.trained_rows)),
            inputs={"batch": b.input("input", "completions", f"{self.name}.training@1"), "state": state.read()},
            outputs=("control",), changes=state, effect=StateEffectKind.TRANSITION_MAYBE, when=When.HOST_COMPUTE,
            implementations=(native(f"python:{identity}"),),
            math=MathContract(parameter_gradients=(state.declaration.resource,), detached_inputs=("batch",),
                forward_rows="batch", parameter_gradient_rows="batch", update_states="state", reduction="custom"))
        b.publish(update, state, output="control", schema=f"{self.name}.result@1", when=When.HOST_COMMIT,
            implementations=(native("canonical.state:publish@1"),))
        generator = torch.Generator(device=facts.device).manual_seed(facts.initialization_seed)
        initialize_parameters = None if definition.initialize is None else _freeze(definition.initialize)
        prepare_data, output_data = _freeze(owner.prepare), _freeze(owner.output)
        def initialize(roots: Roots) -> None:
            if initialize_parameters is not None:
                root, = roots
                initialize_parameters({name: root[5 * index] for index, name in enumerate(shapes)}, generator)
        def prepare(value: CompletionInput, batch: TensorBatch) -> ContextT:
            if value.hidden.shape[1] != facts.hidden_size or value.hidden.device != facts.device:
                raise ValueError("completion hidden rows differ from specialization")
            if facts.max_input_rows is not None and len(value.hidden) > facts.max_input_rows:
                raise ValueError("completion rows exceed the signed input capacity")
            if value.readiness is not None:
                if batch.stream is None:
                    raise ValueError("CUDA completion input requires a CUDA invocation")
                value.readiness.order_on(batch.stream)
                value.hidden.record_stream(batch.stream)
            return prepare_data(value)
        return TensorHostBinding(b.graph, b.layouts,
            MathBinding(FunctionMath(initialize, lambda value: (), prepare, output_data), b.bind_all()),
            facts, AnchorSymbol("host.compute.end"), owner.resource_facts(facts))


class CausalAdapterWorkload(_LearningMath[CausalTrainBatch[torch.Tensor] | None, ContextT, ResultT],
                            BackendWorkload[ContextT, ResultT]):
    """Instance low-rank overlays with an explicit frozen causal-model dependency."""

    name: ClassVar[str] = "causal.adapter"
    namespace: ClassVar[str] = ""
    model: CausalModel

    @property
    @abstractmethod
    def optimizer(self) -> AdamWConfig: ...

    @property
    @abstractmethod
    def phase(self) -> Phase: ...

    def backend_requirements(self) -> BackendRequirements:
        return BackendRequirements(parameter_targets=tuple(target.target for target in self.model.targets),
            model_source=self.model.model_source, resident_bytes=self.model.resident_bytes)

    def _binding(self, facts: BackendCompileFacts) -> tuple[Builder[ContextT], FunctionMath[CausalTrainBatch[torch.Tensor] | TensorStepInput, ContextT, ResultT]]:
        owner = _snapshot(self, (self.model,))
        model = owner.model
        if (model.resident_bytes < 0 or model.resident_bytes and (
            len(facts.timeline.reserved_bytes) != len(facts.timeline.anchors)
            or any(reserved < model.resident_bytes for reserved in facts.timeline.reserved_bytes))):
            raise ValueError("causal model residency exceeds the signed timeline reservation")
        if (facts.targets != model.targets or facts.probe.tensor_dtype != model.tensor_dtype
            or model.device.type != "meta" and facts.device != model.device):
            raise ValueError("causal model and backend projection facts differ")
        if owner.phase not in (Phase.BEFORE_MODEL, Phase.AFTER_SAMPLE):
            raise ValueError("causal adapter updates require BEFORE_MODEL or AFTER_SAMPLE")
        targets = tuple(item.target for item in model.targets)
        if not targets or targets != tuple(sorted(set(targets))) or len(model.initial_factors) != 2 * len(targets):
            raise ValueError("causal adapters require sorted unique targets and one A/B pair per target")
        for index, target in enumerate(model.targets):
            down, up = model.initial_factors[2 * index:2 * index + 2]
            if (down.ndim != 2 or up.ndim != 2 or down.shape[0] < 1 or down.shape[0] != up.shape[1]
                or down.shape[1] != target.input_size or up.shape[0] != target.output_size
                or down.device != model.device or up.device != model.device
                or down.dtype != tensor_dtype(model.tensor_dtype) or up.dtype != down.dtype):
                raise ValueError("causal adapter factors differ from their target shape, device or dtype")
        identity = _fingerprint(owner.loss)
        values = {key: value for key, value in _values(owner).items() if key != "model"}
        b = Builder[ContextT](owner.name, boundary_digest(_fingerprint(values), model.fingerprint,
            model.scale, owner.optimizer, identity, _fingerprint(owner.trained_rows)))
        names = tuple(f"factor_{index}" for index in range(len(model.initial_factors)))
        state = b.state("factors", tuple(Field(name, tuple(factor.shape), model.tensor_dtype)
            for name, factor in zip(names, model.initial_factors, strict=True)),
            namespace=owner.namespace, optimizer=owner.optimizer)
        update = b.compute("update", function=autograd_loss(_freeze(owner.loss), names, _freeze(owner.trained_rows)),
            inputs={"batch": b.input("training", "batch", CausalTrainBatch.schema), "state": state.read()},
            outputs=("updated",), changes=state, effect=StateEffectKind.TRANSITION_MAYBE, when=owner.phase.value,
            implementations=(native(f"python:{identity}", f"compute/{facts.device.type}@1",
                "optimizer/adamw@1", "transition/full_root@1"),),
            math=MathContract(parameter_gradients=(state.declaration.resource,), detached_inputs=("batch",),
                forward_rows="batch", parameter_gradient_rows="batch", update_states="state", reduction="custom"))
        publication = b.publish(update, state, when=owner.phase.value, schema=f"{owner.name}.result@1")
        b.overlay(state, update, publication, tuple(ParameterOverlayFieldBinding(target.target,
            ParameterOverlayOperator.LOW_RANK_ADD,
            (state.field(f"factor_{2 * index}.parameters"), state.field(f"factor_{2 * index + 1}.parameters")), model.scale)
            for index, target in enumerate(model.targets)))
        def initialize(roots: Roots) -> None:
            root, = roots
            for index, factor in enumerate(model.initial_factors):
                root[5 * index].copy_(factor)
        prepare_data, output_data = _freeze(owner.prepare), _freeze(owner.output)
        limit = next((item for item in facts.timeline.task_resources if item.task == update.symbol), None)
        if limit is not None and (len(limit.input_shapes) != 1 or len(limit.input_shapes[0]) != 2
            or limit.input_shapes[0][0] < 1 or limit.input_shapes[0][1] < 2):
            raise ValueError("causal training costs require signed [sequences, tokens] input bounds")
        def prepare(value: CausalTrainBatch[torch.Tensor] | TensorStepInput, batch: TensorBatch) -> ContextT:
            if isinstance(value, TensorStepInput):
                if value.rows.plan_epoch != batch.epoch:
                    raise ValueError("causal input differs from the signed epoch")
                if value.auxiliary is None:
                    return prepare_data(None)
                if not isinstance(value.auxiliary, CausalTrainBatch):
                    raise ValueError("causal adapter requires causal training tensors")
                value = value.auxiliary
            if (value.input_ids.ndim != 2 or value.input_ids.shape != value.labels.shape
                or value.input_ids.shape[0] < 1 or value.input_ids.shape[1] < 2
                or value.input_ids.dtype != torch.int64 or value.labels.dtype != torch.int64
                or value.input_ids.device != model.device or value.labels.device != model.device
                or value.attention_mask is not None and (value.attention_mask.shape != value.input_ids.shape
                    or value.attention_mask.device != model.device)):
                raise ValueError("causal batch must contain aligned token/label tensors")
            if limit is not None and any(size > bound for size, bound in zip(value.input_ids.shape, limit.input_shapes[0], strict=True)):
                raise ValueError("causal auxiliary input exceeds its signed input bounds")
            return prepare_data(value)
        def rows(value: CausalTrainBatch[torch.Tensor] | TensorStepInput) -> tuple[PackedPhaseRow, ...]:
            if isinstance(value, TensorStepInput) and value.rows.total_rows > facts.probe.max_packed_rows:
                raise ValueError("packed input exceeds the signed backend row capacity")
            return ()
        return b, FunctionMath(initialize, rows, prepare, output_data)

    def declare(self, facts: BackendCompileFacts) -> tuple[SemanticGraph, TensorLayoutRequest, BackendAttachmentSpec]:
        builder, _ = self._binding(facts)
        return builder.finish()

    def bind_math(self, facts: BackendCompileFacts, plan: PlanBundle) -> MathBinding[TensorStepInput, ContextT, ResultT]:
        if self.model.device != facts.device:
            raise ValueError("causal mathematics require the materialized model on the execution device")
        builder, functions = self._binding(facts)
        if builder.graph != plan.semantic_graph or builder.layouts != plan.tensor_layout_request:
            raise ValueError("causal mathematics differ from the original signed declaration")
        return MathBinding(functions, builder.bind_all(plan))


@dataclass(frozen=True)
class Activation:
    state: str
    layers: tuple[int, ...]
    first_tokens: int = 0
    last_tokens: int = 2
    operator: ActivationPatchOperator = ActivationPatchOperator.LOW_RANK_AFFINE


def _buffers(state: InstanceState) -> dict[str, Buffer]:
    if state.optimizer is not None or state.initialize is not None:
        raise ValueError("buffer effects require zero-initialized state without an optimizer")
    if state.alignment != 16 or state.mutation is not MutationGranularity.WHOLE_TENSOR:
        raise ValueError("buffer effect layouts use their original target alignment and mutation granularity")
    result: dict[str, Buffer] = {}
    for key, value in state.fields.items():
        if isinstance(value, Buffer):
            result[key] = value
        elif not callable(value):
            result[key] = Buffer(value, state.dtype)
        else:
            raise TypeError("operand-dependent shapes require Buffer(input_size, output_size)")
    return result


@dataclass(frozen=True)
class _Export:
    name: str | Config
    every: int | Config
    max_rows: int | Config
    max_cols: int | Config
    enabled: bool | Config


@dataclass(frozen=True)
class _Train:
    state: str
    update: str
    reduction: LossReduction
    steps: int | Config
    reset: str


@dataclass(frozen=True)
class _Sampler:
    stage: SamplerStage
    beta: float | Config = 0.0


@dataclass(frozen=True)
class _Gradient:
    inputs: tuple[str, ...]
    parameters: bool
    input_rows: str = ""
    parameter_rows: str = ""


def differentiate(*, inputs: tuple[str, ...] = (), parameters: bool = False,
                  input_rows: str = "", parameter_rows: str = "") -> Callable[[F], F]:
    """Keep the declared derivative edges; parameters may stay frozen while inputs differentiate."""
    def decorate(function: F) -> F:
        setattr(function, "__pllm_gradient__", _Gradient(inputs, parameters, input_rows, parameter_rows))
        return function
    return decorate


@dataclass(frozen=True)
class _Overlay:
    state: str
    targets: tuple[ParameterTarget, ...] | Config
    weight: str
    prefill: bool | Config
    decode: bool | Config
    replay: ConditionalLayerReplay | Config | None


def overlay(*, state: str, targets: tuple[ParameterTarget, ...] | Config, weight: str = "value",
            prefill: bool | Config = True, decode: bool | Config = False,
            replay: ConditionalLayerReplay | Config | None = None) -> Callable[[F], F]:
    def decorate(function: F) -> F:
        setattr(function, "__pllm_effect__", _Overlay(state, targets, weight, prefill, decode, replay))
        return function
    return decorate


def on(phase: Phase, *, rows: str = "") -> Callable[[F], F]:
    def decorate(function: F) -> F:
        setattr(function, "__pllm_phase__", phase)
        setattr(function, "__pllm_rows__", rows)
        return function
    return decorate


def hook(layer: int | Config, *, site: HiddenSite = HiddenSite.BLOCK_OUTPUT) -> Callable[[F], F]:
    """Declare a supported hidden capture; never register an eager Python model hook."""
    def decorate(function: F) -> F:
        setattr(function, "__pllm_hidden__", Hidden(layer, site))
        return function
    return decorate


def export(*, name: str | Config = "value", every: int | Config = 1, max_rows: int | Config = 1,
           max_cols: int | Config = 16, enabled: bool | Config = True) -> Callable[[F], F]:
    def decorate(function: F) -> F:
        setattr(function, "__pllm_effect__", _Export(name, every, max_rows, max_cols, enabled))
        return function
    return decorate


def add_logits(function: F) -> F:
    setattr(function, "__pllm_effect__", _Sampler(SamplerStage.PACKED_PRE_FILTER_ADD))
    return function


def guide_sampling(*, beta: float | Config) -> Callable[[F], F]:
    """Project selected hidden predictions through the original post-filter sampler."""
    def decorate(function: F) -> F:
        setattr(function, "__pllm_effect__", _Sampler(SamplerStage.PACKED_POST_FILTER_EXACT, beta))
        return function
    return decorate


def train(*, state: str, update: str = "", reduction: LossReduction = "mean",
          steps: int | Config = 1, reset: str = "") -> Callable[[F], F]:
    def decorate(function: F) -> F:
        setattr(function, "__pllm_effect__", _Train(state, update, reduction, steps, reset))
        return function
    return decorate


def _setting(value: C | Config, config: object, expected: type[C]) -> C:
    resolved: object = value
    if isinstance(value, Config):
        resolved = config
        for part in value.name.split("."):
            resolved = getattr(resolved, part)
    if type(resolved) is not expected:
        raise TypeError(f"{value}: expected {expected.__name__}")
    return cast(C, resolved)


def _targets(value: object) -> tuple[ParameterTarget, ...]:
    if not isinstance(value, tuple):
        raise TypeError("overlay targets require a tuple of parameter targets")
    items = cast(tuple[object, ...], value)
    if not all(isinstance(item, ParameterTarget) for item in items):
        raise TypeError("overlay targets require typed parameter targets")
    return cast(tuple[ParameterTarget, ...], items)


def _fingerprint(value: object, seen: frozenset[int] = frozenset(), *, structured: bool = False) -> str:
    """Sign cold code/config dependencies, never eval annotations or execute code to trace it."""
    if id(value) in seen:
        raise ValueError("recursive authoring dependencies require an explicit native binding")
    seen = seen | {id(value)}
    if isinstance(value, MethodType):
        return _fingerprint(value.__func__, seen, structured=structured)
    if isinstance(value, FunctionType):
        closure = getclosurevars(value)
        globals_used = _global_names(value.__code__)
        if globals_used & {"getattr", "setattr", "hasattr", "globals", "locals", "vars", "eval", "exec", "__import__"}:
            raise ValueError("reflective mathematics requires an explicit native binding")
        dependencies = {**{key: value.__globals__[key] for key in globals_used if key in value.__globals__}, **closure.nonlocals}
        return boundary_digest(sha256(dumps(_code(value.__code__))).hexdigest(),
            tuple((key, _fingerprint(item, seen, structured=structured)) for key, item in sorted(dependencies.items())),
            _fingerprint(value.__defaults__, seen, structured=structured), _fingerprint(value.__kwdefaults__, seen, structured=structured))
    if isinstance(value, ModuleType):
        if structured and value.__name__ == "re":
            return boundary_digest(value.__name__, sys.version)
        if not value.__name__.startswith(("torch", "math")):
            raise ValueError(f"{value.__name__}: module dependencies require an explicit native binding")
        return boundary_digest(value.__name__, torch.__version__)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Enum):
        return boundary_digest(type(value).__qualname__, _fingerprint(value.value, seen))
    if value is TensorUpdate:
        return TensorUpdate.__qualname__
    if structured and is_dataclass(value) and isinstance(value, type):
        return boundary_digest(value.__module__, value.__qualname__,
            tuple((item.name, str(item.type)) for item in fields(value)),
            _fingerprint(value.__init__, seen, structured=True), _fingerprint(getattr(value, "__post_init__", None), seen, structured=True))
    if isinstance(value, type):
        if value is torch.Tensor:
            return value.__qualname__
        if structured and value in (object, Counter, blake2b):
            return boundary_digest(value.__module__, value.__qualname__, sys.version)
        if structured and issubclass(value, Enum):
            return boundary_digest(value.__module__, value.__qualname__,
                tuple((name, _fingerprint(member.value, seen, structured=True)) for name, member in value.__members__.items()))
        raise ValueError("dynamic type dependencies require an explicit native binding")
    if is_dataclass(value) and not isinstance(value, type):
        return boundary_digest(type(value).__qualname__, tuple((field.name,
            _fingerprint(getattr(value, field.name), seen, structured=structured)) for field in fields(value)))
    if isinstance(value, Mapping):
        items = cast(Mapping[str, object], value)
        return boundary_digest(tuple((key, _fingerprint(item, seen, structured=structured)) for key, item in sorted(items.items())))
    if isinstance(value, (tuple, list)):
        return boundary_digest(tuple(_fingerprint(item, seen, structured=structured) for item in cast(tuple[object, ...], value)))
    return boundary_digest(value)


def _code(code: CodeType) -> CodeType:
    return code.replace(co_filename="", co_firstlineno=0, co_linetable=b"", co_consts=tuple(
        _code(value) if isinstance(value, CodeType) else value for value in code.co_consts))


def _global_names(code: CodeType) -> set[str]:
    names = {str(instruction.argval) for instruction in get_instructions(code) if instruction.opname == "LOAD_GLOBAL"}
    for value in code.co_consts:
        if isinstance(value, CodeType):
            names.update(_global_names(value))
    return names


def _freeze(function: G) -> G:
    if isinstance(function, MethodType):
        return cast(G, MethodType(_freeze(function.__func__), function.__self__))
    if not isinstance(function, FunctionType):
        raise TypeError("ordinary operations require a Python function; use an explicit native binding")
    globals_used = _global_names(function.__code__)
    global_values = dict(function.__globals__)
    for key in globals_used & global_values.keys():
        value = global_values[key]
        global_values[key] = _freeze(value) if isinstance(value, FunctionType) else (
            value if isinstance(value, (ModuleType, type)) else deepcopy(value))
    copied = FunctionType(function.__code__, global_values, function.__name__, deepcopy(function.__defaults__),
        None if function.__closure__ is None else tuple(CellType(_freeze(cell.cell_contents)
            if isinstance(cell.cell_contents, FunctionType) else deepcopy(cell.cell_contents)) for cell in function.__closure__))
    copied.__annotations__ = dict(function.__annotations__)
    copied.__kwdefaults__ = deepcopy(function.__kwdefaults__)
    return cast(G, copied)


def _members(owner: object) -> dict[str, object]:
    members: dict[str, object] = {}
    for cls in reversed(type(owner).__mro__):
        members.update(vars(cls))
    return members


def _values(owner: object) -> dict[str, object]:
    values: dict[str, object] = {name: member for name, member in _members(owner).items() if not name.startswith("__")
        and isinstance(member, (str, int, float, bool, tuple, InstanceState, Activation, Recurrence))}
    values.update(vars(owner))
    if is_dataclass(owner) and not isinstance(owner, type):
        values.update((item.name, getattr(owner, item.name)) for item in fields(owner))
    return values


def _snapshot(owner: C, borrowed: tuple[object, ...] = ()) -> C:
    memo = {id(value): value for value in borrowed}
    copied = deepcopy(owner, memo)
    for key, value in _values(copied).items():
        object.__setattr__(copied, key, _freeze(value) if isinstance(value, FunctionType) else deepcopy(value, memo))
    return copied


def _arguments(function: Callable[..., torch.Tensor | TensorUpdate], hidden: Hidden | None = None,
               result_type: type[torch.Tensor] | type[TensorUpdate] = torch.Tensor) -> tuple[tuple[str, object], ...]:
    result: list[tuple[str, object]] = []
    unmarked = 0
    for name, parameter in signature(function, eval_str=False).parameters.items():
        if parameter.kind not in (SignatureParameter.POSITIONAL_OR_KEYWORD, SignatureParameter.KEYWORD_ONLY):
            raise TypeError(f"{name}: tensor operations require named parameters")
        annotation: object = parameter.annotation
        if annotation is torch.Tensor and hidden is not None:
            unmarked += 1
            if unmarked > 1:
                raise TypeError("multiple Tensor inputs require explicit Annotated sources")
            result.append((name, hidden))
            continue
        if get_origin(annotation) is not Annotated:
            raise TypeError(f"{name}: input source requires Annotated[Tensor, source]; string annotations are not evaluated")
        args: tuple[object, ...] = get_args(annotation)
        if len(args) != 2 or args[0] is not torch.Tensor:
            raise TypeError(f"{name}: expected one source on Annotated[Tensor, source]")
        result.append((name, args[1]))
    if signature(function, eval_str=False).return_annotation is not result_type:
        raise TypeError(f"operation must declare a {result_type.__name__} result")
    return tuple(result)


def _invoke(function: Callable[..., torch.Tensor], names: tuple[str, ...]) -> Callable[[Tensors], torch.Tensor]:
    def invoke(inputs: Tensors) -> torch.Tensor:
        return function(**{name: inputs[name] for name in names})
    return invoke


class Workload(_NativeMath, BackendWorkload[TensorFrame, TensorOutput],
               HostWorkload[StateExecution[TensorStepInput, TensorFrame, TensorOutput]]):
    """Python methods and effects compiled once, not runtime hooks or a second IR."""

    name: ClassVar[str] = "python.workload"

    def _build(self, b: TensorBinding) -> None:
        owner = _snapshot(self)
        members = _members(owner)
        values = _values(owner)
        helpers: list[tuple[str, str]] = []
        for cls in type(owner).__mro__:
            if cls is Workload:
                break
            for name, member in vars(cls).items():
                if name.startswith("__") or name in values or any(name in vars(base) for base in Workload.__mro__):
                    continue
                if isinstance(member, FunctionType):
                    function = cast(Callable[..., torch.Tensor], getattr(owner, name))
                    helpers.append((name, _fingerprint(function)))
                    object.__setattr__(owner, name, _freeze(function))
        bindings = tuple((key, _fingerprint(value)) for key, value in sorted(values.items()))
        states: dict[str, State] = {}
        parameters: dict[tuple[str, str], SemanticInputSource | TensorPort] = {}
        definitions: list[tuple[str, str]] = helpers
        for name, member in members.items():
            if isinstance(member, property) and member.fget is not None and (
                signature(member.fget, eval_str=False).return_annotation in (InstanceState, RequestState, Activation, Recurrence)):
                values[name] = member.__get__(owner)
            elif isinstance(member, InstanceState) and name not in values:
                values[name] = member
        for name, definition in values.items():
            if isinstance(definition, InstanceState):
                definitions.append((name, _fingerprint(definition)))
                if definition.optimizer is None:
                    continue
                shapes: dict[str, Shape] = {}
                for key, shape in definition.fields.items():
                    if isinstance(shape, Buffer):
                        raise TypeError("optimizer parameters require tensor shapes, not overlay buffers")
                    shapes[key] = shape
                state, ports = b.parameters(name, shapes, optimizer=definition.optimizer,
                    namespace=definition.namespace,
                    initialize=None if definition.initialize is None else _freeze(definition.initialize),
                    scope=StateScope.REQUEST if isinstance(definition, RequestState) else StateScope.INSTANCE,
                    dtype=TensorDType(str(definition.dtype).removeprefix("torch.")),
                    alignment=definition.alignment, mutation=definition.mutation,
                    keys=definition.keys if isinstance(definition, RequestState) else ())
                states[name] = state
                parameters.update({(name, field): port for field, port in ports.items()})
        for name, effect in values.items():
            if isinstance(effect, Recurrence):
                definitions.append((name, _fingerprint(effect)))
                b.recurrence(effect)
                continue
            if not isinstance(effect, Activation):
                continue
            definition = values[effect.state]
            if not isinstance(definition, InstanceState):
                raise ValueError("activation replacement requires a declared state")
            definitions.append((name, _fingerprint(effect)))
            b.activation_patch(effect.state, _buffers(definition), layers=effect.layers, operator=effect.operator,
                first_tokens=effect.first_tokens, last_tokens=effect.last_tokens, namespace=definition.namespace,
                scope=StateScope.REQUEST if isinstance(definition, RequestState) else StateScope.INSTANCE,
                keys=definition.keys if isinstance(definition, RequestState) else ())

        def port(source: object) -> SemanticInputSource | TensorPort:
            match source:
                case Hidden(layer=layer, site=site):
                    return b.hidden(_setting(layer, owner, int), site=site)
                case Samples(limit=limit):
                    if limit is not None:
                        b.sample_limit = _setting(limit, owner, int)
                    return b.sampler_rows()
                case Parameter(state=state, field=field, per_row=per_row):
                    parameter = parameters[state, field]
                    return b.parameter_rows(parameter) if per_row else parameter
                case StateIndex(state=state):
                    return b.state_metadata(states[state], version=False)
                case StateVersion(state=state):
                    return b.state_metadata(states[state], version=True)
                case Result(operation=operation):
                    return TaskResultInputSource(TaskSymbol(operation), "value")
                case BaseLogits():
                    return b.feedback()
                case SampledTokens():
                    return b.sampled_tokens()
                case _:
                    raise TypeError(f"unsupported tensor source: {type(source).__name__}")

        prediction: tuple[Call, _Sampler] | None = None
        training: tuple[Call, State] | None = None
        for name, member in sorted(members.items()):
            phase: object = getattr(member, "__pllm_phase__", None)
            if not isinstance(phase, Phase):
                continue
            effect: object = getattr(member, "__pllm_effect__", None)
            gradient: object = getattr(member, "__pllm_gradient__", None)
            forward_rows: object = getattr(member, "__pllm_rows__", "")
            if not isinstance(forward_rows, str):
                raise TypeError("operation row selector must name a method")
            if isinstance(effect, _Overlay):
                if gradient is not None or forward_rows:
                    raise ValueError("ordinary overlays update private buffers without cross-operation autograd")
                if phase is not Phase.MODEL_UPDATE:
                    raise ValueError("parameter overlays require MODEL_UPDATE")
                definition = values[effect.state]
                if not isinstance(definition, RequestState):
                    raise ValueError("dense overlay requires a declared request state")
                buffers = _buffers(definition)
                transition = cast(Callable[..., TensorUpdate], getattr(owner, name))
                arguments = _arguments(transition, result_type=TensorUpdate)
                bindings_overlay: dict[str, str] = {}
                for key, source in arguments:
                    if isinstance(source, Operand):
                        bindings_overlay[key] = source.value
                    elif isinstance(source, Parameter) and source.state == effect.state and source.field in buffers:
                        bindings_overlay[key] = source.field
                    else:
                        raise TypeError("overlay inputs must name captured operands or its own state buffers")
                if set(buffers) & {source.value for source in Operand}:
                    raise ValueError("overlay buffers conflict with reserved operand names")
                resolved_targets = _targets(_setting(effect.targets, owner, tuple))
                replay: object = effect.replay
                if isinstance(replay, Config):
                    path = replay.name
                    replay = owner
                    for part in path.split("."):
                        replay = getattr(replay, part)
                if replay is not None and not isinstance(replay, ConditionalLayerReplay):
                    raise TypeError("overlay replay requires a conditional layer replay")
                identity = _fingerprint(transition)
                transition = _freeze(transition)
                def invoke(inputs: Tensors, function: Callable[..., TensorUpdate] = transition,
                           bindings: Mapping[str, str] = bindings_overlay) -> TensorUpdate:
                    return function(**{key: inputs[source] for key, source in bindings.items()})
                b.dense_overlay(name, effect.state, buffers, targets=resolved_targets,
                    weight=effect.weight, namespace=definition.namespace, keys=definition.keys,
                    phases=tuple(p for p, enabled in ((HiddenPhase.PREFILL, effect.prefill), (HiddenPhase.DECODE, effect.decode))
                        if _setting(enabled, owner, bool)), replay=replay, identity=f"python:{identity}", function=invoke)
                definitions.append((name, boundary_digest(identity, bindings_overlay)))
                continue
            function = cast(Callable[..., torch.Tensor], getattr(owner, name))
            hidden: object = getattr(member, "__pllm_hidden__", None)
            arguments = _arguments(function, hidden if isinstance(hidden, Hidden) else None)
            if isinstance(effect, _Train):
                definition = values[effect.state]
                if isinstance(definition, InstanceState) and isinstance(definition.optimizer, SGD):
                    if (gradient is not None or forward_rows or effect.update or not effect.reset
                        or effect.reduction != "weighted_state_mean_then_mean" or phase is not Phase.AFTER_SAMPLE):
                        raise ValueError("SGD batch training requires reset, weighted reduction and AFTER_SAMPLE without inference-row selectors")
                    batch_sources: dict[str, tuple[str, str]] = {}
                    for key, source in arguments:
                        if isinstance(source, Batch) and source.field in ("hidden", "base_logits", "labels", "weights"):
                            batch_sources[key] = "batch", source.field
                        elif isinstance(source, Parameter) and source.state == effect.state and source.field in definition.fields and not source.per_row:
                            batch_sources[key] = "parameter", source.field
                        else:
                            raise ValueError("batch training inputs must name its batch tensors or own parameters")
                    reset = cast(Callable[[Tensors, torch.Generator], None], getattr(owner, effect.reset))
                    identity = boundary_digest(_fingerprint(function), _fingerprint(reset), batch_sources, effect)
                    definitions.append((name, identity))
                    if training is not None:
                        raise ValueError("tensor sampler attachment requires one training operation")
                    state = states[effect.state]
                    training = b.weighted_train(name, state, identity=f"python:{identity}", optimizer=definition.optimizer,
                        steps=_setting(effect.steps, owner, int), reset=_freeze(reset), function=_invoke(_freeze(function), tuple(batch_sources)),
                        arguments=batch_sources), state
                    continue
                if effect.reset or effect.steps != 1:
                    raise ValueError("multi-step/reset training requires explicit SGD batch parameters")
            inputs = {key: port(source) for key, source in arguments}
            identity = _fingerprint(function)
            function = _freeze(function)
            definitions.append((name, boundary_digest(identity,
                tuple((key, _fingerprint(source)) for key, source in arguments))))
            update: State | None = None
            math: MathContract = MathContract()
            parameter_names = tuple(key for key, source in arguments if isinstance(source, Parameter))
            demand = gradient if isinstance(gradient, _Gradient) else _Gradient((), isinstance(effect, _Train))
            if demand.parameters and not parameter_names:
                raise ValueError("parameter gradients require declared Parameter inputs")
            if demand.input_rows and not demand.inputs or demand.parameter_rows and not demand.parameters:
                raise ValueError("gradient row selectors require their declared derivative edges")
            if (len(set(demand.inputs)) != len(demand.inputs) or any(key not in inputs or key in parameter_names for key in demand.inputs)):
                raise ValueError("input gradients must name distinct nonparameter inputs")
            if any(key in demand.inputs and isinstance(source, (Samples, SampledTokens, StateIndex, StateVersion)) for key, source in arguments):
                raise ValueError("indices, versions and drawn tokens are nondifferentiable inputs")
            if isinstance(effect, _Export) and (demand.inputs or demand.parameters or forward_rows):
                raise ValueError("bounded host exports use their own row limits and are gradient boundaries")
            if isinstance(gradient, _Gradient) or isinstance(effect, _Train):
                math = MathContract(input_gradients=demand.inputs,
                    parameter_gradients=tuple(dict.fromkeys(states[source.state].declaration.resource
                        for _, source in arguments if isinstance(source, Parameter))) if demand.parameters else (),
                    detached_inputs=tuple(key for key, source in arguments if not isinstance(source, Parameter) and key not in demand.inputs))
            if isinstance(effect, _Train):
                update = states[effect.state]
                math = replace(math, update_states=parameter_names[0] if parameter_names else "__update_state", reduction=effect.reduction)
            domains = {"forward_rows": forward_rows, "input_gradient_rows": demand.input_rows,
                "parameter_gradient_rows": demand.parameter_rows, "update_states": effect.update if isinstance(effect, _Train) else ""}
            selectors: dict[str, Call] = {}
            sources = dict(arguments)
            for domain, method in domains.items():
                if not method:
                    continue
                selector = cast(Callable[..., torch.Tensor], getattr(owner, method))
                selected_args = _arguments(selector)
                if any(key not in sources or source != sources[key] for key, source in selected_args):
                    raise ValueError("row selector must use the operation's declared inputs")
                if method not in selectors:
                    identity_selector = _fingerprint(selector)
                    definitions.append((method, identity_selector))
                    selectors[method] = b.tensor(f"__{name}_{method}", when=phase.value,
                        identity=f"python:{identity_selector}:{name}:{method}", inputs={key: inputs[key] for key, _ in selected_args})(
                            _invoke(_freeze(selector), tuple(key for key, _ in selected_args)))
                key = f"__{domain}"
                if key in inputs:
                    raise ValueError(f"{key} is reserved for a row-domain input")
                inputs[key] = TaskResultInputSource(selectors[method].symbol, "value")
                math = replace(math, **{domain: key})
            call = b.tensor(name, inputs=inputs, when=phase.value, identity=f"python:{identity}",
                update=update, math=math)(_invoke(function, tuple(key for key, _ in arguments)))
            if isinstance(effect, _Export):
                b.export(call, name=_setting(effect.name, owner, str), every=_setting(effect.every, owner, int),
                    max_rows=_setting(effect.max_rows, owner, int), max_cols=_setting(effect.max_cols, owner, int),
                    enabled=_setting(effect.enabled, owner, bool))
            elif isinstance(effect, _Train):
                if training is not None:
                    raise ValueError("tensor sampler attachment requires one training operation")
                training = call, states[effect.state]
            elif isinstance(effect, _Sampler):
                if prediction is not None:
                    raise ValueError("tensor sampler attachment requires one logits operation")
                prediction = call, effect
        b.validate_gradients(None if training is None else training[0])
        if training is not None:
            call, state = training
            task = next(task for task in b.graph.tasks if task.symbol == call.symbol)
            if task.math is not None and task.math.reduction == "state_mean":
                training = b.stage_update(call, state), state
        if prediction is not None:
            if training is None:
                raise ValueError("logits attachment requires a training operation")
            call, sampler = prediction
            b.logits(call, *training, stage=sampler.stage, beta=_setting(sampler.beta, owner, float))
        if any(isinstance(value, Recurrence) for value in values.values()) and (len(b.graph.tasks) != 1 or b.graph.states):
            raise ValueError("tokenwise recurrence requires a standalone model effect")
        declared = {state.resource for state in b.graph.states}
        if any(name not in declared for name, value in values.items() if isinstance(value, InstanceState)):
            raise ValueError("state buffers require an overlay effect; unused state is not allocated")
        # Include instance-local function replacements and configuration in the original signature.
        b.graph = replace(b.graph, execution_parameters_digest=boundary_digest(
            b.graph.execution_parameters_digest, definitions, bindings))

    def declaration(self, facts: BackendCompileFacts | None = None) -> TensorBinding:
        dimensions = None if facts is None else Dimensions(facts.probe.hidden_size, facts.vocab_size,
            tuple((target.input_size, target.output_size) for target in facts.targets), facts.probe.tensor_dtype, facts.layer_count)
        b = TensorBinding(self.name, boundary_digest(getattr(self, "config", ()), dimensions,
            None if facts is None else facts.initialization_seed), dimensions)
        b.resources = () if facts is None else facts.timeline.task_resources
        b.probe = None if facts is None else facts.probe
        self._build(b)
        resources = () if facts is None else facts.timeline.task_resources
        b.graph = replace(b.graph, tasks=tuple(replace(task,
            implementations=self.implementations(str(task.symbol), resources,
                explicit_math=facts is not None and "compute/explicit_math@1" in facts.timeline.capabilities)
            or task.implementations) for task in b.graph.tasks))
        if facts is not None and b.overlay_targets and tuple(target.target for target in facts.targets) != b.overlay_targets:
            raise ValueError("overlay targets differ from backend specialization")
        return b

    def backend_requirements(self) -> BackendRequirements:
        b = self.declaration()
        return BackendRequirements(tuple(dict.fromkeys(value.source for task in b.graph.tasks for value in task.inputs
            if isinstance(value.source, HiddenInputSource))), b.overlay_targets,
            activation_layers=() if b.attachment is None else tuple(binding.layer for binding in b.attachment.activation_bindings),
            sampler_stage=b.attachment.sampler_stage if b.attachment is not None and b.attachment.kind is AttachmentKind.SAMPLER_TRAINING else None,
            sampler_feedback=any(isinstance(port.source, SamplerFeedbackInputSource) for task in b.graph.tasks for port in task.inputs))

    def declare(self, facts: BackendCompileFacts) -> tuple[SemanticGraph, TensorLayoutRequest, BackendAttachmentSpec]:
        return self.declaration(facts).finish()

    def _functions(self, b: TensorBinding, device: torch.device, seed: int) -> FunctionMath[TensorStepInput, TensorFrame, TensorOutput]:
        generator = torch.Generator(device=device).manual_seed(seed)
        def initialize(roots: Roots) -> None:
            for root in roots:
                for init in b.initializers:
                    init(root, generator)
        def output(data: TensorFrame, invocation: InvocationOutput) -> TensorOutput:
            return invocation.value
        if b.graph.states:
            def output(data: TensorFrame, invocation: InvocationOutput) -> TensorOutput:
                return b.result(data, invocation)
        def rows(value: TensorStepInput) -> tuple[PackedPhaseRow, ...]:
            if b.probe is not None and value.rows.total_rows > b.probe.max_packed_rows:
                raise ValueError("packed input exceeds the signed backend row capacity")
            if b.attachment is not None and b.attachment.kind is AttachmentKind.MODEL_REPLAY:
                return ()
            if b.attachment is not None and b.attachment.kind is AttachmentKind.ACTIVATION_PATCH:
                return value.rows.rows if b.graph.states[0].scope is StateScope.REQUEST else ()
            return value.rows.request_rows if b.overlay_targets else value.rows.decode_rows
        return FunctionMath(initialize, rows, b.bind_prepare(), output)

    def bind_math(self, facts: BackendCompileFacts, plan: PlanBundle) -> MathBinding[TensorStepInput, TensorFrame, TensorOutput]:
        owner = _snapshot(self)
        b = owner.declaration(facts)
        if b.graph != plan.semantic_graph or b.layouts != plan.tensor_layout_request:
            raise ValueError("tensor mathematics differ from the original signed declaration")
        return owner.operations(facts, plan, b)

    def implementations(self, operation: str, resources: tuple[TaskResourceFact, ...], *,
                        explicit_math: bool = False) -> tuple[TaskImplementationSpec, ...]:
        if explicit_math:
            raise ValueError("this workload has no explicit tensor implementation")
        return ()

    def operations(self, facts: BackendCompileFacts, plan: PlanBundle,
                   b: TensorBinding) -> MathBinding[TensorStepInput, TensorFrame, TensorOutput]:
        if plan.timeline.task_resources:
            self.require_resources(plan, self.resource_facts(facts, ExecutionOptions()))
        return MathBinding(self._functions(b, facts.device, facts.initialization_seed), b.bind_all(plan))

    def declare_host(self, facts: HostRuntimeFacts) -> TensorHostBinding[TensorStepInput, TensorFrame, TensorOutput]:
        b = TensorBinding(self.name, boundary_digest(getattr(self, "config", ()), facts.hidden_size, facts.initialization_seed),
            Dimensions(facts.hidden_size, 0), host=True)
        self._build(b)
        if b.graph.states:
            raise ValueError("ordinary host tensor authoring currently supports stateless observations")
        return TensorHostBinding(b.graph, b.layouts, MathBinding(self._functions(b, facts.device, facts.initialization_seed), b.bind_all()), facts)


__all__ = ["Workload", "Phase", "on", "hook", "export", "add_logits", "guide_sampling", "train", "Config", "Hidden", "Samples",
    "BaseLogits", "SampledTokens", "Parameter", "StateIndex", "StateVersion", "Result", "InstanceState", "RequestState",
    "overlay", "Operand", "Buffer", "TensorUpdate", "differentiate", "Activation", "Recurrence", "transaction", "CompletionWorkload",
    "CausalAdapterWorkload", "SGD", "Batch"]
