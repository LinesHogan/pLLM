"""Cold Python authoring boundary. Reflection stops at the original typed graph."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass, field, replace
from enum import Enum
from functools import partial
from inspect import Parameter as SignatureParameter, signature
from types import FunctionType, MethodType
from typing import Any, ClassVar, Literal, TypeVar

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from pllm.authoring import BackendRequirements, BackendWorkload
from pllm.config import ExecutionOptions
from pllm.contracts.backend_attachment import (
    ActivationPatchOperator,
    AttachmentKind,
    BackendAttachmentSpec,
    DecodeRefresh,
    DecodeWindow,
    MissingPolicy,
)
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.parameter_overlay import (
    ParameterEffect as ParameterEffect,
    ParameterOverlayFieldBinding,
    ParameterOverlayOperator,
    ParameterTarget,
)
from pllm.contracts.request_identity import RequestKeyField, canonical_request_key_fields
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import (
    HiddenInputSource,
    LossReduction,
    MathContract,
    SamplerFeedbackInputSource,
    SchedulerTimeline,
    SemanticGraph,
    SemanticInputSource,
    SemanticOutputKind,
    StateEffectKind,
    StateScope,
    TaskResultInputSource,
    TaskSymbol,
)
from pllm.contracts.state_feed import FeedBinding, FeedOrder
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType, TensorLayoutRequest
from pllm.contracts.visibility import Exact as Exact, Visibility as Visibility
from pllm.execution.binding_base import SGD as SGD, Buffer as Buffer, buffer_field, overlay_fields
from pllm.execution.codec import boundary_digest
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.declaration import Call, Dimensions, Shape, State, When, native
from pllm.execution.freeze import (
    fingerprint,
    freeze,
    members as _members,
    owner_snapshot,
    owner_values,
    reachable_functions,
)
from pllm.execution.host_runtime import (
    HostRuntimeFacts,
    HostWorkload,
    TensorHostBinding,
)
from pllm.execution.module_state import (
    ModuleCall,
    ModuleFactory,
    copy_initial,
    initial_values,
    parameter_count,
    skeleton,
)
from pllm.execution.position_state import PositionState as PositionState
from pllm.execution.state_execution import MathBinding, StateExecution
from pllm.execution.step_input import TensorStepInput
from pllm.execution.task_memory import derive_resources
from pllm.execution.tensor_binding import TensorBinding, TensorOutput as TensorOutput
from pllm.execution.tensor_frame import TensorFrame, TensorPort, Tensors, TensorUpdate as TensorUpdate
from pllm.execution.tensor_program import FunctionMath, InvocationOutput, Roots, TaskOutput
from pllm.execution.tensor_task import TensorExport
from pllm.execution.token_program import Over, Tokens
from pllm.execution.trace import TraceEvent
from pllm.kernels.adamw import AdamWConfig

F = TypeVar("F", bound=Callable[..., torch.Tensor | tuple[torch.Tensor, ...] | TensorUpdate])
Initializer = Callable[[Tensors, torch.Generator], None]
C = TypeVar("C")


class Phase(Enum):
    BEFORE_MODEL = When.BEFORE_MODEL
    OBSERVE = When.OBSERVE
    BEFORE_SAMPLE = When.BEFORE_SAMPLE
    AFTER_SAMPLE = When.AFTER_SAMPLE
    MODEL_UPDATE = When.MODEL_UPDATE
    AUTOMATIC = When.AUTOMATIC


class _Source:
    """An argument marker owns its graph input and derivative eligibility."""

    differentiable: ClassVar[bool] = True

    def port(self, context: _Declaration) -> SemanticInputSource | TensorPort:
        raise TypeError(f"unsupported tensor source: {type(self).__name__}")

    def overlay(self, state: str, buffers: Mapping[str, Buffer]) -> str:
        raise TypeError("overlay inputs must name captured operands or its own state buffers")

    def batch(self, state: str, definition: InstanceState) -> tuple[str, str]:
        raise ValueError("batch training inputs must name its batch tensors or own parameters")

    def adapt(self, context: _Declaration) -> Callable[[torch.Tensor], object]:
        """How the method receives the tensor read from this source."""
        return _same


@dataclass(frozen=True)
class Config:
    name: str

    @staticmethod
    def resolve(value: object, owner: object) -> object:
        """Resolve an explicit configuration path against the snapshotted owner."""
        match value:
            case Config(name=name):
                for part in name.split("."):
                    owner = object.__getattribute__(owner, part)
                return owner
            case _:
                return value


@dataclass(frozen=True)
class Hidden(_Source):
    layer: int | Config = 0
    site: HiddenSite = HiddenSite.BLOCK_OUTPUT
    phase: HiddenPhase = HiddenPhase.DECODE

    def port(self, context: _Declaration) -> SemanticInputSource:
        return context.binding.hidden(_setting(self.layer, context.owner, int), site=self.site, phase=self.phase)


@dataclass(frozen=True)
class Samples(_Source):
    limit: int | Config | None = None

    differentiable: ClassVar[bool] = False

    def port(self, context: _Declaration) -> SemanticInputSource:
        if self.limit is not None:
            context.binding.sample_limit = _setting(self.limit, context.owner, int)
        return context.binding.sampler_rows()


@dataclass(frozen=True)
class Batch(_Source):
    field: str

    def batch(self, state: str, definition: InstanceState) -> tuple[str, str]:
        return "batch", self.field


@dataclass(frozen=True)
class Parameter(_Source):
    state: str
    field: str
    per_row: bool = False

    def port(self, context: _Declaration) -> SemanticInputSource | TensorPort:
        parameter = context.parameters[self.state, self.field]
        return context.binding.parameter_rows(parameter) if self.per_row else parameter

    def overlay(self, state: str, buffers: Mapping[str, Buffer]) -> str:
        return self.field if self.state == state and self.field in buffers else super().overlay(state, buffers)

    def batch(self, state: str, definition: InstanceState) -> tuple[str, str]:
        if self.state != state or self.field not in definition.fields or self.per_row:
            return super().batch(state, definition)
        return "parameter", self.field


def _same(value: torch.Tensor) -> torch.Tensor:
    return value


@dataclass(frozen=True)
class _Module(Parameter):
    """A module state, handed to the method as a callable with the module's interface."""

    field: str = "parameters"

    def adapt(self, context: _Declaration) -> Callable[[torch.Tensor], object]:
        definition = context.state_definition(self.state, f"{self.state}: pllm.module needs a declared module state")
        factory, dimensions = definition.module, context.binding.dimensions
        if factory is None:
            raise TypeError(f"{self.state}: pllm.module needs a state declared with module=")
        if dimensions is None:
            return _same  # declared without model dimensions: never invoked
        return partial(ModuleCall, skeleton(factory, dimensions), per_row=definition.scope is StateScope.REQUEST)


@dataclass(frozen=True)
class Capture:
    """Second Annotated marker: on a training task, write this input's value into
    the state's memory field at publication — the next step reads it as the input
    version. Per lineage, the last selected row wins."""

    state: str
    field: str

    def declare(self, state: str, update: State | None) -> LayoutFieldId:
        """Validate a memory publication against the training state's fields."""
        if self.state != state:
            raise ValueError("memory captures target the task's own training state")
        assert update is not None
        if self.field not in {field.name for field in update.fields[update.optimizer_fields :]}:
            raise ValueError(f"{self.field}: captures name a declared memory field")
        return update.field(self.field)


@dataclass(frozen=True)
class StateIndex(_Source):
    state: str

    differentiable: ClassVar[bool] = False

    def port(self, context: _Declaration) -> TensorPort:
        return context.binding.state_metadata(context.states[self.state], version=False)


@dataclass(frozen=True)
class StateVersion(_Source):
    state: str

    differentiable: ClassVar[bool] = False

    def port(self, context: _Declaration) -> TensorPort:
        return context.binding.state_metadata(context.states[self.state], version=True)


@dataclass(frozen=True)
class Result(_Source):
    operation: str

    def port(self, context: _Declaration) -> SemanticInputSource:
        return TaskResultInputSource(TaskSymbol(self.operation), "value")


@dataclass(frozen=True)
class BaseLogits(_Source):
    def port(self, context: _Declaration) -> SemanticInputSource:
        return context.binding.feedback()


@dataclass(frozen=True)
class SampledTokens(_Source):
    differentiable: ClassVar[bool] = False

    def port(self, context: _Declaration) -> TensorPort:
        return context.binding.sampled_tokens()


class Operand(_Source, Enum):
    INPUT = "input"
    TARGET = "target"
    WEIGHT = "weight"
    PROMPT_END = "prompt_end"

    def overlay(self, state: str, buffers: Mapping[str, Buffer]) -> str:
        return self.value


@dataclass(frozen=True)
class _Captured(_Source):
    """An input whose value a training task also writes into a memory field."""

    source: _Source
    capture: Capture


# Inputs are declared as parameter defaults: `h: torch.Tensor = pllm.hidden(0)`. The factories return the
# source marker typed `Any`, so the annotation can name the value the method body receives.


def hidden(
    layer: int | Config = 0,
    site: HiddenSite = HiddenSite.BLOCK_OUTPUT,
    *,
    phase: Literal["decode", "prefill"] = "decode",
) -> Any:
    """One row per request: latest decode, or the last prompt token of completed prefill."""
    return Hidden(layer, site, HiddenPhase(phase))


def parameter(state: str, field: str, *, per_row: bool = False) -> Any:
    """One tensor field of a declared state, aligned to the operation's rows for request state."""
    return Parameter(state, field, per_row)


def module(state: str) -> Any:
    """A state declared with `module=`, called like the module; request state uses each row's own parameters."""
    return _Module(state)


def version(state: str) -> Any:
    """How many updates each row's state has received (0 before the first update)."""
    return StateVersion(state)


def index(state: str) -> Any:
    """Each row's slot in the state pool, for indexing pooled tensors."""
    return StateIndex(state)


def samples(limit: int | Config | None = None) -> Any:
    """How many samples each request has drawn so far."""
    return Samples(limit)


def result(operation: str) -> Any:
    """The value another operation of this workload returned in the same step."""
    return Result(operation)


def base_logits() -> Any:
    """The model's logits before any workload adjustment."""
    return BaseLogits()


def sampled_tokens() -> Any:
    """The tokens the sampler drew this step."""
    return SampledTokens()


def operand(name: Literal["input", "target", "weight", "prompt_end"]) -> Any:
    """A projection tensor or scalar bool marking the span that completes the prompt."""
    return Operand(name)


def capture(source: _Source, *, state: str, field: str) -> Any:
    """Read `source` and, when the training update publishes, write its value into `state.field`.

    The next step reads that field as its input version; per request the last selected row wins.
    """
    return _Captured(source, Capture(state, field))


def batch(field: Literal["hidden", "base_logits", "labels", "weights"]) -> Any:
    """One tensor of the host-supplied `WeightedCEBatch` an SGD training step consumes."""
    return Batch(field)


@dataclass(frozen=True)
class InstanceState:
    fields: Mapping[str, Shape | Buffer] = field(default_factory=lambda: dict[str, Shape | Buffer]())
    optimizer: AdamWConfig | SGD | None = field(default=None, kw_only=True)
    initialize: Callable[[Tensors, torch.Generator], None] | None = None
    namespace: str = ""
    shape: Shape | None = None
    dtype: torch.dtype = torch.float32
    alignment: int = field(default=16, kw_only=True)
    feed: FeedOrder | None = field(default=None, kw_only=True)
    # Plain per-lineage tensors carried across steps; never optimizer-managed.
    memory: Mapping[str, Shape] = field(default_factory=lambda: dict[str, Shape](), kw_only=True)
    # An `nn.Module` built from the model's dimensions; its parameters become the state, initialized as the
    # module initializes them (every request starts from the same values).
    module: ModuleFactory | None = field(default=None, kw_only=True)
    imported: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        if self.module is not None:
            if self.fields or self.shape is not None or self.initialize is not None or self.feed:
                raise ValueError(
                    "a module state takes its fields and initial values from the module; drop fields/shape/initialize"
                )
            object.__setattr__(self, "fields", {"parameters": parameter_count(self.module)})
        if self.feed is not None and (
            self.scope is not StateScope.INSTANCE
            or self.optimizer is not None
            or self.initialize is not None
            or self.memory
            or self.imported
        ):
            raise ValueError("fed state needs InstanceState without an optimizer, initializer or memory fields")
        if self.shape is not None:
            object.__setattr__(self, "fields", {"value": self.shape})
        if self.memory and self.optimizer is None:
            raise ValueError("memory fields ride on the training transition; give the state an optimizer")
        if set(self.memory) & set(self.fields):
            raise ValueError("rename the memory fields that repeat a parameter field name")

    scope: ClassVar[StateScope] = StateScope.INSTANCE

    @property
    def request_keys(self) -> tuple[RequestKeyField, ...]:
        return ()

    def declare(self, name: str, context: _Declaration) -> None:
        """Allocate tensor parameters; buffer layouts belong to their effects."""
        context.definitions.append((name, boundary_digest(fingerprint(self), tuple(self.fields))))
        if name in context.overlay_states or name in context.states:
            return
        if self.feed is not None:
            if context.binding.feeds:
                raise ValueError("declare one fed state per workload")
            context.binding.feeds = (FeedBinding(name, self.feed, tuple(self.fields)),)
        shapes: dict[str, Shape] = {}
        for key, spec in self.fields.items():
            match spec:
                case Buffer():
                    break
                case shape:
                    shapes[key] = shape
        if len(shapes) != len(self.fields) or any(
            type(value) is Overlay and value.state == name for value in context.values.values()
        ):
            context.allocate_buffers(name, self)
            return
        context.states[name], ports = context.binding.parameters(
            name,
            shapes,
            optimizer=self.optimizer,
            namespace=self.namespace,
            initialize=_frozen(self.initialize),
            scope=self.scope,
            dtype=TensorDType(str(self.dtype).removeprefix("torch.")),
            alignment=self.alignment,
            keys=self.request_keys,
            memory=self.memory,
        )
        context.parameters.update({(name, field): port for field, port in ports.items()})
        if self.module is not None:
            context.binding.initial_values[name] = partial(initial_values, self.module, dtype=self.dtype)


@dataclass(frozen=True)
class RequestState(InstanceState):
    """Per-request parameters, read as tensors aligned to the current operation's rows."""

    keys: tuple[RequestKeyField, ...] = (RequestKeyField.REQUEST_ID,)

    def __post_init__(self) -> None:
        super().__post_init__()
        canonical_request_key_fields(self.keys)

    scope: ClassVar[StateScope] = StateScope.REQUEST

    @property
    def request_keys(self) -> tuple[RequestKeyField, ...]:
        return self.keys


@dataclass(frozen=True)
class Activation:
    state: str
    layers: tuple[int, ...]
    first_tokens: int = 0
    last_tokens: int = 2
    operator: ActivationPatchOperator = ActivationPatchOperator.LOW_RANK_AFFINE
    _: KW_ONLY
    decode: DecodeWindow = DecodeWindow.NONE
    decode_refresh: DecodeRefresh = DecodeRefresh.INHERIT
    on_missing: MissingPolicy = MissingPolicy.STOP

    def __post_init__(self) -> None:
        if self.decode is DecodeWindow.NONE and (
            self.decode_refresh is not DecodeRefresh.INHERIT or self.on_missing is not MissingPolicy.STOP
        ):
            raise ValueError("decode window is off; refresh and missing policies require decode rows")

    def declare(self, context: _Declaration) -> None:
        """Read local state, preserving replay/HOLD publication dependencies when declared."""
        definition = context.state_definition(self.state, "Activation needs a declared state")
        if definition.feed is not None:
            raise ValueError("Activation needs local state written from fed inputs")
        state = context.states[self.state]
        context.binding.activation_patch(
            state,
            context.writes.get(self.state) if context.routes or context.binding.feeds else None,
            layers=self.layers,
            operator=self.operator,
            first_tokens=self.first_tokens,
            last_tokens=self.last_tokens,
            decode=self.decode,
            decode_refresh=self.decode_refresh,
            on_missing=self.on_missing,
        )


@dataclass(frozen=True)
class Overlay(ParameterEffect):
    """Read declared state at internal model projections, independently of its producer."""

    def declare(self, context: _Declaration) -> None:
        """Bind logical parameter views to the model's targets."""
        state = context.states[self.state]
        context.binding.overlay(
            state,
            None,
            tuple(
                ParameterOverlayFieldBinding(
                    target,
                    self.operator,
                    tuple(
                        context.binding.field_views.get(
                            (self.state, f"{key}_{i}"), state.view(state.fields[0].name, context.binding.dimensions)
                        )
                        for key in self.applied_fields
                    ),
                    self.state,
                    self.scale,
                )
                for i, target in enumerate(self.targets)
            ),
        )


def _frozen(initialize: Initializer | None) -> Initializer | None:
    return None if initialize is None else freeze(initialize)


def _buffers(state: InstanceState) -> dict[str, Buffer]:
    if state.optimizer is not None:
        raise ValueError("buffer effects update their state themselves; drop the optimizer")
    buffers: dict[str, Buffer] = {}
    for key, value in state.fields.items():
        match value:
            case Buffer():
                buffers[key] = value
            case tuple():
                buffers[key] = Buffer(value, state.dtype)
            case _:
                raise TypeError("operand-dependent shapes require Buffer(input_size, output_size)")
    return buffers


class _Effect:
    """An operation effect owns its declaration and attachment policy."""

    training: ClassVar[bool] = False
    bounded: ClassVar[bool] = False
    # The phase the effect runs in when no `@pllm.on` names one; None means the author must name it.
    phase: ClassVar[Phase | None] = None

    state: str = ""
    update: str = ""
    reduction: LossReduction = "mean"

    def declare(self, operation: _Operation) -> None:
        operation.tensor()

    def results(self, function: Callable[[Tensors], Any]) -> Callable[[Tensors], tuple[torch.Tensor, ...]]:
        """Ordinary tensor methods declare a single result."""
        return lambda inputs: (function(inputs),)

    @property
    def outputs(self) -> tuple[str, ...]:
        return ("value",)


@dataclass(frozen=True, kw_only=True)
class _Write(_Effect):
    state: str
    fields: tuple[str, ...]

    phase: ClassVar[Phase | None] = Phase.AUTOMATIC

    def declare(self, operation: _Operation) -> None:
        context = operation.context
        definition = context.state_definition(self.state, "write needs a declared state")
        if (
            definition.feed is not None
            or not self.fields
            or len(set(self.fields)) != len(self.fields)
            or any((self.state, key) not in context.parameters for key in self.fields)
            or operation.rows
        ):
            raise ValueError("write needs an unfed state, distinct declared fields and no row mask")
        context.definitions.append((operation.name, fingerprint(self)))
        call = operation.tensor()
        state = context.states[self.state]
        update = context.binding.write_result(call, state, self.fields, operation.phase.value)
        context.writes[self.state] = (
            update,
            context.binding.publish(update, state, when=operation.phase.value, name=f"{self.state}.publish"),
        )

    def results(self, function: Callable[[Tensors], Any]) -> Callable[[Tensors], tuple[torch.Tensor, ...]]:
        """Write methods already return their declared tuple."""
        return function

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.fields


def write(*, state: str, fields: tuple[str, ...]) -> Callable[[F], F]:
    """Atomically publish a tuple of computed fields; request values have a leading request axis."""
    return _mark(effect=_Write(state=state, fields=fields))


@dataclass(frozen=True)
class _Export(_Effect):
    name: str | Config
    every: int | Config
    max_rows: int | Config
    max_cols: int | Config
    enabled: bool | Config
    overflow: Literal["drop", "block"]

    bounded: ClassVar[bool] = True
    phase: ClassVar[Phase | None] = Phase.OBSERVE

    def declare(self, operation: _Operation) -> None:
        """Publish bounded tensor observations to the host."""
        owner = operation.context.owner
        export = TensorExport(
            name=_setting(self.name, owner, str),
            every=_setting(self.every, owner, int),
            max_rows=_setting(self.max_rows, owner, int),
            max_cols=_setting(self.max_cols, owner, int),
            enabled=_setting(self.enabled, owner, bool),
            overflow=self.overflow,
        )
        call = operation.tensor(result_schema="canonical.hidden_export@1", export=export)
        operation.context.binding.model_effect(call)


@dataclass(frozen=True, kw_only=True)
class _Train(_Effect):
    state: str
    update: str
    reduction: LossReduction
    steps: int | Config
    reset: str

    training: ClassVar[bool] = True
    phase: ClassVar[Phase | None] = Phase.AUTOMATIC

    def declare(self, operation: _Operation) -> None:
        """Choose optimizer training or the explicit weighted SGD transition."""
        arguments, captures = operation.arguments()
        definition = operation.context.state_definition(self.state, "training requires a declared optimizer state")
        match definition.optimizer:
            case SGD() as optimizer:
                self.weighted(
                    replace(operation, phase=Phase.AFTER_SAMPLE) if operation.phase is Phase.AUTOMATIC else operation,
                    arguments,
                    captures,
                    definition,
                    optimizer,
                )
            case _:
                if self.reset or self.steps != 1:
                    raise ValueError("multi-step/reset training requires explicit SGD batch parameters")
                context = operation.context
                sampler = context.sampler
                request_training = sampler is not None or not context.has_model_effect
                unconditional = request_training and (
                    sampler is None or sampler.stage is SamplerStage.PACKED_POST_FILTER_EXACT
                )
                if unconditional and (operation.rows or self.update):
                    raise ValueError(
                        "request training updates every selected state: remove rows= and custom update_states from @pllm.train"
                    )
                # Token commit owns request training; selectors keep their declared phase.
                when = (
                    When.AFTER_SAMPLE
                    if request_training and operation.phase is Phase.AUTOMATIC
                    else operation.phase.value
                )
                context.train(
                    operation.tensor(
                        arguments,
                        captures,
                        when=when,
                        update_kind=StateEffectKind.UPDATE if unconditional else StateEffectKind.UPDATE_MAYBE,
                    ),
                    self.state,
                )

    def weighted(
        self,
        operation: _Operation,
        arguments: dict[str, _Source],
        captures: dict[str, Capture],
        definition: InstanceState,
        optimizer: SGD,
    ) -> None:
        """Declare the batch transition, including its reset and input bindings."""
        if captures:
            raise ValueError("SGD batch training manages its own state writes")
        if (
            operation.gradient is not None
            or operation.rows
            or self.update
            or not self.reset
            or self.reduction != "weighted_state_mean_then_mean"
            or operation.phase is not Phase.AFTER_SAMPLE
        ):
            raise ValueError(
                "SGD batch training requires reset, weighted reduction and AFTER_SAMPLE without inference-row selectors"
            )
        context = operation.context
        sources = {key: source.batch(self.state, definition) for key, source in arguments.items()}
        reset = context.method(self.reset)
        identity = boundary_digest(fingerprint(operation.function), fingerprint(reset), sources, self)
        context.definitions.append((operation.name, identity))
        context.training = (
            context.binding.weighted_train(
                operation.name,
                context.states[self.state],
                identity=f"python:{identity}",
                optimizer=optimizer,
                steps=_setting(self.steps, context.owner, int),
                reset=freeze(reset),
                function=_invoke(freeze(operation.function), {key: key for key in sources}),
                arguments=sources,
            ),
            context.states[self.state],
        )


@dataclass(frozen=True)
class _Sampler(_Effect):
    stage: SamplerStage
    beta: float | Config | None = None  # None for additive logits, which have no guidance strength

    phase: ClassVar[Phase | None] = Phase.BEFORE_SAMPLE

    def enabled(self, owner: Workload) -> bool:
        """Zero-strength guidance has no sampler attachment."""
        return self.beta is None or _setting(self.beta, owner, float) != 0

    def declare(self, operation: _Operation) -> None:
        """Zero-strength guidance declares no computation or sampler attachment."""
        if self.enabled(operation.context.owner):
            call = operation.tensor(
                output_kind=SemanticOutputKind.SAMPLER,
                sampler_stage=self.stage,
                sampler_beta=0.0 if self.beta is None else _setting(self.beta, operation.context.owner, float),
            )
            if operation.context.prediction is not None:
                raise ValueError("tensor sampler attachment requires one logits operation")
            operation.context.prediction = call, self


@dataclass(frozen=True)
class _Gradient:
    inputs: tuple[str, ...]
    parameters: bool
    input_rows: str = ""
    parameter_rows: str = ""

    def contract(self, operation: _Operation, arguments: dict[str, _Source]) -> MathContract:
        """Identify parameter resources and detachments."""
        parameters = {
            key: parameter for key, source in arguments.items() if (parameter := _typed(source, Parameter)) is not None
        }
        math = (
            MathContract(
                input_gradients=self.inputs,
                parameter_gradients=tuple(
                    dict.fromkeys(
                        operation.context.states[source.state].declaration.resource
                        for source in parameters.values()
                        if not operation.effect.training or source.state == operation.effect.state
                    )
                )
                if self.parameters
                else (),
                detached_inputs=tuple(key for key in arguments if key not in parameters and key not in self.inputs),
            )
            if _typed(operation.gradient, _Gradient) is not None or operation.effect.training
            else MathContract()
        )

        return (
            replace(math, update_states=next(iter(parameters), "__update_state"), reduction=operation.effect.reduction)
            if operation.effect.training
            else math
        )


def _mark(**markers: object) -> Callable[[F], F]:
    """Attach authoring markers; reflection consumes them once during declaration."""

    def decorate(function: F) -> F:
        for name, marker in markers.items():
            setattr(function, f"__pllm_{name}__", marker)
        return function

    return decorate


def differentiate(
    *, inputs: tuple[str, ...] = (), parameters: bool = False, input_rows: str = "", parameter_rows: str = ""
) -> Callable[[F], F]:
    """Keep derivative edges; parameters may stay frozen while inputs differentiate."""
    return _mark(gradient=_Gradient(inputs, parameters, input_rows, parameter_rows))


@dataclass(frozen=True, kw_only=True)
class _Overlay(_Effect):
    state: str
    targets: tuple[ParameterTarget, ...] | Config
    weight: str
    operator: ParameterOverlayOperator | Config
    factors: tuple[str, str] | Config
    scale: float | Config
    prefill: bool | Config
    decode: bool | Config
    visibility: Visibility | Config

    phase: ClassVar[Phase | None] = Phase.MODEL_UPDATE

    def declare(self, operation: _Operation) -> None:
        """Declare a request buffer transition and its parameter overlay attachment."""
        if operation.gradient is not None or operation.rows:
            raise ValueError("ordinary overlays update private buffers without cross-operation autograd")
        if operation.phase is not Phase.MODEL_UPDATE:
            raise ValueError("parameter overlays require MODEL_UPDATE")
        context = operation.context
        definition = context.state_definition(self.state, "parameter overlays require a declared request state")
        if definition.scope is not StateScope.REQUEST:
            raise ValueError("parameter overlays require a declared request state")
        buffers = _buffers(definition)
        bindings = {key: source.overlay(self.state, buffers) for key, source in _arguments(operation.function).items()}
        if set(buffers) & {source.value for source in Operand}:
            raise ValueError("overlay buffers conflict with reserved operand names")
        targets, operator, scale = (
            _targets(_setting(self.targets, context.owner, tuple)),
            _setting(self.operator, context.owner, ParameterOverlayOperator),
            _setting(self.scale, context.owner, float),
        )
        effect = ParameterEffect(
            self.state, targets, self.weight, operator, _setting(self.factors, context.owner, tuple), scale
        )
        applied = effect.applied_fields
        visibility = _setting(self.visibility, context.owner, Visibility)
        identity = fingerprint(operation.function)
        context.binding.parameter_overlay(
            operation.name,
            effect,
            buffers,
            initialize=_frozen(definition.initialize),
            namespace=definition.namespace,
            keys=definition.request_keys,
            phases=tuple(
                p
                for p, enabled in ((HiddenPhase.PREFILL, self.prefill), (HiddenPhase.DECODE, self.decode))
                if _setting(enabled, context.owner, bool)
            ),
            identity=f"python:{identity}",
            function=_invoke(freeze(operation.function), bindings),
            visibility=visibility,
            equations_digest=boundary_digest(context.definitions, context.bindings),
        )
        context.definitions.append(
            (operation.name, boundary_digest(identity, bindings, operator, scale, applied, self.visibility))
        )


def overlay(
    *,
    state: str,
    targets: tuple[ParameterTarget, ...] | Config,
    weight: str = "value",
    operator: ParameterOverlayOperator | Config = ParameterOverlayOperator.DENSE_ADD,
    factors: tuple[str, str] | Config = ("a", "b"),
    scale: float | Config = 1.0,
    prefill: bool | Config = True,
    decode: bool | Config = False,
    visibility: Visibility | Config = Visibility.forward_pinned(),
) -> Callable[[F], F]:
    """Apply residuals using the declared state visibility at each target projection."""

    return _mark(
        effect=_Overlay(
            state=state,
            targets=targets,
            weight=weight,
            operator=operator,
            factors=factors,
            scale=scale,
            prefill=prefill,
            decode=decode,
            visibility=visibility,
        )
    )


def on(phase: Phase) -> Callable[[F], F]:
    """Run the method in `phase`; needed for methods without an effect or to move an effect's phase."""
    return _mark(phase=phase)


def export(
    *,
    name: str | Config = "value",
    every: int | Config = 1,
    max_rows: int | Config = 1,
    max_cols: int | Config = 16,
    enabled: bool | Config = True,
    overflow: Literal["drop", "block"] = "drop",
) -> Callable[[F], F]:
    """Deliver bounded observations to on_export; block only when explicitly requested."""
    if overflow not in ("drop", "block"):
        raise ValueError("export overflow must be 'drop' or 'block'")
    return _mark(effect=_Export(name, every, max_rows, max_cols, enabled, overflow))


def add_logits(*, rows: str = "") -> Callable[[F], F]:
    """Add the returned tensor to the logits before sampling.

    `rows` (here and on the other effects) names a method returning a boolean mask: only those rows run.
    """
    return _mark(effect=_Sampler(SamplerStage.PACKED_PRE_FILTER_ADD), rows=rows)


def guide_sampling(*, beta: float | Config, rows: str = "") -> Callable[[F], F]:
    """Steer sampling toward the returned hidden-state prediction with strength `beta`."""
    return _mark(effect=_Sampler(SamplerStage.PACKED_POST_FILTER_EXACT, beta), rows=rows)


def train(
    *,
    state: str,
    update: str = "",
    reduction: LossReduction = "mean",
    steps: int | Config = 1,
    reset: str = "",
    rows: str = "",
) -> Callable[[F], F]:
    """Minimize the returned per-row loss with `state`'s optimizer; runs after sampling by default."""
    return _mark(effect=_Train(state=state, update=update, reduction=reduction, steps=steps, reset=reset), rows=rows)


def _setting(value: C | Config, config: object, expected: type[C]) -> C:
    resolved = Config.resolve(value, config)
    if (typed := _typed(resolved, expected)) is None or type(resolved) is not expected:
        raise TypeError(f"{value}: expected {expected.__name__}")
    return typed


def _targets(value: tuple[object, ...]) -> tuple[ParameterTarget, ...]:
    targets = tuple(target for item in value if (target := _typed(item, ParameterTarget)) is not None)
    if len(targets) != len(value):
        raise TypeError("overlay targets require typed parameter targets")
    return targets


def layer_route(*, tokens: Tokens = "all", over: Over = "token") -> Callable[[F], F]:
    """Route disjoint token roles through model slices and tensor math.

    ``over="token"`` runs each position independently in causal order. ``over="span"``
    packs each request's contiguous scheduled tokens into [M, H] hidden tensors;
    position and generated flags have shape [M, 1]. Spans cannot load predecessors.
    """
    if over not in ("token", "span"):
        raise ValueError('set layer_route(over=) to "token" or "span"')
    return _mark(tokens=tokens, over=over)


def _values(owner: object) -> dict[str, object]:
    return owner_values(owner, (InstanceState, Activation, Overlay, PositionState))


def _snapshot(owner: C, borrowed: tuple[object, ...] = ()) -> C:
    return owner_snapshot(owner, (InstanceState, Activation, Overlay, PositionState), borrowed)


def _typed(value: object, expected: type[C]) -> C | None:
    """Narrow an untrusted boundary marker to its declared type."""
    return value if isinstance(value, expected) else None


def _arguments(function: Callable[..., torch.Tensor | tuple[torch.Tensor, ...] | TensorUpdate]) -> dict[str, _Source]:
    """Read each parameter's source from its default value, e.g. `h: torch.Tensor = pllm.hidden(0)`."""
    result: dict[str, _Source] = {}
    for name, parameter in signature(function, eval_str=False).parameters.items():
        source = _typed(parameter.default, _Source)
        if source is None or parameter.kind is SignatureParameter.POSITIONAL_ONLY:
            raise TypeError(
                f"{name}: give every input a source as its default, e.g. `{name}: torch.Tensor = pllm.hidden(0)`"
            )
        result[name] = source
    return result


def _invoke(
    function: Callable[..., C],
    bindings: Mapping[str, str],
    adapters: Mapping[str, Callable[[torch.Tensor], object]] = {},
) -> Callable[[Tensors], C]:
    def invoke(inputs: Tensors) -> C:
        return function(**{key: adapters.get(key, _same)(inputs[source]) for key, source in bindings.items()})

    return invoke


@dataclass
class _Operation:
    """A reflected method, with marker types checked once at the boundary."""

    context: _Declaration
    name: str
    phase: Phase
    effect: _Effect
    gradient: object
    rows: str

    @property
    def function(self) -> FunctionType | MethodType:
        return self.context.method(self.name)

    def arguments(self) -> tuple[dict[str, _Source], dict[str, Capture]]:
        """Split captured inputs into their read source and the memory field the update writes."""
        sources, captures = _arguments(self.function), dict[str, Capture]()
        for key, source in sources.items():
            if (captured := _typed(source, _Captured)) is not None:
                sources[key], captures[key] = captured.source, captured.capture
        return sources, captures

    def tensor(
        self,
        arguments: dict[str, _Source] | None = None,
        captures: dict[str, Capture] | None = None,
        *,
        when: When | None = None,
        update_kind: StateEffectKind = StateEffectKind.UPDATE_MAYBE,
        output_kind: SemanticOutputKind = SemanticOutputKind.TEMPORARY,
        result_schema: str = "",
        export: TensorExport | None = None,
        sampler_stage: SamplerStage | None = None,
        sampler_beta: float = 0.0,
    ) -> Call:
        """Declare tensor mathematics, derivative domains and memory publication."""
        if arguments is None:
            arguments, captures = self.arguments()
        context, effect = self.context, self.effect
        inputs, identity = {key: source.port(context) for key, source in arguments.items()}, fingerprint(self.function)
        context.definitions.append(
            (
                self.name,
                boundary_digest(
                    identity,
                    tuple((key, fingerprint(source)) for key, source in arguments.items()),
                    fingerprint(captures),
                ),
            )
        )
        demand = _typed(self.gradient, _Gradient) or _Gradient((), effect.training)
        math = demand.contract(self, arguments)
        update = context.states[effect.state] if effect.training else None
        writes = {key: capture.declare(effect.state, update) for key, capture in (captures or {}).items()}
        math = self.selectors(
            arguments,
            inputs,
            math,
            {
                "forward_rows": self.rows,
                "input_gradient_rows": demand.input_rows,
                "parameter_gradient_rows": demand.parameter_rows,
                "update_states": effect.update,
            },
        )
        return context.binding.tensor(
            self.name,
            inputs=inputs,
            when=self.phase.value if when is None else when,
            identity=f"python:{identity}",
            update=update,
            math=math,
            captures=writes,
            outputs=effect.outputs,
            effect=update_kind,
            output_kind=output_kind,
            result_schema=result_schema,
            export=export,
            sampler_stage=sampler_stage,
            sampler_beta=sampler_beta,
        )(
            effect.results(
                _invoke(
                    freeze(self.function),
                    {key: key for key in arguments},
                    {key: source.adapt(context) for key, source in arguments.items()},
                )
            )
        )

    def selectors(
        self,
        arguments: dict[str, _Source],
        inputs: dict[str, SemanticInputSource | TensorPort],
        math: MathContract,
        domains: dict[str, str],
    ) -> MathContract:
        """Declare each selector once and connect its row-domain result; selectors read their own inputs."""
        selectors: dict[str, Call] = {}
        for domain, method in domains.items():
            if not method:
                continue
            selector = self.context.method(method)
            selected_args = _arguments(selector)
            if method not in selectors:
                identity = fingerprint(selector)
                self.context.definitions.append((method, identity))
                selectors[method] = self.context.binding.tensor(
                    f"__{self.name}_{method}",
                    when=self.phase.value,
                    identity=f"python:{identity}:{self.name}:{method}",
                    inputs={
                        key: inputs[key] if arguments.get(key) == source else source.port(self.context)
                        for key, source in selected_args.items()
                    },
                )(_Effect().results(_invoke(freeze(selector), {key: key for key in selected_args})))
            key = f"__{domain}"
            inputs[key] = TaskResultInputSource(selectors[method].symbol, "value")
            math = replace(math, **{domain: key})
        return math


def _property(member: object) -> Callable[..., object] | None:
    """Find declaration getters without evaluating arbitrary owner properties."""
    match member:
        case property(fget=fget) if fget is not None and signature(fget, eval_str=False).return_annotation in (
            InstanceState,
            RequestState,
            Activation,
            Overlay,
            PositionState,
        ):
            return fget
        case _:
            return None


def _candidates(owner: Workload, values: dict[str, object], members: dict[str, object]) -> dict[str, FunctionType]:
    """Find helper methods in MRO order, excluding framework and value bindings."""
    classes = type(owner).__mro__[: type(owner).__mro__.index(Workload)]
    names = dict.fromkeys(name for cls in classes for name in vars(cls))
    return {
        name: member
        for name in names
        if not name.startswith("__")
        and name not in values
        and not any(name in vars(base) for base in Workload.__mro__)
        and (member := _typed(members[name], FunctionType)) is not None
    }


@dataclass
class _Declaration:
    """Own the binding, reflected owner and ordered definitions during one build."""

    owner: Workload
    binding: TensorBinding
    values: dict[str, object]
    bindings: tuple[tuple[str, str], ...]
    definitions: list[tuple[str, str]] = field(default_factory=lambda: [])
    operations: list[_Operation] = field(default_factory=lambda: [])
    routes: list[tuple[str, Tokens, Over]] = field(default_factory=lambda: [])
    positions: tuple[tuple[str, float], ...] = ()
    overlay_states: set[str] = field(default_factory=lambda: set[str]())
    states: dict[str, State] = field(default_factory=lambda: {})
    parameters: dict[tuple[str, str], SemanticInputSource | TensorPort] = field(default_factory=lambda: {})
    writes: dict[str, tuple[Call, Call]] = field(default_factory=lambda: {})
    prediction: tuple[Call, _Sampler] | None = None
    training: tuple[Call, State] | None = None

    @property
    def sampler(self) -> _Sampler | None:
        """Find active guidance before either prediction or training is emitted."""
        return next(
            (
                sampler
                for operation in self.operations
                if (sampler := _typed(operation.effect, _Sampler)) is not None and sampler.enabled(self.owner)
            ),
            None,
        )

    @property
    def has_model_effect(self) -> bool:
        """Model effects own training unless a sampler pairs with it."""
        return (
            bool(self.routes)
            or any(isinstance(value, (Activation, Overlay)) for value in self.values.values())
            or any(isinstance(operation.effect, (_Export, _Overlay)) for operation in self.operations)
        )

    @classmethod
    def read(cls, workload: Workload, binding: TensorBinding) -> _Declaration:
        """Snapshot and sign reachable mathematics before evaluating state properties."""
        owner = _snapshot(workload)
        values, members = _values(owner), _members(owner)
        context, candidates = cls(owner, binding, values, ()), _candidates(owner, values, members)
        seeds: list[FunctionType] = []
        for name, member in members.items():
            context.read_operation(name, member, candidates, seeds)
            if (getter := _typed(_property(member), FunctionType)) is not None:
                seeds.append(getter)
        for value in values.values():
            if (state := _typed(value, InstanceState)) is not None and state.initialize is not None:
                method = _typed(state.initialize, MethodType)
                if (
                    function := _typed(state.initialize if method is None else method.__func__, FunctionType)
                ) is not None:
                    seeds.append(function)
        context.freeze_helpers(candidates, seeds)
        context.bindings = tuple((key, fingerprint(value)) for key, value in sorted(values.items()))
        for name, member in members.items():
            if _property(member):
                values[name] = object.__getattribute__(owner, name)
        return context

    def read_operation(
        self, name: str, member: object, candidates: dict[str, FunctionType], seeds: list[FunctionType]
    ) -> None:
        """Read decorator attributes once; marker values never drive runtime reflection."""
        if (function := _typed(member, FunctionType)) is None:
            return
        markers = vars(function)
        if (tokens := markers.get("__pllm_tokens__")) is not None:
            if tuple(signature(function).parameters) != ("self", "layers") or any(
                markers.get(f"__pllm_{marker}__") is not None for marker in ("phase", "effect", "gradient", "rows")
            ):
                raise ValueError(
                    "layer_route requires (self, layers) and no stacked markers; use layers.save inside the route instead of @write"
                )
            overlap = {
                role
                for _, selected, _ in self.routes
                for role in ("prefill", "decode")
                if selected in (role, "all") and tokens in (role, "all")
            }
            if overlap:
                raise ValueError(f"layer_route tokens overlap: {', '.join(sorted(overlap))}")
            self.routes.append((name, tokens, markers["__pllm_over__"]))
            seeds.append(function)
            return
        phase, marked, gradient, rows = (
            _typed(markers.get("__pllm_phase__"), Phase),
            _typed(markers.get("__pllm_effect__"), _Effect),
            markers.get("__pllm_gradient__"),
            _typed(markers.get("__pllm_rows__", ""), str),
        )
        if phase is None and marked is None:
            return
        effect = marked or _Effect()
        texts = (*vars(effect).values(), *vars(_typed(gradient, _Gradient) or _Gradient((), False)).values(), rows)
        seeds.extend(candidates[text] for text in texts if type(text) is str and text in candidates)
        if (overlay := _typed(effect, _Overlay)) is not None:
            self.overlay_states.add(overlay.state)
        phase = phase or effect.phase or Phase.AFTER_SAMPLE
        seeds.append(function)
        assert rows is not None
        self.operations.append(_Operation(self, name, phase, effect, gradient, rows))

    def method(self, name: str) -> FunctionType | MethodType:
        """Bind a named authoring method, including an instance-local replacement."""
        return object.__getattribute__(self.owner, name)

    def freeze_helpers(self, candidates: dict[str, FunctionType], seeds: list[FunctionType]) -> None:
        """Freeze only helpers reachable from operations, state properties or markers."""
        reachable = reachable_functions(candidates, tuple(seeds)) | {
            name for name, member in candidates.items() if any(member is seed for seed in seeds)
        }
        for name in candidates:
            if name in reachable:
                function = self.method(name)
                self.definitions.append((name, fingerprint(function)))
                object.__setattr__(self.owner, name, freeze(function))

    def state_definition(self, name: str, error: str) -> InstanceState:
        definition = _typed(self.values.get(name), InstanceState)
        if definition is None:
            raise ValueError(error)
        return definition

    def allocate_buffers(self, name: str, definition: InstanceState) -> None:
        """Register buffer fields as ordinary readable state ports."""
        buffers = _buffers(replace(definition, optimizer=None))
        consumers = tuple(
            consumer
            for value in self.values.values()
            if (consumer := _typed(value, Overlay)) is not None and consumer.state == name
        )
        if consumers:
            (consumer,) = consumers
            fields = overlay_fields(buffers, consumer)
            self.binding.overlay_targets = consumer.targets
        else:
            fields = tuple(
                buffer_field(key, buffer, lambda m: (m.hidden_size, m.hidden_size)) for key, buffer in buffers.items()
            )
        shapes = {field.name: field.shape for field in fields}
        if definition.optimizer is not None:
            dtype = fields[0].dtype
            state, ports = self.binding.parameters(
                name,
                shapes,
                optimizer=definition.optimizer,
                namespace=definition.namespace,
                initialize=_frozen(definition.initialize),
                scope=definition.scope,
                dtype=dtype(self.binding.dimensions)
                if callable(dtype) and self.binding.dimensions is not None
                else TensorDType(str(definition.dtype).removeprefix("torch."))
                if callable(dtype)
                else dtype,
                keys=definition.request_keys,
            )
        else:
            state = self.binding.state(
                name, fields, namespace=definition.namespace, scope=definition.scope, keys=definition.request_keys
            )
            ports = self.binding.parameter_ports(state, shapes, initialize=_frozen(definition.initialize))
        self.states[name] = state
        self.parameters.update({(name, field): port for field, port in ports.items()})

    def declare_values(self) -> None:
        """Allocate ordinary state before effects and operations consume its ports."""
        self.positions = self.position_states()
        for name, value in self.values.items():
            if (state := _typed(value, InstanceState)) is not None:
                state.declare(name, self)
                if state.imported:
                    allocated = self.states[name]
                    self.binding.imports[name] = tuple(
                        view for (resource, _), view in self.binding.field_views.items() if resource == name
                    )
                    update = self.binding.import_update(allocated, f"{name}:import@1")
                    self.writes[name] = (
                        update,
                        self.binding.publish(update, allocated, when=When.BEFORE_MODEL, name=f"{name}.publish"),
                    )

        self.definitions.extend(
            (name, fingerprint(value)) for name, value in self.values.items() if type(value) in (Activation, Overlay)
        )

    def attach_effects(self) -> None:
        """Connect publications before replay and missing-input branches."""
        for value in self.values.values():
            if (effect := _typed(value, Activation) or _typed(value, Overlay)) is not None:
                effect.declare(self)
        if self.routes:
            self.binding.layer_route(
                tuple((tokens, over, self.method(name)) for name, tokens, over in self.routes), self.positions
            )
        if self.binding.feeds:
            self.binding.conditional_input(tuple(feed.resource for feed in self.binding.feeds))

    def position_states(self) -> tuple[tuple[str, float], ...]:
        """Resolve the pre-allocation schema without tracing tensors."""
        states = tuple(
            (name, pos.initial)
            for name, value in sorted(self.values.items())
            if (pos := _typed(value, PositionState)) is not None
        )
        if states and not self.routes:
            raise ValueError("PositionState requires layers.load/save inside a layer_route")
        return states

    def train(self, call: Call, state: str) -> None:
        self.training = call, self.states[state]

    def attach_training(self) -> None:
        """Validate gradients and pair the single prediction and training operations."""
        self.binding.validate_gradients(None if self.training is None else self.training[0])
        if self.prediction is not None:
            if self.training is None:
                raise ValueError("logits attachment requires a training operation")
            call, sampler = self.prediction
            self.binding.logits(
                call,
                *self.training,
                stage=sampler.stage,
            )
        elif self.training is not None and self.binding.attachment is None:
            self.binding.request_training(*self.training)

    def sign_composition(self) -> None:
        """Validate composed effects and sign ordered definitions and owner values."""
        b = self.binding
        if len(b.imports) > 1:
            raise ValueError("StateTensorUpdate supplies one imported resource per workload")
        if not b.graph.tasks and b.attachment is not None:
            state = self.states[b.attachment.state_resource]
            b.compute(
                "model.read",
                inputs={"state": state.read()},
                reads=(state.version(),),
                when=When.MODEL_UPDATE,
                function=lambda data, batch: TaskOutput(),
            )

        frozen = {
            name
            for name, value in self.values.items()
            if (state := _typed(value, InstanceState)) is not None
            and state.module is not None
            and state.optimizer is None
            and not state.imported
            and state.feed is None
            and not any(
                effect.resource == name and effect.kind is not StateEffectKind.READ
                for task in b.graph.tasks
                for effect in task.effects
            )
        }
        b.graph = replace(
            b.graph,
            states=tuple(replace(state, immutable=state.resource in frozen) for state in b.graph.states),
            execution_parameters_digest=boundary_digest(
                b.graph.execution_parameters_digest, self.definitions, self.bindings
            ),
        )


class Workload(
    BackendWorkload[TensorFrame, TensorOutput],
    HostWorkload[StateExecution[TensorStepInput, TensorFrame, TensorOutput]],
):
    """Python methods and effects compiled once, not runtime hooks or a second IR."""

    name: ClassVar[str] = "python.workload"

    def on_export(self, event: TraceEvent) -> None:
        """Consume ordered CPU snapshots in the execution process; never reenter execution.

        The default retains snapshot history only. Own any retained tensors and I/O resources.
        """

    def _build(self, b: TensorBinding) -> None:
        """Declare owner values and operations, then compose and sign their graph."""
        context = _Declaration.read(self, b)
        context.declare_values()
        for operation in sorted(context.operations, key=lambda operation: operation.name):
            operation.effect.declare(operation)
        context.attach_effects()
        context.attach_training()
        context.sign_composition()

    def declaration(self, facts: BackendCompileFacts | None = None) -> TensorBinding:
        dimensions = (
            None
            if facts is None
            else Dimensions(
                facts.probe.hidden_size,
                facts.vocab_size,
                tuple((target.input_size, target.output_size) for target in facts.probe.targets),
                facts.probe.tensor_dtype,
                facts.layer_count,
            )
        )
        b = TensorBinding(
            self.name,
            boundary_digest(
                _values(self).get("config", ()), dimensions, None if facts is None else facts.initialization_seed
            ),
            dimensions,
        )
        b.resources = () if facts is None else facts.timeline.task_resources
        b.probe = None if facts is None else facts.probe
        b.max_requests = 1 if facts is None else facts.max_admitted_lineages
        self._build(b)
        if facts is not None:
            b.configure_scans(facts)
        b.graph = replace(
            b.graph,
            tasks=tuple(
                replace(
                    task,
                    implementations=(
                        (native(f"python.cuda_graph:{self.name}:{task.symbol}@1", "compute/cuda_graph@1"),)
                        if any(fact.task == str(task.symbol) and fact.resident_bytes for fact in b.resources)
                        else task.implementations
                    ),
                )
                if task.symbol in b.tensor_tasks
                else task
                for task in b.graph.tasks
            ),
        )
        return b

    def backend_requirements(self) -> BackendRequirements:
        b = self.declaration()
        return BackendRequirements(
            tuple(
                dict.fromkeys(
                    source
                    for task in b.graph.tasks
                    for value in task.inputs
                    if (source := _typed(value.source, HiddenInputSource)) is not None
                )
            ),
            b.overlay_targets,
            activation_layers=()
            if b.attachment is None
            else tuple(binding.layer for binding in b.attachment.activation_bindings),
            sampler_stage=b.attachment.sampler_stage
            if b.attachment is not None and b.attachment.guides_sampler
            else None,
            sampler_feedback=any(
                _typed(port.source, SamplerFeedbackInputSource) is not None
                for task in b.graph.tasks
                for port in task.inputs
            ),
            position_states=b.position_states,
        )

    def declare(self, facts: BackendCompileFacts) -> tuple[SemanticGraph, TensorLayoutRequest, BackendAttachmentSpec]:
        return self.declaration(facts).finish()

    def _functions(
        self, b: TensorBinding, device: torch.device, seed: int
    ) -> FunctionMath[TensorStepInput, TensorFrame, TensorOutput]:
        generator = torch.Generator(device=device).manual_seed(seed)

        def initializer(resource: str) -> Callable[[Roots, tuple[RequestStateKey | None, ...]], None]:
            init = b.initializers.get(resource)
            if resource in b.initial_values:
                assert b.dimensions is not None
                init = partial(copy_initial, b.initial_values[resource](b.dimensions, seed).to(device))

            def initialize(roots: Roots, keys: tuple[RequestStateKey | None, ...]) -> None:
                if init is not None:
                    for root in roots:
                        init(root, generator)

            return initialize

        def output(data: TensorFrame, invocation: InvocationOutput) -> TensorOutput:
            return b.result(data, invocation) if b.graph.states else invocation.value

        def rows(value: TensorStepInput) -> tuple[PackedPhaseRow, ...]:
            if b.attachment is not None and (
                b.attachment.activation_bindings
                or any(task.replay is not None for task in b.graph.tasks)
                and b.overlay_targets
            ):
                return value.rows.rows if b.graph.states[0].scope is StateScope.REQUEST else ()
            if b.attachment is not None and b.attachment.kind is AttachmentKind.MODEL_REPLAY:
                return ()
            return b.read_rows(value.rows)

        return FunctionMath(
            {str(state.resource): initializer(str(state.resource)) for state in b.graph.states},
            rows,
            b.bind_prepare(),
            output,
            lambda previous, incoming: replace(previous, step=incoming.step, rows=incoming.rows, index=None, plan=None),
            parameter_phases=b.parameter_phases,
        )

    def bind_math(
        self, facts: BackendCompileFacts, plan: PlanBundle
    ) -> MathBinding[TensorStepInput, TensorFrame, TensorOutput]:
        owner = _snapshot(self)
        b = owner.declaration(facts)
        if b.graph != plan.semantic_graph or b.layouts != plan.tensor_layout_request:
            raise ValueError("tensor mathematics differ from the original signed declaration")
        return owner.operations(facts, plan, b)

    def specialize(self, facts: BackendCompileFacts, options: ExecutionOptions) -> BackendCompileFacts:
        """Fund shared page storage for declared feeds before compiling the plan."""
        if any(state.feed is not None for value in _values(self).values() if (state := _typed(value, InstanceState))):
            facts = replace(
                facts,
                providers=tuple(replace(provider, page_capacity=3, shared_memory=True) for provider in facts.providers),
            )
        return super().specialize(facts, options)

    def resource_facts(self, facts: BackendCompileFacts, options: ExecutionOptions) -> SchedulerTimeline:
        """Reserve ordinary tensor scratch and each requested graph's private pool."""
        if options.explicit_math:
            raise ValueError("this workload has no explicit tensor implementation")
        with FakeTensorMode():
            binding = self.declaration(facts)
        return derive_resources(binding, facts, options)

    def operations(
        self, facts: BackendCompileFacts, plan: PlanBundle, b: TensorBinding
    ) -> MathBinding[TensorStepInput, TensorFrame, TensorOutput]:
        if plan.timeline.task_resources:
            b.configure_steps(facts)
            if any(
                task.lane == "side"
                and plan.semantic_graph.tasks[task.task_id].symbol in b.tensor_tasks
                and not b.operations[plan.semantic_graph.tasks[task.task_id].symbol].resident(1)
                for task in plan.schedule_physical.tasks
            ):
                raise ValueError(
                    "side tensor tasks need fixed row inputs; declare row inputs or move the task before the model"
                )
        return MathBinding(
            self._functions(b, facts.device, facts.initialization_seed),
            b.bind_all(plan, model_probe=facts.probe),
            borrowed_resources=tuple(feed.resource for feed in b.feeds),
            feeds=b.feeds,
            on_export=self.on_export if b.has_exports else None,
        )

    def declare_host(self, facts: HostRuntimeFacts) -> TensorHostBinding[TensorStepInput, TensorFrame, TensorOutput]:
        b = TensorBinding(
            self.name,
            boundary_digest(_values(self).get("config", ()), facts.hidden_size, facts.initialization_seed),
            Dimensions(facts.hidden_size, 0),
            host=True,
        )
        self._build(b)
        if b.graph.states:
            raise ValueError("ordinary host tensor authoring currently supports stateless observations")
        return TensorHostBinding(
            b.graph,
            b.layouts,
            MathBinding(
                self._functions(b, facts.device, facts.initialization_seed),
                b.bind_all(),
                on_export=self.on_export if b.has_exports else None,
            ),
            facts,
        )


__all__ = [
    "Workload",
    "layer_route",
    "Phase",
    "on",
    "write",
    "export",
    "add_logits",
    "guide_sampling",
    "train",
    "Config",
    "hidden",
    "parameter",
    "module",
    "version",
    "index",
    "samples",
    "result",
    "base_logits",
    "sampled_tokens",
    "operand",
    "capture",
    "batch",
    "InstanceState",
    "RequestState",
    "overlay",
    "Buffer",
    "TensorUpdate",
    "differentiate",
    "Activation",
    "Overlay",
    "ParameterEffect",
    "Exact",
    "Visibility",
    "SGD",
]
