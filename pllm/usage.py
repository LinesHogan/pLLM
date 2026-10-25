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

from pllm.authoring import BackendCompileFacts, BackendRequirements, BackendWorkload, TensorStepInput
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
    ParameterOverlayOperator,
    ParameterTarget,
)
from pllm.contracts.request_identity import RequestKeyField, canonical_request_key_fields
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import (
    ConditionalLayerReplay,
    HiddenInputSource,
    LossReduction,
    MathContract,
    SamplerFeedbackInputSource,
    SchedulerTimeline,
    SemanticGraph,
    SemanticInputSource,
    StateScope,
    TaskImplementationSpec,
    TaskResourceFact,
    TaskResultInputSource,
    TaskSymbol,
)
from pllm.contracts.state_feed import FeedBinding, FeedOrder
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType, TensorLayoutRequest
from pllm.contracts.visibility import Exact as Exact, Visibility as Visibility
from pllm.execution.binding_base import SGD as SGD, Buffer as Buffer
from pllm.execution.codec import boundary_digest
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
from pllm.execution.state_execution import MathBinding, StateExecution
from pllm.execution.task_memory import derive_resources
from pllm.execution.tensor_binding import TensorBinding, TensorOutput as TensorOutput
from pllm.execution.tensor_frame import TensorFrame, TensorPort, Tensors, TensorUpdate as TensorUpdate
from pllm.execution.tensor_program import FunctionMath, InvocationOutput, Roots, TensorComputation
from pllm.execution.token_program import Tokens
from pllm.kernels.adamw import AdamWConfig

F = TypeVar("F", bound=Callable[..., torch.Tensor | TensorUpdate])
Initializer = Callable[[Tensors, torch.Generator], None]
C = TypeVar("C")


class Phase(Enum):
    BEFORE_MODEL = When.BEFORE_MODEL
    OBSERVE = When.OBSERVE
    BEFORE_SAMPLE = When.BEFORE_SAMPLE
    AFTER_SAMPLE = When.AFTER_SAMPLE
    MODEL_UPDATE = When.MODEL_UPDATE


def _reject(condition: object, message: str, error: type[Exception] = ValueError) -> None:
    """Reject an invalid authoring declaration with its specific error type."""
    if condition:
        raise error(message)


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
        if isinstance(value, Config):
            for part in value.name.split("."):
                owner = object.__getattribute__(owner, part)
            return owner
        return value


@dataclass(frozen=True)
class Hidden(_Source):
    layer: int | Config = 0
    site: HiddenSite = HiddenSite.BLOCK_OUTPUT

    def port(self, context: _Declaration) -> SemanticInputSource:
        return context.binding.hidden(_setting(self.layer, context.owner, int), site=self.site)


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
        if self.field not in ("hidden", "base_logits", "labels", "weights"):
            return super().batch(state, definition)
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
        _reject(factory is None, f"{self.state}: pllm.module needs a state declared with module=", TypeError)
        if factory is None or dimensions is None:
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
        if update is None or self.state != state:
            raise ValueError("memory captures target the task's own training state")
        _reject(
            self.field not in {field.name for field in update.fields[update.optimizer_fields :]},
            f"{self.field}: captures name a declared memory field",
        )
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

    def overlay(self, state: str, buffers: Mapping[str, Buffer]) -> str:
        return self.value


@dataclass(frozen=True)
class _Captured(_Source):
    """An input whose value a training task also writes into a memory field."""

    source: _Source
    capture: Capture


# Inputs are declared as parameter defaults: `h: torch.Tensor = pllm.hidden(0)`. The factories return the
# source marker typed `Any`, so the annotation can name the value the method body receives.


def hidden(layer: int | Config = 0, site: HiddenSite = HiddenSite.BLOCK_OUTPUT) -> Any:
    """The hidden state at `layer` (negative counts from the last layer), one row per request."""
    return Hidden(layer, site)


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


def operand(name: Literal["input", "target", "weight"]) -> Any:
    """A projection operand inside an overlay: its input, target or base weight."""
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

    def __post_init__(self) -> None:
        if self.module is not None:
            _reject(
                self.fields or self.shape is not None or self.initialize is not None or self.memory or self.feed,
                "a module state takes its fields and initial values from the module; drop fields/shape/initialize",
            )
            object.__setattr__(self, "fields", {"parameters": parameter_count(self.module)})
        _reject(
            self.feed is not None
            and (
                self.scope is not StateScope.INSTANCE
                or self.optimizer is not None
                or self.initialize is not None
                or self.memory
            ),
            "fed state needs InstanceState without an optimizer, initializer or memory fields",
        )
        _reject(
            self.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64),
            "optimizer state requires a supported floating dtype",
        )
        if self.shape is not None:
            _reject(self.fields, "declare a single shape or named fields, not both")
            object.__setattr__(self, "fields", {"value": self.shape})
        _reject(not self.fields, "state requires a shape or named fields")
        _reject(
            self.memory and self.optimizer is None,
            "memory fields ride on the training transition and require an optimizer",
        )
        _reject(set(self.memory) & set(self.fields), "memory fields duplicate a parameter field")
        _reject(
            any(isinstance(shape, Buffer) for shape in self.memory.values()),
            "memory fields require tensor shapes, not overlay buffers",
            TypeError,
        )

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
            _reject(context.binding.feeds, "declare one fed state per workload")
            context.binding.feeds = (FeedBinding(name, self.feed, tuple(self.fields)),)
        shapes = {key: shape for key, shape in self.fields.items() if not isinstance(shape, Buffer)}
        if len(shapes) != len(self.fields):
            _reject(
                self.optimizer is not None, "optimizer parameters require tensor shapes, not overlay buffers", TypeError
            )
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
        _reject(
            self.decode is DecodeWindow.NONE
            and (self.decode_refresh is not DecodeRefresh.INHERIT or self.on_missing is not MissingPolicy.STOP),
            "decode window is off; refresh and missing policies require decode rows",
        )

    def allocate(self, context: _Declaration) -> None:
        """Make activation fields readable before tensor operations are declared."""
        definition = context.state_definition(self.state, "Activation needs a declared state")
        _reject(definition.feed is not None, "Activation needs local state written from fed inputs")
        context.allocate_buffers(self.state, definition)

    def declare(self, context: _Declaration) -> None:
        """Attach either a computed or an imported update to the activation."""
        state = context.states[self.state]
        producer = context.writes.get(self.state)
        if producer is None:
            update = context.binding.import_update(state, "canonical.activation:import@1")
            producer = update, context.binding.publish(update, state, when=When.BEFORE_MODEL)
        context.binding.activation_patch(
            state,
            *producer,
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
    """Externally produced parameter overlay: field values arrive as a StateTensorUpdate,
    are imported into private roots and published before the model runs."""

    def declare(self, context: _Declaration) -> None:
        """Import externally produced overlay buffers into private roots."""
        definition = context.state_definition(self.state, "external overlays require a declared state")
        _reject(
            definition.initialize is not None, "imported overlays take their values from the import; drop initialize"
        )
        context.binding.import_overlay(
            self.state,
            _buffers(definition),
            targets=self.targets,
            applied=self.applied_fields,
            operator=self.operator,
            scale=self.scale,
            namespace=definition.namespace,
            keys=definition.request_keys,
            scope=definition.scope,
        )


def _frozen(initialize: Initializer | None) -> Initializer | None:
    return None if initialize is None else freeze(initialize)


def _buffers(state: InstanceState) -> dict[str, Buffer]:
    _reject(
        state.optimizer is not None,
        "buffer effects update their state themselves; drop the optimizer",
    )
    _reject(state.alignment != 16, "buffer effect layouts use their original target alignment")
    _reject(
        any(callable(value) for value in state.fields.values()),
        "operand-dependent shapes require Buffer(input_size, output_size)",
        TypeError,
    )
    return {
        key: value if isinstance(value, Buffer) else Buffer(value, state.dtype)
        for key, value in state.fields.items()
        if not callable(value)
    }


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


@dataclass(frozen=True, kw_only=True)
class _Write(_Effect):
    state: str
    field: str

    phase: ClassVar[Phase | None] = Phase.BEFORE_MODEL

    def declare(self, operation: _Operation) -> None:
        context = operation.context
        definition = context.state_definition(self.state, "write needs a declared state")
        _reject(
            definition.feed is not None
            or definition.optimizer is not None
            or definition.scope is not StateScope.INSTANCE
            or self.state in context.writes
            or self.field not in definition.fields
            or operation.phase is not Phase.BEFORE_MODEL
            or operation.gradient is not None
            or operation.rows,
            "write needs one BEFORE_MODEL writer for a local InstanceState field without an optimizer",
        )
        context.definitions.append((operation.name, fingerprint(self)))
        call = operation.tensor()
        state = context.states[self.state]
        update = context.binding.write_result(call, state, self.field)
        context.writes[self.state] = (
            update,
            context.binding.publish(
                update,
                state,
                when=When.BEFORE_MODEL,
                name=f"{self.state}.publish",
            ),
        )


def write(*, state: str, field: str) -> Callable[[F], F]:
    """Publish a computed tensor into a local state field before the model runs."""
    return _mark(effect=_Write(state=state, field=field))


@dataclass(frozen=True)
class _Export(_Effect):
    name: str | Config
    every: int | Config
    max_rows: int | Config
    max_cols: int | Config
    enabled: bool | Config

    bounded: ClassVar[bool] = True
    phase: ClassVar[Phase | None] = Phase.OBSERVE

    def declare(self, operation: _Operation) -> None:
        """Publish bounded tensor observations to the host."""
        call, owner = operation.tensor(), operation.context.owner
        operation.context.binding.export(
            call,
            name=_setting(self.name, owner, str),
            every=_setting(self.every, owner, int),
            max_rows=_setting(self.max_rows, owner, int),
            max_cols=_setting(self.max_cols, owner, int),
            enabled=_setting(self.enabled, owner, bool),
        )


@dataclass(frozen=True, kw_only=True)
class _Train(_Effect):
    state: str
    update: str
    reduction: LossReduction
    steps: int | Config
    reset: str

    training: ClassVar[bool] = True

    def declare(self, operation: _Operation) -> None:
        """Choose optimizer training or the explicit weighted SGD transition."""
        arguments, captures = operation.arguments()
        definition = operation.context.state_definition(self.state, "training requires a declared optimizer state")
        if isinstance(definition.optimizer, SGD):
            self.weighted(operation, arguments, captures, definition, definition.optimizer)
        else:
            _reject(self.reset or self.steps != 1, "multi-step/reset training requires explicit SGD batch parameters")
            operation.context.train(operation.tensor(arguments, captures), self.state)

    def weighted(
        self,
        operation: _Operation,
        arguments: dict[str, _Source],
        captures: dict[str, Capture],
        definition: InstanceState,
        optimizer: SGD,
    ) -> None:
        """Declare the batch transition, including its reset and input bindings."""
        _reject(captures, "SGD batch training manages its own state writes")
        _reject(
            operation.gradient is not None
            or operation.rows
            or self.update
            or not self.reset
            or self.reduction != "weighted_state_mean_then_mean"
            or operation.phase is not Phase.AFTER_SAMPLE,
            "SGD batch training requires reset, weighted reduction and AFTER_SAMPLE without inference-row selectors",
        )
        context = operation.context
        sources = {key: source.batch(self.state, definition) for key, source in arguments.items()}
        reset = context.method(self.reset)
        identity = boundary_digest(fingerprint(operation.function), fingerprint(reset), sources, self)
        context.definitions.append((operation.name, identity))
        _reject(context.training is not None, "tensor sampler attachment requires one training operation")
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

    def declare(self, operation: _Operation) -> None:
        """Zero-strength guidance declares no computation or sampler attachment."""
        if self.beta is None or _setting(self.beta, operation.context.owner, float) != 0:
            call = operation.tensor()
            _reject(operation.context.prediction is not None, "tensor sampler attachment requires one logits operation")
            operation.context.prediction = call, self


@dataclass(frozen=True)
class _Gradient:
    inputs: tuple[str, ...]
    parameters: bool
    input_rows: str = ""
    parameter_rows: str = ""

    def contract(self, operation: _Operation, arguments: dict[str, _Source]) -> MathContract:
        """Validate derivative edges and identify parameter resources and detachments."""
        parameters = {
            key: parameter for key, source in arguments.items() if (parameter := _typed(source, Parameter)) is not None
        }
        _reject(self.parameters and not parameters, "parameter gradients require declared Parameter inputs")
        _reject(
            self.input_rows and not self.inputs or self.parameter_rows and not self.parameters,
            "gradient row selectors require their declared derivative edges",
        )
        _reject(
            len(set(self.inputs)) != len(self.inputs)
            or any(key not in arguments or key in parameters for key in self.inputs),
            "input gradients must name distinct nonparameter inputs",
        )
        _reject(
            any(not arguments[key].differentiable for key in self.inputs),
            "indices, versions and drawn tokens are nondifferentiable inputs",
        )
        _reject(
            operation.effect.bounded and (self.inputs or self.parameters or operation.rows),
            "bounded host exports use their own row limits and are gradient boundaries",
        )
        math = (
            MathContract(
                input_gradients=self.inputs,
                parameter_gradients=tuple(
                    dict.fromkeys(
                        operation.context.states[source.state].declaration.resource for source in parameters.values()
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
    replay: ConditionalLayerReplay | Config | None
    visibility: Visibility | Config

    phase: ClassVar[Phase | None] = Phase.MODEL_UPDATE

    def declare(self, operation: _Operation) -> None:
        """Declare a request buffer transition and its parameter overlay attachment."""
        _reject(
            operation.gradient is not None or operation.rows,
            "ordinary overlays update private buffers without cross-operation autograd",
        )
        _reject(operation.phase is not Phase.MODEL_UPDATE, "parameter overlays require MODEL_UPDATE")
        context = operation.context
        definition = context.state_definition(self.state, "parameter overlays require a declared request state")
        _reject(definition.scope is not StateScope.REQUEST, "parameter overlays require a declared request state")
        buffers = _buffers(definition)
        bindings = {key: source.overlay(self.state, buffers) for key, source in _arguments(operation.function).items()}
        _reject(
            set(buffers) & {source.value for source in Operand}, "overlay buffers conflict with reserved operand names"
        )
        targets, operator, scale = (
            _targets(_setting(self.targets, context.owner, tuple)),
            _setting(self.operator, context.owner, ParameterOverlayOperator),
            _setting(self.scale, context.owner, float),
        )
        effect = ParameterEffect(
            self.state, targets, self.weight, operator, _setting(self.factors, context.owner, tuple), scale
        )
        applied, replay = effect.applied_fields, Config.resolve(self.replay, context.owner)
        if replay is not None and not isinstance(replay, ConditionalLayerReplay):
            raise TypeError("overlay replay requires a conditional layer replay")
        visibility = _setting(self.visibility, context.owner, Visibility)
        _reject(
            replay is not None and visibility == Exact(),
            "Exact overlays use causal-history KV; remove post-forward layer replay",
        )
        identity = fingerprint(operation.function)
        context.binding.parameter_overlay(
            operation.name,
            self.state,
            buffers,
            initialize=_frozen(definition.initialize),
            targets=targets,
            applied=applied,
            namespace=definition.namespace,
            keys=definition.request_keys,
            phases=tuple(
                p
                for p, enabled in ((HiddenPhase.PREFILL, self.prefill), (HiddenPhase.DECODE, self.decode))
                if _setting(enabled, context.owner, bool)
            ),
            replay=replay,
            identity=f"python:{identity}",
            operator=operator,
            scale=scale,
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
    replay: ConditionalLayerReplay | Config | None = None,
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
            replay=replay,
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
) -> Callable[[F], F]:
    return _mark(effect=_Export(name, every, max_rows, max_cols, enabled))


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
    if not isinstance(resolved, expected) or type(resolved) is not expected:
        raise TypeError(f"{value}: expected {expected.__name__}")
    return resolved


def _targets(value: tuple[object, ...]) -> tuple[ParameterTarget, ...]:
    _reject(
        not all(isinstance(item, ParameterTarget) for item in value),
        "overlay targets require typed parameter targets",
        TypeError,
    )
    return tuple(item for item in value if isinstance(item, ParameterTarget))


def layer_route(*, tokens: Tokens = "all") -> Callable[[F], F]:
    """Declare a straight-line per-token layer program for the selected model rows."""
    if tokens not in ("prefill", "decode", "all"):
        raise ValueError('set layer_route(tokens=) to "prefill", "decode" or "all"')

    return _mark(tokens=tokens)


def _values(owner: object) -> dict[str, object]:
    return owner_values(owner, (InstanceState, Activation, Overlay))


def _snapshot(owner: C, borrowed: tuple[object, ...] = ()) -> C:
    return owner_snapshot(owner, (InstanceState, Activation, Overlay), borrowed)


def _typed(value: object, expected: type[C]) -> C | None:
    """Narrow an untrusted boundary marker to its declared type."""
    return value if isinstance(value, expected) else None


def _arguments(function: Callable[..., torch.Tensor | TensorUpdate]) -> dict[str, _Source]:
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
                _reject(not self.effect.training, f"{key}: only training operations capture inputs into memory")
                sources[key], captures[key] = captured.source, captured.capture
        return sources, captures

    def tensor(
        self,
        arguments: dict[str, _Source] | None = None,
        captures: dict[str, Capture] | None = None,
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
            when=self.phase.value,
            identity=f"python:{identity}",
            update=update,
            math=math,
            captures=writes,
        )(
            _invoke(
                freeze(self.function),
                {key: key for key in arguments},
                {key: source.adapt(context) for key, source in arguments.items()},
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
                )(_invoke(freeze(selector), {key: key for key in selected_args}))
            key = f"__{domain}"
            _reject(key in inputs, f"{key} is reserved for a row-domain input")
            inputs[key] = TaskResultInputSource(selectors[method].symbol, "value")
            math = replace(math, **{domain: key})
        return math


def _property(member: object) -> Callable[..., object] | None:
    """Find declaration getters without evaluating arbitrary owner properties."""
    return (
        member.fget
        if isinstance(member, property)
        and member.fget is not None
        and signature(member.fget, eval_str=False).return_annotation
        in (InstanceState, RequestState, Activation, Overlay)
        else None
    )


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
    routes: list[tuple[str, Tokens]] = field(default_factory=lambda: [])
    overlay_states: set[str] = field(default_factory=lambda: set[str]())
    states: dict[str, State] = field(default_factory=lambda: {})
    parameters: dict[tuple[str, str], SemanticInputSource | TensorPort] = field(default_factory=lambda: {})
    writes: dict[str, tuple[Call, Call]] = field(default_factory=lambda: {})
    prediction: tuple[Call, _Sampler] | None = None
    training: tuple[Call, State] | None = None

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
                if (
                    function := _typed(
                        state.initialize.__func__ if isinstance(state.initialize, MethodType) else state.initialize,
                        FunctionType,
                    )
                ) is not None:
                    seeds.append(function)
        context.freeze_helpers(candidates, seeds)
        context.bindings = tuple((key, fingerprint(value)) for key, value in sorted(values.items()))
        for name, member in members.items():
            if _property(member):
                values[name] = object.__getattribute__(owner, name)
        for value in values.values():
            if isinstance(value, Overlay):
                context.overlay_states.add(value.state)
        return context

    def read_operation(
        self, name: str, member: object, candidates: dict[str, FunctionType], seeds: list[FunctionType]
    ) -> None:
        """Read decorator attributes once; marker values never drive runtime reflection."""
        tokens = getattr(member, "__pllm_tokens__", None)
        if tokens is not None:
            assert tokens in ("prefill", "decode", "all") and isinstance(member, FunctionType)
            self.routes.append((name, tokens))
            seeds.append(member)
            return
        phase, marked, gradient, rows = (
            _typed(getattr(member, "__pllm_phase__", None), Phase),
            _typed(getattr(member, "__pllm_effect__", None), _Effect),
            getattr(member, "__pllm_gradient__", None),
            _typed(getattr(member, "__pllm_rows__", ""), str),
        )
        if phase is None and marked is None:
            return
        effect = marked or _Effect()
        texts = (*vars(effect).values(), *vars(_typed(gradient, _Gradient) or _Gradient((), False)).values(), rows)
        seeds.extend(candidates[text] for text in texts if isinstance(text, str) and text in candidates)
        if (overlay := _typed(effect, _Overlay)) is not None:
            self.overlay_states.add(overlay.state)
        phase = phase or effect.phase or Phase.AFTER_SAMPLE
        if (function := _typed(member, FunctionType)) is not None:
            seeds.append(function)
        if rows is None:
            raise TypeError("operation row selector must name a method")
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
        state = self.binding.buffer_state(
            name,
            _buffers(definition),
            namespace=definition.namespace,
            scope=definition.scope,
            keys=definition.request_keys,
        )
        self.states[name] = state
        ports = self.binding.parameter_ports(
            state, {field.name: field.shape for field in state.fields}, initialize=_frozen(definition.initialize)
        )
        self.parameters.update({(name, field): port for field, port in ports.items()})

    def declare_values(self) -> None:
        """Allocate ordinary state before effects and operations consume its ports."""
        for value in self.values.values():
            if isinstance(value, Activation):
                value.allocate(self)
        for name, value in self.values.items():
            if (state := _typed(value, InstanceState)) is not None:
                state.declare(name, self)

        for name, value in self.values.items():
            if isinstance(value, (Activation, Overlay)):
                self.definitions.append((name, fingerprint(value)))
                if isinstance(value, Overlay):
                    value.declare(self)

    def attach_effects(self) -> None:
        """Connect publications before replay and missing-input branches."""
        for value in self.values.values():
            if isinstance(value, Activation):
                value.declare(self)
        for name, tokens in self.routes:
            self.binding.layer_route(self.method(name), tokens)
        if self.binding.feeds:
            self.binding.conditional_input(tuple(feed.resource for feed in self.binding.feeds))

    def train(self, call: Call, state: str) -> None:
        _reject(self.training is not None, "tensor sampler attachment requires one training operation")
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
                beta=0.0 if sampler.beta is None else _setting(sampler.beta, self.owner, float),
            )
        elif self.training is not None and self.binding.attachment is None:
            self.binding.train_from_hidden(*self.training)

    def sign_composition(self) -> None:
        """Validate composed effects and sign ordered definitions and owner values."""
        b = self.binding
        if self.routes:
            attachment = b.attachment
            composed = attachment is not None and attachment.activation_bindings
            _reject(
                not composed and (len(b.graph.tasks) != 1 or b.graph.states),
                "layer_route requires a standalone model effect or Activation composition",
            )
        _reject(
            any(
                name not in {state.resource for state in b.graph.states}
                for name, value in self.values.items()
                if _typed(value, InstanceState) is not None
            ),
            "state buffers require an overlay effect; unused state is not allocated",
        )

        b.graph = replace(
            b.graph,
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

    @property
    def producers(self) -> tuple[TensorComputation, ...]:
        """Concurrent tensor producers whose storage can overlap every worker anchor."""
        return ()

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
                tuple((target.input_size, target.output_size) for target in facts.targets),
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
                    implementations=self.implementations(
                        str(task.symbol),
                        b.resources,
                        explicit_math=facts is not None and "compute/explicit_math@1" in facts.timeline.capabilities,
                    )
                    or task.implementations,
                )
                if task.symbol in b.tensor_tasks
                else task
                for task in b.graph.tasks
            ),
        )
        _reject(
            facts is not None
            and b.overlay_targets
            and tuple(target.target for target in facts.targets) != b.overlay_targets,
            "overlay targets differ from backend specialization",
        )
        return b

    def backend_requirements(self) -> BackendRequirements:
        b = self.declaration()
        return BackendRequirements(
            tuple(
                dict.fromkeys(
                    value.source
                    for task in b.graph.tasks
                    for value in task.inputs
                    if isinstance(value.source, HiddenInputSource)
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
                isinstance(port.source, SamplerFeedbackInputSource) for task in b.graph.tasks for port in task.inputs
            ),
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

        limits = [
            fact.input_shapes[0][0]
            for fact in b.resources
            if fact.task in b.tensor_tasks and fact.task not in b.exports and fact.input_shapes
        ]
        row_limit = min(limits) if limits and type(self).operations is Workload.operations else None

        def rows(value: TensorStepInput) -> tuple[PackedPhaseRow, ...]:
            _reject(
                row_limit is not None and len(value.rows.decode_rows) > row_limit,
                "operation rows exceed the signed workspace envelope",
            )
            _reject(
                b.probe is not None and value.rows.total_rows > b.probe.max_packed_rows,
                "packed input exceeds the signed backend row capacity",
            )
            if b.attachment is not None and b.attachment.activation_bindings:
                return value.rows.rows if b.graph.states[0].scope is StateScope.REQUEST else ()
            if b.attachment is not None and b.attachment.kind is AttachmentKind.MODEL_REPLAY:
                return ()
            return value.rows.request_rows if b.overlay_targets else value.rows.decode_rows

        return FunctionMath(
            {str(state.resource): initializer(str(state.resource)) for state in b.graph.states},
            rows,
            b.bind_prepare(),
            output,
            lambda previous, incoming: replace(incoming, entries=previous.entries),
            parameter_phases=b.parameter_phases,
        )

    def bind_math(
        self, facts: BackendCompileFacts, plan: PlanBundle
    ) -> MathBinding[TensorStepInput, TensorFrame, TensorOutput]:
        owner = _snapshot(self)
        b = owner.declaration(facts)
        _reject(
            b.graph != plan.semantic_graph or b.layouts != plan.tensor_layout_request,
            "tensor mathematics differ from the original signed declaration",
        )
        return owner.operations(facts, plan, b)

    def specialize(self, facts: BackendCompileFacts, options: ExecutionOptions) -> BackendCompileFacts:
        """Fund shared page storage for declared feeds before compiling the plan."""
        if self.declaration(facts).feeds:
            facts = replace(
                facts,
                providers=tuple(replace(provider, page_capacity=3, shared_memory=True) for provider in facts.providers),
            )
        return super().specialize(facts, options)

    def resource_facts(self, facts: BackendCompileFacts, options: ExecutionOptions) -> SchedulerTimeline:
        """Reserve ordinary tensor scratch and each requested graph's private pool."""
        _reject(options.explicit_math, "this workload has no explicit tensor implementation")
        with FakeTensorMode():
            binding = self.declaration(facts)
        timeline = derive_resources(binding, facts, options)
        concurrent = sum(producer.memory().workspace for producer in self.producers)
        return (
            replace(
                timeline,
                reserved_bytes=tuple(
                    value + concurrent for value in (timeline.reserved_bytes or (0,) * len(timeline.anchors))
                ),
            )
            if concurrent
            else timeline
        )

    def implementations(
        self, operation: str, resources: tuple[TaskResourceFact, ...], *, explicit_math: bool = False
    ) -> tuple[TaskImplementationSpec, ...]:
        _reject(explicit_math, "this workload has no explicit tensor implementation")
        if any(fact.task == operation and fact.resident_bytes for fact in resources):
            return (native(f"python.cuda_graph:{self.name}:{operation}@1", "compute/cuda_graph@1"),)
        return ()

    def operations(
        self, facts: BackendCompileFacts, plan: PlanBundle, b: TensorBinding
    ) -> MathBinding[TensorStepInput, TensorFrame, TensorOutput]:
        if plan.timeline.task_resources and type(self).operations is Workload.operations:
            b.configure_steps(facts, "compute/cuda_graph@1" in plan.timeline.capabilities)
        return MathBinding(
            self._functions(b, facts.device, facts.initialization_seed),
            b.bind_all(plan, model_probe=facts.probe),
            borrowed_resources=tuple(feed.resource for feed in b.feeds),
            feeds=b.feeds,
        )

    def declare_host(self, facts: HostRuntimeFacts) -> TensorHostBinding[TensorStepInput, TensorFrame, TensorOutput]:
        b = TensorBinding(
            self.name,
            boundary_digest(_values(self).get("config", ()), facts.hidden_size, facts.initialization_seed),
            Dimensions(facts.hidden_size, 0),
            host=True,
        )
        self._build(b)
        _reject(b.graph.states, "ordinary host tensor authoring currently supports stateless observations")
        return TensorHostBinding(
            b.graph,
            b.layouts,
            MathBinding(self._functions(b, facts.device, facts.initialization_seed), b.bind_all()),
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
