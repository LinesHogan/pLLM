"""Cold Python authoring boundary. Reflection stops at the original typed graph."""

from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass, field, fields, is_dataclass, replace
from enum import Enum
from inspect import Parameter as SignatureParameter
from inspect import signature
from types import FunctionType, MethodType
from typing import Annotated, ClassVar, TypeVar, cast, get_args, get_origin

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
from pllm.contracts.parameter_overlay import ParameterOverlayOperator, ParameterTarget
from pllm.contracts.request_identity import RequestKeyField, canonical_request_key_fields
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
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType, TensorLayoutRequest
from pllm.execution.codec import boundary_digest
from pllm.execution.declaration import Call, Dimensions, Shape, State, When, native
from pllm.execution.freeze import fingerprint, freeze, owner_snapshot, owner_values, reachable_functions
from pllm.execution.freeze import members as _members
from pllm.execution.host_runtime import (
    HostRuntimeFacts,
    HostWorkload,
    TensorHostBinding,
)
from pllm.execution.replay import Recurrence as Recurrence
from pllm.execution.state_execution import MathBinding, StateExecution
from pllm.execution.task_memory import derive_resources
from pllm.execution.tensor_binding import SGD as SGD
from pllm.execution.tensor_binding import Buffer as Buffer
from pllm.execution.tensor_binding import TensorBinding
from pllm.execution.tensor_binding import TensorOutput as TensorOutput
from pllm.execution.tensor_frame import TensorFrame, TensorPort, Tensors
from pllm.execution.tensor_frame import TensorUpdate as TensorUpdate
from pllm.execution.tensor_program import FunctionMath, InvocationOutput, Roots, TensorComputation
from pllm.kernels.adamw import AdamWConfig

F = TypeVar("F", bound=Callable[..., torch.Tensor | TensorUpdate])
C = TypeVar("C")


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
class Capture:
    """Second Annotated marker: on a training task, write this input's value into
    the state's memory field at publication — the next step reads it as the input
    version. Per lineage, the last selected row wins."""

    state: str
    field: str


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
    # Plain per-lineage tensors carried across steps; never optimizer-managed.
    memory: Mapping[str, Shape] = field(default_factory=lambda: dict[str, Shape](), kw_only=True)

    def __post_init__(self) -> None:
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("optimizer state requires a supported floating dtype")
        if self.shape is not None:
            if self.fields:
                raise ValueError("declare a single shape or named fields, not both")
            object.__setattr__(self, "fields", {"value": self.shape})
        if not self.fields:
            raise ValueError("state requires a shape or named fields")
        if self.memory:
            if self.optimizer is None:
                raise ValueError("memory fields ride on the training transition and require an optimizer")
            if set(self.memory) & set(self.fields):
                raise ValueError("memory fields duplicate a parameter field")
            if any(isinstance(shape, Buffer) for shape in self.memory.values()):
                raise TypeError("memory fields require tensor shapes, not overlay buffers")


@dataclass(frozen=True)
class RequestState(InstanceState):
    """Per-request parameters, read as tensors aligned to the current operation's rows."""

    keys: tuple[RequestKeyField, ...] = (RequestKeyField.REQUEST_ID,)

    def __post_init__(self) -> None:
        super().__post_init__()
        canonical_request_key_fields(self.keys)


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


@dataclass(frozen=True)
class Overlay:
    """Externally produced parameter overlay: field values arrive as a StateTensorUpdate,
    are imported into private roots and published before the model runs."""

    state: str
    targets: tuple[ParameterTarget, ...]
    weight: str = "value"
    operator: ParameterOverlayOperator = ParameterOverlayOperator.DENSE_ADD
    factors: tuple[str, str] = ("a", "b")
    scale: float = 1.0


def _buffers(state: InstanceState) -> dict[str, Buffer]:
    if state.optimizer is not None or state.initialize is not None:
        raise ValueError("buffer effects require zero-initialized state without an optimizer")
    if state.alignment != 16:
        raise ValueError("buffer effect layouts use their original target alignment")
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
    beta: float | Config | None = None  # None for additive logits, which have no guidance strength


@dataclass(frozen=True)
class _Gradient:
    inputs: tuple[str, ...]
    parameters: bool
    input_rows: str = ""
    parameter_rows: str = ""


def differentiate(
    *, inputs: tuple[str, ...] = (), parameters: bool = False, input_rows: str = "", parameter_rows: str = ""
) -> Callable[[F], F]:
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
    operator: ParameterOverlayOperator | Config
    factors: tuple[str, str] | Config
    scale: float | Config
    prefill: bool | Config
    decode: bool | Config
    replay: ConditionalLayerReplay | Config | None


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
) -> Callable[[F], F]:
    """Apply request-state fields to model parameters; dense uses `weight`, low-rank the factor pair."""

    def decorate(function: F) -> F:
        setattr(
            function,
            "__pllm_effect__",
            _Overlay(state, targets, weight, operator, factors, scale, prefill, decode, replay),
        )
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


def export(
    *,
    name: str | Config = "value",
    every: int | Config = 1,
    max_rows: int | Config = 1,
    max_cols: int | Config = 16,
    enabled: bool | Config = True,
) -> Callable[[F], F]:
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


def train(
    *, state: str, update: str = "", reduction: LossReduction = "mean", steps: int | Config = 1, reset: str = ""
) -> Callable[[F], F]:
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


def _values(owner: object) -> dict[str, object]:
    return owner_values(owner, (InstanceState, Activation, Overlay, Recurrence))


def _snapshot(owner: C, borrowed: tuple[object, ...] = ()) -> C:
    return owner_snapshot(owner, (InstanceState, Activation, Overlay, Recurrence), borrowed)


def _arguments(
    function: Callable[..., torch.Tensor | TensorUpdate],
    hidden: Hidden | None = None,
    result_type: type[torch.Tensor] | type[TensorUpdate] = torch.Tensor,
    captures: dict[str, Capture] | None = None,
) -> tuple[tuple[str, object], ...]:
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
            raise TypeError(
                f"{name}: input source requires Annotated[Tensor, source]; string annotations are not evaluated"
            )
        args: tuple[object, ...] = get_args(annotation)
        if len(args) == 3 and isinstance(args[2], Capture) and captures is not None:
            if args[0] is not torch.Tensor:
                raise TypeError(f"{name}: expected one source on Annotated[Tensor, source]")
            captures[name] = args[2]
            result.append((name, args[1]))
            continue
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
        owner = _snapshot(self)
        members = _members(owner)
        values = _values(owner)
        candidates: dict[str, FunctionType] = {}
        for cls in type(owner).__mro__:
            if cls is Workload:
                break
            for name, member in vars(cls).items():
                if name.startswith("__") or name in values or any(name in vars(base) for base in Workload.__mro__):
                    continue
                if isinstance(member, FunctionType) and name not in candidates:
                    candidates[name] = member
        # Helper signing is lazy: only methods reachable from bound mathematics
        # (operations, state-declaring properties, and methods named by effect
        # fields) are fingerprinted and frozen; unused methods neither enter
        # the signature nor can fail the declaration.
        seeds: list[FunctionType] = []
        for name, member in members.items():
            if isinstance(member, FunctionType) and isinstance(getattr(member, "__pllm_phase__", None), Phase):
                seeds.append(member)
            elif (
                isinstance(member, property)
                and member.fget is not None
                and (
                    signature(member.fget, eval_str=False).return_annotation
                    in (InstanceState, RequestState, Activation, Overlay, Recurrence)
                )
            ):
                seeds.append(member.fget)
            for attribute in ("__pllm_effect__", "__pllm_gradient__", "__pllm_rows__"):
                marker = getattr(member, attribute, None)
                texts = (
                    (marker,)
                    if isinstance(marker, str)
                    else (
                        tuple(getattr(marker, item.name) for item in fields(marker))
                        if is_dataclass(marker) and not isinstance(marker, type)
                        else ()
                    )
                )
                seeds.extend(candidates[text] for text in texts if isinstance(text, str) and text in candidates)
        for value in values.values():
            if isinstance(value, InstanceState) and value.initialize is not None:
                initialize = value.initialize.__func__ if isinstance(value.initialize, MethodType) else value.initialize
                if isinstance(initialize, FunctionType):
                    seeds.append(initialize)
        seeded = {name for name, member in candidates.items() if any(member is seed for seed in seeds)}
        reachable = reachable_functions(candidates, tuple(seeds)) | seeded
        helpers: list[tuple[str, str]] = []
        for name in candidates:
            if name not in reachable:
                continue
            function = cast(Callable[..., torch.Tensor], getattr(owner, name))
            helpers.append((name, fingerprint(function)))
            object.__setattr__(owner, name, freeze(function))
        bindings = tuple((key, fingerprint(value)) for key, value in sorted(values.items()))
        states: dict[str, State] = {}
        parameters: dict[tuple[str, str], SemanticInputSource | TensorPort] = {}
        definitions: list[tuple[str, str]] = helpers
        # Method-overlay states are created by parameter_overlay with their own buffer layout.
        overlay_states = {
            effect.state
            for member in members.values()
            if isinstance((effect := getattr(member, "__pllm_effect__", None)), _Overlay)
        }
        for name, member in members.items():
            if (
                isinstance(member, property)
                and member.fget is not None
                and (
                    signature(member.fget, eval_str=False).return_annotation
                    in (InstanceState, RequestState, Activation, Overlay, Recurrence)
                )
            ):
                values[name] = member.__get__(owner)
            elif isinstance(member, InstanceState) and name not in values:
                values[name] = member
        for name, definition in values.items():
            if isinstance(definition, InstanceState):
                definitions.append((name, fingerprint(definition)))
                if name in overlay_states:
                    continue
                if any(isinstance(shape, Buffer) for shape in definition.fields.values()):
                    if definition.optimizer is not None:
                        raise TypeError("optimizer parameters require tensor shapes, not overlay buffers")
                    continue  # Overlay/Activation buffer states are built by their effect branches.
                shapes: dict[str, Shape] = {}
                for key, shape in definition.fields.items():
                    if isinstance(shape, Buffer):
                        raise TypeError("tensor states require tensor shapes, not overlay buffers")
                    shapes[key] = shape
                state, ports = b.parameters(
                    name,
                    shapes,
                    optimizer=definition.optimizer,
                    namespace=definition.namespace,
                    initialize=None if definition.initialize is None else freeze(definition.initialize),
                    scope=StateScope.REQUEST if isinstance(definition, RequestState) else StateScope.INSTANCE,
                    dtype=TensorDType(str(definition.dtype).removeprefix("torch.")),
                    alignment=definition.alignment,
                    keys=definition.keys if isinstance(definition, RequestState) else (),
                    memory=definition.memory,
                )
                states[name] = state
                parameters.update({(name, field): port for field, port in ports.items()})
        for name, effect in values.items():
            if isinstance(effect, Recurrence):
                definitions.append((name, fingerprint(effect)))
                b.recurrence(effect)
                continue
            if isinstance(effect, Overlay):
                definition = values.get(effect.state)
                if not isinstance(definition, InstanceState):
                    raise ValueError("external overlays require a declared state")
                buffers = _buffers(definition)
                applied = (effect.weight,) if effect.operator is ParameterOverlayOperator.DENSE_ADD else effect.factors
                definitions.append((name, fingerprint(effect)))
                b.import_overlay(
                    effect.state,
                    buffers,
                    targets=effect.targets,
                    applied=applied,
                    operator=effect.operator,
                    scale=effect.scale,
                    namespace=definition.namespace,
                    keys=definition.keys if isinstance(definition, RequestState) else (),
                    scope=StateScope.REQUEST if isinstance(definition, RequestState) else StateScope.INSTANCE,
                )
                continue
            if not isinstance(effect, Activation):
                continue
            definition = values.get(effect.state)
            if not isinstance(definition, InstanceState):
                raise ValueError("activation replacement requires a declared state")
            definitions.append((name, fingerprint(effect)))
            b.activation_patch(
                effect.state,
                _buffers(definition),
                layers=effect.layers,
                operator=effect.operator,
                first_tokens=effect.first_tokens,
                last_tokens=effect.last_tokens,
                namespace=definition.namespace,
                decode=effect.decode,
                decode_refresh=effect.decode_refresh,
                on_missing=effect.on_missing,
                scope=StateScope.REQUEST if isinstance(definition, RequestState) else StateScope.INSTANCE,
                keys=definition.keys if isinstance(definition, RequestState) else (),
            )

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
            if isinstance(effect, _Sampler) and effect.beta is not None and _setting(effect.beta, owner, float) == 0:
                continue  # Zero-strength guidance leaves sampling unchanged, so nothing is computed for it.
            if isinstance(effect, _Overlay):
                if gradient is not None or forward_rows:
                    raise ValueError("ordinary overlays update private buffers without cross-operation autograd")
                if phase is not Phase.MODEL_UPDATE:
                    raise ValueError("parameter overlays require MODEL_UPDATE")
                definition = values.get(effect.state)
                if not isinstance(definition, RequestState):
                    raise ValueError("parameter overlays require a declared request state")
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
                operator = _setting(effect.operator, owner, ParameterOverlayOperator)
                scale = _setting(effect.scale, owner, float)
                if operator is ParameterOverlayOperator.DENSE_ADD:
                    applied = (effect.weight,)
                else:
                    factors = _setting(effect.factors, owner, tuple)
                    if len(factors) != 2:
                        raise TypeError("low-rank overlays require two factor field names")
                    applied = factors
                replay: object = effect.replay
                if isinstance(replay, Config):
                    path = replay.name
                    replay = owner
                    for part in path.split("."):
                        replay = getattr(replay, part)
                if replay is not None and not isinstance(replay, ConditionalLayerReplay):
                    raise TypeError("overlay replay requires a conditional layer replay")
                identity = fingerprint(transition)
                transition = freeze(transition)

                def invoke(
                    inputs: Tensors,
                    function: Callable[..., TensorUpdate] = transition,
                    bindings: Mapping[str, str] = bindings_overlay,
                ) -> TensorUpdate:
                    return function(**{key: inputs[source] for key, source in bindings.items()})

                b.parameter_overlay(
                    name,
                    effect.state,
                    buffers,
                    targets=resolved_targets,
                    applied=applied,
                    namespace=definition.namespace,
                    keys=definition.keys,
                    phases=tuple(
                        p
                        for p, enabled in ((HiddenPhase.PREFILL, effect.prefill), (HiddenPhase.DECODE, effect.decode))
                        if _setting(enabled, owner, bool)
                    ),
                    replay=replay,
                    identity=f"python:{identity}",
                    operator=operator,
                    scale=scale,
                    function=invoke,
                )
                definitions.append((name, boundary_digest(identity, bindings_overlay, operator, scale, applied)))
                continue
            function = cast(Callable[..., torch.Tensor], getattr(owner, name))
            hidden: object = getattr(member, "__pllm_hidden__", None)
            capture_marks: dict[str, Capture] = {}
            arguments = _arguments(
                function,
                hidden if isinstance(hidden, Hidden) else None,
                captures=capture_marks if isinstance(effect, _Train) else None,
            )
            if isinstance(effect, _Train):
                definition = values.get(effect.state)
                if not isinstance(definition, InstanceState):
                    raise ValueError("training requires a declared optimizer state")
                if isinstance(definition.optimizer, SGD):
                    if capture_marks:
                        raise ValueError("SGD batch training manages its own state writes")
                    if (
                        gradient is not None
                        or forward_rows
                        or effect.update
                        or not effect.reset
                        or effect.reduction != "weighted_state_mean_then_mean"
                        or phase is not Phase.AFTER_SAMPLE
                    ):
                        raise ValueError(
                            "SGD batch training requires reset, weighted reduction and AFTER_SAMPLE without inference-row selectors"
                        )
                    batch_sources: dict[str, tuple[str, str]] = {}
                    for key, source in arguments:
                        if isinstance(source, Batch) and source.field in ("hidden", "base_logits", "labels", "weights"):
                            batch_sources[key] = "batch", source.field
                        elif (
                            isinstance(source, Parameter)
                            and source.state == effect.state
                            and source.field in definition.fields
                            and not source.per_row
                        ):
                            batch_sources[key] = "parameter", source.field
                        else:
                            raise ValueError("batch training inputs must name its batch tensors or own parameters")
                    reset = cast(Callable[[Tensors, torch.Generator], None], getattr(owner, effect.reset))
                    identity = boundary_digest(fingerprint(function), fingerprint(reset), batch_sources, effect)
                    definitions.append((name, identity))
                    if training is not None:
                        raise ValueError("tensor sampler attachment requires one training operation")
                    state = states[effect.state]
                    training = (
                        b.weighted_train(
                            name,
                            state,
                            identity=f"python:{identity}",
                            optimizer=definition.optimizer,
                            steps=_setting(effect.steps, owner, int),
                            reset=freeze(reset),
                            function=_invoke(freeze(function), tuple(batch_sources)),
                            arguments=batch_sources,
                        ),
                        state,
                    )
                    continue
                if effect.reset or effect.steps != 1:
                    raise ValueError("multi-step/reset training requires explicit SGD batch parameters")
            inputs = {key: port(source) for key, source in arguments}
            identity = fingerprint(function)
            function = freeze(function)
            definitions.append(
                (
                    name,
                    boundary_digest(
                        identity,
                        tuple((key, fingerprint(source)) for key, source in arguments),
                        fingerprint(capture_marks),
                    ),
                )
            )
            update: State | None = None
            math: MathContract = MathContract()
            parameter_names = tuple(key for key, source in arguments if isinstance(source, Parameter))
            demand = gradient if isinstance(gradient, _Gradient) else _Gradient((), isinstance(effect, _Train))
            if demand.parameters and not parameter_names:
                raise ValueError("parameter gradients require declared Parameter inputs")
            if demand.input_rows and not demand.inputs or demand.parameter_rows and not demand.parameters:
                raise ValueError("gradient row selectors require their declared derivative edges")
            if len(set(demand.inputs)) != len(demand.inputs) or any(
                key not in inputs or key in parameter_names for key in demand.inputs
            ):
                raise ValueError("input gradients must name distinct nonparameter inputs")
            if any(
                key in demand.inputs and isinstance(source, (Samples, SampledTokens, StateIndex, StateVersion))
                for key, source in arguments
            ):
                raise ValueError("indices, versions and drawn tokens are nondifferentiable inputs")
            if isinstance(effect, _Export) and (demand.inputs or demand.parameters or forward_rows):
                raise ValueError("bounded host exports use their own row limits and are gradient boundaries")
            if isinstance(gradient, _Gradient) or isinstance(effect, _Train):
                math = MathContract(
                    input_gradients=demand.inputs,
                    parameter_gradients=tuple(
                        dict.fromkeys(
                            states[source.state].declaration.resource
                            for _, source in arguments
                            if isinstance(source, Parameter)
                        )
                    )
                    if demand.parameters
                    else (),
                    detached_inputs=tuple(
                        key
                        for key, source in arguments
                        if not isinstance(source, Parameter) and key not in demand.inputs
                    ),
                )
            if isinstance(effect, _Train):
                update = states[effect.state]
                math = replace(
                    math,
                    update_states=parameter_names[0] if parameter_names else "__update_state",
                    reduction=effect.reduction,
                )
            captures: dict[str, LayoutFieldId] = {}
            for key, capture in capture_marks.items():
                if update is None or capture.state != cast(_Train, effect).state:
                    raise ValueError("memory captures target the task's own training state")
                if capture.field not in {field.name for field in update.fields[update.optimizer_fields :]}:
                    raise ValueError(f"{capture.field}: captures name a declared memory field")
                captures[key] = update.field(capture.field)
            domains = {
                "forward_rows": forward_rows,
                "input_gradient_rows": demand.input_rows,
                "parameter_gradient_rows": demand.parameter_rows,
                "update_states": effect.update if isinstance(effect, _Train) else "",
            }
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
                    identity_selector = fingerprint(selector)
                    definitions.append((method, identity_selector))
                    selectors[method] = b.tensor(
                        f"__{name}_{method}",
                        when=phase.value,
                        identity=f"python:{identity_selector}:{name}:{method}",
                        inputs={key: inputs[key] for key, _ in selected_args},
                    )(_invoke(freeze(selector), tuple(key for key, _ in selected_args)))
                key = f"__{domain}"
                if key in inputs:
                    raise ValueError(f"{key} is reserved for a row-domain input")
                inputs[key] = TaskResultInputSource(selectors[method].symbol, "value")
                math = replace(math, **{domain: key})
            call = b.tensor(
                name,
                inputs=inputs,
                when=phase.value,
                identity=f"python:{identity}",
                update=update,
                math=math,
                captures=captures,
            )(_invoke(function, tuple(key for key, _ in arguments)))
            if isinstance(effect, _Export):
                b.export(
                    call,
                    name=_setting(effect.name, owner, str),
                    every=_setting(effect.every, owner, int),
                    max_rows=_setting(effect.max_rows, owner, int),
                    max_cols=_setting(effect.max_cols, owner, int),
                    enabled=_setting(effect.enabled, owner, bool),
                )
            elif isinstance(effect, _Train):
                if training is not None:
                    raise ValueError("tensor sampler attachment requires one training operation")
                training = call, states[effect.state]
            elif isinstance(effect, _Sampler):
                if prediction is not None:
                    raise ValueError("tensor sampler attachment requires one logits operation")
                prediction = call, effect
        b.validate_gradients(None if training is None else training[0])
        if prediction is not None:
            if training is None:
                raise ValueError("logits attachment requires a training operation")
            call, sampler = prediction
            b.logits(
                call,
                *training,
                stage=sampler.stage,
                beta=0.0 if sampler.beta is None else _setting(sampler.beta, owner, float),
            )
        elif training is not None and b.attachment is None:
            b.train_from_hidden(*training)
        if any(isinstance(value, Recurrence) for value in values.values()):
            attachment = b.attachment
            composed = (
                attachment is not None
                and attachment.activation_bindings
                and {state.resource for state in b.graph.states} == {attachment.state_resource}
                and {task.symbol for task in b.graph.tasks} == {symbol for _, symbol in attachment.tasks}
            )
            if not composed and (len(b.graph.tasks) != 1 or b.graph.states):
                raise ValueError("tokenwise recurrence requires a standalone model effect")
        declared = {state.resource for state in b.graph.states}
        if any(name not in declared for name, value in values.items() if isinstance(value, InstanceState)):
            raise ValueError("state buffers require an overlay effect; unused state is not allocated")
        # Include instance-local function replacements and configuration in the original signature.
        b.graph = replace(
            b.graph,
            execution_parameters_digest=boundary_digest(b.graph.execution_parameters_digest, definitions, bindings),
        )

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
                getattr(self, "config", ()), dimensions, None if facts is None else facts.initialization_seed
            ),
            dimensions,
        )
        b.resources = () if facts is None else facts.timeline.task_resources
        b.probe = None if facts is None else facts.probe
        self._build(b)
        resources = () if facts is None else facts.timeline.task_resources
        b.graph = replace(
            b.graph,
            tasks=tuple(
                replace(
                    task,
                    implementations=self.implementations(
                        str(task.symbol),
                        resources,
                        explicit_math=facts is not None and "compute/explicit_math@1" in facts.timeline.capabilities,
                    )
                    or task.implementations,
                )
                for task in b.graph.tasks
            ),
        )
        if (
            facts is not None
            and b.overlay_targets
            and tuple(target.target for target in facts.targets) != b.overlay_targets
        ):
            raise ValueError("overlay targets differ from backend specialization")
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

        def initializer(resource: str) -> Callable[[Roots], None]:
            init = b.initializers.get(resource)

            def initialize(roots: Roots) -> None:
                if init is not None:
                    for root in roots:
                        init(root, generator)

            return initialize

        def output(data: TensorFrame, invocation: InvocationOutput) -> TensorOutput:
            return invocation.value

        if b.graph.states:

            def output(data: TensorFrame, invocation: InvocationOutput) -> TensorOutput:
                return b.result(data, invocation)

        limits = [
            fact.input_shapes[0][0]
            for fact in b.resources
            if fact.task in b.tensor_tasks and fact.task not in b.exports and fact.input_shapes
        ]
        row_limit = min(limits) if limits and type(self).operations is Workload.operations else None

        def rows(value: TensorStepInput) -> tuple[PackedPhaseRow, ...]:
            if row_limit is not None and len(value.rows.decode_rows) > row_limit:
                raise ValueError("operation rows exceed the signed workspace envelope")
            if b.probe is not None and value.rows.total_rows > b.probe.max_packed_rows:
                raise ValueError("packed input exceeds the signed backend row capacity")
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
        )

    def bind_math(
        self, facts: BackendCompileFacts, plan: PlanBundle
    ) -> MathBinding[TensorStepInput, TensorFrame, TensorOutput]:
        owner = _snapshot(self)
        b = owner.declaration(facts)
        if b.graph != plan.semantic_graph or b.layouts != plan.tensor_layout_request:
            raise ValueError("tensor mathematics differ from the original signed declaration")
        return owner.operations(facts, plan, b)

    def resource_facts(self, facts: BackendCompileFacts, options: ExecutionOptions) -> SchedulerTimeline:
        """Reserve ordinary tensor scratch and each requested graph's private pool."""
        if options.explicit_math:
            raise ValueError("this workload has no explicit tensor implementation")
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
        if explicit_math:
            raise ValueError("this workload has no explicit tensor implementation")
        if any(fact.task == operation and fact.resident_bytes for fact in resources):
            return (native(f"python.cuda_graph:{self.name}:{operation}@1", "compute/cuda_graph@1"),)
        return ()

    def operations(
        self, facts: BackendCompileFacts, plan: PlanBundle, b: TensorBinding
    ) -> MathBinding[TensorStepInput, TensorFrame, TensorOutput]:
        if plan.timeline.task_resources and type(self).operations is Workload.operations:
            b.configure_steps(facts, "compute/cuda_graph@1" in plan.timeline.capabilities)
        return MathBinding(
            self._functions(b, facts.device, facts.initialization_seed), b.bind_all(plan, model_probe=facts.probe)
        )

    def declare_host(self, facts: HostRuntimeFacts) -> TensorHostBinding[TensorStepInput, TensorFrame, TensorOutput]:
        b = TensorBinding(
            self.name,
            boundary_digest(getattr(self, "config", ()), facts.hidden_size, facts.initialization_seed),
            Dimensions(facts.hidden_size, 0),
            host=True,
        )
        self._build(b)
        if b.graph.states:
            raise ValueError("ordinary host tensor authoring currently supports stateless observations")
        return TensorHostBinding(
            b.graph,
            b.layouts,
            MathBinding(self._functions(b, facts.device, facts.initialization_seed), b.bind_all()),
            facts,
        )


__all__ = [
    "Workload",
    "Phase",
    "on",
    "hook",
    "export",
    "add_logits",
    "guide_sampling",
    "train",
    "Config",
    "Hidden",
    "Samples",
    "BaseLogits",
    "SampledTokens",
    "Parameter",
    "StateIndex",
    "StateVersion",
    "Result",
    "InstanceState",
    "RequestState",
    "overlay",
    "Operand",
    "Buffer",
    "TensorUpdate",
    "differentiate",
    "Activation",
    "Overlay",
    "Recurrence",
    "SGD",
    "Batch",
]
