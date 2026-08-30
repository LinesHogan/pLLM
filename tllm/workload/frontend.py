"""Single-decorator frontend for auxiliary workloads."""

from __future__ import annotations

import inspect
import torch
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum, StrEnum
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Generic, Never, ParamSpec, Protocol, Self, TypeVar, cast, get_origin, get_type_hints, overload
from weakref import WeakKeyDictionary

from tllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from tllm.contracts.request_identity import RequestKeyField
from tllm.workload.config import ConfigAtom, ConfigField, WorkloadConfig
from tllm.workload.staging import StagedPort
from tllm.workload.host import HostOutput
from tllm.workload.ir import (
    AffineAccessIR,
    AffineExprIR,
    ApproximationCriteria,
    ApproximationPolicy,
    DomainIR,
    EffectIR,
    EffectKind,
    HiddenLayerKind,
    ImplementationIR,
    ImplementationKind,
    OptimizerSpec,
    StateIR,
    StateScope,
    StateStorage,
    TaskIR,
    TaskInputIR,
    TaskOutputIR,
    TimingIR,
    ValueIR,
    ValueSourceKind,
    ValueTargetKind,
    WorkloadEdge,
    WorkloadEdgeKind,
    WorkloadIR,
)

if TYPE_CHECKING:
    from tllm.ports.consumer import BaseConsumer


_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T")
def _name(value: str, label: str) -> str:
    if not value:
        raise ValueError(f"{label} must be non-empty")
    return value


@dataclass(frozen=True, slots=True)
class RequestKey:
    fields: tuple[RequestKeyField, ...]

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("request_key requires at least one field")
        if len(self.fields) != len(set(self.fields)):
            raise ValueError("request_key fields must be unique")


def request_key(*fields: RequestKeyField) -> RequestKey:
    return RequestKey(tuple(fields))


@dataclass(frozen=True, slots=True)
class _FixedLearningRate:
    value: float

    def option(self) -> str:
        if self.value <= 0:
            raise ValueError("optimizer learning rate must be positive")
        return self.value.hex()


@dataclass(frozen=True, slots=True)
class _ConfiguredLearningRate:
    field_name: str

    def option(self) -> str:
        return f"field:{self.field_name}"


_LearningRate = _FixedLearningRate | _ConfiguredLearningRate


class AdamWSpec(OptimizerSpec):
    def __init__(
        self,
        *,
        lr: _LearningRate,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(
            kind="adamw",
            options=(
                ("betas", repr(tuple(float(value) for value in betas))),
                ("eps", float(eps).hex()),
                ("lr", lr.option()),
                ("weight_decay", float(weight_decay).hex()),
            ),
        )

    @classmethod
    def fixed(
        cls,
        lr: float,
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> Self:
        return cls(
            lr=_FixedLearningRate(lr),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    @classmethod
    def configured(
        cls,
        lr: ConfigField[float],
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> Self:
        return cls(
            lr=_ConfiguredLearningRate(lr.name),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )


class AdamSpec(OptimizerSpec):
    """Signed Adam declaration (no decoupled weight decay).

    Optimizer kinds are an open extension point: ``kind`` plus the sorted
    ``options`` pairs are the signed identity, so future fast-weight or
    weight-overlay optimizers plug in here without changing the State or
    Region contracts.
    """

    def __init__(
        self,
        *,
        lr: _LearningRate,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(
            kind="adam",
            options=(
                ("betas", repr(tuple(float(value) for value in betas))),
                ("eps", float(eps).hex()),
                ("lr", lr.option()),
            ),
        )

    @classmethod
    def fixed(
        cls,
        lr: float,
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> Self:
        return cls(lr=_FixedLearningRate(lr), betas=betas, eps=eps)

    @classmethod
    def configured(
        cls,
        lr: ConfigField[float],
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> Self:
        return cls(lr=_ConfiguredLearningRate(lr.name), betas=betas, eps=eps)


StateValueT = TypeVar("StateValueT")
ConfiguredT_co = TypeVar("ConfiguredT_co", bound=ConfigAtom, covariant=True)
ConfiguredT = TypeVar("ConfiguredT", bound=ConfigAtom)
PortT = TypeVar("PortT")
PortT_co = TypeVar("PortT_co", covariant=True)


class _StateOwner(Protocol[StateValueT]):
    def state_value(self, state: "State[StateValueT]") -> StateValueT: ...


class _StateReference(Protocol):
    @property
    def scope(self) -> StateScope: ...

    @property
    def optimizer(self) -> OptimizerSpec | None: ...

    @property
    def transition_kind(self) -> str: ...

    @property
    def name(self) -> str: ...


class _StateEffectTarget(Protocol):
    @property
    def effect_state(self) -> _StateReference: ...

    def make_effect(self, kind: EffectKind) -> "StateEffect": ...


class _StateSelection(Protocol):
    @property
    def selected_state(self) -> _StateReference: ...


class ConfiguredValue(Protocol[ConfiguredT_co]):
    def resolve(self, config: WorkloadConfig) -> ConfiguredT_co: ...


@dataclass(frozen=True, slots=True)
class ConstantValue(Generic[ConfiguredT]):
    value: ConfiguredT

    def resolve(self, config: WorkloadConfig) -> ConfiguredT:
        del config
        return self.value


@dataclass(frozen=True, slots=True)
class ConfigFieldValue(Generic[ConfiguredT]):
    name: str

    def resolve(self, config: WorkloadConfig) -> ConfiguredT:
        return cast(ConfiguredT, config.raw_value(self.name))


@dataclass(frozen=True, slots=True)
class NoConfiguredValue:
    def resolve(self, config: WorkloadConfig) -> Never:
        del config
        raise RuntimeError("this declaration has no configured value")


NO_CONFIGURED_VALUE = NoConfiguredValue()


class _PublishedModuleOwner(Protocol):
    def publish_module(
        self,
        state: PublishedModule[object],
        values: Mapping[str, torch.Tensor],
    ) -> None: ...


class _OptimizerOwner(Protocol):
    def optimize_state(
        self, state: TrainableState[object], loss: torch.Tensor
    ) -> None: ...


def configured(
    value: ConfiguredT | ConfigField[ConfiguredT],
) -> ConfiguredValue[ConfiguredT]:
    if isinstance(value, ConfigField):
        return ConfigFieldValue(value.name)
    return ConstantValue(value)


class State(Generic[StateValueT]):
    """A named state family declared on a workload class."""

    storage = StateStorage.DYNAMIC

    def __init__(
        self,
        *,
        scope: StateScope,
        key: RequestKey | None = None,
        parameters: str | tuple[str, ...] | None,
        optimizer: OptimizerSpec | None = None,
        transition: str = "optimizer",
    ) -> None:
        self.scope = StateScope(scope)
        self._infer_parameter_name = parameters is None
        if parameters is None:
            parameters = ("__state_name__",)
        if isinstance(parameters, str):
            parameters = (parameters,)
        if not parameters or any(not parameter for parameter in parameters):
            raise TypeError("State.parameters must be a non-empty str or tuple[str, ...]")
        if self.scope is StateScope.REQUEST and key is None:
            raise ValueError("REQUEST State requires key=request_key(...)")
        if self.scope is not StateScope.REQUEST and key is not None:
            raise ValueError("only REQUEST State may declare a request_key")
        self.key = key
        self.parameters = parameters
        self.optimizer = optimizer
        self.transition_kind = _name(transition, "State.transition")
        self._name = ""
        self._owner: type[object] | None = None
        self._initializer: Callable[..., StateValueT] | None = None

    @classmethod
    def replica(
        cls,
        *,
        optimizer: OptimizerSpec,
        transition: str = "optimizer",
    ) -> "TrainableState[StateValueT]":
        return TrainableState(
            scope=StateScope.REPLICA,
            parameters=None,
            optimizer=optimizer,
            transition=transition,
        )

    @classmethod
    def published_module(cls) -> "PublishedModule[StateValueT]":
        return PublishedModule(
            scope=StateScope.REPLICA,
            parameters=None,
            optimizer=None,
            transition="host_publish",
        )

    @classmethod
    def service(
        cls, *, transition: str = "service_transition"
    ) -> "ServiceState[StateValueT]":
        """Declare replica-owned typed service state with an explicit transition."""

        return ServiceState(
            scope=StateScope.REPLICA,
            parameters=None,
            optimizer=None,
            transition=transition,
        )

    def __set_name__(self, owner: type[object], name: str) -> None:
        self._owner = owner
        self._name = _name(name, "State name")
        if self._infer_parameter_name:
            self.parameters = (self._name,)

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> Self: ...

    @overload
    def __get__(
        self,
        instance: _StateOwner[StateValueT],
        owner: type[object] | None = None,
    ) -> StateValueT: ...

    def __get__(
        self,
        instance: _StateOwner[StateValueT] | None,
        owner: type[object] | None = None,
    ) -> "Self | StateValueT":
        if instance is None:
            return self
        return instance.state_value(self)

    def initialize(self, function: Callable[..., StateValueT]) -> Callable[..., StateValueT]:
        """Bind the cold initializer for this State."""

        if not callable(function):
            raise TypeError("State.initialize requires a callable")
        if self._initializer is not None:
            raise RuntimeError(f"State {self.name!r} already has an initializer")
        self._initializer = function
        return function

    @property
    def initializer(self) -> Callable[..., StateValueT]:
        if self._initializer is None:
            raise RuntimeError(f"State {self.name!r} has no initializer")
        return self._initializer

    @property
    def name(self) -> str:
        if not self._name:
            raise ValueError("State must be assigned to a workload class attribute")
        return self._name

    def to_ir(self) -> StateIR:
        return StateIR(
            name=self.name,
            scope=self.scope,
            request_key=() if self.key is None else self.key.fields,
            parameters=self.parameters,
            optimizer=self.optimizer,
            storage=self.storage,
            transition=self.transition_kind,
        )

    @property
    def effect_state(self) -> Self:
        return self

    def make_effect(self, kind: EffectKind) -> "StateEffect":
        return StateEffect(kind, self)

    def selection(self) -> "StateSelectionInput[StateValueT]":
        """Bind a task-local input to this request-owned state family."""

        if self.scope is not StateScope.REQUEST:
            raise ValueError("State.selection requires a REQUEST State")
        return StateSelectionInput(self)


class TrainableState(State[StateValueT]):
    optimizer: OptimizerSpec
    storage = StateStorage.OPTIMIZER_STABLE


class ServiceState(State[StateValueT]):
    """Non-optimizer state mutated only by a signed extern task."""


class PublishedModule(State[StateValueT]):
    def __init__(
        self,
        *,
        scope: StateScope,
        key: RequestKey | None = None,
        parameters: str | tuple[str, ...] | None,
        optimizer: OptimizerSpec | None = None,
        transition: str = "host_publish",
    ) -> None:
        super().__init__(
            scope=scope,
            key=key,
            parameters=parameters,
            optimizer=optimizer,
            transition=transition,
        )
        self.port: StagedPort[Mapping[str, torch.Tensor]] = StagedPort()

    def __set_name__(self, owner: type[object], name: str) -> None:
        super().__set_name__(owner, name)
        self.port.__set_name__(owner, f"{name}__publish")


def _state_id(state: _StateReference) -> str:
    return state.name


@dataclass(frozen=True, slots=True)
class StateSelectionInput(Generic[StateValueT]):
    """The exact request-state lease consumed by one task declaration."""

    selected_state: _StateReference

    def __post_init__(self) -> None:
        if self.selected_state.scope is not StateScope.REQUEST:
            raise ValueError("StateSelectionInput requires a REQUEST State")

    @property
    def effect_state(self) -> _StateReference:
        return self.selected_state

    def make_effect(self, kind: EffectKind) -> "StateEffect":
        return StateEffect(kind, self.selected_state, self)


@dataclass(frozen=True, slots=True)
class StateEffect:
    kind: EffectKind
    state: _StateReference
    selection: _StateSelection | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", EffectKind(self.kind))
        if self.kind is EffectKind.EXTERN_IO:
            raise ValueError("extern_io uses ExternalEffect, not StateEffect")
        if self.state.scope is StateScope.REQUEST and self.selection is None:
            raise ValueError("request-state effects require State.selection()")
        if self.state.scope is not StateScope.REQUEST and self.selection is not None:
            raise ValueError("only request-state effects may declare a selection")
        if self.selection is not None and self.selection.selected_state is not self.state:
            raise ValueError("StateEffect selection must belong to its state")

    @property
    def resource_name(self) -> str:
        return _state_id(self.state)


@dataclass(frozen=True, slots=True)
class ExternalEffect:
    resource: str
    kind: EffectKind = field(init=False, default=EffectKind.EXTERN_IO)

    def __post_init__(self) -> None:
        object.__setattr__(self, "resource", _name(self.resource, "extern_io resource"))

    @property
    def resource_name(self) -> str:
        return self.resource


WorkloadEffect = StateEffect | ExternalEffect


def read(target: _StateEffectTarget) -> StateEffect:
    return _effect(EffectKind.READ, target)


def current(target: _StateEffectTarget) -> StateEffect:
    return _effect(EffectKind.CURRENT, target)


def transition(target: _StateEffectTarget) -> StateEffect:
    state = target.effect_state
    if state.optimizer is None and state.transition_kind == "optimizer":
        raise ValueError("transition requires a State optimizer or explicit version conversion")
    return _effect(EffectKind.TRANSITION, target)


def maybe_transition(
    target: _StateEffectTarget,
) -> StateEffect:
    """Declare a *conditional* transition: the version may or may not advance.

    Use this when whether the update fires is data-dependent and cannot be
    proven at signing time (e.g. a zero-kept-row step that skips the optimizer
    update entirely).  Ordering, merge legality, and capture safety treat it
    exactly like a strict ``transition`` — the signed graph cannot prove the
    mutation will not happen — while the runtime version audit accepts either
    ``+0`` (skipped) or ``+1`` (fired) per mutating effect, never more.  A
    conditional transition therefore never weakens the signed ordering
    contract; it only widens the audited version delta.
    """
    state = target.effect_state
    if state.optimizer is None and state.transition_kind == "optimizer":
        raise ValueError(
            "maybe_transition requires a State optimizer or explicit version conversion"
        )
    return _effect(EffectKind.TRANSITION_MAYBE, target)


def next(target: _StateEffectTarget) -> StateEffect:
    return _effect(EffectKind.NEXT, target)


def extern_io(resource: str) -> ExternalEffect:
    """Declare an external I/O effect on a named channel (e.g. "trace:file").

    The resource names the external channel, not a workload State.  A task
    with an extern_io effect can never be internalized into a merged Region
    (the planner forces a Region boundary: the signed graph has no ordering
    proof for external side effects) and is never capture-safe.  This is the
    effect-level complement of ``implementation_kind=EXTERN``: an EXTERN
    implementation says *who* runs the task (a registered extern executor),
    while extern_io says the task *observes or mutates something outside* the
    signed state model regardless of who runs it.
    """
    return ExternalEffect(resource)


def _effect(
    kind: EffectKind,
    target: _StateEffectTarget,
) -> StateEffect:
    return target.make_effect(kind)


@dataclass(frozen=True, slots=True)
class HiddenInput:
    layer: str
    layer_kind: HiddenLayerKind
    site: HiddenSite
    phase: HiddenPhase
    path: ConfiguredValue[str] | NoConfiguredValue = NO_CONFIGURED_VALUE

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer", _name(self.layer, "HiddenInput.layer"))
        object.__setattr__(self, "layer_kind", HiddenLayerKind(self.layer_kind))
        object.__setattr__(self, "site", HiddenSite(self.site))
        object.__setattr__(self, "phase", HiddenPhase(self.phase))
        if self.site is HiddenSite.NONE or self.phase is HiddenPhase.NONE:
            raise ValueError("hidden input requires site and phase")


@dataclass(frozen=True, slots=True)
class ResultInput:
    task: str
    output: str


@dataclass(frozen=True, slots=True)
class ProvidedInput:
    name: str


@dataclass(frozen=True, slots=True)
class StepIdInput:
    pass


@dataclass(frozen=True, slots=True)
class StagedInput(Generic[PortT_co]):
    port: StagedPort[PortT_co]


@dataclass(frozen=True, slots=True)
class SamplerRuntimeInput:
    kind: ValueSourceKind


@dataclass(frozen=True, slots=True)
class SinkOutput:
    name: str


@dataclass(frozen=True, slots=True)
class TemporaryOutput:
    storage: str = "temporary"


@dataclass(frozen=True, slots=True)
class OptimizeOutput:
    _state: object
    at: Before | After | Between
    conditional: bool = False

    def trainable_state(self) -> TrainableState[object]:
        return cast(TrainableState[object], self._state)


@dataclass(frozen=True, slots=True)
class SamplerOutput:
    name: str
    kind: SamplerOutputKind
    beta: ConfiguredValue[float]
    enabled: ConfiguredValue[bool]
    backend: ConfiguredValue[str]
    required: bool = False


class SamplerOutputKind(StrEnum):
    HIDDEN_REWEIGHT = "hidden_reweight"
    ADD_LOGITS = "add_logits"


@dataclass(frozen=True, slots=True)
class HostOutputSpec(Generic[PortT_co]):
    port: HostOutput[PortT_co]


InputSpec = (
    HiddenInput
    | ResultInput
    | ProvidedInput
    | StepIdInput
    | StagedInput[object]
    | SamplerRuntimeInput
    | _StateSelection
)
OutputSpec = (
    SinkOutput | TemporaryOutput | OptimizeOutput | SamplerOutput | HostOutputSpec[object]
)


def hidden(
    layer: int | str | object = "*",
    site: HiddenSite | str = HiddenSite.POST,
    phase: HiddenPhase | str = HiddenPhase.DECODE,
    *,
    path: str | ConfigField[str] | None = None,
) -> HiddenInput:
    if layer is Ellipsis:
        layer = "*"
    if not isinstance(layer, (int, str)) or isinstance(layer, bool):
        raise TypeError("hidden.layer must be int, str, or Ellipsis")
    if isinstance(layer, int):
        layer_kind = HiddenLayerKind.INDEX
    elif layer == "*":
        layer_kind = HiddenLayerKind.WILDCARD
    else:
        layer_kind = HiddenLayerKind.PATH
    return HiddenInput(
        str(layer),
        layer_kind,
        HiddenSite(site),
        HiddenPhase(phase),
        NO_CONFIGURED_VALUE if path is None else configured(path),
    )


def result(task_name: str, output_name: str) -> ResultInput:
    return ResultInput(_name(task_name, "result.task"), _name(output_name, "result.output"))


def provided(name: str) -> ProvidedInput:
    return ProvidedInput(_name(name, "provided.name"))


def step_id() -> StepIdInput:
    return StepIdInput()


def staged(port: StagedPort[PortT]) -> StagedInput[PortT]:
    return StagedInput(port)


def sink(name: str) -> SinkOutput:
    return SinkOutput(_name(name, "sink.name"))


def temporary() -> TemporaryOutput:
    return TemporaryOutput()


def host_output(port: HostOutput[PortT]) -> HostOutputSpec[PortT]:
    return HostOutputSpec(port)


def optimize(
    state: TrainableState[StateValueT], *, at: Before | After | Between
) -> OptimizeOutput:
    return OptimizeOutput(state, at)


def maybe_optimize(
    state: TrainableState[StateValueT], *, at: Before | After | Between
) -> OptimizeOutput:
    """Optimize only when the task returns a Tensor instead of ``None``."""

    return OptimizeOutput(state, at, conditional=True)


class _SamplerOutputs:
    def request_ids(self) -> SamplerRuntimeInput:
        return SamplerRuntimeInput(ValueSourceKind.SAMPLER_REQUEST_IDS)

    def sample_indices(self) -> SamplerRuntimeInput:
        return SamplerRuntimeInput(ValueSourceKind.SAMPLER_SAMPLE_IDXS)

    def sample_index_tensor(self) -> SamplerRuntimeInput:
        return SamplerRuntimeInput(
            ValueSourceKind.SAMPLER_SAMPLE_IDX_TENSOR
        )

    def logits(self) -> SamplerRuntimeInput:
        return SamplerRuntimeInput(ValueSourceKind.SAMPLER_LOGITS)

    def sampled_tokens(self) -> SamplerRuntimeInput:
        return SamplerRuntimeInput(ValueSourceKind.SAMPLED_TOKENS)

    def hidden_reweight(
        self,
        name: str = "guidance",
        *,
        beta: float | ConfigField[float],
        enabled: bool | ConfigField[bool] = True,
        backend: str | ConfigField[str] = "post_filter_exact",
        required: bool = False,
    ) -> SamplerOutput:
        return SamplerOutput(
            name=_name(name, "sampler output"),
            kind=SamplerOutputKind.HIDDEN_REWEIGHT,
            beta=configured(beta),
            enabled=configured(enabled),
            backend=configured(backend),
            required=bool(required),
        )

    def add_logits(
        self,
        name: str = "logit_delta",
        *,
        enabled: bool | ConfigField[bool] = True,
        required: bool = True,
    ) -> SamplerOutput:
        return SamplerOutput(
            name=_name(name, "sampler output"),
            kind=SamplerOutputKind.ADD_LOGITS,
            beta=ConstantValue(1.0),
            enabled=configured(enabled),
            backend=ConstantValue("pre_filter_dense"),
            required=required,
        )


sampler = _SamplerOutputs()


@dataclass(frozen=True, slots=True)
class Before:
    anchor: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor", _name(self.anchor, "Before.anchor"))


@dataclass(frozen=True, slots=True)
class After:
    anchor: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor", _name(self.anchor, "After.anchor"))


@dataclass(frozen=True, slots=True)
class Between:
    start: str
    end: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _name(self.start, "Between.start"))
        object.__setattr__(self, "end", _name(self.end, "Between.end"))
        if self.start == self.end:
            raise ValueError("Between requires distinct anchors")


Timing = Before | After | Between


@dataclass(frozen=True, slots=True)
class AffineExpr:
    coefficients: tuple[tuple[str, int], ...] = ()
    constant: int = 0

    def _combine(self, other: "AffineExpr", sign: int) -> "AffineExpr":
        coefficients: dict[str, int] = dict(self.coefficients)
        for name, value in other.coefficients:
            coefficients[name] = coefficients.get(name, 0) + sign * value
        return AffineExpr(tuple(sorted((name, value) for name, value in coefficients.items() if value)), self.constant + sign * other.constant)

    def __add__(self, other: "int | Index | AffineExpr") -> "AffineExpr":
        return self._combine(_as_affine(other), 1)

    def __radd__(self, other: "int | Index | AffineExpr") -> "AffineExpr":
        return self + other

    def __sub__(self, other: "int | Index | AffineExpr") -> "AffineExpr":
        return self._combine(_as_affine(other), -1)

    def __rsub__(self, other: "int | Index | AffineExpr") -> "AffineExpr":
        return _as_affine(other)._combine(self, -1)

    def __mul__(self, factor: int) -> "AffineExpr":
        if isinstance(factor, bool):
            raise TypeError("affine expressions may only be multiplied by an int")
        return AffineExpr(tuple((name, value * factor) for name, value in self.coefficients), self.constant * factor)

    __rmul__ = __mul__

    def to_ir(self) -> AffineExprIR:
        return AffineExprIR(self.coefficients, self.constant)


@dataclass(frozen=True, slots=True)
class Index:
    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "Index.name"))

    def expr(self) -> AffineExpr:
        return AffineExpr(((self.name, 1),), 0)

    def __add__(self, other: int | Index | AffineExpr) -> AffineExpr:
        return self.expr() + other

    __radd__ = __add__

    def __sub__(self, other: int | Index | AffineExpr) -> AffineExpr:
        return self.expr() - other

    def __rsub__(self, other: int | Index | AffineExpr) -> AffineExpr:
        return other - self.expr()

    def __mul__(self, factor: int) -> AffineExpr:
        return self.expr() * factor

    __rmul__ = __mul__


def _as_affine(value: int | Index | AffineExpr) -> AffineExpr:
    if isinstance(value, AffineExpr):
        return value
    if isinstance(value, Index):
        return value.expr()
    if isinstance(value, bool):
        raise TypeError("expected an affine Index, AffineExpr, or int")
    return AffineExpr((), value)


@dataclass(frozen=True, slots=True)
class Domain:
    name: str
    indices: tuple[Index, ...] = ()
    bounds: tuple[tuple[str, int | str, int | str], ...] = ()
    parameters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "Domain.name"))
        names = tuple(index.name for index in self.indices)
        if len(names) != len(set(names)):
            raise ValueError("Domain index names must be unique")
        if any(not value for value in self.parameters):
            raise TypeError("Domain.parameters must be tuple[str, ...]")

    def to_ir(self) -> DomainIR:
        return DomainIR(
            name=self.name,
            indices=tuple(index.name for index in self.indices),
            bounds=tuple((str(index), str(lower), str(upper)) for index, lower, upper in self.bounds),
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class AffineAccess:
    value: str
    indices: tuple[Index | AffineExpr | int, ...]
    mode: str = "read"

    def to_ir(self) -> AffineAccessIR:
        return AffineAccessIR(
            value=_name(self.value, "AffineAccess.value"),
            indices=tuple(_as_affine(index).to_ir() for index in self.indices),
            mode=self.mode,
        )


@dataclass(frozen=True, slots=True)
class ExternImplementation:
    namespace: str
    name: str
    version: str = "1"
    input_types: tuple[str, ...] = ()
    output_types: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    # When set, task() fills input_types/output_types from the function's
    # annotations (the encoding rule is the type's __name__, None -> "None").
    derive_types: bool = False
    kind: ImplementationKind = ImplementationKind.EXTERN

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "namespace", _name(self.namespace, "ExternImplementation.namespace")
        )
        object.__setattr__(self, "name", _name(self.name, "ExternImplementation.name"))
        object.__setattr__(self, "kind", ImplementationKind(self.kind))
        object.__setattr__(self, "version", _name(self.version, "ExternImplementation.version"))
        for values, field_name in (
            (self.input_types, "input_types"),
            (self.output_types, "output_types"),
            (self.capabilities, "capabilities"),
        ):
            if any(not item for item in values):
                raise TypeError(f"ExternImplementation.{field_name} must be tuple[str, ...]")
        if self.derive_types and (self.input_types or self.output_types):
            raise ValueError(
                "ExternImplementation with derive_types=True must not also "
                "declare input_types/output_types"
            )

    @property
    def key(self) -> str:
        return f"{self.namespace}:{self.name}@{self.version}"

    def to_ir(self) -> ImplementationIR:
        if self.derive_types:
            raise RuntimeError("extern() implementation escaped task() without type derivation")
        return ImplementationIR(
            kind=self.kind,
            key=self.key,
            input_types=self.input_types,
            output_types=self.output_types,
            capabilities=self.capabilities,
        )


def extern(key: object, *, capabilities: tuple[str, ...] = ()) -> ExternImplementation:
    """Declare an extern implementation by key; task() derives the signed
    input/output types from the function annotations."""
    if not isinstance(key, str):
        raise TypeError("extern key must be str")
    namespace, sep, rest = key.partition(":")
    name, at, version = rest.partition("@")
    if not sep or not namespace or not name:
        raise ValueError("extern key must have the form 'namespace:name[@version]'")
    return ExternImplementation(
        namespace=namespace,
        name=name,
        version=version if at else "1",
        capabilities=capabilities,
        derive_types=True,
    )


def _annotation_type_name(hint: object, label: str) -> str:
    """Encode one resolved annotation as its signed type string."""
    if hint is None or hint is type(None):
        return "None"
    origin = get_origin(hint)
    if isinstance(origin, type):
        return origin.__name__
    if not isinstance(hint, type):
        raise TypeError(
            f"{label} annotation {hint!r} is not a concrete class; "
            "extern types must be plain classes"
        )
    return hint.__name__


class ExecutorRegistry:
    def __init__(self) -> None:
        self._executors: dict[str, Callable[..., object]] = {}

    def register(self, implementation: ExternImplementation | str, executor: Callable[..., object]) -> None:
        key = implementation.key if isinstance(implementation, ExternImplementation) else _name(implementation, "executor key")
        if not callable(executor):
            raise TypeError("executor must be callable")
        if key in self._executors:
            raise ValueError(f"executor {key!r} is already registered")
        self._executors[key] = executor

    def resolve(self, implementation: ExternImplementation | str) -> Callable[..., object]:
        key = implementation.key if isinstance(implementation, ExternImplementation) else implementation
        try:
            return self._executors[key]
        except KeyError as exc:
            raise KeyError(f"executor {key!r} is not registered") from exc


@dataclass(frozen=True, slots=True)
class TaskDecl:
    inputs: tuple[tuple[str, InputSpec], ...]
    outputs: tuple[tuple[str, OutputSpec], ...]
    effects: tuple[WorkloadEffect, ...]
    timing: Timing
    implementations: tuple[ExternImplementation, ...]
    domain: Domain
    accesses: tuple[AffineAccess, ...]
    predicate: ConfigPredicate | None = None


@dataclass(frozen=True, slots=True)
class ConfigPredicate:
    field_name: str
    expected: ConfigAtom

    def matches(self, config: WorkloadConfig) -> bool:
        return config.raw_value(self.field_name) == self.expected


def when(field: ConfigField[ConfiguredT], expected: ConfiguredT) -> ConfigPredicate:
    """Select a semantic task variant from Rust-normalized config."""

    return ConfigPredicate(field.name, expected)


_TASK_DECLARATIONS: WeakKeyDictionary[FunctionType, TaskDecl] = WeakKeyDictionary()


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    """One immutable task specification and its author or generated executor."""

    name: str
    declaration: TaskDecl
    executor: FunctionType
    generated_after: str | None = None
    is_gradient_task: bool = False
    gradient_tensor_parameters: frozenset[str] = frozenset()

    def bind(self, instance: object) -> Callable[..., object]:
        bound = MethodType(self.executor, instance)
        if not self.is_gradient_task:
            return bound

        parameter_names = tuple(name for name, _source in self.declaration.inputs)

        def run_with_grad(*args: object, **kwargs: object) -> object:
            with torch.inference_mode(False), torch.enable_grad():
                return bound(
                    *(
                        cast(torch.Tensor, value).clone()
                        if parameter_names[index] in self.gradient_tensor_parameters
                        else value
                        for index, value in enumerate(args)
                    ),
                    **{
                        key: (
                            cast(torch.Tensor, value).clone()
                            if key in self.gradient_tensor_parameters
                            else value
                        )
                        for key, value in kwargs.items()
                    },
                )

        return run_with_grad


def _mapping(value: Mapping[str, _T] | None, label: str) -> tuple[tuple[str, _T], ...]:
    if value is None:
        return ()
    result: list[tuple[str, _T]] = []
    for key, item in value.items():
        result.append((_name(key, f"{label} key"), item))
    return tuple(result)


@overload
def task(function: Callable[_P, _R], /) -> Callable[_P, _R]: ...


@overload
def task(
    function: None = None,
    /,
    *,
    inputs: Mapping[str, InputSpec] | None = None,
    outputs: Mapping[str, OutputSpec] | None = None,
    effects: tuple[WorkloadEffect, ...] = (),
    timing: Timing = Before("step.end"),
    implementation: ExternImplementation | tuple[ExternImplementation, ...] | None = None,
    domain: Domain = Domain("scalar"),
    accesses: tuple[AffineAccess, ...] = (),
    when: ConfigPredicate | None = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...


def task(
    function: Callable[_P, _R] | None = None,
    /,
    *,
    inputs: Mapping[str, InputSpec] | None = None,
    outputs: Mapping[str, OutputSpec] | None = None,
    effects: tuple[WorkloadEffect, ...] = (),
    timing: Timing = Before("step.end"),
    implementation: ExternImplementation | tuple[ExternImplementation, ...] | None = None,
    domain: Domain = Domain("scalar"),
    accesses: tuple[AffineAccess, ...] = (),
    when: ConfigPredicate | None = None,
) -> Callable[_P, _R] | Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    task_inputs = _mapping(inputs, "task.inputs")
    task_outputs = _mapping(outputs, "task.outputs")
    if implementation is None:
        implementations: tuple[ExternImplementation, ...] = ()
    elif isinstance(implementation, ExternImplementation):
        implementations = (implementation,)
    else:
        implementations = implementation

    def decorate(target: Callable[_P, _R]) -> Callable[_P, _R]:
        nonlocal implementations
        if not isinstance(target, FunctionType):
            raise TypeError("task requires a Python function")
        parameters = set(inspect.signature(target).parameters)
        parameters.discard("self")
        parameters.discard("cls")
        unknown = tuple(name for name, _ in task_inputs if name not in parameters)
        if unknown:
            raise ValueError(f"task inputs are not function parameters: {', '.join(unknown)}")
        for effect in effects:
            match effect:
                case StateEffect(selection=selection) if selection is not None:
                    ports = tuple(
                        name for name, source in task_inputs if source is selection
                    )
                    if len(ports) != 1:
                        raise ValueError(
                            "request-state effect requires its exact selection input exactly once"
                        )
                case _:
                    pass
        if implementations:
            try:
                hints = get_type_hints(target)
            except (NameError, TypeError) as exc:
                raise TypeError("extern task annotations must resolve") from exc
            missing = tuple(name for name in parameters if name not in hints)
            if missing or "return" not in hints:
                raise TypeError("extern task inputs and output require concrete annotations")
            if any(item.derive_types for item in implementations):
                undeclared = tuple(
                    name for name in parameters if name not in dict(task_inputs)
                )
                if undeclared:
                    raise TypeError(
                        "extern task parameters must all be declared inputs: "
                        + ", ".join(undeclared)
                    )
                derived_inputs = tuple(
                    _annotation_type_name(hints[name], f"task input {name!r}")
                    for name, _ in task_inputs
                )
                derived_outputs = (
                    _annotation_type_name(hints["return"], "task return"),
                )
                implementations = tuple(
                    replace(
                        item,
                        input_types=derived_inputs,
                        output_types=derived_outputs,
                        derive_types=False,
                    )
                    if item.derive_types
                    else item
                    for item in implementations
                )

        _TASK_DECLARATIONS[target] = TaskDecl(
            task_inputs,
            task_outputs,
            effects,
            timing,
            implementations,
            domain,
            accesses,
            when,
        )
        return target

    return decorate if function is None else decorate(function)


class WorkloadAuthority(str, Enum):
    """The execution runtime explicitly authorized for a declaration."""

    NATIVE_REGION = "native_region"
    LEGACY_ADAPTER = "legacy_adapter"


LegacyConsumerFactory = Callable[[object], "BaseConsumer[object]"]


@dataclass(frozen=True, slots=True)
class NoLegacyConsumerAdapter:
    def build(self, instance: object) -> "BaseConsumer[object]":
        _ = instance
        raise RuntimeError("workload definition has no legacy consumer binding")


NO_LEGACY_CONSUMER_ADAPTER = NoLegacyConsumerAdapter()


@dataclass(frozen=True, slots=True)
class LegacyConsumerAdapter:
    """Explicit compatibility binding captured when a workload is declared."""

    factory: LegacyConsumerFactory

    def build(self, instance: object) -> "BaseConsumer[object]":
        return self.factory(instance)


LegacyConsumerBinding = NoLegacyConsumerAdapter | LegacyConsumerAdapter


@dataclass(frozen=True, slots=True)
class WorkloadDefinition:
    workload_id: str
    approximation: ApproximationPolicy
    authority: WorkloadAuthority = WorkloadAuthority.NATIVE_REGION
    approximation_criteria: ApproximationCriteria | None = None
    tasks: tuple[TaskDefinition, ...] = ()
    hidden_inputs: tuple[HiddenInput, ...] = ()
    sampler_outputs: tuple[SamplerOutput, ...] = ()
    gradient_tasks: tuple[str, ...] = ()
    sampler_inputs: tuple[ValueSourceKind, ...] = ()
    states: tuple[State[object], ...] = ()
    staged_ports: tuple[StagedPort[object], ...] = ()
    host_outputs: tuple[HostOutput[object], ...] = ()
    legacy_consumer: LegacyConsumerBinding = NO_LEGACY_CONSUMER_ADAPTER


@dataclass(frozen=True, slots=True)
class NativeWorkloadDefinition:
    workload_id: str
    approximation: ApproximationPolicy
    authority: WorkloadAuthority
    approximation_criteria: ApproximationCriteria | None
    config_type: type[WorkloadConfig]
    enabled: ConfigFieldValue[bool]
    tasks: tuple[TaskDefinition, ...] = ()
    hidden_inputs: tuple[HiddenInput, ...] = ()
    sampler_outputs: tuple[SamplerOutput, ...] = ()
    gradient_tasks: tuple[str, ...] = ()
    sampler_inputs: tuple[ValueSourceKind, ...] = ()
    states: tuple[State[object], ...] = ()
    staged_ports: tuple[StagedPort[object], ...] = ()
    host_outputs: tuple[HostOutput[object], ...] = ()
    legacy_consumer: LegacyConsumerBinding = NO_LEGACY_CONSUMER_ADAPTER


_Definition = WorkloadDefinition | NativeWorkloadDefinition
_WORKLOAD_DEFINITIONS: WeakKeyDictionary[type[object], _Definition] = WeakKeyDictionary()


@dataclass(frozen=True, slots=True)
class WorkloadDeclaration:
    ir: WorkloadIR
    instance: object
    executors: tuple[tuple[str, Callable[..., object]], ...]
    authority: WorkloadAuthority = WorkloadAuthority.NATIVE_REGION
    implementation_executors: tuple[
        tuple[str, str, Callable[..., object]], ...
    ] = ()
    # Host-side parity validation parameters for approximation workloads;
    # deliberately outside the signed IR and workload identity.
    approximation_criteria: ApproximationCriteria | None = None

    def executor(
        self,
        task_name: str,
        implementation_id: str | None = None,
    ) -> Callable[..., object]:
        """Return the callable declared for one exact task implementation."""

        if implementation_id is None:
            for name, executor in self.executors:
                if name == task_name:
                    return executor
            raise KeyError(task_name)
        for name, key, executor in self.implementation_executors:
            if name == task_name and key == implementation_id:
                return executor
        raise KeyError((task_name, implementation_id))


def workload(
    cls: type[_T] | None = None,
    /,
    *,
    id: str | None = None,
    approximation: ApproximationPolicy = ApproximationPolicy.EXACT,
    authority: WorkloadAuthority = WorkloadAuthority.NATIVE_REGION,
    approximation_criteria: ApproximationCriteria | None = None,
    config: type[WorkloadConfig] | None = None,
    enabled: ConfigField[bool] | None = None,
    legacy_consumer: LegacyConsumerFactory | None = None,
) -> type[_T] | Callable[[type[_T]], type[_T]]:
    def decorate(target: type[_T]) -> type[_T]:
        policy = ApproximationPolicy(approximation)
        resolved_authority = WorkloadAuthority(authority)
        if resolved_authority is WorkloadAuthority.LEGACY_ADAPTER:
            if legacy_consumer is None:
                raise TypeError(
                    "legacy-adapter workload declaration requires legacy_consumer="
                )
            legacy_binding: LegacyConsumerBinding = LegacyConsumerAdapter(
                legacy_consumer
            )
        else:
            if legacy_consumer is not None:
                raise TypeError(
                    "legacy_consumer= is only valid with LEGACY_ADAPTER authority"
                )
            legacy_binding = NO_LEGACY_CONSUMER_ADAPTER
        if approximation_criteria is not None:
            if policy is ApproximationPolicy.EXACT:
                raise ValueError(
                    "an exact workload must not declare approximation criteria"
                )
        workload_id = id or f"{target.__module__}.{target.__qualname__}"
        if config is None and enabled is None:
            definition: _Definition = WorkloadDefinition(
                _name(workload_id, "workload.id"),
                policy,
                resolved_authority,
                approximation_criteria,
                legacy_consumer=legacy_binding,
            )
        elif config is not None and enabled is not None:
            if resolved_authority is WorkloadAuthority.LEGACY_ADAPTER:
                raise TypeError(
                    "legacy-adapter declarations cannot use the native config surface"
                )
            definition = NativeWorkloadDefinition(
                workload_id=_name(workload_id, "workload.id"),
                approximation=policy,
                authority=resolved_authority,
                approximation_criteria=approximation_criteria,
                config_type=config,
                enabled=ConfigFieldValue(enabled.name),
                legacy_consumer=legacy_binding,
            )
        else:
            raise TypeError(
                "native workload config and enabled ConfigField must be declared together"
            )
        members = _class_members(target)
        state_list: list[State[object]] = []
        for _name_, member in members:
            if isinstance(member, State):
                state_list.append(cast(State[object], member))
        states = tuple(state_list)
        author_tasks = _author_task_definitions(target)
        tasks = _expanded_task_definitions(author_tasks, states)
        hidden_inputs, sampler_outputs, gradient_tasks, sampler_inputs = _author_metadata(tasks)
        published_ports: list[StagedPort[object]] = []
        for state in states:
            if isinstance(state, PublishedModule):
                published_ports.append(cast(StagedPort[object], state.port))
        definition = replace(
            definition,
            tasks=tasks,
            hidden_inputs=hidden_inputs,
            sampler_outputs=sampler_outputs,
            gradient_tasks=gradient_tasks,
            sampler_inputs=sampler_inputs,
            states=states,
            staged_ports=(
                *published_ports,
                *(
                    cast(StagedPort[object], member)
                    for _name_, member in members
                    if isinstance(member, StagedPort)
                ),
            ),
            host_outputs=tuple(
                cast(HostOutput[object], member)
                for _name_, member in members
                if isinstance(member, HostOutput)
            ),
        )
        _WORKLOAD_DEFINITIONS[target] = definition
        # Validate at class definition time without constructing an instance.
        _build_ir(target, definition)
        return target

    return decorate if cls is None else decorate(cls)


def _author_task_definitions(target: type[object]) -> tuple[TaskDefinition, ...]:
    definitions: list[TaskDefinition] = []
    for name, member in _class_members(target):
        if not isinstance(member, FunctionType):
            continue
        declaration = _TASK_DECLARATIONS.get(member)
        if declaration is None:
            continue
        hints = get_type_hints(member)
        definitions.append(
            TaskDefinition(
                name,
                declaration,
                member,
                gradient_tensor_parameters=frozenset(
                    parameter
                    for parameter, _source in declaration.inputs
                    if hints.get(parameter) is torch.Tensor
                ),
            )
        )
    return tuple(definitions)


def task_declaration(function: FunctionType) -> TaskDecl:
    try:
        return _TASK_DECLARATIONS[function]
    except KeyError as exc:
        raise TypeError("function is not decorated with @task") from exc


def _expanded_task_definitions(
    author_tasks: tuple[TaskDefinition, ...],
    states: tuple[State[object], ...],
) -> tuple[TaskDefinition, ...]:
    publishers = _published_module_tasks(states)
    optimized = _optimized_task_definitions(author_tasks)
    source_order = {
        definition.name: index
        for index, definition in enumerate((*author_tasks, *publishers))
    }

    def task_position(entry: TaskDefinition) -> tuple[int, bool]:
        if entry.generated_after is not None:
            return source_order[entry.generated_after], True
        return source_order[entry.name], False

    return tuple(
        sorted(
            (*optimized, *publishers),
            key=task_position,
        )
    )


def _published_module_tasks(
    states: tuple[State[object], ...],
) -> tuple[TaskDefinition, ...]:
    generated: list[TaskDefinition] = []
    for state in states:
        if not isinstance(state, PublishedModule):
            continue
        published = state
        generated_name = f"publish__{_state_id(published)}"

        def publisher_executor(
            published: PublishedModule[object],
        ) -> Callable[..., None]:
            def publish(
                self: _PublishedModuleOwner,
                values: Mapping[str, torch.Tensor],
            ) -> None:
                self.publish_module(published, values)

            return publish

        generated.append(
            TaskDefinition(
                name=generated_name,
                declaration=TaskDecl(
                    inputs=(("values", StagedInput(published.port)),),
                    outputs=(),
                    effects=(transition(published),),
                    timing=Before("next.prepare"),
                    implementations=(ExternImplementation(
                        namespace="tllm.workload",
                        name="published_module_transition",
                        version="1",
                        input_types=("Mapping",),
                        output_types=("None",),
                    ),),
                    domain=Domain("scalar"),
                    accesses=(),
                ),
                executor=publisher_executor(published),
            )
        )
    return tuple(generated)


def _author_metadata(
    tasks: tuple[TaskDefinition, ...],
) -> tuple[
    tuple[HiddenInput, ...],
    tuple[SamplerOutput, ...],
    tuple[str, ...],
    tuple[ValueSourceKind, ...],
]:
    hidden_inputs: list[HiddenInput] = []
    sampler_outputs: list[SamplerOutput] = []
    gradient_tasks: list[str] = []
    sampler_inputs: list[ValueSourceKind] = []
    for definition in tasks:
        declaration = definition.declaration
        hidden_inputs.extend(
            source
            for _parameter, source in declaration.inputs
            if isinstance(source, HiddenInput)
        )
        sampler_outputs.extend(
            output
            for _output_name, output in declaration.outputs
            if isinstance(output, SamplerOutput)
        )
        sampler_inputs.extend(
            source.kind
            for _parameter, source in declaration.inputs
            if isinstance(source, SamplerRuntimeInput)
        )
        if definition.is_gradient_task:
            gradient_tasks.append(definition.name)
    if len(sampler_outputs) > 1:
        raise ValueError("one workload may declare at most one sampler output")
    return (
        tuple(hidden_inputs),
        tuple(sampler_outputs),
        tuple(gradient_tasks),
        tuple(dict.fromkeys(sampler_inputs)),
    )


def _optimized_task_definitions(
    author_tasks: tuple[TaskDefinition, ...],
) -> tuple[TaskDefinition, ...]:
    """Lower author-level optimize outputs to ordinary signed tasks."""

    pending: list[tuple[str, str, OptimizeOutput, ConfigPredicate | None]] = []
    expanded: list[TaskDefinition] = []
    known_names = {entry.name for entry in author_tasks}
    for entry in author_tasks:
        task_name = entry.name
        declaration = entry.declaration
        optimized = [
            (output_name, output)
            for output_name, output in declaration.outputs
            if isinstance(output, OptimizeOutput)
        ]
        if not optimized:
            expanded.append(entry)
            continue
        effects = list(declaration.effects)
        rewritten_outputs: list[tuple[str, OutputSpec]] = []
        for output_name, output in declaration.outputs:
            if not isinstance(output, OptimizeOutput):
                rewritten_outputs.append((output_name, output))
                continue
            trainable_state = output.trainable_state()
            has_read_effect = False
            for effect in effects:
                match effect:
                    case StateEffect(kind=EffectKind.READ, state=state):
                        has_read_effect = state is trainable_state
                    case ExternalEffect() | StateEffect():
                        pass
                if has_read_effect:
                    break
            if not has_read_effect:
                effects.append(read(trainable_state))
            rewritten_outputs.append((output_name, TemporaryOutput()))
            pending.append((task_name, output_name, output, declaration.predicate))
        expanded.append(
            replace(
                entry,
                declaration=replace(
                    declaration,
                    outputs=tuple(rewritten_outputs),
                    effects=tuple(effects),
                ),
                is_gradient_task=True,
            )
        )

    generated: list[TaskDefinition] = []
    for task_name, output_name, output, predicate in pending:
        state = output.trainable_state()
        generated_name = f"{task_name}__optimize_{_state_id(state)}"
        if generated_name in known_names:
            raise ValueError(f"generated optimizer task {generated_name!r} already exists")
        known_names.add(generated_name)

        def optimizer_executor(state: TrainableState[object]) -> Callable[..., None]:
            def apply_loss(self: _OptimizerOwner, loss: torch.Tensor) -> None:
                self.optimize_state(state, loss)

            return apply_loss

        def conditional_optimizer_executor(
            state: TrainableState[object],
        ) -> Callable[..., None]:
            def apply_loss(
                self: _OptimizerOwner, loss: torch.Tensor | None
            ) -> None:
                if loss is not None:
                    self.optimize_state(state, loss)

            return apply_loss

        conditional = output.conditional

        generated.append(
            TaskDefinition(
                name=generated_name,
                declaration=TaskDecl(
                    inputs=(("loss", ResultInput(task_name, output_name)),),
                    outputs=(),
                    effects=((maybe_transition(state) if conditional else transition(state)),),
                    timing=output.at,
                    implementations=(ExternImplementation(
                        namespace="tllm.workload",
                        name="optimizer_transition",
                        kind=(
                            ImplementationKind.EXTERN
                            if conditional
                            else ImplementationKind.OPTIMIZER_TRANSITION
                        ),
                        version="1" if conditional else "2",
                        input_types=("Tensor",),
                        output_types=("None",),
                        capabilities=(
                            "runtime.async_host_safe",
                            "runtime.side_stream_safe",
                        ),
                    ),),
                    domain=Domain("scalar"),
                    accesses=(),
                    predicate=predicate,
                ),
                executor=(
                    conditional_optimizer_executor(state)
                    if conditional
                    else optimizer_executor(state)
                ),
                generated_after=task_name,
            )
        )
    return (*expanded, *generated)


def _timing_ir(value: Timing) -> TimingIR:
    if isinstance(value, Between):
        return TimingIR(value.start, value.end)
    if isinstance(value, Before):
        return TimingIR("step.begin", value.anchor)
    return TimingIR(value.anchor, "generate.end")


def _input_ir(name: str, value: InputSpec) -> TaskInputIR:
    match value:
        case StateSelectionInput(selected_state=state):
            return TaskInputIR(name, ValueSourceKind.STATE_SELECTION, _state_id(state))
        case _:
            pass
    if isinstance(value, HiddenInput):
        return TaskInputIR(
            name,
            ValueSourceKind.HIDDEN,
            f"hidden:{value.layer}:{value.site}:{value.phase}",
            layer=value.layer,
            layer_kind=value.layer_kind.value,
            site=value.site,
            phase=value.phase,
        )
    if isinstance(value, ResultInput):
        return TaskInputIR(name, ValueSourceKind.RESULT, value.task, output=value.output)
    if isinstance(value, ProvidedInput):
        return TaskInputIR(name, ValueSourceKind.PROVIDED, value.name)
    if isinstance(value, StagedInput):
        return TaskInputIR(name, ValueSourceKind.STAGED, value.port.key)
    if isinstance(value, SamplerRuntimeInput):
        return TaskInputIR(name, value.kind, value.kind.value)
    return TaskInputIR(name, ValueSourceKind.STEP_ID, "step_id")


def _output_ir(name: str, value: OutputSpec) -> TaskOutputIR:
    if isinstance(value, SinkOutput):
        return TaskOutputIR(name, ValueTargetKind.SINK, value.name)
    if isinstance(value, SamplerOutput):
        return TaskOutputIR(name, ValueTargetKind.SAMPLER, value.name)
    if isinstance(value, HostOutputSpec):
        return TaskOutputIR(name, ValueTargetKind.HOST, value.port.key)
    return TaskOutputIR(name, ValueTargetKind.TEMPORARY, f"temporary:{name}")


def _class_members(cls: type[object]) -> tuple[tuple[str, object], ...]:
    members: dict[str, object] = {}
    for base in reversed(cls.__mro__[:-1]):
        members.update(vars(base))
    return tuple(members.items())


def _build_ir(cls: type[object], definition: _Definition) -> WorkloadIR:
    states = definition.states
    if len({_state_id(state) for state in states}) != len(states):
        raise ValueError("workload contains duplicate state names")
    parameter_owners: dict[str, str] = {}
    for state in states:
        for parameter in state.parameters:
            prior = parameter_owners.get(parameter)
            if prior is not None:
                raise ValueError(f"parameter {parameter!r} is owned by both {prior!r} and {_state_id(state)!r}")
            parameter_owners[parameter] = _state_id(state)

    raw_tasks = definition.tasks
    task_names = tuple(entry.name for entry in raw_tasks)
    if len(task_names) != len(set(task_names)):
        raise ValueError("workload contains duplicate task names")
    known_state_ids = {id(state) for state in states}
    known_task_names = set(task_names)
    output_names = {
        entry.name: {output for output, _ in entry.declaration.outputs}
        for entry in raw_tasks
    }
    task_irs: list[TaskIR] = []
    edges: list[WorkloadEdge] = []
    values: list[ValueIR] = []
    consumers: dict[tuple[str, str], list[str]] = {}

    for ordinal, entry in enumerate(raw_tasks):
        name = entry.name
        function = entry.executor
        declaration = entry.declaration
        for effect in declaration.effects:
            match effect:
                case ExternalEffect(resource_name=resource_name):
                    if resource_name in {_state_id(state) for state in states}:
                        raise ValueError(
                            f"task {name!r} extern_io channel {resource_name!r} "
                            "collides with a state name"
                        )
                case StateEffect(state=state):
                    if id(state) not in known_state_ids:
                        raise ValueError(
                            f"task {name!r} references state from another workload"
                        )
        for parameter, input_spec in declaration.inputs:
            if isinstance(input_spec, ResultInput):
                if input_spec.task not in known_task_names:
                    raise ValueError(f"task {name!r} references unknown task {input_spec.task!r}")
                if input_spec.output not in output_names[input_spec.task]:
                    raise ValueError(f"task {name!r} references unknown output {input_spec.output!r}")
                consumers.setdefault((input_spec.task, input_spec.output), []).append(name)
                edges.append(WorkloadEdge(input_spec.task, name, WorkloadEdgeKind.DATA, f"{input_spec.output}->{parameter}"))
        if declaration.implementations:
            implementations = tuple(value.to_ir() for value in declaration.implementations)
        else:
            implementations = (
                ImplementationIR(
                    ImplementationKind.CAPTURED,
                    f"{function.__module__}:{function.__qualname__}@1",
                    capabilities=(
                        ("runtime.async_host_safe", "torch.export")
                        if entry.is_gradient_task
                        else ("torch.export",)
                    ),
                ),
            )
        effect_irs: list[EffectIR] = []
        for effect in declaration.effects:
            match effect:
                case StateEffect(selection=selection) if selection is not None:
                    ports = tuple(
                        parameter
                        for parameter, input_spec in declaration.inputs
                        if input_spec is selection
                    )
                    if len(ports) != 1:
                        raise ValueError(
                            f"task {name!r} request-state effect must bind its exact selection input exactly once"
                        )
                    effect_irs.append(
                        EffectIR(
                            effect.kind,
                            effect.resource_name,
                            selection_port=ports[0],
                        )
                    )
                case StateEffect() | ExternalEffect():
                    effect_irs.append(EffectIR(effect.kind, effect.resource_name))
        task_irs.append(
            TaskIR(
                name=name,
                ordinal=ordinal,
                inputs=tuple(_input_ir(param, value) for param, value in declaration.inputs),
                outputs=tuple(_output_ir(output, value) for output, value in declaration.outputs),
                effects=tuple(effect_irs),
                timing=_timing_ir(declaration.timing),
                implementations=implementations,
                domain=declaration.domain.to_ir(),
                accesses=tuple(access.to_ir() for access in declaration.accesses),
            )
        )

    # State effects are version-order constraints. Reads may commute, while a
    # transition is ordered against every adjacent access to the same state.
    # extern_io channels serialize the same way: the signed graph has no
    # ordering proof for external side effects, so declaration order is kept.
    ordered_kinds = frozenset(
        {EffectKind.TRANSITION, EffectKind.TRANSITION_MAYBE, EffectKind.EXTERN_IO}
    )
    state_effects: dict[str, list[tuple[str, EffectKind]]] = {}
    for task_ir in task_irs:
        for effect in task_ir.effects:
            accesses = state_effects.setdefault(effect.state, [])
            if accesses and (effect.kind in ordered_kinds or accesses[-1][1] in ordered_kinds):
                edges.append(WorkloadEdge(accesses[-1][0], task_ir.name, WorkloadEdgeKind.EFFECT, effect.state))
            accesses.append((task_ir.name, effect.kind))

    for task_ir in task_irs:
        for output in task_ir.outputs:
            values.append(
                ValueIR(
                    value_id=f"{definition.workload_id}:{task_ir.name}:{output.name}",
                    producer=task_ir.name,
                    output=output.name,
                    consumers=tuple(consumers.get((task_ir.name, output.name), ())),
                )
            )

    # Deterministic topological order, retaining class order for ties.
    ordinal_by_name = {task.name: task.ordinal for task in task_irs}
    indegree = {task.name: 0 for task in task_irs}
    successors: dict[str, set[str]] = {task.name: set() for task in task_irs}
    for edge in edges:
        if edge.target not in successors[edge.source]:
            successors[edge.source].add(edge.target)
            indegree[edge.target] += 1
    ready = sorted(
        (name for name, degree in indegree.items() if degree == 0),
        key=ordinal_by_name.__getitem__,
    )
    schedule: list[str] = []
    while ready:
        current_name = ready.pop(0)
        schedule.append(current_name)
        for successor in sorted(
            successors[current_name], key=ordinal_by_name.__getitem__
        ):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
        ready.sort(key=ordinal_by_name.__getitem__)
    if len(schedule) != len(task_irs):
        cycle = ", ".join(name for name, degree in indegree.items() if degree)
        raise ValueError(f"workload task dependency cycle: {cycle}")

    return WorkloadIR(
        workload_id=definition.workload_id,
        states=tuple(state.to_ir() for state in states),
        tasks=tuple(task_irs),
        values=tuple(values),
        edges=tuple(edges),
        schedule=tuple(schedule),
        approximation=definition.approximation,
    )


def _workload_definition_for(cls: type[object]) -> _Definition | None:
    """Find the immutable definition without reflecting task attributes."""

    for candidate in cls.__mro__:
        definition = _WORKLOAD_DEFINITIONS.get(candidate)
        if definition is not None:
            return definition
    return None


def definition_for_workload_class(cls: type[object]) -> _Definition | None:
    return _workload_definition_for(cls)


class _ConfiguredWorkloadInstance(Protocol):
    config: WorkloadConfig


def _disabled_workload_noop(_instance: object) -> None:
    pass


_DISABLED_WORKLOAD_TASK = TaskDefinition(
    name="__disabled__",
    declaration=TaskDecl(
        inputs=(),
        outputs=(),
        effects=(),
        timing=Before("step.end"),
        implementations=(
            ExternImplementation(
                namespace="tllm.workload",
                name="disabled_noop",
                version="1",
                input_types=(),
                output_types=("None",),
            ),
        ),
        domain=Domain("scalar"),
        accesses=(),
    ),
    executor=_disabled_workload_noop,
)


def _definition_for_instance(
    instance: object, definition: _Definition
) -> _Definition:
    match definition:
        case NativeWorkloadDefinition():
            config = cast(_ConfiguredWorkloadInstance, instance).config
        case WorkloadDefinition():
            predicates = tuple(
                entry.declaration.predicate
                for entry in definition.tasks
                if entry.declaration.predicate is not None
            )
            if not predicates:
                return definition
            raise TypeError("config predicates require a native WorkloadConfig")
    enabled = definition.enabled.resolve(config)
    if not enabled:
        return replace(definition, tasks=(_DISABLED_WORKLOAD_TASK,))
    return replace(
        definition,
        tasks=tuple(
            entry
            for entry in definition.tasks
            if (
                entry.declaration.predicate is None
                or entry.declaration.predicate.matches(config)
            )
        ),
    )


def declare_workload(
    instance: object,
    *,
    executor_registry: ExecutorRegistry | None = None,
) -> WorkloadDeclaration:
    if isinstance(instance, type):
        raise TypeError("declare_workload expects a workload instance")
    definition = _workload_definition_for(type(instance))
    if definition is None:
        raise TypeError("instance class is not decorated with @workload")
    specialized_definition = _definition_for_instance(instance, definition)
    ir = _build_ir(type(instance), specialized_definition)
    executors: list[tuple[str, Callable[..., object]]] = []
    implementation_executors: list[tuple[str, str, Callable[..., object]]] = []
    tasks_by_name = {entry.name: entry for entry in specialized_definition.tasks}
    for task in ir.tasks:
        entry = tasks_by_name[task.name]
        default_executor = entry.bind(instance)
        executors.append((task.name, default_executor))
        for implementation in task.implementations:
            executor = default_executor
            if (
                executor_registry is not None
                and implementation.kind is ImplementationKind.EXTERN
            ):
                executor = executor_registry.resolve(implementation.key)
            implementation_executors.append(
                (task.name, implementation.key, executor)
            )
    return WorkloadDeclaration(
        ir=ir,
        instance=instance,
        executors=tuple(executors),
        authority=definition.authority,
        implementation_executors=tuple(implementation_executors),
        approximation_criteria=definition.approximation_criteria,
    )


__all__ = [
    "AdamWSpec",
    "AffineAccess",
    "AffineExpr",
    "After",
    "ApproximationCriteria",
    "Before",
    "Between",
    "ConfigPredicate",
    "Domain",
    "ExecutorRegistry",
    "ExternalEffect",
    "ExternImplementation",
    "Index",
    "RequestKey",
    "State",
    "StateEffect",
    "StateSelectionInput",
    "WorkloadEffect",
    "WorkloadDeclaration",
    "WorkloadAuthority",
    "current",
    "declare_workload",
    "definition_for_workload_class",
    "hidden",
    "maybe_transition",
    "maybe_optimize",
    "next",
    "provided",
    "read",
    "request_key",
    "result",
    "sink",
    "step_id",
    "task",
    "task_declaration",
    "temporary",
    "transition",
    "when",
    "workload",
]
