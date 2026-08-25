"""Single-decorator frontend for auxiliary workloads."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from types import FunctionType
from typing import Any, ParamSpec, TypeVar, cast, get_type_hints, overload

from tllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from tllm.workload.ir import (
    AffineAccessIR,
    AffineExprIR,
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


_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T")
_REQUEST_KEY_FIELDS = frozenset({"request_id", "prompt_idx", "sample_idx"})


def _name(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str")
    if not value:
        raise ValueError(f"{label} must be non-empty")
    return value


@dataclass(frozen=True, slots=True)
class RequestKey:
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("request_key requires at least one field")
        normalized = tuple(_name(field, "request_key field") for field in self.fields)
        unknown = tuple(field for field in normalized if field not in _REQUEST_KEY_FIELDS)
        if unknown:
            raise ValueError(f"unsupported request_key fields: {', '.join(unknown)}")
        if len(normalized) != len(set(normalized)):
            raise ValueError("request_key fields must be unique")
        object.__setattr__(self, "fields", normalized)


def request_key(*fields: str) -> RequestKey:
    return RequestKey(tuple(fields))


class AdamWSpec(OptimizerSpec):
    def __init__(
        self,
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr <= 0:
            raise ValueError("AdamWSpec.lr must be positive")
        super().__init__(
            kind="adamw",
            options=(
                ("betas", repr(tuple(float(value) for value in betas))),
                ("eps", float(eps).hex()),
                ("lr", float(lr).hex()),
                ("weight_decay", float(weight_decay).hex()),
            ),
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
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0:
            raise ValueError("AdamSpec.lr must be positive")
        super().__init__(
            kind="adam",
            options=(
                ("betas", repr(tuple(float(value) for value in betas))),
                ("eps", float(eps).hex()),
                ("lr", float(lr).hex()),
            ),
        )


class State:
    """A named state family declared on a workload class."""

    def __init__(
        self,
        *,
        scope: StateScope,
        key: RequestKey | None = None,
        parameters: str | tuple[str, ...],
        optimizer: OptimizerSpec | None = None,
        transition: str = "optimizer",
    ) -> None:
        self.scope = StateScope(scope)
        if isinstance(parameters, str):
            parameters = (parameters,)
        if not isinstance(parameters, tuple) or not parameters or any(
            not isinstance(parameter, str) or not parameter for parameter in parameters
        ):
            raise TypeError("State.parameters must be a non-empty str or tuple[str, ...]")
        if self.scope is StateScope.REQUEST and key is None:
            raise ValueError("REQUEST State requires key=request_key(...)")
        if self.scope is not StateScope.REQUEST and key is not None:
            raise ValueError("only REQUEST State may declare a request_key")
        if key is not None and not isinstance(key, RequestKey):
            raise TypeError("State.key must be RequestKey or None")
        if optimizer is not None and not isinstance(optimizer, OptimizerSpec):
            raise TypeError("State.optimizer must be an OptimizerSpec")
        self.key = key
        self.parameters = parameters
        self.optimizer = optimizer
        self.transition_kind = _name(transition, "State.transition")
        self._name = ""
        self._owner: type[object] | None = None

    def __set_name__(self, owner: type[object], name: str) -> None:
        self._owner = owner
        self._name = _name(name, "State name")

    def __get__(self, instance: object, owner: type[object] | None = None) -> "State":
        return self

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
            transition=self.transition_kind,
        )


@dataclass(frozen=True, slots=True)
class StateEffect:
    kind: EffectKind
    state: State


def read(state: State) -> StateEffect:
    return _effect(EffectKind.READ, state)


def current(state: State) -> StateEffect:
    return _effect(EffectKind.CURRENT, state)


def transition(state: State) -> StateEffect:
    if not isinstance(state, State):
        raise TypeError("transition expects a State")
    if state.optimizer is None and state.transition_kind == "optimizer":
        raise ValueError("transition requires a State optimizer or explicit version conversion")
    return StateEffect(EffectKind.TRANSITION, state)


def next(state: State) -> StateEffect:
    return _effect(EffectKind.NEXT, state)


def _effect(kind: EffectKind, state: State) -> StateEffect:
    if not isinstance(state, State):
        raise TypeError(f"{kind.value} expects a State")
    return StateEffect(kind, state)


@dataclass(frozen=True, slots=True)
class HiddenInput:
    layer: str
    layer_kind: HiddenLayerKind
    site: HiddenSite
    phase: HiddenPhase

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
class SinkOutput:
    name: str


@dataclass(frozen=True, slots=True)
class TemporaryOutput:
    storage: str = "temporary"


InputSpec = HiddenInput | ResultInput | ProvidedInput | StepIdInput
OutputSpec = SinkOutput | TemporaryOutput


def hidden(
    layer: int | str | object = "*",
    site: HiddenSite | str = HiddenSite.POST,
    phase: HiddenPhase | str = HiddenPhase.DECODE,
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
    )


def result(task_name: str, output_name: str) -> ResultInput:
    return ResultInput(_name(task_name, "result.task"), _name(output_name, "result.output"))


def provided(name: str) -> ProvidedInput:
    return ProvidedInput(_name(name, "provided.name"))


def step_id() -> StepIdInput:
    return StepIdInput()


def sink(name: str) -> SinkOutput:
    return SinkOutput(_name(name, "sink.name"))


def temporary() -> TemporaryOutput:
    return TemporaryOutput()


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
        if not isinstance(factor, int) or isinstance(factor, bool):
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
    if isinstance(value, int) and not isinstance(value, bool):
        return AffineExpr((), value)
    raise TypeError("expected an affine Index, AffineExpr, or int")


@dataclass(frozen=True, slots=True)
class Domain:
    name: str
    indices: tuple[Index, ...] = ()
    bounds: tuple[tuple[str, int | str, int | str], ...] = ()
    parameters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "Domain.name"))
        if not isinstance(self.indices, tuple) or any(not isinstance(index, Index) for index in self.indices):
            raise TypeError("Domain.indices must be tuple[Index, ...]")
        names = tuple(index.name for index in self.indices)
        if len(names) != len(set(names)):
            raise ValueError("Domain index names must be unique")
        if not isinstance(self.parameters, tuple) or any(not isinstance(value, str) or not value for value in self.parameters):
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

    def __post_init__(self) -> None:
        for field_name in ("namespace", "name", "version"):
            object.__setattr__(self, field_name, _name(getattr(self, field_name), f"ExternImplementation.{field_name}"))
        for field_name in ("input_types", "output_types", "capabilities"):
            value = getattr(self, field_name)
            if not isinstance(value, tuple) or any(not isinstance(item, str) or not item for item in value):
                raise TypeError(f"ExternImplementation.{field_name} must be tuple[str, ...]")

    @property
    def key(self) -> str:
        return f"{self.namespace}:{self.name}@{self.version}"

    def to_ir(self) -> ImplementationIR:
        return ImplementationIR(
            kind=ImplementationKind.EXTERN,
            key=self.key,
            input_types=self.input_types,
            output_types=self.output_types,
            capabilities=self.capabilities,
        )


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
    effects: tuple[StateEffect, ...]
    timing: Timing
    implementations: tuple[ExternImplementation, ...]
    domain: Domain
    accesses: tuple[AffineAccess, ...]


def _mapping(value: Mapping[str, _T] | None, expected: tuple[type, ...], label: str) -> tuple[tuple[str, _T], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    result: list[tuple[str, _T]] = []
    for key, item in value.items():
        if not isinstance(item, expected):
            raise TypeError(f"{label}[{key!r}] has unsupported descriptor {type(item).__name__}")
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
    effects: tuple[StateEffect, ...] = (),
    timing: Timing = Before("step.end"),
    implementation: ExternImplementation | tuple[ExternImplementation, ...] | None = None,
    domain: Domain = Domain("scalar"),
    accesses: tuple[AffineAccess, ...] = (),
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...


def task(
    function: Callable[_P, _R] | None = None,
    /,
    *,
    inputs: Mapping[str, InputSpec] | None = None,
    outputs: Mapping[str, OutputSpec] | None = None,
    effects: tuple[StateEffect, ...] = (),
    timing: Timing = Before("step.end"),
    implementation: ExternImplementation | tuple[ExternImplementation, ...] | None = None,
    domain: Domain = Domain("scalar"),
    accesses: tuple[AffineAccess, ...] = (),
) -> Callable[_P, _R] | Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    task_inputs = _mapping(
        inputs,
        (HiddenInput, ResultInput, ProvidedInput, StepIdInput),
        "task.inputs",
    )
    task_outputs = _mapping(outputs, (SinkOutput, TemporaryOutput), "task.outputs")
    if not isinstance(effects, tuple) or any(not isinstance(effect, StateEffect) for effect in effects):
        raise TypeError("task.effects must be tuple[StateEffect, ...]")
    if not isinstance(timing, (Before, After, Between)):
        raise TypeError("task.timing must be Before, After, or Between")
    if implementation is None:
        implementations: tuple[ExternImplementation, ...] = ()
    elif isinstance(implementation, ExternImplementation):
        implementations = (implementation,)
    elif isinstance(implementation, tuple) and all(isinstance(item, ExternImplementation) for item in implementation):
        implementations = implementation
    else:
        raise TypeError("task.implementation must be ExternImplementation or tuple")
    if not isinstance(domain, Domain):
        raise TypeError("task.domain must be Domain")
    if not isinstance(accesses, tuple) or any(not isinstance(access, AffineAccess) for access in accesses):
        raise TypeError("task.accesses must be tuple[AffineAccess, ...]")

    def decorate(target: Callable[_P, _R]) -> Callable[_P, _R]:
        if not isinstance(target, FunctionType):
            raise TypeError("task requires a Python function")
        parameters = set(inspect.signature(target).parameters)
        parameters.discard("self")
        parameters.discard("cls")
        unknown = tuple(name for name, _ in task_inputs if name not in parameters)
        if unknown:
            raise ValueError(f"task inputs are not function parameters: {', '.join(unknown)}")
        if implementations:
            try:
                hints = get_type_hints(target)
            except (NameError, TypeError) as exc:
                raise TypeError("extern task annotations must resolve") from exc
            missing = tuple(name for name in parameters if name not in hints)
            if missing or "return" not in hints:
                raise TypeError("extern task inputs and output require concrete annotations")

        @wraps(target)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return target(*args, **kwargs)

        setattr(
            wrapped,
            "__tllm_task__",
            TaskDecl(task_inputs, task_outputs, effects, timing, implementations, domain, accesses),
        )
        return wrapped

    return decorate if function is None else decorate(function)


class WorkloadAuthority(str, Enum):
    """The execution runtime explicitly authorized for a declaration."""

    NATIVE_REGION = "native_region"
    LEGACY_ADAPTER = "legacy_adapter"


@dataclass(frozen=True, slots=True)
class WorkloadDefinition:
    workload_id: str
    approximation: ApproximationPolicy
    authority: WorkloadAuthority = WorkloadAuthority.NATIVE_REGION


@dataclass(frozen=True, slots=True)
class WorkloadDeclaration:
    ir: WorkloadIR
    instance: object
    executors: tuple[tuple[str, Callable[..., object]], ...]
    authority: WorkloadAuthority = WorkloadAuthority.NATIVE_REGION
    implementation_executors: tuple[
        tuple[str, str, Callable[..., object]], ...
    ] = ()

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
) -> type[_T] | Callable[[type[_T]], type[_T]]:
    def decorate(target: type[_T]) -> type[_T]:
        if not isinstance(target, type):
            raise TypeError("workload must decorate a class")
        workload_id = id or f"{target.__module__}.{target.__qualname__}"
        definition = WorkloadDefinition(
            _name(workload_id, "workload.id"),
            ApproximationPolicy(approximation),
            WorkloadAuthority(authority),
        )
        setattr(target, "__tllm_workload__", definition)
        # Validate at class definition time without constructing an instance.
        _build_ir(target, definition)
        return target

    return decorate if cls is None else decorate(cls)


def _timing_ir(value: Timing) -> TimingIR:
    if isinstance(value, Between):
        return TimingIR(value.start, value.end)
    if isinstance(value, Before):
        return TimingIR("step.begin", value.anchor)
    return TimingIR(value.anchor, "generate.end")


def _input_ir(name: str, value: InputSpec) -> TaskInputIR:
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
    return TaskInputIR(name, ValueSourceKind.STEP_ID, "step_id")


def _output_ir(name: str, value: OutputSpec) -> TaskOutputIR:
    if isinstance(value, SinkOutput):
        return TaskOutputIR(name, ValueTargetKind.SINK, value.name)
    return TaskOutputIR(name, ValueTargetKind.TEMPORARY, f"temporary:{name}")


def _class_members(cls: type[object]) -> tuple[tuple[str, object], ...]:
    members: dict[str, object] = {}
    for base in reversed(cls.__mro__[:-1]):
        members.update(vars(base))
    return tuple(members.items())


def _build_ir(cls: type[object], definition: WorkloadDefinition) -> WorkloadIR:
    members = _class_members(cls)
    states = tuple(value for _, value in members if isinstance(value, State))
    if len({state.name for state in states}) != len(states):
        raise ValueError("workload contains duplicate state names")
    parameter_owners: dict[str, str] = {}
    for state in states:
        for parameter in state.parameters:
            prior = parameter_owners.get(parameter)
            if prior is not None:
                raise ValueError(f"parameter {parameter!r} is owned by both {prior!r} and {state.name!r}")
            parameter_owners[parameter] = state.name

    raw_tasks: list[tuple[str, Callable[..., object], TaskDecl]] = []
    for name, member in members:
        try:
            declaration = member.__tllm_task__  # type: ignore[attr-defined]
        except AttributeError:
            declaration = None
        if isinstance(declaration, TaskDecl):
            raw_tasks.append((name, cast(Callable[..., object], member), declaration))
    known_states = set(states)
    known_task_names = {name for name, _, _ in raw_tasks}
    output_names = {name: {output for output, _ in decl.outputs} for name, _, decl in raw_tasks}
    task_irs: list[TaskIR] = []
    edges: list[WorkloadEdge] = []
    values: list[ValueIR] = []
    consumers: dict[tuple[str, str], list[str]] = {}

    for ordinal, (name, function, declaration) in enumerate(raw_tasks):
        for effect in declaration.effects:
            if effect.state not in known_states:
                raise ValueError(f"task {name!r} references state from another workload")
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
                    capabilities=("torch.export",),
                ),
            )
        task_irs.append(
            TaskIR(
                name=name,
                ordinal=ordinal,
                inputs=tuple(_input_ir(param, value) for param, value in declaration.inputs),
                outputs=tuple(_output_ir(output, value) for output, value in declaration.outputs),
                effects=tuple(EffectIR(effect.kind, effect.state.name) for effect in declaration.effects),
                timing=_timing_ir(declaration.timing),
                implementations=implementations,
                domain=declaration.domain.to_ir(),
                accesses=tuple(access.to_ir() for access in declaration.accesses),
            )
        )

    # State effects are version-order constraints. Reads may commute, while a
    # transition is ordered against every adjacent access to the same state.
    state_effects: dict[str, list[tuple[str, EffectKind]]] = {}
    for task_ir in task_irs:
        for effect in task_ir.effects:
            accesses = state_effects.setdefault(effect.state, [])
            if accesses and (effect.kind is EffectKind.TRANSITION or accesses[-1][1] is EffectKind.TRANSITION):
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


def declare_workload(
    instance: object,
    *,
    executor_registry: ExecutorRegistry | None = None,
) -> WorkloadDeclaration:
    if isinstance(instance, type):
        raise TypeError("declare_workload expects a workload instance")
    if executor_registry is not None and not isinstance(
        executor_registry, ExecutorRegistry
    ):
        raise TypeError("executor_registry must be ExecutorRegistry or None")
    try:
        definition = type(instance).__tllm_workload__  # type: ignore[attr-defined]
    except AttributeError:
        definition = None
    if not isinstance(definition, WorkloadDefinition):
        raise TypeError("instance class is not decorated with @workload")
    ir = _build_ir(type(instance), definition)
    executors: list[tuple[str, Callable[..., object]]] = []
    implementation_executors: list[tuple[str, str, Callable[..., object]]] = []
    for task in ir.tasks:
        default_executor = getattr(instance, task.name)
        if not callable(default_executor):
            raise TypeError(f"workload task {task.name!r} is not callable")
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
    )


__all__ = [
    "AdamWSpec",
    "AffineAccess",
    "AffineExpr",
    "After",
    "Before",
    "Between",
    "Domain",
    "ExecutorRegistry",
    "ExternImplementation",
    "Index",
    "RequestKey",
    "State",
    "StateEffect",
    "WorkloadDeclaration",
    "WorkloadAuthority",
    "current",
    "declare_workload",
    "hidden",
    "next",
    "provided",
    "read",
    "request_key",
    "result",
    "sink",
    "step_id",
    "task",
    "temporary",
    "transition",
    "workload",
]
