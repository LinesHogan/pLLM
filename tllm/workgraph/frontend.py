"""Typed, transparent Python front-end contracts for WorkGraph capture."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import StrEnum
from functools import wraps
from types import FunctionType
from typing import Callable, ParamSpec, Protocol, TypeVar, cast, get_type_hints, overload

from tllm.ports.address import PortAddressDesc
from tllm.ports.port_id import PortId
from tllm.workgraph.ir import (
    GraphSafetyKind,
    NodeKind,
    PortRef,
    TimingMobilityKind,
    ValueDesc,
)


_P = ParamSpec("_P")
_R = TypeVar("_R")
_R_co = TypeVar("_R_co", covariant=True)
_T = TypeVar("_T")


def _require_str(value: object, label: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str")
    if nonempty and not value:
        raise TypeError(f"{label} must be non-empty str")
    return value


def _require_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be int")
    return value


def _require_instance(value: object, expected: type[_T], label: str) -> _T:
    if not isinstance(value, expected):
        raise TypeError(f"{label} must be {expected.__name__}")
    return value


def _require_tuple(value: object, expected: type[_T], label: str) -> tuple[_T, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be tuple[{expected.__name__}, ...]")
    items = cast(tuple[object, ...], value)
    if any(not isinstance(item, expected) for item in items):
        raise TypeError(f"{label} must contain only {expected.__name__}")
    return cast(tuple[_T, ...], items)


class FlowCaptureCode(StrEnum):
    """Stable classifications emitted by the flow capture front-end."""

    OPAQUE_PYTHON_REGION = "WG200_OPAQUE_PYTHON_REGION"
    STRICT_EXPORT_FAILURE = "WG201_STRICT_EXPORT_FAILURE"
    DATA_DEPENDENT_CONTROL_FLOW = "WG202_DATA_DEPENDENT_CONTROL_FLOW"
    HOST_IO = "WG203_HOST_IO"
    UNREGISTERED_CLOSURE_STATE = "WG204_UNREGISTERED_CLOSURE_STATE"
    UNKNOWN_SIDE_EFFECT = "WG205_UNKNOWN_SIDE_EFFECT"
    UNBOUNDED_DYNAMIC_SHAPE = "WG206_UNBOUNDED_DYNAMIC_SHAPE"
    UNSUPPORTED_MUTATION = "WG207_UNSUPPORTED_MUTATION"


class CaptureStatus(StrEnum):
    CAPTURED = "captured"
    OPAQUE_BARRIER = "opaque_barrier"


class PortBindingDirection(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


class ShapeGuardKind(StrEnum):
    STATIC = "static"
    RANGE = "range"


class Effect(StrEnum):
    PURE = "pure"
    READ_STATE = "read_state"
    WRITE_STATE = "write_state"
    RNG = "rng"
    HOST_IO = "host_io"
    OPAQUE = "opaque"


class StateScope(StrEnum):
    NONE = "none"
    REQUEST = "request"
    REPLICA = "replica"
    SHARD = "shard"
    GLOBAL = "global"


class Device(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class Capability(StrEnum):
    EAGER_ONLY = "eager_only"
    CAPTURE_SAFE = "capture_safe"
    REPLAY_SAFE = "replay_safe"


@dataclass(frozen=True, slots=True)
class _BeforeNextUse:
    """Freshness singleton requiring completion before the next value use."""


BeforeNextUse = _BeforeNextUse()


@dataclass(frozen=True, slots=True, order=True)
class ArtifactKey:
    """Stable leaf-artifact identity, independent of node and executor keys."""

    namespace: str
    name: str
    version: int = 1

    def __post_init__(self) -> None:
        namespace = _require_str(cast(object, self.namespace), "ArtifactKey.namespace", nonempty=True)
        name = _require_str(cast(object, self.name), "ArtifactKey.name", nonempty=True)
        if "/" in namespace or "\\" in namespace:
            raise ValueError("ArtifactKey.namespace must not contain path separators")
        if "/" in name or "\\" in name:
            raise ValueError("ArtifactKey.name must not contain path separators")
        version = _require_int(cast(object, self.version), "ArtifactKey.version")
        if version < 1:
            raise ValueError("ArtifactKey.version must be >= 1")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)

    @classmethod
    def from_callable(cls, function: Callable[..., object]) -> "ArtifactKey":
        if not isinstance(function, FunctionType):
            raise TypeError("captured callable must be a Python function")
        namespace = function.__module__
        name = function.__qualname__
        if not name:
            raise TypeError("captured callable must have a stable qualified name")
        return cls(namespace=namespace, name=name)

    @property
    def value(self) -> str:
        return f"{self.namespace}:{self.name}@{self.version}"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class SourceSpan:
    file: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int

    def __post_init__(self) -> None:
        file = _require_str(cast(object, self.file), "SourceSpan.file", nonempty=True)
        start_line = _require_int(cast(object, self.start_line), "SourceSpan.start_line")
        start_column = _require_int(cast(object, self.start_column), "SourceSpan.start_column")
        end_line = _require_int(cast(object, self.end_line), "SourceSpan.end_line")
        end_column = _require_int(cast(object, self.end_column), "SourceSpan.end_column")
        if start_line < 1 or end_line < start_line:
            raise ValueError("SourceSpan line range is invalid")
        if start_column < 0 or end_column < 0:
            raise ValueError("SourceSpan columns must be non-negative")
        if start_line == end_line and end_column < start_column:
            raise ValueError("SourceSpan column range is invalid")
        object.__setattr__(self, "file", file)
        object.__setattr__(self, "start_line", start_line)
        object.__setattr__(self, "start_column", start_column)
        object.__setattr__(self, "end_line", end_line)
        object.__setattr__(self, "end_column", end_column)

    @classmethod
    def from_callable(cls, function: Callable[..., object]) -> "SourceSpan":
        file = inspect.getsourcefile(function) or inspect.getfile(function)
        try:
            source_lines, start_line = inspect.getsourcelines(function)
        except (OSError, TypeError):
            return cls(file=file, start_line=1, start_column=0, end_line=1, end_column=0)
        end_line = start_line + max(len(source_lines) - 1, 0)
        end_column = len(source_lines[-1].rstrip("\n")) if source_lines else 0
        return cls(
            file=file,
            start_line=start_line,
            start_column=0,
            end_line=end_line,
            end_column=end_column,
        )


@dataclass(frozen=True, slots=True)
class ShapeGuardExpr:
    value_path: tuple[int, ...]
    dimension: int
    kind: ShapeGuardKind
    lower_bound: int
    upper_bound: int

    def __post_init__(self) -> None:
        value_path = _require_tuple(cast(object, self.value_path), int, "ShapeGuardExpr.value_path")
        if any(isinstance(part, bool) or part < 0 for part in value_path):
            raise ValueError("ShapeGuardExpr.value_path must contain non-negative ints")
        dimension = _require_int(cast(object, self.dimension), "ShapeGuardExpr.dimension")
        if dimension < 0:
            raise ValueError("ShapeGuardExpr.dimension must be non-negative")
        kind = _require_instance(cast(object, self.kind), ShapeGuardKind, "ShapeGuardExpr.kind")
        lower_bound = _require_int(cast(object, self.lower_bound), "ShapeGuardExpr.lower_bound")
        upper_bound = _require_int(cast(object, self.upper_bound), "ShapeGuardExpr.upper_bound")
        if lower_bound < 0 or upper_bound < lower_bound:
            raise ValueError("ShapeGuardExpr bounds are invalid")
        if kind == ShapeGuardKind.STATIC and lower_bound != upper_bound:
            raise ValueError("static ShapeGuardExpr requires equal bounds")
        object.__setattr__(self, "value_path", value_path)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "lower_bound", lower_bound)
        object.__setattr__(self, "upper_bound", upper_bound)


@dataclass(frozen=True, slots=True)
class ShapeGuard:
    expressions: tuple[ShapeGuardExpr, ...] = ()

    def __post_init__(self) -> None:
        expressions = _require_tuple(
            cast(object, self.expressions), ShapeGuardExpr, "ShapeGuard.expressions"
        )
        object.__setattr__(self, "expressions", expressions)


@dataclass(frozen=True, slots=True)
class PortBindingDesc:
    """Explicit mapping between one structured port and one captured pytree leaf."""

    port: PortRef
    direction: PortBindingDirection
    value_path: tuple[int, ...]

    def __post_init__(self) -> None:
        port_value = cast(object, self.port)
        if not isinstance(port_value, (PortAddressDesc, PortId)):
            raise TypeError("PortBindingDesc.port must be PortId or PortAddressDesc")
        direction = _require_instance(
            cast(object, self.direction), PortBindingDirection, "PortBindingDesc.direction"
        )
        value_path = _require_tuple(cast(object, self.value_path), int, "PortBindingDesc.value_path")
        if any(isinstance(part, bool) or part < 0 for part in value_path):
            raise ValueError("PortBindingDesc.value_path must contain non-negative ints")
        object.__setattr__(self, "port", port_value)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "value_path", value_path)


@dataclass(frozen=True, slots=True)
class FlowCaptureDiagnostic:
    code: FlowCaptureCode
    message: str
    source_span: SourceSpan
    detail: str = ""

    def __post_init__(self) -> None:
        code = _require_instance(
            cast(object, self.code), FlowCaptureCode, "FlowCaptureDiagnostic.code"
        )
        message = _require_str(
            cast(object, self.message), "FlowCaptureDiagnostic.message", nonempty=True
        )
        source_span = _require_instance(
            cast(object, self.source_span), SourceSpan, "FlowCaptureDiagnostic.source_span"
        )
        detail = _require_str(cast(object, self.detail), "FlowCaptureDiagnostic.detail")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "source_span", source_span)
        object.__setattr__(self, "detail", detail)


@dataclass(frozen=True, slots=True)
class CapturedFlow:
    """Pure metadata for one control-plane capture specialization."""

    artifact_key: ArtifactKey
    source_span: SourceSpan
    shape_guard: ShapeGuard = ShapeGuard()
    inputs: tuple[PortBindingDesc, ...] = ()
    outputs: tuple[PortBindingDesc, ...] = ()
    values: tuple[ValueDesc, ...] = ()
    effects: tuple[Effect, ...] = (Effect.OPAQUE,)
    diagnostics: tuple[FlowCaptureDiagnostic, ...] = ()
    status: CaptureStatus = CaptureStatus.OPAQUE_BARRIER
    node_kind: NodeKind = NodeKind.BARRIER
    mobility: TimingMobilityKind = TimingMobilityKind.PINNED
    graph_safety: GraphSafetyKind = GraphSafetyKind.EAGER_ONLY

    def __post_init__(self) -> None:
        artifact_key = _require_instance(
            cast(object, self.artifact_key), ArtifactKey, "CapturedFlow.artifact_key"
        )
        source_span = _require_instance(
            cast(object, self.source_span), SourceSpan, "CapturedFlow.source_span"
        )
        shape_guard = _require_instance(
            cast(object, self.shape_guard), ShapeGuard, "CapturedFlow.shape_guard"
        )
        inputs = _require_tuple(cast(object, self.inputs), PortBindingDesc, "CapturedFlow.inputs")
        outputs = _require_tuple(cast(object, self.outputs), PortBindingDesc, "CapturedFlow.outputs")
        values = _require_tuple(cast(object, self.values), ValueDesc, "CapturedFlow.values")
        effects = _require_tuple(cast(object, self.effects), Effect, "CapturedFlow.effects")
        diagnostics = _require_tuple(
            cast(object, self.diagnostics), FlowCaptureDiagnostic, "CapturedFlow.diagnostics"
        )
        status = _require_instance(cast(object, self.status), CaptureStatus, "CapturedFlow.status")
        node_kind = _require_instance(cast(object, self.node_kind), NodeKind, "CapturedFlow.node_kind")
        mobility = _require_instance(
            cast(object, self.mobility), TimingMobilityKind, "CapturedFlow.mobility"
        )
        graph_safety = _require_instance(
            cast(object, self.graph_safety), GraphSafetyKind, "CapturedFlow.graph_safety"
        )
        if status == CaptureStatus.OPAQUE_BARRIER:
            if node_kind != NodeKind.BARRIER:
                raise ValueError("opaque capture must lower to BarrierNode")
            if mobility != TimingMobilityKind.PINNED:
                raise ValueError("opaque capture must be pinned")
            if graph_safety != GraphSafetyKind.EAGER_ONLY:
                raise ValueError("opaque capture must be eager-only")
            if Effect.OPAQUE not in effects:
                raise ValueError("opaque capture must retain an opaque effect")
            if not diagnostics:
                raise ValueError("opaque capture must explain why it is a barrier")
        binding_keys = tuple(
            (binding.direction, binding.value_path) for binding in (*inputs, *outputs)
        )
        if len(set(binding_keys)) != len(binding_keys):
            raise ValueError("CapturedFlow port bindings must map each value path exactly once")
        object.__setattr__(self, "artifact_key", artifact_key)
        object.__setattr__(self, "source_span", source_span)
        object.__setattr__(self, "shape_guard", shape_guard)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "node_kind", node_kind)
        object.__setattr__(self, "mobility", mobility)
        object.__setattr__(self, "graph_safety", graph_safety)


@dataclass(frozen=True, slots=True)
class TypeDesc:
    module: str
    qualname: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "module",
            _require_str(cast(object, self.module), "TypeDesc.module", nonempty=True),
        )
        object.__setattr__(
            self,
            "qualname",
            _require_str(cast(object, self.qualname), "TypeDesc.qualname", nonempty=True),
        )

    @classmethod
    def from_annotation(cls, annotation: object) -> "TypeDesc":
        if not isinstance(annotation, type):
            raise TypeError("extern annotations must name concrete types")
        return cls(module=annotation.__module__, qualname=annotation.__qualname__)


@dataclass(frozen=True, slots=True)
class ExternInputDesc:
    position: int
    name: str
    value_type: TypeDesc

    def __post_init__(self) -> None:
        position = _require_int(cast(object, self.position), "ExternInputDesc.position")
        if position < 0:
            raise ValueError("ExternInputDesc.position must be non-negative")
        object.__setattr__(self, "position", position)
        object.__setattr__(
            self,
            "name",
            _require_str(cast(object, self.name), "ExternInputDesc.name", nonempty=True),
        )
        object.__setattr__(
            self,
            "value_type",
            _require_instance(cast(object, self.value_type), TypeDesc, "ExternInputDesc.value_type"),
        )


@dataclass(frozen=True, slots=True)
class ExternContract:
    inputs: tuple[ExternInputDesc, ...]
    output: TypeDesc
    effect: Effect
    state: StateScope
    device: Device
    capability: Capability
    bindings: tuple[PortBindingDesc, ...] = ()

    def __post_init__(self) -> None:
        inputs = _require_tuple(cast(object, self.inputs), ExternInputDesc, "ExternContract.inputs")
        output = _require_instance(cast(object, self.output), TypeDesc, "ExternContract.output")
        effect = _require_instance(cast(object, self.effect), Effect, "ExternContract.effect")
        state = _require_instance(cast(object, self.state), StateScope, "ExternContract.state")
        device = _require_instance(cast(object, self.device), Device, "ExternContract.device")
        capability = _require_instance(
            cast(object, self.capability), Capability, "ExternContract.capability"
        )
        bindings = _require_tuple(
            cast(object, self.bindings), PortBindingDesc, "ExternContract.bindings"
        )
        if effect == Effect.PURE and state != StateScope.NONE:
            raise ValueError("pure extern cannot declare mutable state scope")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "effect", effect)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "capability", capability)
        object.__setattr__(self, "bindings", bindings)


class FlowCallable(Protocol[_P, _R_co]):
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R_co: ...

    def capture(self, *args: _P.args, **kwargs: _P.kwargs) -> CapturedFlow: ...


def _capture_placeholder(
    function: Callable[..., object],
    bindings: tuple[PortBindingDesc, ...],
) -> CapturedFlow:
    span = SourceSpan.from_callable(function)
    diagnostic = FlowCaptureDiagnostic(
        code=FlowCaptureCode.OPAQUE_PYTHON_REGION,
        message="flow capture has not produced a proven strict export graph",
        source_span=span,
        detail="the original Python function remains authoritative",
    )
    return CapturedFlow(
        artifact_key=ArtifactKey.from_callable(function),
        source_span=span,
        inputs=tuple(binding for binding in bindings if binding.direction == PortBindingDirection.INPUT),
        outputs=tuple(binding for binding in bindings if binding.direction == PortBindingDirection.OUTPUT),
        diagnostics=(diagnostic,),
    )


@overload
def flow(function: Callable[_P, _R], /) -> FlowCallable[_P, _R]: ...


@overload
def flow(
    function: None = None,
    /,
    *,
    freshness: _BeforeNextUse = BeforeNextUse,
    bindings: tuple[PortBindingDesc, ...] = (),
) -> Callable[[Callable[_P, _R]], FlowCallable[_P, _R]]: ...


def flow(
    function: Callable[_P, _R] | None = None,
    /,
    *,
    freshness: _BeforeNextUse = BeforeNextUse,
    bindings: tuple[PortBindingDesc, ...] = (),
) -> FlowCallable[_P, _R] | Callable[[Callable[_P, _R]], FlowCallable[_P, _R]]:
    """Wrap a function transparently; only ``.capture`` enters control-plane capture."""

    if freshness is not BeforeNextUse:
        raise TypeError("flow.freshness must be the BeforeNextUse singleton")
    bindings = _require_tuple(cast(object, bindings), PortBindingDesc, "flow.bindings")

    def decorate(target: Callable[_P, _R]) -> FlowCallable[_P, _R]:
        if not isinstance(target, FunctionType):
            raise TypeError("flow currently requires a Python function")

        @wraps(target)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return target(*args, **kwargs)

        def capture(*_args: _P.args, **_kwargs: _P.kwargs) -> CapturedFlow:
            inspect.signature(target).bind(*_args, **_kwargs)
            return _capture_placeholder(target, bindings)

        setattr(wrapped, "capture", capture)
        setattr(wrapped, "flow_freshness", freshness)
        setattr(wrapped, "flow_bindings", bindings)
        return cast(FlowCallable[_P, _R], wrapped)

    if function is None:
        return decorate
    return decorate(function)


def _resolved_signature(function: Callable[..., object]) -> inspect.Signature:
    signature = inspect.signature(function)
    try:
        hints = get_type_hints(function)
    except (NameError, TypeError) as exc:
        raise TypeError("extern annotations must resolve to concrete types") from exc
    parameters: list[inspect.Parameter] = []
    for parameter in signature.parameters.values():
        if parameter.annotation is inspect.Parameter.empty or parameter.name not in hints:
            raise TypeError(f"extern input {parameter.name!r} requires a concrete annotation")
        parameters.append(parameter.replace(annotation=hints[parameter.name]))
    if signature.return_annotation is inspect.Signature.empty or "return" not in hints:
        raise TypeError("extern output requires a concrete return annotation")
    return signature.replace(parameters=parameters, return_annotation=hints["return"])


def extern(
    *,
    effect: Effect,
    state: StateScope,
    device: Device,
    capability: Capability,
    bindings: tuple[PortBindingDesc, ...] = (),
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Declare a typed opaque implementation while retaining normal Python calls."""

    effect = _require_instance(cast(object, effect), Effect, "extern.effect")
    state = _require_instance(cast(object, state), StateScope, "extern.state")
    device = _require_instance(cast(object, device), Device, "extern.device")
    capability = _require_instance(cast(object, capability), Capability, "extern.capability")
    bindings = _require_tuple(cast(object, bindings), PortBindingDesc, "extern.bindings")

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        if not isinstance(function, FunctionType):
            raise TypeError("extern currently requires a Python function")
        signature = _resolved_signature(function)
        inputs = tuple(
            ExternInputDesc(
                position=position,
                name=parameter.name,
                value_type=TypeDesc.from_annotation(parameter.annotation),
            )
            for position, parameter in enumerate(signature.parameters.values())
        )
        contract = ExternContract(
            inputs=inputs,
            output=TypeDesc.from_annotation(signature.return_annotation),
            effect=effect,
            state=state,
            device=device,
            capability=capability,
            bindings=bindings,
        )

        @wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return function(*args, **kwargs)

        setattr(wrapped, "__signature__", signature)
        setattr(wrapped, "extern_contract", contract)
        return wrapped

    return decorate


__all__ = [
    "ArtifactKey",
    "BeforeNextUse",
    "Capability",
    "CaptureStatus",
    "CapturedFlow",
    "Device",
    "Effect",
    "ExternContract",
    "ExternInputDesc",
    "FlowCallable",
    "FlowCaptureCode",
    "FlowCaptureDiagnostic",
    "PortBindingDesc",
    "PortBindingDirection",
    "ShapeGuard",
    "ShapeGuardExpr",
    "ShapeGuardKind",
    "SourceSpan",
    "StateScope",
    "TypeDesc",
    "extern",
    "flow",
]
