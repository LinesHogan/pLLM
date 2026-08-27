"""Typed, transparent Python front-end contracts for WorkGraph capture."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import wraps
from types import FunctionType
from typing import Callable, ParamSpec, Protocol, TypeVar, cast, get_type_hints, overload

from tllm.contracts.physical_plan import (
    ArtifactKey,
    Device,
    TensorDTypeDesc,
    TensorDeviceDesc,
)
from tllm.ports.address import PortAddressDesc
from tllm.ports.base import TimingDeadline
from tllm.ports.port_id import PortId
from tllm.workgraph.ir import (
    GraphSafetyKind,
    NodeKind,
    PortRef,
    TimingMobilityKind,
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


class TensorLayout(StrEnum):
    STRIDED = "strided"
    SPARSE_COO = "sparse_coo"
    SPARSE_CSR = "sparse_csr"
    SPARSE_CSC = "sparse_csc"
    SPARSE_BSR = "sparse_bsr"
    SPARSE_BSC = "sparse_bsc"
    MKLDNN = "mkldnn"


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


class Capability(StrEnum):
    EAGER_ONLY = "eager_only"
    CAPTURE_SAFE = "capture_safe"
    REPLAY_SAFE = "replay_safe"


class OperatorKind(StrEnum):
    ATEN = "aten"
    CUSTOM = "custom"


class OperatorAttributeKind(StrEnum):
    NONE = "none"
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    TUPLE = "tuple"
    LIST = "list"
    MAPPING = "mapping"
    SLICE = "slice"
    ELLIPSIS = "ellipsis"
    DTYPE = "dtype"
    DEVICE = "device"
    LAYOUT = "layout"
    MEMORY_FORMAT = "memory_format"


@dataclass(frozen=True, slots=True)
class _BeforeNextUse:
    """Freshness singleton requiring completion before the next value use."""


BeforeNextUse = _BeforeNextUse()


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
class DynamicDim:
    """One typed, positional input dimension admitted by strict export."""

    value_path: tuple[int, ...]
    dimension: int
    lower_bound: int = 1
    upper_bound: int | None = None

    def __post_init__(self) -> None:
        value_path = _require_tuple(cast(object, self.value_path), int, "DynamicDim.value_path")
        if len(value_path) != 1 or isinstance(value_path[0], bool) or value_path[0] < 0:
            raise ValueError("DynamicDim currently requires one positional input index")
        dimension = _require_int(cast(object, self.dimension), "DynamicDim.dimension")
        lower_bound = _require_int(cast(object, self.lower_bound), "DynamicDim.lower_bound")
        upper_value = cast(object, self.upper_bound)
        upper_bound: int | None
        if upper_value is None:
            upper_bound = None
        else:
            upper_bound = _require_int(upper_value, "DynamicDim.upper_bound")
        if dimension < 0 or lower_bound < 0:
            raise ValueError("DynamicDim dimension and lower bound must be non-negative")
        if upper_bound is not None and upper_bound < lower_bound:
            raise ValueError("DynamicDim upper bound must be >= lower bound")
        object.__setattr__(self, "value_path", value_path)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "lower_bound", lower_bound)
        object.__setattr__(self, "upper_bound", upper_bound)


@dataclass(frozen=True, slots=True)
class UseDefDesc:
    producer: int
    inputs: tuple[int, ...]
    users: tuple[int, ...]

    def __post_init__(self) -> None:
        producer = _require_int(cast(object, self.producer), "UseDefDesc.producer")
        inputs = _require_tuple(cast(object, self.inputs), int, "UseDefDesc.inputs")
        users = _require_tuple(cast(object, self.users), int, "UseDefDesc.users")
        if producer < 0 or any(value < 0 for value in (*inputs, *users)):
            raise ValueError("UseDefDesc ordinals must be non-negative")
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "users", users)


@dataclass(frozen=True, slots=True)
class CapturedValueDesc:
    value_id: str
    version: int
    shape: tuple[str, ...]
    dtype: TensorDTypeDesc
    device: TensorDeviceDesc
    layout: TensorLayout
    stride: tuple[str, ...]
    use_def: UseDefDesc
    alias_set: str
    effect: Effect

    def __post_init__(self) -> None:
        value_id = _require_str(
            cast(object, self.value_id), "CapturedValueDesc.value_id", nonempty=True
        )
        version = _require_int(cast(object, self.version), "CapturedValueDesc.version")
        shape = _require_tuple(cast(object, self.shape), str, "CapturedValueDesc.shape")
        dtype = _require_instance(
            cast(object, self.dtype), TensorDTypeDesc, "CapturedValueDesc.dtype"
        )
        device = _require_instance(
            cast(object, self.device), TensorDeviceDesc, "CapturedValueDesc.device"
        )
        layout = _require_instance(
            cast(object, self.layout), TensorLayout, "CapturedValueDesc.layout"
        )
        stride = _require_tuple(cast(object, self.stride), str, "CapturedValueDesc.stride")
        use_def = _require_instance(
            cast(object, self.use_def), UseDefDesc, "CapturedValueDesc.use_def"
        )
        alias_set = _require_str(
            cast(object, self.alias_set), "CapturedValueDesc.alias_set", nonempty=True
        )
        effect = _require_instance(cast(object, self.effect), Effect, "CapturedValueDesc.effect")
        if version < 1:
            raise ValueError("CapturedValueDesc requires version >= 1")
        object.__setattr__(self, "value_id", value_id)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "stride", stride)
        object.__setattr__(self, "use_def", use_def)
        object.__setattr__(self, "alias_set", alias_set)
        object.__setattr__(self, "effect", effect)


@dataclass(frozen=True, slots=True)
class OperatorAttributeDesc:
    path: tuple[str, ...]
    kind: OperatorAttributeKind
    value: str

    def __post_init__(self) -> None:
        path = _require_tuple(cast(object, self.path), str, "OperatorAttributeDesc.path")
        if not path or any(not segment for segment in path):
            raise ValueError("OperatorAttributeDesc.path must contain non-empty segments")
        kind = _require_instance(
            cast(object, self.kind), OperatorAttributeKind, "OperatorAttributeDesc.kind"
        )
        value = _require_str(cast(object, self.value), "OperatorAttributeDesc.value")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class LeafOperatorDesc:
    ordinal: int
    namespace: str
    name: str
    overload: str
    inputs: tuple[int, ...]
    outputs: tuple[str, ...]
    module_path: tuple[str, ...] = ()
    operator_kind: OperatorKind = OperatorKind.ATEN
    effects: tuple[Effect, ...] = (Effect.PURE,)
    capabilities: tuple[Capability, ...] = (Capability.CAPTURE_SAFE,)
    attributes: tuple[OperatorAttributeDesc, ...] = ()

    def __post_init__(self) -> None:
        ordinal = _require_int(cast(object, self.ordinal), "LeafOperatorDesc.ordinal")
        if ordinal < 0:
            raise ValueError("LeafOperatorDesc.ordinal must be non-negative")
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(
            self,
            "namespace",
            _require_str(cast(object, self.namespace), "LeafOperatorDesc.namespace", nonempty=True),
        )
        object.__setattr__(
            self,
            "name",
            _require_str(cast(object, self.name), "LeafOperatorDesc.name", nonempty=True),
        )
        object.__setattr__(
            self,
            "overload",
            _require_str(cast(object, self.overload), "LeafOperatorDesc.overload"),
        )
        object.__setattr__(
            self,
            "inputs",
            _require_tuple(cast(object, self.inputs), int, "LeafOperatorDesc.inputs"),
        )
        object.__setattr__(
            self,
            "outputs",
            _require_tuple(cast(object, self.outputs), str, "LeafOperatorDesc.outputs"),
        )
        module_path = _require_tuple(
            cast(object, self.module_path), str, "LeafOperatorDesc.module_path"
        )
        if any(not segment or "/" in segment or "\\" in segment for segment in module_path):
            raise ValueError("LeafOperatorDesc.module_path must contain stable path segments")
        operator_kind = _require_instance(
            cast(object, self.operator_kind), OperatorKind, "LeafOperatorDesc.operator_kind"
        )
        effects = _require_tuple(cast(object, self.effects), Effect, "LeafOperatorDesc.effects")
        capabilities = _require_tuple(
            cast(object, self.capabilities), Capability, "LeafOperatorDesc.capabilities"
        )
        attributes = _require_tuple(
            cast(object, self.attributes), OperatorAttributeDesc, "LeafOperatorDesc.attributes"
        )
        if not effects or not capabilities:
            raise ValueError("LeafOperatorDesc requires typed effect and capability facts")
        if operator_kind == OperatorKind.ATEN and self.namespace != "aten":
            raise ValueError("ATen LeafOperatorDesc must use the 'aten' namespace")
        if operator_kind == OperatorKind.CUSTOM and self.namespace == "aten":
            raise ValueError("custom LeafOperatorDesc must not use the 'aten' namespace")
        object.__setattr__(self, "module_path", module_path)
        object.__setattr__(self, "operator_kind", operator_kind)
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "capabilities", capabilities)
        if len({attribute.path for attribute in attributes}) != len(attributes):
            raise ValueError("LeafOperatorDesc attributes must use unique paths")
        object.__setattr__(self, "attributes", attributes)


@dataclass(frozen=True, slots=True)
class LeafArtifactDesc:
    artifact_key: ArtifactKey
    operators: tuple[LeafOperatorDesc, ...]
    input_values: tuple[str, ...]
    output_values: tuple[str, ...]
    terminal_ordinal: int = 0

    def __post_init__(self) -> None:
        artifact_key = _require_instance(
            cast(object, self.artifact_key), ArtifactKey, "LeafArtifactDesc.artifact_key"
        )
        operators = _require_tuple(
            cast(object, self.operators), LeafOperatorDesc, "LeafArtifactDesc.operators"
        )
        input_values = _require_tuple(
            cast(object, self.input_values), str, "LeafArtifactDesc.input_values"
        )
        output_values = _require_tuple(
            cast(object, self.output_values), str, "LeafArtifactDesc.output_values"
        )
        terminal_ordinal = _require_int(
            cast(object, self.terminal_ordinal), "LeafArtifactDesc.terminal_ordinal"
        )
        if terminal_ordinal < 0:
            raise ValueError("LeafArtifactDesc.terminal_ordinal must be non-negative")
        object.__setattr__(self, "artifact_key", artifact_key)
        object.__setattr__(self, "operators", operators)
        object.__setattr__(self, "input_values", input_values)
        object.__setattr__(self, "output_values", output_values)
        object.__setattr__(self, "terminal_ordinal", terminal_ordinal)


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
    values: tuple[CapturedValueDesc, ...] = ()
    leaf_artifact: LeafArtifactDesc | None = None
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
        values = _require_tuple(
            cast(object, self.values), CapturedValueDesc, "CapturedFlow.values"
        )
        leaf_value = cast(object, self.leaf_artifact)
        leaf_artifact: LeafArtifactDesc | None
        if leaf_value is None:
            leaf_artifact = None
        else:
            leaf_artifact = _require_instance(
                leaf_value, LeafArtifactDesc, "CapturedFlow.leaf_artifact"
            )
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
            if values or leaf_artifact is not None:
                raise ValueError("opaque capture must discard every partial leaf artifact")
        else:
            if node_kind != NodeKind.COMPUTE:
                raise ValueError("successful capture must lower to a ComputeNode")
            if graph_safety not in {
                GraphSafetyKind.CAPTURE_SAFE,
                GraphSafetyKind.REPLAY_STATEFUL,
            }:
                raise ValueError("successful capture must be capture-safe or replay-stateful")
            if leaf_artifact is None or not values:
                raise ValueError("successful capture requires a complete typed leaf artifact")
            if diagnostics:
                raise ValueError("successful capture cannot retain failure diagnostics")
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
        object.__setattr__(self, "leaf_artifact", leaf_artifact)
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
        input_positions = {item.position for item in inputs}
        if any(
            binding.direction == PortBindingDirection.INPUT
            and (
                len(binding.value_path) != 1
                or binding.value_path[0] not in input_positions
            )
            for binding in bindings
        ):
            raise ValueError(
                "extern input bindings must reference a declared input position"
            )
        port_bindings = tuple(
            (binding.port, binding.direction) for binding in bindings
        )
        value_bindings = tuple(
            (binding.direction, binding.value_path) for binding in bindings
        )
        if len(set(port_bindings)) != len(port_bindings):
            raise ValueError("extern bindings must map each port exactly once")
        if len(set(value_bindings)) != len(value_bindings):
            raise ValueError("extern bindings must map each value path exactly once")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "effect", effect)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "capability", capability)
        object.__setattr__(self, "bindings", bindings)


class ExternCallable(Protocol[_P, _R_co]):
    extern_contract: ExternContract

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R_co: ...


# ---------------------------------------------------------------------------
# Program declaration vocabulary (plan: ESamp program compiler).
#
# These descriptors extend ``flow``/``extern`` with the Program-level
# declaration surface: where inputs come from, which named outputs escape to
# sinks, cross-function value references, and state keys. They are pure
# control-plane data; ``tllm.program`` lowers them.
# ---------------------------------------------------------------------------


def _program_str(value: object, label: str) -> str:
    """Program-name validator: TypeError for non-str, ValueError for empty."""
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{label} must be a non-empty str")
    return value


@dataclass(frozen=True, slots=True)
class HiddenSource:
    """flow input source: hidden rows from one layer/site/phase port."""

    layer: int
    site: str
    phase: str

    def __post_init__(self) -> None:
        if not isinstance(self.layer, int) or isinstance(self.layer, bool):  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard
            raise TypeError("hidden layer must be an int")
        _program_str(cast(object, self.site), "hidden.site")
        _program_str(cast(object, self.phase), "hidden.phase")


@dataclass(frozen=True, slots=True)
class ProvidedSource:
    """flow/extern input source: a tensor the caller supplies in the bundle.

    Unlike ``hidden()``, no engine tap is declared — the consumer hands the
    value over directly (device_lease entry key = declared input name).
    """


@dataclass(frozen=True, slots=True)
class SinkRef:
    """flow named output handed to a registered sink (e.g. ``guidance``)."""

    name: str

    def __post_init__(self) -> None:
        _program_str(cast(object, self.name), "sink.name")


@dataclass(frozen=True, slots=True)
class LocalRef:
    """flow named output used only by Program-internal data edges."""


@dataclass(frozen=True, slots=True)
class ResultRef:
    """extern input source: one named output of another flow."""

    flow: str
    output: str

    def __post_init__(self) -> None:
        _program_str(cast(object, self.flow), "result.flow")
        _program_str(cast(object, self.output), "result.output")


@dataclass(frozen=True, slots=True)
class PromptIdxKey:
    """State key source: prompt identity from bundle metadata."""


class RequestKeyField(StrEnum):
    REQUEST_ID = "request_id"
    PROMPT_IDX = "prompt_idx"
    SAMPLE_IDX = "sample_idx"


@dataclass(frozen=True, slots=True)
class RequestStateKey:
    """Composite REQUEST-state identity sourced from typed row metadata."""

    fields: tuple[RequestKeyField, ...]

    def __post_init__(self) -> None:
        normalized = tuple(RequestKeyField(field) for field in self.fields)
        if not normalized:
            raise ValueError("request state key requires at least one field")
        if len(normalized) != len(set(normalized)):
            raise ValueError("request state key fields must be unique")
        object.__setattr__(self, "fields", normalized)


def hidden(layer: int, site: str, phase: str) -> HiddenSource:
    return HiddenSource(layer=layer, site=site, phase=phase)


def provided() -> ProvidedSource:
    return ProvidedSource()


def sink(name: str) -> SinkRef:
    return SinkRef(name=name)


def local() -> LocalRef:
    return LocalRef()


def result(flow_name: str, output_name: str) -> ResultRef:
    return ResultRef(flow=flow_name, output=output_name)


def request_prompt_idx() -> PromptIdxKey:
    return PromptIdxKey()


def request_state_key(*fields: RequestKeyField | str) -> RequestStateKey:
    return RequestStateKey(tuple(RequestKeyField(field) for field in fields))


def request_sample_key() -> RequestStateKey:
    return request_state_key(RequestKeyField.REQUEST_ID, RequestKeyField.SAMPLE_IDX)


@dataclass(frozen=True, slots=True)
class FlowProgramDecl:
    """Program-mode metadata recorded by ``@flow(inputs/outputs/state)``."""

    inputs: tuple[tuple[str, HiddenSource | ProvidedSource], ...]
    outputs: tuple[tuple[str, SinkRef | LocalRef], ...]
    state: str | None


@dataclass(frozen=True, slots=True)
class ExternProgramDecl:
    """Program-mode metadata recorded by ``@extern(inputs/after/deadline)``."""

    inputs: tuple[tuple[str, ResultRef | HiddenSource | ProvidedSource], ...]
    after: tuple[str, ...]
    state: str | None
    deadline: TimingDeadline | None


def _program_mapping(
    value: object,
    value_types: tuple[type, ...],
    label: str,
) -> tuple[tuple[str, object], ...]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping of name to descriptor")
    items: list[tuple[str, object]] = []
    for key, item in cast(Mapping[object, object], value).items():
        key_str = _require_str(key, f"{label} key", nonempty=True)
        if not isinstance(item, value_types):
            raise TypeError(
                f"{label}[{key_str!r}] must be {'/'.join(t.__name__ for t in value_types)}, "
                f"got {type(item).__name__}"
            )
        items.append((key_str, item))
    return tuple(items)


def _check_param_names(pairs: tuple[tuple[str, object], ...], function: Callable[..., object], label: str) -> None:
    params = set(inspect.signature(function).parameters)
    params.discard("self")
    for key, _ in pairs:
        if key not in params:
            raise ValueError(f"{label} key {key!r} is not a parameter of {function.__name__}")


class FlowCallable(Protocol[_P, _R_co]):
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R_co: ...

    def capture(self, *args: _P.args, **kwargs: _P.kwargs) -> CapturedFlow: ...


@overload
def flow(function: Callable[_P, _R], /) -> FlowCallable[_P, _R]: ...


@overload
def flow(
    function: None = None,
    /,
    *,
    freshness: _BeforeNextUse = BeforeNextUse,
    bindings: tuple[PortBindingDesc, ...] = (),
    dynamic_dims: tuple[DynamicDim, ...] = (),
    inputs: Mapping[str, HiddenSource | ProvidedSource] | None = None,
    outputs: Mapping[str, SinkRef | LocalRef] | None = None,
    state: str | None = None,
) -> Callable[[Callable[_P, _R]], FlowCallable[_P, _R]]: ...


def flow(
    function: Callable[_P, _R] | None = None,
    /,
    *,
    freshness: _BeforeNextUse = BeforeNextUse,
    bindings: tuple[PortBindingDesc, ...] = (),
    dynamic_dims: tuple[DynamicDim, ...] = (),
    inputs: Mapping[str, HiddenSource | ProvidedSource] | None = None,
    outputs: Mapping[str, SinkRef | LocalRef] | None = None,
    state: str | None = None,
) -> FlowCallable[_P, _R] | Callable[[Callable[_P, _R]], FlowCallable[_P, _R]]:
    """Wrap a function transparently; only ``.capture`` enters control-plane capture.

    ``inputs``/``outputs``/``state`` are the keyword-only Program declaration
    surface (see ``tllm.program``); omitting all three preserves the legacy
    decorator exactly.
    """

    if freshness is not BeforeNextUse:
        raise TypeError("flow.freshness must be the BeforeNextUse singleton")
    bindings = _require_tuple(cast(object, bindings), PortBindingDesc, "flow.bindings")
    dynamic_dims = _require_tuple(cast(object, dynamic_dims), DynamicDim, "flow.dynamic_dims")
    program_inputs: tuple[tuple[str, object], ...] = ()
    program_outputs: tuple[tuple[str, object], ...] = ()
    if inputs is not None:
        program_inputs = _program_mapping(inputs, (HiddenSource, ProvidedSource), "flow.inputs")
    if outputs is not None:
        program_outputs = _program_mapping(outputs, (SinkRef, LocalRef), "flow.outputs")
    if state is not None:
        _require_str(cast(object, state), "flow.state", nonempty=True)
    program_mode = inputs is not None or outputs is not None or state is not None

    def decorate(target: Callable[_P, _R]) -> FlowCallable[_P, _R]:
        if not isinstance(target, FunctionType):
            raise TypeError("flow currently requires a Python function")
        program_decl: FlowProgramDecl | None = None
        if program_mode:
            _check_param_names(program_inputs, target, "flow.inputs")
            program_decl = FlowProgramDecl(
                inputs=cast(tuple[tuple[str, HiddenSource | ProvidedSource], ...], program_inputs),
                outputs=cast(tuple[tuple[str, SinkRef | LocalRef], ...], program_outputs),
                state=state,
            )

        @wraps(target)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return target(*args, **kwargs)

        def capture(*_args: _P.args, **_kwargs: _P.kwargs) -> CapturedFlow:
            inspect.signature(target).bind(*_args, **_kwargs)
            from tllm.workgraph.capture import capture_flow

            captured, exported_program = capture_flow(
                target,
                tuple(cast(tuple[object, ...], _args)),
                cast(dict[str, object], _kwargs),
                bindings=bindings,
                dynamic_dims=dynamic_dims,
            )
            if exported_program is not None:
                capture_handles[captured.shape_guard] = exported_program
            return captured

        capture_handles: dict[ShapeGuard, object] = {}
        setattr(wrapped, "capture", capture)
        setattr(wrapped, "flow_freshness", freshness)
        setattr(wrapped, "flow_bindings", bindings)
        setattr(wrapped, "flow_dynamic_dims", dynamic_dims)
        setattr(wrapped, "flow_program", program_decl)
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
        if parameter.name in ("self", "cls"):
            # Method receivers carry the Program instance, not an extern input.
            continue
        if parameter.annotation is inspect.Parameter.empty or parameter.name not in hints:
            raise TypeError(f"extern input {parameter.name!r} requires a concrete annotation")
        parameters.append(parameter.replace(annotation=hints[parameter.name]))
    if signature.return_annotation is inspect.Signature.empty or "return" not in hints:
        raise TypeError("extern output requires a concrete return annotation")
    return signature.replace(parameters=parameters, return_annotation=hints["return"])


def extern(
    *,
    effect: Effect | None = None,
    state: StateScope | str | None = None,
    device: Device | None = None,
    capability: Capability | None = None,
    bindings: tuple[PortBindingDesc, ...] = (),
    inputs: Mapping[str, ResultRef | HiddenSource | ProvidedSource] | None = None,
    after: tuple[str, ...] = (),
    deadline: TimingDeadline | None = None,
) -> Callable[[Callable[_P, _R]], ExternCallable[_P, _R]]:
    """Declare a typed opaque implementation while retaining normal Python calls.

    Two surfaces share this decorator:

    - legacy: ``effect``/``state``(StateScope)/``device``/``capability`` are all
      required and no Program metadata is recorded;
    - Program: ``inputs``/``after``/``deadline`` and a string ``state`` name
      declare cross-function data edges and timing; the contract fields default
      to ``WRITE_STATE``/``CUDA``/``CAPTURE_SAFE``.
    """

    bindings = _require_tuple(cast(object, bindings), PortBindingDesc, "extern.bindings")
    program_mode = (
        inputs is not None
        or bool(after)
        or deadline is not None
        # StateScope is a StrEnum; a plain str state name means Program mode.
        or (isinstance(state, str) and not isinstance(state, StateScope))
    )
    program_inputs: tuple[tuple[str, object], ...] = ()
    program_after: tuple[str, ...] = ()
    if program_mode:
        if inputs is not None:
            program_inputs = _program_mapping(inputs, (ResultRef, HiddenSource, ProvidedSource), "extern.inputs")
        if after:
            if not isinstance(cast(object, after), tuple):
                raise TypeError("extern.after must be a tuple of flow names")
            for name in after:
                _require_str(cast(object, name), "extern.after entry", nonempty=True)
            program_after = after
        if deadline is not None and not isinstance(cast(object, deadline), TimingDeadline):
            raise TypeError("extern.deadline must be a TimingDeadline")
        if state is not None:
            _require_str(cast(object, state), "extern.state", nonempty=True)
        if effect is None:
            effect = Effect.WRITE_STATE
        if device is None:
            device = Device.CUDA
        if capability is None:
            capability = Capability.CAPTURE_SAFE
        if state is None:
            raise ValueError("extern Program mode requires a state name")
    else:
        if effect is None or state is None or device is None or capability is None:
            raise TypeError(
                "extern requires effect/state/device/capability in legacy mode"
            )
    effect = _require_instance(cast(object, effect), Effect, "extern.effect")
    device = _require_instance(cast(object, device), Device, "extern.device")
    capability = _require_instance(cast(object, capability), Capability, "extern.capability")
    state_scope: StateScope
    if program_mode:
        # Program state ownership is carried by ExternProgramDecl.state; the
        # contract scope is a placeholder until tllm.program lowers the node.
        state_scope = StateScope.NONE
    else:
        state_scope = _require_instance(cast(object, state), StateScope, "extern.state")

    def decorate(function: Callable[_P, _R]) -> ExternCallable[_P, _R]:
        if not isinstance(function, FunctionType):
            raise TypeError("extern currently requires a Python function")
        program_decl: ExternProgramDecl | None = None
        if program_mode:
            _check_param_names(program_inputs, function, "extern.inputs")
            program_decl = ExternProgramDecl(
                inputs=cast(tuple[tuple[str, ResultRef | HiddenSource | ProvidedSource], ...], program_inputs),
                after=program_after,
                state=cast(str | None, state),
                deadline=deadline,
            )
        signature = _resolved_signature(function)
        inputs_desc = tuple(
            ExternInputDesc(
                position=position,
                name=parameter.name,
                value_type=TypeDesc.from_annotation(parameter.annotation),
            )
            for position, parameter in enumerate(signature.parameters.values())
        )
        contract = ExternContract(
            inputs=inputs_desc,
            output=TypeDesc.from_annotation(signature.return_annotation),
            effect=effect,
            state=state_scope,
            device=device,
            capability=capability,
            bindings=bindings,
        )

        @wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return function(*args, **kwargs)

        setattr(wrapped, "__signature__", signature)
        setattr(wrapped, "extern_contract", contract)
        setattr(wrapped, "extern_program", program_decl)
        return cast(ExternCallable[_P, _R], wrapped)

    return decorate


__all__ = [
    "ArtifactKey",
    "BeforeNextUse",
    "Capability",
    "CaptureStatus",
    "CapturedFlow",
    "CapturedValueDesc",
    "Device",
    "DynamicDim",
    "Effect",
    "ExternContract",
    "ExternCallable",
    "ExternInputDesc",
    "FlowCallable",
    "FlowCaptureCode",
    "FlowCaptureDiagnostic",
    "LeafArtifactDesc",
    "LeafOperatorDesc",
    "OperatorKind",
    "OperatorAttributeDesc",
    "OperatorAttributeKind",
    "PortBindingDesc",
    "PortBindingDirection",
    "ShapeGuard",
    "ShapeGuardExpr",
    "ShapeGuardKind",
    "SourceSpan",
    "StateScope",
    "TensorDTypeDesc",
    "TensorDeviceDesc",
    "TensorLayout",
    "TypeDesc",
    "UseDefDesc",
    "ExternProgramDecl",
    "FlowProgramDecl",
    "HiddenSource",
    "ProvidedSource",
    "LocalRef",
    "PromptIdxKey",
    "RequestKeyField",
    "RequestStateKey",
    "ResultRef",
    "SinkRef",
    "extern",
    "flow",
    "hidden",
    "local",
    "provided",
    "request_prompt_idx",
    "request_sample_key",
    "request_state_key",
    "result",
    "sink",
]
