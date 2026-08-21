"""Read-only LSP-style views over frozen WorkGraph capture metadata.

This adapter deliberately has no capture, compilation, execution, or runtime
object dependency.  It copies the small set of source and inferred facts that
an editor needs into frozen descriptors and only creates JSON mappings at the
outer ``to_dict`` boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias, cast

from tllm.workgraph.shadow import ShadowParityPlan
from tllm.workgraph.frontend import (
    ArtifactKey,
    Capability,
    CaptureStatus,
    CapturedFlow,
    CapturedValueDesc,
    Effect,
    FlowCaptureCode,
    ShapeGuard,
    SourceSpan,
    TensorDTypeDesc,
    TensorDeviceDesc,
    TensorLayout,
    UseDefDesc,
)
from tllm.workgraph.hierarchy import (
    CaptureHierarchy,
    PartitionBoundary,
    RegionKey,
    RegionKind,
    RootFlowDesc,
    build_capture_root,
)
from tllm.workgraph.ir import (
    ExecutorKey,
    NodeKey,
    WorkGraphDiagnostic,
    WorkGraphDiagnosticCode,
)


class LspDiagnosticSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFORMATION = "information"
    HINT = "hint"


LspDiagnosticCode: TypeAlias = FlowCaptureCode | WorkGraphDiagnosticCode


def _tuple_of(value: object, expected: type[object], label: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be a tuple")
    items = cast(tuple[object, ...], value)
    if any(not isinstance(item, expected) for item in items):
        raise TypeError(f"{label} must contain only {expected.__name__}")
    return items


def _tuple_of_int(value: object, label: str) -> tuple[int, ...]:
    items = _tuple_of(value, int, label)
    typed = cast(tuple[int, ...], items)
    if any(isinstance(item, bool) or item < 0 for item in typed):
        raise ValueError(f"{label} must contain non-negative integers")
    return typed


def _tuple_of_str(value: object, label: str) -> tuple[str, ...]:
    items = _tuple_of(value, str, label)
    typed = cast(tuple[str, ...], items)
    if any(not item for item in typed):
        raise ValueError(f"{label} must contain non-empty strings")
    return typed


def _require_span(value: object, label: str) -> SourceSpan:
    if not isinstance(value, SourceSpan):
        raise TypeError(f"{label} must be SourceSpan")
    return value


def _require_code(value: object, label: str) -> LspDiagnosticCode:
    if not isinstance(value, (FlowCaptureCode, WorkGraphDiagnosticCode)):
        raise TypeError(f"{label} must be a typed WorkGraph diagnostic code")
    return value


@dataclass(frozen=True, slots=True)
class LspDiagnostic:
    """One source-aware, read-only diagnostic for an editor client."""

    code: LspDiagnosticCode
    message: str
    source_span: SourceSpan | None
    severity: LspDiagnosticSeverity = LspDiagnosticSeverity.ERROR
    detail: str = ""

    def __post_init__(self) -> None:
        code = _require_code(self.code, "LspDiagnostic.code")
        if not isinstance(cast(object, self.message), str) or not self.message:
            raise TypeError("LspDiagnostic.message must be a non-empty string")
        source_span = (
            None
            if self.source_span is None
            else _require_span(self.source_span, "LspDiagnostic.source_span")
        )
        if not isinstance(cast(object, self.severity), LspDiagnosticSeverity):
            raise TypeError("LspDiagnostic.severity must be LspDiagnosticSeverity")
        if not isinstance(cast(object, self.detail), str):
            raise TypeError("LspDiagnostic.detail must be a string")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "source_span", source_span)


@dataclass(frozen=True, slots=True)
class LspValueDescriptor:
    """Tensor facts shown by a value hover."""

    value_id: str
    version: int
    shape: tuple[str, ...]
    dtype: TensorDTypeDesc
    device: TensorDeviceDesc
    layout: TensorLayout
    stride: tuple[str, ...]
    effect: Effect
    alias_set: str
    use_def: UseDefDesc

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.value_id), str) or not self.value_id:
            raise TypeError("LspValueDescriptor.value_id must be a non-empty string")
        if (
            not isinstance(cast(object, self.version), int)
            or isinstance(cast(object, self.version), bool)
            or self.version < 1
        ):
            raise ValueError("LspValueDescriptor.version must be a positive integer")
        shape = _tuple_of_str(self.shape, "LspValueDescriptor.shape")
        stride = _tuple_of_str(self.stride, "LspValueDescriptor.stride")
        if not isinstance(cast(object, self.dtype), TensorDTypeDesc):
            raise TypeError("LspValueDescriptor.dtype must be TensorDTypeDesc")
        if not isinstance(cast(object, self.device), TensorDeviceDesc):
            raise TypeError("LspValueDescriptor.device must be TensorDeviceDesc")
        if not isinstance(cast(object, self.layout), TensorLayout):
            raise TypeError("LspValueDescriptor.layout must be TensorLayout")
        if not isinstance(cast(object, self.effect), Effect):
            raise TypeError("LspValueDescriptor.effect must be Effect")
        if not isinstance(cast(object, self.alias_set), str) or not self.alias_set:
            raise TypeError("LspValueDescriptor.alias_set must be a non-empty string")
        if not isinstance(cast(object, self.use_def), UseDefDesc):
            raise TypeError("LspValueDescriptor.use_def must be UseDefDesc")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "stride", stride)


@dataclass(frozen=True, slots=True)
class LspHoverDescriptor:
    """A typed hover for a flow or one captured Tensor value."""

    subject: str
    source_span: SourceSpan
    value: LspValueDescriptor | None
    effects: tuple[Effect, ...]
    shape_guard: ShapeGuard
    region: RegionKey | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.subject), str) or not self.subject:
            raise TypeError("LspHoverDescriptor.subject must be a non-empty string")
        _require_span(self.source_span, "LspHoverDescriptor.source_span")
        effects = _tuple_of(self.effects, Effect, "LspHoverDescriptor.effects")
        if not isinstance(cast(object, self.shape_guard), ShapeGuard):
            raise TypeError("LspHoverDescriptor.shape_guard must be ShapeGuard")
        if self.region is not None and not isinstance(cast(object, self.region), RegionKey):
            raise TypeError("LspHoverDescriptor.region must be RegionKey or None")
        if self.value is not None and not isinstance(cast(object, self.value), LspValueDescriptor):
            raise TypeError("LspHoverDescriptor.value must be LspValueDescriptor or None")
        object.__setattr__(self, "effects", cast(tuple[Effect, ...], effects))


@dataclass(frozen=True, slots=True)
class LspRegionDescriptor:
    """A read-only root, task, or leaf region descriptor."""

    key: RegionKey
    artifact_key: ArtifactKey
    kind: RegionKind
    source_span: SourceSpan
    status: CaptureStatus
    parent: RegionKey | None
    boundary: PartitionBoundary | None
    node_key: NodeKey | None
    executor: ExecutorKey | None
    operator_ordinals: tuple[int, ...]
    input_values: tuple[str, ...]
    output_values: tuple[str, ...]
    dependencies: tuple[RegionKey, ...]
    effects: tuple[Effect, ...]
    capabilities: tuple[Capability, ...]

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.key), RegionKey):
            raise TypeError("LspRegionDescriptor.key must be RegionKey")
        if not isinstance(cast(object, self.artifact_key), ArtifactKey):
            raise TypeError("LspRegionDescriptor.artifact_key must be ArtifactKey")
        if self.key.artifact_key != self.artifact_key or self.key.kind != self.kind:
            raise ValueError("LspRegionDescriptor key does not match its identity")
        if not isinstance(cast(object, self.kind), RegionKind):
            raise TypeError("LspRegionDescriptor.kind must be RegionKind")
        _require_span(self.source_span, "LspRegionDescriptor.source_span")
        if not isinstance(cast(object, self.status), CaptureStatus):
            raise TypeError("LspRegionDescriptor.status must be CaptureStatus")
        if self.parent is not None and not isinstance(cast(object, self.parent), RegionKey):
            raise TypeError("LspRegionDescriptor.parent must be RegionKey or None")
        if self.boundary is not None and not isinstance(
            cast(object, self.boundary), PartitionBoundary
        ):
            raise TypeError("LspRegionDescriptor.boundary must be PartitionBoundary or None")
        if self.node_key is not None and not isinstance(cast(object, self.node_key), NodeKey):
            raise TypeError("LspRegionDescriptor.node_key must be NodeKey or None")
        if self.executor is not None and not isinstance(cast(object, self.executor), ExecutorKey):
            raise TypeError("LspRegionDescriptor.executor must be ExecutorKey or None")
        operator_ordinals = _tuple_of_int(
            self.operator_ordinals, "LspRegionDescriptor.operator_ordinals"
        )
        if len(set(operator_ordinals)) != len(operator_ordinals):
            raise ValueError("LspRegionDescriptor.operator_ordinals must be unique")
        input_values = _tuple_of_str(self.input_values, "LspRegionDescriptor.input_values")
        output_values = _tuple_of_str(self.output_values, "LspRegionDescriptor.output_values")
        dependencies = _tuple_of(self.dependencies, RegionKey, "LspRegionDescriptor.dependencies")
        effects = _tuple_of(self.effects, Effect, "LspRegionDescriptor.effects")
        capabilities = _tuple_of(self.capabilities, Capability, "LspRegionDescriptor.capabilities")
        if self.kind == RegionKind.ROOT:
            if self.parent is not None or self.boundary is not None:
                raise ValueError("root LSP region cannot have a parent or boundary")
            if self.node_key is not None or self.executor is not None:
                raise ValueError("root LSP region cannot have a runtime identity")
        elif self.kind == RegionKind.TASK:
            if self.parent is None or self.parent.kind != RegionKind.ROOT:
                raise ValueError("task LSP region must be a child of a root")
            if self.boundary is None:
                raise ValueError("task LSP region requires a partition boundary")
            if self.node_key is None or self.executor is None:
                raise ValueError("task LSP region requires stable descriptor identities")
        elif self.kind in {RegionKind.LEAF, RegionKind.OPAQUE}:
            if self.parent is None or self.parent.kind != RegionKind.TASK:
                raise ValueError("leaf LSP region must be a child of a task")
            if self.boundary is not None:
                raise ValueError("leaf LSP region cannot have a partition boundary")
            if self.node_key is not None or self.executor is not None:
                raise ValueError("leaf LSP region cannot have a runtime identity")
            if self.kind == RegionKind.OPAQUE and operator_ordinals:
                raise ValueError("opaque LSP region cannot expose operator ordinals")
        else:
            raise ValueError("unsupported LSP region kind")
        object.__setattr__(self, "operator_ordinals", operator_ordinals)
        object.__setattr__(self, "input_values", input_values)
        object.__setattr__(self, "output_values", output_values)
        object.__setattr__(self, "dependencies", cast(tuple[RegionKey, ...], dependencies))
        object.__setattr__(self, "effects", cast(tuple[Effect, ...], effects))
        object.__setattr__(self, "capabilities", cast(tuple[Capability, ...], capabilities))


@dataclass(frozen=True, slots=True)
class LspSnapshot:
    """Frozen read-only document data for an LSP client."""

    artifact_keys: tuple[ArtifactKey, ...]
    shape_guards: tuple[ShapeGuard, ...]
    diagnostics: tuple[LspDiagnostic, ...]
    hovers: tuple[LspHoverDescriptor, ...]
    regions: tuple[LspRegionDescriptor, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        artifact_keys = _tuple_of(self.artifact_keys, ArtifactKey, "LspSnapshot.artifact_keys")
        shape_guards = _tuple_of(self.shape_guards, ShapeGuard, "LspSnapshot.shape_guards")
        diagnostics = _tuple_of(self.diagnostics, LspDiagnostic, "LspSnapshot.diagnostics")
        hovers = _tuple_of(self.hovers, LspHoverDescriptor, "LspSnapshot.hovers")
        regions = _tuple_of(self.regions, LspRegionDescriptor, "LspSnapshot.regions")
        if self.schema_version != 1:
            raise ValueError("LspSnapshot.schema_version must be 1")
        if len(set(artifact_keys)) != len(artifact_keys):
            raise ValueError("LspSnapshot.artifact_keys must be unique")
        region_values = cast(tuple[LspRegionDescriptor, ...], regions)
        if len({region.key for region in region_values}) != len(region_values):
            raise ValueError("LspSnapshot.regions must have unique keys")
        if any(region.artifact_key not in artifact_keys for region in region_values):
            raise ValueError("LspSnapshot region references an unknown artifact")
        object.__setattr__(self, "artifact_keys", cast(tuple[ArtifactKey, ...], artifact_keys))
        object.__setattr__(self, "shape_guards", cast(tuple[ShapeGuard, ...], shape_guards))
        object.__setattr__(self, "diagnostics", cast(tuple[LspDiagnostic, ...], diagnostics))
        object.__setattr__(self, "hovers", cast(tuple[LspHoverDescriptor, ...], hovers))
        object.__setattr__(self, "regions", region_values)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_keys": [_artifact_key_dict(key) for key in self.artifact_keys],
            "shape_guards": [_shape_guard_dict(guard) for guard in self.shape_guards],
            "diagnostics": [_diagnostic_dict(item) for item in self.diagnostics],
            "hovers": [_hover_dict(item) for item in self.hovers],
            "regions": [_region_dict(item) for item in self.regions],
        }

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _artifact_key_dict(key: ArtifactKey) -> dict[str, object]:
    return {
        "namespace": key.namespace,
        "name": key.name,
        "version": key.version,
        "digest": key.digest,
        "value": key.value,
    }


def _source_span_dict(span: SourceSpan | None) -> dict[str, object] | None:
    if span is None:
        return None
    return {
        "file": span.file,
        "start_line": span.start_line,
        "start_column": span.start_column,
        "end_line": span.end_line,
        "end_column": span.end_column,
    }


def _shape_guard_dict(guard: ShapeGuard) -> list[dict[str, object]]:
    return [
        {
            "value_path": list(expression.value_path),
            "dimension": expression.dimension,
            "kind": expression.kind.value,
            "lower_bound": expression.lower_bound,
            "upper_bound": expression.upper_bound,
        }
        for expression in guard.expressions
    ]


def _diagnostic_dict(diagnostic: LspDiagnostic) -> dict[str, object]:
    return {
        "code": diagnostic.code.value,
        "message": diagnostic.message,
        "source_span": _source_span_dict(diagnostic.source_span),
        "severity": diagnostic.severity.value,
        "detail": diagnostic.detail,
    }


def _value_dict(value: LspValueDescriptor) -> dict[str, object]:
    return {
        "value_id": value.value_id,
        "version": value.version,
        "shape": list(value.shape),
        "dtype": {"namespace": value.dtype.namespace, "name": value.dtype.name},
        "device": {"kind": value.device.kind.value, "index": value.device.index},
        "layout": value.layout.value,
        "stride": list(value.stride),
        "effect": value.effect.value,
        "alias_set": value.alias_set,
        "use_def": {
            "producer": value.use_def.producer,
            "inputs": list(value.use_def.inputs),
            "users": list(value.use_def.users),
        },
    }


def _hover_dict(hover: LspHoverDescriptor) -> dict[str, object]:
    return {
        "subject": hover.subject,
        "source_span": _source_span_dict(hover.source_span),
        "value": None if hover.value is None else _value_dict(hover.value),
        "effects": [effect.value for effect in hover.effects],
        "shape_guard": _shape_guard_dict(hover.shape_guard),
        "region": None if hover.region is None else hover.region.value,
    }


def _region_dict(region: LspRegionDescriptor) -> dict[str, object]:
    return {
        "key": region.key.value,
        "artifact_key": region.artifact_key.value,
        "kind": region.kind.value,
        "source_span": _source_span_dict(region.source_span),
        "status": region.status.value,
        "parent": None if region.parent is None else region.parent.value,
        "boundary": None if region.boundary is None else region.boundary.value,
        "node_key": None if region.node_key is None else region.node_key.value,
        "executor": None if region.executor is None else region.executor.value,
        "operator_ordinals": list(region.operator_ordinals),
        "input_values": list(region.input_values),
        "output_values": list(region.output_values),
        "dependencies": [dependency.value for dependency in region.dependencies],
        "effects": [effect.value for effect in region.effects],
        "capabilities": [capability.value for capability in region.capabilities],
    }


def _value_descriptor(value: CapturedValueDesc) -> LspValueDescriptor:
    return LspValueDescriptor(
        value_id=value.value_id,
        version=value.version,
        shape=value.shape,
        dtype=value.dtype,
        device=value.device,
        layout=value.layout,
        stride=value.stride,
        effect=value.effect,
        alias_set=value.alias_set,
        use_def=value.use_def,
    )


def _plan_diagnostic(diagnostic: WorkGraphDiagnostic, source_span: SourceSpan | None) -> LspDiagnostic:
    return LspDiagnostic(
        code=diagnostic.code,
        message=diagnostic.message,
        source_span=source_span,
        severity=LspDiagnosticSeverity.WARNING,
        detail=diagnostic.detail,
    )


def _regions_for_root(root: RootFlowDesc) -> tuple[LspRegionDescriptor, ...]:
    artifact = root.children[0].children[0].leaf_artifact
    root_input_values = () if artifact is None else artifact.input_values
    root_output_values = () if artifact is None else artifact.output_values
    regions: list[LspRegionDescriptor] = [
        LspRegionDescriptor(
            key=root.key,
            artifact_key=root.artifact_key,
            kind=RegionKind.ROOT,
            source_span=root.source_span,
            status=root.status,
            parent=None,
            boundary=None,
            node_key=None,
            executor=None,
            operator_ordinals=(),
            input_values=root_input_values,
            output_values=root_output_values,
            dependencies=(),
            effects=root.effects,
            capabilities=tuple(
                dict.fromkeys(
                    capability for task in root.children for capability in task.capabilities
                )
            ),
        )
    ]
    for task in root.children:
        regions.append(
            LspRegionDescriptor(
                key=task.key,
                artifact_key=root.artifact_key,
                kind=RegionKind.TASK,
                source_span=root.source_span,
                status=root.status,
                parent=root.key,
                boundary=task.boundary,
                node_key=task.node_key,
                executor=task.executor,
                operator_ordinals=task.operator_ordinals,
                input_values=task.input_values,
                output_values=task.output_values,
                dependencies=task.dependencies,
                effects=task.effects,
                capabilities=task.capabilities,
            )
        )
        leaf = task.children[0]
        regions.append(
            LspRegionDescriptor(
                key=leaf.key,
                artifact_key=root.artifact_key,
                kind=leaf.kind,
                source_span=root.source_span,
                status=root.status,
                parent=task.key,
                boundary=None,
                node_key=None,
                executor=None,
                operator_ordinals=leaf.operator_ordinals,
                input_values=task.input_values,
                output_values=task.output_values,
                dependencies=(),
                effects=task.effects,
                capabilities=task.capabilities,
            )
        )
    return tuple(regions)


def _regions_for_hierarchy(hierarchy: CaptureHierarchy) -> tuple[LspRegionDescriptor, ...]:
    return tuple(region for root in hierarchy.roots for region in _regions_for_root(root))


def _hovers_for_root(root: RootFlowDesc) -> tuple[LspHoverDescriptor, ...]:
    hovers = [
        LspHoverDescriptor(
            subject=root.artifact_key.value,
            source_span=root.source_span,
            value=None,
            effects=root.effects,
            shape_guard=root.shape_guard,
            region=root.key,
        )
    ]
    hovers.extend(
        LspHoverDescriptor(
            subject=f"{value.value_id}@{value.version}",
            source_span=root.source_span,
            value=_value_descriptor(value),
            effects=(value.effect,),
            shape_guard=root.shape_guard,
            region=root.key,
        )
        for value in root.values
    )
    return tuple(hovers)


def _snapshot_from_capture(capture: CapturedFlow) -> LspSnapshot:
    root = build_capture_root(capture)
    regions = _regions_for_root(root)
    return LspSnapshot(
        artifact_keys=(root.artifact_key,),
        shape_guards=(root.shape_guard,),
        diagnostics=tuple(
            LspDiagnostic(
                code=diagnostic.code,
                message=diagnostic.message,
                source_span=diagnostic.source_span,
                detail=diagnostic.detail,
            )
            for diagnostic in root.diagnostics
        ),
        hovers=_hovers_for_root(root),
        regions=regions,
    )


def _snapshot_from_plan(plan: ShadowParityPlan) -> LspSnapshot:
    hierarchy = plan.hierarchy
    if hierarchy is None:
        source_span = None
        diagnostics = tuple(_plan_diagnostic(item, source_span) for item in plan.diagnostics)
        return LspSnapshot(
            artifact_keys=(),
            shape_guards=(),
            diagnostics=diagnostics,
            hovers=(),
            regions=(),
        )
    roots = hierarchy.roots
    source_span = roots[0].source_span if roots else None
    diagnostics = [
        _plan_diagnostic(item, source_span)
        for item in plan.diagnostics
    ]
    diagnostics.extend(
        diagnostic
        for root in roots
        for diagnostic in (
            LspDiagnostic(
                code=item.code,
                message=item.message,
                source_span=item.source_span,
                detail=item.detail,
            )
            for item in root.diagnostics
        )
    )
    return LspSnapshot(
        artifact_keys=tuple(root.artifact_key for root in roots),
        shape_guards=tuple(root.shape_guard for root in roots),
        diagnostics=tuple(diagnostics),
        hovers=tuple(hover for root in roots for hover in _hovers_for_root(root)),
        regions=_regions_for_hierarchy(hierarchy),
    )


def build_lsp_snapshot(snapshot: CapturedFlow | ShadowParityPlan) -> LspSnapshot:
    """Build a frozen read-only LSP view from a pure WorkGraph snapshot."""

    if isinstance(snapshot, CapturedFlow):
        return _snapshot_from_capture(snapshot)
    if isinstance(cast(object, snapshot), ShadowParityPlan):
        return _snapshot_from_plan(snapshot)
    raise TypeError("build_lsp_snapshot expects CapturedFlow or ShadowParityPlan")


def adapt_lsp(snapshot: CapturedFlow | ShadowParityPlan) -> LspSnapshot:
    """Alias for callers that name the adapter operation directly."""

    return build_lsp_snapshot(snapshot)


__all__ = [
    "LspDiagnostic",
    "LspDiagnosticCode",
    "LspDiagnosticSeverity",
    "LspHoverDescriptor",
    "LspRegionDescriptor",
    "LspSnapshot",
    "LspValueDescriptor",
    "adapt_lsp",
    "build_lsp_snapshot",
]
