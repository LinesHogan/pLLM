"""Hierarchical capture sidecar for the Phase 1 compatibility compiler.

The runtime schedule remains the flat SemanticGraph/DataCentricIR projection.
This module records root-flow, task-region, and leaf-artifact relationships
without retaining callbacks, exported programs, Tensors, or runtime plans.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import cast

from tllm.ports.address import PortAddressDesc, port_address_desc
from tllm.ports.port_id import PortId
from tllm.producers.registry import ExecutionPlan
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.workgraph.frontend import (
    ArtifactKey,
    Capability,
    CaptureStatus,
    CapturedFlow,
    CapturedValueDesc,
    Effect,
    FlowCaptureDiagnostic,
    LeafArtifactDesc,
    LeafOperatorDesc,
    OperatorKind,
    PortBindingDesc,
    PortBindingDirection,
    ShapeGuard,
    SourceSpan,
)
from tllm.workgraph.ir import (
    DataCentricIR,
    EdgeKind,
    ExecutorKey,
    LeaseDesc,
    NodeKey,
    NodeKind,
    SemanticGraph,
)


class RegionKind(StrEnum):
    ROOT = "root"
    TASK = "task"
    LEAF = "leaf"
    OPAQUE = "opaque"


class PartitionBoundary(StrEnum):
    ROOT = "root"
    MODULE = "module"
    CUSTOM_OP = "custom_op"
    EFFECT = "effect"
    CAPABILITY = "capability"
    OPAQUE = "opaque"


def _typed_tuple(value: object, item_type: type[object], label: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be tuple")
    items = cast(tuple[object, ...], value)
    if any(not isinstance(item, item_type) for item in items):
        raise TypeError(f"{label} must contain only {item_type.__name__}")
    return items


@dataclass(frozen=True, slots=True, order=True)
class RegionKey:
    artifact_key: ArtifactKey
    kind: RegionKind
    ordinal: int = 0

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.artifact_key), ArtifactKey):
            raise TypeError("RegionKey.artifact_key must be ArtifactKey")
        if not isinstance(cast(object, self.kind), RegionKind):
            raise TypeError("RegionKey.kind must be RegionKind")
        ordinal = cast(object, self.ordinal)
        if not isinstance(ordinal, int) or isinstance(ordinal, bool):
            raise TypeError("RegionKey.ordinal must be int")
        if self.ordinal < 0:
            raise ValueError("RegionKey.ordinal must be non-negative")

    @property
    def value(self) -> str:
        return f"{self.artifact_key.value}:{self.kind.value}:{self.ordinal}"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class LeafRegionDesc:
    key: RegionKey
    artifact_key: ArtifactKey
    kind: RegionKind
    operator_ordinals: tuple[int, ...] = ()
    leaf_artifact: LeafArtifactDesc | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.key), RegionKey):
            raise TypeError("LeafRegionDesc.key must be RegionKey")
        if not isinstance(cast(object, self.artifact_key), ArtifactKey):
            raise TypeError("LeafRegionDesc.artifact_key must be ArtifactKey")
        if self.kind not in {RegionKind.LEAF, RegionKind.OPAQUE}:
            raise ValueError("LeafRegionDesc.kind must be leaf or opaque")
        ordinals = _typed_tuple(self.operator_ordinals, int, "LeafRegionDesc.operator_ordinals")
        typed_ordinals = cast(tuple[int, ...], ordinals)
        if any(ordinal < 0 for ordinal in typed_ordinals):
            raise ValueError("LeafRegionDesc.operator_ordinals must be non-negative")
        if self.key.kind != self.kind or self.key.artifact_key != self.artifact_key:
            raise ValueError("LeafRegionDesc key must match its kind and artifact")
        if self.kind == RegionKind.OPAQUE:
            if self.leaf_artifact is not None or typed_ordinals:
                raise ValueError("opaque leaf must not retain a partial export artifact")
        else:
            if not isinstance(self.leaf_artifact, LeafArtifactDesc):
                raise TypeError("exported leaf requires LeafArtifactDesc")
            if self.leaf_artifact.artifact_key != self.artifact_key:
                raise ValueError("leaf artifact identity differs from its region")
            known = {operator.ordinal for operator in self.leaf_artifact.operators}
            if not set(typed_ordinals).issubset(known):
                raise ValueError("leaf operator ordinals must reference the export artifact")
            if not typed_ordinals and known:
                raise ValueError("non-empty export artifact requires explicit operator coverage")
        object.__setattr__(self, "operator_ordinals", typed_ordinals)


@dataclass(frozen=True, slots=True)
class TaskRegionDesc:
    key: RegionKey
    node_key: NodeKey
    executor: ExecutorKey
    boundary: PartitionBoundary
    module_path: tuple[str, ...]
    operator_ordinals: tuple[int, ...]
    input_values: tuple[str, ...]
    output_values: tuple[str, ...]
    effects: tuple[Effect, ...]
    capabilities: tuple[Capability, ...]
    dependencies: tuple[RegionKey, ...]
    children: tuple[LeafRegionDesc, ...]
    kind: RegionKind = RegionKind.TASK

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.key), RegionKey) or self.key.kind != RegionKind.TASK:
            raise TypeError("TaskRegionDesc.key must identify a task region")
        if not isinstance(cast(object, self.node_key), NodeKey):
            raise TypeError("TaskRegionDesc.node_key must be NodeKey")
        if not isinstance(cast(object, self.executor), ExecutorKey):
            raise TypeError("TaskRegionDesc.executor must be ExecutorKey")
        if not isinstance(cast(object, self.boundary), PartitionBoundary):
            raise TypeError("TaskRegionDesc.boundary must be PartitionBoundary")
        if self.kind != RegionKind.TASK:
            raise ValueError("TaskRegionDesc.kind must be task")
        module_path = cast(
            tuple[str, ...], _typed_tuple(self.module_path, str, "TaskRegionDesc.module_path")
        )
        operator_ordinals = cast(
            tuple[int, ...],
            _typed_tuple(self.operator_ordinals, int, "TaskRegionDesc.operator_ordinals"),
        )
        input_values = cast(
            tuple[str, ...], _typed_tuple(self.input_values, str, "TaskRegionDesc.input_values")
        )
        output_values = cast(
            tuple[str, ...], _typed_tuple(self.output_values, str, "TaskRegionDesc.output_values")
        )
        effects = cast(tuple[Effect, ...], _typed_tuple(self.effects, Effect, "TaskRegionDesc.effects"))
        capabilities = cast(
            tuple[Capability, ...],
            _typed_tuple(self.capabilities, Capability, "TaskRegionDesc.capabilities"),
        )
        dependencies = cast(
            tuple[RegionKey, ...],
            _typed_tuple(self.dependencies, RegionKey, "TaskRegionDesc.dependencies"),
        )
        children = cast(
            tuple[LeafRegionDesc, ...],
            _typed_tuple(self.children, LeafRegionDesc, "TaskRegionDesc.children"),
        )
        if len(children) != 1:
            raise ValueError("TaskRegionDesc must own exactly one leaf region")
        if children[0].operator_ordinals != operator_ordinals:
            raise ValueError("task and leaf operator coverage must match")
        if len(set(operator_ordinals)) != len(operator_ordinals):
            raise ValueError("TaskRegionDesc contains duplicate operator ordinals")
        if not effects or not capabilities:
            raise ValueError("TaskRegionDesc requires typed effect and capability facts")
        if self.key in dependencies:
            raise ValueError("TaskRegionDesc cannot depend on itself")
        if children[0].leaf_artifact is not None:
            operators_by_ordinal = {
                operator.ordinal: operator for operator in children[0].leaf_artifact.operators
            }
            selected = tuple(operators_by_ordinal[ordinal] for ordinal in operator_ordinals)
            if any(
                operator.module_path != module_path
                or operator.effects != effects
                or operator.capabilities != capabilities
                for operator in selected
            ):
                raise ValueError("task facts must match every selected leaf operator")
        object.__setattr__(self, "module_path", module_path)
        object.__setattr__(self, "operator_ordinals", operator_ordinals)
        object.__setattr__(self, "input_values", input_values)
        object.__setattr__(self, "output_values", output_values)
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "children", children)


@dataclass(frozen=True, slots=True)
class RootFlowDesc:
    key: RegionKey
    artifact_key: ArtifactKey
    source_span: SourceSpan
    status: CaptureStatus
    shape_guard: ShapeGuard
    inputs: tuple[PortBindingDesc, ...]
    outputs: tuple[PortBindingDesc, ...]
    values: tuple[CapturedValueDesc, ...]
    effects: tuple[Effect, ...]
    diagnostics: tuple[FlowCaptureDiagnostic, ...]
    children: tuple[TaskRegionDesc, ...]
    kind: RegionKind = RegionKind.ROOT

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.key), RegionKey) or self.key.kind != RegionKind.ROOT:
            raise TypeError("RootFlowDesc.key must identify a root region")
        if self.key.artifact_key != self.artifact_key:
            raise ValueError("RootFlowDesc key must match its artifact")
        if not self.artifact_key.digest:
            raise ValueError("RootFlowDesc artifact must include a structural digest")
        if not isinstance(cast(object, self.source_span), SourceSpan):
            raise TypeError("RootFlowDesc.source_span must be SourceSpan")
        if not isinstance(cast(object, self.status), CaptureStatus):
            raise TypeError("RootFlowDesc.status must be CaptureStatus")
        if not isinstance(cast(object, self.shape_guard), ShapeGuard):
            raise TypeError("RootFlowDesc.shape_guard must be ShapeGuard")
        inputs = cast(
            tuple[PortBindingDesc, ...],
            _typed_tuple(self.inputs, PortBindingDesc, "RootFlowDesc.inputs"),
        )
        outputs = cast(
            tuple[PortBindingDesc, ...],
            _typed_tuple(self.outputs, PortBindingDesc, "RootFlowDesc.outputs"),
        )
        values = cast(
            tuple[CapturedValueDesc, ...],
            _typed_tuple(self.values, CapturedValueDesc, "RootFlowDesc.values"),
        )
        effects = cast(
            tuple[Effect, ...], _typed_tuple(self.effects, Effect, "RootFlowDesc.effects")
        )
        diagnostics = cast(
            tuple[FlowCaptureDiagnostic, ...],
            _typed_tuple(
                self.diagnostics,
                FlowCaptureDiagnostic,
                "RootFlowDesc.diagnostics",
            ),
        )
        children = cast(
            tuple[TaskRegionDesc, ...],
            _typed_tuple(self.children, TaskRegionDesc, "RootFlowDesc.children"),
        )
        if self.kind != RegionKind.ROOT or not children:
            raise ValueError("RootFlowDesc must contain at least one task")
        if any(child.key.artifact_key != self.artifact_key for child in children):
            raise ValueError("root task belongs to another artifact")
        task_keys = {child.key for child in children}
        if len(task_keys) != len(children):
            raise ValueError("RootFlowDesc contains duplicate task keys")
        for child in children:
            if any(dependency not in task_keys for dependency in child.dependencies):
                raise ValueError("task dependency references another root")
        positions = {child.key: index for index, child in enumerate(children)}
        for child in children:
            if any(positions[dependency] >= positions[child.key] for dependency in child.dependencies):
                raise ValueError("task dependencies must follow topological order")
        if self.status == CaptureStatus.OPAQUE_BARRIER:
            if len(children) != 1 or children[0].boundary != PartitionBoundary.OPAQUE:
                raise ValueError("opaque root must contain one opaque task")
            if values or not diagnostics or Effect.OPAQUE not in effects:
                raise ValueError("opaque root must retain diagnostics/effect and no partial values")
        else:
            if not values or diagnostics:
                raise ValueError("captured root requires inferred values and no failure diagnostics")
            previous_operator: LeafOperatorDesc | None = None
            covered = tuple(ordinal for child in children for ordinal in child.operator_ordinals)
            leaf_artifact = cast(LeafArtifactDesc, children[0].children[0].leaf_artifact)
            expected = tuple(operator.ordinal for operator in leaf_artifact.operators)
            if len(covered) != len(set(covered)) or set(covered) != set(expected):
                raise ValueError("task regions must cover every leaf operator exactly once")
            operators_by_ordinal = {
                operator.ordinal: operator for operator in leaf_artifact.operators
            }
            for child in children:
                selected = tuple(
                    operators_by_ordinal[ordinal] for ordinal in child.operator_ordinals
                )
                expected_boundary = (
                    PartitionBoundary.ROOT
                    if not selected
                    else _partition_boundary(previous_operator, selected[0])
                )
                if child.boundary != expected_boundary:
                    raise ValueError("task boundary does not match its leaf operator facts")
                if selected:
                    previous_operator = selected[-1]
            expected_effects = tuple(
                dict.fromkeys(
                    effect
                    for operator in leaf_artifact.operators
                    for effect in operator.effects
                )
            ) or (Effect.PURE,)
            if effects != expected_effects:
                raise ValueError("captured root effects must match its leaf operators")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "children", children)


@dataclass(frozen=True, slots=True)
class CaptureAttachment:
    node_key: NodeKey
    artifact_key: ArtifactKey
    bindings: tuple[PortBindingDesc, ...] = ()
    anchor_node_key: NodeKey | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.node_key), NodeKey):
            raise TypeError("CaptureAttachment.node_key must be NodeKey")
        if not isinstance(cast(object, self.artifact_key), ArtifactKey):
            raise TypeError("CaptureAttachment.artifact_key must be ArtifactKey")
        bindings = cast(
            tuple[PortBindingDesc, ...],
            _typed_tuple(self.bindings, PortBindingDesc, "CaptureAttachment.bindings"),
        )
        if len(set(bindings)) != len(bindings):
            raise ValueError("CaptureAttachment contains duplicate bindings")
        anchor_node_key = cast(object, self.anchor_node_key)
        if anchor_node_key is not None and not isinstance(anchor_node_key, NodeKey):
            raise TypeError("CaptureAttachment.anchor_node_key must be NodeKey or None")
        object.__setattr__(self, "bindings", bindings)


@dataclass(frozen=True, slots=True)
class CaptureLeaseDesc:
    value_id: str
    live_start: NodeKey
    live_end: NodeKey
    reader_count: int
    materialized: bool = False

    def __post_init__(self) -> None:
        value_id = cast(object, self.value_id)
        if not isinstance(value_id, str) or not value_id:
            raise TypeError("CaptureLeaseDesc.value_id must be non-empty str")
        if not isinstance(cast(object, self.live_start), NodeKey) or not isinstance(
            cast(object, self.live_end), NodeKey
        ):
            raise TypeError("CaptureLeaseDesc live range endpoints must be NodeKey")
        reader_count = cast(object, self.reader_count)
        if not isinstance(reader_count, int) or isinstance(reader_count, bool):
            raise TypeError("CaptureLeaseDesc.reader_count must be int")
        if self.reader_count < 0:
            raise ValueError("CaptureLeaseDesc.reader_count must be non-negative")
        if self.materialized:
            raise ValueError("Phase 1 capture leases cannot be materialized")


@dataclass(frozen=True, slots=True)
class CaptureDataCentricIR:
    leases: tuple[CaptureLeaseDesc, ...] = ()
    moves: tuple[()] = ()

    def __post_init__(self) -> None:
        leases = cast(
            tuple[CaptureLeaseDesc, ...],
            _typed_tuple(self.leases, CaptureLeaseDesc, "CaptureDataCentricIR.leases"),
        )
        if self.moves:
            raise ValueError("Phase 1 capture data IR must not contain Move")
        if len({lease.value_id for lease in leases}) != len(leases):
            raise ValueError("CaptureDataCentricIR contains duplicate lease values")
        object.__setattr__(self, "leases", leases)


@dataclass(frozen=True, slots=True)
class CaptureHierarchy:
    roots: tuple[RootFlowDesc, ...]
    attachments: tuple[CaptureAttachment, ...]
    data_centric: CaptureDataCentricIR
    schema_version: int = 2

    def __post_init__(self) -> None:
        roots = cast(
            tuple[RootFlowDesc, ...],
            _typed_tuple(self.roots, RootFlowDesc, "CaptureHierarchy.roots"),
        )
        attachments = cast(
            tuple[CaptureAttachment, ...],
            _typed_tuple(self.attachments, CaptureAttachment, "CaptureHierarchy.attachments"),
        )
        if self.schema_version != 2:
            raise ValueError("CaptureHierarchy.schema_version must be 2")
        if not roots:
            raise ValueError("CaptureHierarchy requires at least one captured root")
        root_artifacts = {root.artifact_key for root in roots}
        attachment_artifacts = {attachment.artifact_key for attachment in attachments}
        if len(root_artifacts) != len(roots):
            raise ValueError("CaptureHierarchy contains duplicate root artifacts")
        if len(attachment_artifacts) != len(attachments):
            raise ValueError("CaptureHierarchy requires one attachment per artifact")
        if root_artifacts != attachment_artifacts:
            raise ValueError("every capture root requires one explicit attachment")
        if not isinstance(cast(object, self.data_centric), CaptureDataCentricIR):
            raise TypeError("CaptureHierarchy.data_centric must be CaptureDataCentricIR")
        task_positions = {
            task.node_key: (root_index, position)
            for root_index, root in enumerate(roots)
            for position, task in enumerate(root.children)
        }
        for lease in self.data_centric.leases:
            if lease.live_start not in task_positions or lease.live_end not in task_positions:
                raise ValueError("capture lease references an unknown task node")
            start_root, start_position = task_positions[lease.live_start]
            end_root, end_position = task_positions[lease.live_end]
            if start_root != end_root or start_position > end_position:
                raise ValueError("capture lease live range is not topologically ordered")
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "attachments", attachments)

    @property
    def root(self) -> RootFlowDesc:
        if len(self.roots) != 1:
            raise ValueError("CaptureHierarchy.root requires exactly one captured flow")
        return self.roots[0]

    def to_dict(self) -> dict[str, object]:
        def region_key(key: RegionKey) -> str:
            return key.value

        def artifact_key(key: ArtifactKey) -> dict[str, object]:
            return {
                "namespace": key.namespace,
                "name": key.name,
                "version": key.version,
                "digest": key.digest,
            }

        def port_ref(port: PortId | PortAddressDesc) -> dict[str, object]:
            if isinstance(port, PortId):
                return {
                    "type": "port_id",
                    "namespace": port.namespace,
                    "name": port.name,
                    "version": port.version,
                }
            return {
                "type": "port_address",
                "kind": port.kind,
                "locator_type": port.locator_type,
                "locator_fields": [
                    {
                        "path": list(field.path),
                        "value_type": field.value_type,
                        "value": field.value,
                    }
                    for field in port.locator_fields
                ],
            }

        def binding_desc(binding: PortBindingDesc) -> dict[str, object]:
            return {
                "port": port_ref(binding.port),
                "direction": binding.direction.value,
                "value_path": list(binding.value_path),
            }

        def operator_desc(operator: LeafOperatorDesc) -> dict[str, object]:
            return {
                "ordinal": operator.ordinal,
                "namespace": operator.namespace,
                "name": operator.name,
                "overload": operator.overload,
                "inputs": list(operator.inputs),
                "outputs": list(operator.outputs),
                "module_path": list(operator.module_path),
                "operator_kind": operator.operator_kind.value,
                "effects": [effect.value for effect in operator.effects],
                "capabilities": [item.value for item in operator.capabilities],
                "attributes": [
                    {
                        "path": list(attribute.path),
                        "kind": attribute.kind.value,
                        "value": attribute.value,
                    }
                    for attribute in operator.attributes
                ],
            }

        def leaf_desc(task: TaskRegionDesc) -> dict[str, object]:
            leaf = task.children[0]
            artifact = leaf.leaf_artifact
            return {
                "kind": leaf.kind.value,
                "artifact": leaf.artifact_key.value,
                "artifact_key": artifact_key(leaf.artifact_key),
                "terminal_ordinal": None if artifact is None else artifact.terminal_ordinal,
                "operators": (
                    []
                    if artifact is None
                    else [
                        operator_desc(operator)
                        for operator in artifact.operators
                        if operator.ordinal in task.operator_ordinals
                    ]
                ),
            }

        return {
            "schema_version": self.schema_version,
            "roots": [
                {
                    "key": region_key(root.key),
                    "artifact": root.artifact_key.value,
                    "artifact_key": artifact_key(root.artifact_key),
                    "status": root.status.value,
                    "source": {
                        "file": root.source_span.file,
                        "line": root.source_span.start_line,
                        "start_line": root.source_span.start_line,
                        "start_column": root.source_span.start_column,
                        "end_line": root.source_span.end_line,
                        "end_column": root.source_span.end_column,
                    },
                    "inputs": [binding_desc(binding) for binding in root.inputs],
                    "outputs": [binding_desc(binding) for binding in root.outputs],
                    "shape_guard": [
                        {
                            "value_path": list(expression.value_path),
                            "dimension": expression.dimension,
                            "kind": expression.kind.value,
                            "lower_bound": expression.lower_bound,
                            "upper_bound": expression.upper_bound,
                        }
                        for expression in root.shape_guard.expressions
                    ],
                    "values": [
                        {
                            "value_id": value.value_id,
                            "version": value.version,
                            "shape": list(value.shape),
                            "dtype": f"{value.dtype.namespace}.{value.dtype.name}",
                            "device": {
                                "kind": value.device.kind.value,
                                "index": value.device.index,
                            },
                            "layout": value.layout.value,
                            "stride": list(value.stride),
                            "producer": value.use_def.producer,
                            "inputs": list(value.use_def.inputs),
                            "users": list(value.use_def.users),
                            "alias_set": value.alias_set,
                            "effect": value.effect.value,
                        }
                        for value in root.values
                    ],
                    "effects": [effect.value for effect in root.effects],
                    "diagnostics": [
                        {
                            "code": diagnostic.code.value,
                            "message": diagnostic.message,
                            "file": diagnostic.source_span.file,
                            "line": diagnostic.source_span.start_line,
                            "detail": diagnostic.detail,
                        }
                        for diagnostic in root.diagnostics
                    ],
                    "tasks": [
                        {
                            "key": region_key(task.key),
                            "node_key": str(task.node_key),
                            "executor": str(task.executor),
                            "boundary": task.boundary.value,
                            "module_path": list(task.module_path),
                            "operators": list(task.operator_ordinals),
                            "inputs": list(task.input_values),
                            "outputs": list(task.output_values),
                            "effects": [effect.value for effect in task.effects],
                            "capabilities": [item.value for item in task.capabilities],
                            "dependencies": [region_key(item) for item in task.dependencies],
                            "leaf": leaf_desc(task),
                        }
                        for task in root.children
                    ],
                }
                for root in self.roots
            ],
            "attachments": [
                {
                    "artifact": attachment.artifact_key.value,
                    "artifact_key": artifact_key(attachment.artifact_key),
                    "node_key": str(attachment.node_key),
                    "anchor_node_key": (
                        None
                        if attachment.anchor_node_key is None
                        else str(attachment.anchor_node_key)
                    ),
                    "bindings": [binding_desc(binding) for binding in attachment.bindings],
                }
                for attachment in self.attachments
            ],
            "leases": [
                {
                    "value_id": lease.value_id,
                    "live_start": str(lease.live_start),
                    "live_end": str(lease.live_end),
                    "reader_count": lease.reader_count,
                    "materialized": lease.materialized,
                }
                for lease in self.data_centric.leases
            ],
            "moves": [],
        }


def _partition_boundary(
    previous: LeafOperatorDesc | None,
    current: LeafOperatorDesc,
) -> PartitionBoundary:
    if current.operator_kind == OperatorKind.CUSTOM:
        return PartitionBoundary.CUSTOM_OP
    if previous is None:
        return PartitionBoundary.ROOT
    if previous.module_path != current.module_path:
        return PartitionBoundary.MODULE
    if previous.effects != current.effects:
        return PartitionBoundary.EFFECT
    if previous.capabilities != current.capabilities:
        return PartitionBoundary.CAPABILITY
    return PartitionBoundary.ROOT


def _partition_operators(
    operators: tuple[LeafOperatorDesc, ...],
) -> tuple[tuple[PartitionBoundary, tuple[LeafOperatorDesc, ...]], ...]:
    groups: list[tuple[PartitionBoundary, list[LeafOperatorDesc]]] = []
    previous: LeafOperatorDesc | None = None
    previous_facts: tuple[object, ...] | None = None
    for operator in operators:
        facts: tuple[object, ...] = (
            operator.module_path,
            operator.operator_kind,
            operator.effects,
            operator.capabilities,
        )
        split = previous_facts is None or facts != previous_facts or operator.operator_kind == OperatorKind.CUSTOM
        if split:
            groups.append((_partition_boundary(previous, operator), [operator]))
        else:
            groups[-1][1].append(operator)
        previous = operator
        previous_facts = facts
    return tuple((boundary, tuple(group)) for boundary, group in groups)


def _task_values(
    capture: CapturedFlow,
    task_ordinals: frozenset[int],
    ordinal_to_task: dict[int, RegionKey],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[RegionKey, ...]]:
    leaf = cast(LeafArtifactDesc, capture.leaf_artifact)
    inputs: list[str] = []
    outputs: list[str] = []
    dependencies: set[RegionKey] = set()
    leaf_outputs = set(leaf.output_values)
    for value in capture.values:
        value_key = f"{value.value_id}@{value.version}"
        producer_task = ordinal_to_task.get(value.use_def.producer)
        user_tasks = {
            ordinal_to_task[user]
            for user in value.use_def.users
            if user in ordinal_to_task
        }
        current_key = ordinal_to_task[next(iter(task_ordinals))]
        if current_key in user_tasks and producer_task != current_key:
            inputs.append(value_key)
            if producer_task is not None:
                dependencies.add(producer_task)
        if producer_task == current_key and (
            any(user != current_key for user in user_tasks) or value_key in leaf_outputs
        ):
            outputs.append(value_key)
    return tuple(inputs), tuple(outputs), tuple(sorted(dependencies, key=str))


def _validate_capture_ssa(capture: CapturedFlow) -> None:
    if capture.status != CaptureStatus.CAPTURED:
        return
    leaf = cast(LeafArtifactDesc, capture.leaf_artifact)
    value_keys = tuple(f"{value.value_id}@{value.version}" for value in capture.values)
    if len(set(value_keys)) != len(value_keys):
        raise ValueError("captured flow contains duplicate value identities")
    known_values = set(value_keys)
    if not set((*leaf.input_values, *leaf.output_values)).issubset(known_values):
        raise ValueError("leaf input/output references an unknown captured value")
    value_ordinals = {value.use_def.producer for value in capture.values}
    operator_ordinals = {operator.ordinal for operator in leaf.operators}
    known_user_ordinals = value_ordinals | operator_ordinals
    for value, value_key in zip(capture.values, value_keys, strict=True):
        if not set(value.use_def.inputs).issubset(value_ordinals):
            raise ValueError("captured value use-def references an unknown input value")
        unknown_users = set(value.use_def.users) - known_user_ordinals
        if unknown_users and (
            unknown_users != {leaf.terminal_ordinal} or value_key not in leaf.output_values
        ):
            raise ValueError("captured value use-def references an unknown user")
    for operator in leaf.operators:
        if not set(operator.inputs).issubset(value_ordinals):
            raise ValueError("leaf operator references unsupported non-Tensor SSA input")
        if not set(operator.outputs).issubset(known_values):
            raise ValueError("leaf operator references an unknown output value")


def _successful_root(capture: CapturedFlow) -> RootFlowDesc:
    leaf = cast(LeafArtifactDesc, capture.leaf_artifact)
    partitions = _partition_operators(leaf.operators)
    if not partitions:
        partitions = ((PartitionBoundary.ROOT, ()),)
    root_key = RegionKey(capture.artifact_key, RegionKind.ROOT, 0)
    ordinal_to_task: dict[int, RegionKey] = {}
    for task_ordinal, (_boundary, operators) in enumerate(partitions):
        task_key = RegionKey(capture.artifact_key, RegionKind.TASK, task_ordinal)
        for operator in operators:
            ordinal_to_task[operator.ordinal] = task_key
    tasks: list[TaskRegionDesc] = []
    for task_ordinal, (boundary, operators) in enumerate(partitions):
        task_key = RegionKey(capture.artifact_key, RegionKind.TASK, task_ordinal)
        operator_ordinals = tuple(operator.ordinal for operator in operators)
        if operators:
            input_values, output_values, dependencies = _task_values(
                capture,
                frozenset(operator_ordinals),
                ordinal_to_task,
            )
            if task_ordinal > 0:
                dependencies = tuple(
                    sorted(
                        {
                            *dependencies,
                            RegionKey(
                                capture.artifact_key,
                                RegionKind.TASK,
                                task_ordinal - 1,
                            ),
                        },
                        key=str,
                    )
                )
            module_path = operators[0].module_path
            effects = operators[0].effects
            capabilities = operators[0].capabilities
        else:
            input_values = leaf.input_values
            output_values = leaf.output_values
            dependencies = ()
            module_path = ()
            effects = capture.effects
            capabilities = (Capability.CAPTURE_SAFE,)
        leaf_region = LeafRegionDesc(
            key=RegionKey(capture.artifact_key, RegionKind.LEAF, task_ordinal),
            artifact_key=capture.artifact_key,
            kind=RegionKind.LEAF,
            operator_ordinals=operator_ordinals,
            leaf_artifact=leaf,
        )
        tasks.append(
            TaskRegionDesc(
                key=task_key,
                node_key=NodeKey("capture_task", capture.artifact_key.value, task_ordinal),
                executor=ExecutorKey("capture", capture.artifact_key.value),
                boundary=boundary,
                module_path=module_path,
                operator_ordinals=operator_ordinals,
                input_values=input_values,
                output_values=output_values,
                effects=effects,
                capabilities=capabilities,
                dependencies=dependencies,
                children=(leaf_region,),
            )
        )
    return RootFlowDesc(
        key=root_key,
        artifact_key=capture.artifact_key,
        source_span=capture.source_span,
        status=capture.status,
        shape_guard=capture.shape_guard,
        inputs=capture.inputs,
        outputs=capture.outputs,
        values=capture.values,
        effects=capture.effects,
        diagnostics=capture.diagnostics,
        children=tuple(tasks),
    )


def _opaque_root(capture: CapturedFlow) -> RootFlowDesc:
    task_key = RegionKey(capture.artifact_key, RegionKind.TASK, 0)
    leaf = LeafRegionDesc(
        key=RegionKey(capture.artifact_key, RegionKind.OPAQUE, 0),
        artifact_key=capture.artifact_key,
        kind=RegionKind.OPAQUE,
    )
    task = TaskRegionDesc(
        key=task_key,
        node_key=NodeKey("capture_task", capture.artifact_key.value, 0),
        executor=ExecutorKey("legacy_capture", capture.artifact_key.value),
        boundary=PartitionBoundary.OPAQUE,
        module_path=(),
        operator_ordinals=(),
        input_values=(),
        output_values=(),
        effects=(Effect.OPAQUE,),
        capabilities=(Capability.EAGER_ONLY,),
        dependencies=(),
        children=(leaf,),
    )
    return RootFlowDesc(
        key=RegionKey(capture.artifact_key, RegionKind.ROOT, 0),
        artifact_key=capture.artifact_key,
        source_span=capture.source_span,
        status=capture.status,
        shape_guard=capture.shape_guard,
        inputs=capture.inputs,
        outputs=capture.outputs,
        values=capture.values,
        effects=capture.effects,
        diagnostics=capture.diagnostics,
        children=(task,),
    )


def build_capture_root(capture: CapturedFlow) -> RootFlowDesc:
    """Build one pure root/task/leaf view without attaching runtime plans."""

    if not isinstance(cast(object, capture), CapturedFlow):
        raise TypeError("build_capture_root.capture must be CapturedFlow")
    _validate_capture_ssa(capture)
    if capture.status == CaptureStatus.CAPTURED:
        return _successful_root(capture)
    return _opaque_root(capture)


def _capture_leases(
    captures: tuple[CapturedFlow, ...],
    roots: tuple[RootFlowDesc, ...],
) -> tuple[CaptureLeaseDesc, ...]:
    root_by_artifact = {root.artifact_key: root for root in roots}
    leases: list[CaptureLeaseDesc] = []
    for capture in captures:
        if capture.status != CaptureStatus.CAPTURED:
            continue
        task_by_operator = {
            ordinal: task
            for task in root_by_artifact[capture.artifact_key].children
            for ordinal in task.operator_ordinals
        }
        for value in capture.values:
            producer_task = task_by_operator.get(value.use_def.producer)
            user_tasks = tuple(
                dict.fromkeys(
                    task_by_operator[user]
                    for user in value.use_def.users
                    if user in task_by_operator
                )
            )
            start_task = producer_task if producer_task is not None else (user_tasks[0] if user_tasks else None)
            end_task = user_tasks[-1] if user_tasks else producer_task
            if start_task is None or end_task is None:
                continue
            leases.append(
                CaptureLeaseDesc(
                    value_id=f"{capture.artifact_key.value}:{value.value_id}@{value.version}",
                    live_start=start_task.node_key,
                    live_end=end_task.node_key,
                    reader_count=len(user_tasks),
                )
            )
    return tuple(leases)


def _validate_attachments(
    captures: tuple[CapturedFlow, ...],
    attachments: tuple[CaptureAttachment, ...],
    dispatch_plan: DispatchPlan,
    execution_plan: ExecutionPlan,
    semantic: SemanticGraph | None,
) -> None:
    capture_by_artifact = {capture.artifact_key: capture for capture in captures}
    flow_by_node = {
        NodeKey.consumer_flow(target.flow_id.rsplit("#", 1)[0], int(target.flow_id.rsplit("#", 1)[1])): target.flow
        for target in dispatch_plan.flow_targets()
    }
    producer_nodes = {
        NodeKey.producer(planned.entry.producer_id)
        for planned in execution_plan.producers
    }
    known_nodes = set(flow_by_node) | producer_nodes
    if semantic is not None:
        known_nodes = {node.node_key for node in semantic.nodes}
    for attachment in attachments:
        if attachment.artifact_key not in capture_by_artifact:
            raise ValueError("CaptureAttachment references an unknown artifact")
        if attachment.node_key not in known_nodes:
            raise ValueError("CaptureAttachment references an unknown runtime node")
        if attachment.node_key not in flow_by_node:
            raise ValueError("CaptureAttachment must target an existing consumer flow node")
        capture = capture_by_artifact[attachment.artifact_key]
        expected_bindings = (*capture.inputs, *capture.outputs)
        if len(expected_bindings) != len(attachment.bindings) or set(expected_bindings) != set(attachment.bindings):
            raise ValueError("CaptureAttachment must bind every capture port exactly")
        for binding in attachment.bindings:
            if isinstance(binding.port, PortId):
                producer_id = execution_plan.port_sources.get(binding.port)
                if producer_id is None or NodeKey.producer(producer_id) not in producer_nodes:
                    raise ValueError("CaptureAttachment references an unknown producer PortId")
                continue
            flow = flow_by_node.get(attachment.node_key)
            if flow is None:
                raise ValueError("PortAddressDesc attachment must target a consumer flow")
            declared = (
                tuple(port_address_desc(item) for item in flow.reads)
                if binding.direction == PortBindingDirection.INPUT
                else tuple(port_address_desc(item) for item in flow.writes)
            )
            if binding.port not in declared:
                raise ValueError("CaptureAttachment PortAddressDesc is not declared by the target flow")
        if attachment.anchor_node_key is not None:
            if semantic is None:
                raise ValueError("anchor attachment requires a concrete semantic graph")
            anchor_nodes = {
                node.node_key for node in semantic.nodes if node.kind == NodeKind.ANCHOR
            }
            if attachment.anchor_node_key not in anchor_nodes:
                raise ValueError("CaptureAttachment references an unknown anchor")
            if (
                attachment.anchor_node_key,
                attachment.node_key,
                EdgeKind.ANCHOR,
            ) not in semantic.edges:
                raise ValueError("CaptureAttachment anchor does not attach to the target node")


def build_hierarchy(
    *,
    captures: tuple[CapturedFlow, ...],
    attachments: tuple[CaptureAttachment, ...],
    dispatch_plan: DispatchPlan,
    execution_plan: ExecutionPlan,
    semantic: SemanticGraph | None = None,
) -> CaptureHierarchy:
    """Build and validate the pure-data capture sidecar."""
    typed_captures = cast(
        tuple[CapturedFlow, ...], _typed_tuple(captures, CapturedFlow, "build_hierarchy.captures")
    )
    typed_attachments = cast(
        tuple[CaptureAttachment, ...],
        _typed_tuple(attachments, CaptureAttachment, "build_hierarchy.attachments"),
    )
    if len({capture.artifact_key for capture in typed_captures}) != len(typed_captures):
        raise ValueError("build_hierarchy captures contain duplicate artifacts")
    _validate_attachments(
        typed_captures,
        typed_attachments,
        dispatch_plan,
        execution_plan,
        semantic,
    )
    roots = tuple(build_capture_root(capture) for capture in typed_captures)
    return CaptureHierarchy(
        roots=roots,
        attachments=typed_attachments,
        data_centric=CaptureDataCentricIR(leases=_capture_leases(typed_captures, roots)),
    )


def apply_attachment_live_ranges(
    data: DataCentricIR,
    hierarchy: CaptureHierarchy,
    execution_plan: ExecutionPlan,
) -> DataCentricIR:
    """Project explicit producer-port readers into metadata-only Phase 1 leases."""
    node_position = {node.node_key: index for index, node in enumerate(data.nodes)}
    readers_by_port: dict[PortId, set[NodeKey]] = {}
    for attachment in hierarchy.attachments:
        for binding in attachment.bindings:
            if binding.direction == PortBindingDirection.INPUT and isinstance(binding.port, PortId):
                readers_by_port.setdefault(binding.port, set()).add(attachment.node_key)
    if not readers_by_port:
        return data
    value_by_port = {
        value.port: value
        for value in data.values
        if isinstance(value.port, PortId)
    }
    leases_by_key = {lease.key: lease for lease in data.leases}
    producer_lease_keys: dict[NodeKey, set[str]] = {}
    for port, readers in readers_by_port.items():
        producer_id = execution_plan.port_sources[port]
        producer_key = NodeKey.producer(producer_id)
        value = value_by_port[port]
        live_end = max(readers, key=lambda key: node_position[key])
        lease = LeaseDesc(
            value=value,
            storage="backend_owned",
            live_start=str(producer_key),
            live_end=str(live_end),
            reader_count=len(readers),
            stable_address=False,
            materialized=False,
        )
        leases_by_key[lease.key] = lease
        producer_lease_keys.setdefault(producer_key, set()).add(lease.key)
    nodes = tuple(
        replace(
            node,
            lease_keys=tuple(
                dict.fromkeys((*node.lease_keys, *sorted(producer_lease_keys.get(node.node_key, set()))))
            ),
        )
        for node in data.nodes
    )
    return replace(
        data,
        nodes=nodes,
        leases=tuple(sorted(leases_by_key.values(), key=lambda lease: lease.key)),
    )


__all__ = [
    "CaptureAttachment",
    "CaptureDataCentricIR",
    "CaptureHierarchy",
    "CaptureLeaseDesc",
    "LeafRegionDesc",
    "PartitionBoundary",
    "RegionKey",
    "RegionKind",
    "RootFlowDesc",
    "TaskRegionDesc",
    "apply_attachment_live_ranges",
    "build_capture_root",
    "build_hierarchy",
]
