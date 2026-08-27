"""Legacy capture-region lowering for single-device WorkGraph execution.

The frozen descriptor vocabulary itself now lives in
``tllm.contracts.physical_plan`` and is re-exported here so existing
legacy import paths keep identical class identity.  What remains in
this module is the legacy shadow-plan lowering (capture regions and
task partitions) that only the residual Program/workgraph path uses.

This module is a control-plane boundary.  It deliberately contains no Tensor,
CUDA stream/event, CUDAGraph, callback, request id, or backend runtime handle.
Those objects belong to the Python physical executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

from tllm.contracts.physical_plan import (
    AnchorPlacementDesc,
    ArtifactKey,
    BufferBucketDesc,
    BufferBucketKey,
    BufferOwnershipKind,
    CapturePolicy,
    ExecutorKey,
    FallbackPolicy,
    FenceDependencyDesc,
    FenceKey,
    FenceKind,
    NodeKey,
    PhysicalFlowPlan,
    PhysicalNodePlan,
    StreamClass,
)
from tllm.workgraph.frontend import (
    Capability,
    CaptureStatus,
    ExternContract,
)
from tllm.workgraph.hierarchy import CaptureAttachment, RootFlowDesc
from tllm.workgraph.ir import (
    EdgeKind,
    ExecutionNodeDesc,
)

if TYPE_CHECKING:
    from tllm.workgraph.shadow import ShadowParityPlan


_T = TypeVar("_T")


def _instance(value: object, expected: type[_T], label: str) -> _T:
    if not isinstance(value, expected):
        raise TypeError(f"{label} must be {expected.__name__}")
    return value


@dataclass(frozen=True, slots=True)
class PhysicalCaptureRegion:
    """One explicitly attached capture root and its executable task plan."""

    attachment: CaptureAttachment
    plan: PhysicalFlowPlan
    executor_contract: ExternContract | None = None

    def __post_init__(self) -> None:
        attachment = _instance(
            self.attachment,
            CaptureAttachment,
            "PhysicalCaptureRegion.attachment",
        )
        plan = _instance(self.plan, PhysicalFlowPlan, "PhysicalCaptureRegion.plan")
        contract = self.executor_contract
        if contract is not None:
            contract = _instance(
                contract,
                ExternContract,
                "PhysicalCaptureRegion.executor_contract",
            )
        if not plan.nodes:
            raise ValueError("PhysicalCaptureRegion.plan must contain a capture task")
        if any(node.node_key.namespace != "capture_task" for node in plan.nodes):
            raise ValueError("PhysicalCaptureRegion.plan may contain only capture tasks")
        if any(node.artifact_key != attachment.artifact_key for node in plan.nodes):
            raise ValueError("PhysicalCaptureRegion plan artifact differs from its attachment")
        if contract is not None:
            attached_ports = {
                (binding.port, binding.direction)
                for binding in attachment.bindings
            }
            implementation_ports = {
                (binding.port, binding.direction)
                for binding in contract.bindings
            }
            if implementation_ports != attached_ports:
                raise ValueError(
                    "physical region executor contract must bind every attached port"
                )
        object.__setattr__(self, "attachment", attachment)
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "executor_contract", contract)


@dataclass(frozen=True, slots=True)
class PhysicalRegionExecutorDesc:
    """Frozen external implementation selected for one attached region."""

    artifact_key: ArtifactKey
    executor: ExecutorKey
    contract: ExternContract

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_key",
            _instance(
                self.artifact_key,
                ArtifactKey,
                "PhysicalRegionExecutorDesc.artifact_key",
            ),
        )
        object.__setattr__(
            self,
            "executor",
            _instance(
                self.executor,
                ExecutorKey,
                "PhysicalRegionExecutorDesc.executor",
            ),
        )
        object.__setattr__(
            self,
            "contract",
            _instance(
                self.contract,
                ExternContract,
                "PhysicalRegionExecutorDesc.contract",
            ),
        )


def _fence_kind(edge_kind: EdgeKind) -> FenceKind:
    if edge_kind == EdgeKind.LEASE:
        return FenceKind.RETIRE
    if edge_kind == EdgeKind.EFFECT:
        return FenceKind.COMPLETION
    return FenceKind.READY


def _static_shape(shape: tuple[str, ...]) -> tuple[int, ...] | None:
    dimensions: list[int] = []
    for expression in shape:
        try:
            dimension = int(expression)
        except ValueError:
            return None
        if dimension < 1:
            return None
        dimensions.append(dimension)
    return tuple(dimensions) if dimensions else None


def _capture_policy(root: RootFlowDesc) -> CapturePolicy:
    if root.status != CaptureStatus.CAPTURED:
        return CapturePolicy.DISABLED
    admitted = {Capability.CAPTURE_SAFE, Capability.REPLAY_SAFE}
    if any(
        capability not in admitted
        for task in root.children
        for capability in task.capabilities
    ):
        return CapturePolicy.DISABLED
    return CapturePolicy.PREFER


def _execution_order(
    nodes: tuple[ExecutionNodeDesc, ...],
    edges: tuple[tuple[NodeKey, NodeKey, EdgeKind], ...],
) -> tuple[ExecutionNodeDesc, ...]:
    by_key = {node.node_key: node for node in nodes}
    source_order = {node.node_key: ordinal for ordinal, node in enumerate(nodes)}
    successors: dict[NodeKey, list[NodeKey]] = {node.node_key: [] for node in nodes}
    indegree: dict[NodeKey, int] = {node.node_key: 0 for node in nodes}
    seen_edges: set[tuple[NodeKey, NodeKey]] = set()
    for producer, consumer, _kind in edges:
        edge = (producer, consumer)
        if producer not in by_key or consumer not in by_key or edge in seen_edges:
            continue
        seen_edges.add(edge)
        successors[producer].append(consumer)
        indegree[consumer] += 1
    ready = sorted(
        (key for key, count in indegree.items() if count == 0),
        key=source_order.__getitem__,
    )
    ordered: list[ExecutionNodeDesc] = []
    while ready:
        key = ready.pop(0)
        ordered.append(by_key[key])
        for successor in sorted(successors[key], key=source_order.__getitem__):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort(key=source_order.__getitem__)
    if len(ordered) != len(nodes):
        raise ValueError("physical execution subgraph contains a cycle")
    return tuple(ordered)


def compile_physical_flow_plan(plan: ShadowParityPlan) -> PhysicalFlowPlan:
    """Lower one verified shadow plan into frozen single-device descriptors."""

    from tllm.workgraph.shadow import ShadowParityPlan
    from tllm.workgraph.lowering import WorkGraphError

    raw_plan = cast(object, plan)
    if not isinstance(raw_plan, ShadowParityPlan):
        raise TypeError("compile_physical_flow_plan.plan must be ShadowParityPlan")
    plan = raw_plan
    if plan.epoch < 1:
        raise ValueError("physical plan requires a compiled workgraph epoch")
    if plan.backend is None or plan.backend.graph_mode == "full":
        raise WorkGraphError(
            "WG120_UNSUPPORTED_GRAPH_MODE",
            "physical execution supports only eager and piecewise graph modes",
        )

    artifact_by_node: dict[NodeKey, ArtifactKey] = {}
    capture_policy_by_node: dict[NodeKey, CapturePolicy] = {}
    buckets: list[BufferBucketDesc] = []
    bucket_by_node: dict[NodeKey, BufferBucketKey] = {}
    legacy_node_keys = {
        node.node_key
        for node in plan.data.nodes
        if node.executor is not None and node.executor.namespace == "legacy"
    }
    if plan.hierarchy is not None:
        roots = {root.artifact_key: root for root in plan.hierarchy.roots}
        for attachment in plan.hierarchy.attachments:
            existing_artifact = artifact_by_node.get(attachment.node_key)
            if existing_artifact is not None and existing_artifact != attachment.artifact_key:
                raise ValueError("one physical node cannot own multiple capture artifacts")
            artifact_by_node[attachment.node_key] = attachment.artifact_key
            root = roots[attachment.artifact_key]
            policy = (
                CapturePolicy.DISABLED
                if attachment.node_key in legacy_node_keys
                else _capture_policy(root)
            )
            capture_policy_by_node[attachment.node_key] = policy
            if policy == CapturePolicy.DISABLED:
                continue
            leaf = root.children[0].children[0]
            leaf_artifact = leaf.leaf_artifact
            if leaf_artifact is None or len(leaf_artifact.output_values) != 1:
                continue
            output_id = leaf_artifact.output_values[0]
            output_value = next(
                (
                    value
                    for value in root.values
                    if f"{value.value_id}@{value.version}" == output_id
                ),
                None,
            )
            if output_value is None:
                continue
            shape = _static_shape(output_value.shape)
            if shape is None:
                continue
            bucket_key = BufferBucketKey(
                namespace="capture",
                name=f"artifact-{attachment.artifact_key.digest[:16]}",
                ordinal=len(buckets),
            )
            buckets.append(
                BufferBucketDesc(
                    key=bucket_key,
                    shape=shape,
                    dtype=output_value.dtype,
                    device=output_value.device,
                    slots=1,
                )
            )
            bucket_by_node[attachment.node_key] = bucket_key

    execution_nodes = _execution_order(
        tuple(node for node in plan.data.nodes if node.executor is not None),
        plan.data.edges,
    )
    physical_keys = {node.node_key for node in execution_nodes}
    incoming: dict[NodeKey, list[FenceDependencyDesc]] = {
        node.node_key: [] for node in execution_nodes
    }
    for ordinal, (producer, consumer, edge_kind) in enumerate(plan.data.edges):
        if producer not in physical_keys or consumer not in physical_keys:
            continue
        incoming[consumer].append(
            FenceDependencyDesc(
                fence=FenceKey("workgraph", "dependency", ordinal),
                producer=producer,
                consumer=consumer,
                kind=_fence_kind(edge_kind),
            )
        )

    nodes: list[PhysicalNodePlan] = []
    for node in execution_nodes:
        assert node.executor is not None
        artifact_key = artifact_by_node.get(node.node_key)
        capture_policy = (
            capture_policy_by_node.get(node.node_key, CapturePolicy.DISABLED)
            if node.executor.namespace != "legacy"
            else CapturePolicy.DISABLED
        )
        stream = (
            StreamClass.SIDE_TRAIN
            if node.node_key.namespace == "consumer_flow"
            else StreamClass.DEFAULT
        )
        nodes.append(
            PhysicalNodePlan(
                node_key=node.node_key,
                executor=node.executor,
                artifact_key=artifact_key,
                stream=stream,
                fences=tuple(incoming[node.node_key]),
                buffer_bucket=bucket_by_node.get(node.node_key),
                capture_policy=capture_policy,
                fallback_policy=FallbackPolicy.EAGER,
            )
        )
    physical = PhysicalFlowPlan(
        plan_epoch=plan.epoch,
        nodes=tuple(nodes),
        buffer_buckets=tuple(buckets),
    )
    try:
        from tllm import core as _core

        _core.validate_physical_flow_plan(physical)
    except (TypeError, ValueError) as exc:
        raise WorkGraphError(
            "WG400_RUST_PHYSICAL_VALIDATION",
            "Rust physical-plan descriptor validation failed",
        ) from exc
    return physical


def compile_physical_capture_region(
    plan: ShadowParityPlan,
    artifact_key: ArtifactKey,
    executor_desc: PhysicalRegionExecutorDesc | None = None,
    *,
    allow_task_partition: bool = False,
) -> PhysicalCaptureRegion:
    """Lower one complete hierarchy root into an independently runnable plan.

    Compatibility producer/consumer nodes are intentionally absent.  They
    remain authoritative legacy boundaries while this plan owns only the
    explicitly migrated capture task.

    ``allow_task_partition`` is the Program-lowering escape hatch: stage
    slicing (``tllm.program``) produces one capture task per hook stage, so
    Program callers opt in and receive one physical node per task in
    dependency order. The legacy path keeps requiring exactly one complete
    capture task (``WG421_UNSUPPORTED_CAPTURE_PARTITION``).
    """

    from tllm.workgraph.shadow import ShadowParityPlan
    from tllm.workgraph.lowering import WorkGraphError

    raw_plan = cast(object, plan)
    if not isinstance(raw_plan, ShadowParityPlan):
        raise TypeError("compile_physical_capture_region.plan must be ShadowParityPlan")
    key = _instance(
        artifact_key,
        ArtifactKey,
        "compile_physical_capture_region.artifact_key",
    )
    implementation = executor_desc
    if implementation is not None:
        implementation = _instance(
            implementation,
            PhysicalRegionExecutorDesc,
            "compile_physical_capture_region.executor_desc",
        )
        if implementation.artifact_key != key:
            raise ValueError("physical region executor artifact does not match capture")
    hierarchy = raw_plan.hierarchy
    if hierarchy is None:
        raise WorkGraphError(
            "WG420_UNKNOWN_CAPTURE_REGION",
            "physical capture region requires a hierarchy artifact",
        )
    root = next(
        (candidate for candidate in hierarchy.roots if candidate.artifact_key == key),
        None,
    )
    if root is None:
        raise WorkGraphError(
            "WG420_UNKNOWN_CAPTURE_REGION",
            f"capture artifact {key.value!r} is not present in the hierarchy",
        )
    attachment = next(
        (
            candidate
            for candidate in hierarchy.attachments
            if candidate.artifact_key == key
        ),
        None,
    )
    if attachment is None:
        raise WorkGraphError(
            "WG420_UNKNOWN_CAPTURE_REGION",
            "capture artifact has no explicit runtime attachment",
        )
    if root.status != CaptureStatus.CAPTURED:
        raise WorkGraphError(
            "WG422_OPAQUE_CAPTURE_REGION",
            "opaque capture artifacts cannot become physical regions",
        )
    if allow_task_partition:
        # Program lowering: one physical node per stage capture task.
        return _compile_partitioned_region(
            raw_plan, root, attachment, key, implementation
        )
    if len(root.children) != 1:
        raise WorkGraphError(
            "WG421_UNSUPPORTED_CAPTURE_PARTITION",
            "physical execution currently requires one complete capture task",
        )
    task = root.children[0]
    leaf = task.children[0]
    artifact = leaf.leaf_artifact
    if artifact is None:
        raise WorkGraphError(
            "WG422_OPAQUE_CAPTURE_REGION",
            "capture task has no complete leaf artifact",
        )

    # An empty exported leaf has no graph work to replay. A selected eager-only
    # extern may still perform real runtime work, but it must run via its fixed
    # handler instead of inflating graph replay or fallback counters.
    policy = _capture_policy(root)
    if not artifact.operators or (
        implementation is not None
        and implementation.contract.capability == Capability.EAGER_ONLY
    ):
        policy = CapturePolicy.DISABLED
    physical = PhysicalFlowPlan(
        plan_epoch=raw_plan.epoch,
        nodes=(
            PhysicalNodePlan(
                node_key=task.node_key,
                executor=(
                    task.executor
                    if implementation is None
                    else implementation.executor
                ),
                stream=StreamClass.SIDE_FORWARD,
                artifact_key=key,
                capture_policy=policy,
                fallback_policy=FallbackPolicy.EAGER,
            ),
        ),
    )
    try:
        from tllm import core as _core

        _core.validate_physical_flow_plan(physical)
    except (TypeError, ValueError) as exc:
        raise WorkGraphError(
            "WG400_RUST_PHYSICAL_VALIDATION",
            "Rust capture-region descriptor validation failed",
        ) from exc
    return PhysicalCaptureRegion(
        attachment=attachment,
        plan=physical,
        executor_contract=(
            None if implementation is None else implementation.contract
        ),
    )


def _compile_partitioned_region(
    plan: ShadowParityPlan,
    root: RootFlowDesc,
    attachment: CaptureAttachment,
    key: ArtifactKey,
    implementation: PhysicalRegionExecutorDesc | None,
) -> PhysicalCaptureRegion:
    """Program-only lowering: one physical node per stage capture task.

    Reachable only via ``allow_task_partition=True``; the legacy single-task
    rule (WG421) is untouched for direct captures.
    """
    from tllm.workgraph.lowering import WorkGraphError

    policy = _capture_policy(root)
    if implementation is not None and implementation.contract.capability == Capability.EAGER_ONLY:
        policy = CapturePolicy.DISABLED
    task_node_by_region = {task.key: task.node_key for task in root.children}
    nodes: list[PhysicalNodePlan] = []
    fence_ordinal = 0
    for task in root.children:
        leaf = task.children[0]
        task_policy = policy
        if leaf.leaf_artifact is None or not leaf.leaf_artifact.operators:
            task_policy = CapturePolicy.DISABLED
        fences: list[FenceDependencyDesc] = []
        for dependency in task.dependencies:
            producer = task_node_by_region.get(dependency)
            if producer is None:
                raise WorkGraphError(
                    "WG420_UNKNOWN_CAPTURE_REGION",
                    "program stage depends on a task outside its capture root",
                )
            fences.append(
                FenceDependencyDesc(
                    fence=FenceKey("workgraph", "partition", fence_ordinal),
                    producer=producer,
                    consumer=task.node_key,
                    kind=FenceKind.READY,
                )
            )
            fence_ordinal += 1
        nodes.append(
            PhysicalNodePlan(
                node_key=task.node_key,
                executor=(
                    task.executor if implementation is None else implementation.executor
                ),
                stream=StreamClass.SIDE_FORWARD,
                artifact_key=key,
                fences=tuple(fences),
                capture_policy=task_policy,
                fallback_policy=FallbackPolicy.EAGER,
            )
        )
    physical = PhysicalFlowPlan(
        plan_epoch=plan.epoch,
        nodes=tuple(nodes),
    )
    try:
        from tllm import core as _core

        _core.validate_physical_flow_plan(physical)
    except (TypeError, ValueError) as exc:
        raise WorkGraphError(
            "WG400_RUST_PHYSICAL_VALIDATION",
            "Rust partitioned capture-region descriptor validation failed",
        ) from exc
    return PhysicalCaptureRegion(
        attachment=attachment,
        plan=physical,
        executor_contract=None if implementation is None else implementation.contract,
    )


__all__ = [
    "AnchorPlacementDesc",
    "BufferBucketDesc",
    "BufferBucketKey",
    "BufferOwnershipKind",
    "CapturePolicy",
    "FallbackPolicy",
    "FenceDependencyDesc",
    "FenceKey",
    "FenceKind",
    "PhysicalFlowPlan",
    "PhysicalCaptureRegion",
    "PhysicalNodePlan",
    "PhysicalRegionExecutorDesc",
    "StreamClass",
    "compile_physical_flow_plan",
    "compile_physical_capture_region",
]
