"""Frozen physical-plan descriptors for single-device WorkGraph execution.

This module is a control-plane boundary.  It deliberately contains no Tensor,
CUDA stream/event, CUDAGraph, callback, request id, or backend runtime handle.
Those objects belong to the Python physical executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, TypeVar, cast

from tllm.workgraph.frontend import (
    ArtifactKey,
    Capability,
    CaptureStatus,
    TensorDTypeDesc,
    TensorDeviceDesc,
)
from tllm.workgraph.hierarchy import CaptureAttachment, RootFlowDesc
from tllm.workgraph.ir import (
    EdgeKind,
    ExecutionNodeDesc,
    ExecutorKey,
    NodeKey,
)

if TYPE_CHECKING:
    from tllm.workgraph.compiler import Phase1aPlan


_T = TypeVar("_T")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str")
    if not value:
        raise ValueError(f"{label} must be non-empty")
    if "/" in value or "\\" in value:
        raise ValueError(f"{label} must not contain path separators")
    if "0x" in value.lower() or "<" in value or ">" in value:
        raise ValueError(f"{label} must not contain runtime repr or address text")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be int")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _instance(value: object, expected: type[_T], label: str) -> _T:
    if not isinstance(value, expected):
        raise TypeError(f"{label} must be {expected.__name__}")
    return value


def _tuple_of(value: object, expected: type[_T], label: str) -> tuple[_T, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be tuple")
    items = cast(tuple[object, ...], value)
    if any(not isinstance(item, expected) for item in items):
        raise TypeError(f"{label} must contain only {expected.__name__}")
    return cast(tuple[_T, ...], items)


class StreamClass(StrEnum):
    DEFAULT = "default"
    SIDE_FORWARD = "side_forward"
    SIDE_TRAIN = "side_train"


class FenceKind(StrEnum):
    READY = "ready"
    COMPLETION = "completion"
    RETIRE = "retire"


class CapturePolicy(StrEnum):
    DISABLED = "disabled"
    PREFER = "prefer"
    REQUIRE = "require"


class FallbackPolicy(StrEnum):
    EAGER = "eager"
    REJECT = "reject"


class BufferOwnershipKind(StrEnum):
    BORROWED = "borrowed"
    MANAGED = "managed"


@dataclass(frozen=True, slots=True, order=True)
class FenceKey:
    namespace: str
    name: str
    ordinal: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", _text(self.namespace, "FenceKey.namespace"))
        object.__setattr__(self, "name", _text(self.name, "FenceKey.name"))
        object.__setattr__(self, "ordinal", _integer(self.ordinal, "FenceKey.ordinal"))

    @property
    def value(self) -> str:
        return f"{self.namespace}:{self.name}/{self.ordinal}"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class BufferBucketKey:
    namespace: str
    name: str
    ordinal: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", _text(self.namespace, "BufferBucketKey.namespace"))
        object.__setattr__(self, "name", _text(self.name, "BufferBucketKey.name"))
        object.__setattr__(self, "ordinal", _integer(self.ordinal, "BufferBucketKey.ordinal"))

    @property
    def value(self) -> str:
        return f"{self.namespace}:{self.name}/{self.ordinal}"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class BufferBucketDesc:
    key: BufferBucketKey
    shape: tuple[int, ...]
    dtype: TensorDTypeDesc
    device: TensorDeviceDesc
    slots: int = 1
    ownership: BufferOwnershipKind = BufferOwnershipKind.MANAGED

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _instance(self.key, BufferBucketKey, "BufferBucketDesc.key"))
        raw_shape = cast(object, self.shape)
        if not isinstance(raw_shape, tuple):
            raise TypeError("BufferBucketDesc.shape must be tuple")
        shape = tuple(
            _integer(dimension, "BufferBucketDesc.shape dimension", minimum=1)
            for dimension in cast(tuple[object, ...], raw_shape)
        )
        if not shape:
            raise ValueError("BufferBucketDesc.shape must be non-empty")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(
            self,
            "dtype",
            _instance(self.dtype, TensorDTypeDesc, "BufferBucketDesc.dtype"),
        )
        _text(self.dtype.namespace, "BufferBucketDesc.dtype.namespace")
        _text(self.dtype.name, "BufferBucketDesc.dtype.name")
        object.__setattr__(
            self,
            "device",
            _instance(self.device, TensorDeviceDesc, "BufferBucketDesc.device"),
        )
        object.__setattr__(self, "slots", _integer(self.slots, "BufferBucketDesc.slots", minimum=1))
        object.__setattr__(
            self,
            "ownership",
            _instance(self.ownership, BufferOwnershipKind, "BufferBucketDesc.ownership"),
        )


@dataclass(frozen=True, slots=True)
class FenceDependencyDesc:
    fence: FenceKey
    producer: NodeKey
    consumer: NodeKey
    kind: FenceKind

    def __post_init__(self) -> None:
        object.__setattr__(self, "fence", _instance(self.fence, FenceKey, "FenceDependencyDesc.fence"))
        producer = _instance(self.producer, NodeKey, "FenceDependencyDesc.producer")
        consumer = _instance(self.consumer, NodeKey, "FenceDependencyDesc.consumer")
        if producer == consumer:
            raise ValueError("FenceDependencyDesc producer and consumer must differ")
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "consumer", consumer)
        object.__setattr__(self, "kind", _instance(self.kind, FenceKind, "FenceDependencyDesc.kind"))


@dataclass(frozen=True, slots=True)
class PhysicalNodePlan:
    node_key: NodeKey
    executor: ExecutorKey
    stream: StreamClass
    fences: tuple[FenceDependencyDesc, ...] = ()
    artifact_key: ArtifactKey | None = None
    buffer_bucket: BufferBucketKey | None = None
    capture_policy: CapturePolicy = CapturePolicy.DISABLED
    fallback_policy: FallbackPolicy = FallbackPolicy.EAGER

    def __post_init__(self) -> None:
        node_key = _instance(self.node_key, NodeKey, "PhysicalNodePlan.node_key")
        _text(node_key.namespace, "PhysicalNodePlan.node_key.namespace")
        _text(node_key.owner, "PhysicalNodePlan.node_key.owner")
        object.__setattr__(self, "node_key", node_key)
        executor = _instance(self.executor, ExecutorKey, "PhysicalNodePlan.executor")
        _text(executor.namespace, "PhysicalNodePlan.executor.namespace")
        _text(executor.name, "PhysicalNodePlan.executor.name")
        _text(executor.version, "PhysicalNodePlan.executor.version")
        object.__setattr__(self, "executor", executor)
        object.__setattr__(self, "stream", _instance(self.stream, StreamClass, "PhysicalNodePlan.stream"))
        fences = _tuple_of(self.fences, FenceDependencyDesc, "PhysicalNodePlan.fences")
        if any(fence.consumer != node_key for fence in fences):
            raise ValueError("PhysicalNodePlan fences must target the owning node")
        if len({fence.fence for fence in fences}) != len(fences):
            raise ValueError("PhysicalNodePlan fences must have unique keys")
        object.__setattr__(self, "fences", fences)
        if self.artifact_key is not None:
            artifact_key = _instance(
                self.artifact_key, ArtifactKey, "PhysicalNodePlan.artifact_key"
            )
            if len(artifact_key.digest) != 64:
                raise ValueError("physical ArtifactKey requires a SHA-256 digest")
            object.__setattr__(
                self,
                "artifact_key",
                artifact_key,
            )
        if self.buffer_bucket is not None:
            object.__setattr__(
                self,
                "buffer_bucket",
                _instance(self.buffer_bucket, BufferBucketKey, "PhysicalNodePlan.buffer_bucket"),
            )
        capture_policy = _instance(
            self.capture_policy, CapturePolicy, "PhysicalNodePlan.capture_policy"
        )
        fallback_policy = _instance(
            self.fallback_policy, FallbackPolicy, "PhysicalNodePlan.fallback_policy"
        )
        if capture_policy == CapturePolicy.REQUIRE and fallback_policy == FallbackPolicy.EAGER:
            raise ValueError("required capture cannot declare eager fallback")
        if capture_policy != CapturePolicy.DISABLED and self.artifact_key is None:
            raise ValueError("enabled capture requires an ArtifactKey")
        object.__setattr__(self, "capture_policy", capture_policy)
        object.__setattr__(self, "fallback_policy", fallback_policy)


@dataclass(frozen=True, slots=True)
class PhysicalFlowPlan:
    plan_epoch: int
    nodes: tuple[PhysicalNodePlan, ...]
    buffer_buckets: tuple[BufferBucketDesc, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        epoch = _integer(self.plan_epoch, "PhysicalFlowPlan.plan_epoch", minimum=1)
        nodes = _tuple_of(self.nodes, PhysicalNodePlan, "PhysicalFlowPlan.nodes")
        buckets = _tuple_of(
            self.buffer_buckets, BufferBucketDesc, "PhysicalFlowPlan.buffer_buckets"
        )
        if self.schema_version != 1:
            raise ValueError("PhysicalFlowPlan.schema_version must be 1")
        if len({node.node_key for node in nodes}) != len(nodes):
            raise ValueError("PhysicalFlowPlan nodes must have unique NodeKey values")
        bucket_keys = {bucket.key for bucket in buckets}
        if len(bucket_keys) != len(buckets):
            raise ValueError("PhysicalFlowPlan buffer buckets must have unique keys")
        if any(
            node.buffer_bucket is not None and node.buffer_bucket not in bucket_keys
            for node in nodes
        ):
            raise ValueError("PhysicalFlowPlan node references an unknown buffer bucket")
        node_keys = {node.node_key for node in nodes}
        positions = {node.node_key: ordinal for ordinal, node in enumerate(nodes)}
        fence_keys: set[FenceKey] = set()
        for node in nodes:
            for fence in node.fences:
                if fence.producer not in node_keys:
                    raise ValueError("PhysicalFlowPlan fence references an unknown producer")
                if fence.fence in fence_keys:
                    raise ValueError("PhysicalFlowPlan fences must have unique keys")
                fence_keys.add(fence.fence)
                if positions[fence.producer] >= positions[fence.consumer]:
                    raise ValueError("PhysicalFlowPlan nodes must be in dependency order")
        object.__setattr__(self, "plan_epoch", epoch)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "buffer_buckets", buckets)


@dataclass(frozen=True, slots=True)
class PhysicalCaptureRegion:
    """One explicitly attached capture root and its executable task plan."""

    attachment: CaptureAttachment
    plan: PhysicalFlowPlan

    def __post_init__(self) -> None:
        attachment = _instance(
            self.attachment,
            CaptureAttachment,
            "PhysicalCaptureRegion.attachment",
        )
        plan = _instance(self.plan, PhysicalFlowPlan, "PhysicalCaptureRegion.plan")
        if not plan.nodes:
            raise ValueError("PhysicalCaptureRegion.plan must contain a capture task")
        if any(node.node_key.namespace != "capture_task" for node in plan.nodes):
            raise ValueError("PhysicalCaptureRegion.plan may contain only capture tasks")
        if any(node.artifact_key != attachment.artifact_key for node in plan.nodes):
            raise ValueError("PhysicalCaptureRegion plan artifact differs from its attachment")
        object.__setattr__(self, "attachment", attachment)
        object.__setattr__(self, "plan", plan)


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


def compile_physical_flow_plan(plan: Phase1aPlan) -> PhysicalFlowPlan:
    """Lower one verified shadow plan into frozen single-device descriptors."""

    from tllm.workgraph.compiler import Phase1aPlan
    from tllm.workgraph.lowering import WorkGraphError

    raw_plan = cast(object, plan)
    if not isinstance(raw_plan, Phase1aPlan):
        raise TypeError("compile_physical_flow_plan.plan must be Phase1aPlan")
    plan = raw_plan
    if plan.epoch < 1:
        raise ValueError("physical plan requires a compiled Phase 1 epoch")
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
    plan: Phase1aPlan,
    artifact_key: ArtifactKey,
) -> PhysicalCaptureRegion:
    """Lower one complete hierarchy root into an independently runnable plan.

    Compatibility producer/consumer nodes are intentionally absent.  They
    remain authoritative legacy boundaries while this plan owns only the
    explicitly migrated capture task.
    """

    from tllm.workgraph.compiler import Phase1aPlan
    from tllm.workgraph.lowering import WorkGraphError

    raw_plan = cast(object, plan)
    if not isinstance(raw_plan, Phase1aPlan):
        raise TypeError("compile_physical_capture_region.plan must be Phase1aPlan")
    key = _instance(
        artifact_key,
        ArtifactKey,
        "compile_physical_capture_region.artifact_key",
    )
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

    # An alias-only handoff has no CUDA work to capture.  Treating its empty
    # exported graph as replay would inflate replay counters without doing
    # work, so it uses the normal eager handler and records no fallback.
    policy = (
        CapturePolicy.DISABLED
        if not artifact.operators
        else _capture_policy(root)
    )
    physical = PhysicalFlowPlan(
        plan_epoch=raw_plan.epoch,
        nodes=(
            PhysicalNodePlan(
                node_key=task.node_key,
                executor=task.executor,
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
    return PhysicalCaptureRegion(attachment=attachment, plan=physical)


__all__ = [
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
    "StreamClass",
    "compile_physical_flow_plan",
    "compile_physical_capture_region",
]
