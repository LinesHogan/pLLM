"""Control-plane identity keys and frozen physical-plan descriptors.

This module is the native home of the single-device physical execution
vocabulary: stable identity keys (``ArtifactKey``/``NodeKey``/``ExecutorKey``),
tensor metadata descriptors, and the frozen ``PhysicalFlowPlan`` record that
both the schedule middle-end lowering and the runtime executor consume.

It is a control-plane boundary.  It deliberately contains no Tensor, CUDA
stream/event, CUDAGraph, callback, request id, or backend runtime handle.
Those objects belong to the Python physical executor.

The legacy ``tllm.workgraph`` modules re-export these names so existing
legacy import paths keep working with identical class identity; new native
code must import from here, never from ``tllm.workgraph``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import FunctionType
from typing import Callable, TypeVar, cast

_T = TypeVar("_T")


def _require_str(value: object, label: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str")
    if nonempty and not value:
        raise ValueError(f"{label} must be non-empty")
    return value


def _require_label(value: object, label: str, *, nonempty: bool = False) -> str:
    # Historical workgraph.frontend semantics: an empty label is a TypeError
    # there but a ValueError in workgraph.ir.  Both exception contracts are
    # kept verbatim so existing callers observe identical failures.
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


def _stable_segment(value: str, label: str) -> str:
    result = _require_str(value, label, nonempty=True)
    if "/" in result or "\\" in result:
        raise ValueError(f"{label} must not contain '/' or '\\\\'")
    return result


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


def _tuple_of(value: object, expected: type[_T], label: str) -> tuple[_T, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be tuple")
    items = cast(tuple[object, ...], value)
    if any(not isinstance(item, expected) for item in items):
        raise TypeError(f"{label} must contain only {expected.__name__}")
    return cast(tuple[_T, ...], items)


class Device(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True, slots=True)
class TensorDeviceDesc:
    kind: Device
    index: int | None = None

    def __post_init__(self) -> None:
        kind = _require_instance(cast(object, self.kind), Device, "TensorDeviceDesc.kind")
        index_value = cast(object, self.index)
        index: int | None
        if index_value is None:
            index = None
        else:
            index = _require_int(index_value, "TensorDeviceDesc.index")
            if index < 0:
                raise ValueError("TensorDeviceDesc.index must be non-negative")
        if kind == Device.CPU and index is not None:
            raise ValueError("CPU TensorDeviceDesc cannot declare a device index")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "index", index)


@dataclass(frozen=True, slots=True)
class TensorDTypeDesc:
    namespace: str
    name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "namespace",
            _require_label(cast(object, self.namespace), "TensorDTypeDesc.namespace", nonempty=True),
        )
        object.__setattr__(
            self,
            "name",
            _require_label(cast(object, self.name), "TensorDTypeDesc.name", nonempty=True),
        )


@dataclass(frozen=True, slots=True, order=True)
class ArtifactKey:
    """Stable leaf-artifact identity, independent of node and executor keys."""

    namespace: str
    name: str
    version: int = 1
    digest: str = ""

    def __post_init__(self) -> None:
        namespace = _require_label(cast(object, self.namespace), "ArtifactKey.namespace", nonempty=True)
        name = _require_label(cast(object, self.name), "ArtifactKey.name", nonempty=True)
        if "/" in namespace or "\\" in namespace:
            raise ValueError("ArtifactKey.namespace must not contain path separators")
        if "/" in name or "\\" in name:
            raise ValueError("ArtifactKey.name must not contain path separators")
        version = _require_int(cast(object, self.version), "ArtifactKey.version")
        if version < 1:
            raise ValueError("ArtifactKey.version must be >= 1")
        digest = _require_label(cast(object, self.digest), "ArtifactKey.digest")
        if digest and (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("ArtifactKey.digest must be an empty string or lowercase SHA-256")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "digest", digest)

    @classmethod
    def from_callable(
        cls,
        function: Callable[..., object],
        *,
        digest: str = "",
    ) -> "ArtifactKey":
        if not isinstance(function, FunctionType):
            raise TypeError("captured callable must be a Python function")
        namespace = function.__module__
        name = function.__qualname__
        if not name:
            raise TypeError("captured callable must have a stable qualified name")
        return cls(namespace=namespace, name=name, digest=digest)

    @property
    def value(self) -> str:
        suffix = "" if not self.digest else f"#{self.digest}"
        return f"{self.namespace}:{self.name}@{self.version}{suffix}"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class NodeKey:
    """Stable logical node identity.

    ``str(NodeKey(...))`` is the persisted form.  The three components are
    intentionally independent of object addresses, request ids, Tensor reprs
    and source line numbers.
    """

    namespace: str
    owner: str
    ordinal: int = 0

    def __post_init__(self) -> None:
        namespace = _require_str(self.namespace, "NodeKey.namespace", nonempty=True)
        owner = _require_str(self.owner, "NodeKey.owner", nonempty=True)
        if "/" in namespace:
            raise ValueError("NodeKey.namespace must not contain '/'")
        if namespace == "anchor":
            segments = owner.split("/")
            if len(segments) != 4 or any(not segment or "\\" in segment for segment in segments):
                raise ValueError(
                    "anchor NodeKey.owner must contain exactly four non-empty '/'-separated segments"
                )
        else:
            owner = _stable_segment(owner, "NodeKey.owner")
        ordinal = _require_int(self.ordinal, "NodeKey.ordinal")
        if ordinal < 0:
            raise ValueError("NodeKey.ordinal must be a non-negative int")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "ordinal", ordinal)

    @property
    def value(self) -> str:
        if self.namespace == "anchor":
            return f"{self.namespace}/{self.owner}"
        return f"{self.namespace}/{self.owner}/{self.ordinal}"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def consumer_flow(cls, consumer_id: str, ordinal: int) -> "NodeKey":
        return cls("consumer_flow", consumer_id, ordinal)

    @classmethod
    def producer(cls, producer_id: str) -> "NodeKey":
        return cls("producer", producer_id, 0)

    @classmethod
    def anchor(cls, backend: str, runner: str, graph_mode: str, hook: str) -> "NodeKey":
        return cls("anchor", f"{backend}/{runner}/{graph_mode}/{hook}", 0)


@dataclass(frozen=True, slots=True, order=True)
class ExecutorKey:
    """Stable identity for an implementation, without storing its callback."""

    namespace: str
    name: str
    version: str = "1"

    def __post_init__(self) -> None:
        namespace = _require_str(self.namespace, "ExecutorKey.namespace", nonempty=True)
        name = _require_str(self.name, "ExecutorKey.name", nonempty=True)
        version = _require_str(self.version, "ExecutorKey.version", nonempty=True)
        if "/" in namespace or "/" in name:
            raise ValueError("ExecutorKey namespace/name must not contain '/'")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)

    @property
    def value(self) -> str:
        return f"{self.namespace}:{self.name}@{self.version}"

    def __str__(self) -> str:
        return self.value


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
        object.__setattr__(self, "key", _require_instance(self.key, BufferBucketKey, "BufferBucketDesc.key"))
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
            _require_instance(self.dtype, TensorDTypeDesc, "BufferBucketDesc.dtype"),
        )
        _text(self.dtype.namespace, "BufferBucketDesc.dtype.namespace")
        _text(self.dtype.name, "BufferBucketDesc.dtype.name")
        object.__setattr__(
            self,
            "device",
            _require_instance(self.device, TensorDeviceDesc, "BufferBucketDesc.device"),
        )
        object.__setattr__(self, "slots", _integer(self.slots, "BufferBucketDesc.slots", minimum=1))
        object.__setattr__(
            self,
            "ownership",
            _require_instance(self.ownership, BufferOwnershipKind, "BufferBucketDesc.ownership"),
        )


@dataclass(frozen=True, slots=True)
class FenceDependencyDesc:
    fence: FenceKey
    producer: NodeKey
    consumer: NodeKey
    kind: FenceKind

    def __post_init__(self) -> None:
        object.__setattr__(self, "fence", _require_instance(self.fence, FenceKey, "FenceDependencyDesc.fence"))
        producer = _require_instance(self.producer, NodeKey, "FenceDependencyDesc.producer")
        consumer = _require_instance(self.consumer, NodeKey, "FenceDependencyDesc.consumer")
        if producer == consumer:
            raise ValueError("FenceDependencyDesc producer and consumer must differ")
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "consumer", consumer)
        object.__setattr__(self, "kind", _require_instance(self.kind, FenceKind, "FenceDependencyDesc.kind"))


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
        node_key = _require_instance(self.node_key, NodeKey, "PhysicalNodePlan.node_key")
        _text(node_key.namespace, "PhysicalNodePlan.node_key.namespace")
        _text(node_key.owner, "PhysicalNodePlan.node_key.owner")
        object.__setattr__(self, "node_key", node_key)
        executor = _require_instance(self.executor, ExecutorKey, "PhysicalNodePlan.executor")
        _text(executor.namespace, "PhysicalNodePlan.executor.namespace")
        _text(executor.name, "PhysicalNodePlan.executor.name")
        _text(executor.version, "PhysicalNodePlan.executor.version")
        object.__setattr__(self, "executor", executor)
        object.__setattr__(self, "stream", _require_instance(self.stream, StreamClass, "PhysicalNodePlan.stream"))
        fences = _tuple_of(self.fences, FenceDependencyDesc, "PhysicalNodePlan.fences")
        if any(fence.consumer != node_key for fence in fences):
            raise ValueError("PhysicalNodePlan fences must target the owning node")
        if len({fence.fence for fence in fences}) != len(fences):
            raise ValueError("PhysicalNodePlan fences must have unique keys")
        object.__setattr__(self, "fences", fences)
        if self.artifact_key is not None:
            artifact_key = _require_instance(
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
                _require_instance(self.buffer_bucket, BufferBucketKey, "PhysicalNodePlan.buffer_bucket"),
            )
        capture_policy = _require_instance(
            self.capture_policy, CapturePolicy, "PhysicalNodePlan.capture_policy"
        )
        fallback_policy = _require_instance(
            self.fallback_policy, FallbackPolicy, "PhysicalNodePlan.fallback_policy"
        )
        if capture_policy == CapturePolicy.REQUIRE and fallback_policy == FallbackPolicy.EAGER:
            raise ValueError("required capture cannot declare eager fallback")
        if capture_policy != CapturePolicy.DISABLED and self.artifact_key is None:
            raise ValueError("enabled capture requires an ArtifactKey")
        object.__setattr__(self, "capture_policy", capture_policy)
        object.__setattr__(self, "fallback_policy", fallback_policy)


@dataclass(frozen=True, slots=True)
class AnchorPlacementDesc:
    node_key: NodeKey
    after_anchor: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "node_key",
            _require_instance(self.node_key, NodeKey, "AnchorPlacementDesc.node_key"),
        )
        object.__setattr__(
            self,
            "after_anchor",
            _text(self.after_anchor, "AnchorPlacementDesc.after_anchor"),
        )


@dataclass(frozen=True, slots=True)
class PhysicalFlowPlan:
    plan_epoch: int
    nodes: tuple[PhysicalNodePlan, ...]
    buffer_buckets: tuple[BufferBucketDesc, ...] = ()
    anchor_placements: tuple[AnchorPlacementDesc, ...] = ()
    schedule_signature: str = ""
    schema_version: int = 1

    def __post_init__(self) -> None:
        epoch = _integer(self.plan_epoch, "PhysicalFlowPlan.plan_epoch", minimum=1)
        nodes = _tuple_of(self.nodes, PhysicalNodePlan, "PhysicalFlowPlan.nodes")
        buckets = _tuple_of(
            self.buffer_buckets, BufferBucketDesc, "PhysicalFlowPlan.buffer_buckets"
        )
        placements = _tuple_of(
            self.anchor_placements,
            AnchorPlacementDesc,
            "PhysicalFlowPlan.anchor_placements",
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
        if placements and tuple(placement.node_key for placement in placements) != tuple(
            node.node_key for node in nodes
        ):
            raise ValueError(
                "PhysicalFlowPlan anchor placements must cover nodes in execution order"
            )
        signature = self.schedule_signature
        if signature and (
            len(signature) != 64
            or any(character not in "0123456789abcdef" for character in signature)
        ):
            raise ValueError(
                "PhysicalFlowPlan.schedule_signature must be lowercase SHA-256"
            )
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
        object.__setattr__(self, "anchor_placements", placements)


__all__ = [
    "AnchorPlacementDesc",
    "ArtifactKey",
    "BufferBucketDesc",
    "BufferBucketKey",
    "BufferOwnershipKind",
    "CapturePolicy",
    "Device",
    "ExecutorKey",
    "FallbackPolicy",
    "FenceDependencyDesc",
    "FenceKey",
    "FenceKind",
    "NodeKey",
    "PhysicalFlowPlan",
    "PhysicalNodePlan",
    "StreamClass",
    "TensorDTypeDesc",
    "TensorDeviceDesc",
]
