"""Contract tests for the Rust physical WorkGraph plan validator."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from tllm import core
from tllm.workgraph.frontend import (
    ArtifactKey,
    Device,
    TensorDTypeDesc,
    TensorDeviceDesc,
)
from tllm.workgraph.ir import (
    ExecutorKey,
    NodeKey,
)
from tllm.workgraph.physical import (
    BufferBucketDesc,
    BufferBucketKey,
    BufferOwnershipKind,
    CapturePolicy,
    FallbackPolicy,
    FenceDependencyDesc,
    FenceKey,
    FenceKind,
    PhysicalFlowPlan,
    PhysicalNodePlan,
    StreamClass,
)


# These frozen structural descriptors let the test cross the Rust boundary
# with values that Python's stricter descriptor constructors reject locally.
@dataclass(frozen=True, slots=True)
class _DType:
    namespace: str
    name: str


@dataclass(frozen=True, slots=True)
class _Device:
    kind: str
    index: int | None = None


@dataclass(frozen=True, slots=True)
class _Key:
    namespace: str
    name: str
    ordinal: int = 0


@dataclass(frozen=True, slots=True)
class _NodeKey:
    namespace: str
    owner: str
    ordinal: int = 0


@dataclass(frozen=True, slots=True)
class _Executor:
    namespace: str
    name: str
    version: str = "1"


@dataclass(frozen=True, slots=True)
class _Artifact:
    namespace: str
    name: str
    version: int = 1
    digest: str = "a" * 64


@dataclass(frozen=True, slots=True)
class _Bucket:
    key: _Key
    shape: tuple[int, ...]
    dtype: _DType
    device: _Device
    slots: int = 1
    ownership: str = "managed"


@dataclass(frozen=True, slots=True)
class _Fence:
    fence: _Key
    producer: _NodeKey
    consumer: _NodeKey
    kind: str


@dataclass(frozen=True, slots=True)
class _Node:
    node_key: _NodeKey
    executor: _Executor
    stream: str = "default"
    fences: tuple[_Fence, ...] = ()
    artifact_key: _Artifact | None = None
    buffer_bucket: _Key | None = None
    capture_policy: str = "disabled"
    fallback_policy: str = "eager"


@dataclass(frozen=True, slots=True)
class _Plan:
    plan_epoch: int
    nodes: tuple[_Node, ...]
    buffer_buckets: tuple[_Bucket, ...] = ()
    schema_version: int = 1


def _real_plan() -> PhysicalFlowPlan:
    producer = NodeKey.producer("physical.source")
    consumer = NodeKey.consumer_flow("physical.consumer", 0)
    bucket_key = BufferBucketKey("physical", "hidden")
    bucket = BufferBucketDesc(
        key=bucket_key,
        shape=(2, 4),
        dtype=TensorDTypeDesc("torch", "float32"),
        device=TensorDeviceDesc(Device.CUDA, index=0),
        slots=2,
        ownership=BufferOwnershipKind.MANAGED,
    )
    dependency = FenceDependencyDesc(
        fence=FenceKey("physical", "ready"),
        producer=producer,
        consumer=consumer,
        kind=FenceKind.READY,
    )
    return PhysicalFlowPlan(
        plan_epoch=7,
        nodes=(
            PhysicalNodePlan(
                node_key=producer,
                executor=ExecutorKey("producer", "physical.source"),
                stream=StreamClass.DEFAULT,
            ),
            PhysicalNodePlan(
                node_key=consumer,
                executor=ExecutorKey("consumer", "physical.consumer"),
                stream=StreamClass.SIDE_TRAIN,
                fences=(dependency,),
                artifact_key=ArtifactKey(
                    "capture", "physical.consumer", digest="a" * 64
                ),
                buffer_bucket=bucket_key,
                capture_policy=CapturePolicy.PREFER,
                fallback_policy=FallbackPolicy.REJECT,
            ),
        ),
        buffer_buckets=(bucket,),
    )


def _structural_plan(
    *,
    plan_epoch: int = 1,
    schema_version: int = 1,
    nodes: tuple[_Node, ...] | None = None,
    buckets: tuple[_Bucket, ...] = (),
) -> _Plan:
    source = _NodeKey("producer", "physical.source")
    target = _NodeKey("consumer_flow", "physical.consumer")
    if nodes is None:
        nodes = (
            _Node(source, _Executor("producer", "physical.source")),
            _Node(
                target,
                _Executor("consumer", "physical.consumer"),
                fences=(
                    _Fence(
                        _Key("physical", "ready"),
                        source,
                        target,
                        "ready",
                    ),
                ),
            ),
        )
    return _Plan(
        plan_epoch=plan_epoch,
        nodes=nodes,
        buffer_buckets=buckets,
        schema_version=schema_version,
    )


def _bucket(
    *,
    key: _Key | None = None,
    dtype: _DType | None = None,
    device: _Device | None = None,
    ownership: str = "managed",
) -> _Bucket:
    return _Bucket(
        key=_Key("physical", "hidden") if key is None else key,
        shape=(2, 4),
        dtype=_DType("torch", "float32") if dtype is None else dtype,
        device=_Device("cuda", 0) if device is None else device,
        ownership=ownership,
    )


def _assert_rejected(plan: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        core.validate_physical_flow_plan(plan)


def test_rust_validator_accepts_frozen_physical_plan_and_returns_node_order() -> None:
    result = core.validate_physical_flow_plan(_real_plan())

    assert isinstance(result, core.WorkGraphValidation)
    assert result.topological_order == [
        "producer/physical.source/0",
        "consumer_flow/physical.consumer/0",
    ]


@pytest.mark.parametrize(
    ("plan_epoch", "schema_version"),
    [(0, 1), (1, 0), (1, 2)],
)
def test_rust_validator_rejects_schema_and_epoch_corruption(
    plan_epoch: int, schema_version: int
) -> None:
    _assert_rejected(_structural_plan(plan_epoch=plan_epoch, schema_version=schema_version))


def test_rust_validator_rejects_duplicate_nodes_and_buffer_buckets() -> None:
    base = _structural_plan()
    _assert_rejected(replace(base, nodes=(base.nodes[0], base.nodes[0])))

    bucket = _bucket()
    _assert_rejected(replace(base, buffer_buckets=(bucket, bucket)))


def test_rust_validator_rejects_unknown_bucket_and_fence_producer() -> None:
    base = _structural_plan()
    target = base.nodes[1]
    unknown_bucket = replace(
        target,
        buffer_bucket=_Key("physical", "missing"),
    )
    _assert_rejected(replace(base, nodes=(base.nodes[0], unknown_bucket)))

    unknown_producer = replace(
        target,
        fences=(
            _Fence(
                _Key("physical", "ready"),
                _NodeKey("producer", "missing"),
                target.node_key,
                "ready",
            ),
        ),
    )
    _assert_rejected(replace(base, nodes=(base.nodes[0], unknown_producer)))


@pytest.mark.parametrize(
    "target",
    [
        replace(_structural_plan().nodes[1], stream="side_copy"),
        replace(_structural_plan().nodes[1], capture_policy="always"),
        replace(_structural_plan().nodes[1], fallback_policy="retry"),
        replace(_structural_plan().nodes[1], capture_policy="prefer"),
        replace(
            _structural_plan().nodes[1],
            capture_policy="require",
            fallback_policy="eager",
        ),
    ],
)
def test_rust_validator_rejects_illegal_stream_capture_and_fallback(
    target: _Node,
) -> None:
    base = _structural_plan()
    _assert_rejected(replace(base, nodes=(base.nodes[0], target)))


@pytest.mark.parametrize(
    "bucket",
    [
        _bucket(ownership="runtime_lease"),
        _bucket(device=_Device("xpu", 0)),
        _bucket(device=_Device("cpu", 0)),
        _bucket(device=_Device("cuda", -1)),
    ],
)
def test_rust_validator_rejects_illegal_bucket_ownership_and_device(
    bucket: _Bucket,
) -> None:
    _assert_rejected(_structural_plan(buckets=(bucket,)))


def test_rust_validator_rejects_runtime_object_like_nested_descriptor() -> None:
    runtime_dtype = _DType("torch", "<Tensor object at 0xdeadbeef>")
    _assert_rejected(_structural_plan(buckets=(_bucket(dtype=runtime_dtype),)))


@pytest.mark.parametrize(
    ("namespace", "owner"),
    [
        ("producer<Node object at 0xdeadbeef>", "physical.source"),
        ("producer", "physical.source<Node object at 0xdeadbeef>"),
    ],
)
def test_rust_validator_rejects_runtime_identity_in_physical_node_key(
    namespace: str,
    owner: str,
) -> None:
    base = _structural_plan()
    runtime_source = _NodeKey(namespace, owner)
    target = base.nodes[1]
    target = replace(
        target,
        fences=(replace(target.fences[0], producer=runtime_source),),
    )
    runtime_source_node = replace(base.nodes[0], node_key=runtime_source)

    _assert_rejected(replace(base, nodes=(runtime_source_node, target)))
