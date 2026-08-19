from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.workgraph.frontend import Device, TensorDTypeDesc, TensorDeviceDesc
from tllm.workgraph.physical import (
    BufferBucketDesc,
    BufferBucketKey,
    BufferOwnershipKind,
    FenceKey,
)
from tllm.runtime.scheduling.physical import BufferLease, Fence


class _FakeEvent:
    def __init__(self, name: str, trace: list[str]) -> None:
        self.name = name
        self.trace = trace
        self.done = False

    def record(self, stream: object) -> None:
        del stream
        self.trace.append(f"record:{self.name}")

    def query(self) -> bool:
        self.trace.append(f"query:{self.name}")
        return self.done


class _FakeStream:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace

    def wait_event(self, event: _FakeEvent) -> None:
        self.trace.append(f"wait:{event.name}")


def _fence(name: str, trace: list[str]) -> Fence:
    return Fence(FenceKey("physical", name), _FakeEvent(name, trace))


def _bucket(*, ownership: BufferOwnershipKind = BufferOwnershipKind.MANAGED) -> BufferBucketDesc:
    return BufferBucketDesc(
        key=BufferBucketKey("physical", "hidden"),
        shape=(2, 3),
        dtype=TensorDTypeDesc(namespace="torch", name="float32"),
        device=TensorDeviceDesc(Device.CPU),
        slots=2,
        ownership=ownership,
    )


class PhysicalExecutorUnitTest(unittest.TestCase):
    def test_fence_records_waits_and_reports_event_completion(self) -> None:
        trace: list[str] = []
        event = _FakeEvent("ready", trace)
        fence = Fence(FenceKey("physical", "ready"), event)
        stream = _FakeStream(trace)

        self.assertFalse(fence.complete)
        fence.record(stream)
        fence.wait(stream)
        self.assertEqual(trace, ["query:ready", "record:ready", "wait:ready"])

        event.done = True
        self.assertTrue(fence.complete)
        self.assertEqual(trace[-1], "query:ready")

    def test_managed_buffer_lease_waits_and_retires_without_global_sync(self) -> None:
        trace: list[str] = []
        ready = _fence("ready", trace)
        completion = _fence("completion", trace)
        retire = _fence("retire", trace)
        lease = BufferLease(
            bucket=_bucket(),
            tensor=torch.empty((2, 3), dtype=torch.float32),
            plan_epoch=1,
            slot=0,
            ready_fence=ready,
            completion_fence=completion,
            retire_fence=retire,
        )
        stream = _FakeStream(trace)

        with mock.patch(
            "torch.cuda.synchronize",
            side_effect=AssertionError("physical executor must use fence dependencies"),
        ):
            lease.wait_ready(stream)
            lease.publish_completion(stream)
            self.assertFalse(lease.reusable)
            lease.publish_retire(stream)
            self.assertFalse(lease.reusable)

            completion.event.done = True  # type: ignore[attr-defined]
            self.assertFalse(lease.reusable)
            retire.event.done = True  # type: ignore[attr-defined]
            self.assertTrue(lease.reusable)

        self.assertEqual(
            trace[:3],
            ["wait:ready", "record:completion", "record:retire"],
        )

    def test_buffer_lease_validates_tensor_layout_epoch_and_slot(self) -> None:
        trace: list[str] = []
        bucket = _bucket()
        fences = {
            name: _fence(name, trace)
            for name in ("ready", "completion", "retire")
        }

        with self.assertRaises(ValueError):
            BufferLease(
                bucket=bucket,
                tensor=torch.empty((2, 4), dtype=torch.float32),
                plan_epoch=1,
                slot=0,
                ready_fence=fences["ready"],
                completion_fence=fences["completion"],
                retire_fence=fences["retire"],
            )
        with self.assertRaises(ValueError):
            BufferLease(
                bucket=bucket,
                tensor=torch.empty((2, 3), dtype=torch.float64),
                plan_epoch=1,
                slot=0,
                ready_fence=fences["ready"],
                completion_fence=fences["completion"],
                retire_fence=fences["retire"],
            )
        with self.assertRaises(ValueError):
            BufferLease(
                bucket=bucket,
                tensor=torch.empty((2, 3), dtype=torch.float32),
                plan_epoch=0,
                slot=0,
                ready_fence=fences["ready"],
                completion_fence=fences["completion"],
                retire_fence=fences["retire"],
            )
        with self.assertRaises(ValueError):
            BufferLease(
                bucket=bucket,
                tensor=torch.empty((2, 3), dtype=torch.float32),
                plan_epoch=1,
                slot=2,
                ready_fence=fences["ready"],
                completion_fence=fences["completion"],
                retire_fence=fences["retire"],
            )

    def test_ownership_requires_the_expected_fence_contract(self) -> None:
        trace: list[str] = []
        managed = _bucket(ownership=BufferOwnershipKind.MANAGED)
        borrowed = _bucket(ownership=BufferOwnershipKind.BORROWED)
        ready = _fence("ready", trace)
        completion = _fence("completion", trace)
        retire = _fence("retire", trace)

        with self.assertRaises(ValueError):
            BufferLease(
                bucket=managed,
                tensor=torch.empty((2, 3)),
                plan_epoch=1,
                slot=0,
                ready_fence=ready,
                completion_fence=completion,
                retire_fence=None,
            )
        with self.assertRaises(ValueError):
            BufferLease(
                bucket=borrowed,
                tensor=torch.empty((2, 3)),
                plan_epoch=1,
                slot=0,
                ready_fence=None,
                completion_fence=completion,
                retire_fence=retire,
            )


if __name__ == "__main__":
    unittest.main()
