"""Red CPU-only contracts for Phase 3 pinned-memory transfer execution.

The metadata planner lives in :mod:`tllm.workgraph.phase3`; this module pins
the runtime side of that boundary.  Every tensor, fence, and copy operation is
fake so the tests exercise ownership and ordering without requiring CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import unittest

from tllm.workgraph import phase3


def _transfer_runtime(test_case: unittest.TestCase):
    """Import the future runtime surface as an explicit per-test red failure."""

    try:
        return importlib.import_module("tllm.runtime.scheduling.phase3_transfer")
    except (ImportError, ModuleNotFoundError) as exc:
        test_case.fail(
            "Phase 3 transfer runtime is not implemented: "
            "tllm.runtime.scheduling.phase3_transfer is required"
        )
        raise AssertionError from exc


@dataclass
class _FakeTensor:
    shape: tuple[int, ...]
    dtype: str
    device: str
    payload: list[float]


class _FakeFence:
    def __init__(self, name: str, *, complete: bool = False) -> None:
        self.name = name
        self.complete = complete

    def query(self) -> bool:
        return self.complete


class _FakeTensorCopyAdapter:
    """A CPU-only tensor allocator and asynchronous copy oracle."""

    def __init__(self) -> None:
        self.allocations: list[_FakeTensor] = []
        self.calls: list[tuple[str, str, str, int | None]] = []
        self.fences: list[_FakeFence] = []

    def allocate_pinned(self, shape: tuple[int, ...], dtype: str) -> _FakeTensor:
        tensor = _FakeTensor(shape, dtype, "cpu:pinned", [0.0] * _size(shape))
        self.allocations.append(tensor)
        return tensor

    def copy(
        self,
        kind: str,
        source: _FakeTensor,
        destination: _FakeTensor,
        *,
        staging: _FakeTensor | None = None,
    ) -> _FakeFence:
        slot_id = None
        if staging is not None:
            slot_id = next(
                index for index, tensor in enumerate(self.allocations) if tensor is staging
            )
        self.calls.append((kind, source.device, destination.device, slot_id))
        destination.payload[:] = source.payload
        fence = _FakeFence(f"copy-{len(self.fences)}")
        self.fences.append(fence)
        return fence


def _size(shape: tuple[int, ...]) -> int:
    size = 1
    for dimension in shape:
        size *= dimension
    return size


def _move(source: object, destination: object, kind: phase3.MoveKind) -> phase3.Move:
    value = phase3.ValueKey("hidden", 1)
    return phase3.Move(
        value=value,
        source=phase3.Residency(value, source),
        destination=phase3.Residency(value, destination),
        kind=kind,
        byte_count=16,
    )


class Phase3PinnedMemoryPoolTest(unittest.TestCase):
    def test_pool_is_bounded_and_uses_typed_pinned_allocations(self) -> None:
        transfer = _transfer_runtime(self)
        adapter = _FakeTensorCopyAdapter()
        pool = transfer.PinnedMemoryPool(capacity=2, adapter=adapter)

        first = pool.acquire(shape=(4,), dtype="float32")
        second = pool.acquire(shape=(4,), dtype="float32")

        self.assertIs(first.tensor, adapter.allocations[0])
        self.assertIs(second.tensor, adapter.allocations[1])
        self.assertEqual(len(adapter.allocations), 2)
        with self.assertRaises((BufferError, RuntimeError)):
            pool.acquire(shape=(4,), dtype="float32")

    def test_unfinished_retire_fence_forbids_slot_reuse(self) -> None:
        transfer = _transfer_runtime(self)
        adapter = _FakeTensorCopyAdapter()
        pool = transfer.PinnedMemoryPool(capacity=1, adapter=adapter)
        slot = pool.acquire(shape=(4,), dtype="float32")
        retire = _FakeFence("retire", complete=False)

        pool.retire(slot, retire)
        with self.assertRaises((BufferError, RuntimeError)):
            pool.acquire(shape=(4,), dtype="float32")

        retire.complete = True
        pool.reap()
        reused = pool.acquire(shape=(4,), dtype="float32")
        self.assertIs(reused, slot)
        self.assertEqual(len(adapter.allocations), 1)


class Phase3CopyExecutorTest(unittest.TestCase):
    def test_host_staged_copy_holds_one_slot_until_second_segment_retires(self) -> None:
        transfer = _transfer_runtime(self)
        adapter = _FakeTensorCopyAdapter()
        pool = transfer.PinnedMemoryPool(capacity=1, adapter=adapter)
        executor = transfer.CopyExecutor(pool=pool, adapter=adapter, capacity=1)
        source = _FakeTensor((4,), "float32", "cuda:0", [1.0, 2.0, 3.0, 4.0])
        destination = _FakeTensor((4,), "float32", "cuda:1", [0.0] * 4)
        value = phase3.ValueKey("hidden", 1)
        request = phase3.TransferRequest(
            value=value,
            source=phase3.Residency(value, phase3.Placement("cuda", 0)),
            destination=phase3.Residency(value, phase3.Placement("cuda", 1)),
            byte_count=16,
        )
        moves = phase3.compile_moves(
            (request,),
            topology=phase3.DeviceTopology(peer_access=frozenset()),
        )

        handle = executor.submit(
            moves,
            source=source,
            destination=destination,
        )
        self.assertEqual(tuple(move.kind for move in moves), (phase3.MoveKind.D2H, phase3.MoveKind.H2D))
        self.assertEqual(len(adapter.calls), 1)
        self.assertEqual(adapter.calls[0][0], "d2h")
        self.assertEqual(pool.in_use, 1)

        adapter.fences[0].complete = True
        executor.poll()
        self.assertEqual(len(adapter.calls), 2)
        self.assertEqual(adapter.calls[1][0], "h2d")
        self.assertEqual(adapter.calls[0][3], adapter.calls[1][3])
        self.assertEqual(pool.in_use, 1)

        adapter.fences[1].complete = True
        executor.poll()
        self.assertTrue(handle.complete)
        self.assertEqual(pool.in_use, 0)
        self.assertEqual(destination.payload, source.payload)

    def test_copy_engine_capacity_rejects_second_inflight_copy(self) -> None:
        transfer = _transfer_runtime(self)
        adapter = _FakeTensorCopyAdapter()
        pool = transfer.PinnedMemoryPool(capacity=2, adapter=adapter)
        executor = transfer.CopyExecutor(pool=pool, adapter=adapter, capacity=1)
        source = _FakeTensor((2,), "float32", "cpu", [1.0, 2.0])
        destination_a = _FakeTensor((2,), "float32", "cuda:0", [0.0, 0.0])
        destination_b = _FakeTensor((2,), "float32", "cuda:0", [0.0, 0.0])
        h2d_a = (_move(phase3.Placement("cpu"), phase3.Placement("cuda", 0), phase3.MoveKind.H2D),)
        h2d_b = (_move(phase3.Placement("cpu"), phase3.Placement("cuda", 0), phase3.MoveKind.H2D),)

        first = executor.submit(h2d_a, source=source, destination=destination_a)
        with self.assertRaises((BufferError, RuntimeError)):
            executor.submit(h2d_b, source=source, destination=destination_b)

        adapter.fences[0].complete = True
        executor.poll()
        self.assertTrue(first.complete)
        second = executor.submit(h2d_b, source=source, destination=destination_b)
        self.assertFalse(second.complete)

    def test_fake_adapter_executes_cpu_h2d_without_torch_or_cuda(self) -> None:
        transfer = _transfer_runtime(self)
        adapter = _FakeTensorCopyAdapter()
        pool = transfer.PinnedMemoryPool(capacity=1, adapter=adapter)
        executor = transfer.CopyExecutor(pool=pool, adapter=adapter, capacity=1)
        source = _FakeTensor((2,), "float32", "cpu", [5.0, 6.0])
        destination = _FakeTensor((2,), "float32", "cuda:0", [0.0, 0.0])
        move = (_move(phase3.Placement("cpu"), phase3.Placement("cuda", 0), phase3.MoveKind.H2D),)

        handle = executor.submit(move, source=source, destination=destination)
        self.assertEqual(adapter.calls, [("h2d", "cpu", "cuda:0", 0)])
        self.assertFalse(handle.complete)
        adapter.fences[0].complete = True
        executor.poll()
        self.assertEqual(destination.payload, [5.0, 6.0])
        self.assertTrue(handle.complete)


if __name__ == "__main__":
    unittest.main()
