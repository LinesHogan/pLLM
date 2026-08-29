"""Local-MP transport contracts for typed workload ports."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
import unittest
from unittest import mock

import torch

if TYPE_CHECKING:
    from tllm.schedule import WorkloadCompilation

from tllm.runtime import residual_runtime
from tllm.runtime.mp.control import ControlPlane
from tllm.runtime.mp import port_wire
from tllm.runtime.mp.port_wire import (
    SharedPortPayload,
    decode_port_payload,
    encode_port_payload,
)
from tllm.runtime.mp.worker_extension import TllmWorkerExtension
from tllm.runtime.runtime import TllmRuntime
from tllm.workload.host import HostOutput
from tllm.workload.runtime_values import SKIPPED
from tllm.workload.staging import RuntimePortStore, StagedPort


@dataclass(frozen=True, slots=True)
class _Payload:
    name: str
    count: int


class _Execution:
    def __init__(self) -> None:
        self.prepared: tuple[str, str, object] | None = None
        self.committed: tuple[str, str] | None = None
        self.aborted: tuple[str, str] | None = None

    def prepare_port(self, key: str, transaction: str, value: object) -> None:
        self.prepared = (key, transaction, value)

    def commit_port(self, key: str, transaction: str) -> int:
        self.committed = (key, transaction)
        return 4

    def abort_port(self, key: str, transaction: str) -> None:
        self.aborted = (key, transaction)


class _PortOwner:
    batch: StagedPort[torch.Tensor] = StagedPort()


class _OutputOwner:
    control: HostOutput[str] = HostOutput()


class _OutputWorkload:
    def __init__(self, label: str) -> None:
        self.label = label

    def drain_port(self, port: HostOutput[str]) -> tuple[str, ...]:
        del port
        return (self.label,)


class _ExecutionView:
    def __init__(self, workload: object) -> None:
        self.workload = workload

    def workload_instance(self, workload_type: type[object]) -> object:
        del workload_type
        return self.workload


class _LocalPortWorkload:
    def __init__(self) -> None:
        self.store = RuntimePortStore()
        self.fail_commit = False

    def stage_port(self, port: StagedPort[torch.Tensor], value: torch.Tensor) -> int:
        return port.stage(self.store, value)

    def prepare_port(self, key: str, transaction: str, value: object) -> None:
        self.store.prepare(key, transaction, value)

    def commit_port(self, key: str, transaction: str) -> int:
        if self.fail_commit:
            raise RuntimeError("local commit failed")
        return self.store.commit(key, transaction)

    def abort_port(self, key: str, transaction: str) -> None:
        self.store.abort(key, transaction)


class _PortWorker:
    def __init__(self, worker_id: str = "worker-0") -> None:
        self.worker_id = worker_id
        self.store = RuntimePortStore()

    def tllm_ping(self) -> dict[str, object]:
        return {"pid": 1, "worker_id": self.worker_id}

    def tllm_prepare_workload_port(
        self, key: str, transaction: str, wire: dict[str, object]
    ) -> dict[str, object]:
        self.store.prepare(key, transaction, decode_port_payload(wire))
        return {
            "worker_id": self.worker_id,
            "key": key,
            "transaction": transaction,
            "prepared": True,
        }

    def tllm_commit_workload_port(
        self, key: str, transaction: str
    ) -> dict[str, object]:
        return {
            "worker_id": self.worker_id,
            "key": key,
            "transaction": transaction,
            "committed": True,
            "sequence": self.store.commit(key, transaction),
        }

    def tllm_abort_workload_port(
        self, key: str, transaction: str
    ) -> dict[str, object]:
        self.store.abort(key, transaction)
        return {
            "worker_id": self.worker_id,
            "key": key,
            "transaction": transaction,
            "aborted": True,
        }


class _WrongCommitIdentityWorker(_PortWorker):
    def tllm_commit_workload_port(
        self, key: str, transaction: str
    ) -> dict[str, object]:
        reply = super().tllm_commit_workload_port(key, transaction)
        reply["transaction"] = "wrong-transaction"
        return reply


class _PortEngineCore:
    def __init__(self) -> None:
        self.workers = (_PortWorker("worker-0"), _PortWorker("worker-1"))
        self.failures: set[tuple[str, int]] = set()
        self.reply_limits: dict[str, int] = {}

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[object, ...] = (),
    ) -> list[object]:
        del timeout
        replies: list[object] = []
        for index, worker in enumerate(self.workers):
            limit = self.reply_limits.get(method)
            if limit is not None and index >= limit:
                continue
            if (method, index) in self.failures:
                raise RuntimeError(f"worker {index} failed {method}")
            replies.append(getattr(worker, method)(*args))
        return replies


class _PortLLM:
    def __init__(self, core: _PortEngineCore) -> None:
        self.llm_engine = type("Engine", (), {"engine_core": core})()


class _FailingBuffer:
    def __setitem__(self, _key: object, _value: object) -> None:
        raise RuntimeError("shared-memory write failed")

    def release(self) -> None:
        pass


class _FailingReleaseBuffer(_FailingBuffer):
    def release(self) -> None:
        raise RuntimeError("buffer release failed")


class _FakeSharedMemory:
    def __init__(
        self,
        *args: object,
        buffer: _FailingBuffer | None = None,
        **kwargs: object,
    ) -> None:
        del args, kwargs
        self.name = "test-shared-memory"
        self.buf = _FailingBuffer() if buffer is None else buffer
        self.closed = False
        self.unlinked = False

    def close(self) -> None:
        self.closed = True

    def unlink(self) -> None:
        self.unlinked = True


class _CleanupFailingPayload:
    def __init__(self, payload: SharedPortPayload) -> None:
        self.wire = payload.wire
        self._payload = payload

    def release(self) -> None:
        self._payload.release()
        raise RuntimeError("cleanup failed")


def _runtime_with_local_port() -> tuple[TllmRuntime, _LocalPortWorkload]:
    workload = _LocalPortWorkload()
    runtime = TllmRuntime.__new__(TllmRuntime)
    runtime._workloads_by_type = {_PortOwner: workload}
    runtime._workload_fail_stop_reason = None
    runtime._workload_compilation = None
    runtime._port_transaction_owner = "test-runtime"
    runtime._port_transaction_sequence = 0
    return runtime, workload


class WorkloadPortTransportUnitTest(unittest.TestCase):
    def test_local_workload_never_borrows_a_foreign_active_runtime(self) -> None:
        runtime, local = _runtime_with_local_port()
        compilation = SimpleNamespace(
            decision=SimpleNamespace(signature="a" * 64),
            physical=SimpleNamespace(plan_epoch=2),
        )
        runtime._workload_compilation = cast("WorkloadCompilation", compilation)
        foreign = _LocalPortWorkload()
        foreign_execution = SimpleNamespace(
            compilation=compilation,
            plan_epoch=2,
            workload_instance=lambda _workload_type: foreign,
        )
        foreign_runtime, _ = _runtime_with_local_port()
        with (
            mock.patch("tllm.runtime.runtime._active_runtime", foreign_runtime),
            mock.patch.object(
                residual_runtime.RUNTIME,
                "workload_execution",
                foreign_execution,
            ),
            mock.patch.object(
                residual_runtime.RUNTIME,
                "workload_execution_store",
                SimpleNamespace(next_execution=foreign_execution),
            ),
        ):
            self.assertIs(runtime.local_workload(_PortOwner), local)

    def test_drain_reads_current_not_activated_next_execution(self) -> None:
        runtime, _local = _runtime_with_local_port()
        current = _ExecutionView(_OutputWorkload("current"))
        staged = _ExecutionView(_OutputWorkload("staged"))
        store = SimpleNamespace(current=current, next_execution=staged)
        with (
            mock.patch("tllm.runtime.runtime._active_runtime", runtime),
            mock.patch.object(
                residual_runtime.RUNTIME,
                "workload_execution_store",
                store,
            ),
        ):
            self.assertEqual(runtime.drain(_OutputOwner.control), ("current",))

    def test_committed_value_can_roll_back_only_before_consumption(self) -> None:
        store = RuntimePortStore()
        store.prepare("input", "test:1", "value")
        store.commit("input", "test:1")
        store.abort("input", "test:1")
        self.assertIs(store.take("input"), SKIPPED)
        store.abort("input", "test:1")

        store.prepare("input", "test:2", "value")
        store.commit("input", "test:2")
        self.assertEqual(store.take("input"), "value")
        store.prepare("input", "test:3", "value")
        store.commit("input", "test:3")
        self.assertEqual(store.take("input"), "value")
        with self.assertRaisesRegex(RuntimeError, "consumed"):
            store.abort("input", "test:3")
        with self.assertRaisesRegex(RuntimeError, "superseded"):
            store.abort("input", "test:1")

    def test_tensor_and_tensor_map_use_shared_safetensors(self) -> None:
        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        encoded = encode_port_payload(tensor)
        try:
            decoded = cast(torch.Tensor, decode_port_payload(encoded.wire))
            self.assertTrue(torch.equal(decoded, tensor))
        finally:
            encoded.release()

        encoded_map = encode_port_payload({"left": tensor, "right": tensor + 1})
        try:
            decoded_map = cast(
                dict[str, torch.Tensor], decode_port_payload(encoded_map.wire)
            )
            self.assertTrue(torch.equal(decoded_map["left"], tensor))
            self.assertTrue(torch.equal(decoded_map["right"], tensor + 1))
        finally:
            encoded_map.release()

    def test_dataclass_round_trip_uses_json_not_pickle(self) -> None:
        encoded = encode_port_payload(_Payload("batch", 3))
        decoded = decode_port_payload(encoded.wire)
        self.assertEqual(decoded, _Payload("batch", 3))

    def test_shared_memory_is_unlinked_after_write_or_close_failure(self) -> None:
        memory = mock.Mock()
        memory.close.side_effect = RuntimeError("close failed")
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            SharedPortPayload({}, memory).release()
        memory.unlink.assert_called_once_with()

        fake_memory = _FakeSharedMemory()
        with (
            mock.patch.object(port_wire, "SharedMemory", return_value=fake_memory),
            self.assertRaisesRegex(RuntimeError, "shared-memory write failed"),
        ):
            encode_port_payload(torch.ones(1))
        self.assertTrue(fake_memory.closed)
        self.assertTrue(fake_memory.unlinked)

        release_failure_memory = _FakeSharedMemory(
            buffer=_FailingReleaseBuffer()
        )
        with (
            mock.patch.object(
                port_wire, "SharedMemory", return_value=release_failure_memory
            ),
            self.assertRaisesRegex(RuntimeError, "buffer release failed"),
        ):
            encode_port_payload(torch.ones(1))
        self.assertTrue(release_failure_memory.closed)
        self.assertTrue(release_failure_memory.unlinked)

    def test_worker_extension_prepares_commits_and_aborts_exact_transaction(self) -> None:
        execution = _Execution()
        encoded = encode_port_payload(torch.ones(2, 3))
        extension = TllmWorkerExtension()
        try:
            with mock.patch.object(
                residual_runtime.RUNTIME, "workload_execution", execution
            ):
                prepared = extension.tllm_prepare_workload_port(
                    "workload:input", "tx-1", encoded.wire
                )
                committed = extension.tllm_commit_workload_port(
                    "workload:input", "tx-1"
                )
                aborted = extension.tllm_abort_workload_port(
                    "workload:input", "tx-2"
                )
        finally:
            encoded.release()
        self.assertTrue(prepared["prepared"])
        self.assertEqual(committed["sequence"], 4)
        self.assertTrue(aborted["aborted"])
        assert execution.prepared is not None
        self.assertTrue(
            torch.equal(cast(torch.Tensor, execution.prepared[2]), torch.ones(2, 3))
        )

    def test_worker_port_transaction_targets_activated_next_execution(self) -> None:
        current = _Execution()
        staged = _Execution()
        encoded = encode_port_payload(torch.ones(2, 3))
        extension = TllmWorkerExtension()
        try:
            with (
                mock.patch.object(
                    residual_runtime.RUNTIME,
                    "workload_execution",
                    current,
                ),
                mock.patch.object(
                    residual_runtime.RUNTIME,
                    "workload_execution_store",
                    SimpleNamespace(next_execution=staged),
                ),
            ):
                extension.tllm_prepare_workload_port(
                    "workload:input", "replacement-tx", encoded.wire
                )
                extension.tllm_commit_workload_port(
                    "workload:input", "replacement-tx"
                )
        finally:
            encoded.release()
        self.assertIsNone(current.prepared)
        self.assertIsNone(current.committed)
        self.assertIsNotNone(staged.prepared)
        self.assertEqual(staged.committed, ("workload:input", "replacement-tx"))

    def test_multiworker_commit_failure_rolls_back_committed_prefix(self) -> None:
        core = _PortEngineCore()
        plane = ControlPlane(_PortLLM(core))
        runtime, local = _runtime_with_local_port()
        core.failures.add(("tllm_commit_workload_port", 1))

        with (
            mock.patch(
                "tllm.runtime.mp.control.active_control_plane",
                return_value=plane,
            ),
            self.assertRaisesRegex(RuntimeError, "worker 1 failed"),
        ):
            runtime.stage(_PortOwner.batch, torch.ones(2, 3))

        self.assertIs(local.store.take(_PortOwner.batch.key), SKIPPED)
        for worker in core.workers:
            self.assertIs(worker.store.take(_PortOwner.batch.key), SKIPPED)

        core.failures.clear()
        with mock.patch(
            "tllm.runtime.mp.control.active_control_plane",
            return_value=plane,
        ):
            self.assertEqual(runtime.stage(_PortOwner.batch, torch.ones(2, 3)), 1)
        for worker in core.workers:
            value = cast(torch.Tensor, worker.store.take(_PortOwner.batch.key))
            self.assertTrue(torch.equal(value, torch.ones(2, 3)))

    def test_truncated_prepare_reply_is_rejected_and_rolled_back(self) -> None:
        core = _PortEngineCore()
        core.reply_limits["tllm_prepare_workload_port"] = 1
        plane = ControlPlane(_PortLLM(core))
        runtime, local = _runtime_with_local_port()

        with (
            mock.patch(
                "tllm.runtime.mp.control.active_control_plane",
                return_value=plane,
            ),
            self.assertRaisesRegex(RuntimeError, "prepare quorum"),
        ):
            runtime.stage(_PortOwner.batch, torch.ones(2, 3))

        self.assertIs(local.store.take(_PortOwner.batch.key), SKIPPED)
        for worker in core.workers:
            self.assertIs(worker.store.take(_PortOwner.batch.key), SKIPPED)

    def test_truncated_commit_reply_is_rejected_and_rolled_back(self) -> None:
        core = _PortEngineCore()
        core.reply_limits["tllm_commit_workload_port"] = 1
        plane = ControlPlane(_PortLLM(core))
        runtime, local = _runtime_with_local_port()

        with (
            mock.patch(
                "tllm.runtime.mp.control.active_control_plane",
                return_value=plane,
            ),
            self.assertRaisesRegex(RuntimeError, "commit quorum"),
        ):
            runtime.stage(_PortOwner.batch, torch.ones(2, 3))

        self.assertIs(local.store.take(_PortOwner.batch.key), SKIPPED)
        for worker in core.workers:
            self.assertIs(worker.store.take(_PortOwner.batch.key), SKIPPED)

    def test_duplicate_ping_identity_is_rejected_before_prepare(self) -> None:
        core = _PortEngineCore()
        core.workers = (_PortWorker("duplicate"), _PortWorker("duplicate"))
        plane = ControlPlane(_PortLLM(core))
        runtime, local = _runtime_with_local_port()

        with (
            mock.patch(
                "tllm.runtime.mp.control.active_control_plane",
                return_value=plane,
            ),
            self.assertRaisesRegex(RuntimeError, "duplicate worker identities"),
        ):
            runtime.stage(_PortOwner.batch, torch.ones(2, 3))

        self.assertIs(local.store.take(_PortOwner.batch.key), SKIPPED)
        for worker in core.workers:
            self.assertIs(worker.store.take(_PortOwner.batch.key), SKIPPED)

    def test_local_commit_failure_rolls_back_remote_commit_quorum(self) -> None:
        core = _PortEngineCore()
        plane = ControlPlane(_PortLLM(core))
        runtime, local = _runtime_with_local_port()
        local.fail_commit = True

        with (
            mock.patch(
                "tllm.runtime.mp.control.active_control_plane",
                return_value=plane,
            ),
            self.assertRaisesRegex(RuntimeError, "local commit failed"),
        ):
            runtime.stage(_PortOwner.batch, torch.ones(2, 3))

        self.assertIs(local.store.take(_PortOwner.batch.key), SKIPPED)
        for worker in core.workers:
            self.assertIs(worker.store.take(_PortOwner.batch.key), SKIPPED)

    def test_cleanup_failure_does_not_misreport_committed_publication(self) -> None:
        core = _PortEngineCore()
        plane = ControlPlane(_PortLLM(core))
        runtime, local = _runtime_with_local_port()
        payload = cast(
            SharedPortPayload, encode_port_payload(torch.ones(2, 3))
        )

        with (
            mock.patch(
                "tllm.runtime.mp.control.active_control_plane",
                return_value=plane,
            ),
            mock.patch.object(
                port_wire,
                "encode_port_payload",
                return_value=_CleanupFailingPayload(payload),
            ),
            self.assertWarnsRegex(RuntimeWarning, "cleanup failed"),
        ):
            sequence = runtime.stage(_PortOwner.batch, torch.ones(2, 3))

        self.assertEqual(sequence, 1)
        local_value = cast(torch.Tensor, local.store.take(_PortOwner.batch.key))
        self.assertTrue(torch.equal(local_value, torch.ones(2, 3)))
        for worker in core.workers:
            value = cast(torch.Tensor, worker.store.take(_PortOwner.batch.key))
            self.assertTrue(torch.equal(value, torch.ones(2, 3)))

    def test_parent_worker_sequence_mismatch_rolls_back_and_fail_stops(self) -> None:
        core = _PortEngineCore()
        plane = ControlPlane(_PortLLM(core))
        runtime, local = _runtime_with_local_port()
        self.assertEqual(
            local.stage_port(_PortOwner.batch, torch.zeros(1)),
            1,
        )
        _ = local.store.take(_PortOwner.batch.key)

        with (
            mock.patch(
                "tllm.runtime.mp.control.active_control_plane",
                return_value=plane,
            ),
            self.assertRaisesRegex(RuntimeError, "parent/worker sequence"),
        ):
            runtime.stage(_PortOwner.batch, torch.ones(2, 3))

        self.assertIs(local.store.take(_PortOwner.batch.key), SKIPPED)
        for worker in core.workers:
            self.assertIs(worker.store.take(_PortOwner.batch.key), SKIPPED)
        with self.assertRaisesRegex(RuntimeError, "fail-stopped"):
            runtime.stage(_PortOwner.batch, torch.ones(2, 3))

    def test_mismatched_commit_vote_is_rejected_and_rolled_back(self) -> None:
        core = _PortEngineCore()
        core.workers = (
            _PortWorker("worker-0"),
            _WrongCommitIdentityWorker("worker-1"),
        )
        plane = ControlPlane(_PortLLM(core))
        runtime, local = _runtime_with_local_port()

        with (
            mock.patch(
                "tllm.runtime.mp.control.active_control_plane",
                return_value=plane,
            ),
            self.assertRaisesRegex(RuntimeError, "commit reply identity"),
        ):
            runtime.stage(_PortOwner.batch, torch.ones(2, 3))

        self.assertIs(local.store.take(_PortOwner.batch.key), SKIPPED)
        for worker in core.workers:
            self.assertIs(worker.store.take(_PortOwner.batch.key), SKIPPED)

    def test_unconfirmed_rollback_fail_stops_future_publication(self) -> None:
        core = _PortEngineCore()
        core.failures.update(
            {
                ("tllm_commit_workload_port", 1),
                ("tllm_abort_workload_port", 1),
            }
        )
        plane = ControlPlane(_PortLLM(core))
        runtime, _local = _runtime_with_local_port()

        with (
            mock.patch(
                "tllm.runtime.mp.control.active_control_plane",
                return_value=plane,
            ),
            self.assertRaisesRegex(RuntimeError, "rollback quorum was not confirmed"),
        ):
            runtime.stage(_PortOwner.batch, torch.ones(2, 3))

        with self.assertRaisesRegex(RuntimeError, "fail-stopped"):
            runtime.stage(_PortOwner.batch, torch.ones(2, 3))


if __name__ == "__main__":
    unittest.main()
