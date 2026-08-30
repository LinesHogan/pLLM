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
    decode_inline_port_payload,
    decode_port_payload,
    encode_inline_port_payload,
    encode_port_payload,
)
from tllm.runtime.mp.worker_extension import TllmWorkerExtension
from tllm.runtime.runtime import (
    StagedPortTransactionReceipt,
    TllmRuntime,
)
from tllm.contracts.workload_ports import (
    PortTransactionStatus,
    SKIPPED,
    StagedPortTransactionResolution,
)
from tllm.workload.host import HostOutput
from tllm.workload.staging import (
    RuntimePortStore,
    StagedPort,
)


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
    metadata: StagedPort[_Payload] = StagedPort()


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

    def port_transaction_status(
        self, key: str, transaction: str
    ) -> PortTransactionStatus:
        return self.store.transaction_status(key, transaction)

    def acknowledge_remote_port_consumed(
        self, key: str, transaction: str
    ) -> None:
        self.store.acknowledge_remote_consumed(key, transaction)


class _PortRuntime(TllmRuntime[_LocalPortWorkload]):
    @classmethod
    def create(cls, workload: _LocalPortWorkload) -> _PortRuntime:
        runtime = cls.__new__(cls)
        runtime._workloads_by_type = {
            _PortOwner: workload,
            _LocalPortWorkload: workload,
        }
        runtime._workload_fail_stop_reason = None
        runtime._workload_compilation = cast(
            "WorkloadCompilation",
            SimpleNamespace(
                decision=SimpleNamespace(signature="a" * 64),
                physical=SimpleNamespace(plan_epoch=1),
            ),
        )
        runtime._port_transaction_owner = f"test-runtime-{id(runtime)}"
        runtime._port_transaction_sequence = 0
        runtime._staged_port_transactions = {}
        return runtime

    @property
    def fail_stopped(self) -> bool:
        return self._workload_fail_stop_reason is not None

    def set_compilation_for_test(
        self, compilation: WorkloadCompilation
    ) -> None:
        self._workload_compilation = compilation


@dataclass(slots=True)
class _ForeignExecution:
    compilation: WorkloadCompilation
    plan_epoch: int
    workload: _LocalPortWorkload

    def workload_instance(
        self, workload_type: type[_LocalPortWorkload]
    ) -> _LocalPortWorkload:
        del workload_type
        return self.workload


class _PortWorker:
    def __init__(self, worker_id: str = "worker-0") -> None:
        self.worker_id = worker_id
        self.store = RuntimePortStore()
        self.signature = "a" * 64
        self.plan_epoch = 1

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

    def tllm_abort_workload_port_transaction(
        self,
        keys: list[str],
        transaction: str,
        signature: str,
        plan_epoch: int,
    ) -> dict[str, object]:
        if signature != self.signature or plan_epoch != self.plan_epoch:
            raise RuntimeError("worker port transaction epoch mismatch")
        statuses = tuple(
            self.store.transaction_status(key, transaction) for key in keys
        )
        if all(status is PortTransactionStatus.PENDING for status in statuses):
            for key in keys:
                self.store.abort(key, transaction)
            resolution = "aborted"
        elif all(
            status is PortTransactionStatus.CONSUMED for status in statuses
        ):
            resolution = "consumed"
        else:
            raise RuntimeError("worker port transaction members diverged")
        return {
            "worker_id": self.worker_id,
            "keys": keys,
            "transaction": transaction,
            "signature": signature,
            "plan_epoch": plan_epoch,
            "resolution": resolution,
        }

    def tllm_confirm_workload_port_transaction_consumed(
        self,
        keys: list[str],
        transaction: str,
        signature: str,
        plan_epoch: int,
    ) -> dict[str, object]:
        if signature != self.signature or plan_epoch != self.plan_epoch:
            raise RuntimeError("worker port transaction epoch mismatch")
        if not all(
            self.store.transaction_status(key, transaction)
            is PortTransactionStatus.CONSUMED
            for key in keys
        ):
            raise RuntimeError("worker port transaction was not consumed")
        return {
            "worker_id": self.worker_id,
            "keys": keys,
            "transaction": transaction,
            "signature": signature,
            "plan_epoch": plan_epoch,
            "resolution": "consumed",
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


def _runtime_with_local_port() -> tuple[_PortRuntime, _LocalPortWorkload]:
    workload = _LocalPortWorkload()
    return _PortRuntime.create(workload), workload


class WorkloadPortTransportUnitTest(unittest.TestCase):
    def test_inline_output_wire_round_trips_tensor_and_typed_dataclass(self) -> None:
        tensor = torch.arange(4, dtype=torch.float32)
        decoded_tensor = decode_inline_port_payload(
            encode_inline_port_payload(tensor)
        )
        self.assertTrue(torch.equal(cast(torch.Tensor, decoded_tensor), tensor))
        self.assertEqual(
            decode_inline_port_payload(
                encode_inline_port_payload(_Payload("output", 4))
            ),
            _Payload("output", 4),
        )

    def test_stage_pair_commits_or_rolls_back_both_ports_as_one_transaction(self) -> None:
        runtime, local = _runtime_with_local_port()
        tensor = torch.ones(2)
        metadata = _Payload("pair", 2)

        receipt = runtime.stage_pair(
            _PortOwner.batch,
            tensor,
            _PortOwner.metadata,
            metadata,
        )
        self.assertEqual(
            (receipt.first_sequence, receipt.second_sequence),
            (1, 1),
        )
        self.assertIs(local.store.take(_PortOwner.batch.key), tensor)
        self.assertEqual(local.store.take(_PortOwner.metadata.key), metadata)
        self.assertIs(
            runtime.resolve_staged_transaction(receipt),
            StagedPortTransactionResolution.CONSUMED,
        )
        runtime.settle_staged_transaction(receipt)

        _PortOwner.metadata.stage(local.store, _Payload("occupied", 1))
        with self.assertRaisesRegex(RuntimeError, "pending value"):
            runtime.stage_pair(
                _PortOwner.batch,
                tensor,
                _PortOwner.metadata,
                metadata,
            )
        self.assertIs(local.store.take(_PortOwner.batch.key), SKIPPED)

    def test_pending_stage_pair_receipt_aborts_and_can_be_replaced(self) -> None:
        runtime, local = _runtime_with_local_port()
        receipt = runtime.stage_pair(
            _PortOwner.batch,
            torch.ones(2),
            _PortOwner.metadata,
            _Payload("first", 2),
        )

        self.assertIs(
            runtime.resolve_staged_transaction(receipt),
            StagedPortTransactionResolution.CANCELLED,
        )
        self.assertIs(local.store.take(_PortOwner.batch.key), SKIPPED)
        self.assertIs(local.store.take(_PortOwner.metadata.key), SKIPPED)

        replacement = runtime.stage_pair(
            _PortOwner.batch,
            torch.ones(2),
            _PortOwner.metadata,
            _Payload("replacement", 2),
        )
        self.assertEqual(
            (replacement.first_sequence, replacement.second_sequence),
            (1, 1),
        )
        self.assertIs(
            runtime.resolve_staged_transaction(replacement),
            StagedPortTransactionResolution.CANCELLED,
        )

    def test_stage_pair_receipt_identity_is_exact(self) -> None:
        runtime, _local = _runtime_with_local_port()
        foreign_runtime, _foreign = _runtime_with_local_port()
        receipt = runtime.stage_pair(
            _PortOwner.batch,
            torch.ones(2),
            _PortOwner.metadata,
            _Payload("identity", 2),
        )
        forged = StagedPortTransactionReceipt(
            runtime_owner=receipt.runtime_owner,
            transaction=receipt.transaction,
            first_port=receipt.first_port,
            second_port=receipt.second_port,
            first_sequence=receipt.first_sequence + 1,
            second_sequence=receipt.second_sequence,
        )

        with self.assertRaisesRegex(ValueError, "does not match its journal"):
            runtime.resolve_staged_transaction(forged)
        with self.assertRaisesRegex(ValueError, "another runtime"):
            foreign_runtime.resolve_staged_transaction(receipt)
        self.assertIs(
            runtime.resolve_staged_transaction(receipt),
            StagedPortTransactionResolution.CANCELLED,
        )

    def test_consumed_stage_pair_receipt_requires_explicit_settlement(
        self,
    ) -> None:
        runtime, local = _runtime_with_local_port()
        receipt = runtime.stage_pair(
            _PortOwner.batch,
            torch.ones(2),
            _PortOwner.metadata,
            _Payload("consumed", 2),
        )
        local.store.take(_PortOwner.batch.key)
        local.store.take(_PortOwner.metadata.key)

        with self.assertRaisesRegex(RuntimeError, "unresolved transaction"):
            runtime.stage_pair(
                _PortOwner.batch,
                torch.ones(2),
                _PortOwner.metadata,
                _Payload("too-early", 2),
            )
        self.assertIs(
            runtime.resolve_staged_transaction(receipt),
            StagedPortTransactionResolution.CONSUMED,
        )
        self.assertFalse(runtime.fail_stopped)
        runtime.settle_staged_transaction(receipt)

    def test_stage_pair_releases_first_payload_when_second_encode_fails(self) -> None:
        runtime, _local = _runtime_with_local_port()
        first_payload = mock.Mock(wire={"kind": "first"})
        plane = mock.Mock()
        with (
            mock.patch(
                "tllm.runtime.mp.control.active_control_plane",
                return_value=plane,
            ),
            mock.patch(
                "tllm.runtime.mp.port_wire.encode_port_payload",
                side_effect=(first_payload, RuntimeError("second encode failed")),
            ),
            self.assertRaisesRegex(RuntimeError, "second encode failed"),
        ):
            runtime.stage_pair(
                _PortOwner.batch,
                torch.ones(1),
                _PortOwner.metadata,
                _Payload("metadata", 1),
            )
        first_payload.release.assert_called_once_with()

    def test_stage_pair_sequence_divergence_rolls_back_and_fail_stops(self) -> None:
        runtime, local = _runtime_with_local_port()
        plane = mock.Mock()
        plane.worker_quorum.return_value = SimpleNamespace(worker_ids=frozenset({"w"}))
        plane.commit_workload_port.return_value = (
            SimpleNamespace(sequence=9),
        )
        payloads = (
            mock.Mock(wire={"kind": "first"}),
            mock.Mock(wire={"kind": "second"}),
        )
        with (
            mock.patch(
                "tllm.runtime.mp.control.active_control_plane",
                return_value=plane,
            ),
            mock.patch(
                "tllm.runtime.mp.port_wire.encode_port_payload",
                side_effect=payloads,
            ),
            self.assertRaisesRegex(RuntimeError, "sequence quorum mismatch"),
        ):
            runtime.stage_pair(
                _PortOwner.batch,
                torch.ones(1),
                _PortOwner.metadata,
                _Payload("metadata", 1),
            )
        self.assertTrue(runtime.fail_stopped)
        self.assertIs(local.store.take(_PortOwner.batch.key), SKIPPED)
        self.assertIs(local.store.take(_PortOwner.metadata.key), SKIPPED)

    def test_stage_pair_commits_tensor_and_metadata_to_worker_quorum(self) -> None:
        core = _PortEngineCore()
        plane = ControlPlane(_PortLLM(core))
        runtime, local = _runtime_with_local_port()
        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        metadata = _Payload("pair", 2)

        with mock.patch(
            "tllm.runtime.mp.control.active_control_plane",
            return_value=plane,
        ):
            receipt = runtime.stage_pair(
                _PortOwner.batch,
                tensor,
                _PortOwner.metadata,
                metadata,
            )
            self.assertEqual(
                (receipt.first_sequence, receipt.second_sequence),
                (1, 1),
            )

        self.assertTrue(
            torch.equal(
                cast(torch.Tensor, local.store.take(_PortOwner.batch.key)), tensor
            )
        )
        self.assertEqual(local.store.take(_PortOwner.metadata.key), metadata)
        for worker in core.workers:
            self.assertTrue(
                torch.equal(
                    cast(
                        torch.Tensor,
                        worker.store.take(_PortOwner.batch.key),
                    ),
                    tensor,
                )
            )
            self.assertEqual(
                worker.store.take(_PortOwner.metadata.key), metadata
            )

    def test_mp_consumption_ack_retires_parent_mirror_and_preserves_sequence(
        self,
    ) -> None:
        core = _PortEngineCore()
        plane = ControlPlane(_PortLLM(core))
        runtime, local = _runtime_with_local_port()

        with mock.patch(
            "tllm.runtime.mp.control.active_control_plane",
            return_value=plane,
        ):
            receipt = runtime.stage_pair(
                _PortOwner.batch,
                torch.ones(2),
                _PortOwner.metadata,
                _Payload("first", 2),
            )
            for worker in core.workers:
                worker.store.take(_PortOwner.batch.key)
                worker.store.take(_PortOwner.metadata.key)
            self.assertIs(
                runtime.resolve_staged_transaction(receipt),
                StagedPortTransactionResolution.CONSUMED,
            )
            runtime.settle_staged_transaction(receipt)

            self.assertIs(local.store.take(_PortOwner.batch.key), SKIPPED)
            self.assertIs(local.store.take(_PortOwner.metadata.key), SKIPPED)
            replacement = runtime.stage_pair(
                _PortOwner.batch,
                torch.ones(2),
                _PortOwner.metadata,
                _Payload("replacement", 2),
            )
            self.assertEqual(
                (replacement.first_sequence, replacement.second_sequence),
                (2, 2),
            )
            self.assertIs(
                runtime.resolve_staged_transaction(replacement),
                StagedPortTransactionResolution.CANCELLED,
            )

        self.assertIs(local.store.take(_PortOwner.batch.key), SKIPPED)
        self.assertIs(local.store.take(_PortOwner.metadata.key), SKIPPED)
        for worker in core.workers:
            self.assertIs(worker.store.take(_PortOwner.batch.key), SKIPPED)
            self.assertIs(worker.store.take(_PortOwner.metadata.key), SKIPPED)

    def test_mp_receipt_pins_the_original_control_plane(self) -> None:
        core = _PortEngineCore()
        plane = ControlPlane(_PortLLM(core))
        runtime, local = _runtime_with_local_port()
        with mock.patch(
            "tllm.runtime.mp.control.active_control_plane",
            return_value=plane,
        ):
            receipt = runtime.stage_pair(
                _PortOwner.batch,
                torch.ones(2),
                _PortOwner.metadata,
                _Payload("pending", 2),
            )

        self.assertIs(
            runtime.resolve_staged_transaction(receipt),
            StagedPortTransactionResolution.CANCELLED,
        )
        self.assertIs(local.store.take(_PortOwner.batch.key), SKIPPED)
        self.assertIs(local.store.take(_PortOwner.metadata.key), SKIPPED)

    def test_mp_worker_consumption_divergence_fail_stops_resolution(self) -> None:
        core = _PortEngineCore()
        plane = ControlPlane(_PortLLM(core))
        runtime, _local = _runtime_with_local_port()
        with mock.patch(
            "tllm.runtime.mp.control.active_control_plane",
            return_value=plane,
        ):
            receipt = runtime.stage_pair(
                _PortOwner.batch,
                torch.ones(2),
                _PortOwner.metadata,
                _Payload("diverged", 2),
            )
        core.workers[0].store.take(_PortOwner.batch.key)
        core.workers[0].store.take(_PortOwner.metadata.key)

        with self.assertRaisesRegex(RuntimeError, "worker states diverged"):
            runtime.resolve_staged_transaction(receipt)
        self.assertTrue(runtime.fail_stopped)

    def test_mp_receipt_rejects_resolution_against_another_epoch(self) -> None:
        core = _PortEngineCore()
        plane = ControlPlane(_PortLLM(core))
        runtime, _local = _runtime_with_local_port()
        with mock.patch(
            "tllm.runtime.mp.control.active_control_plane",
            return_value=plane,
        ):
            receipt = runtime.stage_pair(
                _PortOwner.batch,
                torch.ones(2),
                _PortOwner.metadata,
                _Payload("epoch", 2),
            )
        for worker in core.workers:
            worker.signature = "b" * 64
            worker.plan_epoch = 2

        with self.assertRaisesRegex(RuntimeError, "epoch mismatch"):
            runtime.resolve_staged_transaction(receipt)
        self.assertTrue(runtime.fail_stopped)

    def test_mp_transaction_resolution_quorum_failure_fail_stops(self) -> None:
        core = _PortEngineCore()
        plane = ControlPlane(_PortLLM(core))
        runtime, _local = _runtime_with_local_port()
        with mock.patch(
            "tllm.runtime.mp.control.active_control_plane",
            return_value=plane,
        ):
            receipt = runtime.stage_pair(
                _PortOwner.batch,
                torch.ones(2),
                _PortOwner.metadata,
                _Payload("resolution-quorum", 2),
            )
        core.failures.add(("tllm_abort_workload_port_transaction", 1))

        with self.assertRaisesRegex(RuntimeError, "worker 1 failed"):
            runtime.resolve_staged_transaction(receipt)
        self.assertTrue(runtime.fail_stopped)

    def test_mp_transaction_settlement_quorum_failure_fail_stops(self) -> None:
        core = _PortEngineCore()
        plane = ControlPlane(_PortLLM(core))
        runtime, _local = _runtime_with_local_port()
        with mock.patch(
            "tllm.runtime.mp.control.active_control_plane",
            return_value=plane,
        ):
            receipt = runtime.stage_pair(
                _PortOwner.batch,
                torch.ones(2),
                _PortOwner.metadata,
                _Payload("settlement-quorum", 2),
            )
        for worker in core.workers:
            worker.store.take(_PortOwner.batch.key)
            worker.store.take(_PortOwner.metadata.key)
        self.assertIs(
            runtime.resolve_staged_transaction(receipt),
            StagedPortTransactionResolution.CONSUMED,
        )
        core.failures.add(
            ("tllm_confirm_workload_port_transaction_consumed", 1)
        )

        with self.assertRaisesRegex(RuntimeError, "worker 1 failed"):
            runtime.settle_staged_transaction(receipt)
        self.assertTrue(runtime.fail_stopped)

    def test_stage_pair_unconfirmed_rollback_fail_stops_runtime(self) -> None:
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
            self.assertRaisesRegex(
                RuntimeError, "rollback quorum was not confirmed"
            ),
        ):
            runtime.stage_pair(
                _PortOwner.batch,
                torch.ones(2, 3),
                _PortOwner.metadata,
                _Payload("pair", 2),
            )

        with self.assertRaisesRegex(RuntimeError, "fail-stopped"):
            runtime.stage_pair(
                _PortOwner.batch,
                torch.ones(2, 3),
                _PortOwner.metadata,
                _Payload("pair", 2),
            )

    def test_local_workload_never_borrows_a_foreign_active_runtime(self) -> None:
        runtime, local = _runtime_with_local_port()
        compilation = SimpleNamespace(
            decision=SimpleNamespace(signature="a" * 64),
            physical=SimpleNamespace(plan_epoch=2),
        )
        typed_compilation = cast("WorkloadCompilation", compilation)
        runtime.set_compilation_for_test(typed_compilation)
        foreign = _LocalPortWorkload()
        foreign_execution = _ForeignExecution(
            compilation=typed_compilation,
            plan_epoch=2,
            workload=foreign,
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
            self.assertIs(runtime.local_workload(local), local)

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

    def test_worker_port_transaction_wire_is_exactly_typed(self) -> None:
        extension = TllmWorkerExtension()
        invalid = (
            ("not-an-array", "tx-1", "a" * 64, 1),
            ([1], "tx-1", "a" * 64, 1),
            ([""], "tx-1", "a" * 64, 1),
            (["input"], "", "a" * 64, 1),
            (["input"], "tx-1", "not-a-signature", 1),
            (["input"], "tx-1", "a" * 64, True),
        )
        for keys, transaction, signature, plan_epoch in invalid:
            with (
                self.subTest(
                    keys=keys,
                    transaction=transaction,
                    signature=signature,
                    plan_epoch=plan_epoch,
                ),
                self.assertRaises((TypeError, ValueError)),
            ):
                extension.tllm_abort_workload_port_transaction(
                    keys,
                    transaction,
                    signature,
                    plan_epoch,
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
