"""Local-MP transport contracts for typed workload ports."""

from __future__ import annotations

from dataclasses import dataclass
import unittest
from unittest import mock

import torch

from tllm.runtime import residual_runtime
from tllm.runtime.mp.port_wire import decode_port_payload, encode_port_payload
from tllm.runtime.mp.worker_extension import TllmWorkerExtension


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


class WorkloadPortTransportUnitTest(unittest.TestCase):
    def test_tensor_and_tensor_map_use_shared_safetensors(self) -> None:
        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        encoded = encode_port_payload(tensor)
        try:
            decoded = decode_port_payload(encoded.wire)
            self.assertTrue(torch.equal(decoded, tensor))
        finally:
            encoded.release()

        encoded_map = encode_port_payload({"left": tensor, "right": tensor + 1})
        try:
            decoded_map = decode_port_payload(encoded_map.wire)
            self.assertTrue(torch.equal(decoded_map["left"], tensor))
            self.assertTrue(torch.equal(decoded_map["right"], tensor + 1))
        finally:
            encoded_map.release()

    def test_dataclass_round_trip_uses_json_not_pickle(self) -> None:
        encoded = encode_port_payload(_Payload("batch", 3))
        decoded = decode_port_payload(encoded.wire)
        self.assertEqual(decoded, _Payload("batch", 3))

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
        self.assertTrue(torch.equal(execution.prepared[2], torch.ones(2, 3)))


if __name__ == "__main__":
    unittest.main()
