"""Owned shared input/output storage and the closed multiprocessing codec."""

import json
import os
from contextlib import ExitStack
from dataclasses import replace

import pytest
import torch

from pllm.contracts.codec import decode_boundary, encode_boundary
from pllm.contracts.request_identity import RequestIdentity
from pllm.execution.host_runtime import CompletionRows
from pllm.execution.shared_tensor import SharedTensor
from pllm.host import CausalTrainBatch, Completion, CompletionBatch, FinishReason
from pllm.io import StateImport

IDENTITIES = (RequestIdentity("a", 9, 0), RequestIdentity("b", 7, 0), RequestIdentity("c", 11, 0))
BATCH = CompletionBatch(
    tuple(Completion(identity, 0, "feedback", (1,), "1", FinishReason.LENGTH) for identity in IDENTITIES)
)


def test_codec_round_trips_payloads_and_shared_tensors_own_their_mapping() -> None:
    with ExitStack() as lifetime:
        source = torch.arange(6, dtype=torch.float32).reshape(2, 3).T
        descriptor = SharedTensor.export(source, lifetime)
        for value, target in (
            (CausalTrainBatch(descriptor, descriptor, weights=descriptor), CausalTrainBatch[SharedTensor]),
            (StateImport((descriptor,)), StateImport[SharedTensor]),
            (BATCH, CompletionBatch),
        ):
            assert decode_boundary(encode_boundary(value), target) == value
        wire = json.loads(encode_boundary(BATCH))
        wire["entries"][0]["finish_reason"] = "unknown"
        with pytest.raises(ValueError):
            decode_boundary(json.dumps(wire), CompletionBatch)
        mapped = descriptor.map(os.getpid())
        torch.testing.assert_close(mapped, source)
        with pytest.raises(ValueError):
            replace(descriptor, process_id=0).map(os.getpid())
    # The mapping outlives the exporter's lifetime and is a private copy.
    mapped.fill_(7)
    assert torch.all(mapped == 7) and not torch.all(source == 7)


def test_completion_capture_orders_rows_and_requires_all_identities() -> None:
    hidden = torch.zeros(3, 2)
    capture = CompletionRows(IDENTITIES, hidden)
    with pytest.raises(RuntimeError):
        capture.require_complete()
    capture.write((IDENTITIES[2], IDENTITIES[0]), torch.tensor([[2.0, 2.0], [0.0, 0.0]]))
    capture.write((IDENTITIES[1], IDENTITIES[0]), torch.tensor([[1.0, 1.0], [3.0, 3.0]]))
    capture.require_complete()
    torch.testing.assert_close(hidden, torch.tensor([[3.0, 3.0], [1.0, 1.0], [2.0, 2.0]]))
