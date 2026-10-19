"""Owned shared input/output storage and the closed multiprocessing codec."""

import json
import os
from contextlib import ExitStack
from dataclasses import replace

import pytest
import torch

from pllm.authoring import CausalTrainBatch, StateTensorUpdate
from pllm.contracts.feedback import Completion, CompletionBatch, FinishReason
from pllm.contracts.request_identity import RequestIdentity
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.host_runtime import CompletionCapture
from pllm.execution.shared_tensor import CpuFileStorage, SharedTensor

IDENTITIES = (RequestIdentity("a", 9, 0), RequestIdentity("b", 7, 0), RequestIdentity("c", 11, 0))


def _completion_batch() -> CompletionBatch:
    return CompletionBatch(
        tuple(Completion(identity, 0, "feedback", (1,), "1", FinishReason.LENGTH) for identity in IDENTITIES)
    )


def test_mp_codec_round_trips_registered_payloads() -> None:
    with ExitStack() as lifetime:
        descriptor = SharedTensor.export(torch.arange(6, dtype=torch.float32).reshape(2, 3), lifetime)
        for value, target in (
            (CausalTrainBatch(descriptor, descriptor), CausalTrainBatch[SharedTensor]),
            (StateTensorUpdate((descriptor,)), StateTensorUpdate[SharedTensor]),
            (_completion_batch(), CompletionBatch),
        ):
            assert decode_boundary(encode_boundary(value), target) == value


@pytest.mark.parametrize(
    "field, malformed",
    (
        ("output_index", True),
        ("token_ids", [False]),
        ("identity", "not-an-identity"),
        ("finish_reason", "unknown"),
    ),
)
def test_codec_rejects_malformed_wire_entries(field, malformed) -> None:
    wire = json.loads(encode_boundary(_completion_batch()))
    wire["entries"][0][field] = malformed
    with pytest.raises(ValueError):
        decode_boundary(json.dumps(wire), CompletionBatch)


def test_shared_tensor_mapping_is_owned_by_the_mapping() -> None:
    with ExitStack() as lifetime:
        source = torch.arange(6, dtype=torch.float32).reshape(2, 3).T
        descriptor = SharedTensor.export(source, lifetime)
        mapped = descriptor.map(os.getpid())
        assert isinstance(descriptor.storage, CpuFileStorage)
        torch.testing.assert_close(mapped, source)
        for bad in (
            replace(descriptor, process_id=0),
            replace(descriptor, storage=replace(descriptor.storage, inode=0)),
            replace(descriptor, field=replace(descriptor.field, shape=(6, 2))),
        ):
            with pytest.raises(ValueError):
                bad.map(os.getpid())
        with pytest.raises(OSError):
            replace(descriptor, storage=replace(descriptor.storage, descriptor=2**30)).map(os.getpid())
    # The mapping, not an FD or remote version ledger, owns this non-recycled output.
    mapped.fill_(7)
    assert torch.all(mapped == 7) and not torch.all(source == 7)


def test_completion_capture_orders_rows_and_requires_all_identities() -> None:
    hidden = torch.zeros(3, 2)
    capture = CompletionCapture(IDENTITIES, hidden)
    with pytest.raises(RuntimeError):
        capture.require_complete()
    with pytest.raises(ValueError):
        capture.write(IDENTITIES, torch.ones(3, 1))
    capture.write((IDENTITIES[2], IDENTITIES[0]), torch.tensor([[2.0, 2.0], [0.0, 0.0]]))
    capture.write((IDENTITIES[1], IDENTITIES[0]), torch.tensor([[1.0, 1.0], [3.0, 3.0]]))
    batch = _completion_batch()
    completed = capture.finish(batch)
    assert completed.hidden is hidden and completed.completions is batch
    torch.testing.assert_close(hidden, torch.tensor([[3.0, 3.0], [1.0, 1.0], [2.0, 2.0]]))
