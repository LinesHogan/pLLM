"""Owned shared input/output storage and the closed multiprocessing codec."""

from dataclasses import replace
from contextlib import ExitStack
import json
import os

import pytest
import torch

from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.authoring import CausalTrainBatch, StateTensorUpdate
from pllm.execution.shared_tensor import CpuFileStorage, SharedTensor
from pllm.execution.host_runtime import CompletionCapture
from pllm.contracts.feedback import Completion, CompletionBatch, FinishReason
from pllm.contracts.request_identity import RequestIdentity


def test_closed_mp_codec_round_trips_registered_payload_and_shared_tensors() -> None:
    with ExitStack() as lifetime:
        source = torch.arange(6, dtype=torch.float32).reshape(2, 3).T
        descriptor = SharedTensor.export(source, lifetime)
        for value, target in ((CausalTrainBatch(descriptor, descriptor), CausalTrainBatch[SharedTensor]),
                              (StateTensorUpdate((descriptor,)), StateTensorUpdate[SharedTensor])):
            assert decode_boundary(encode_boundary(value), target) == value
        mapped = descriptor.map(os.getpid())
        storage = descriptor.storage
        assert isinstance(storage, CpuFileStorage)
        torch.testing.assert_close(mapped, source)
        for bad in (replace(descriptor, process_id=0), replace(descriptor, storage=replace(storage, inode=0)),
                    replace(descriptor, field=replace(descriptor.field, shape=(6, 2)))):
            with pytest.raises(ValueError):
                bad.map(os.getpid())
        with pytest.raises(OSError):
            replace(descriptor, storage=replace(storage, descriptor=2**30)).map(os.getpid())
    # The mapping, not an FD or remote version ledger, owns this non-recycled output.
    mapped.fill_(7)
    assert torch.all(mapped == 7) and not torch.all(source == 7)
    identities = (RequestIdentity("a", 9, 0), RequestIdentity("b", 7, 0), RequestIdentity("c", 11, 0))
    capture = CompletionCapture(identities, mapped)
    with pytest.raises(RuntimeError, match="missing"):
        capture.require_complete()
    with pytest.raises(ValueError, match="sampling rows"):
        capture.write(identities, torch.ones(3, 1))
    capture.write((identities[2], identities[0]), torch.tensor([[2., 2.], [0., 0.]]))
    capture.write((identities[1], identities[0]), torch.tensor([[1., 1.], [3., 3.]]))
    batch = CompletionBatch(tuple(Completion(identity, 0, "feedback", (1,), "1", FinishReason.LENGTH)
        for identity in identities))
    assert decode_boundary(encode_boundary(batch), CompletionBatch) == batch
    for field, malformed in (("output_index", True), ("token_ids", [False]),
                              ("identity", "not-an-identity"), ("finish_reason", "unknown")):
        wire = json.loads(encode_boundary(batch))
        wire["entries"][0][field] = malformed
        with pytest.raises(ValueError, match="boundary"):
            decode_boundary(json.dumps(wire), CompletionBatch)
    completed = capture.finish(batch)
    assert completed.hidden is mapped and completed.completions is batch
    torch.testing.assert_close(mapped, torch.tensor([[3., 3.], [1., 1.], [2., 2.]]))
