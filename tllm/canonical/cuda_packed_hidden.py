"""Opaque CUDA readiness capability for one packed hidden snapshot batch."""

from __future__ import annotations

import torch
from dataclasses import dataclass

from tllm.canonical.cuda_stream import CudaStreamReadiness
from tllm.contracts.packed_hidden import PackedHiddenSnapshotBatch


@dataclass(frozen=True, slots=True)
class _CudaPackedHiddenProducerAssertion:
    stream: torch.cuda.Stream
    readiness: CudaStreamReadiness


class CudaPackedHiddenSnapshotBatch:
    __slots__ = ("_batch", "_readiness")

    def __init__(
        self,
        batch: PackedHiddenSnapshotBatch,
        assertion: _CudaPackedHiddenProducerAssertion,
    ) -> None:
        if (
            type(batch) is not PackedHiddenSnapshotBatch
            or type(assertion) is not _CudaPackedHiddenProducerAssertion
            or batch.snapshots[0].hidden.device != assertion.stream.device
            or assertion.stream.device.type != "cuda"
        ):
            raise ValueError(
                "CUDA packed hidden snapshots require one owning producer stream"
            )
        self._batch = batch
        self._readiness = assertion.readiness

    @classmethod
    def assert_produced_on(
        cls,
        batch: PackedHiddenSnapshotBatch,
        producer_stream: torch.cuda.Stream,
    ) -> "CudaPackedHiddenSnapshotBatch":
        """Trusted capture-producer assertion; this does not infer provenance."""

        if type(producer_stream) is not torch.cuda.Stream:
            raise TypeError("CUDA packed hidden producer must own its stream")
        return cls(
            batch,
            _CudaPackedHiddenProducerAssertion(
                producer_stream,
                CudaStreamReadiness(producer_stream),
            ),
        )

    @property
    def batch(self) -> PackedHiddenSnapshotBatch:
        return self._batch

    def order_on(self, consumer_stream: torch.cuda.Stream) -> None:
        self._readiness.order_on(consumer_stream)


__all__ = ["CudaPackedHiddenSnapshotBatch"]
