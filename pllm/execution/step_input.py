"""The packed rows, hidden taps and projection operands one engine step hands to a compiled workload."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from pllm.contracts.host_batch import HostBatch, map_host_batch
from pllm.contracts.packed_hidden import HiddenRowBatch
from pllm.contracts.packed_rows import PackedPhaseRowBatch, PackedSamplerRowBatch
from pllm.contracts.parameter_overlay import ProjectionOperands
from pllm.execution.cuda_stream import CudaStreamReadiness


@dataclass(frozen=True, slots=True)
class StepInput:
    rows: PackedPhaseRowBatch
    sampler_rows: PackedSamplerRowBatch
    hidden: HiddenRowBatch | None = None
    operands: tuple[ProjectionOperands, ...] = ()
    batch: HostBatch[torch.Tensor] | None = None
    hidden_readiness: CudaStreamReadiness | None = None
    batch_readiness: CudaStreamReadiness | None = None
    import_rows: Mapping[str, int] = field(default_factory=lambda: dict[str, int]())

    def __post_init__(self) -> None:
        assert self.sampler_rows.row_batch is self.rows
        if self.hidden is None:
            assert self.hidden_readiness is None
        else:
            assert self.hidden.row_batch is self.rows
            device = self.hidden.taps[0].storage.device
            readiness = None if self.hidden_readiness is None else self.hidden_readiness.device
            assert readiness == (device if device.type == "cuda" else None)

    @classmethod
    def of_batch(cls, batch: HostBatch[torch.Tensor], readiness: CudaStreamReadiness | None = None) -> StepInput:
        """A host step without request rows, carrying one host batch its CUDA producer readies at `readiness`."""
        rows = PackedPhaseRowBatch(0, 0, ())
        return cls(rows, PackedSamplerRowBatch(rows, ()), batch=batch, batch_readiness=readiness)

    def order_host_batch(self) -> None:
        """Order the current stream after the host batch producer and keep its tensors alive for that stream."""
        if self.batch is None or self.batch_readiness is None:
            return
        stream = torch.cuda.current_stream(self.batch_readiness.device)
        self.batch_readiness.order_on(stream)
        map_host_batch(self.batch, lambda tensor: tensor.record_stream(stream))

    def hidden_batch(self, stream: torch.cuda.Stream | None) -> HiddenRowBatch:
        if self.hidden is None or (self.hidden_readiness is None) != (stream is None):
            raise ValueError("hidden taps differ from the input device")
        if self.hidden_readiness is not None and stream is not None:
            self.hidden_readiness.order_on(stream)
            for tap in self.hidden.taps:
                tap.storage.record_stream(stream)
        return self.hidden
