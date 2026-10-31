"""The packed rows, hidden captures and operands one engine step hands to a compiled workload."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pllm.contracts.auxiliary import AuxiliaryInput, map_auxiliary
from pllm.contracts.packed_hidden import HiddenRowBatch
from pllm.contracts.packed_rows import PackedPhaseRowBatch, PackedSamplerRowBatch
from pllm.contracts.parameter_overlay import CapturedParameterOperand
from pllm.execution.cuda_stream import CudaStreamReadiness


@dataclass(frozen=True, slots=True)
class TensorStepInput:
    rows: PackedPhaseRowBatch
    sampler_rows: PackedSamplerRowBatch
    hidden: HiddenRowBatch | None = None
    operands: tuple[CapturedParameterOperand, ...] = ()
    auxiliary: AuxiliaryInput[torch.Tensor] | None = None
    hidden_readiness: CudaStreamReadiness | None = None
    auxiliary_readiness: CudaStreamReadiness | None = None

    def __post_init__(self) -> None:
        assert self.sampler_rows.row_batch is self.rows
        if self.hidden is None:
            assert self.hidden_readiness is None
        else:
            assert self.hidden.row_batch is self.rows
            device = self.hidden.captures[0].storage.device
            readiness = None if self.hidden_readiness is None else self.hidden_readiness.device
            assert readiness == (device if device.type == "cuda" else None)

    @classmethod
    def of_batch(
        cls, auxiliary: AuxiliaryInput[torch.Tensor], readiness: CudaStreamReadiness | None = None
    ) -> TensorStepInput:
        """A host step without request rows, carrying one auxiliary batch its CUDA producer readies at `readiness`."""
        rows = PackedPhaseRowBatch(0, 0, ())
        return cls(rows, PackedSamplerRowBatch(rows, ()), auxiliary=auxiliary, auxiliary_readiness=readiness)

    def order_auxiliary(self) -> None:
        """Order the current stream after the auxiliary producer and keep its tensors alive for that stream."""
        if self.auxiliary is None or self.auxiliary_readiness is None:
            return
        stream = torch.cuda.current_stream(self.auxiliary_readiness.device)
        self.auxiliary_readiness.order_on(stream)
        map_auxiliary(self.auxiliary, lambda tensor: tensor.record_stream(stream))

    def hidden_batch(self, stream: torch.cuda.Stream | None) -> HiddenRowBatch:
        if self.hidden is None or (self.hidden_readiness is None) != (stream is None):
            raise ValueError("hidden captures differ from the input device")
        if self.hidden_readiness is not None and stream is not None:
            self.hidden_readiness.order_on(stream)
            for capture in self.hidden.captures:
                capture.storage.record_stream(stream)
        return self.hidden
