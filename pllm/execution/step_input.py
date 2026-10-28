"""The packed rows, hidden captures and operands one engine step hands to a compiled workload."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pllm.contracts.auxiliary import AuxiliaryInput
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

    def __post_init__(self) -> None:
        if self.sampler_rows.row_batch is not self.rows:
            raise ValueError("tensor input requires the original packed row provenance")
        if self.hidden is None:
            if self.hidden_readiness is not None:
                raise ValueError("hidden readiness requires its original captures")
        else:
            if self.hidden.row_batch is not self.rows:
                raise ValueError("hidden input belongs to another packed step")
            device = self.hidden.captures[0].storage.device
            if (
                (device.type == "cuda") != (self.hidden_readiness is not None)
                or self.hidden_readiness is not None
                and self.hidden_readiness.device != device
            ):
                raise ValueError("hidden captures require readiness from their original device")

    def hidden_batch(self, stream: torch.cuda.Stream | None) -> HiddenRowBatch:
        if self.hidden is None or (self.hidden_readiness is None) != (stream is None):
            raise ValueError("hidden captures differ from the input device")
        if self.hidden_readiness is not None and stream is not None:
            self.hidden_readiness.order_on(stream)
            for capture in self.hidden.captures:
                capture.storage.record_stream(stream)
        return self.hidden
