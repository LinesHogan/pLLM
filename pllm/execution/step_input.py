"""The packed rows, token history, hidden taps and projection operands one engine step hands to a compiled workload."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch
import triton

from pllm.contracts.host_batch import HostBatch
from pllm.contracts.packed_hidden import HiddenRowBatch
from pllm.contracts.packed_rows import PackedPhaseRowBatch, PackedSamplerRowBatch, label_rows
from pllm.contracts.parameter_overlay import Projection, ProjectionOperands
from pllm.contracts.request_identity import RequestIdentity
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.kernels.input_tokens import input_token_rows

if TYPE_CHECKING:
    from pllm.execution.tensor_frame import UpdateOutput


_NO_UPDATES: Mapping[tuple[RequestIdentity, Projection], UpdateOutput] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class StepInput:
    rows: PackedPhaseRowBatch
    sampler_rows: PackedSamplerRowBatch
    hidden: HiddenRowBatch | None = None
    operands: tuple[ProjectionOperands, ...] = ()
    batch: HostBatch[torch.Tensor] | None = None
    input_tokens: torch.Tensor | None = None
    hidden_readiness: CudaStreamReadiness | None = None
    import_rows: Mapping[str, int] = field(default_factory=lambda: dict[str, int]())
    overlay_updates: Mapping[tuple[RequestIdentity, Projection], UpdateOutput] = field(
        default_factory=lambda: _NO_UPDATES
    )

    force_import: bool = False

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
    def of_batch(
        cls,
        batch: HostBatch[torch.Tensor],
        *,
        identities: tuple[RequestIdentity, ...] = (),
        import_rows: Mapping[str, int] | None = None,
        force_import: bool = False,
    ) -> StepInput:
        """Build one host invocation, with a selectable prompt row for each request import."""
        n = len(identities)
        sampler = label_rows(identities, [1] * n, [1] * n, [0] * n, 0)
        return cls(
            sampler.row_batch, sampler, batch=batch, import_rows=dict(import_rows or {}), force_import=force_import
        )

    def hidden_batch(self, stream: torch.cuda.Stream | None) -> HiddenRowBatch:
        if self.hidden is None or (self.hidden_readiness is None) != (stream is None):
            raise ValueError("hidden taps differ from the input device")
        if self.hidden_readiness is not None and stream is not None:
            self.hidden_readiness.order_on(stream)
            for tap in self.hidden.taps:
                tap.storage.record_stream(stream)
        return self.hidden


def gather_input_tokens(
    history: torch.Tensor,
    requests: torch.Tensor,
    starts: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Write ids without host reads; requests holds [request count, row count, native slots...]."""
    input_token_rows[(len(output),)](
        history,
        requests,
        starts,
        positions,
        output,
        history.stride(0),
        output.shape[1],
        triton.next_power_of_2(len(requests) - 2),
        triton.next_power_of_2(output.shape[1]),
    )
