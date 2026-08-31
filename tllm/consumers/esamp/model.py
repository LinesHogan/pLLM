#!/usr/bin/env python3
"""Low-rank model used by ESamp train engine."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tllm.consumers.esamp.model_bank_backend import (
    TorchModelBankForwardBackend,
)


class NoOutputLayerNorm(torch.nn.Module):
    """Closed parameter-free output-normalization variant."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class LowRankGatedResidualModel(torch.nn.Module):
    """Low-rank gated residual projector with optional output layernorm."""

    def __init__(
        self,
        *,
        hidden_size: int,
        rank: int,
        use_output_layernorm: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank = max(1, min(int(rank), self.hidden_size))
        self.use_output_layernorm = bool(use_output_layernorm)

        self.a = torch.nn.Parameter(torch.empty((self.hidden_size, self.rank), device=device, dtype=dtype))
        self.g = torch.nn.Parameter(torch.empty((self.hidden_size, self.rank), device=device, dtype=dtype))
        self.b = torch.nn.Parameter(torch.empty((self.rank, self.hidden_size), device=device, dtype=dtype))
        self.gate_bias = torch.nn.Parameter(torch.ones((self.rank,), device=device, dtype=dtype))
        self.out_ln: torch.nn.LayerNorm | NoOutputLayerNorm = (
            torch.nn.LayerNorm(
                self.hidden_size,
                elementwise_affine=True,
                device=device,
                dtype=dtype,
            )
            if self.use_output_layernorm
            else NoOutputLayerNorm()
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.a.normal_(mean=0.0, std=0.02)
            self.g.normal_(mean=0.0, std=0.02)
            self.b.normal_(mean=0.0, std=0.02)
            self.gate_bias.fill_(1.0)
            match self.out_ln:
                case torch.nn.LayerNorm() as out_ln:
                    out_ln.reset_parameters()
                case NoOutputLayerNorm():
                    pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = torch.matmul(x, self.a)
        gate = F.silu(torch.matmul(x, self.g) + self.gate_bias)
        delta = torch.matmul(up * gate, self.b)
        out = delta + x
        return self.out_ln(out)


class BatchedLowRankGatedResidualModel(torch.nn.Module):
    """One fixed-layout low-rank bank indexed by typed request slots."""

    def __init__(
        self,
        *,
        capacity: int,
        hidden_size: int,
        rank: int,
        use_output_layernorm: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if capacity < 1 or hidden_size < 1 or rank < 1:
            raise ValueError(
                "batched low-rank state requires positive layout dimensions"
            )
        self.capacity = capacity
        self.hidden_size = hidden_size
        self.rank = min(rank, hidden_size)
        self.use_output_layernorm = use_output_layernorm
        self.a = torch.nn.Parameter(
            torch.randn(
                (capacity, hidden_size, self.rank),
                device=device,
                dtype=dtype,
            )
            * 0.02
        )
        self.g = torch.nn.Parameter(
            torch.randn(
                (capacity, hidden_size, self.rank),
                device=device,
                dtype=dtype,
            )
            * 0.02
        )
        self.b = torch.nn.Parameter(
            torch.randn(
                (capacity, self.rank, hidden_size),
                device=device,
                dtype=dtype,
            )
            * 0.02
        )
        self.gate_bias = torch.nn.Parameter(
            torch.ones(
                (capacity, self.rank), device=device, dtype=dtype
            )
        )
        output_weight = torch.ones(
            (capacity, hidden_size), device=device, dtype=dtype
        )
        output_bias = torch.zeros(
            (capacity, hidden_size), device=device, dtype=dtype
        )
        self.out_ln_weight: torch.Tensor
        self.out_ln_bias: torch.Tensor
        if use_output_layernorm:
            self.out_ln_weight = torch.nn.Parameter(output_weight)
            self.out_ln_bias = torch.nn.Parameter(output_bias)
        else:
            self.register_buffer("out_ln_weight", output_weight)
            self.register_buffer("out_ln_bias", output_bias)

    def forward(
        self, slot_ids: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        if (
            slot_ids.ndim != 1
            or source.ndim != 2
            or int(slot_ids.shape[0]) != int(source.shape[0])
            or int(source.shape[1]) != self.hidden_size
        ):
            raise ValueError(
                "batched low-rank forward requires aligned slot and hidden rows"
            )
        return TorchModelBankForwardBackend().forward(
            slot_ids=slot_ids,
            src=source,
            params=self,
            use_output_layernorm=self.use_output_layernorm,
        )
