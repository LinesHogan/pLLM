#!/usr/bin/env python3
"""In-Place-TTT-inspired side-adapter consumer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from tllm.consumers.base import BaseConsumer
from tllm.consumers.devlist.common import LowRankAdapter
from tllm.contracts.gpu_stage import DeviceTensorLease
from tllm.contracts.port_bundle import PortBundle
from tllm.contracts.runtime_context import RuntimeContext
from tllm.ports.base import ConsumerFlow
from tllm.ports.request_meta import RequestMeta
from tllm.ports.residual_stream import ResidualStream


@dataclass
class InPlaceTTTConsumerConfig:
    consumer_id: str = "inplace_ttt"
    enabled: bool = True
    hidden_size: int = 0
    rank: int = 16
    learning_rate: float = 1e-3
    chunk_size: int = 4096
    source_layer: int = 0
    target_layer: int = -1
    phases: tuple[Literal["prefill", "decode"], ...] = ("prefill",)

    def __post_init__(self) -> None:
        normalized: list[Literal["prefill", "decode"]] = []
        for phase in self.phases:
            if phase not in ("prefill", "decode"):
                raise ValueError("InPlaceTTTConsumerConfig phases must contain only 'prefill' or 'decode'")
            if phase not in normalized:
                normalized.append(phase)
        if not normalized:
            raise ValueError("InPlaceTTTConsumerConfig requires at least one phase")
        self.phases = tuple(normalized)


@dataclass(frozen=True)
class InPlaceTTTStats:
    update_count: int = 0
    loss_count: int = 0
    trained_rows: int = 0
    last_loss: float = 0.0


def pad_rows_to_chunk(rows: torch.Tensor, *, chunk_size: int) -> tuple[torch.Tensor, int]:
    valid_rows = int(rows.shape[0])
    chunk = max(1, int(chunk_size))
    remainder = valid_rows % chunk
    if remainder == 0:
        return rows, valid_rows
    pad_rows = chunk - remainder
    padding = torch.zeros((pad_rows, *rows.shape[1:]), dtype=rows.dtype, device=rows.device)
    return torch.cat([rows, padding], dim=0), valid_rows


class InPlaceTTTEngine(nn.Module):
    """Small fast adapter that learns target-source hidden deltas."""

    def __init__(self, config: InPlaceTTTConsumerConfig) -> None:
        super().__init__()
        self.config = config
        self.adapter: LowRankAdapter | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self._update_count = 0
        self._loss_count = 0
        self._trained_rows = 0
        self._last_loss = 0.0
        if int(config.hidden_size) > 0:
            self._build_adapter(int(config.hidden_size))

    @staticmethod
    def _detached_train_input(tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.detach()
        if torch.is_inference(out):
            out = out.clone()
        return out

    def _build_adapter(self, hidden_size: int) -> None:
        self.config.hidden_size = int(hidden_size)
        self.adapter = LowRankAdapter(
            in_dim=int(hidden_size),
            out_dim=int(hidden_size),
            rank=int(self.config.rank),
        )
        self.optimizer = torch.optim.Adam(self.adapter.parameters(), lr=float(self.config.learning_rate))

    def ensure_shape(self, rows: torch.Tensor) -> None:
        if rows.ndim != 2:
            raise RuntimeError("In-Place TTT expects 2D hidden rows")
        hidden_size = int(rows.shape[1])
        if self.adapter is None:
            self._build_adapter(hidden_size)
        elif int(self.config.hidden_size) != hidden_size:
            raise RuntimeError("In-Place TTT hidden size changed after adapter initialization")
        assert self.adapter is not None
        first_param = next(self.adapter.parameters(), None)
        if first_param is not None and (first_param.device != rows.device or first_param.dtype != rows.dtype):
            self.to(device=rows.device, dtype=rows.dtype)

    def update(self, source: torch.Tensor, target: torch.Tensor) -> None:
        if not bool(self.config.enabled):
            return
        if source.shape != target.shape:
            raise RuntimeError("In-Place TTT source and target rows must have matching shapes")
        if int(source.shape[0]) <= 0:
            return
        self.ensure_shape(source)
        assert self.adapter is not None
        assert self.optimizer is not None
        source_train = self._detached_train_input(source)
        target_train = self._detached_train_input(target)
        padded_source, valid_rows = pad_rows_to_chunk(source_train, chunk_size=int(self.config.chunk_size))
        padded_target, _ = pad_rows_to_chunk(target_train, chunk_size=int(self.config.chunk_size))
        desired_delta = padded_target - padded_source
        pred_delta = self.adapter(padded_source)
        loss = F.mse_loss(pred_delta[:valid_rows], desired_delta[:valid_rows])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._update_count += 1
        self._loss_count += 1
        self._trained_rows += int(valid_rows)
        self._last_loss = float(loss.detach().cpu())

    def read_stats(self) -> InPlaceTTTStats:
        return InPlaceTTTStats(
            update_count=int(self._update_count),
            loss_count=int(self._loss_count),
            trained_rows=int(self._trained_rows),
            last_loss=float(self._last_loss),
        )


class InPlaceTTTConsumer(BaseConsumer):
    """tLLM-native In-Place-TTT-inspired hidden-pair adaptation consumer."""

    def __init__(self, config: InPlaceTTTConsumerConfig, *, engine: InPlaceTTTEngine | None = None) -> None:
        self.config = config
        self.engine = engine or InPlaceTTTEngine(config)

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    def flows(self) -> Sequence[ConsumerFlow]:
        if not bool(self.config.enabled):
            return ()
        flows: list[ConsumerFlow] = []
        for phase in self.config.phases:
            delivery = "device_lease" if phase == "decode" else "bundle"
            ownership = "runtime_lease" if phase == "decode" else "borrowed"
            window = "out_of_band" if phase == "decode" else "background"
            flows.append(
                ConsumerFlow(
                    reads=(
                        ResidualStream.read(
                            layer=int(self.config.source_layer),
                            site="block_output",
                            phase=phase,
                            role="source",
                        ),
                        ResidualStream.read(
                            layer=int(self.config.target_layer),
                            site="block_output",
                            phase=phase,
                            role="target",
                        ),
                        RequestMeta.read(),
                    ),
                    writes=(),
                    window=window,
                    delivery=delivery,
                    ownership=ownership,
                    bundle_key=("engine_step_id", "phase"),
                )
            )
        return flows

    def _extract_hidden_pair(self, bundle: PortBundle) -> tuple[torch.Tensor, torch.Tensor] | None:
        lease = bundle.entries.get("device_lease")
        if isinstance(lease, DeviceTensorLease):
            source = lease.entries.get("source")
            target = lease.entries.get("target")
            active_rows = max(0, int(lease.active_rows))
            if isinstance(source, torch.Tensor):
                source = source[:active_rows]
            if isinstance(target, torch.Tensor):
                target = target[:active_rows]
        else:
            source = bundle.entries.get("source")
            target = bundle.entries.get("target")
        if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
            return None
        return source, target

    def consume_bundle(self, bundle: PortBundle, ctx: RuntimeContext) -> None:
        _ = ctx
        pair = self._extract_hidden_pair(bundle)
        if pair is None:
            return
        self.engine.update(pair[0], pair[1])

    def read_stats(self) -> InPlaceTTTStats:
        return self.engine.read_stats()
