#!/usr/bin/env python3
"""In-Place-TTT-inspired side-adapter consumer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from opt_einsum import contract
import torch
from torch import nn
from torch.nn import functional as F

from tllm.consumers.base import BaseConsumer
from tllm.consumers._common.alignment import LowRankAdapter
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
    tail_rows: int = 0
    last_loss: float = 0.0


@dataclass(frozen=True)
class InPlaceTTTExactToyState:
    """Toy exact cache matching upstream-shaped ``past_h/past_t/past_w``."""

    past_h: torch.Tensor | None = None
    past_t: torch.Tensor | None = None
    past_w: torch.Tensor | None = None


@dataclass(frozen=True)
class InPlaceTTTExactToyStep:
    output: torch.Tensor
    state: InPlaceTTTExactToyState
    update_count: int = 0
    trained_rows: int = 0
    tail_rows: int = 0


def pad_rows_to_chunk(rows: torch.Tensor, *, chunk_size: int) -> tuple[torch.Tensor, int]:
    valid_rows = int(rows.shape[0])
    chunk = max(1, int(chunk_size))
    remainder = valid_rows % chunk
    if remainder == 0:
        return rows, valid_rows
    pad_rows = chunk - remainder
    padding = torch.zeros((pad_rows, *rows.shape[1:]), dtype=rows.dtype, device=rows.device)
    return torch.cat([rows, padding], dim=0), valid_rows


def _empty_toy_output(h: torch.Tensor, down_proj_weight: torch.Tensor) -> torch.Tensor:
    return h.new_empty((0, int(down_proj_weight.shape[0])))


def _validate_exact_toy_tensors(h: torch.Tensor, t: torch.Tensor, down_proj_weight: torch.Tensor) -> None:
    if h.ndim != 2 or t.ndim != 2 or down_proj_weight.ndim != 2:
        raise RuntimeError("In-Place TTT exact toy math expects 2D h, t, and down_proj weight")
    if int(h.shape[0]) != int(t.shape[0]):
        raise RuntimeError("In-Place TTT exact toy h and t must have matching rows")
    if int(h.shape[1]) != int(down_proj_weight.shape[1]):
        raise RuntimeError("In-Place TTT exact toy h width must match down_proj input width")
    if int(t.shape[1]) != int(down_proj_weight.shape[0]):
        raise RuntimeError("In-Place TTT exact toy t width must match down_proj output width")


def _validate_ttt_proj_weight(ttt_proj_weight: torch.Tensor | None, down_proj_weight: torch.Tensor) -> None:
    if ttt_proj_weight is None:
        return
    if ttt_proj_weight.ndim != 2:
        raise RuntimeError("In-Place TTT exact toy ttt_proj weight must be 2D")
    hidden = int(down_proj_weight.shape[0])
    if tuple(ttt_proj_weight.shape) != (hidden, hidden):
        raise RuntimeError("In-Place TTT exact toy ttt_proj weight must be hidden_size x hidden_size")


def run_exact_toy_down_proj_step(
    h: torch.Tensor,
    t: torch.Tensor,
    down_proj_weight: torch.Tensor,
    *,
    state: InPlaceTTTExactToyState | None = None,
    chunk_size: int,
    learning_rate: float,
    ttt_proj_weight: torch.Tensor | None = None,
) -> InPlaceTTTExactToyStep:
    """Run a dependency-light exact toy step for MLP down-proj fast weights.

    The helper models the consumer-side math target for upstream-shaped
    In-Place TTT alignment. ``past_w`` is the effective ``down_proj.weight``.
    Full chunks update that effective weight with a local linear least-squares
    gradient step, while below-chunk rows are cached as ``past_h/past_t`` and do
    not update.
    """

    _validate_exact_toy_tensors(h, t, down_proj_weight)
    _validate_ttt_proj_weight(ttt_proj_weight, down_proj_weight)
    current_state = state or InPlaceTTTExactToyState()
    fast_w = current_state.past_w
    if fast_w is None:
        fast_w = down_proj_weight.detach().clone()
    else:
        fast_w = fast_w.to(device=down_proj_weight.device, dtype=down_proj_weight.dtype).clone()
    if tuple(fast_w.shape) != tuple(down_proj_weight.shape):
        raise RuntimeError("In-Place TTT exact toy past_w must match down_proj weight shape")

    h_work = h
    t_work = t
    if current_state.past_h is not None or current_state.past_t is not None:
        if current_state.past_h is None or current_state.past_t is None:
            raise RuntimeError("In-Place TTT exact toy past_h and past_t must be present together")
        past_h = current_state.past_h.to(device=h.device, dtype=h.dtype)
        past_t = current_state.past_t.to(device=t.device, dtype=t.dtype)
        _validate_exact_toy_tensors(past_h, past_t, down_proj_weight)
        h_work = torch.cat([past_h, h], dim=0)
        t_work = torch.cat([past_t, t], dim=0)

    chunk = max(1, int(chunk_size))
    total_rows = int(h_work.shape[0])
    full_rows = (total_rows // chunk) * chunk
    outputs: list[torch.Tensor] = []
    update_count = 0
    for start in range(0, full_rows, chunk):
        end = start + chunk
        h_chunk = h_work[start:end]
        t_chunk = t_work[start:end]
        outputs.append(h_chunk.matmul(fast_w.t()))
        if ttt_proj_weight is None:
            dw = contract("c h, c d -> d h", h_chunk, t_chunk)
        else:
            proj = ttt_proj_weight.to(device=t_chunk.device, dtype=t_chunk.dtype)
            dw = contract("c h, c d, d e -> e h", h_chunk, t_chunk, proj)
        fast_w = fast_w + (dw * float(learning_rate)).to(device=fast_w.device, dtype=fast_w.dtype)
        update_count += 1

    tail_h = h_work[full_rows:]
    tail_t = t_work[full_rows:]
    tail_rows = int(tail_h.shape[0])
    if tail_rows > 0:
        outputs.append(tail_h.matmul(fast_w.t()))
    output = torch.cat(outputs, dim=0) if outputs else _empty_toy_output(h_work, down_proj_weight)
    next_state = InPlaceTTTExactToyState(
        past_h=tail_h.detach().clone() if tail_rows > 0 else None,
        past_t=tail_t.detach().clone() if tail_rows > 0 else None,
        past_w=fast_w.detach().clone(),
    )
    return InPlaceTTTExactToyStep(
        output=output,
        state=next_state,
        update_count=update_count,
        trained_rows=full_rows,
        tail_rows=tail_rows,
    )


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
        self._tail_source: torch.Tensor | None = None
        self._tail_target: torch.Tensor | None = None
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

    def _store_tail(self, source: torch.Tensor, target: torch.Tensor) -> None:
        if int(source.shape[0]) <= 0:
            self._tail_source = None
            self._tail_target = None
            return
        self._tail_source = source.detach().clone()
        self._tail_target = target.detach().clone()

    def _append_cached_tail(self, source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._tail_source is None or self._tail_target is None:
            return source, target
        tail_source = self._tail_source.to(device=source.device, dtype=source.dtype)
        tail_target = self._tail_target.to(device=target.device, dtype=target.dtype)
        return torch.cat([tail_source, source], dim=0), torch.cat([tail_target, target], dim=0)

    def _update_full_chunk(self, source: torch.Tensor, target: torch.Tensor) -> None:
        assert self.adapter is not None
        assert self.optimizer is not None
        desired_delta = target - source
        pred_delta = self.adapter(source)
        loss = F.mse_loss(pred_delta, desired_delta)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._update_count += 1
        self._loss_count += 1
        self._trained_rows += int(source.shape[0])
        self._last_loss = float(loss.detach().cpu())

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
        source_train, target_train = self._append_cached_tail(source_train, target_train)
        chunk_size = max(1, int(self.config.chunk_size))
        total_rows = int(source_train.shape[0])
        full_rows = (total_rows // chunk_size) * chunk_size
        if full_rows <= 0:
            self._store_tail(source_train, target_train)
            return

        for start in range(0, full_rows, chunk_size):
            end = start + chunk_size
            self._update_full_chunk(source_train[start:end], target_train[start:end])
        self._store_tail(source_train[full_rows:], target_train[full_rows:])

    def read_stats(self) -> InPlaceTTTStats:
        tail_rows = 0 if self._tail_source is None else int(self._tail_source.shape[0])
        return InPlaceTTTStats(
            update_count=int(self._update_count),
            loss_count=int(self._loss_count),
            trained_rows=int(self._trained_rows),
            tail_rows=tail_rows,
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
