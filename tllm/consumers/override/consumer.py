#!/usr/bin/env python3
"""OverRIDE-inspired iterative reweighting consumer.

The tensor math is a compiled ``OverRIDEProgram``; this wrapper is plain
Python: it parses the iteration from request ids, groups rows per
iteration, and stashes detached (hidden, logits) rows across the sampler
boundary. The training flow recomputes ``logits + head(hidden)`` at
``after_sample`` time, which is exactly equivalent to the old stashed
q-logits: no parameter update can happen between prepare and after_sample
(the update *is* the after_sample step).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from tllm.consumers.base import BaseConsumer
from tllm.consumers._common.bundle_inputs import provided_bundle
from tllm.consumers.override.config import OverRIDEConsumerConfig
from tllm.consumers.override.program import OverRIDEProgram
from tllm.ports.base import ConsumerFlow
from tllm.program import ProgramConsumer, StateSpec, compile_program
from tllm.workgraph.frontend import DynamicDim, StateScope

__all__ = [
    "OverRIDEConsumer",
    "OverRIDEConsumerConfig",
    "OverRIDEProgram",
    "OverRIDEStats",
]


@dataclass(frozen=True)
class OverRIDEStats:
    reweight_count: int = 0
    modified_row_count: int = 0
    loss_count: int = 0
    update_count: int = 0
    last_loss: float = 0.0


@dataclass(frozen=True)
class _StashedRow:
    head_idx: int
    hidden: torch.Tensor
    logits: torch.Tensor


class OverRIDEConsumer(BaseConsumer[OverRIDEConsumerConfig]):
    """tLLM-native consumer for OverRIDE-style iterative reweighting."""

    def __init__(self, config: OverRIDEConsumerConfig) -> None:
        self.config = config
        self._compiled: ProgramConsumer | None = None
        self._step_id = 0
        self._reweight_count = 0
        self._modified_row_count = 0
        self._loss_sum_seen = 0.0
        self._last_loss = 0.0
        self._pending: dict[int, _StashedRow] = {}
        from tllm.consumers.override.sampler_provider import OverRIDESamplerModifierProvider

        self._sampler_provider = OverRIDESamplerModifierProvider(consumer=self)

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    @property
    def compiled(self) -> ProgramConsumer | None:
        return self._compiled

    def flows(self) -> Sequence[ConsumerFlow]:
        return ()

    def sampler_modifier_provider(self):
        return self._sampler_provider

    # -- iteration bookkeeping (plain Python) ---------------------------------

    @staticmethod
    def iteration_from_request_id(request_id: str) -> int:
        if "_" not in str(request_id):
            return 1
        idx_str = str(request_id).split("_", 1)[0]
        try:
            return int(idx_str) + 1
        except ValueError:
            return 1

    @classmethod
    def iterations_from_request_ids(cls, request_ids: Sequence[str]) -> tuple[int, ...]:
        return tuple(cls.iteration_from_request_id(request_id) for request_id in request_ids)

    @property
    def _num_heads(self) -> int:
        return max(0, int(self.config.num_iterations) - 1)

    def _prev_head_idx(self, iteration: int) -> int | None:
        idx = int(iteration) - 2
        return idx if 0 <= idx < self._num_heads else None

    def _cur_head_idx(self, iteration: int) -> int | None:
        idx = int(iteration) - 1
        return idx if 0 <= idx < self._num_heads else None

    # -- compiled program ------------------------------------------------------

    def _ensure_compiled(self, hidden: torch.Tensor, logits: torch.Tensor) -> None:
        if self._compiled is not None:
            return
        if hidden.ndim != 2 or logits.ndim != 2:
            raise RuntimeError("OverRIDE expects 2D hidden and logits tensors")
        if int(hidden.shape[0]) != int(logits.shape[0]):
            raise RuntimeError("OverRIDE requires hidden/logits row counts to match")
        hidden_size = int(hidden.shape[1])
        vocab_size = int(logits.shape[1])
        if int(self.config.hidden_size) > 0 and int(self.config.hidden_size) != hidden_size:
            raise RuntimeError("OverRIDE hidden size does not match configured hidden_size")
        if int(self.config.vocab_size) > 0 and int(self.config.vocab_size) != vocab_size:
            raise RuntimeError("OverRIDE vocab size does not match configured vocab_size")
        program = OverRIDEProgram(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            rank=int(self.config.rank),
            num_heads=self._num_heads,
            lr=float(self.config.learning_rate),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        rows = 2  # export requires example rows strictly above the lower bound
        hidden_ex = torch.zeros((rows, hidden_size), device=hidden.device, dtype=hidden.dtype)
        logits_ex = torch.zeros((rows, vocab_size), device=hidden.device, dtype=hidden.dtype)
        idx_ex = torch.zeros((rows,), device=hidden.device, dtype=torch.long)
        scale_ex = torch.zeros((rows, 1), device=hidden.device, dtype=hidden.dtype)
        tokens_ex = torch.zeros((rows,), device=hidden.device, dtype=torch.long)
        forward_dims = (
            DynamicDim(value_path=(0,), dimension=0, lower_bound=1, upper_bound=65536),
            DynamicDim(value_path=(1,), dimension=0, lower_bound=1, upper_bound=65536),
            DynamicDim(value_path=(2,), dimension=0, lower_bound=1, upper_bound=65536),
            DynamicDim(value_path=(3,), dimension=0, lower_bound=1, upper_bound=65536),
        )
        self._compiled = compile_program(
            program,
            states={"heads": StateSpec(scope=StateScope.REPLICA, optimizer=torch.optim.Adam)},
            examples={
                "reweight": (hidden_ex, logits_ex, idx_ex, scale_ex),
                "forward": (hidden_ex, logits_ex, idx_ex, tokens_ex),
            },
            dynamic_dims={"reweight": forward_dims[:4], "forward": forward_dims},
        )

    # -- reweighting (read-only flow) ------------------------------------------

    def reweight_logits(self, hidden: torch.Tensor, logits: torch.Tensor, *, iteration: int) -> torch.Tensor:
        """Return logits reweighted by the previous iteration's head, if any."""
        return self.reweight_logits_batch(hidden, logits, iterations=(int(iteration),) * int(logits.shape[0]))

    def reweight_logits_batch(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        *,
        request_ids: Sequence[str] | None = None,
        iterations: Sequence[int] | None = None,
        row_ids: tuple[int, ...] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reweight mixed-iteration rows; stash trainable rows for after_sample."""
        if hidden.ndim != 2 or logits.ndim != 2:
            raise RuntimeError("OverRIDE expects 2D hidden and logits tensors")
        row_count = int(logits.shape[0])
        if int(hidden.shape[0]) != row_count:
            raise RuntimeError("OverRIDE requires hidden/logits row counts to match")
        resolved = self._resolve_iterations(
            row_count=row_count, request_ids=request_ids, iterations=iterations
        )
        normalized_row_ids = self._normalize_row_ids(
            tuple(range(row_count)) if row_ids is None else row_ids, row_count
        )
        if not bool(self.config.enabled) or self._num_heads <= 0:
            return logits

        prev = [self._prev_head_idx(iteration) for iteration in resolved]
        head_idx = torch.tensor(
            [idx if idx is not None else 0 for idx in prev],
            device=hidden.device,
            dtype=torch.long,
        )
        scale = torch.tensor(
            [[float(self.config.lambd)] if idx is not None else [0.0] for idx in prev],
            device=hidden.device,
            dtype=hidden.dtype,
        )
        self._ensure_compiled(hidden, logits)
        assert self._compiled is not None
        reweighted = self._compiled.run_flow(
            "reweight", hidden.detach(), logits.detach(), head_idx, scale
        )["reweighted"]
        modified = sum(1 for idx in prev if idx is not None)
        if modified:
            self._reweight_count += 1
            self._modified_row_count += modified

        self._pending.clear()
        for row_pos, iteration in enumerate(resolved):
            cur_idx = self._cur_head_idx(iteration)
            if cur_idx is None:
                continue
            self._pending[normalized_row_ids[row_pos]] = _StashedRow(
                head_idx=cur_idx,
                hidden=hidden[row_pos : row_pos + 1].detach(),
                logits=logits[row_pos : row_pos + 1].detach(),
            )
        return reweighted

    # -- training (compiled program) --------------------------------------------

    def update_from_sampled_tokens(
        self,
        sampled_token_ids: torch.Tensor,
        *,
        row_ids: tuple[int, ...] | torch.Tensor | None = None,
    ) -> None:
        """Train the stashed current-iteration heads against sampled tokens."""
        if not bool(self.config.enabled) or not self._pending:
            return
        targets = sampled_token_ids.reshape(-1)
        normalized_row_ids = self._normalize_row_ids(
            tuple(range(int(targets.numel()))) if row_ids is None else row_ids,
            int(targets.numel()),
        )
        selected: list[tuple[_StashedRow, torch.Tensor]] = []
        for token_idx, row_id in enumerate(normalized_row_ids):
            pending = self._pending.pop(row_id, None)
            if pending is None:
                continue
            selected.append((pending, targets[token_idx : token_idx + 1]))
        if not selected:
            return

        hidden = torch.cat([row.hidden for row, _t in selected], dim=0)
        logits = torch.cat([row.logits for row, _t in selected], dim=0)
        head_idx = torch.tensor(
            [row.head_idx for row, _t in selected], device=hidden.device, dtype=torch.long
        )
        tokens = torch.cat([t for _row, t in selected], dim=0).to(device=hidden.device)
        self._ensure_compiled(hidden, logits)
        assert self._compiled is not None
        self._step_id += 1
        loss_sum_before = float(self._compiled.loss_sum)
        bundle = provided_bundle(
            consumer_id=self.consumer_id,
            step_id=self._step_id,
            entries={
                "hidden": hidden,
                "logits": logits,
                "head_idx": head_idx,
                "tokens": tokens,
            },
        )
        self._compiled.consume_flow(self._compiled.flow_ordinal("forward"), bundle, None)
        self._last_loss = float(self._compiled.loss_sum) - loss_sum_before

    # -- row-id / iteration validation (plain Python) -----------------------------

    @staticmethod
    def _normalize_row_ids(row_ids: tuple[int, ...] | torch.Tensor, row_count: int) -> tuple[int, ...]:
        if isinstance(row_ids, torch.Tensor):
            if row_ids.ndim != 1:
                raise RuntimeError("OverRIDE row id tensors must be 1D")
            integer_dtypes = {
                torch.int8,
                torch.uint8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
            if row_ids.dtype not in integer_dtypes:
                raise RuntimeError("OverRIDE row id tensors must use an integer dtype")
            normalized = tuple(int(row_id) for row_id in row_ids.detach().cpu().tolist())
        else:
            if not isinstance(row_ids, tuple):
                raise RuntimeError("OverRIDE row ids must be a tuple of integers")
            normalized = row_ids
            if any(type(row_id) is not int for row_id in row_ids):
                raise RuntimeError("OverRIDE row ids must be integers")
        if len(normalized) != int(row_count):
            raise RuntimeError("OverRIDE requires one row id per logits row")
        if len(set(normalized)) != len(normalized):
            raise RuntimeError("OverRIDE row ids must be unique")
        return normalized

    def _resolve_iterations(
        self,
        *,
        row_count: int,
        request_ids: Sequence[str] | None,
        iterations: Sequence[int] | None,
    ) -> tuple[int, ...]:
        if request_ids is not None and iterations is not None:
            raise RuntimeError("OverRIDE accepts request_ids or iterations, not both")
        if iterations is None:
            if request_ids is None:
                raise RuntimeError("OverRIDE requires request_ids or explicit iterations")
            resolved = self.iterations_from_request_ids(request_ids)
        else:
            resolved = tuple(int(iteration) for iteration in iterations)
        if len(resolved) != int(row_count):
            raise RuntimeError("OverRIDE requires one iteration id per logits row")
        return resolved

    def read_stats(self) -> OverRIDEStats:
        compiled = self._compiled
        return OverRIDEStats(
            reweight_count=int(self._reweight_count),
            modified_row_count=int(self._modified_row_count),
            loss_count=0 if compiled is None else int(compiled.loss_count),
            update_count=0 if compiled is None else int(compiled.update_count),
            last_loss=float(self._last_loss),
        )
