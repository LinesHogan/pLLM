#!/usr/bin/env python3
"""BaseConsumer wrapper driving the compiled TLM Program.

The wrapper owns only orchestration: online/offline batching, the upstream
TTL predict/train ordering, and stats. All training execution — lanes,
functional optimizer transitions, CUDA graph capture — is derived by
``tllm.program.compile_program`` from ``TLMProgram``'s declarations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import torch

from tllm.contracts.gpu_stage import DeviceTensorLease
from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.request_meta_view import RowBatchMeta
from tllm.consumers.base import BaseConsumer
from tllm.consumers.tlm.config import TLMConsumerConfig
from tllm.consumers.tlm.program import TLMProgram
from tllm.ports.base import ConsumerFlow
from tllm.program import ProgramConsumer, StateSpec, compile_program
from tllm.workgraph.frontend import DynamicDim, StateScope

_T = TypeVar("_T")


@dataclass(frozen=True)
class TLMStats:
    update_count: int = 0
    loss_count: int = 0
    buffered_count: int = 0
    last_loss: float = 0.0


class TLMConsumer(BaseConsumer[TLMConsumerConfig]):
    """Local online/offline test-time learning consumer (Program-backed)."""

    def __init__(self, config: TLMConsumerConfig) -> None:
        self.config = config
        self._compiled: ProgramConsumer | None = None
        self._step_id = 0
        self._offline: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._loss_sum_seen = 0.0
        self._last_loss = 0.0

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    @property
    def compiled(self) -> ProgramConsumer | None:
        return self._compiled

    def flows(self) -> Sequence[ConsumerFlow]:
        return ()

    # -- compilation -----------------------------------------------------

    def _ensure_compiled(self, hidden: torch.Tensor, logits: torch.Tensor) -> None:
        if self._compiled is not None:
            return
        if hidden.ndim != 2:
            raise RuntimeError("TLM expects 2D hidden rows")
        program = TLMProgram(
            hidden_size=int(hidden.shape[1]),
            vocab_size=int(logits.shape[-1]),
            rank=int(self.config.rank),
            lr=float(self.config.learning_rate),
            threshold=float(self.config.threshold),
            lamb=float(self.config.lamb),
            ignore_index=int(self.config.ignore_index),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        rows = max(2, int(hidden.shape[0]))  # export requires rows > lower bound
        labels_ex = torch.zeros((rows,), dtype=torch.long, device=hidden.device)
        forward_ex = (
            torch.zeros((rows, int(hidden.shape[1])), device=hidden.device, dtype=hidden.dtype),
            torch.zeros((rows, int(logits.shape[-1])), device=hidden.device, dtype=hidden.dtype),
            labels_ex,
            torch.zeros((rows, int(logits.shape[-1])), device=hidden.device, dtype=hidden.dtype),
        )
        predict_ex = (forward_ex[0], forward_ex[1])
        row_dims = tuple(
            DynamicDim(value_path=(i,), dimension=0, lower_bound=1, upper_bound=65536)
            for i in range(4)
        )
        self._compiled = compile_program(
            program,
            states={"adapters": StateSpec(scope=StateScope.REPLICA, optimizer=torch.optim.Adam)},
            examples={"forward": forward_ex, "predict": predict_ex},
            dynamic_dims={
                "forward": row_dims,
                "predict": row_dims[:2],
            },
        )

    # -- driving the compiled program -------------------------------------

    def _train_batch(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> None:
        if not bool(self.config.enabled):
            return
        self._ensure_compiled(hidden, logits)
        assert self._compiled is not None
        rows = int(hidden.shape[0])
        self._step_id += 1
        bundle = PortBundle(
            key=BundleKey(
                engine_step_id=self._step_id,
                phase="decode",
                request_id=self.consumer_id,
                sample_idx=0,
            ),
            entries={
                "device_lease": DeviceTensorLease(
                    entries={
                        "hidden": hidden.detach(),
                        "logits": logits.detach(),
                        "labels": labels.detach(),
                        "base_logits": base_logits.detach(),
                    },
                    active_rows=rows,
                ),
                "request_meta": RowBatchMeta(
                    request_ids=tuple(f"{self.consumer_id}-{i}" for i in range(rows)),
                    prompt_idxs=tuple(0 for _ in range(rows)),
                    sample_idxs=tuple(0 for _ in range(rows)),
                    phase="decode",
                    engine_step_id=self._step_id,
                ),
            },
        )
        self._compiled.consume_flow(0, bundle, None)
        loss_sum = float(self._compiled.loss_sum)
        self._last_loss = loss_sum - self._loss_sum_seen
        self._loss_sum_seen = loss_sum

    def predict(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Read-only adapted logits against the current bank state."""
        self._ensure_compiled(hidden, logits)
        assert self._compiled is not None
        out = self._compiled.run_flow("predict", hidden, logits)
        return out["adapted_logits"]

    # -- public API ---------------------------------------------------------

    def update_online(
        self,
        hidden: torch.Tensor,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
        base_logits: torch.Tensor | None = None,
    ) -> None:
        self._train_batch(hidden, logits, labels, logits if base_logits is None else base_logits)

    def update_offline(
        self,
        hidden: torch.Tensor,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
        base_logits: torch.Tensor | None = None,
    ) -> None:
        if not bool(self.config.enabled):
            return
        self._offline.append(
            (
                hidden.detach(),
                logits.detach(),
                labels.detach(),
                (logits if base_logits is None else base_logits).detach(),
            )
        )

    def flush_offline(self) -> None:
        if not self._offline:
            return
        hidden = torch.cat([item[0] for item in self._offline], dim=0)
        logits = torch.cat([item[1] for item in self._offline], dim=0)
        labels = torch.cat([item[2].reshape(-1) for item in self._offline], dim=0)
        base_logits = torch.cat([item[3] for item in self._offline], dim=0)
        self._offline.clear()
        self._train_batch(hidden, logits, labels, base_logits)

    def run_ordered_step(
        self,
        *,
        predict: Callable[[], _T],
        hidden: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        base_logits: torch.Tensor | None = None,
    ) -> _T:
        """Run the observable upstream TTL ordering for a single step."""
        if self.config.mode == "online":
            result = predict()
            self.update_online(hidden, logits=logits, labels=labels, base_logits=base_logits)
            return result
        self.update_offline(hidden, logits=logits, labels=labels, base_logits=base_logits)
        self.flush_offline()
        return predict()

    def read_stats(self) -> TLMStats:
        compiled = self._compiled
        return TLMStats(
            update_count=0 if compiled is None else int(compiled.update_count),
            loss_count=0 if compiled is None else int(compiled.loss_count),
            buffered_count=sum(int(item[0].shape[0]) for item in self._offline),
            last_loss=float(self._last_loss),
        )
