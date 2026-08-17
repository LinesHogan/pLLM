#!/usr/bin/env python3
"""BaseConsumer wrapper around the TLM training engine."""

from __future__ import annotations

from typing import Callable, Sequence, TypeVar

import torch

from tllm.contracts.test_time_lora import LoRAScope, TestTimeLoRAUpdate
from tllm.consumers.base import BaseConsumer
from tllm.consumers.tlm.config import TLMConsumerConfig
from tllm.consumers.tlm.engine import (
    TLMEngine,
    TLMStats,
    _validate_lora_update_against_known_target,
)
from tllm.ports.base import ConsumerFlow


_T = TypeVar("_T")


class TLMConsumer(BaseConsumer):
    """Local online/offline test-time learning consumer."""

    def __init__(self, config: TLMConsumerConfig, *, engine: TLMEngine | None = None) -> None:
        self.config = config
        self.engine = engine or TLMEngine(config)

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    def flows(self) -> Sequence[ConsumerFlow]:
        return ()

    def update_online(
        self,
        hidden: torch.Tensor,
        scores: torch.Tensor | None = None,
        *,
        logits: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        base_logits: torch.Tensor | None = None,
        allow_scalar_fallback: bool = False,
    ) -> None:
        self.engine.update_online(
            hidden,
            scores,
            logits=logits,
            labels=labels,
            base_logits=base_logits,
            allow_scalar_fallback=allow_scalar_fallback,
        )

    def update_offline(
        self,
        hidden: torch.Tensor,
        scores: torch.Tensor | None = None,
        *,
        logits: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        base_logits: torch.Tensor | None = None,
        allow_scalar_fallback: bool = False,
    ) -> None:
        self.engine.update_offline(
            hidden,
            scores,
            logits=logits,
            labels=labels,
            base_logits=base_logits,
            allow_scalar_fallback=allow_scalar_fallback,
        )

    def flush_offline(self) -> None:
        self.engine.flush_offline()

    def read_stats(self) -> TLMStats:
        return self.engine.read_stats()

    def export_test_time_lora_update(
        self,
        *,
        adapter_key: str,
        lora_int_id: int,
        target_module_name: str,
        rank: int,
        scope: LoRAScope = "global",
        request_ids: tuple[str, ...] = (),
        step_id: int = -1,
        adapter_slot: int = 0,
        alpha: float = 1.0,
    ) -> TestTimeLoRAUpdate:
        return self.engine.export_test_time_lora_update(
            adapter_key=adapter_key,
            lora_int_id=lora_int_id,
            target_module_name=target_module_name,
            rank=rank,
            scope=scope,
            request_ids=request_ids,
            step_id=step_id,
            adapter_slot=adapter_slot,
            alpha=alpha,
        )

    def run_lora_learning_step(
        self,
        *,
        predict: Callable[[], _T],
        hidden: torch.Tensor,
        lora_service: object,
        adapter_key: str,
        lora_int_id: int,
        target_module_name: str,
        rank: int | None = None,
        request_ids: tuple[str, ...] = (),
        step_id: int = -1,
        adapter_slot: int = 0,
        alpha: float = 1.0,
        apply_lora_update: Callable[[], object] | None = None,
        scores: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        base_logits: torch.Tensor | None = None,
        allow_scalar_fallback: bool = False,
    ) -> _T:
        """Run one upstream-ordered predict/train step and stage the LoRA update."""

        stage_update = getattr(lora_service, "stage_update", None)
        if not callable(stage_update):
            raise TypeError("lora_service must provide stage_update")
        update_rank = int(self.config.rank) if rank is None else rank

        def apply_staged_update() -> None:
            if apply_lora_update is not None:
                apply_lora_update()
                return
            commit_staged_updates = getattr(lora_service, "commit_staged_updates", None)
            if callable(commit_staged_updates):
                commit_staged_updates()

        def train_and_stage() -> None:
            if self.config.mode == "online":
                self.update_online(
                    hidden,
                    scores,
                    logits=logits,
                    labels=labels,
                    base_logits=base_logits,
                    allow_scalar_fallback=allow_scalar_fallback,
                )
            elif self.config.mode == "offline":
                self.update_offline(
                    hidden,
                    scores,
                    logits=logits,
                    labels=labels,
                    base_logits=base_logits,
                    allow_scalar_fallback=allow_scalar_fallback,
                )
                self.flush_offline()
            else:
                raise ValueError("TLM ordered step mode must be 'online' or 'offline'")

            update = self.export_test_time_lora_update(
                adapter_key=adapter_key,
                lora_int_id=lora_int_id,
                target_module_name=target_module_name,
                rank=update_rank,
                scope="request" if request_ids else "global",
                request_ids=request_ids,
                step_id=step_id,
                adapter_slot=adapter_slot,
                alpha=alpha,
            )
            _validate_lora_update_against_known_target(lora_service=lora_service, update=update)
            stage_update(update)
            apply_staged_update()

        if self.config.mode == "online":
            result = predict()
            train_and_stage()
            return result
        if self.config.mode == "offline":
            train_and_stage()
            return predict()
        raise ValueError("TLM ordered step mode must be 'online' or 'offline'")
