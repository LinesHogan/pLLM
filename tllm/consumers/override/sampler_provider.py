#!/usr/bin/env python3
"""OverRIDE-owned provider for same-step sampler intervention."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.consumers.override.consumer import OverRIDEReweightEngine
from tllm.runtime.sampler_bridge.types import CandidateModifierState, SamplerStepView


@dataclass
class _PreparedSamplerStep:
    engine_step_id: int
    row_count: int
    row_ids: tuple[int, ...]


@dataclass
class OverRIDESamplerModifierProvider:
    engine: OverRIDEReweightEngine
    _prepared_step: _PreparedSamplerStep | None = None

    def is_active(self) -> bool:
        return bool(self.engine.config.enabled)

    @staticmethod
    def _internal_decode_row_ids(row_count: int) -> tuple[int, ...]:
        return tuple(range(int(row_count)))

    def prepare_step(self, view: SamplerStepView) -> CandidateModifierState | None:
        if not self.is_active():
            return None
        row_count = int(view.logits.shape[0])
        if row_count <= 0:
            return None
        if len(view.request_ids) != row_count:
            raise RuntimeError("OverRIDE sampler provider requires one request id per decode row")
        row_ids = self._internal_decode_row_ids(row_count)
        result = self.engine.reweight_logits_batch(
            view.source_hidden,
            view.logits,
            request_ids=view.request_ids,
            row_ids=row_ids,
        )
        self._prepared_step = _PreparedSamplerStep(
            engine_step_id=int(view.engine_step_id),
            row_count=row_count,
            row_ids=row_ids,
        )
        affected_row_ids = torch.arange(row_count, device=view.logits.device, dtype=torch.long)
        dense_logits = ((2.0 * view.logits) - result.logits).detach()
        return CandidateModifierState(
            beta=1.0,
            backend="pre_filter_dense",
            affected_row_ids=affected_row_ids,
            pred_hidden=view.source_hidden.detach(),
            lm_head_weight=torch.empty((0, 0), device=view.logits.device, dtype=view.logits.dtype),
            lm_head_bias=None,
            precomputed_dense_logits=dense_logits,
        )

    def after_sample(self, view: SamplerStepView, sampled_token_ids: torch.Tensor) -> None:
        prepared = self._prepared_step
        if prepared is None:
            raise RuntimeError("OverRIDE sampler provider has no prepared sampler step")
        if int(view.engine_step_id) != prepared.engine_step_id:
            raise RuntimeError("OverRIDE sampler provider received sampled tokens for a different prepared sampler step")
        row_count = int(sampled_token_ids.reshape(-1).numel())
        if row_count != prepared.row_count:
            raise RuntimeError("OverRIDE sampler provider requires one sampled token per prepared decode row")
        self.engine.update_from_sampled_tokens_batch(
            sampled_token_ids,
            row_ids=prepared.row_ids,
        )
        self._prepared_step = None
