"""Runtime service for sampler outputs declared by workload tasks."""

from __future__ import annotations

import torch
from typing import cast

from tllm.ports.sampler import CandidateModifierState, SamplerBackend
from tllm.runtime.sampler_bridge.types import SamplerStepView
from tllm.runtime.vllm_patch.lm_head import resolve_lm_head
from tllm.runtime.sampler_bridge.context import SamplerStepContext
from tllm.runtime.workload_sinks import MissingSinkRecord, SinkRecord, SinkStore
from tllm.workload.config import WorkloadConfig
from tllm.workload.frontend import SamplerOutput, SamplerOutputKind


class DeclaredSamplerOutput:
    """Consume one signed sampler sink without workload-owned callbacks."""

    def __init__(
        self,
        policy: SamplerOutput,
        config: WorkloadConfig,
        store: SinkStore,
        context: SamplerStepContext,
    ) -> None:
        self._policy = policy
        self._config = config
        self._store = store
        self._context = context

    def is_active(self) -> bool:
        return bool(self._policy.enabled.resolve(self._config)) and float(
            self._policy.beta.resolve(self._config)
        ) != 0.0

    def prepare_step(self, view: SamplerStepView) -> CandidateModifierState | None:
        self._context.prepare(view)
        if not self.is_active():
            return None
        match self._store.lookup(self._policy.name, view.engine_step_id):
            case MissingSinkRecord():
                if self._policy.required and view.decode_count > 0:
                    raise RuntimeError(
                        f"sampler output {self._policy.name!r} was not published"
                    )
                return None
            case SinkRecord() as record:
                pass
        identity = record.identity
        rows = view.decode_count
        if (
            identity.rows != rows
            or identity.request_ids != tuple(view.request_ids[:rows])
            or identity.prompt_idxs != tuple(view.prompt_idxs[:rows])
            or identity.sample_idxs != tuple(view.sample_idxs[:rows])
            or identity.phase != view.phase
            or record.value.device != view.device
        ):
            raise RuntimeError(
                f"sampler output {self._policy.name!r} does not match the active step"
            )
        affected = torch.arange(rows, device=view.device, dtype=torch.long)
        match self._policy.kind:
            case SamplerOutputKind.HIDDEN_REWEIGHT:
                head = resolve_lm_head(view.model)
                return CandidateModifierState(
                    beta=float(self._policy.beta.resolve(self._config)),
                    backend=cast(
                        SamplerBackend,
                        self._policy.backend.resolve(self._config),
                    ),
                    affected_row_ids=affected,
                    pred_hidden=record.value,
                    lm_head_weight=head.weight,
                    lm_head_bias=head.bias,
                )
            case SamplerOutputKind.ADD_LOGITS:
                return CandidateModifierState(
                    beta=1.0,
                    backend="pre_filter_dense",
                    affected_row_ids=affected,
                    pred_hidden=view.source_hidden.detach(),
                    lm_head_weight=torch.empty(
                        (0, 0), device=view.device, dtype=view.logits.dtype
                    ),
                    lm_head_bias=None,
                    precomputed_dense_logits=(view.logits + record.value).detach(),
                )

    def after_sample(
        self, view: SamplerStepView, sampled_token_ids: torch.Tensor
    ) -> None:
        self._context.sampled(view, sampled_token_ids)


__all__ = ["DeclaredSamplerOutput"]
