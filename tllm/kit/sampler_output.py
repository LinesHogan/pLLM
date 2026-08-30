"""Runtime service for sampler outputs declared by workload tasks."""

from __future__ import annotations

import torch
from typing import cast

from tllm.ports.sampler import (
    CandidateModifierState,
    SamplerBackend,
    SamplerHiddenReweightBatch,
)
from tllm.runtime.sampler_bridge.types import SamplerStepView
from tllm.runtime.vllm_patch.lm_head import resolve_lm_head
from tllm.runtime.sampler_bridge.context import SamplerStepContext
from tllm.runtime.sampler_bridge.provider import SamplerInterventionStage
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

    def intervention_stage(self) -> SamplerInterventionStage:
        backend = self._policy.backend.resolve(self._config)
        if (
            self._policy.kind is SamplerOutputKind.ADD_LOGITS
            or backend == "pre_filter_dense"
        ):
            return SamplerInterventionStage.PRE_FILTER
        return SamplerInterventionStage.POST_FILTER

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
        ):
            raise RuntimeError(
                f"sampler output {self._policy.name!r} does not match the active step"
            )
        match self._policy.kind:
            case SamplerOutputKind.HIDDEN_REWEIGHT:
                match record.value:
                    case SamplerHiddenReweightBatch(
                        pred_hidden=pred_hidden,
                        affected_row_ids=affected_row_ids,
                        total_rows=total_rows,
                    ):
                        if (
                            pred_hidden.ndim != 2
                            or int(pred_hidden.shape[0])
                            != len(affected_row_ids)
                            or pred_hidden.device != view.device
                            or total_rows != rows
                        ):
                            raise RuntimeError(
                                f"sampler output {self._policy.name!r} has invalid hidden subset metadata"
                            )
                        if not affected_row_ids:
                            return None
                        affected = torch.tensor(
                            affected_row_ids,
                            device=view.device,
                            dtype=torch.long,
                        )
                        row_map = torch.full(
                            (rows,),
                            -1,
                            device=view.device,
                            dtype=torch.long,
                        )
                        row_map.index_copy_(
                            0,
                            affected,
                            torch.arange(
                                len(affected_row_ids),
                                device=view.device,
                                dtype=torch.long,
                            ),
                        )
                    case torch.Tensor() as pred_hidden:
                        if (
                            pred_hidden.ndim != 2
                            or int(pred_hidden.shape[0]) != rows
                            or pred_hidden.device != view.device
                        ):
                            raise RuntimeError(
                                f"sampler output {self._policy.name!r} does not match the active step"
                            )
                        affected = torch.arange(
                            rows, device=view.device, dtype=torch.long
                        )
                        row_map = None
                head = resolve_lm_head(view.model)
                return CandidateModifierState(
                    beta=float(self._policy.beta.resolve(self._config)),
                    backend=cast(
                        SamplerBackend,
                        self._policy.backend.resolve(self._config),
                    ),
                    affected_row_ids=affected,
                    pred_hidden=pred_hidden,
                    lm_head_weight=head.weight,
                    lm_head_bias=head.bias,
                    pred_hidden_row_map=row_map,
                )
            case SamplerOutputKind.ADD_LOGITS:
                match record.value:
                    case torch.Tensor() as logit_delta:
                        if (
                            logit_delta.ndim != 2
                            or int(logit_delta.shape[0]) != rows
                            or logit_delta.device != view.device
                        ):
                            raise RuntimeError(
                                f"sampler output {self._policy.name!r} does not match the active step"
                            )
                    case SamplerHiddenReweightBatch():
                        raise RuntimeError(
                            f"sampler output {self._policy.name!r} requires a dense logit delta"
                        )
                affected = torch.arange(
                    rows, device=view.device, dtype=torch.long
                )
                return CandidateModifierState(
                    beta=1.0,
                    backend="pre_filter_dense",
                    affected_row_ids=affected,
                    pred_hidden=view.source_hidden.detach(),
                    lm_head_weight=torch.empty(
                        (0, 0), device=view.device, dtype=view.logits.dtype
                    ),
                    lm_head_bias=None,
                    precomputed_dense_logits=(view.logits + logit_delta).detach(),
                )

    def after_sample(
        self, view: SamplerStepView, sampled_token_ids: torch.Tensor
    ) -> None:
        self._context.sampled(view, sampled_token_ids)


__all__ = ["DeclaredSamplerOutput"]
