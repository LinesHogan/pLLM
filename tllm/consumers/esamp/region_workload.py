"""Native ESamp Region workload — the algorithm, nothing else.

ESamp: a low-rank gated residual distiller reads the first and last
layers' decode-phase hidden states, predicts the target hidden, trains by
MSE on a signed AdamW transition, and — when intervention is enabled —
reweights the sampler's candidates with the prediction.

Everything else is owned by tllm.kit: publication storage and row identity
(SinkStore), sampler alignment checks (SamplerModifier), the grad context
(from the signed effects), optimizer lifecycle, specialization key, and
worker reconstruction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from tllm.consumers.esamp.model import LowRankGatedResidualModel
from tllm.kit import RegionAlgorithm, candidate_modifier_state
from tllm.ports.residual_stream import ResidualLocator
from tllm.ports.sampler import CandidateModifierState
from tllm.runtime.sampler_bridge.types import SamplerStepView
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.workload import (
    AdamWSpec,
    Before,
    Between,
    State,
    StateScope,
    extern,
    hidden,
    read,
    result,
    sink,
    task,
    temporary,
    transition,
    workload,
)


@dataclass
class ESampRegionConfig:
    """Everything the native ESamp Region supports — and nothing it does not."""

    source_layer_path: str = "model.model.layers[0].input_layernorm"
    target_layer_path: str = "model.model.layers[-1].input_layernorm"
    enable_esamp_training: bool = True
    distiller_hidden_dim: int = 128
    distiller_lr: float = 1e-3
    output_layernorm: bool = True
    enable_distiller_intervention: bool = False
    distiller_beta: float = 0.0
    distiller_sampler_backend: Literal["post_filter_exact"] = "post_filter_exact"


@workload(id="esamp.region")
class ESampRegionWorkload(RegionAlgorithm[ESampRegionConfig]):
    """Replica-state ESamp algorithm ready for native Region materialization."""

    algorithm = "esamp-exact-region-v1"
    config_type = ESampRegionConfig
    enabled_field = "enable_esamp_training"

    distiller = State(
        scope=StateScope.REPLICA, parameters="distiller", optimizer=AdamWSpec(lr=1e-3)
    )

    def residual_paths(self) -> Mapping[object, str]:
        return {
            ResidualLocator(0, "block_output", "decode"): self.config.source_layer_path,
            ResidualLocator(-1, "block_output", "decode"): self.config.target_layer_path,
        }

    def init_parameters(
        self, spec: RuntimeSpecializationSpec
    ) -> Mapping[str, torch.nn.Module]:
        return {
            "distiller": LowRankGatedResidualModel(
                hidden_size=int(spec.hidden_size),
                rank=max(1, min(int(self.config.distiller_hidden_dim), int(spec.hidden_size))),
                use_output_layernorm=bool(self.config.output_layernorm),
                device=spec.device,
                dtype=spec.dtype,
            )
        }

    def optimizer_overrides(self) -> Mapping[str, object]:
        # The signed optimizer lr is nominal; the physical lr is pinned per
        # instance by the specialization key.
        return {"lr": float(self.config.distiller_lr)}

    @task(
        inputs={
            "source": hidden(0, "block_output", "decode"),
            "target": hidden(-1, "block_output", "decode"),
        },
        outputs={"guidance": sink("guidance"), "loss": temporary()},
        effects=(read(distiller),),
        timing=Between("model.execute", "sampler.pre_draw"),
    )
    def predict_and_loss(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The clones detach the engine's inference-mode rows so autograd can
        # save them for the signed update.
        prediction = self.model(source.clone())
        loss = F.mse_loss(prediction.float(), target.clone().float())
        return prediction.detach(), loss

    @task(
        inputs={"loss": result("predict_and_loss", "loss")},
        effects=(transition(distiller),),
        timing=Before("next.prepare"),
        implementation=extern("tllm.esamp:adamw_transition@1"),
    )
    def update(self, loss: torch.Tensor) -> None:
        self.step_update(loss)

    def modifier_active(self) -> bool:
        return (
            bool(self.config.enable_distiller_intervention)
            and float(self.config.distiller_beta) != 0.0
        )

    def modify_candidates(
        self, value: torch.Tensor, view: SamplerStepView
    ) -> CandidateModifierState:
        return candidate_modifier_state(
            beta=float(self.config.distiller_beta),
            backend=self.config.distiller_sampler_backend,
            pred_hidden=value,
            view=view,
        )

    def extra_stats(self) -> Mapping[str, int]:
        return {"guidance_rows": self.resources.sinks.latest_rows("guidance")}


__all__ = ["ESampRegionConfig", "ESampRegionWorkload"]
