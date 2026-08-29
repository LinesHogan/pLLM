"""Exact ESamp workload expressed only as state, data flow, and algorithm."""

from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn.functional as F

from tllm.consumers.esamp.model import LowRankGatedResidualModel
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.kit.workload import Workload
from tllm.workload import (
    AdamWSpec,
    Before,
    Between,
    ConfigField,
    NumericConstraint,
    State,
    TrainableState,
    WorkloadConfig,
    config_field,
    hidden,
    optimize,
    sampler,
    task,
    workload,
)


_ESAMP_ENABLED = ConfigField(True)


class ESampConfig(WorkloadConfig):
    source_layer_path: str = config_field(
        default="model.model.layers[0].input_layernorm"
    )
    target_layer_path: str = config_field(
        default="model.model.layers[-1].input_layernorm"
    )
    enabled: bool = cast(bool, _ESAMP_ENABLED)
    distiller_hidden_dim: int = config_field(
        default=128, constraint=NumericConstraint.POSITIVE
    )
    distiller_lr: float = config_field(
        default=1e-3, constraint=NumericConstraint.POSITIVE
    )
    output_layernorm: bool = config_field(default=True)
    intervention: bool = config_field(default=False)
    distiller_beta: float = config_field(
        default=0.0, constraint=NumericConstraint.NONNEGATIVE
    )
    sampler_backend: Literal["post_filter_exact"] = config_field(
        default="post_filter_exact"
    )


class ESampWorkload(Workload[ESampConfig]):
    distiller: State[LowRankGatedResidualModel] = (
        State[LowRankGatedResidualModel].replica(
            optimizer=AdamWSpec(lr=ESampConfig.distiller_lr)
        )
    )

    @distiller.initialize
    def init_distiller(
        self, spec: RuntimeSpecializationSpec
    ) -> LowRankGatedResidualModel:
        return LowRankGatedResidualModel(
            hidden_size=spec.hidden_size,
            rank=min(self.config.distiller_hidden_dim, spec.hidden_size),
            use_output_layernorm=self.config.output_layernorm,
            device=spec.device,
            dtype=spec.dtype,
        )

    @task(
        inputs={
            "source": hidden(
                0,
                "block_output",
                "decode",
                path=ESampConfig.source_layer_path,
            ),
            "target": hidden(
                -1,
                "block_output",
                "decode",
                path=ESampConfig.target_layer_path,
            ),
        },
        outputs={
            "guidance": sampler.hidden_reweight(
                beta=ESampConfig.distiller_beta,
                enabled=ESampConfig.intervention,
                backend=ESampConfig.sampler_backend,
            ),
            "loss": optimize(
                cast(TrainableState[object], distiller), at=Before("next.prepare")
            ),
        },
        timing=Between("model.execute", "sampler.pre_draw"),
    )
    def predict_and_loss(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self.distiller(source)
        return prediction, F.mse_loss(prediction.float(), target.float())


workload(ESampWorkload, id="esamp", config=ESampConfig, enabled=_ESAMP_ENABLED)


__all__ = ["ESampConfig", "ESampWorkload"]
