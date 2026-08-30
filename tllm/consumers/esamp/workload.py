"""Exact ESamp workload expressed only as state, data flow, and algorithm."""

from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn.functional as F

from tllm.consumers.esamp.model import LowRankGatedResidualModel
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.torch_adapter import torch_assert_async
from tllm.kit.workload import Workload
from tllm.workload import (
    AdamWSpec,
    Before,
    Between,
    ConfigField,
    ConfigKind,
    NumericConstraint,
    State,
    TrainableState,
    WorkloadConfig,
    config_field,
    hidden,
    optimize,
    read,
    result,
    sampler,
    task,
    temporary,
    workload,
)


_ESAMP_ENABLED = ConfigField.boolean(True)
_ESAMP_DISTILLER_LR = ConfigField.floating(
    1e-3, NumericConstraint.POSITIVE
)


class ESampConfig(WorkloadConfig):
    source_layer_path: str = config_field(
        default="model.model.layers[0].input_layernorm", kind=ConfigKind.STR
    )
    target_layer_path: str = config_field(
        default="model.model.layers[-1].input_layernorm", kind=ConfigKind.STR
    )
    enabled: bool = cast(bool, _ESAMP_ENABLED)
    distiller_hidden_dim: int = config_field(
        default=128, kind=ConfigKind.INT, constraint=NumericConstraint.POSITIVE
    )
    distiller_lr: float = cast(float, _ESAMP_DISTILLER_LR)
    output_layernorm: bool = config_field(default=True, kind=ConfigKind.BOOL)
    intervention: bool = config_field(default=False, kind=ConfigKind.BOOL)
    distiller_beta: float = config_field(
        default=0.0, kind=ConfigKind.FLOAT, constraint=NumericConstraint.NONNEGATIVE
    )
    sampler_backend: Literal["post_filter_exact"] = config_field(
        default="post_filter_exact", kind=ConfigKind.STR
    )
    training_row_selection: Literal["none", "first_per_prompt"] = config_field(
        default="none", kind=ConfigKind.STR
    )


class ESampWorkload(Workload[ESampConfig]):
    distiller: TrainableState[LowRankGatedResidualModel] = (
        State[LowRankGatedResidualModel].replica(
            optimizer=AdamWSpec.configured(_ESAMP_DISTILLER_LR)
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
            "sample_index_tensor": sampler.sample_index_tensor(),
        },
        outputs={
            "guidance": sampler.hidden_reweight(
                beta=ESampConfig.distiller_beta,
                enabled=ESampConfig.intervention,
                backend=ESampConfig.sampler_backend,
            ),
            "training_source": temporary(),
            "target": temporary(),
            "sample_index_tensor": temporary(),
        },
        effects=(read(distiller),),
        timing=Between("model.execute", "sampler.pre_draw"),
    )
    def predict(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        sample_index_tensor: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        prediction = self.distiller(source)
        return prediction, source, target, sample_index_tensor

    @task(
        inputs={
            "source": result("predict", "training_source"),
            "target": result("predict", "target"),
            "sample_index_tensor": result(
                "predict", "sample_index_tensor"
            ),
        },
        outputs={
            "loss": optimize(
                distiller, at=Before("next.prepare")
            ),
        },
        timing=Between("token.commit", "next.prepare"),
    )
    def loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        sample_index_tensor: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self.distiller(source)
        if self.config.training_row_selection == "first_per_prompt":
            per_row_loss = (prediction.float() - target.float()).square().mean(
                dim=1
            )
            selected = (sample_index_tensor == 0).to(per_row_loss)
            selected_count = selected.sum()
            torch_assert_async(
                selected_count > 0,
                "first_per_prompt requires at least one sample_idx=0 row",
            )
            loss = (per_row_loss * selected).sum() / selected_count
        else:
            loss = F.mse_loss(prediction.float(), target.float())
        return loss


workload(ESampWorkload, id="esamp", config=ESampConfig, enabled=_ESAMP_ENABLED)


__all__ = ["ESampConfig", "ESampWorkload"]
