"""Exact ESamp workload expressed only as state, data flow, and algorithm."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from tllm.consumers.esamp.model import LowRankGatedResidualModel
from tllm.contracts.packed_rows import (
    PackedPhaseRowBatch,
    PackedSamplerRowBatch,
)
from tllm.ports.sampler import PackedSamplerGuidanceBatch
from tllm.runtime.request_trainable_state import (
    RequestStateLossBatch,
    RequestStateSelectionBatch,
)
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
    RequestKeyField,
    RequestTrainableState,
    State,
    TrainableState,
    WorkloadConfig,
    config_field,
    declared_config_field,
    hidden,
    optimize,
    packed_hidden,
    packed_request_rows,
    packed_sampler_rows,
    packed_selection,
    read,
    result,
    sampler,
    task,
    temporary,
    transition,
    request_key,
    when,
    workload,
)


_ESAMP_ENABLED = ConfigField.boolean(True)
_ESAMP_DISTILLER_LR = ConfigField.floating(
    1e-3, NumericConstraint.POSITIVE
)
_ESAMP_PER_REQUEST_MODE = ConfigField.boolean(False)


class ESampConfig(WorkloadConfig):
    source_layer_path: str = config_field(
        default="model.layers[0].input_layernorm", kind=ConfigKind.STR
    )
    target_layer_path: str = config_field(
        default="model.layers[-1].input_layernorm", kind=ConfigKind.STR
    )
    enabled: bool = declared_config_field(_ESAMP_ENABLED)
    distiller_hidden_dim: int = config_field(
        default=128, kind=ConfigKind.INT, constraint=NumericConstraint.POSITIVE
    )
    distiller_lr: float = declared_config_field(_ESAMP_DISTILLER_LR)
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
    per_request_models: bool = declared_config_field(_ESAMP_PER_REQUEST_MODE)


class ESampWorkload(Workload[ESampConfig]):
    distiller: TrainableState[LowRankGatedResidualModel] = (
        State[LowRankGatedResidualModel].replica(
            optimizer=AdamWSpec.configured(_ESAMP_DISTILLER_LR)
        )
    )
    request_distiller: RequestTrainableState[LowRankGatedResidualModel] = (
        State[LowRankGatedResidualModel].request_trainable(
            key=request_key(RequestKeyField.REQUEST_ID),
            optimizer=AdamWSpec.configured(_ESAMP_DISTILLER_LR),
        )
    )
    request_rows = packed_request_rows()
    request_sampler_rows = packed_sampler_rows(rows=request_rows)
    request_distiller_selection = packed_selection(
        request_distiller.selection(), rows=request_rows
    )

    def __init__(self, config: ESampConfig) -> None:
        if (
            config.per_request_models
            and config.training_row_selection != "none"
        ):
            raise ValueError(
                "per_request_models=True requires training_row_selection='none'"
            )
        if config.per_request_models and not config.output_layernorm:
            raise ValueError(
                "per_request_models=True requires output_layernorm=True"
            )
        super().__init__(config)

    @distiller.initialize
    def init_distiller(
        self, spec: RuntimeSpecializationSpec
    ) -> LowRankGatedResidualModel:
        return self._new_distiller(spec)

    @request_distiller.initialize
    def init_request_distiller(
        self, spec: RuntimeSpecializationSpec
    ) -> LowRankGatedResidualModel:
        train_model = self._new_distiller(spec)
        # Legacy ESamp constructs a train/forward pair for each owner, then
        # copies the train weights into the forward member.  Request state
        # needs only the trainable module, but preserving the paired random
        # initialization keeps the eager numerical contract exact.
        self._new_distiller(spec)
        return train_model

    def _new_distiller(
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
        when=when(_ESAMP_PER_REQUEST_MODE, False),
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
        when=when(_ESAMP_PER_REQUEST_MODE, False),
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

    @task(
        inputs={
            "rows": request_rows,
            "sampler_rows": request_sampler_rows,
            "source": packed_hidden(
                0,
                rows=request_rows,
                path=ESampConfig.source_layer_path,
            ),
            "target": packed_hidden(
                -1,
                rows=request_rows,
                path=ESampConfig.target_layer_path,
            ),
            "selection": request_distiller_selection,
        },
        outputs={
            "guidance": sampler.hidden_reweight(
                beta=ESampConfig.distiller_beta,
                enabled=ESampConfig.intervention,
                backend=ESampConfig.sampler_backend,
            ),
            "training_source": temporary(),
            "training_target": temporary(),
        },
        effects=(read(request_distiller_selection),),
        timing=Between("model.execute", "sampler.pre_draw"),
        when=when(_ESAMP_PER_REQUEST_MODE, True),
    )
    def request_predict(
        self,
        rows: PackedPhaseRowBatch,
        sampler_rows: PackedSamplerRowBatch,
        source: torch.Tensor,
        target: torch.Tensor,
        selection: RequestStateSelectionBatch,
    ) -> tuple[
        PackedSamplerGuidanceBatch,
        torch.Tensor,
        torch.Tensor,
    ]:
        groups = self.request_distiller.groups(selection)
        active_leases = {
            group.lease
            for group in groups
            if group.before_version > 0
        }
        affected_row_ids = tuple(
            link.sampler_row_index
            for link in sampler_rows.links
            if selection.rows[link.packed_row_index].lease in active_leases
        )
        with torch.no_grad():
            predictions_by_sampler_row = source.new_empty(
                (len(sampler_rows.links), int(source.shape[1]))
            )
            for group in groups:
                if group.before_version <= 0:
                    continue
                sampler_row_ids = tuple(
                    link.sampler_row_index
                    for link in sampler_rows.links
                    if selection.rows[link.packed_row_index].lease
                    == group.lease
                )
                if not sampler_row_ids:
                    continue
                packed_row_ids = torch.tensor(
                    tuple(
                        sampler_rows.links[row_id].packed_row_index
                        for row_id in sampler_row_ids
                    ),
                    device=source.device,
                    dtype=torch.long,
                )
                local_row_ids = torch.searchsorted(
                    group.canonical_row_indices, packed_row_ids
                )
                predictions_by_sampler_row.index_copy_(
                    0,
                    torch.tensor(
                        sampler_row_ids,
                        device=source.device,
                        dtype=torch.long,
                    ),
                    group.module(
                        source.index_select(0, group.canonical_row_indices)
                    ).index_select(0, local_row_ids),
                )
            prediction = predictions_by_sampler_row.index_select(
                0,
                torch.tensor(
                    affected_row_ids,
                    device=source.device,
                    dtype=torch.long,
                ),
            )
        return (
            PackedSamplerGuidanceBatch(
                sampler_rows=sampler_rows,
                pred_hidden=prediction,
                affected_row_ids=affected_row_ids,
            ),
            source,
            target,
        )

    @task(
        inputs={
            "rows": request_rows,
            "source": result("request_predict", "training_source"),
            "target": result("request_predict", "training_target"),
            "selection": request_distiller_selection,
        },
        outputs={"loss_batch": temporary()},
        effects=(read(request_distiller_selection),),
        timing=Between("token.commit", "next.prepare"),
        when=when(_ESAMP_PER_REQUEST_MODE, True),
    )
    def request_loss(
        self,
        rows: PackedPhaseRowBatch,
        source: torch.Tensor,
        target: torch.Tensor,
        selection: RequestStateSelectionBatch,
    ) -> RequestStateLossBatch:
        del rows
        losses = tuple(
            F.mse_loss(
                group.module(
                    source.index_select(0, group.canonical_row_indices)
                ).float(),
                target.index_select(0, group.canonical_row_indices).float(),
            )
            for group in self.request_distiller.groups(selection)
        )
        return RequestStateLossBatch(selection, losses)

    @task(
        inputs={
            "rows": request_rows,
            "loss_batch": result("request_loss", "loss_batch"),
            "selection": request_distiller_selection,
        },
        effects=(transition(request_distiller_selection),),
        timing=Before("next.prepare"),
        when=when(_ESAMP_PER_REQUEST_MODE, True),
    )
    def request_train(
        self,
        rows: PackedPhaseRowBatch,
        loss_batch: RequestStateLossBatch,
        selection: RequestStateSelectionBatch,
    ) -> None:
        del rows
        if loss_batch.selection is not selection:
            raise RuntimeError("request loss batch selection differs from transition")
        self.optimize_request_state(ESampWorkload.request_distiller, loss_batch)


workload(ESampWorkload, id="esamp", config=ESampConfig, enabled=_ESAMP_ENABLED)


__all__ = ["ESampConfig", "ESampWorkload"]
