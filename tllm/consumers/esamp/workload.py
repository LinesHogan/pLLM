"""Exact ESamp workload expressed only as state, data flow, and algorithm."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from tllm.consumers.esamp.model import (
    BatchedLowRankGatedResidualModel,
    LowRankGatedResidualModel,
)
from tllm.contracts.hidden_locator import HiddenPhase
from tllm.contracts.packed_rows import (
    PackedPhaseRowBatch,
    PackedSamplerRowBatch,
)
from tllm.ports.sampler import PackedSamplerGuidanceBatch
from tllm.runtime.request_trainable_state import (
    FixedBatchedSlotView,
    ResidentSlotView,
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
    FixedBatchedRequestTrainableState,
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
_ESAMP_STATE_MODE: ConfigField[
    Literal["shared", "per_request", "model_bank"]
] = ConfigField.choice(
    "shared",
    ("shared", "per_request", "model_bank"),
)


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
    state_mode: Literal["shared", "per_request", "model_bank"] = (
        declared_config_field(_ESAMP_STATE_MODE)
    )
    request_capacity: int = config_field(
        default=0,
        kind=ConfigKind.INT,
        constraint=NumericConstraint.NONNEGATIVE,
    )


class ESampWorkload(Workload[ESampConfig]):
    distiller: TrainableState[LowRankGatedResidualModel] = (
        State[LowRankGatedResidualModel].replica(
            optimizer=AdamWSpec.configured(_ESAMP_DISTILLER_LR)
        )
    )
    request_distiller: RequestTrainableState[
        LowRankGatedResidualModel
    ] = State[LowRankGatedResidualModel].request_trainable(
        key=request_key(RequestKeyField.REQUEST_ID),
        optimizer=AdamWSpec.configured(_ESAMP_DISTILLER_LR),
    )
    request_distiller_bank: FixedBatchedRequestTrainableState[
        BatchedLowRankGatedResidualModel
    ] = (
        State[BatchedLowRankGatedResidualModel].request_batched_trainable(
            key=request_key(RequestKeyField.REQUEST_ID),
            optimizer=AdamWSpec.configured(_ESAMP_DISTILLER_LR),
        )
    )
    request_rows = packed_request_rows()
    request_sampler_rows = packed_sampler_rows(rows=request_rows)
    request_distiller_selection = packed_selection(
        request_distiller.selection(),
        rows=request_rows,
        phase=HiddenPhase.DECODE,
    )
    request_distiller_bank_selection = packed_selection(
        request_distiller_bank.selection(),
        rows=request_rows,
        phase=HiddenPhase.DECODE,
    )

    def __init__(self, config: ESampConfig) -> None:
        if config.state_mode != "model_bank" and config.request_capacity:
            raise ValueError(
                "request_capacity applies only to model_bank state"
            )
        if (
            config.state_mode != "shared"
            and config.training_row_selection != "none"
        ):
            raise ValueError(
                "request state requires training_row_selection='none'"
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
        self._new_distiller(spec)
        return train_model

    @request_distiller_bank.initialize
    def init_request_distiller_bank(
        self, spec: RuntimeSpecializationSpec
    ) -> BatchedLowRankGatedResidualModel:
        return BatchedLowRankGatedResidualModel(
            capacity=self.config.request_capacity or spec.rows,
            hidden_size=spec.hidden_size,
            rank=min(self.config.distiller_hidden_dim, spec.hidden_size),
            use_output_layernorm=self.config.output_layernorm,
            device=spec.device,
            dtype=spec.dtype,
        )

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
        when=when(_ESAMP_STATE_MODE, "shared"),
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
        when=when(_ESAMP_STATE_MODE, "shared"),
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
        when=when(_ESAMP_STATE_MODE, "per_request"),
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
        del rows
        match self.request_distiller.selected(selection):
            case ResidentSlotView(groups=groups):
                pass
            case _:
                raise RuntimeError(
                    "ESamp resident request state requires resident storage"
                )
        active_leases = {
            group.lease
            for group in groups
            if group.before_version > 0
        }
        lease_by_packed_row = {
            row.canonical_row_index: row.lease
            for row in selection.rows
        }
        affected_row_ids = tuple(
            link.sampler_row_index
            for link in sampler_rows.links
            if (
                link.packed_row_index in lease_by_packed_row
                and lease_by_packed_row[link.packed_row_index]
                in active_leases
            )
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
                    if (
                        link.packed_row_index in lease_by_packed_row
                        and lease_by_packed_row[link.packed_row_index]
                        == group.lease
                    )
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
                    group.canonical_row_indices,
                    packed_row_ids,
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
        when=when(_ESAMP_STATE_MODE, "per_request"),
    )
    def request_loss(
        self,
        rows: PackedPhaseRowBatch,
        source: torch.Tensor,
        target: torch.Tensor,
        selection: RequestStateSelectionBatch,
    ) -> RequestStateLossBatch:
        del rows
        match self.request_distiller.selected(selection):
            case ResidentSlotView(groups=groups):
                pass
            case _:
                raise RuntimeError(
                    "ESamp resident request state requires resident storage"
                )
        losses = tuple(
            F.mse_loss(
                group.module(
                    source.index_select(0, group.canonical_row_indices)
                ).float(),
                target.index_select(0, group.canonical_row_indices).float(),
            )
            for group in groups
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
        when=when(_ESAMP_STATE_MODE, "per_request"),
    )
    def request_train(
        self,
        rows: PackedPhaseRowBatch,
        loss_batch: RequestStateLossBatch,
        selection: RequestStateSelectionBatch,
    ) -> None:
        del rows
        if loss_batch.selection is not selection:
            raise RuntimeError(
                "request loss batch selection differs from transition"
            )
        self.optimize_request_state(
            ESampWorkload.request_distiller,
            loss_batch,
        )

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
            "selection": request_distiller_bank_selection,
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
        effects=(read(request_distiller_bank_selection),),
        timing=Between("model.execute", "sampler.pre_draw"),
        when=when(_ESAMP_STATE_MODE, "model_bank"),
    )
    def bank_predict(
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
        match self.request_distiller_bank.selected(selection):
            case FixedBatchedSlotView(module=bank):
                pass
            case _:
                raise RuntimeError(
                    "ESamp request state requires fixed batched storage"
                )
        active_leases = {
            lease
            for lease, version in zip(
                selection.leases,
                selection.before_versions,
                strict=True,
            )
            if version > 0
        }
        lease_by_packed_row = {
            row.canonical_row_index: row.lease
            for row in selection.rows
        }
        affected_links = tuple(
            link
            for link in sampler_rows.links
            if (
                link.packed_row_index in lease_by_packed_row
                and lease_by_packed_row[link.packed_row_index]
                in active_leases
            )
        )
        affected_row_ids = tuple(
            link.sampler_row_index for link in affected_links
        )
        slot_by_packed_row = {
            row.canonical_row_index: row.lease.slot.slot_id
            for row in selection.rows
        }
        packed_row_ids = torch.tensor(
            tuple(
                link.packed_row_index for link in affected_links
            ),
            device=source.device,
            dtype=torch.long,
        )
        slot_ids = torch.tensor(
            tuple(
                slot_by_packed_row[link.packed_row_index]
                for link in affected_links
            ),
            device=source.device,
            dtype=torch.long,
        )
        with torch.no_grad():
            prediction = bank(
                slot_ids,
                source.index_select(0, packed_row_ids),
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
            "source": result("bank_predict", "training_source"),
            "target": result("bank_predict", "training_target"),
            "selection": request_distiller_bank_selection,
        },
        outputs={"loss_batch": temporary()},
        effects=(read(request_distiller_bank_selection),),
        timing=Between("token.commit", "next.prepare"),
        when=when(_ESAMP_STATE_MODE, "model_bank"),
    )
    def bank_loss(
        self,
        rows: PackedPhaseRowBatch,
        source: torch.Tensor,
        target: torch.Tensor,
        selection: RequestStateSelectionBatch,
    ) -> RequestStateLossBatch:
        del rows
        match self.request_distiller_bank.selected(selection):
            case FixedBatchedSlotView(module=bank, unique_slot_ids=slot_ids):
                pass
            case _:
                raise RuntimeError(
                    "ESamp request state requires fixed batched storage"
                )
        first_rows = tuple(
            next(
                row.canonical_row_index
                for row in selection.rows
                if row.lease == lease
            )
            for lease in selection.leases
        )
        row_ids = torch.tensor(
            first_rows,
            device=source.device,
            dtype=torch.long,
        )
        prediction = bank(slot_ids, source.index_select(0, row_ids))
        per_slot_loss = (
            prediction.float() - target.index_select(0, row_ids).float()
        ).square().mean(dim=1)
        losses = tuple(
            loss
            for loss in per_slot_loss.unbind()
        )
        return RequestStateLossBatch(selection, losses)

    @task(
        inputs={
            "rows": request_rows,
            "loss_batch": result("bank_loss", "loss_batch"),
            "selection": request_distiller_bank_selection,
        },
        effects=(transition(request_distiller_bank_selection),),
        timing=Before("next.prepare"),
        when=when(_ESAMP_STATE_MODE, "model_bank"),
    )
    def bank_train(
        self,
        rows: PackedPhaseRowBatch,
        loss_batch: RequestStateLossBatch,
        selection: RequestStateSelectionBatch,
    ) -> None:
        del rows
        if loss_batch.selection is not selection:
            raise RuntimeError("request loss batch selection differs from transition")
        self.optimize_request_state(
            ESampWorkload.request_distiller_bank,
            loss_batch,
        )


workload(ESampWorkload, id="esamp", config=ESampConfig, enabled=_ESAMP_ENABLED)


__all__ = ["ESampConfig", "ESampWorkload"]
