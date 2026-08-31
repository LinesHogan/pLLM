"""CUDA task implementations selected by the canonical Rust scheduler."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.canonical.cuda_stream import CudaStreamReadiness
from tllm.canonical.kernels.low_rank_gated import (
    LowRankGatedLayout,
    low_rank_gated_forward,
)
from tllm.canonical.ptts import (
    CudaDensePageGradientBatch,
    CudaDensePagePublication,
    CudaDensePageStateProvider,
    CudaDensePageVersionPin,
    DensePageAdamWConfig,
    DensePageAdamWFieldBinding,
)
from tllm.canonical.ptts.request_state import (
    RequestStateDirectory,
    RequestStateSelection,
    RequestStateTransitionLease,
)
from tllm.canonical.workloads.esamp_request_types import (
    ESampRequestConfig,
    ESampRequestStepInput,
    ESAMP_EXP_AVG_FIELD,
    ESAMP_EXP_AVG_SQ_FIELD,
    ESAMP_GRADIENT_FIELD,
    ESAMP_PARAMETER_FIELD,
    ESAMP_REQUEST_CUDA_LOSS_IMPLEMENTATION,
    ESAMP_REQUEST_CUDA_PREDICT_IMPLEMENTATION,
    ESAMP_REQUEST_CUDA_TRAIN_IMPLEMENTATION,
    ESAMP_STEP_FIELD,
)
from tllm.contracts.sampler import PackedSamplerGuidanceBatch
from tllm.contracts.semantic_graph import ImplementationId
from tllm.contracts.state_ref import StateRef


@dataclass(frozen=True, slots=True)
class ESampCudaPredictionPending:
    pass


@dataclass(frozen=True, slots=True)
class ESampCudaPredicted:
    guidance: PackedSamplerGuidanceBatch
    training_source: torch.Tensor
    training_target: torch.Tensor
    state_vectors: torch.Tensor
    row_state_indices: torch.Tensor
    readiness: CudaStreamReadiness


ESampCudaPrediction = ESampCudaPredictionPending | ESampCudaPredicted


@dataclass(frozen=True, slots=True)
class ESampCudaOptimizationPending:
    pass


@dataclass(frozen=True, slots=True)
class ESampCudaOptimized:
    losses: tuple[torch.Tensor, ...]
    gradients: CudaDensePageGradientBatch


ESampCudaOptimization = ESampCudaOptimizationPending | ESampCudaOptimized


@dataclass(frozen=True, slots=True)
class ESampCudaPublicationPending:
    pass


@dataclass(frozen=True, slots=True)
class ESampCudaPublished:
    before_refs: tuple[StateRef, ...]
    after_refs: tuple[StateRef, ...]
    publication: CudaDensePagePublication


ESampCudaPublication = ESampCudaPublicationPending | ESampCudaPublished


@dataclass(slots=True)
class ESampCudaStepContext:
    step: ESampRequestStepInput
    source: torch.Tensor
    target: torch.Tensor
    selection: RequestStateSelection
    transition_lease: RequestStateTransitionLease
    state_pin: CudaDensePageVersionPin
    stream: torch.cuda.Stream
    prediction: ESampCudaPrediction
    optimization: ESampCudaOptimization
    publication: ESampCudaPublication


class ESampCudaPredictTask:
    __slots__ = ("_config", "_layout", "_provider")

    def __init__(
        self,
        config: ESampRequestConfig,
        layout: LowRankGatedLayout,
        provider: CudaDensePageStateProvider,
    ) -> None:
        self._config = config
        self._layout = layout
        self._provider = provider

    @property
    def implementation_id(self) -> ImplementationId:
        return ESAMP_REQUEST_CUDA_PREDICT_IMPLEMENTATION

    def execute(self, context: ESampCudaStepContext) -> None:
        state_vectors = self._provider.pinned_field_rows(
            context.state_pin,
            ESAMP_PARAMETER_FIELD,
            context.stream,
        )
        with torch.cuda.stream(context.stream):
            packed_row_indices = torch.tensor(
                tuple(row.canonical_row_index for row in context.selection.rows),
                dtype=torch.long,
                device=context.source.device,
            )
            row_state_indices = torch.tensor(
                tuple(row.state_index for row in context.selection.rows),
                dtype=torch.long,
                device=context.source.device,
            )
            training_source = context.source.index_select(0, packed_row_indices)
            training_target = context.target.index_select(0, packed_row_indices)
            local_row_by_packed = {
                row.canonical_row_index: position
                for position, row in enumerate(context.selection.rows)
            }
            affected_row_ids: list[int] = []
            prediction_positions: list[int] = []
            if self._config.intervention:
                for link in context.step.sampler_rows.links:
                    if link.packed_row_index not in local_row_by_packed:
                        continue
                    local_position = local_row_by_packed[link.packed_row_index]
                    selected_row = context.selection.rows[local_position]
                    selected_ref = context.selection.refs[selected_row.state_index]
                    if int(selected_ref.version_id) > 0:
                        affected_row_ids.append(link.sampler_row_index)
                        prediction_positions.append(local_position)
            if prediction_positions:
                all_predictions = low_rank_gated_forward(
                    training_source,
                    state_vectors,
                    row_state_indices,
                    self._layout,
                )
                pred_hidden = all_predictions.index_select(
                    0,
                    torch.tensor(
                        prediction_positions,
                        dtype=torch.long,
                        device=all_predictions.device,
                    ),
                )
            else:
                pred_hidden = training_source.new_empty(
                    (0, self._layout.hidden_size)
                )
            readiness = CudaStreamReadiness(context.stream)
        context.prediction = ESampCudaPredicted(
            PackedSamplerGuidanceBatch(
                context.step.sampler_rows,
                pred_hidden,
                tuple(affected_row_ids),
            ),
            training_source,
            training_target,
            state_vectors,
            row_state_indices,
            readiness,
        )


class ESampCudaLossTask:
    __slots__ = ("_layout", "_provider")

    def __init__(
        self,
        layout: LowRankGatedLayout,
        provider: CudaDensePageStateProvider,
    ) -> None:
        self._layout = layout
        self._provider = provider

    @property
    def implementation_id(self) -> ImplementationId:
        return ESAMP_REQUEST_CUDA_LOSS_IMPLEMENTATION

    def execute(self, context: ESampCudaStepContext) -> None:
        match context.prediction:
            case ESampCudaPredicted() as predicted:
                pass
            case _:
                raise RuntimeError("CUDA ESamp loss requires prediction")
        with torch.cuda.stream(context.stream):
            parameters = predicted.state_vectors.detach().requires_grad_(True)
            prediction = low_rank_gated_forward(
                predicted.training_source,
                parameters,
                predicted.row_state_indices,
                self._layout,
            )
            row_losses = (
                prediction.float() - predicted.training_target.float()
            ).square().mean(dim=1)
            state_count = int(parameters.shape[0])
            loss_sums = torch.zeros(
                (state_count,),
                dtype=row_losses.dtype,
                device=row_losses.device,
            ).index_add(0, predicted.row_state_indices, row_losses)
            loss_counts = torch.zeros_like(loss_sums).index_add(
                0,
                predicted.row_state_indices,
                torch.ones_like(row_losses),
            )
            losses = loss_sums / loss_counts
            (gradient,) = torch.autograd.grad(losses.sum(), parameters)
            gradient_values = tuple(
                gradient[index].detach() for index in range(state_count)
            )
        context.optimization = ESampCudaOptimized(
            tuple(loss.detach() for loss in losses),
            self._provider.bind_gradients(
                gradient_values,
                context.stream,
            ),
        )


class ESampCudaTrainTask:
    __slots__ = ("_config", "_directory", "_provider")

    def __init__(
        self,
        config: ESampRequestConfig,
        directory: RequestStateDirectory,
        provider: CudaDensePageStateProvider,
    ) -> None:
        self._config = config
        self._directory = directory
        self._provider = provider

    @property
    def implementation_id(self) -> ImplementationId:
        return ESAMP_REQUEST_CUDA_TRAIN_IMPLEMENTATION

    def execute(self, context: ESampCudaStepContext) -> None:
        match context.optimization:
            case ESampCudaOptimized() as optimized:
                pass
            case _:
                raise RuntimeError("CUDA ESamp train requires loss")
        logical = self._directory.prepare_transition(
            context.selection,
            context.transition_lease,
        )
        try:
            physical = self._provider.prepare_transition_batch(
                logical.before_refs,
                logical.after_refs,
                context.stream,
            )
            publication = self._provider.submit_adamw_transition(
                physical,
                DensePageAdamWFieldBinding(
                    ESAMP_PARAMETER_FIELD,
                    ESAMP_GRADIENT_FIELD,
                    ESAMP_STEP_FIELD,
                    ESAMP_EXP_AVG_FIELD,
                    ESAMP_EXP_AVG_SQ_FIELD,
                ),
                DensePageAdamWConfig(
                    self._config.learning_rate,
                    self._config.betas,
                    self._config.eps,
                    self._config.weight_decay,
                ),
                optimized.gradients,
                context.stream,
            )
        except Exception:
            self._directory.abort_transition(logical)
            raise
        self._directory.commit_transition(logical)
        context.publication = ESampCudaPublished(
            logical.before_refs,
            logical.after_refs,
            publication,
        )


__all__ = [
    "ESampCudaLossTask",
    "ESampCudaOptimizationPending",
    "ESampCudaOptimized",
    "ESampCudaPredicted",
    "ESampCudaPredictionPending",
    "ESampCudaPredictTask",
    "ESampCudaPublicationPending",
    "ESampCudaPublished",
    "ESampCudaStepContext",
    "ESampCudaTrainTask",
]
