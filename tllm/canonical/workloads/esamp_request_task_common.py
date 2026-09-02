"""Shared ESamp request task contracts and math helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.canonical.cuda_stream import CudaStreamReadiness
from tllm.canonical.kernels.low_rank_gated import LowRankGatedLayout, low_rank_gated_forward
from tllm.canonical.ptts import CudaDensePageGradientBatch, CudaDensePagePublication
from tllm.canonical.ptts import CudaDensePageStateProvider, CudaDensePageVersionPin
from tllm.canonical.ptts.dense_page import (
    DensePageAdamWConfig,
    DensePageAdamWFieldBinding,
    DensePageStateProvider,
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
    ESAMP_REQUEST_LOSS_IMPLEMENTATION,
    ESAMP_REQUEST_PREDICT_IMPLEMENTATION,
    ESAMP_REQUEST_TRAIN_IMPLEMENTATION,
    ESAMP_STEP_FIELD,
)
from tllm.contracts.sampler import PackedSamplerGuidanceBatch
from tllm.contracts.semantic_graph import ImplementationId
from tllm.contracts.state_ref import StateRef


@dataclass(frozen=True, slots=True)
class ESampPredictionPending:
    pass


@dataclass(frozen=True, slots=True)
class ESampPredicted:
    guidance: PackedSamplerGuidanceBatch
    training_source: torch.Tensor
    training_target: torch.Tensor
    state_vectors: torch.Tensor
    row_state_indices: torch.Tensor
    readiness: CudaStreamReadiness | None = None


ESampPrediction = ESampPredictionPending | ESampPredicted


@dataclass(frozen=True, slots=True)
class ESampOptimizationPending:
    pass


@dataclass(frozen=True, slots=True)
class ESampOptimized:
    losses: tuple[torch.Tensor, ...]
    gradients: tuple[torch.Tensor, ...] | CudaDensePageGradientBatch


ESampOptimization = ESampOptimizationPending | ESampOptimized


@dataclass(frozen=True, slots=True)
class ESampPublicationPending:
    pass


@dataclass(frozen=True, slots=True)
class ESampPublished:
    before_refs: tuple[StateRef, ...]
    after_refs: tuple[StateRef, ...]
    publication: CudaDensePagePublication | None = None


ESampPublication = ESampPublicationPending | ESampPublished


@dataclass(slots=True)
class ESampCpuStepContext:
    step: ESampRequestStepInput
    source: torch.Tensor
    target: torch.Tensor
    selection: RequestStateSelection
    transition_lease: RequestStateTransitionLease
    prediction: ESampPrediction
    optimization: ESampOptimization
    publication: ESampPublication


@dataclass(slots=True)
class ESampCudaStepContext(ESampCpuStepContext):
    state_pin: CudaDensePageVersionPin
    stream: torch.cuda.Stream


def esamp_adamw_field_binding() -> DensePageAdamWFieldBinding:
    return DensePageAdamWFieldBinding(
        ESAMP_PARAMETER_FIELD,
        ESAMP_GRADIENT_FIELD,
        ESAMP_STEP_FIELD,
        ESAMP_EXP_AVG_FIELD,
        ESAMP_EXP_AVG_SQ_FIELD,
    )


def esamp_adamw_config(config: ESampRequestConfig) -> DensePageAdamWConfig:
    return DensePageAdamWConfig(
        config.learning_rate, config.betas, config.eps, config.weight_decay
    )


def selected_training_tensors(
    selection: RequestStateSelection,
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed_row_indices = torch.tensor(
        tuple(row.canonical_row_index for row in selection.rows),
        dtype=torch.long,
        device=source.device,
    )
    row_state_indices = torch.tensor(
        tuple(row.state_index for row in selection.rows),
        dtype=torch.long,
        device=source.device,
    )
    return (
        source.index_select(0, packed_row_indices),
        target.index_select(0, packed_row_indices),
        row_state_indices,
    )


def build_prediction(
    *,
    config: ESampRequestConfig,
    step: ESampRequestStepInput,
    selection: RequestStateSelection,
    training_source: torch.Tensor,
    training_target: torch.Tensor,
    state_vectors: torch.Tensor,
    row_state_indices: torch.Tensor,
    layout: LowRankGatedLayout,
    readiness: CudaStreamReadiness | None = None,
) -> ESampPredicted:
    local_row_by_packed = {
        row.canonical_row_index: position
        for position, row in enumerate(selection.rows)
    }
    affected_row_ids: list[int] = []
    prediction_positions: list[int] = []
    if config.intervention:
        for link in step.sampler_rows.links:
            if link.packed_row_index not in local_row_by_packed:
                continue
            local_position = local_row_by_packed[link.packed_row_index]
            selected_row = selection.rows[local_position]
            selected_ref = selection.refs[selected_row.state_index]
            if int(selected_ref.version_id) > 0:
                affected_row_ids.append(link.sampler_row_index)
                prediction_positions.append(local_position)
    if prediction_positions:
        all_predictions = low_rank_gated_forward(
            training_source, state_vectors, row_state_indices, layout
        )
        pred_hidden = all_predictions.index_select(
            0,
            torch.tensor(
                tuple(prediction_positions),
                dtype=torch.long,
                device=all_predictions.device,
            ),
        )
    else:
        pred_hidden = training_source.new_empty((0, layout.hidden_size))
    return ESampPredicted(
        PackedSamplerGuidanceBatch(
            step.sampler_rows,
            pred_hidden,
            tuple(affected_row_ids),
        ),
        training_source,
        training_target,
        state_vectors,
        row_state_indices,
        readiness,
    )


def compute_low_rank_losses_and_gradients(
    predicted: ESampPredicted,
    layout: LowRankGatedLayout,
    *,
    clone_outputs: bool,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    parameters = predicted.state_vectors.detach().requires_grad_(True)
    prediction = low_rank_gated_forward(
        predicted.training_source, parameters, predicted.row_state_indices, layout
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
    if clone_outputs:
        return (
            tuple(loss.detach().clone() for loss in losses),
            tuple(gradient[index].detach().clone() for index in range(state_count)),
        )
    return tuple(loss.detach() for loss in losses), tuple(
        gradient[index].detach() for index in range(state_count)
    )


def _predicted(prediction: ESampPrediction, message: str) -> ESampPredicted:
    match prediction:
        case ESampPredicted() as predicted:
            return predicted
        case _:
            raise RuntimeError(message)


def _optimized(optimization: ESampOptimization, message: str) -> ESampOptimized:
    match optimization:
        case ESampOptimized() as optimized:
            return optimized
        case _:
            raise RuntimeError(message)


class ESampCpuPredictTask:
    __slots__ = ("_config", "_layout", "_provider")

    def __init__(
        self,
        config: ESampRequestConfig,
        layout: LowRankGatedLayout,
        provider: DensePageStateProvider,
    ) -> None:
        self._config = config
        self._layout = layout
        self._provider = provider

    @property
    def implementation_id(self) -> ImplementationId:
        return ESAMP_REQUEST_PREDICT_IMPLEMENTATION

    def execute(self, context: ESampCpuStepContext) -> None:
        training_source, training_target, row_state_indices = selected_training_tensors(
            context.selection,
            context.source,
            context.target,
        )
        state_vectors = torch.stack(
            tuple(
                self._provider.snapshot(ref)[int(ESAMP_PARAMETER_FIELD)]
                for ref in context.selection.refs
            )
        )
        context.prediction = build_prediction(
            config=self._config,
            step=context.step,
            selection=context.selection,
            training_source=training_source,
            training_target=training_target,
            state_vectors=state_vectors,
            row_state_indices=row_state_indices,
            layout=self._layout,
        )


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
            training_source, training_target, row_state_indices = (
                selected_training_tensors(
                    context.selection,
                    context.source,
                    context.target,
                )
            )
            prediction = build_prediction(
                config=self._config,
                step=context.step,
                selection=context.selection,
                training_source=training_source,
                training_target=training_target,
                state_vectors=state_vectors,
                row_state_indices=row_state_indices,
                layout=self._layout,
                readiness=CudaStreamReadiness(context.stream),
            )
        context.prediction = prediction


class ESampCpuLossTask:
    __slots__ = ("_layout",)

    def __init__(self, layout: LowRankGatedLayout) -> None:
        self._layout = layout

    @property
    def implementation_id(self) -> ImplementationId:
        return ESAMP_REQUEST_LOSS_IMPLEMENTATION

    def execute(self, context: ESampCpuStepContext) -> None:
        losses, gradients = compute_low_rank_losses_and_gradients(
            _predicted(context.prediction, "ESamp loss requires the planned prediction"),
            self._layout,
            clone_outputs=True,
        )
        context.optimization = ESampOptimized(losses, gradients)


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
        predicted = _predicted(context.prediction, "CUDA ESamp loss requires prediction")
        with torch.cuda.stream(context.stream):
            losses, gradient_values = compute_low_rank_losses_and_gradients(
                predicted,
                self._layout,
                clone_outputs=False,
            )
        context.optimization = ESampOptimized(
            losses,
            self._provider.bind_gradients(
                gradient_values,
                context.stream,
            ),
        )


class ESampCpuTrainTask:
    __slots__ = ("_config", "_directory", "_provider")

    def __init__(
        self,
        config: ESampRequestConfig,
        directory: RequestStateDirectory,
        provider: DensePageStateProvider,
    ) -> None:
        self._config = config
        self._directory = directory
        self._provider = provider

    @property
    def implementation_id(self) -> ImplementationId:
        return ESAMP_REQUEST_TRAIN_IMPLEMENTATION

    def execute(self, context: ESampCpuStepContext) -> None:
        optimized = _optimized(
            context.optimization,
            "ESamp train requires the planned loss",
        )
        if type(optimized.gradients) is not tuple:
            raise RuntimeError("ESamp train requires CPU tuple gradients")
        logical = self._directory.prepare_transition(
            context.selection,
            context.transition_lease,
        )
        try:
            physical = self._provider.prepare_transition_batch(
                logical.before_refs,
                logical.after_refs,
            )
        except Exception:
            self._directory.abort_transition(logical)
            raise
        try:
            self._provider.compute_adamw_transition(
                physical,
                esamp_adamw_field_binding(),
                esamp_adamw_config(self._config),
                optimized.gradients,
            )
        except Exception:
            self._directory.abort_transition(logical)
            raise
        try:
            self._provider.publish_transition_batch(physical)
        except Exception:
            self._provider.abort_transition_batch(physical)
            self._directory.abort_transition(logical)
            raise
        self._directory.commit_transition(logical)
        context.publication = ESampPublished(logical.before_refs, logical.after_refs)


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
        optimized = _optimized(
            context.optimization,
            "CUDA ESamp train requires loss",
        )
        if type(optimized.gradients) is not CudaDensePageGradientBatch:
            raise RuntimeError("CUDA ESamp train requires CUDA-bound gradients")
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
                esamp_adamw_field_binding(),
                esamp_adamw_config(self._config),
                optimized.gradients,
                context.stream,
            )
        except Exception:
            self._directory.abort_transition(logical)
            raise
        self._directory.commit_transition(logical)
        context.publication = ESampPublished(
            logical.before_refs,
            logical.after_refs,
            publication,
        )
