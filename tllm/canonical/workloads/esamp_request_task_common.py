"""Shared ESamp request task contracts and math helpers."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Protocol

import torch

from tllm.canonical.cuda_stream import CudaStreamReadiness
from tllm.canonical.kernels.low_rank_gated import LowRankGatedLayout, low_rank_gated_forward
from tllm.canonical.ptts import CudaDensePageStateProvider, CudaDensePageVersionPin
from tllm.canonical.ptts.dense_page_cuda_types import CudaDensePageRelease
from tllm.canonical.ptts.dense_page import (
    DensePageAdamWConfig,
    DensePageAdamWFieldBinding,
    DensePageStateProvider,
    DensePageVersionPin,
)
from tllm.canonical.ptts.request_state import (
    RequestStateDirectory,
    RequestStateSelection,
    RequestStateTransitionLease,
    RequestStateTransitionBatch,
)
from tllm.canonical.ptts.publication import RequestStatePublicationReceipt
from tllm.canonical.workloads.esamp_request_types import (
    ESampRequestConfig,
    ESampRequestStepInput,
    ESAMP_EXP_AVG_FIELD,
    ESAMP_EXP_AVG_SQ_FIELD,
    ESAMP_GRADIENT_FIELD,
    ESAMP_PARAMETER_FIELD,
    ESAMP_STEP_FIELD,
)
from tllm.contracts.sampler import PackedSamplerGuidanceBatch
from tllm.contracts.semantic_graph import ImplementationId
from tllm.contracts.state_ref import StateRef


TensorGradients = tuple[torch.Tensor, ...]


class ESampStateKernel(Protocol):
    @property
    def stream(self) -> torch.cuda.Stream | None: ...

    def read(self, refs: tuple[StateRef, ...]) -> torch.Tensor: ...

    def release(self) -> CudaDensePageRelease | None: ...

    def submit(
        self, logical: RequestStateTransitionBatch, config: ESampRequestConfig,
        gradients: TensorGradients,
    ) -> RequestStatePublicationReceipt: ...


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
    gradients: TensorGradients


ESampOptimization = ESampOptimizationPending | ESampOptimized


@dataclass(frozen=True, slots=True)
class ESampPublicationPending:
    pass


ESampPublication = ESampPublicationPending | RequestStatePublicationReceipt


@dataclass(slots=True)
class ESampStepContext:
    step: ESampRequestStepInput
    source: torch.Tensor
    target: torch.Tensor
    selection: RequestStateSelection
    transition_lease: RequestStateTransitionLease
    prediction: ESampPrediction
    optimization: ESampOptimization
    publication: ESampPublication
    kernel: ESampStateKernel


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


@dataclass(frozen=True, slots=True)
class ESampCpuStateKernel:
    provider: DensePageStateProvider
    state_pin: DensePageVersionPin
    stream: None = None

    def read(self, refs: tuple[StateRef, ...]) -> torch.Tensor:
        return torch.stack(tuple(self.provider.snapshot(ref)[int(ESAMP_PARAMETER_FIELD)] for ref in refs))

    def release(self) -> None:
        self.provider.release_pin(self.state_pin)

    def submit(
        self, logical: RequestStateTransitionBatch, config: ESampRequestConfig,
        gradients: TensorGradients,
    ) -> RequestStatePublicationReceipt:
        physical = self.provider.prepare_transition_batch(logical.before_refs, logical.after_refs)
        self.provider.compute_adamw_transition(
            physical, esamp_adamw_field_binding(), esamp_adamw_config(config), gradients,
        )
        try:
            receipt = RequestStatePublicationReceipt(
                logical.operation_id, physical.transaction_id, logical.before_refs, logical.after_refs,
            )
            self.provider.publish_transition_batch(physical)
        except Exception:
            self.provider.abort_transition_batch(physical)
            raise
        return receipt


@dataclass(frozen=True, slots=True)
class ESampCudaStateKernel:
    provider: CudaDensePageStateProvider
    state_pin: CudaDensePageVersionPin
    stream: torch.cuda.Stream

    def read(self, refs: tuple[StateRef, ...]) -> torch.Tensor:
        return self.provider.pinned_field_rows(self.state_pin, ESAMP_PARAMETER_FIELD, self.stream)

    def release(self) -> CudaDensePageRelease:
        return self.provider.release_pin(self.state_pin, self.stream)

    def submit(
        self, logical: RequestStateTransitionBatch, config: ESampRequestConfig,
        gradients: TensorGradients,
    ) -> RequestStatePublicationReceipt:
        bound = self.provider.bind_gradients(gradients, self.stream)
        try:
            physical = self.provider.prepare_transition_batch(
                logical.before_refs, logical.after_refs, self.stream,
            )
            publication = self.provider.submit_adamw_transition(
                physical, esamp_adamw_field_binding(), esamp_adamw_config(config), bound, self.stream,
            )
        except Exception:
            self.provider.discard_gradients(bound)
            raise
        return RequestStatePublicationReceipt(
            logical.operation_id, physical.transaction_id, logical.before_refs, logical.after_refs, publication,
        )


class ESampPredictTask:
    __slots__ = ("_config", "_layout", "implementation_id")

    def __init__(
        self, config: ESampRequestConfig, layout: LowRankGatedLayout,
        implementation_id: ImplementationId,
    ) -> None:
        self._config = config
        self._layout = layout
        self.implementation_id = implementation_id

    def execute(self, context: ESampStepContext) -> None:
        state_vectors = context.kernel.read(context.selection.refs)
        stream = context.kernel.stream
        with nullcontext() if stream is None else torch.cuda.stream(stream):
            training_source, training_target, row_state_indices = selected_training_tensors(
                context.selection, context.source, context.target,
            )
            context.prediction = build_prediction(
                config=self._config, step=context.step, selection=context.selection,
                training_source=training_source, training_target=training_target,
                state_vectors=state_vectors, row_state_indices=row_state_indices,
                layout=self._layout,
                readiness=CudaStreamReadiness(stream) if stream is not None else None,
            )


class ESampLossTask:
    __slots__ = ("_layout", "implementation_id")

    def __init__(self, layout: LowRankGatedLayout, implementation_id: ImplementationId) -> None:
        self._layout = layout
        self.implementation_id = implementation_id

    def execute(self, context: ESampStepContext) -> None:
        match context.prediction:
            case ESampPredicted() as predicted:
                pass
            case ESampPredictionPending():
                raise RuntimeError("ESamp loss requires the planned prediction")
        stream = context.kernel.stream
        with nullcontext() if stream is None else torch.cuda.stream(stream):
            losses, gradients = compute_low_rank_losses_and_gradients(
                predicted, self._layout, clone_outputs=stream is None,
            )
        context.optimization = ESampOptimized(losses, gradients)


class ESampTrainTask:
    __slots__ = ("_config", "_directory", "implementation_id")

    def __init__(
        self, config: ESampRequestConfig, directory: RequestStateDirectory,
        implementation_id: ImplementationId,
    ) -> None:
        self._config = config
        self._directory = directory
        self.implementation_id = implementation_id

    def execute(self, context: ESampStepContext) -> None:
        match context.optimization:
            case ESampOptimized() as optimized:
                pass
            case ESampOptimizationPending():
                raise RuntimeError("ESamp train requires the planned loss")
        logical = self._directory.prepare_transition(context.selection, context.transition_lease)
        try:
            receipt = context.kernel.submit(logical, self._config, optimized.gradients)
        except Exception:
            self._directory.abort_transition(logical)
            raise
        self._directory.commit_transition(logical)
        context.publication = receipt
