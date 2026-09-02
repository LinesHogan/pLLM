"""CUDA task implementations selected by the canonical Rust scheduler."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.canonical.cuda_stream import CudaStreamReadiness
from tllm.canonical.kernels.low_rank_gated import (
    LowRankGatedLayout,
)
from tllm.canonical.ptts import (
    CudaDensePageGradientBatch,
    CudaDensePageStateProvider,
    CudaDensePageVersionPin,
)
from tllm.canonical.ptts.request_state import (
    RequestStateDirectory,
    RequestStateSelection,
    RequestStateTransitionLease,
)
from tllm.canonical.workloads.esamp_request_task_common import (
    ESampOptimization,
    ESampOptimized,
    ESampPredicted,
    ESampPrediction,
    ESampPublication,
    ESampPublished,
    build_prediction,
    compute_low_rank_losses_and_gradients,
    esamp_adamw_config,
    esamp_adamw_field_binding,
    selected_training_tensors,
)
from tllm.canonical.workloads.esamp_request_types import (
    ESampRequestConfig,
    ESampRequestStepInput,
    ESAMP_PARAMETER_FIELD,
    ESAMP_REQUEST_CUDA_LOSS_IMPLEMENTATION,
    ESAMP_REQUEST_CUDA_PREDICT_IMPLEMENTATION,
    ESAMP_REQUEST_CUDA_TRAIN_IMPLEMENTATION,
)
from tllm.contracts.semantic_graph import ImplementationId


@dataclass(slots=True)
class ESampCudaStepContext:
    step: ESampRequestStepInput
    source: torch.Tensor
    target: torch.Tensor
    selection: RequestStateSelection
    transition_lease: RequestStateTransitionLease
    state_pin: CudaDensePageVersionPin
    stream: torch.cuda.Stream
    prediction: ESampPrediction
    optimization: ESampOptimization
    publication: ESampPublication


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
            readiness = CudaStreamReadiness(context.stream)
            prediction = build_prediction(
                config=self._config,
                step=context.step,
                selection=context.selection,
                training_source=training_source,
                training_target=training_target,
                state_vectors=state_vectors,
                row_state_indices=row_state_indices,
                layout=self._layout,
                readiness=readiness,
            )
        context.prediction = prediction


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
            case ESampPredicted() as predicted:
                pass
            case _:
                raise RuntimeError("CUDA ESamp loss requires prediction")
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
            case ESampOptimized() as optimized:
                pass
            case _:
                raise RuntimeError("CUDA ESamp train requires loss")
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


__all__ = [
    "ESampCudaLossTask",
    "ESampCudaPredictTask",
    "ESampCudaStepContext",
    "ESampCudaTrainTask",
]
