"""CPU math/state task implementations selected by the canonical scheduler."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.canonical.kernels.low_rank_gated import (
    LowRankGatedLayout,
)
from tllm.canonical.ptts.dense_page import (
    DensePageStateProvider,
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
    ESAMP_REQUEST_LOSS_IMPLEMENTATION,
    ESAMP_REQUEST_PREDICT_IMPLEMENTATION,
    ESAMP_REQUEST_TRAIN_IMPLEMENTATION,
)
from tllm.contracts.semantic_graph import ImplementationId


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


class ESampCpuLossTask:
    __slots__ = ("_layout",)

    def __init__(self, layout: LowRankGatedLayout) -> None:
        self._layout = layout

    @property
    def implementation_id(self) -> ImplementationId:
        return ESAMP_REQUEST_LOSS_IMPLEMENTATION

    def execute(self, context: ESampCpuStepContext) -> None:
        match context.prediction:
            case ESampPredicted() as predicted:
                pass
            case _:
                raise RuntimeError("ESamp loss requires the planned prediction")
        losses, gradients = compute_low_rank_losses_and_gradients(
            predicted,
            self._layout,
            clone_outputs=True,
        )
        context.optimization = ESampOptimized(
            losses,
            gradients,
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
        match context.optimization:
            case ESampOptimized() as optimized:
                pass
            case _:
                raise RuntimeError("ESamp train requires the planned loss")
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
        context.publication = ESampPublished(
            logical.before_refs,
            logical.after_refs,
        )


__all__ = [
    "ESampCpuLossTask",
    "ESampCpuPredictTask",
    "ESampCpuStepContext",
    "ESampCpuTrainTask",
]
