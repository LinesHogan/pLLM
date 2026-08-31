"""Strict public values for the clean-room ESamp request slice."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.canonical.ptts.dense_page_planning import DensePageCpuProviderFacts
from tllm.contracts.packed_rows import PackedPhaseRowBatch, PackedSamplerRowBatch
from tllm.contracts.request_identity import (
    RequestKeyField,
    canonical_request_key_fields,
)
from tllm.contracts.sampler import PackedSamplerGuidanceBatch
from tllm.contracts.semantic_graph import (
    ImplementationId,
    SchedulerTimeline,
)
from tllm.contracts.state_ref import StateRef
from tllm.contracts.state_representation import StatePhysicalPlan
from tllm.contracts.tensor_layout import LayoutFieldId, TensorDType


ESAMP_REQUEST_LOSS_IMPLEMENTATION = ImplementationId(
    "canonical.esamp_request:cpu_loss@1"
)
ESAMP_REQUEST_PREDICT_IMPLEMENTATION = ImplementationId(
    "canonical.esamp_request:cpu_predict@1"
)
ESAMP_REQUEST_TRAIN_IMPLEMENTATION = ImplementationId(
    "canonical.esamp_request:cpu_train@1"
)

ESAMP_PARAMETER_FIELD = LayoutFieldId(0)
ESAMP_GRADIENT_FIELD = LayoutFieldId(1)
ESAMP_STEP_FIELD = LayoutFieldId(2)
ESAMP_EXP_AVG_FIELD = LayoutFieldId(3)
ESAMP_EXP_AVG_SQ_FIELD = LayoutFieldId(4)


@dataclass(frozen=True, slots=True)
class ESampRequestConfig:
    rank: int = 128
    source_layer: int = 0
    target_layer: int = -1
    output_layernorm: bool = True
    intervention: bool = False
    intervention_beta: float = 0.0
    learning_rate: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    key_fields: tuple[RequestKeyField, ...] = (RequestKeyField.REQUEST_ID,)

    def __post_init__(self) -> None:
        if type(self.betas) is not tuple or len(self.betas) != 2:
            raise ValueError("ESamp AdamW betas must be an exact pair")
        beta_one, beta_two = self.betas
        canonical_request_key_fields(self.key_fields)
        if (
            type(self.rank) is not int
            or self.rank < 1
            or type(self.source_layer) is not int
            or type(self.target_layer) is not int
            or type(self.output_layernorm) is not bool
            or type(self.intervention) is not bool
            or type(self.intervention_beta) is not float
            or type(self.learning_rate) is not float
            or type(beta_one) is not float
            or type(beta_two) is not float
            or type(self.eps) is not float
            or type(self.weight_decay) is not float
            or self.intervention_beta < 0.0
            or self.learning_rate <= 0.0
            or not 0.0 <= beta_one < 1.0
            or not 0.0 <= beta_two < 1.0
            or self.eps <= 0.0
            or self.weight_decay < 0.0
        ):
            raise ValueError("ESamp request configuration is invalid")


@dataclass(frozen=True, slots=True)
class ESampRequestSpecialization:
    hidden_size: int
    tensor_dtype: TensorDType
    initialization_seed: int

    def __post_init__(self) -> None:
        if (
            type(self.hidden_size) is not int
            or type(self.tensor_dtype) is not TensorDType
            or type(self.initialization_seed) is not int
            or self.hidden_size < 1
            or self.tensor_dtype is not TensorDType.FLOAT32
            or self.initialization_seed < 0
        ):
            raise ValueError(
                "CPU ESamp specialization requires positive float32 dimensions"
            )


@dataclass(frozen=True, slots=True)
class ESampRequestCompileOptions:
    timeline: SchedulerTimeline
    providers: tuple[DensePageCpuProviderFacts, ...]
    max_admitted_lineages: int
    state_memory_budget_bytes: int
    plan_epoch: int = 1

    def __post_init__(self) -> None:
        if type(self.providers) is not tuple or any(
            type(provider) is not DensePageCpuProviderFacts
            for provider in self.providers
        ):
            raise ValueError("ESamp providers must be exact CPU provider facts")
        provider_ids = tuple(provider.provider_id for provider in self.providers)
        if (
            type(self.timeline) is not SchedulerTimeline
            or type(self.max_admitted_lineages) is not int
            or type(self.state_memory_budget_bytes) is not int
            or type(self.plan_epoch) is not int
            or not self.providers
            or tuple(sorted(provider_ids)) != provider_ids
            or len(set(provider_ids)) != len(provider_ids)
            or self.max_admitted_lineages < 1
            or self.state_memory_budget_bytes < 0
            or self.plan_epoch < 1
        ):
            raise ValueError("ESamp request compile options are not canonical")


@dataclass(frozen=True, slots=True)
class ESampRequestStepInput:
    rows: PackedPhaseRowBatch
    sampler_rows: PackedSamplerRowBatch
    source: torch.Tensor
    target: torch.Tensor

    def __post_init__(self) -> None:
        if (
            self.sampler_rows.row_batch != self.rows
            or self.source.ndim != 2
            or self.target.shape != self.source.shape
            or int(self.source.shape[0]) != self.rows.total_rows
            or self.target.dtype is not self.source.dtype
            or self.target.device != self.source.device
            or self.source.device.type != "cpu"
            or torch.is_inference(self.source)
            or torch.is_inference(self.target)
        ):
            raise ValueError(
                "ESamp request step requires aligned ordinary CPU packed tensors"
            )


@dataclass(frozen=True, slots=True)
class ESampRequestStepResult:
    step_id: int
    guidance: PackedSamplerGuidanceBatch
    sampler_beta: float
    losses: tuple[torch.Tensor, ...]
    before_refs: tuple[StateRef, ...]
    after_refs: tuple[StateRef, ...]

    def __post_init__(self) -> None:
        if (
            self.step_id < 0
            or self.sampler_beta < 0.0
            or not self.losses
            or len(self.losses) != len(self.before_refs)
            or len(self.before_refs) != len(self.after_refs)
        ):
            raise ValueError("ESamp request result is not state-aligned")


@dataclass(frozen=True, slots=True)
class ESampRequestDiagnostics:
    active_lineages: int
    loss_count: int
    update_count: int
    completed_sampler_pins: int
    state_plan: StatePhysicalPlan


__all__ = [
    "ESampRequestCompileOptions",
    "ESampRequestConfig",
    "ESampRequestDiagnostics",
    "ESampRequestSpecialization",
    "ESampRequestStepInput",
    "ESampRequestStepResult",
]
