"""Strict public values for the clean-room ESamp request slice."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import torch

from tllm.canonical.ptts.page_bundle_planning import PageBundleProviderFacts
from tllm.contracts.hidden_locator import HiddenLayerKind, HiddenSite
from tllm.contracts.packed_hidden import (
    PackedHiddenResourceBinding,
)
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
from tllm.contracts.state_representation import StateDeviceKind
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
ESAMP_REQUEST_CUDA_LOSS_IMPLEMENTATION = ImplementationId(
    "canonical.esamp_request:cuda_loss@1"
)
ESAMP_REQUEST_CUDA_PREDICT_IMPLEMENTATION = ImplementationId(
    "canonical.esamp_request:cuda_predict@1"
)
ESAMP_REQUEST_CUDA_TRAIN_IMPLEMENTATION = ImplementationId(
    "canonical.esamp_request:cuda_train@1"
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

    def execution_parameters_digest(self) -> str:
        """Bind every mathematical parameter into the signed execution identity."""

        fields = (
            "canonical.esamp_request.parameters@1",
            f"rank={self.rank}",
            f"source_layer={self.source_layer}",
            f"target_layer={self.target_layer}",
            f"output_layernorm={int(self.output_layernorm)}",
            f"intervention={int(self.intervention)}",
            f"intervention_beta={self.intervention_beta.hex()}",
            f"learning_rate={self.learning_rate.hex()}",
            f"beta_one={self.betas[0].hex()}",
            f"beta_two={self.betas[1].hex()}",
            f"eps={self.eps.hex()}",
            f"weight_decay={self.weight_decay.hex()}",
            "key_fields=" + ",".join(field.value for field in self.key_fields),
        )
        return sha256("\x00".join(fields).encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class ESampRequestSpecialization:
    hidden_size: int
    tensor_dtype: TensorDType
    initialization_seed: int
    hidden_bindings: "ESampRequestPackedHiddenBindings"
    device_kind: StateDeviceKind
    device_ordinal: int

    def __post_init__(self) -> None:
        if (
            type(self.hidden_size) is not int
            or type(self.tensor_dtype) is not TensorDType
            or type(self.initialization_seed) is not int
            or type(self.hidden_bindings) is not ESampRequestPackedHiddenBindings
            or type(self.device_kind) is not StateDeviceKind
            or type(self.device_ordinal) is not int
            or self.hidden_size < 1
            or self.tensor_dtype is not TensorDType.FLOAT32
            or self.initialization_seed < 0
            or self.device_ordinal < 0
            or (
                self.device_kind is StateDeviceKind.CPU
                and self.device_ordinal != 0
            )
        ):
            raise ValueError(
                "CPU ESamp specialization requires positive float32 dimensions"
            )

    def execution_parameters_digest(self, config: ESampRequestConfig) -> str:
        """Bind specialization facts and mathematical config into one identity."""

        if type(config) is not ESampRequestConfig:
            raise TypeError("ESamp execution identity requires exact config")
        source = self.hidden_bindings.source
        target = self.hidden_bindings.target
        fields = (
            "canonical.esamp_request.execution@1",
            config.execution_parameters_digest(),
            f"hidden_size={self.hidden_size}",
            f"tensor_dtype={self.tensor_dtype.value}",
            f"initialization_seed={self.initialization_seed}",
            f"device_kind={self.device_kind.value}",
            f"device_ordinal={self.device_ordinal}",
            f"source_layer={source.layer}",
            f"source_layer_kind={source.layer_kind.value}",
            f"source_site={source.site.value}",
            f"source_module_path={source.module_path}",
            f"target_layer={target.layer}",
            f"target_layer_kind={target.layer_kind.value}",
            f"target_site={target.site.value}",
            f"target_module_path={target.module_path}",
        )
        return sha256("\x00".join(fields).encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class ESampRequestPackedHiddenBindings:
    source: PackedHiddenResourceBinding
    target: PackedHiddenResourceBinding

    def __post_init__(self) -> None:
        if (
            type(self.source) is not PackedHiddenResourceBinding
            or type(self.target) is not PackedHiddenResourceBinding
            or self.source.layer_kind is not HiddenLayerKind.INDEX
            or self.target.layer_kind is not HiddenLayerKind.INDEX
            or self.source.site is not HiddenSite.BLOCK_OUTPUT
            or self.target.site is not HiddenSite.BLOCK_OUTPUT
            or self.source.locator_key == self.target.locator_key
        ):
            raise ValueError(
                "ESamp packed hidden bindings require distinct indexed block outputs"
            )

    @property
    def resources(self) -> tuple[PackedHiddenResourceBinding, ...]:
        return tuple(
            sorted(
                (self.source, self.target),
                key=lambda binding: binding.locator_key,
            )
        )


@dataclass(frozen=True, slots=True)
class ESampRequestCompileOptions:
    timeline: SchedulerTimeline
    providers: tuple[PageBundleProviderFacts, ...]
    max_admitted_lineages: int
    state_memory_budget_bytes: int
    plan_epoch: int = 1

    def __post_init__(self) -> None:
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


__all__ = [
    "ESampRequestCompileOptions",
    "ESampRequestConfig",
    "ESampRequestPackedHiddenBindings",
    "ESampRequestSpecialization",
    "ESampRequestStepResult",
]
