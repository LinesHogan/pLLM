#!/usr/bin/env python3
"""Configuration for the BaseConsumer ESamp wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from tllm.consumers.esamp.initializers.svd import SVDModelBankInitializerConfig
from tllm.ports.consumer import BaseConsumer
from tllm.runtime.config import (
    AdaptationStreamMode,
    ModelBankForwardBackendName,
    normalize_adaptation_stream_mode,
    normalize_model_bank_forward_backend,
)
from tllm.runtime.sampler_bridge.types import SamplerBackend, normalize_sampler_backend
from tllm.ports.base import RowSelection

__all__ = [
    "AdaptationStreamMode",
    "BaseESampConsumer",
    "ESampConsumerConfig",
    "ModelBankForwardBackendName",
    "normalize_adaptation_stream_mode",
    "normalize_model_bank_forward_backend",
]


@dataclass
class ESampConsumerConfig:
    consumer_id: str = "esamp"
    implementation: str = "legacy"
    graph_scratch_rows: int = 0
    source_layer_path: str = "model.model.layers[0].input_layernorm"
    target_layer_path: str = "model.model.layers[-1].input_layernorm"
    enable_esamp_training: bool = True
    enable_workgraph_execution: bool = True
    distiller_hidden_dim: int = 128
    distiller_lr: float = 1e-3
    per_request_models: bool = False
    per_request_model_bank: bool = False
    training_row_selection: RowSelection = RowSelection.ALL
    model_bank_slots: int = 0
    model_bank_flush_interval: int = 1
    model_bank_rank: int = 64
    model_bank_use_output_layernorm: bool = True
    model_bank_initializer: SVDModelBankInitializerConfig | None = None
    model_bank_train_cudagraph: bool = False
    model_bank_forward_backend: ModelBankForwardBackendName = "torch"
    adaptation_pipeline_slots: int = 4
    adaptation_stream_mode: AdaptationStreamMode = "dual"
    adaptation_stream_priority: int = 0
    trace_per_request_losses: bool = False
    trace_interval: int = 1
    trace_max_points: int = 0
    enable_distiller_intervention: bool = False
    distiller_beta: float = 0.0
    distiller_sampler_backend: SamplerBackend = "post_filter_exact"
    compact_capture_lane: bool = False

    def __post_init__(self) -> None:
        self.implementation = str(self.implementation).strip() or "legacy"
        if self.implementation not in ("legacy", "program", "native"):
            raise ValueError(
                "implementation must be 'legacy', 'program', or 'native', "
                f"got {self.implementation!r}"
            )
        if bool(self.per_request_model_bank) and not bool(self.per_request_models):
            raise ValueError("per_request_model_bank=True requires per_request_models=True")
        try:
            self.training_row_selection = RowSelection(str(self.training_row_selection))
        except ValueError as exc:
            raise ValueError(
                "training_row_selection must be 'none' or 'first_per_prompt'"
            ) from exc
        self.distiller_sampler_backend = normalize_sampler_backend(self.distiller_sampler_backend)
        self.model_bank_forward_backend = normalize_model_bank_forward_backend(self.model_bank_forward_backend)
        self.adaptation_stream_mode = normalize_adaptation_stream_mode(self.adaptation_stream_mode)
        self.adaptation_stream_priority = int(self.adaptation_stream_priority)


class BaseESampConsumer(BaseConsumer[ESampConsumerConfig]):
    """Shared base of the legacy and program ESamp consumers.

    Lives in the engine-free config module so the plugin registry's reverse
    type lookup stays precise for both implementations.
    """
