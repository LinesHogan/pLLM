#!/usr/bin/env python3
"""Residual-runtime configuration and per-process state.

This module owns the shared mutable state object (``RUNTIME``) that hooks,
producers, and consumers read during a step. It deliberately contains no
hook installation, dispatch, or lifecycle logic — see
``tllm.runtime.lifecycle`` and ``tllm.runtime.residual_runtime`` for those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

import torch

from tllm.consumers.base import BaseConsumer
from tllm.ports.residual_stream import ResidualLocator
from tllm.producers.registry import ExecutionPlan
from tllm.runtime.config import AdaptationStreamMode
from tllm.runtime.consumer_registry import ConsumerRegistry
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.fast_weight.state import FastWeightState
from tllm.runtime.resources import SamplerPrecomputeState
from tllm.runtime.reward_outcome_runtime import RewardOutcomeBatch, RewardOutcomeStore
from tllm.runtime.retry_control_runtime import RetryControlBatch, RetryControlStore
from tllm.runtime.test_time.lora import TestTimeLoRAService
from tllm.runtime.token_batch_metadata import RuntimeTokenBatchMeta
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.ports.residual_write import ResidualWriteLease, WriteLeaseKey
from tllm.runtime.ports.tap_handle import TapHandle
from tllm.runtime.scheduling.deferred import DeferredWorkQueue
from tllm.runtime.fusion import FusionExecutor
from tllm.runtime.stats import PlacementStats
from tllm.runtime.update_barrier_runtime import UpdateBarrierBatch, UpdateBarrierStore

if TYPE_CHECKING:
    from tllm.runtime.producer_executor import ProducerExecutor
    from tllm.runtime.scheduling.plan_store import PlanSnapshotStore
    from tllm.runtime.scheduling.phase3_transfer import PlacementRuntimeBinding
    from tllm.runtime.scheduling.phase4_collective import (
        DistributedPlanSnapshotStore,
        DistributedRuntimeBinding,
    )
    from tllm.runtime.scheduling.phase5_adaptive import AdaptiveRuntimeBinding
    from tllm.runtime.scheduling.workgraph_executor import WorkGraphRuntimeStatsReader
    from tllm.workgraph.anchors import BackendResolution
    from tllm.workgraph.shadow import ShadowParityPlan
    from tllm.workgraph.frontend import CapturedFlow
    from tllm.workgraph.hierarchy import CaptureAttachment


RuntimeConsumer = BaseConsumer


@dataclass
class ResidualRuntimeConfig:
    """Runtime-owned mechanism configuration only.

    Consumer-algorithm settings (distiller dims/lr/beta, model bank policy,
    training toggles, tracing) live in each consumer's own config — never
    here. What remains is what the hook layer itself reads: tap topology,
    scratch capacity, adaptation stream policy and capture-lane toggles.
    """

    graph_scratch_rows: int = 0
    tap_layer_paths: list[str] = field(default_factory=list)
    source_layer_path: str = "model.model.layers[0].input_layernorm"
    target_layer_path: str = "model.model.layers[-1].input_layernorm"
    adaptation_pipeline_slots: int = 4
    adaptation_stream_mode: AdaptationStreamMode = "dual"
    adaptation_stream_priority: int = 0
    compact_capture_lane: bool = False
    fast_weight_enabled: bool = False
    fast_weight_layers: tuple[int, ...] = (0,)
    fast_weight_chunk_size: int = 4096
    fast_weight_lr: float = 1e-3
    fast_weight_mode: str = "exact"
    fast_weight_decode_update_enabled: bool = False
    fast_weight_key_mode: str = "static"
    fast_weight_static_request_id: str = "__tllm_fast_weight_static__"
    fast_weight_max_slots: int = 64
    fast_weight_max_bytes: int = 0
    fast_weight_fail_on_unsupported: bool = False


@dataclass
class ResidualRuntimeState:
    config: ResidualRuntimeConfig = field(default_factory=ResidualRuntimeConfig)

    decode_row_idx: Optional[torch.Tensor] = None
    decode_valid_mask: Optional[torch.Tensor] = None
    decode_count: int = 0
    decode_prompt_idxs: list[int] = field(default_factory=list)
    decode_sample_idxs: list[int] = field(default_factory=list)
    decode_request_ids: list[str] = field(default_factory=list)
    decode_prompt_idx_buf: torch.Tensor | None = None
    decode_sample_idx_buf: torch.Tensor | None = None
    decode_prompt_idx_tensor: torch.Tensor | None = None
    decode_sample_idx_tensor: torch.Tensor | None = None
    decode_compact_row_idx: Optional[torch.Tensor] = None
    decode_compact_count: int = 0
    decode_compact_row_ids: tuple[int, ...] = ()

    tap_layers: Dict[str, torch.nn.Module] = field(default_factory=dict)
    tap_scratch: Dict[str, torch.Tensor] = field(default_factory=dict)
    tap_decode_hidden: Dict[str, torch.Tensor] = field(default_factory=dict)
    tap_decode_hidden_compact: Dict[str, torch.Tensor] = field(default_factory=dict)
    capture_full_residual_rows: bool = True
    residual_raw_paths: Dict[ResidualLocator, str] = field(default_factory=dict)
    residual_bindings: Dict[str, ResidualPathBinding] = field(default_factory=dict)
    write_leases: Dict[WriteLeaseKey, ResidualWriteLease] = field(default_factory=dict)
    tap_handles: Dict[str, TapHandle] = field(default_factory=dict)

    source_resolved_path: str = ""
    target_resolved_path: str = ""
    launch_consumer_from_hooks: bool = True
    reqid_to_promptidx: Dict[str, int] = field(default_factory=dict)
    reqid_to_sampleidx: Dict[str, int] = field(default_factory=dict)
    event_step_id: int = 0
    decode_post_logits_launched_step_id: int = -1
    sampler_precompute: SamplerPrecomputeState = field(default_factory=SamplerPrecomputeState)
    path_hotspot_enabled: bool = False
    path_hotspot_cpu_ms: dict[str, float] = field(default_factory=dict)
    path_hotspot_counts: dict[str, int] = field(default_factory=dict)

    consumer: Optional[RuntimeConsumer] = None
    consumer_registry: Optional[ConsumerRegistry] = None
    dispatch_plan: Optional[DispatchPlan] = None
    execution_plan: Optional[ExecutionPlan] = None
    # Startup-only WorkGraph shadow metadata. Runtime dispatch does not read
    # this field; the existing plans above remain authoritative.
    workgraph_snapshot: Optional["ShadowParityPlan"] = None
    workgraph_captures: tuple["CapturedFlow", ...] = ()
    workgraph_attachments: tuple["CaptureAttachment", ...] = ()
    # The latest concrete backend context is captured at the backend boundary
    # and reused by cold shadow refreshes after later plan changes.
    workgraph_backend_resolution: Optional["BackendResolution"] = None
    workgraph_refresh_pending: bool = False
    # Physical descriptors are published only by the cold compiler path and
    # become visible when the next step acquires its prepare-time snapshot.
    physical_plan_store: Optional["PlanSnapshotStore"] = None
    placement_runtime_binding: Optional["PlacementRuntimeBinding"] = None
    adaptive_runtime_binding: Optional["AdaptiveRuntimeBinding"] = None
    distributed_runtime_binding: Optional["DistributedRuntimeBinding"] = None
    distributed_plan_store: Optional["DistributedPlanSnapshotStore"] = None
    distributed_prepared_step_id: int = -1
    physical_executor: Optional["WorkGraphRuntimeStatsReader"] = None
    physical_prepared_step_id: int = -1
    producer_executor: Optional["ProducerExecutor"] = None
    deferred_work_queue: DeferredWorkQueue = field(default_factory=DeferredWorkQueue)
    fusion_executor: FusionExecutor = field(default_factory=FusionExecutor)
    placement_stats: PlacementStats = field(default_factory=PlacementStats)
    reward_outcome_store: RewardOutcomeStore = field(default_factory=RewardOutcomeStore)
    retry_control_store: RetryControlStore = field(default_factory=RetryControlStore)
    update_barrier_store: UpdateBarrierStore = field(default_factory=UpdateBarrierStore)
    last_reward_outcomes: RewardOutcomeBatch | None = None
    last_retry_controls: RetryControlBatch | None = None
    last_update_barriers: UpdateBarrierBatch | None = None
    test_time_lora_enabled: bool = False
    test_time_lora_service: Optional[TestTimeLoRAService] = None
    fast_weight_state: Optional[FastWeightState] = None
    fast_weight_handles: dict[str, object] = field(default_factory=dict)
    fast_weight_hook_spec: Optional[tuple[object, ...]] = None
    fast_weight_phase: str = "prefill"
    fast_weight_current_token_batch_meta: Optional[RuntimeTokenBatchMeta] = None
    fast_weight_installed_count: int = 0
    fast_weight_skipped_count: int = 0

    def reset_step(self) -> None:
        self.decode_count = 0
        self.decode_prompt_idxs = []
        self.decode_sample_idxs = []
        self.decode_request_ids = []
        self.decode_prompt_idx_buf = self.decode_prompt_idx_buf
        self.decode_sample_idx_buf = self.decode_sample_idx_buf
        self.decode_prompt_idx_tensor = None
        self.decode_sample_idx_tensor = None
        self.decode_compact_count = 0
        self.decode_compact_row_ids = ()
        self.fast_weight_current_token_batch_meta = None
        self.decode_post_logits_launched_step_id = -1
        self.sampler_precompute.reset_step()


RUNTIME = ResidualRuntimeState()
