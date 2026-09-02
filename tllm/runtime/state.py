#!/usr/bin/env python3
"""Residual-runtime configuration and per-process state.

This module owns the shared mutable state object (``RUNTIME``) that hooks
read during a step. It deliberately contains no
hook installation, dispatch, or lifecycle logic — see
``tllm.runtime.lifecycle`` and ``tllm.runtime.residual_runtime`` for those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

import torch

from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime.config import AdaptationStreamMode
from tllm.runtime.resources import SamplerPrecomputeState
from tllm.runtime.request_lifecycle import RequestLifecycleLedger
from tllm.runtime.reward_outcome_runtime import RewardOutcomeBatch, RewardOutcomeStore
from tllm.runtime.retry_control_runtime import RetryControlBatch, RetryControlStore
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.ports.residual_write import ResidualWriteLease, WriteLeaseKey
from tllm.runtime.ports.tap_handle import TapHandle
from tllm.runtime.stats import PlacementStats
from tllm.runtime.update_barrier_runtime import UpdateBarrierBatch, UpdateBarrierStore

if TYPE_CHECKING:
    from tllm.schedule.contracts import ScheduleGraph
    from tllm.schedule.coordinator import WorkloadCompilation
    from tllm.runtime.workload_execution import (
        NativeWorkloadExecution,
        NativeWorkloadExecutionStore,
    )
    from tllm.runtime.torch_adapter import CudaLongHostStaging
    from tllm.runtime.scheduling.workgraph_executor import WorkGraphRuntimeStatsReader


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
    decode_identity_staging: "CudaLongHostStaging | None" = None
    decode_logit_row_selector_buf: torch.Tensor | None = None
    decode_logit_row_selector_staging: "CudaLongHostStaging | None" = None
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
    # Raw vLLM aliases are owned until a signed request-finished cleanup has
    # acknowledged the closed RequestIdentity batch.
    request_lifecycle: RequestLifecycleLedger = field(
        default_factory=RequestLifecycleLedger
    )
    event_step_id: int = 0
    decode_post_logits_launched_step_id: int = -1
    sampler_precompute: SamplerPrecomputeState = field(default_factory=SamplerPrecomputeState)
    path_hotspot_enabled: bool = False
    path_hotspot_cpu_ms: dict[str, float] = field(default_factory=dict)
    path_hotspot_counts: dict[str, int] = field(default_factory=dict)

    physical_executor: Optional["WorkGraphRuntimeStatsReader"] = None
    physical_prepared_step_id: int = -1
    # Immutable new-middle-end output activated by the native physical executor.
    workload_compilation: Optional["WorkloadCompilation"] = None
    workload_execution: Optional["NativeWorkloadExecution"] = None
    workload_execution_store: Optional["NativeWorkloadExecutionStore"] = None
    native_workloads_pending: bool = False
    native_schedule_graph: Optional["ScheduleGraph"] = None
    native_hidden_path_aliases: dict[str, str] = field(default_factory=dict)
    placement_stats: PlacementStats = field(default_factory=PlacementStats)
    reward_outcome_store: RewardOutcomeStore = field(default_factory=RewardOutcomeStore)
    retry_control_store: RetryControlStore = field(default_factory=RetryControlStore)
    update_barrier_store: UpdateBarrierStore = field(default_factory=UpdateBarrierStore)
    last_reward_outcomes: RewardOutcomeBatch | None = None
    last_retry_controls: RetryControlBatch | None = None
    last_update_barriers: UpdateBarrierBatch | None = None

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
        self.decode_post_logits_launched_step_id = -1
        self.sampler_precompute.reset_step()


RUNTIME = ResidualRuntimeState()
