"""What a workload requires of a backend, and what the backend reports about the model and device before the
workload is compiled against it."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch

from pllm.contracts.backend_attachment import ModelProbe
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.kv_layout import KVLayout
from pllm.contracts.parameter_overlay import Projection
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import HiddenInputSource, SchedulerTimeline
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts


@dataclass(frozen=True, slots=True)
class BackendRequirements:
    """What a workload needs the backend to provide before compiling against it: hidden-state taps, adapter
    targets, sampler stages, model-site rewrites and the like. Each input marker, effect and state declaration
    states its own part; `combine` merges the parts."""

    checkpoint_names: tuple[str, ...] = field(default=(), kw_only=True)
    exact_phases: tuple[HiddenPhase, ...] = field(default=(), kw_only=True)
    hidden_inputs: tuple[HiddenInputSource, ...] = ()
    input_token_history: int = -1
    # The layers pllm.layer_logits reads, as declared (negative from the end).
    layer_logits: tuple[int, ...] = ()
    parameter_targets: tuple[Projection, ...] = ()
    activation_layers: tuple[int, ...] = ()
    sampler_stages: tuple[SamplerStage, ...] = ()
    sampler_feedback: bool = False
    position_states: tuple[tuple[str, float], ...] = ()
    # Whether a @pllm.custom_forward reruns model layers, which vLLM's CUDA graphs cannot capture.
    custom_forward: bool = False
    # Whether a @pllm.custom_residual rewrites sublayer adds or value rows.
    custom_residual: bool = False
    stateful_residual: bool = False
    custom_attention: bool = False
    kv_layouts: tuple[KVLayout, ...] = ()
    # Whether a pllm.model.Adapter reads an InstanceState; CUDA-graph overlays bind request state only.
    instance_adapter: bool = False
    causal_training: bool = False
    feeds: bool = False
    # The smallest pllm.sample_index(limit=); requests must use a lower sample index.
    sample_limit: int | None = None

    @staticmethod
    def combine(parts: Iterable[BackendRequirements]) -> BackendRequirements:
        """Merge the parts: sequences concatenate in order (taps and phases once each), flags any, the token history
        is the longest and the sample limit the smallest."""
        parts = tuple(parts)
        return BackendRequirements(
            checkpoint_names=tuple(name for part in parts for name in part.checkpoint_names),
            exact_phases=tuple(dict.fromkeys(phase for part in parts for phase in part.exact_phases)),
            hidden_inputs=tuple(dict.fromkeys(source for part in parts for source in part.hidden_inputs)),
            input_token_history=max((part.input_token_history for part in parts), default=-1),
            layer_logits=tuple(layer for part in parts for layer in part.layer_logits),
            parameter_targets=tuple(target for part in parts for target in part.parameter_targets),
            activation_layers=tuple(layer for part in parts for layer in part.activation_layers),
            sampler_stages=tuple(stage for part in parts for stage in part.sampler_stages),
            sampler_feedback=any(part.sampler_feedback for part in parts),
            position_states=tuple(state for part in parts for state in part.position_states),
            custom_forward=any(part.custom_forward for part in parts),
            custom_residual=any(part.custom_residual for part in parts),
            stateful_residual=any(part.stateful_residual for part in parts),
            custom_attention=any(part.custom_attention for part in parts),
            kv_layouts=tuple(layout for part in parts for layout in part.kv_layouts),
            instance_adapter=any(part.instance_adapter for part in parts),
            causal_training=any(part.causal_training for part in parts),
            feeds=any(part.feeds for part in parts),
            sample_limit=min((part.sample_limit for part in parts if part.sample_limit is not None), default=None),
        )


@dataclass(frozen=True, slots=True)
class BackendCompileFacts:
    probe: ModelProbe
    timeline: SchedulerTimeline
    providers: tuple[PageBundleProviderFacts, ...]
    max_admitted_lineages: int
    state_page_budget_bytes: int
    initialization_seed: int
    vocab_size: int = 0
    layer_count: int = 0
    checkpoint: str = ""
    guidance_capacity: int | None = None
    guidance_rejection: str | None = None

    @property
    def device(self) -> torch.device:
        """The worker's device, or an explicitly selected ordinal for shape-only planning."""
        if self.probe.device_kind.value != "cuda":
            return torch.device("cpu")
        index = self.probe.device_index
        if index is None:
            index = torch.cuda.current_device() if torch.cuda.is_initialized() else 0
        return torch.device("cuda", index)
