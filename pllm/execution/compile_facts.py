"""What a workload requires of a backend, and what the backend reports about the model and device before the
workload is compiled against it."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch

from pllm.contracts.backend_attachment import BackendAttachmentSpec, ModelProbe
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.kv_layout import KVLayout
from pllm.contracts.parameter_overlay import Projection
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import (
    HiddenInputSource,
    SchedulerTimeline,
    SemanticGraph,
    StateEffectKind,
    StateScope,
)
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts


def prefix_cache_reason(
    reads: Iterable[tuple[str, StateScope, bool]], *, custom_attention: bool, in_forward: bool, last_tokens: bool
) -> str | None:
    """Identify a model dependency missing from the prefix cache key."""
    if custom_attention:
        return "@pllm.custom_attention needs full causal attention metadata"
    if in_forward:
        return "@pllm.train_in_forward changes model state during the forward"
    if last_tokens:
        return "ActivationPatch(last_tokens > 0) depends on the prompt length"
    for name, scope, mutable in reads:
        if scope is StateScope.REQUEST or mutable:
            return f"the model reads {scope.value} state {name!r} whose value can differ between requests"
    return None


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
    custom_value_site: bool = False
    custom_attention_facts: bool = False
    prefix_cache_unsafe_reason: str | None = None
    trains_on_batch: bool = False
    imports: bool = False
    causal_training: bool = False
    feeds: bool = False
    # The smallest pllm.sample_index(limit=); requests must use a lower sample index.
    sample_limit: int | None = None

    @property
    def custom_attention_tp_unsafe(self) -> bool:
        """Cross-head facts and altered KV storage need coordination across TP ranks."""
        return self.custom_attention_facts or any(layout.keep != "all" for layout in self.kv_layouts)

    def check_prefix_cache(self) -> None:
        """Reject a token-keyed cache for a workload with a state- or position-dependent prefix."""
        if self.prefix_cache_unsafe_reason:
            raise ValueError(f"disable prefix caching (enable_prefix_caching=False): {self.prefix_cache_unsafe_reason}")

    @classmethod
    def from_model_reads(
        cls,
        graph: SemanticGraph,
        attachment: BackendAttachmentSpec,
        *,
        external_states: set[str],
        custom_attention: bool,
    ) -> BackendRequirements:
        """Classify cache reuse using the lowered model reads, task writes and activation windows."""
        written = {
            effect.resource
            for task in graph.tasks
            for effect in task.effects
            if effect.kind is not StateEffectKind.READ
        }
        reason = prefix_cache_reason(
            (
                (state.resource, state.scope, state.resource in written or state.resource in external_states)
                for state in graph.states
                if state.resource in graph.model_reads
            ),
            custom_attention=custom_attention,
            in_forward=attachment.in_forward is not None,
            last_tokens=any(patch.last_tokens > 0 for patch in attachment.activation_bindings),
        )
        return cls(prefix_cache_unsafe_reason=reason)

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
            custom_value_site=any(part.custom_value_site for part in parts),
            custom_attention_facts=any(part.custom_attention_facts for part in parts),
            prefix_cache_unsafe_reason=next(
                (part.prefix_cache_unsafe_reason for part in parts if part.prefix_cache_unsafe_reason), None
            ),
            trains_on_batch=any(part.trains_on_batch for part in parts),
            imports=any(part.imports for part in parts),
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
