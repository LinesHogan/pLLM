"""Preemption state policy shared by engine sessions and schedule diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pllm.contracts.backend_attachment import BackendAttachment
from pllm.contracts.host_batch import StateImport
from pllm.contracts.semantic_graph import HostServiceInputSource, SemanticGraph, StateEffectKind

if TYPE_CHECKING:
    from pllm.execution.model_sites import ResidualRewrite


def preemption_policy(attachment: BackendAttachment, sites: tuple[ResidualRewrite, ...]) -> Literal["keep", "reset"]:
    """Model-state effects replay from admission; sampler and hidden-only effects retain state.

    Adapters and activation patches whose state no task rewrites while requests run (it is frozen, or imported
    before each request's first forward) also retain it: vLLM's recompute reads the version the request pinned, so
    it rebuilds the same K/V. A reset also needs the plan's recompute_rejection to be empty before replay begins.
    """
    graph = attachment.plan.semantic_graph
    adaptive = any(task.custom_forward is not None and task.custom_forward.adaptive for task in graph.tasks)
    applied = {
        binding.resource for binding in (*attachment.spec.overlay_bindings, *attachment.spec.activation_bindings)
    }
    custom = any(not rewrite.patch for rewrite in sites)
    return "reset" if applied & _rewritten(graph) or custom or attachment.probe.custom_attention or adaptive else "keep"


def _rewritten(graph: SemanticGraph) -> set[str]:
    """States a task rewrites while requests run; a StateImport lands before its requests' first forward."""
    return {
        effect.resource
        for task in graph.tasks
        if not any(
            isinstance(port.source, HostServiceInputSource) and port.source.schema == StateImport.schema
            for port in task.inputs
        )
        for effect in task.effects
        if effect.kind not in (StateEffectKind.READ, StateEffectKind.PUBLISH)
    }
