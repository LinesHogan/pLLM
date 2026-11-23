"""Preemption state policy shared by engine sessions and schedule diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pllm.contracts.backend_attachment import BackendAttachment, DecodeRefresh
from pllm.contracts.host_batch import StateImport
from pllm.contracts.request_identity import RequestKeyField
from pllm.contracts.semantic_graph import HostServiceInputSource, SemanticGraph, StateEffectKind, StateScope

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
        *(str(binding.resource) for binding in attachment.spec.overlay_bindings),
        *(binding.route.resource for binding in attachment.spec.overlay_bindings if binding.route is not None),
        *(str(binding.resource) for binding in attachment.spec.activation_bindings),
        *(
            str(binding.route_resource)
            for binding in attachment.spec.activation_bindings
            if binding.route_resource is not None
        ),
    }
    step_refreshed = {
        *(
            res
            for binding in attachment.spec.overlay_bindings
            if binding.route is not None and binding.route.refresh == "step"
            for res in (str(binding.resource), binding.route.resource)
        ),
        *(
            str(binding.resource)
            for binding in attachment.spec.activation_bindings
            if binding.decode_refresh is DecodeRefresh.STEP
        ),
        *(
            str(binding.route_resource)
            for binding in attachment.spec.activation_bindings
            if binding.route_resource is not None and binding.route_refresh is DecodeRefresh.STEP
        ),
    }
    custom = any(not rewrite.patch for rewrite in sites)
    if (
        applied & _rewritten(graph)
        or applied & _shared_imported_request(graph)
        or step_refreshed & _live_imported_instance(graph)
        or custom
        or attachment.probe.custom_attention
        or adaptive
    ):
        return "reset"
    return "keep"


def _live_imported_instance(graph: SemanticGraph) -> set[str]:
    """Imported instance states that `llm.import_state` can mutate between steps while requests remain in flight."""
    state = graph.host_state(StateImport.schema)
    if state is not None and state.scope is StateScope.INSTANCE:
        return {str(state.resource)}
    return set()


def _shared_imported_request(graph: SemanticGraph) -> set[str]:
    """Imported request states whose keys omit `request_id`, so concurrent online admissions can overwrite a lineage."""
    state = graph.host_state(StateImport.schema)
    if state is not None and state.scope is StateScope.REQUEST and RequestKeyField.REQUEST_ID not in state.key_fields:
        return {str(state.resource)}
    return set()


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
