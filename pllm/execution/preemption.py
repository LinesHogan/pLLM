"""Preemption state policy shared by engine sessions and schedule diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pllm.contracts.backend_attachment import BackendAttachment

if TYPE_CHECKING:
    from pllm.execution.model_sites import ResidualRewrite


def preemption_policy(attachment: BackendAttachment, sites: tuple[ResidualRewrite, ...]) -> Literal["keep", "reset"]:
    """Model-state effects replay from admission; sampler and hidden-only effects retain state.

    A reset also needs the plan's recompute_rejection to be empty before replay begins.
    """
    adaptive = any(
        task.custom_forward is not None and task.custom_forward.adaptive
        for task in attachment.plan.semantic_graph.tasks
    )
    return (
        "reset"
        if attachment.spec.overlay_bindings
        or attachment.spec.activation_bindings
        or sites
        or attachment.probe.custom_attention
        or adaptive
        else "keep"
    )
