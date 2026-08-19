"""Typed Phase 1a WorkGraph handoff for concrete vLLM drivers."""

from __future__ import annotations

from tllm.backends.protocol import BackendCapabilities
from tllm.runtime.graph_policy import GraphMode
from tllm.workgraph.anchors import BackendResolution


def record_resolved_backend(
    *,
    core: object,
    capabilities: BackendCapabilities,
    graph_mode: GraphMode,
) -> None:
    """Hand off pure backend facts when the real residual runtime is active.

    Driver tests often supply isolated fake cores. Those have no global
    WorkGraph state and intentionally keep their existing behavior. The live
    runtime is identified by its module object, not by backend-id strings or
    structural attribute probes.
    """

    from tllm.runtime import residual_runtime

    if core is residual_runtime:
        residual_runtime.record_workgraph_backend_resolution(
            resolution=BackendResolution(
                capabilities=capabilities,
                graph_mode=graph_mode,
            )
        )


__all__ = ["record_resolved_backend"]
