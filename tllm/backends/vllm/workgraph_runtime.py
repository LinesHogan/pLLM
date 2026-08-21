"""Typed WorkGraph handoff for concrete vLLM drivers."""

from __future__ import annotations

from tllm.backends.protocol import BackendCapabilities
from tllm.runtime.graph_policy import GraphMode
from tllm.workgraph.anchors import BackendResolution


def cold_specialize_loaded_runtime(*, core: object, runner: object) -> None:
    """Publish model-layout facts while still inside the vLLM quarantine."""

    from tllm.runtime import residual_runtime

    if core is not residual_runtime:
        return
    from tllm.runtime.specialization import RuntimeSpecializationSpec

    model_config = runner.model_config  # type: ignore[attr-defined]
    spec = RuntimeSpecializationSpec(
        rows=max(1, int(runner.max_num_reqs)),  # type: ignore[attr-defined]
        hidden_size=int(model_config.get_hidden_size()),
        dtype=model_config.dtype,
        device=runner.device,  # type: ignore[attr-defined]
    )
    residual_runtime.cold_specialize_workgraph(spec)


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


__all__ = ["cold_specialize_loaded_runtime", "record_resolved_backend"]
