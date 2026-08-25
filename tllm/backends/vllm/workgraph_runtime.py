"""Typed WorkGraph handoff for concrete vLLM drivers."""

from __future__ import annotations

from typing import cast

import torch

from tllm.backends.protocol import BackendCapabilities
from tllm.runtime.graph_policy import GraphMode
from tllm.runtime.static_resources import static_device_headroom_bytes
from tllm.workgraph.anchors import BackendResolution


def runner_static_headroom_bytes(runner: object, *, configured_budget: int) -> int:
    """Snapshot free device memory once after model load, never on a step path."""
    return static_device_headroom_bytes(
        torch.device(runner.device),  # type: ignore[attr-defined]
        configured_budget=configured_budget,
    )


def cold_compile_loaded_workloads(*, core: object, runner: object) -> None:
    """Compile once at first prepare, after engine memory initialization."""

    from tllm.runtime import residual_runtime

    if core is not residual_runtime:
        return

    from tllm.runtime.runtime import active_runtime

    runtime = active_runtime()
    if not runtime.has_compilable_workloads:
        return
    if runtime.workload_compilation is not None:
        return
    from tllm.backends.timeline import export_main_workload_timeline

    resolution = residual_runtime.RUNTIME.workgraph_backend_resolution
    if resolution is None:
        raise RuntimeError("workload compilation requires resolved backend capabilities")
    options = runtime.workload_compile_options
    headroom = runner_static_headroom_bytes(
        runner,
        configured_budget=int(options.memory_budget_bytes),
    )
    timeline = export_main_workload_timeline(
        resolution.capabilities,
        graph_mode=resolution.graph_mode.name.lower(),
        static_headroom_bytes=headroom,
        backend_version=resolution.backend_version,
    )
    compilation = runtime.compile_workloads(timeline)
    residual_runtime.publish_workload_compilation(
        compilation,
        bindings=runtime.workload_execution_bindings,
    )


def runner_tensor_parallel_size(runner: object) -> int:
    """Read the resolved TP size at the backend quarantine boundary."""

    parallel_config = getattr(runner, "parallel_config", None)
    if parallel_config is None:
        parallel_config = getattr(
            getattr(runner, "vllm_config", None), "parallel_config", None
        )
    value = getattr(parallel_config, "tensor_parallel_size", 1)
    tp_size = int(value)
    if tp_size < 1:
        raise RuntimeError(f"vLLM reported invalid tensor_parallel_size={tp_size}")
    return tp_size


def _runner_max_prefill_tokens(runner: object) -> int:
    """Read the exact vLLM prefill capacity from one supported runner field."""

    for name in ("max_num_tokens", "max_num_batched_tokens"):
        try:
            value = getattr(runner, name)
        except AttributeError:
            continue
        if type(value) is not int or value < 1:
            raise RuntimeError(
                f"vLLM runner {name} must be an exact positive integer for "
                "runtime specialization"
            )
        return cast(int, value)
    raise RuntimeError(
        "vLLM runner must expose max_num_tokens or max_num_batched_tokens "
        "for runtime specialization"
    )


def cold_specialize_loaded_runtime(*, core: object, runner: object) -> None:
    """Publish model-layout facts while still inside the vLLM quarantine."""

    from tllm.runtime import residual_runtime

    if core is not residual_runtime:
        return
    from tllm.runtime.specialization import RuntimeSpecializationSpec

    model_config = runner.model_config  # type: ignore[attr-defined]
    from tllm.backends.protocol import runner_binding_for

    binding = runner_binding_for(runner)
    model_layer_count: int | None = None
    if binding is not None:
        raw_layer_count = binding.driver.runner_model_layer_count(runner)
        if type(raw_layer_count) is not int or raw_layer_count < 1:
            raise RuntimeError("vLLM backend driver must report model layer count >= 1")
        model_layer_count = raw_layer_count
    spec = RuntimeSpecializationSpec(
        rows=max(1, int(runner.max_num_reqs)),  # type: ignore[attr-defined]
        hidden_size=int(model_config.get_hidden_size()),
        dtype=model_config.dtype,
        device=runner.device,  # type: ignore[attr-defined]
        model_layer_count=model_layer_count,
        max_prefill_tokens=_runner_max_prefill_tokens(runner),
    )
    from tllm.runtime.runtime import active_runtime

    active_runtime().cold_specialize_workloads(spec)
    residual_runtime.cold_specialize_workgraph(spec)


def record_resolved_backend(
    *,
    core: object,
    capabilities: BackendCapabilities,
    graph_mode: GraphMode,
    tp_size: int = 1,
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
                tp_size=tp_size,
            )
        )


__all__ = [
    "cold_compile_loaded_workloads",
    "cold_specialize_loaded_runtime",
    "record_resolved_backend",
    "runner_tensor_parallel_size",
    "runner_static_headroom_bytes",
]
