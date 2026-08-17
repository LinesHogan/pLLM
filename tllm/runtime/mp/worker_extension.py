#!/usr/bin/env python3
"""Worker extension exposing tLLM control-plane methods to collective_rpc.

vLLM mixes this class into every worker (``worker_extension_cls`` in
ParallelConfig, resolved by qualified name inside the worker process, so it
is spawn-safe). All methods have plain string names and take/return only
msgpack-safe data (str/int/float/bool/list/dict/None): string method names
cross the client→EngineCore boundary WITHOUT
``VLLM_ALLOW_INSECURE_SERIALIZATION``; callables would not.

Every method is a thin delegate to the process-local runtime
(``tllm.runtime.residual_runtime``): in the parent under mp=0, in the
EngineCore child under mp=1. No torch objects cross the wire; stats that
involve CUDA events are resolved to floats child-side before returning.
"""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol, cast


class _WorkerView(Protocol):
    """The one vLLM Worker attribute this extension touches."""

    model_runner: Any  # set in v1 Worker.__init__; None only mid-construction


def _runner_of(worker: object) -> Any:
    return cast(_WorkerView, worker).model_runner


def _to_plain(value: Any, *, _depth: int = 0) -> Any:
    """Convert stats payloads to msgpack-safe data; reject anything else."""
    if _depth > 8:
        raise TypeError(f"stats payload too deep: {type(value).__name__}")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _to_plain(asdict(value), _depth=_depth + 1)
    if isinstance(value, dict):
        return {str(k): _to_plain(v, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v, _depth=_depth + 1) for v in value]
    raise TypeError(
        f"stats payload value of type {type(value).__name__} is not wire-safe; "
        "resolve CUDA events/tensors to floats before returning"
    )


class TllmWorkerExtension:
    """Methods are invoked by qualified string name via collective_rpc."""

    def tllm_ping(self) -> dict[str, Any]:
        """Liveness/provenance probe: which process is answering."""
        from tllm.runtime import residual_runtime as core

        return {
            "pid": os.getpid(),
            "patch_installed": bool(core._PATCH_INSTALLED),
            "consumer": None if core.RUNTIME.consumer is None else str(core.RUNTIME.consumer.consumer_id),
            "mapping_size": len(core.RUNTIME.reqid_to_promptidx),
        }

    def tllm_push_request_mapping(self, mappings: list[list[Any]]) -> int:
        """Install request_id -> (prompt_idx, sample_idx) rows.

        Each entry is ``[request_id, prompt_idx, sample_idx]``. Called by the
        parent BEFORE the corresponding add_request so the FIFO input socket
        guarantees the mapping lands before the request is schedulable.
        """
        from tllm.runtime import residual_runtime as core

        count = 0
        for entry in mappings:
            request_id, prompt_idx, sample_idx = str(entry[0]), int(entry[1]), int(entry[2])
            core.RUNTIME.reqid_to_promptidx[request_id] = prompt_idx
            core.RUNTIME.reqid_to_sampleidx[request_id] = sample_idx
            count += 1
        return count

    def tllm_pull_stats(self) -> dict[str, Any]:
        """Read and reset runtime stats, resolved to plain data child-side."""
        from tllm.runtime import residual_runtime as core

        return {
            "esamp": _to_plain(core.read_and_reset_esamp_stats(sync=True)),
            "esamp_per_request": _to_plain(core.read_and_reset_esamp_per_request_stats(sync=True)),
            "distiller_timing": _to_plain(core.read_and_reset_distiller_timing_stats(sync=True)),
            "path_hotspot": _to_plain(core.read_and_reset_path_hotspot_stats(sync=True)),
        }

    def tllm_synchronize(self) -> None:
        """Drain outstanding async consumer work in this process."""
        from tllm.runtime import residual_runtime as core

        core.synchronize_esamp()

    def tllm_apply_enable_flags(self, flags: dict[str, bool]) -> None:
        """Apply supported enable flags to this process's runtime."""
        from tllm.runtime import residual_runtime as core

        supported = ("esamp_training_enabled", "test_time_lora_enabled")
        unknown = sorted(set(flags) - set(supported))
        if unknown:
            raise ValueError(f"unsupported enable flag(s) {unknown}; supported: {list(supported)}")
        if "esamp_training_enabled" in flags:
            core.set_esamp_training_enabled(bool(flags["esamp_training_enabled"]))
        if "test_time_lora_enabled" in flags:
            core.set_test_time_lora_enabled(bool(flags["test_time_lora_enabled"]))

    def tllm_register_consumer(self, name: str, config: dict[str, Any], primary: bool = False) -> None:
        """Rebuild a consumer from its spec and activate it in this process."""
        from tllm.consumers import registry as consumer_registry
        from tllm.runtime.mp.bootstrap import ConsumerSpec

        consumer = consumer_registry.build_consumer_from_spec(ConsumerSpec(name=str(name), config=dict(config)))
        consumer_registry.activate_consumer(consumer, primary=bool(primary))

    def tllm_bind_runner_runtime(self) -> None:
        """Record this worker's runner in the runner->runtime routing table.

        String-method replacement for the old closure RPC (closures need
        VLLM_ALLOW_INSECURE_SERIALIZATION and cannot cross to the child).
        """
        from tllm.runtime.runtime import active_runtime, bind_runner_runtime

        runner = _runner_of(self)
        if runner is not None:
            bind_runner_runtime(runner, active_runtime())

    def tllm_runner_name(self) -> str:
        """Report the model runner class actually serving this worker."""
        runner = _runner_of(self)
        return "unknown" if runner is None else f"{type(runner).__module__}.{type(runner).__name__}"
