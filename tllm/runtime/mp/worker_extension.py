#!/usr/bin/env python3
"""Worker extension exposing residual tLLM control-plane methods.

Workers expose the runtime controls still consumed by the vLLM integration:
liveness, request mapping, stats, synchronization, runtime flags, and runner
provenance.
"""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Mapping, Protocol, cast

if TYPE_CHECKING:
    from tllm.backends.vllm.versioned_parameter_overlay import (
        VLLMParameterOverlayRunner,
    )


class _WorkerView(Protocol):
    model_runner: VLLMParameterOverlayRunner | None


WireValue = None | str | int | float | bool | list["WireValue"] | dict[str, "WireValue"]


def _runner_of(worker: object) -> VLLMParameterOverlayRunner | None:
    return cast(_WorkerView, worker).model_runner


def _to_plain(value: object, *, _depth: int = 0) -> WireValue:
    if _depth > 16:
        raise TypeError("stats payload nesting is too deep for the worker wire")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _to_plain(asdict(value), _depth=_depth + 1)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            str(key): _to_plain(item, _depth=_depth + 1)
            for key, item in mapping.items()
        }
    if isinstance(value, (list, tuple)):
        items = cast(list[object] | tuple[object, ...], value)
        return [_to_plain(item, _depth=_depth + 1) for item in items]
    raise TypeError(
        f"stats payload value of type {type(value).__name__} is not wire-safe; "
        "resolve CUDA events/tensors to floats before returning"
    )


def _wire_text(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    return value


def _wire_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{label} must be a non-negative integer")
    return value


class TllmWorkerExtension:
    """Methods are invoked by qualified string name via collective_rpc."""

    model_runner: object | None = None

    def _tllm_worker_id(self) -> str:
        return f"{os.getpid()}:{id(self):x}"

    def tllm_ping(self) -> dict[str, WireValue]:
        """Liveness/provenance probe: which process is answering."""
        from tllm.runtime import residual_runtime as core

        return {
            "pid": os.getpid(),
            "worker_id": self._tllm_worker_id(),
            "patch_installed": core.runtime_patch_installed(),
            "mapping_size": len(core.RUNTIME.reqid_to_promptidx),
        }

    def tllm_push_request_mapping(self, mappings: list[list[object]]) -> int:
        """Install request_id -> (prompt_idx, sample_idx) rows."""
        from tllm.runtime import residual_runtime as core
        from tllm.runtime.request_lifecycle import register_request_alias

        count = 0
        for entry in mappings:
            request_id = _wire_text(entry[0], "request_id")
            prompt_idx = _wire_nonnegative_int(entry[1], "prompt_idx")
            core.RUNTIME.reqid_to_promptidx[request_id] = prompt_idx
            if entry[2] is None:
                core.RUNTIME.reqid_to_sampleidx.pop(request_id, None)
            else:
                core.RUNTIME.reqid_to_sampleidx[request_id] = _wire_nonnegative_int(
                    entry[2], "sample_idx"
                )
            register_request_alias(core.RUNTIME, request_id)
            count += 1
        return count

    def tllm_pull_stats(
        self, category: str = "all", sync: bool = True
    ) -> dict[str, WireValue]:
        """Read and reset runtime stats, resolved to plain data child-side."""
        from tllm.runtime import residual_runtime as core

        supported = {
            "all",
            "esamp",
            "esamp_per_request",
            "distiller_timing",
            "path_hotspot",
            "placement",
            "workgraph",
        }
        if category not in supported:
            raise ValueError(
                f"unsupported stats category {category!r}; supported: {sorted(supported)}"
            )
        payload: dict[str, WireValue] = {
            "esamp": {"loss_avg": 0.0, "loss_count": 0, "trace_losses": []},
            "esamp_per_request": {},
            "distiller_timing": {},
            "path_hotspot": {},
            "placement": {},
            "workgraph": {},
        }
        if category in ("all", "esamp"):
            payload["esamp"] = _to_plain(core.read_and_reset_esamp_stats(sync=sync))
        if category in ("all", "esamp_per_request"):
            payload["esamp_per_request"] = _to_plain(
                core.read_and_reset_esamp_per_request_stats(sync=sync)
            )
        if category in ("all", "distiller_timing"):
            payload["distiller_timing"] = _to_plain(
                core.read_and_reset_distiller_timing_stats(sync=sync)
            )
        if category in ("all", "path_hotspot"):
            payload["path_hotspot"] = _to_plain(
                core.read_and_reset_path_hotspot_stats(sync=sync)
            )
        if category in ("all", "placement"):
            payload["placement"] = _to_plain(core.read_and_reset_placement_stats())
        if category in ("all", "workgraph"):
            payload["workgraph"] = _to_plain(core.read_workgraph_runtime_stats())
        return payload

    def tllm_synchronize(self) -> None:
        """Drain outstanding async residual work in this process."""
        from tllm.runtime import residual_runtime as core

        core.synchronize_esamp()

    def tllm_apply_enable_flags(self, flags: dict[str, bool]) -> None:
        """Apply runtime flags on this worker."""
        from tllm.runtime import residual_runtime as core

        supported = {"esamp_training_enabled"}
        unknown = set(flags) - supported
        if unknown:
            raise ValueError(f"unsupported tLLM enable flags: {sorted(unknown)}")
        if "esamp_training_enabled" in flags:
            core.set_esamp_training_enabled(bool(flags["esamp_training_enabled"]))

    def tllm_runner_name(self) -> str:
        """Report the model runner class actually serving this worker."""
        runner = _runner_of(self)
        return type(runner).__name__ if runner is not None else "unbound"
