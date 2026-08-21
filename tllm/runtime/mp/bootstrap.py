#!/usr/bin/env python3
"""Cross-process bootstrap spec for the tLLM runtime.

Under ``VLLM_ENABLE_V1_MULTIPROCESSING=1`` the vLLM EngineCore (and with it
the model runner) lives in a child process. The parent snapshots everything
needed to reconstruct the runtime — runtime mechanism config plus
(name, config) pairs for every registered consumer — into a JSON file, and
the child rehydrates from it (see ``tllm.runtime.mp.plugin``).

Specs are plain JSON-safe data: consumer/engine *instances* hold torch
parameters and CUDA graphs and can never cross the boundary, so the child
rebuilds them from configs via the consumer registry.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Mapping

TLLM_BOOTSTRAP_SPEC_ENV = "TLLM_BOOTSTRAP_SPEC_PATH"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ConsumerSpec:
    """JSON-safe description of one registered consumer."""

    name: str
    config: dict[str, Any]


@dataclass(frozen=True)
class BootstrapSpec:
    runtime_config: dict[str, Any]
    primary_consumer: ConsumerSpec | None = None
    consumers: tuple[ConsumerSpec, ...] = ()
    enable_flags: dict[str, bool] = field(default_factory=dict[str, bool])
    #: Type names of live consumers that no registered builder can describe.
    #: They survive fork (memory inheritance) but are LOST under spawn — the
    #: plugin warns loudly when it drops them in a spawn-built child.
    skipped_consumers: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "runtime_config": self.runtime_config,
            "primary_consumer": (
                None
                if self.primary_consumer is None
                else {"name": self.primary_consumer.name, "config": self.primary_consumer.config}
            ),
            "consumers": [{"name": c.name, "config": c.config} for c in self.consumers],
            "enable_flags": self.enable_flags,
            "skipped_consumers": list(self.skipped_consumers),
        }

    @staticmethod
    def from_json_dict(data: Mapping[str, Any]) -> "BootstrapSpec":
        version = int(data.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported bootstrap spec schema_version {version}; expected {SCHEMA_VERSION}")
        primary_raw = data.get("primary_consumer")
        primary = None
        if primary_raw is not None:
            primary = ConsumerSpec(name=str(primary_raw["name"]), config=dict(primary_raw["config"]))
        consumers = tuple(
            ConsumerSpec(name=str(entry["name"]), config=dict(entry["config"]))
            for entry in data.get("consumers", ())
        )
        return BootstrapSpec(
            runtime_config=dict(data.get("runtime_config", {})),
            primary_consumer=primary,
            consumers=consumers,
            enable_flags={str(k): bool(v) for k, v in dict(data.get("enable_flags", {})).items()},
            skipped_consumers=tuple(str(n) for n in data.get("skipped_consumers", ())),
            schema_version=version,
        )


def write_bootstrap_spec(spec: BootstrapSpec) -> str:
    """Persist the spec to a temp file and return its path."""
    fd, path = tempfile.mkstemp(prefix="tllm_bootstrap_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(spec.to_json_dict(), handle, sort_keys=True)
    return path


def read_bootstrap_spec(path: str) -> BootstrapSpec:
    with open(path, encoding="utf-8") as handle:
        return BootstrapSpec.from_json_dict(json.load(handle))


def capture_bootstrap_spec() -> BootstrapSpec:
    """Snapshot the process-local runtime into a spec.

    Runs in whichever process holds the configured runtime (the parent).
    """
    from tllm.common.dataclass_json import dataclass_to_jsonable
    from tllm.runtime import residual_runtime

    runtime = residual_runtime.RUNTIME
    from tllm.runtime.consumer_plugins import describe_consumer_or_none, spec_for_consumer

    primary = None
    skipped: list[str] = []
    if runtime.consumer is not None:
        primary_spec = describe_consumer_or_none(runtime.consumer)
        if primary_spec is None:
            skipped.append(type(runtime.consumer).__name__)
        else:
            primary = primary_spec
    consumers: list[ConsumerSpec] = []
    registry = runtime.consumer_registry
    if registry is not None:
        for consumer in registry.consumers():
            spec = describe_consumer_or_none(consumer)
            if spec is None:
                skipped.append(type(consumer).__name__)
            else:
                consumers.append(spec)
    enable_flags = {
        "test_time_lora_enabled": bool(runtime.test_time_lora_enabled),
        "fast_weight_enabled": bool(runtime.config.fast_weight_enabled),
    }
    if primary is not None and primary.name == "esamp":
        enable_flags["esamp_training_enabled"] = bool(primary.config.get("enable_esamp_training", True))
    return BootstrapSpec(
        runtime_config=dataclass_to_jsonable(runtime.config),
        primary_consumer=primary,
        consumers=tuple(consumers),
        enable_flags=enable_flags,
        skipped_consumers=tuple(skipped),
    )


def runtime_matches_spec(spec: BootstrapSpec) -> bool:
    """True when the process-local runtime already reflects the spec.

    Under fork, the child inherits the parent's fully configured runtime and
    reconciliation must be a no-op; under spawn the child starts empty and
    this returns False so the plugin rehydrates.
    """
    try:
        current = capture_bootstrap_spec()
    except Exception:
        return False
    return current.to_json_dict() == spec.to_json_dict()


def apply_bootstrap_spec(spec: BootstrapSpec) -> None:
    """Rehydrate runtime config + consumers from the spec (child side)."""
    from tllm.runtime.consumer_plugins import build_consumer_from_spec
    from tllm.runtime import residual_runtime

    residual_runtime.configure_runtime(**spec.runtime_config)
    primary = None if spec.primary_consumer is None else build_consumer_from_spec(spec.primary_consumer)
    residual_runtime.set_runtime_consumer(primary)
    residual_runtime.replace_dispatch_consumers([build_consumer_from_spec(s) for s in spec.consumers])
    if "esamp_training_enabled" in spec.enable_flags:
        residual_runtime.set_esamp_training_enabled(spec.enable_flags["esamp_training_enabled"])
    if "test_time_lora_enabled" in spec.enable_flags:
        residual_runtime.set_test_time_lora_enabled(spec.enable_flags["test_time_lora_enabled"])
    if "fast_weight_enabled" in spec.enable_flags:
        residual_runtime.set_fast_weight_enabled(spec.enable_flags["fast_weight_enabled"])


def export_bootstrap_spec_env() -> str:
    """Snapshot the current runtime, write the spec, and export its path.

    Called by ``lifecycle.make_llm`` before the LLM (and with it the
    EngineCore child) is constructed; the environment variable propagates to
    the child under both fork and spawn.
    """
    path = write_bootstrap_spec(capture_bootstrap_spec())
    os.environ[TLLM_BOOTSTRAP_SPEC_ENV] = path
    return path


def refresh_exported_bootstrap_spec() -> None:
    """Refresh the active parent-side spec after a runtime control change."""
    path = os.environ.get(TLLM_BOOTSTRAP_SPEC_ENV, "")
    if not path:
        return
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "0") == "1":
        from tllm.runtime.mp.control import active_control_plane

        if active_control_plane() is None:
            return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(capture_bootstrap_spec().to_json_dict(), handle, sort_keys=True)
