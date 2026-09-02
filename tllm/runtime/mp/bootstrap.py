#!/usr/bin/env python3
"""Cross-process bootstrap spec for the tLLM runtime.

Under ``VLLM_ENABLE_V1_MULTIPROCESSING=1`` the vLLM EngineCore (and with it
the model runner) lives in a child process. The parent snapshots runtime
mechanism config into a JSON file, and the child rehydrates from it (see
``tllm.runtime.mp.plugin``).
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
class BootstrapSpec:
    runtime_config: dict[str, Any]
    enable_flags: dict[str, bool] = field(default_factory=dict[str, bool])
    schema_version: int = SCHEMA_VERSION

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "runtime_config": self.runtime_config,
            "enable_flags": self.enable_flags,
        }

    @staticmethod
    def from_json_dict(data: Mapping[str, Any]) -> "BootstrapSpec":
        version = int(data.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported bootstrap spec schema_version {version}; expected {SCHEMA_VERSION}")
        return BootstrapSpec(
            runtime_config=dict(data.get("runtime_config", {})),
            enable_flags={str(k): bool(v) for k, v in dict(data.get("enable_flags", {})).items()},
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
    enable_flags: dict[str, bool] = {}
    return BootstrapSpec(
        runtime_config=dataclass_to_jsonable(runtime.config),
        enable_flags=enable_flags,
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
    """Rehydrate runtime config from the spec (child side)."""
    from tllm.runtime import residual_runtime

    residual_runtime.configure_runtime(**spec.runtime_config)
    if "esamp_training_enabled" in spec.enable_flags:
        residual_runtime.set_esamp_training_enabled(spec.enable_flags["esamp_training_enabled"])


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
