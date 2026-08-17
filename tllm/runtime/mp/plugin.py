#!/usr/bin/env python3
"""vLLM general plugin: child-process bootstrap for multiprocess mode.

Registered as a ``vllm.general_plugins`` entry point. vLLM executes general
plugins exactly once per process — including the EngineCore child and every
worker — under both fork and spawn, making this the sanctioned seam for
re-installing tLLM's runner patches and rehydrating the runtime where the
model actually runs.

The plugin is a no-arg callable and must be safe in ANY vLLM process: with
no bootstrap spec in the environment it does nothing.
"""

from __future__ import annotations

import json
import os

from tllm.runtime.mp.bootstrap import (
    TLLM_BOOTSTRAP_SPEC_ENV,
    apply_bootstrap_spec,
    read_bootstrap_spec,
    runtime_matches_spec,
)


def _write_marker(spec_path: str, *, matched: bool, applied: bool) -> None:
    """Drop a per-process marker next to the spec for smoke verification.

    A2's ``tllm_ping`` control-plane method replaces this as the supported
    liveness probe; the marker exists so A1 can prove a non-parent process
    completed bootstrap without any RPC machinery.
    """
    marker = f"{spec_path}.pid{os.getpid()}"
    with open(marker, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "matched": matched, "applied": applied}, handle)


def tllm_general_plugin() -> None:
    """Install tLLM hooks and rebuild the runtime in this process.

    Idempotent: under fork the child inherits the parent's patched classes
    and configured RUNTIME, so both steps below short-circuit; under spawn
    both run for real.
    """
    path = os.environ.get(TLLM_BOOTSTRAP_SPEC_ENV, "").strip()
    if not path:
        return
    spec = read_bootstrap_spec(path)

    from tllm.runtime import lifecycle, residual_runtime

    lifecycle.install_runner_patch(residual_runtime)
    if runtime_matches_spec(spec):
        _write_marker(path, matched=True, applied=False)
        return
    if spec.skipped_consumers:
        import sys

        print(
            f"tllm bootstrap: consumer(s) {list(spec.skipped_consumers)} have no registered "
            "builder and are LOST in this spawn-built child process; register them via "
            "tllm.consumers.registry to make them reconstructable",
            file=sys.stderr,
        )
    apply_bootstrap_spec(spec)
    _write_marker(path, matched=False, applied=True)
