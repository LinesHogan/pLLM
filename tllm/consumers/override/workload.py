"""OverRIDE declaration entry point with a switchable implementation.

``implementation='consumer'`` keeps the legacy Program consumer (build it via
``tllm.consumers.override.register.build``); ``implementation='region'``
returns the native Region workload for ``TllmRuntime(workloads=...)``.  Both
implementations stay available side by side during the migration transition.
"""

from __future__ import annotations

from tllm.consumers.override.config import OverRIDEConsumerConfig


def build_override_workload(config: OverRIDEConsumerConfig | None = None) -> object:
    """Build a config-specialized OverRIDE declaration or native workload."""

    resolved = config or OverRIDEConsumerConfig()
    if resolved.implementation == "region":
        from tllm.consumers.override.region_workload import (
            build_override_region_workload,
        )

        return build_override_region_workload(resolved)
    if resolved.implementation == "consumer":
        raise ValueError(
            "implementation='consumer' remains a legacy Program consumer; use "
            "tllm.consumers.override.register.build(config) with "
            "TllmRuntime(consumers=...)"
        )
    raise ValueError(
        f"unknown OverRIDE implementation {resolved.implementation!r}; "
        "expected 'consumer' or 'region'"
    )


__all__ = ["build_override_workload"]
