"""Legacy OverRIDE migration dispatcher.

``implementation='consumer'`` keeps the legacy Program consumer (build it via
``tllm.consumers.override.register.build``); ``implementation='native'``
returns the native workload for ``TllmRuntime(workloads=...)``. Both
implementations stay available side by side during the migration transition.
"""

from __future__ import annotations

from tllm.consumers.override.config import OverRIDEConsumerConfig


def build_override_workload(config: OverRIDEConsumerConfig | None = None) -> object:
    """Build a config-specialized OverRIDE declaration or native workload."""

    resolved = config or OverRIDEConsumerConfig()
    if resolved.implementation == "native":
        from tllm.consumers.override.workload import (
            OverRIDEConfig,
            OverRIDEWorkload,
        )

        return OverRIDEWorkload(
            OverRIDEConfig(
                enabled=resolved.enabled,
                hidden_size=resolved.hidden_size,
                vocab_size=resolved.vocab_size,
                rank=resolved.rank,
                num_iterations=resolved.num_iterations,
                lambd=resolved.lambd,
                learning_rate=resolved.learning_rate,
                source_layer_path=resolved.source_layer_path,
            )
        )
    if resolved.implementation == "consumer":
        raise ValueError(
            "implementation='consumer' remains a legacy Program consumer; use "
            "tllm.consumers.override.register.build(config) with "
            "TllmRuntime(consumers=...)"
        )
    raise ValueError(
        f"unknown OverRIDE implementation {resolved.implementation!r}; "
        "expected 'consumer' or 'native'"
    )


__all__ = ["build_override_workload"]
