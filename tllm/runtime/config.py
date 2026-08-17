#!/usr/bin/env python3
"""Generic runtime configuration, decoupled from any specific algorithm.

ESamp-specific settings (layers, distiller dims/lr, model bank, intervention
beta/backend) live in ``tllm.consumers.esamp.config.ESampConsumerConfig`` —
not here. ``TllmRuntimeConfig`` only covers what the core runtime owns:
lifecycle, streams, buffers, failure isolation and CUDA-graph policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - py<3.11
    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)


class AdaptationStreamMode(StrEnum):
    """How adaptation (test-time training) work is scheduled on CUDA streams."""

    DUAL = "dual"
    SINGLE = "single"
    SERIAL = "serial"


def normalize_adaptation_stream_mode(value: str) -> AdaptationStreamMode:
    mode = str(value).strip().lower()
    try:
        return AdaptationStreamMode(mode)
    except ValueError as exc:
        choices = ", ".join(m.value for m in AdaptationStreamMode)
        raise ValueError(f"unknown adaptation_stream_mode={value!r}; expected one of: {choices}") from exc


class ModelBankForwardBackendName(StrEnum):
    """Forward implementation selector for per-request low-rank model banks."""

    TORCH = "torch"
    TRITON_GROUPED = "triton_grouped"


def normalize_model_bank_forward_backend(name: str | None) -> ModelBankForwardBackendName:
    key = str(name or "torch").strip().lower() or "torch"
    if key in {"torch", "reference", "default"}:
        return ModelBankForwardBackendName.TORCH
    if key in {"triton", "triton_grouped", "experimental_triton_grouped"}:
        return ModelBankForwardBackendName.TRITON_GROUPED
    raise ValueError(f"unsupported model-bank forward backend: {key}")


class ConsumerErrorPolicy(str, Enum):
    """What happens when a consumer raises inside a step."""

    FAIL_STEP = "fail_step"  # propagate; the StepSession transitions to FAILED
    ISOLATE = "isolate"  # log + disable the consumer, keep serving


@dataclass(frozen=True)
class StreamPolicy:
    adaptation_stream_mode: str = "dual"  # "single" | "dual"
    adaptation_stream_priority: int = 0
    adaptation_pipeline_slots: int = 4


@dataclass(frozen=True)
class BufferPolicy:
    graph_scratch_rows: int = 0
    # Default ownership for port payloads when a flow does not say otherwise.
    default_ownership: str = "borrowed"  # "borrowed" | "runtime_lease"


@dataclass(frozen=True)
class FailurePolicy:
    on_consumer_error: ConsumerErrorPolicy = ConsumerErrorPolicy.FAIL_STEP


@dataclass(frozen=True)
class GraphPolicy:
    # Allow consumers to launch work from inside CUDA-graph capture/replay.
    # Requires every involved producer to declare GraphSafety.CAPTURE_SAFE.
    allow_consumer_launch_in_graph: bool = False


@dataclass(frozen=True)
class TllmRuntimeConfig:
    stream: StreamPolicy = StreamPolicy()
    buffer: BufferPolicy = BufferPolicy()
    failure: FailurePolicy = FailurePolicy()
    graph: GraphPolicy = GraphPolicy()
