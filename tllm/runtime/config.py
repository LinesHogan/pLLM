#!/usr/bin/env python3
"""Shared runtime vocabulary, decoupled from any specific algorithm.

ESamp-specific settings (layers, distiller dims/lr, model bank, intervention
beta/backend) live in ``tllm.consumers.esamp.config.ESampConsumerConfig`` —
not here.
"""

from __future__ import annotations

from enum import Enum


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
