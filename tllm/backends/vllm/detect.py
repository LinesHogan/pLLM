#!/usr/bin/env python3
"""Structural detection of the installed vLLM backend.

Version strings select candidate families; structure probes confirm. Any
unrecognised combination raises instead of silently guessing.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VllmBackendInfo:
    version: str
    version_tuple: tuple[int, int, int]
    mrv1_runner_cls: Any
    mrv2_runner_cls: Any | None
    mrv1_prepare_inputs_arity: int  # params beyond `self`; -1 when missing
    mrv1_has_sample_tokens: bool
    mrv2_has_sample_tokens: bool
    sampler_returns_tuple: bool
    native_min_p: bool


def _parse_version(version: str) -> tuple[int, int, int]:
    parts: list[int] = []
    for chunk in str(version).split(".")[:3]:
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits or 0))
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


def _positional_arity(fn: Any) -> int:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return -1
    count = 0
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        count += 1
    return count


def _sampler_sample_returns_tuple(sampler_cls: Any) -> bool:
    sample = getattr(sampler_cls, "sample", None)
    if sample is None:
        return False
    try:
        annotation = inspect.signature(sample).return_annotation
    except (TypeError, ValueError):
        return False
    text = str(annotation)
    return "tuple" in text


def _has_native_min_p() -> bool:
    try:
        from vllm.v1.sample.logits_processor import builtin
    except Exception:
        return False
    return any("minp" in name.lower() or "min_p" in name.lower() for name in dir(builtin))


def probe_vllm() -> VllmBackendInfo:
    import vllm

    version = str(vllm.__version__)
    version_tuple = _parse_version(version)

    mrv1_cls = None
    mrv1_arity = -1
    mrv1_has_sample_tokens = False
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as _MRv1

        mrv1_cls = _MRv1
        mrv1_arity = _positional_arity(getattr(_MRv1, "_prepare_inputs", None))
        mrv1_has_sample_tokens = hasattr(_MRv1, "sample_tokens")
    except Exception:
        mrv1_cls = None

    mrv2_cls = None
    mrv2_has_sample_tokens = False
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner as _MRv2

        mrv2_cls = _MRv2
        mrv2_has_sample_tokens = hasattr(_MRv2, "sample_tokens")
    except Exception:
        mrv2_cls = None

    sampler_tuple = False
    try:
        from vllm.v1.sample.sampler import Sampler as _MRv1Sampler

        sampler_tuple = _sampler_sample_returns_tuple(_MRv1Sampler)
    except Exception:
        sampler_tuple = False

    return VllmBackendInfo(
        version=version,
        version_tuple=version_tuple,
        mrv1_runner_cls=mrv1_cls,
        mrv2_runner_cls=mrv2_cls,
        mrv1_prepare_inputs_arity=mrv1_arity,
        mrv1_has_sample_tokens=mrv1_has_sample_tokens,
        mrv2_has_sample_tokens=mrv2_has_sample_tokens,
        sampler_returns_tuple=sampler_tuple,
        native_min_p=_has_native_min_p(),
    )
