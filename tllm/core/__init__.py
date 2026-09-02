#!/usr/bin/env python3
"""Thin Python shim over the Rust core (``tllm._core``).

Hard dependency: there is no runtime fallback to a Python implementation.

Build the extension once per toolchain change:

    cd rust/tllm_core && maturin develop --release
"""

from __future__ import annotations

try:
    from tllm._core import (  # noqa: F401
        PackedPrefillRowPlan,
        PrefillRowSpan,
        DecodeLocalization,
        DecodePositions,
        RowPartition,
        SamplerPlan,
        localize_decode_positions,
        localize_decode_rows,
        partition_rows,
        plan_packed_prefill_rows,
        plan_prefill_rows,
        plan_sampler_step,
    )
except ImportError as exc:  # pragma: no cover - build guard
    raise ImportError(
        "tllm._core is not built. Build it with:\n"
        "    cd rust/tllm_core && maturin develop --release\n"
        "(requires the uv venv active and rustup toolchain installed)"
    ) from exc

__all__ = [
    "PackedPrefillRowPlan",
    "PrefillRowSpan",
    "DecodeLocalization",
    "DecodePositions",
    "RowPartition",
    "SamplerPlan",
    "localize_decode_positions",
    "localize_decode_rows",
    "partition_rows",
    "plan_packed_prefill_rows",
    "plan_prefill_rows",
    "plan_sampler_step",
]
