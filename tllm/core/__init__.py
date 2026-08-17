#!/usr/bin/env python3
"""Thin Python shim over the Rust core (``tllm._core``).

Hard dependency: there is no runtime fallback to a Python implementation.
The legacy Python implementations live under ``test/core/oracles/`` as
differential oracles only (see their headers for the sunset plan).

Build the extension once per toolchain change:

    cd rust/tllm_core && maturin develop --release
"""

from __future__ import annotations

try:
    from tllm._core import (  # noqa: F401
        CompiledPlan,
        DecodeLocalization,
        DecodePositions,
        RowPartition,
        SamplerPlan,
        compile_execution_plan,
        localize_decode_positions,
        localize_decode_rows,
        partition_rows,
        plan_sampler_step,
    )
except ImportError as exc:  # pragma: no cover - build guard
    raise ImportError(
        "tllm._core is not built. Build it with:\n"
        "    cd rust/tllm_core && maturin develop --release\n"
        "(requires the uv venv active and rustup toolchain installed)"
    ) from exc

__all__ = [
    "CompiledPlan",
    "DecodeLocalization",
    "DecodePositions",
    "RowPartition",
    "SamplerPlan",
    "compile_execution_plan",
    "localize_decode_positions",
    "localize_decode_rows",
    "partition_rows",
    "plan_sampler_step",
]
