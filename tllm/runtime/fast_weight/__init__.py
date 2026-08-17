#!/usr/bin/env python3
"""Generic fast-weight overlay infrastructure.

Per-request / per-key low-rank or full fast-weight overlays on linear
layers: install/restore forward wrappers, apply bounded per-step row
updates, and track slot state. Algorithm-agnostic — the In-Place-TTT
consumer drives this through configuration; other consumers may too.
"""

from tllm.runtime.fast_weight.state import (  # noqa: F401
    FastWeightKey,
    FastWeightState,
    apply_down_proj_fast_weight,
    module_weight_view,
    normalize_fast_weight_mode,
    run_down_proj_fast_weight_step,
)
