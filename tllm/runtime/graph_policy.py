#!/usr/bin/env python3
"""Formal CUDA-graph policy for hook-driven producer/consumer work.

Python hooks run at *capture* time, not replay time; anything that must
execute per step under a replayed graph has to operate on address-stable
persistent buffers and be declared ``GraphSafety.CAPTURE_SAFE`` (or
``REPLAY_STATEFUL``). This module lifts the previous ad-hoc
``_runner_uses_compilation_or_cudagraph`` check into an explicit contract:

* resolve the runner's effective :class:`GraphMode`;
* decide whether consumers may launch from hooks in that mode;
* validate a compiled execution plan against the mode at startup.
"""

from __future__ import annotations

import os
from typing import Any

from tllm.backends.protocol import GraphSafety
# Canonical home: tllm.workgraph.graph_mode. Re-exported here; existing
# runtime/backends users keep importing it from this module.
from tllm.workgraph.graph_mode import GraphMode

DEBUG_GRAPH_MODE_ENV = "TLLM_DEBUG_GRAPH_MODE"


class GraphPolicyError(RuntimeError):
    pass


def _debug_forced_graph_mode() -> GraphMode | None:
    """tLLM-developer-only override; never part of any consumer contract."""
    raw = os.environ.get(DEBUG_GRAPH_MODE_ENV, "").strip().lower()
    if not raw:
        return None
    mapping = {"eager": GraphMode.EAGER, "piecewise": GraphMode.PIECEWISE, "full": GraphMode.FULL}
    try:
        return mapping[raw]
    except KeyError as exc:
        raise GraphPolicyError(
            f"{DEBUG_GRAPH_MODE_ENV} must be one of {sorted(mapping)}, got {raw!r}"
        ) from exc


def resolve_graph_mode(runner: Any) -> GraphMode:
    """Best-effort structural read of the runner's effective graph mode."""
    forced = _debug_forced_graph_mode()
    if forced is not None:
        return forced
    if bool(getattr(runner, "enforce_eager", False)):
        return GraphMode.EAGER
    cfg = getattr(runner, "compilation_config", None)
    if cfg is None:
        # vLLM runners expose use_cuda_graph when graphs are active.
        if getattr(runner, "use_cuda_graph", None) is True:
            return GraphMode.FULL
        return GraphMode.EAGER
    mode = getattr(cfg, "cudagraph_mode", None)
    name = getattr(mode, "name", None)
    if isinstance(name, str):
        upper = name.upper()
        if upper in ("NONE",):
            return GraphMode.EAGER
        if "FULL" in upper:
            return GraphMode.FULL
        if "PIECEWISE" in upper:
            return GraphMode.PIECEWISE
    try:
        value = int(mode)
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        level = getattr(cfg, "level", 0)
        try:
            level = int(level)
        except (TypeError, ValueError):
            level = 0
        return GraphMode.PIECEWISE if level > 0 else GraphMode.EAGER
    return GraphMode.PIECEWISE



_SAFETY_RANK = {
    GraphSafety.EAGER_ONLY: 0,
    GraphSafety.CAPTURE_SAFE: 1,
    GraphSafety.REPLAY_STATEFUL: 2,
}

_MODE_REQUIREMENT = {
    GraphMode.EAGER: GraphSafety.EAGER_ONLY,
    GraphMode.PIECEWISE: GraphSafety.CAPTURE_SAFE,
    GraphMode.FULL: GraphSafety.REPLAY_STATEFUL,
}


def assert_plan_graph_compatible(plan: Any, mode: GraphMode) -> None:
    """Fail fast if any planned producer is unsafe for the runner's mode."""
    required = _MODE_REQUIREMENT[mode]
    for planned in getattr(plan, "producers", ()) or ():
        safety = planned.entry.producer.graph_safety
        if isinstance(safety, str):
            safety = GraphSafety(safety)
        if _SAFETY_RANK[safety] < _SAFETY_RANK[required]:
            raise GraphPolicyError(
                f"producer {planned.entry.producer_id!r} is {safety.name} but the "
                f"runner is in {mode.name} mode which requires {required.name}. "
                "Use enforce_eager or a capture-safe producer."
            )
