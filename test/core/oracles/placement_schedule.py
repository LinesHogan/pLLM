#!/usr/bin/env python3
"""REFERENCE oracle — 仅作 Rust core 对照基准,禁止运行时引用。

Reference implementation of the B1 identity placement compiler
(``tllm._core.compile_placement_schedule``). Sunset plan: once the Rust core
shows oracle matrix equivalence + dual-runner parity for two consecutive
development cycles, delete this file and evolve only in Rust.
"""

from __future__ import annotations

HOOK_POINTS = (
    "load_model.pre",
    "load_model.post",
    "prepare_inputs.pre",
    "prepare_inputs.post",
    "layer.pre",
    "layer.post",
    "block.end",
    "stack.end",
    "execute_model.pre",
    "execute_model.post",
    "sample_tokens.pre",
    "sample_tokens.post",
)
IN_CAPTURE_HOOKS = ("layer.pre", "layer.post", "block.end", "stack.end")
MOBILITIES = ("pinned", "deferrable", "advanceable", "free")
DEADLINES = ("step_end", "next_prepare", "generate_end")
OBSERVATIONS = ("at_declared_window", "latest_at_execution", "stale_ok")
GRAPH_MODES = ("eager", "piecewise", "full")


class OraclePlacementError(RuntimeError):
    pass


def _qr(s: str) -> str:
    return f"'{s}'"


def _qr_list(v) -> str:
    return "[" + ", ".join(_qr(x) for x in v) + "]"


def oracle_compile_placement_schedule(flows, graph_mode: str):
    """Return the identity schedule [(flow_id, declared_hook)] or raise.

    ``flows`` is a sequence of mappings with keys: flow_id, declared_hook,
    mobility, deadline, max_defer_steps, observes_version.
    """
    if graph_mode not in GRAPH_MODES:
        raise OraclePlacementError(
            f"unsupported graph mode {_qr(graph_mode)}; expected one of {_qr_list(GRAPH_MODES)}"
        )
    placements = []
    for flow in flows:
        flow_id = flow["flow_id"]
        declared_hook = flow["declared_hook"]
        mobility = flow.get("mobility") or "pinned"
        deadline = flow.get("deadline") or "step_end"
        max_defer = int(flow.get("max_defer_steps") or 0)
        observes = flow.get("observes_version") or "at_declared_window"
        if declared_hook not in HOOK_POINTS:
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} declares unknown hook point {_qr(declared_hook)}; "
                f"expected one of {_qr_list(HOOK_POINTS)}"
            )
        if mobility not in MOBILITIES:
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} declares unsupported timing mobility {_qr(mobility)}; "
                f"expected one of {_qr_list(MOBILITIES)}"
            )
        if deadline not in DEADLINES:
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} declares unsupported timing deadline {_qr(deadline)}; "
                f"expected one of {_qr_list(DEADLINES)}"
            )
        if observes not in OBSERVATIONS:
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} declares unsupported version observation {_qr(observes)}; "
                f"expected one of {_qr_list(OBSERVATIONS)}"
            )
        if max_defer < 0:
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} declares negative max_defer_steps {max_defer}"
            )
        if mobility == "pinned":
            if max_defer != 0:
                raise OraclePlacementError(
                    f"flow {_qr(flow_id)} is pinned but declares max_defer_steps={max_defer}"
                )
            if observes != "at_declared_window":
                raise OraclePlacementError(
                    f"flow {_qr(flow_id)} is pinned but declares observes_version={_qr(observes)}"
                )
        if mobility in ("deferrable", "free") and max_defer < 1:
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} is {mobility} but declares max_defer_steps={max_defer} (must be >=1)"
            )
        if mobility in ("deferrable", "free") and deadline == "step_end":
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} is {mobility} with deadline 'step_end'; "
                "deferral is across steps, so this admits no movement"
            )
        if mobility == "advanceable" and observes == "at_declared_window":
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} is advanceable but observes 'at_declared_window'; "
                "running earlier can never observe that version"
            )
        if graph_mode != "eager" and mobility != "pinned" and declared_hook in IN_CAPTURE_HOOKS:
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} declares mobility {_qr(mobility)} at in-capture hook "
                f"{_qr(declared_hook)} under graph mode {_qr(graph_mode)}; "
                "timing movement across the capture boundary requires the full placement compiler — "
                "pin the flow or use graph mode 'eager'"
            )
        placements.append((flow_id, declared_hook))
    return placements
