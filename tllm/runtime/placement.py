#!/usr/bin/env python3
"""Placement compilation: validate flow timing contracts via the Rust core.

B1 scope: the Rust ``compile_placement_schedule`` emits the IDENTITY schedule
(every flow at its declared hook) and acts as the single validation authority
for TimingContract combinations at plan-compile time. Real placement
computation lands in B5; B3's debug cross-check consumes this module's
schedule table.

Today's effective dispatch point for every per-step flow is the decode
dispatch inside the execute_model wrapper, so the declared hook is
``execute_model.post``; post-generate flows (generation-trace reads, reward /
retry / barrier writes) live outside per-step placement and are excluded.
"""

from __future__ import annotations

from typing import Sequence

from tllm.core import compile_placement_schedule
from tllm.ports.base import ConsumerFlow, TimingContract, TimingMobility
from tllm.runtime.graph_policy import GraphMode

#: Where per-step flows actually dispatch today (decode bundle dispatch runs
#: inside the execute_model wrapper, after the model forward).
DECLARED_DISPATCH_HOOK = "execute_model.post"


def flow_timing_desc(flow_id: str, flow: ConsumerFlow) -> dict[str, object]:
    """Build the plain-data descriptor the Rust compiler consumes."""
    raw = flow.timing
    # ConsumerFlow.__post_init__ always normalizes to TimingContract; the
    # isinstance keeps the type checker honest about the declared union.
    timing = raw if isinstance(raw, TimingContract) else TimingContract(mobility=TimingMobility(str(raw)))
    return {
        "flow_id": str(flow_id),
        "declared_hook": DECLARED_DISPATCH_HOOK,
        "mobility": str(timing.mobility),
        "deadline": str(timing.deadline),
        "max_defer_steps": int(timing.max_defer_steps),
        "observes_version": str(timing.observes_version),
    }


def validate_flow_timing(
    flows: Sequence[tuple[str, ConsumerFlow]],
    *,
    graph_mode: GraphMode = GraphMode.EAGER,
) -> None:
    """Fail fast on illegal timing contracts / graph-mode combinations."""
    if not flows:
        return
    compile_placement_schedule(
        [flow_timing_desc(flow_id, flow) for flow_id, flow in flows],
        graph_mode.name.lower(),
    )
