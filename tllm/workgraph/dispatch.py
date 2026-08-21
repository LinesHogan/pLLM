#!/usr/bin/env python3
"""Read-only dispatch-plan surface consumed by the workgraph IR.

The runtime's concrete ``DispatchPlan`` stays authoritative; workgraph
modules depend only on this structural surface so the layer direction stays
one-way (runtime may import workgraph, never the reverse).
"""

from __future__ import annotations

from typing import Protocol, Sequence

from tllm.ports.base import ConsumerFlow


class FlowDispatchTargetLike(Protocol):
    """The fields of a runtime flow-dispatch target the IR reads."""

    @property
    def consumer_id(self) -> str: ...

    @property
    def flow(self) -> ConsumerFlow: ...

    @property
    def flow_id(self) -> str: ...

    @property
    def flow_ordinal(self) -> int: ...

    @property
    def placement_hook(self) -> str: ...


class DispatchPlanLike(Protocol):
    """The read surface of a runtime dispatch plan the IR consumes."""

    @property
    def fusion_groups(self) -> tuple[tuple[str, ...], ...]: ...

    def flow_targets(self) -> Sequence[FlowDispatchTargetLike]: ...
