#!/usr/bin/env python3
"""Residual-stream producers on the first-class producer contract.

These wrap the existing decode/prefill localization engines
(``tllm.producer.decode`` / ``tllm.producer.prefill``) so the declaration
surface (ports, hooks, graph safety) is machine-readable and plannable. The
localization math itself is unchanged and still driven by the backend
drivers; the producer objects are what the ExecutionPlanCompiler reasons
about.
"""

from __future__ import annotations

from typing import Any

from tllm.backends.protocol import GraphSafety
from tllm.ports.base import PortKind
from tllm.ports.port_id import PortId
from tllm.ports.port_spec import PortSpec
from tllm.producers.base import BindContext, BoundProducer, HookPoint

RESIDUAL_DECODE_PORT = PortId("tllm", "residual.decode", 1)
RESIDUAL_PREFILL_PORT = PortId("tllm", "residual.prefill", 1)


class _LegacyBoundProducer:
    """Bound producer that delegates to the legacy hook-driven path.

    The backend drivers already invoke the localization engines via
    ``port_runtime_hooks``; ``emit`` is therefore a no-op placeholder until
    the drivers dispatch through the plan directly. Keeping it explicit
    makes the migration visible and testable.
    """

    def __init__(self, producer_id: str, context: BindContext) -> None:
        self.producer_id = producer_id
        self.context = context
        self.emitted: list[HookPoint] = []

    def emit(self, hook: HookPoint, step: Any, sink: Any) -> None:
        self.emitted.append(hook)

    def close(self) -> None:
        self.emitted.clear()


class DecodeResidualProducer:
    """Decode-row residual localization (graph-safe capture path)."""

    producer_id = "tllm.residual.decode"
    outputs = (
        PortSpec(
            id=RESIDUAL_DECODE_PORT,
            kind=PortKind.RESIDUAL_STREAM,
            payload_type="HiddenBatch",
            description="decode-step capture-layer hidden rows localized per request",
            writable=True,
            write_modes=("replace", "add"),
        ),
    )
    requires: tuple = ()
    hook_points = frozenset(
        {HookPoint.PREPARE_INPUTS_POST, HookPoint.EXECUTE_MODEL_POST}
    )
    graph_safety = GraphSafety.CAPTURE_SAFE

    def bind(self, context: BindContext) -> BoundProducer:
        return _LegacyBoundProducer(self.producer_id, context)


class PrefillResidualProducer:
    """Prefill-row residual localization (eager-first)."""

    producer_id = "tllm.residual.prefill"
    outputs = (
        PortSpec(
            id=RESIDUAL_PREFILL_PORT,
            kind=PortKind.RESIDUAL_STREAM,
            payload_type="HiddenBatch",
            description="prefill-step capture-layer hidden rows localized per request",
        ),
    )
    requires: tuple = ()
    hook_points = frozenset(
        {HookPoint.PREPARE_INPUTS_POST, HookPoint.EXECUTE_MODEL_POST}
    )
    graph_safety = GraphSafety.EAGER_ONLY

    def bind(self, context: BindContext) -> BoundProducer:
        return _LegacyBoundProducer(self.producer_id, context)


def default_residual_producers() -> tuple[Any, ...]:
    return (DecodeResidualProducer(), PrefillResidualProducer())
