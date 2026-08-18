#!/usr/bin/env python3
"""Residual-stream producers on the first-class producer contract.

The backend drivers materialize these ports; these objects are the typed
declarations consumed by the execution-plan compiler.
"""

from __future__ import annotations

from tllm.backends.protocol import GraphSafety
from tllm.ports.base import PortKind
from tllm.ports.port_id import PortId
from tllm.ports.port_spec import PortSpec
from tllm.producers.base import HookPoint, ProducerDescriptor

RESIDUAL_DECODE_PORT = PortId("tllm", "residual.decode", 1)
RESIDUAL_PREFILL_PORT = PortId("tllm", "residual.prefill", 1)


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
            phases=("decode",),
        ),
    )
    requires: tuple[PortId, ...] = ()
    hook_points = frozenset(
        {
            HookPoint.PREPARE_INPUTS_POST,
            HookPoint.LAYER_POST,
            HookPoint.EXECUTE_MODEL_POST,
        }
    )
    graph_safety = GraphSafety.CAPTURE_SAFE

class PrefillResidualProducer:
    """Prefill-row residual localization (eager-first)."""

    producer_id = "tllm.residual.prefill"
    outputs = (
        PortSpec(
            id=RESIDUAL_PREFILL_PORT,
            kind=PortKind.RESIDUAL_STREAM,
            payload_type="HiddenBatch",
            description="prefill-step capture-layer hidden rows localized per request",
            phases=("prefill",),
        ),
    )
    requires: tuple[PortId, ...] = ()
    hook_points = frozenset(
        {HookPoint.PREPARE_INPUTS_POST, HookPoint.EXECUTE_MODEL_POST}
    )
    graph_safety = GraphSafety.EAGER_ONLY

def default_residual_producers() -> tuple[ProducerDescriptor, ...]:
    return (DecodeResidualProducer(), PrefillResidualProducer())
