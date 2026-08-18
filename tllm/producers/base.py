#!/usr/bin/env python3
"""Compiler-visible producer descriptors.

Producer descriptors are declarative, symmetric with consumer flows: they
declares the ports it offers (``outputs``), the ports it consumes from other
producers (``requires``), the backend hook points it needs
(``hook_points``) and its CUDA-graph safety. Backend drivers own the actual
Tensor/CUDA materialization; the Rust plan resolves and validates these
descriptors at startup.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, cast, runtime_checkable

from tllm.backends.protocol import GraphSafety
from tllm.ports.port_id import PortId
from tllm.ports.port_spec import PortSpec


class HookPoint(str, Enum):
    """Backend-agnostic hook points a producer can attach to.

    Names mirror ``RUNTIME_EVENT_POINTS`` in
    ``tllm.runtime.vllm_patch.port_runtime_hooks``.
    """

    LOAD_MODEL_PRE = "load_model.pre"
    LOAD_MODEL_POST = "load_model.post"
    PREPARE_INPUTS_PRE = "prepare_inputs.pre"
    PREPARE_INPUTS_POST = "prepare_inputs.post"
    LAYER_PRE = "layer.pre"
    LAYER_POST = "layer.post"
    BLOCK_END = "block.end"
    STACK_END = "stack.end"
    EXECUTE_MODEL_PRE = "execute_model.pre"
    EXECUTE_MODEL_POST = "execute_model.post"
    SAMPLE_TOKENS_PRE = "sample_tokens.pre"
    SAMPLER_PRE_DRAW = "sampler.pre_draw"
    SAMPLE_TOKENS_POST = "sample_tokens.post"

    @classmethod
    def all(cls) -> frozenset["HookPoint"]:
        return frozenset(cls)


@runtime_checkable
class ProducerDescriptor(Protocol):
    producer_id: str
    outputs: tuple[PortSpec, ...]
    requires: tuple[PortId, ...]
    hook_points: frozenset[HookPoint]
    graph_safety: GraphSafety


# Import compatibility for code written against the original name. It now
# intentionally denotes a declaration, not an unimplemented executor.
Producer = ProducerDescriptor


def validate_producer(producer: object) -> ProducerDescriptor:
    """Validate an untyped plugin once, then return the narrowed protocol."""
    producer_id = getattr(producer, "producer_id", None)
    if not isinstance(producer_id, str) or not producer_id:
        raise TypeError(f"producer {producer!r} must declare a non-empty producer_id")
    outputs = getattr(producer, "outputs", None)
    if not isinstance(outputs, tuple) or not outputs:
        raise TypeError(f"producer {producer_id!r} must declare a non-empty outputs tuple")
    output_values = cast("tuple[object, ...]", outputs)
    if not all(isinstance(spec, PortSpec) for spec in output_values):
        raise TypeError(f"producer {producer_id!r} outputs must contain only PortSpec values")
    output_specs = cast("tuple[PortSpec, ...]", output_values)
    ids = [spec.id for spec in output_specs]
    if len(set(ids)) != len(ids):
        raise TypeError(f"producer {producer_id!r} declares duplicate port ids: {ids}")
    requires = getattr(producer, "requires", ())
    if not isinstance(requires, tuple) or not all(
        isinstance(port, PortId) for port in cast("tuple[object, ...]", requires)
    ):
        raise TypeError(f"producer {producer_id!r} requires must be a tuple of PortId values")
    hook_points = getattr(producer, "hook_points", None)
    if not isinstance(hook_points, frozenset) or not all(
        isinstance(hook, HookPoint) for hook in cast("frozenset[object]", hook_points)
    ):
        raise TypeError(f"producer {producer_id!r} hook_points must be a frozenset of HookPoint values")
    hook_points = cast("frozenset[HookPoint]", hook_points)
    unknown = set(hook_points) - set(HookPoint.all())
    if unknown:
        raise TypeError(f"producer {producer_id!r} declares unknown hook points: {unknown}")
    safety = getattr(producer, "graph_safety", None)
    if not isinstance(safety, GraphSafety):
        raise TypeError(f"producer {producer_id!r} graph_safety must be a GraphSafety value")
    return cast(ProducerDescriptor, producer)
