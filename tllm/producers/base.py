#!/usr/bin/env python3
"""First-class producer contracts.

Producers are declarative plugins, symmetric with consumers: a producer
declares the ports it offers (``outputs``), the ports it consumes from other
producers (``requires``), the backend hook points it needs
(``hook_points``) and its CUDA-graph safety. ``bind()`` turns the
declaration into a runtime object whose ``emit()`` is invoked by the
backend driver at each declared hook point.

Adding a new producer must only require adding one producer module and a
test — never edits to backend drivers or the runtime core.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, Sequence, runtime_checkable

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
    SAMPLE_TOKENS_POST = "sample_tokens.post"

    @classmethod
    def all(cls) -> frozenset["HookPoint"]:
        return frozenset(cls)


class PortSink(Protocol):
    """Destination handed to ``BoundProducer.emit`` for publishing data."""

    def publish(self, port: PortId, payload: Any, *, step: Any) -> None:
        ...


@runtime_checkable
class BoundProducer(Protocol):
    def emit(self, hook: HookPoint, step: Any, sink: PortSink) -> None:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class Producer(Protocol):
    producer_id: str
    outputs: tuple[PortSpec, ...]
    requires: tuple[PortId, ...]
    hook_points: frozenset[HookPoint]
    graph_safety: GraphSafety

    def bind(self, context: "BindContext") -> BoundProducer:
        ...


class BindContext:
    """Everything a producer needs at bind time."""

    def __init__(
        self,
        *,
        runtime: Any,
        core: Any,
        backend_capabilities: Any = None,
        config: Any = None,
    ) -> None:
        self.runtime = runtime
        self.core = core
        self.backend_capabilities = backend_capabilities
        self.config = config


def validate_producer(producer: Any) -> None:
    """Structural validation with clear errors (fail at registration time)."""
    producer_id = getattr(producer, "producer_id", None)
    if not isinstance(producer_id, str) or not producer_id:
        raise TypeError(f"producer {producer!r} must declare a non-empty producer_id")
    outputs = getattr(producer, "outputs", None)
    if not isinstance(outputs, tuple) or not outputs:
        raise TypeError(f"producer {producer_id!r} must declare a non-empty outputs tuple")
    ids = [spec.id for spec in outputs]
    if len(set(ids)) != len(ids):
        raise TypeError(f"producer {producer_id!r} declares duplicate port ids: {ids}")
    hook_points = getattr(producer, "hook_points", frozenset())
    unknown = set(hook_points) - set(HookPoint.all())
    if unknown:
        raise TypeError(f"producer {producer_id!r} declares unknown hook points: {unknown}")
    if not callable(getattr(producer, "bind", None)):
        raise TypeError(f"producer {producer_id!r} must implement bind(context)")


def producer_requires(producer: Any) -> tuple[PortId, ...]:
    return tuple(getattr(producer, "requires", ()) or ())


def producer_hook_points(producer: Any) -> frozenset[HookPoint]:
    return frozenset(getattr(producer, "hook_points", frozenset()) or frozenset())


def producer_graph_safety(producer: Any) -> GraphSafety:
    safety = getattr(producer, "graph_safety", GraphSafety.CAPTURE_SAFE)
    if isinstance(safety, GraphSafety):
        return safety
    return GraphSafety(str(safety))


def producer_output_specs(producer: Any) -> Sequence[PortSpec]:
    return tuple(getattr(producer, "outputs", ()) or ())
