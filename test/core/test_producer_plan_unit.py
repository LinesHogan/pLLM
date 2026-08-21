#!/usr/bin/env python3
"""Unit tests for the first-class producer layer and plan compiler."""

from __future__ import annotations

import unittest

from tllm.backends.protocol import BackendCapabilities, GraphSafety, SamplerStage
from tllm.ports.base import ConsumerFlow, PortKind, PortRead
from tllm.ports.port_id import PortId
from tllm.ports.port_spec import PortSpec
from tllm.ports.residual_stream import ResidualStream
from tllm.producers.base import HookPoint
from tllm.producers.registry import (
    ExecutionPlanError,
    ProducerRegistry,
    compile_execution_plan,
)


def _port(ns: str, name: str, kind: PortKind, version: int = 1) -> PortSpec:
    return PortSpec(id=PortId(ns, name, version), kind=kind, payload_type="HiddenBatch")


def _producer(
    pid: str,
    outputs,
    *,
    requires=(),
    hooks=frozenset({HookPoint.EXECUTE_MODEL_POST}),
    safety=GraphSafety.CAPTURE_SAFE,
):
    class _P:
        producer_id = pid
        hook_points = hooks
        graph_safety = safety

        def __init__(self) -> None:
            self.outputs = tuple(outputs)
            self.requires = tuple(requires)

    return _P()


class _Consumer:
    consumer_id = "test.consumer"

    def __init__(self, kinds) -> None:
        self._kinds = tuple(kinds)

    def flows(self):
        return (
            ConsumerFlow(
                reads=[PortRead(kind=k) for k in self._kinds],
                writes=(),
                window="same_step",
            ),
        )


class ExecutionPlanCompilerUnitTest(unittest.TestCase):
    def test_exact_read_phase_selects_only_matching_residual_producer(self) -> None:
        decode = PortSpec(
            id=PortId("tllm", "residual.decode", 1),
            kind=PortKind.RESIDUAL_STREAM,
            payload_type="HiddenBatch",
            phases=("decode",),
        )
        prefill = PortSpec(
            id=PortId("tllm", "residual.prefill", 1),
            kind=PortKind.RESIDUAL_STREAM,
            payload_type="HiddenBatch",
            phases=("prefill",),
        )
        reg = ProducerRegistry()
        reg.register(_producer("decode", [decode]))
        reg.register(_producer("prefill", [prefill]))

        class _DecodeConsumer:
            consumer_id = "decode.consumer"

            def flows(self):
                return (
                    ConsumerFlow(
                        reads=(
                            ResidualStream.read(
                                layer=0,
                                site="block_output",
                                phase="decode",
                            ),
                        ),
                        writes=(),
                        window="same_step",
                    ),
                )

        plan = compile_execution_plan(registry=reg, consumers=[_DecodeConsumer()])

        self.assertEqual(
            [planned.entry.producer_id for planned in plan.producers],
            ["decode"],
        )

    def test_missing_producer_fails_fast(self) -> None:
        reg = ProducerRegistry()
        reg.register(_producer("a", [_port("tllm", "x", PortKind.LOGITS)]))
        with self.assertRaisesRegex(ExecutionPlanError, "no registered producer"):
            compile_execution_plan(
                registry=reg, consumers=[_Consumer([PortKind.RESIDUAL_STREAM])]
            )

    def test_ambiguous_priority_fails_fast(self) -> None:
        reg = ProducerRegistry()
        reg.register(_producer("p1", [_port("tllm", "a", PortKind.RESIDUAL_STREAM)]))
        reg.register(_producer("p2", [_port("tllm", "a", PortKind.RESIDUAL_STREAM)]))
        with self.assertRaisesRegex(ExecutionPlanError, "disambiguate"):
            compile_execution_plan(
                registry=reg, consumers=[_Consumer([PortKind.RESIDUAL_STREAM])]
            )

    def test_higher_priority_wins(self) -> None:
        reg = ProducerRegistry()
        reg.register(_producer("p1", [_port("tllm", "a", PortKind.RESIDUAL_STREAM)]))
        reg.register(
            _producer("p2", [_port("tllm", "a", PortKind.RESIDUAL_STREAM)]), priority=10
        )
        plan = compile_execution_plan(
            registry=reg, consumers=[_Consumer([PortKind.RESIDUAL_STREAM])]
        )
        self.assertEqual([p.entry.producer_id for p in plan.producers], ["p2"])

    def test_cycle_detected(self) -> None:
        a = _port("tllm", "a", PortKind.RESIDUAL_STREAM)
        b = _port("tllm", "b", PortKind.HIDDEN_TARGET)
        reg = ProducerRegistry()
        reg.register(_producer("pa", [a], requires=(b.id,)))
        reg.register(_producer("pb", [b], requires=(a.id,)))
        with self.assertRaisesRegex(ExecutionPlanError, "cycle"):
            compile_execution_plan(
                registry=reg, consumers=[_Consumer([PortKind.RESIDUAL_STREAM])]
            )

    def test_graph_safety_incompatible_fails_fast(self) -> None:
        reg = ProducerRegistry()
        reg.register(
            _producer(
                "pa",
                [_port("tllm", "a", PortKind.RESIDUAL_STREAM)],
                safety=GraphSafety.REPLAY_STATEFUL,
            )
        )
        caps = BackendCapabilities(
            backend_id="test-eager",
            runner_family="mrv1",
            sampler_stages=frozenset({SamplerStage.AFTER_DRAW}),
            native_min_p=True,
            tuple_sampler_return=True,
            split_execute_and_sample=True,
            graph_safety=GraphSafety.EAGER_ONLY,
        )
        with self.assertRaisesRegex(ExecutionPlanError, "EAGER_ONLY"):
            compile_execution_plan(
                registry=reg,
                consumers=[_Consumer([PortKind.RESIDUAL_STREAM])],
                backend_capabilities=caps,
            )


if __name__ == "__main__":
    unittest.main()
