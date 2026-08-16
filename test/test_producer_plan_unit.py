#!/usr/bin/env python3
"""Unit tests for the first-class producer layer and plan compiler."""

from __future__ import annotations

import unittest

from tllm.backends.protocol import BackendCapabilities, GraphSafety, SamplerStage
from tllm.ports.base import ConsumerFlow, PortKind, PortRead
from tllm.ports.port_id import PortId
from tllm.ports.port_spec import PortSpec
from tllm.producers.base import BindContext, HookPoint
from tllm.producers.registry import (
    ExecutionPlanError,
    ProducerRegistry,
    compile_execution_plan,
)


def _port(ns: str, name: str, kind: PortKind, version: int = 1) -> PortSpec:
    return PortSpec(id=PortId(ns, name, version), kind=kind, payload_type="HiddenBatch")


class _Bound:
    def __init__(self) -> None:
        self.emitted = []

    def emit(self, hook, step, sink) -> None:
        self.emitted.append(hook)

    def close(self) -> None:
        pass


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

        def bind(self, context: BindContext) -> _Bound:
            return _Bound()

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


class PortIdUnitTest(unittest.TestCase):
    def test_roundtrip(self) -> None:
        pid = PortId("tllm", "residual.decode", 2)
        self.assertEqual(str(pid), "tllm:residual.decode@2")
        self.assertEqual(PortId.parse(str(pid)), pid)

    def test_rejects_malformed(self) -> None:
        for bad in ("noversion", "ns:name@x", ":name@1", "ns:@1"):
            with self.assertRaises(ValueError, msg=bad):
                PortId.parse(bad)


class ProducerRegistryUnitTest(unittest.TestCase):
    def test_duplicate_producer_id_rejected(self) -> None:
        reg = ProducerRegistry()
        reg.register(_producer("a", [_port("tllm", "x", PortKind.RESIDUAL_STREAM)]))
        with self.assertRaises(ExecutionPlanError):
            reg.register(_producer("a", [_port("tllm", "y", PortKind.LOGITS)]))

    def test_same_port_may_be_offered_by_multiple_candidates(self) -> None:
        reg = ProducerRegistry()
        reg.register(_producer("base", [_port("tllm", "x", PortKind.RESIDUAL_STREAM)]))
        reg.register(
            _producer("override", [_port("tllm", "x", PortKind.RESIDUAL_STREAM)]),
            priority=5,
        )
        self.assertEqual(len(reg.candidates_for_kind(PortKind.RESIDUAL_STREAM)), 2)

    def test_multiple_candidates_per_kind_allowed(self) -> None:
        reg = ProducerRegistry()
        reg.register(_producer("raw", [_port("tllm", "res.raw", PortKind.RESIDUAL_STREAM)]))
        reg.register(
            _producer("norm", [_port("tllm", "res.norm", PortKind.RESIDUAL_STREAM)]),
            priority=5,
        )
        self.assertEqual(len(reg.candidates_for_kind(PortKind.RESIDUAL_STREAM)), 2)


class ExecutionPlanCompilerUnitTest(unittest.TestCase):
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

    def test_complementary_kind_producers_both_planned(self) -> None:
        reg = ProducerRegistry()
        reg.register(_producer("decode", [_port("tllm", "res.decode", PortKind.RESIDUAL_STREAM)]))
        reg.register(_producer("prefill", [_port("tllm", "res.prefill", PortKind.RESIDUAL_STREAM)]))
        plan = compile_execution_plan(
            registry=reg, consumers=[_Consumer([PortKind.RESIDUAL_STREAM])]
        )
        self.assertEqual(
            sorted(p.entry.producer_id for p in plan.producers), ["decode", "prefill"]
        )

    def test_dependency_chain_topologically_ordered(self) -> None:
        raw = _port("tllm", "res.raw", PortKind.RESIDUAL_STREAM)
        norm = _port("tllm", "res.norm", PortKind.HIDDEN_TARGET)
        reg = ProducerRegistry()
        reg.register(_producer("norm", [norm], requires=(raw.id,), hooks=frozenset({HookPoint.LAYER_POST})))
        reg.register(_producer("raw", [raw]))
        plan = compile_execution_plan(
            registry=reg,
            consumers=[_Consumer([PortKind.HIDDEN_TARGET])],
        )
        self.assertEqual(
            [p.entry.producer_id for p in plan.producers], ["raw", "norm"]
        )
        self.assertEqual(plan.port_sources[norm.id], "norm")

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

    def test_missing_requirement_fails_fast(self) -> None:
        a = _port("tllm", "a", PortKind.RESIDUAL_STREAM)
        missing = PortId("tllm", "ghost", 1)
        reg = ProducerRegistry()
        reg.register(_producer("pa", [a], requires=(missing,)))
        with self.assertRaisesRegex(ExecutionPlanError, "no registered producer offers"):
            compile_execution_plan(
                registry=reg, consumers=[_Consumer([PortKind.RESIDUAL_STREAM])]
            )

    def test_unsupported_hook_fails_fast(self) -> None:
        reg = ProducerRegistry()
        reg.register(
            _producer(
                "pa",
                [_port("tllm", "a", PortKind.RESIDUAL_STREAM)],
                hooks=frozenset({HookPoint.LAYER_POST}),
            )
        )
        with self.assertRaisesRegex(ExecutionPlanError, "hook points"):
            compile_execution_plan(
                registry=reg,
                consumers=[_Consumer([PortKind.RESIDUAL_STREAM])],
                supported_hooks=frozenset({HookPoint.EXECUTE_MODEL_POST}),
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

    def test_enforced_kinds_limits_consumer_check(self) -> None:
        reg = ProducerRegistry()
        reg.register(_producer("pa", [_port("tllm", "a", PortKind.RESIDUAL_STREAM)]))
        # LOGITS has no producer, but is not enforced -> compiles fine.
        plan = compile_execution_plan(
            registry=reg,
            consumers=[_Consumer([PortKind.RESIDUAL_STREAM, PortKind.LOGITS])],
            enforced_kinds=frozenset({PortKind.RESIDUAL_STREAM}),
        )
        self.assertEqual(len(plan.producers), 1)


class RuntimeIntegrationUnitTest(unittest.TestCase):
    def test_default_plan_compiles_for_esamp_consumer(self) -> None:
        from tllm.runtime.execution_plan import compile_default_plan
        from tllm.consumers.esamp.consumer import ESampConsumer
        from tllm.consumers.esamp.config import ESampConsumerConfig

        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=8))
        plan = compile_default_plan(consumers=[consumer])
        self.assertTrue(plan.producers)

    def test_runtime_refresh_compiles_and_stores_plan(self) -> None:
        from tllm.runtime import residual_runtime as rr

        class _ResidualReader:
            consumer_id = "test.residual_reader"

            def flows(self):
                return (
                    ConsumerFlow(
                        reads=[PortRead(kind=PortKind.RESIDUAL_STREAM)],
                        writes=(),
                        window="same_step",
                    ),
                )

        prev_consumer = rr.RUNTIME.consumer
        rr.register_dispatch_consumer(_ResidualReader())
        try:
            self.assertIsNotNone(rr.RUNTIME.execution_plan)
        finally:
            rr.clear_dispatch_consumers()
            rr.RUNTIME.consumer = None
            rr._refresh_dispatch_plan()
        self.assertIsNone(rr.RUNTIME.execution_plan)
        rr.RUNTIME.consumer = prev_consumer
        rr._refresh_dispatch_plan()


if __name__ == "__main__":
    unittest.main()
