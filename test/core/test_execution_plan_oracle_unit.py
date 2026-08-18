#!/usr/bin/env python3
"""Differential matrix: Rust ``tllm._core.compile_execution_plan`` vs the
legacy Python oracle (``test/core/oracles/execution_plan.py``).

The oracle is the verbatim legacy implementation; any divergence blocks the
oracle's sunset. Randomized DAGs use a seeded PRNG for reproducibility.
"""

from __future__ import annotations

import random
import unittest

from tllm import _core
from tllm.producers.ir import (
    PlanConsumerDesc,
    PlanConsumerFlowDesc,
    PlanConsumerReadDesc,
    PlanConsumerWriteDesc,
    PlanPortSpecDesc,
    PlanProducerDesc,
)

from test.core.oracles.execution_plan import (
    OracleExecutionPlanError,
    consumer_desc,
    oracle_compile_execution_plan,
    producer_desc,
)

_KINDS = ["residual_stream", "logits", "sampler", "request_meta", "generation_trace"]
_HOOKS = ["prepare_inputs.post", "execute_model.post", "sample_tokens.post"]
_SAFETIES = ["EAGER_ONLY", "CAPTURE_SAFE", "REPLAY_STATEFUL"]


def _rust_call(producers, consumers, *, enforced_kinds=None, supported_hooks=None, backend=None):
    producer_descs = [
        PlanProducerDesc(
            id=p.id,
            priority=p.priority,
            outputs=tuple(
                PlanPortSpecDesc(
                    port=str(spec.id),
                    kind=spec.kind.value,
                    writable=spec.writable,
                    write_modes=tuple(spec.write_modes),
                    phases=tuple(spec.phases),
                )
                for spec in p.outputs
            ),
            requires=tuple(str(required) for required in p.requires),
            hook_points=tuple(hook.value for hook in p.hook_points),
            graph_safety=p.graph_safety.name,
        )
        for p in producers
    ]
    consumer_descs = []
    for c in consumers:
        consumer_descs.append(
            PlanConsumerDesc(
                id=c.consumer_id,
                flows=tuple(
                    PlanConsumerFlowDesc(
                        reads=tuple(
                            PlanConsumerReadDesc(
                                kind=read.kind.value,
                                role=str(getattr(read, "role", "")),
                                phase=str(getattr(read, "phase", "")),
                            )
                            for read in flow.reads
                        ),
                        writes=tuple(
                            PlanConsumerWriteDesc(
                                kind=write.kind.value,
                                mode=write.mode,
                                phase=str(getattr(write, "phase", "")),
                            )
                            for write in flow.writes
                        ),
                        bundle_key=tuple(getattr(flow, "bundle_key", ())),
                    )
                    for flow in c.flows()
                ),
            )
        )
    return _core.compile_execution_plan(
        producer_descs,
        consumer_descs,
        enforced_kinds=None if enforced_kinds is None else [k for k in enforced_kinds],
        supported_hooks=None if supported_hooks is None else [h for h in supported_hooks],
        backend=backend,
    )


def _oracle_call(producers, consumers, *, enforced_kinds=None, supported_hooks=None, backend=None):
    from tllm.ports.base import PortKind
    from tllm.producers.base import HookPoint

    return oracle_compile_execution_plan(
        producers=producers,
        consumers=consumers,
        enforced_kinds=None if enforced_kinds is None else frozenset(PortKind(k) for k in enforced_kinds),
        supported_hooks=None if supported_hooks is None else frozenset(HookPoint(h) for h in supported_hooks),
        backend=backend,
    )


def _assert_same(test_case, producers, consumers, **kwargs) -> None:
    rust_result = rust_error = None
    oracle_result = oracle_error = None
    try:
        plan = _rust_call(producers, consumers, **kwargs)
        rust_result = (
            plan.ordered_producer_ids,
            dict(plan.port_sources),
            plan.capture_resident_producer_ids,
        )
    except RuntimeError as exc:
        rust_error = str(exc)
    try:
        ordered, sources, capture_resident = _oracle_call(producers, consumers, **kwargs)
        oracle_result = (ordered, sources, capture_resident)
    except OracleExecutionPlanError as exc:
        oracle_error = str(exc)
    if oracle_error is not None:
        test_case.assertIsNotNone(rust_error, "oracle failed but Rust succeeded")
        test_case.assertEqual(rust_error, oracle_error)
    else:
        test_case.assertIsNone(rust_error, f"Rust failed but oracle succeeded: {rust_error}")
        test_case.assertEqual(rust_result, oracle_result)


class ExecutionPlanOracleMatrixUnitTest(unittest.TestCase):
    def test_phase_specific_read_selects_only_matching_producer(self) -> None:
        producers = [
            producer_desc(
                id="p.decode",
                outputs=[{
                    "port": "tllm:residual.decode@1",
                    "kind": "residual_stream",
                    "phases": ["decode"],
                }],
            ),
            producer_desc(
                id="p.prefill",
                outputs=[{
                    "port": "tllm:residual.prefill@1",
                    "kind": "residual_stream",
                    "phases": ["prefill"],
                }],
            ),
        ]
        consumers = [
            consumer_desc(
                id="c.decode",
                reads=[{"kind": "residual_stream", "phase": "decode"}],
            )
        ]

        _assert_same(self, producers, consumers)
        plan = _rust_call(producers, consumers)
        self.assertEqual(plan.ordered_producer_ids, ["p.decode"])

    def test_write_validation_respects_phase(self) -> None:
        producers = [
            producer_desc(
                id="p.prefill",
                outputs=[{
                    "port": "tllm:residual.prefill@1",
                    "kind": "residual_stream",
                    "writable": True,
                    "write_modes": ["replace"],
                    "phases": ["prefill"],
                }],
            )
        ]
        consumers = [
            consumer_desc(
                id="c.decode",
                writes=[{
                    "kind": "residual_stream",
                    "mode": "replace",
                    "phase": "decode",
                }],
            )
        ]

        _assert_same(self, producers, consumers)

    def test_capture_resident_ids_require_layer_post_and_graph_safety(self) -> None:
        producers = [
            producer_desc(
                id="p.capture",
                outputs=[{"port": "tllm:a@1", "kind": "residual_stream"}],
                hook_points=["layer.post"],
                graph_safety="CAPTURE_SAFE",
            ),
            producer_desc(
                id="p.eager",
                outputs=[{"port": "tllm:b@1", "kind": "residual_stream"}],
                hook_points=["layer.post"],
                graph_safety="EAGER_ONLY",
            ),
            producer_desc(
                id="p.host",
                outputs=[{"port": "tllm:c@1", "kind": "residual_stream"}],
                hook_points=["execute_model.post"],
                graph_safety="REPLAY_STATEFUL",
            ),
        ]
        consumers = [consumer_desc(id="c0", reads=["residual_stream"])]

        _assert_same(self, producers, consumers)
        plan = _rust_call(producers, consumers)
        self.assertEqual(plan.capture_resident_producer_ids, ["p.capture"])

    def test_rust_rejects_duplicate_bundle_fields_from_typed_ir(self) -> None:
        consumer = PlanConsumerDesc(
            id="duplicate.fields",
            flows=(
                PlanConsumerFlowDesc(
                    reads=(
                        PlanConsumerReadDesc(kind="logits", role=""),
                        PlanConsumerReadDesc(kind="logits", role=""),
                    ),
                    writes=(),
                    bundle_key=(),
                ),
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "duplicate bundle field 'logits'"):
            _core.compile_execution_plan([], [consumer])

    def test_rust_rejects_unknown_bundle_key_from_typed_ir(self) -> None:
        consumer = PlanConsumerDesc(
            id="bad.key",
            flows=(
                PlanConsumerFlowDesc(
                    reads=(),
                    writes=(),
                    bundle_key=("engine_setp_id",),
                ),
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "unknown bundle_key field 'engine_setp_id'"):
            _core.compile_execution_plan([], [consumer])

    def test_fixed_scenarios(self) -> None:
        p_res = producer_desc(
            id="p.residual",
            outputs=[{"port": "tllm:residual.decode@1", "kind": "residual_stream",
                      "writable": True, "write_modes": ["replace", "add"]}],
            hook_points=["prepare_inputs.post"],
        )
        p_logits = producer_desc(
            id="p.logits",
            outputs=[{"port": "tllm:logits@1", "kind": "logits"}],
            requires=["tllm:residual.decode@1"],
        )
        c_read = consumer_desc(id="c.read", reads=["residual_stream"])
        c_write = consumer_desc(id="c.write", writes=[{"kind": "residual_stream", "mode": "add"}])

        cases = [
            ([p_res], [c_read], {}),
            ([p_res, p_logits], [c_read], {}),
            ([p_res], [c_read, c_write], {}),
            ([p_res], [c_write], {}),
            ([p_res], [consumer_desc(id="c.bad", writes=[{"kind": "residual_stream", "mode": "append"}])], {}),
            ([p_res], [consumer_desc(id="c.miss", reads=["kv_cache"])], {}),
            ([p_res, p_logits], [c_read], {"enforced_kinds": ["logits"]}),
            ([p_res], [c_read], {"supported_hooks": ["execute_model.post"]}),
            ([p_res], [c_read], {"backend": ("be0", "EAGER_ONLY")}),
            ([producer_desc(id="p.eager", outputs=[{"port": "tllm:residual.decode@1", "kind": "residual_stream"}],
                            graph_safety="EAGER_ONLY")],
             [c_read], {"backend": ("be0", "EAGER_ONLY")}),
        ]
        for producers, consumers, kwargs in cases:
            with self.subTest(producers=[p.id for p in producers], consumers=[c.consumer_id for c in consumers], **kwargs):
                _assert_same(self, producers, consumers, **kwargs)

    def test_priority_tie_error_message(self) -> None:
        producers = [
            producer_desc(id="p.b", outputs=[{"port": "tllm:x@1", "kind": "logits"}]),
            producer_desc(id="p.a", outputs=[{"port": "tllm:x@1", "kind": "logits"}]),
        ]
        _assert_same(self, producers, [consumer_desc(id="c0", reads=["logits"])])

    def test_transitive_requirement_uses_highest_priority_exact_provider(self) -> None:
        producers = [
            producer_desc(
                id="p.source.low",
                priority=1,
                outputs=[{"port": "tllm:x@1", "kind": "hidden_target"}],
            ),
            producer_desc(
                id="p.source.high",
                priority=5,
                outputs=[{"port": "tllm:x@1", "kind": "hidden_target"}],
            ),
            producer_desc(
                id="p.sink",
                outputs=[{"port": "tllm:y@1", "kind": "logits"}],
                requires=["tllm:x@1"],
            ),
        ]

        _assert_same(self, producers, [consumer_desc(id="c0", reads=["logits"])])
        plan = _rust_call(producers, [consumer_desc(id="c0", reads=["logits"])])
        self.assertEqual(plan.ordered_producer_ids, ["p.source.high", "p.sink"])
        self.assertEqual(dict(plan.port_sources)["tllm:x@1"], "p.source.high")

    def test_transitive_requirement_rejects_equal_priority_providers(self) -> None:
        producers = [
            producer_desc(
                id="p.source.b",
                outputs=[{"port": "tllm:x@1", "kind": "hidden_target"}],
            ),
            producer_desc(
                id="p.source.a",
                outputs=[{"port": "tllm:x@1", "kind": "hidden_target"}],
            ),
            producer_desc(
                id="p.sink",
                outputs=[{"port": "tllm:y@1", "kind": "logits"}],
                requires=["tllm:x@1"],
            ),
        ]

        _assert_same(self, producers, [consumer_desc(id="c0", reads=["logits"])])

    def test_cycle_detection(self) -> None:
        producers = [
            producer_desc(id="p.a", outputs=[{"port": "tllm:a@1", "kind": "logits"}],
                          requires=["tllm:b@1"]),
            producer_desc(id="p.b", outputs=[{"port": "tllm:b@1", "kind": "sampler"}],
                          requires=["tllm:a@1"]),
        ]
        _assert_same(self, producers, [consumer_desc(id="c0", reads=["logits", "sampler"])])

    def test_randomized_dags(self) -> None:
        rng = random.Random(20260817)
        for trial in range(8):
            n_producers = rng.randint(1, 6)
            ports = [f"tllm:p{i}@1" for i in range(n_producers)]
            producers = []
            for i in range(n_producers):
                # DAG by construction: producer i may only require ports < i.
                requires = [p for p in ports[:i] if rng.random() < 0.4]
                kind = rng.choice(_KINDS)
                producers.append(
                    producer_desc(
                        id=f"p{i}",
                        priority=rng.choice([0, 0, 1]),
                        outputs=[{
                            "port": ports[i],
                            "kind": kind,
                            "writable": rng.random() < 0.5,
                            "write_modes": ["replace"] if rng.random() < 0.5 else [],
                        }] + ([{
                            "port": f"tllm:p{i}b@1",
                            "kind": rng.choice(_KINDS),
                        }] if rng.random() < 0.3 else []),
                        requires=requires,
                        hook_points=[h for h in _HOOKS if rng.random() < 0.7] or ["execute_model.post"],
                        graph_safety=rng.choice(_SAFETIES),
                    )
                )
            consumers = []
            for j in range(rng.randint(1, 3)):
                reads = rng.sample(_KINDS, rng.randint(1, 2))
                writes = ([{"kind": rng.choice(_KINDS), "mode": rng.choice(["", "replace", "add"])}]
                          if rng.random() < 0.4 else [])
                consumers.append(consumer_desc(id=f"c{j}", reads=reads, writes=writes))
            kwargs = {}
            if rng.random() < 0.5:
                kwargs["enforced_kinds"] = rng.sample(_KINDS, rng.randint(1, 3))
            if rng.random() < 0.3:
                kwargs["supported_hooks"] = rng.sample(_HOOKS, rng.randint(1, 3))
            if rng.random() < 0.3:
                kwargs["backend"] = ("be.rand", rng.choice(_SAFETIES))
            with self.subTest(trial=trial):
                _assert_same(self, producers, consumers, **kwargs)


if __name__ == "__main__":
    unittest.main()
