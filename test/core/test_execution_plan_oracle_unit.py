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
    prod_dicts = [
        {
            "id": p.id,
            "priority": p.priority,
            "outputs": [
                {
                    "port": str(spec.id),
                    "kind": spec.kind.value,
                    "writable": spec.writable,
                    "write_modes": list(spec.write_modes),
                }
                for spec in p.outputs
            ],
            "requires": [str(r) for r in p.requires],
            "hook_points": [h.value for h in p.hook_points],
            "graph_safety": p.graph_safety.name,
        }
        for p in producers
    ]
    cons_dicts = []
    for c in consumers:
        flow = c.flows()[0]
        cons_dicts.append(
            {
                "id": c.consumer_id,
                "reads": [r.kind.value for r in flow.reads],
                "writes": [{"kind": w.kind.value, "mode": w.mode} for w in flow.writes],
            }
        )
    return _core.compile_execution_plan(
        prod_dicts,
        cons_dicts,
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
        rust_result = (plan.ordered_producer_ids, dict(plan.port_sources))
    except RuntimeError as exc:
        rust_error = str(exc)
    try:
        ordered, sources = _oracle_call(producers, consumers, **kwargs)
        oracle_result = (ordered, sources)
    except OracleExecutionPlanError as exc:
        oracle_error = str(exc)
    if oracle_error is not None:
        test_case.assertIsNotNone(rust_error, "oracle failed but Rust succeeded")
        test_case.assertEqual(rust_error, oracle_error)
    else:
        test_case.assertIsNone(rust_error, f"Rust failed but oracle succeeded: {rust_error}")
        test_case.assertEqual(rust_result, oracle_result)


class ExecutionPlanOracleMatrixUnitTest(unittest.TestCase):
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
        for trial in range(60):
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
