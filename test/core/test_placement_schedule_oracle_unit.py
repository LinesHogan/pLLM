#!/usr/bin/env python3
"""Differential test: Rust placement compiler vs the Python oracle.

B1 scope pins the identity schedule plus the full validation matrix; error
messages must match the oracle verbatim. Randomized descriptors use a seeded
PRNG for reproducibility.
"""

from __future__ import annotations

import random
import unittest
from dataclasses import replace

from tllm.core import compile_placement_schedule
from test.core.oracles.placement_schedule import (
    DEADLINES,
    GRAPH_MODES,
    HOOK_POINTS,
    IN_CAPTURE_HOOKS,
    MOBILITIES,
    OBSERVATIONS,
    OraclePlacementError,
    oracle_compile_placement_schedule,
    oracle_execution_order,
)
from tllm.runtime.placement import (
    PlacementFlowDesc,
    PlacementHookCost,
    PlacementResourceWriteDesc,
    PlacementWriteDesc,
    VersionedStateResourceDesc,
)


def _desc(
    flow_id: str,
    hook: str,
    mobility=None,
    deadline=None,
    max_defer=None,
    observes=None,
    *,
    consumer_id: str = "",
    reads: list[str] | None = None,
    writes: list[dict[str, object]] | None = None,
    resource_reads: tuple[VersionedStateResourceDesc, ...] = (),
    resource_writes: tuple[PlacementResourceWriteDesc, ...] = (),
) -> PlacementFlowDesc:
    return PlacementFlowDesc(
        flow_id=flow_id,
        consumer_id=consumer_id,
        declared_hook=hook,
        mobility=mobility,
        deadline=deadline,
        max_defer_steps=max_defer,
        observes_version=observes,
        reads=tuple(reads or ()),
        writes=tuple(
            PlacementWriteDesc(
                port=str(write["port"]),
                order=int(write["order"]),
                mode=str(write["mode"]),
                fusion_candidate=bool(write["fusion_candidate"]),
            )
            for write in (writes or ())
        ),
        resource_reads=resource_reads,
        resource_writes=resource_writes,
        earliest_hook=hook,
        allowed_hooks=(hook, "sample_tokens.pre", "sampler.pre_draw", "sample_tokens.post")
        if hook == "execute_model.post"
        else (hook,),
    )


def _rust_outcome(flows, graph_mode, hook_costs=None):
    try:
        schedule = compile_placement_schedule(flows, graph_mode, hook_costs)
    except RuntimeError as exc:
        return ("err", str(exc))
    return ("ok", [tuple(p) for p in schedule.placements])


def _oracle_outcome(flows, graph_mode, hook_costs=None):
    try:
        placements = oracle_compile_placement_schedule(flows, graph_mode, hook_costs)
    except OraclePlacementError as exc:
        return ("err", str(exc))
    return ("ok", [tuple(p) for p in placements])


def _assert_same(test_case: unittest.TestCase, flows, graph_mode, hook_costs=None) -> None:
    rust = _rust_outcome(flows, graph_mode, hook_costs)
    oracle = _oracle_outcome(flows, graph_mode, hook_costs)
    # Rust's multi-line in-capture message uses a line-continuation; normalize
    # whitespace before comparing so the oracle stays readable.
    norm = lambda out: (out[0], " ".join(str(out[1]).split()) if isinstance(out[1], str) else out[1])
    test_case.assertEqual(norm(rust), norm(oracle))
    if rust[0] == "ok":
        schedule = compile_placement_schedule(flows, graph_mode, hook_costs)
        test_case.assertEqual(schedule.execution_order, oracle_execution_order(flows))


class PlacementScheduleOracleUnitTest(unittest.TestCase):
    def test_versioned_state_resource_raw_war_waw_match_oracle(self) -> None:
        params_v0 = VersionedStateResourceDesc(
            kind="model_parameters",
            namespace="esamp.shared",
            name="distiller",
            version=0,
        )
        params_v1 = replace(params_v0, version=1)
        params_v2 = replace(params_v0, version=2)
        flows = [
            # Deliberately reverse declaration order: resource versions, not
            # Python list position, define the parameter publication chain.
            _desc(
                "next_sampler_read",
                "execute_model.post",
                resource_reads=(params_v1,),
            ),
            _desc(
                "later_optimizer_write",
                "execute_model.post",
                resource_writes=(PlacementResourceWriteDesc(params_v2),),
            ),
            _desc(
                "optimizer_write",
                "execute_model.post",
                resource_reads=(params_v0,),
                resource_writes=(PlacementResourceWriteDesc(params_v1),),
            ),
            _desc(
                "sampler_read",
                "execute_model.post",
                resource_reads=(params_v0,),
            ),
        ]

        _assert_same(self, flows, "eager")
        self.assertEqual(
            compile_placement_schedule(flows, "eager").execution_order,
            ["sampler_read", "optimizer_write", "next_sampler_read", "later_optimizer_write"],
        )

    def test_duplicate_state_resource_version_writer_is_rejected(self) -> None:
        params_v1 = VersionedStateResourceDesc(
            kind="model_parameters",
            namespace="esamp.shared",
            name="distiller",
            version=1,
        )
        flows = [
            _desc(
                "optimizer_a",
                "execute_model.post",
                resource_writes=(PlacementResourceWriteDesc(params_v1),),
            ),
            _desc(
                "optimizer_b",
                "execute_model.post",
                resource_writes=(PlacementResourceWriteDesc(params_v1),),
            ),
        ]

        _assert_same(self, flows, "eager")
        outcome, message = _rust_outcome(flows, "eager")
        self.assertEqual(outcome, "err")
        self.assertIn("write state resource", message)
        self.assertIn("version 1", message)

    def test_valid_combinations_match_oracle(self) -> None:
        valid = [
            _desc("pinned_default", "layer.post"),
            _desc("pinned_explicit", "execute_model.post", mobility="pinned", deadline="step_end", max_defer=0, observes="at_declared_window"),
            _desc("deferrable", "execute_model.post", mobility="deferrable", deadline="next_prepare", max_defer=2, observes="latest_at_execution"),
            _desc("free", "sample_tokens.pre", mobility="free", deadline="generate_end", max_defer=8, observes="stale_ok"),
            _desc("advanceable", "stack.end", mobility="advanceable", deadline="step_end", max_defer=0, observes="stale_ok"),
        ]
        for graph_mode in GRAPH_MODES:
            with self.subTest(graph_mode=graph_mode):
                flows = [d for d in valid if not (graph_mode != "eager" and d.flow_id == "advanceable")]
                _assert_same(self, flows, graph_mode)
                outcome, payload = _rust_outcome(flows, graph_mode)
                self.assertEqual(outcome, "ok")
                self.assertEqual(payload, oracle_compile_placement_schedule(flows, graph_mode))

    def test_deferrable_flow_moves_to_sampler_bubble(self) -> None:
        flows = [
            _desc(
                "movable",
                "execute_model.post",
                mobility="deferrable",
                deadline="next_prepare",
                max_defer=2,
                observes="latest_at_execution",
                reads=["residual:source"],
            )
        ]

        _assert_same(self, flows, "eager")
        self.assertEqual(_rust_outcome(flows, "eager"), ("ok", [("movable", "sampler.pre_draw")]))

    def test_step_lifetime_device_lease_moves_only_within_sampler_window(self) -> None:
        flow = replace(
            _desc(
                "step_lease",
                "execute_model.post",
                mobility="deferrable",
                deadline="next_prepare",
                max_defer=1,
                observes="at_declared_window",
            ),
            delivery="device_lease",
            ownership="runtime_lease",
            lease_lifetime="step",
            allowed_hooks=(
                "prepare_inputs.post",
                "execute_model.post",
                "sample_tokens.pre",
                "sampler.pre_draw",
                "sample_tokens.post",
            ),
        )

        _assert_same(self, [flow], "eager")
        self.assertEqual(_rust_outcome([flow], "eager"), ("ok", [("step_lease", "sampler.pre_draw")]))

    def test_movable_device_lease_requires_step_lifetime_and_next_prepare(self) -> None:
        base = _desc(
            "lease",
            "execute_model.post",
            mobility="deferrable",
            deadline="next_prepare",
            max_defer=1,
            observes="at_declared_window",
        )
        cases = (
            replace(base, delivery="device_lease", ownership="runtime_lease"),
            replace(
                base,
                delivery="device_lease",
                ownership="runtime_lease",
                lease_lifetime="step",
                deadline="generate_end",
            ),
        )

        for flow in cases:
            with self.subTest(lifetime=flow.lease_lifetime, deadline=flow.deadline):
                _assert_same(self, [flow], "eager")
                self.assertEqual(_rust_outcome([flow], "eager")[0], "err")

    def test_dependency_hazards_pin_work_that_cannot_cross_later_flow(self) -> None:
        write_x = [{"port": "x", "order": 0, "mode": "replace", "fusion_candidate": False}]
        cases = (
            ("raw", ["none"], write_x, ["x"], []),
            ("war", ["x"], [], ["none"], write_x),
            ("waw", ["none"], write_x, ["none"], [{**write_x[0], "order": 1}]),
        )
        for name, reads_a, writes_a, reads_b, writes_b in cases:
            with self.subTest(name=name):
                flows = [
                    _desc(
                        "movable",
                        "execute_model.post",
                        mobility="deferrable",
                        deadline="next_prepare",
                        max_defer=1,
                        observes="latest_at_execution",
                        reads=reads_a,
                        writes=writes_a,
                        consumer_id="a",
                    ),
                    _desc("later", "execute_model.post", reads=reads_b, writes=writes_b, consumer_id="b"),
                ]
                _assert_same(self, flows, "eager")
                self.assertEqual(_rust_outcome(flows, "eager")[1][0], ("movable", "execute_model.post"))

    def test_same_consumer_state_order_pins_earlier_flow(self) -> None:
        flows = [
            _desc(
                "first",
                "execute_model.post",
                mobility="deferrable",
                deadline="next_prepare",
                max_defer=1,
                observes="latest_at_execution",
                consumer_id="stateful",
            ),
            _desc("second", "execute_model.post", consumer_id="stateful"),
        ]

        _assert_same(self, flows, "eager")
        self.assertEqual(_rust_outcome(flows, "eager")[1][0], ("first", "execute_model.post"))

    def test_validation_matrix(self) -> None:
        cases = [
            ("unknown hook", [_desc("f", "nowhere")], "eager"),
            ("unknown mobility", [_desc("f", "layer.post", mobility="teleport")], "eager"),
            ("unknown deadline", [_desc("f", "layer.post", deadline="eventually")], "eager"),
            ("unknown observation", [_desc("f", "layer.post", observes="psychic")], "eager"),
            ("negative defer", [_desc("f", "layer.post", max_defer=-1)], "eager"),
            ("pinned with defer", [_desc("f", "layer.post", mobility="pinned", max_defer=1)], "eager"),
            ("pinned with observation", [_desc("f", "layer.post", mobility="pinned", observes="stale_ok")], "eager"),
            ("deferrable without defer budget", [_desc("f", "execute_model.post", mobility="deferrable", deadline="next_prepare", max_defer=0)], "eager"),
            ("deferrable with step_end", [_desc("f", "execute_model.post", mobility="deferrable", deadline="step_end", max_defer=1)], "eager"),
            ("free with step_end", [_desc("f", "execute_model.post", mobility="free", deadline="step_end", max_defer=1)], "eager"),
            ("advanceable observing declared window", [_desc("f", "layer.pre", mobility="advanceable", observes="at_declared_window")], "eager"),
            ("movable in-capture under piecewise", [_desc("f", "layer.post", mobility="deferrable", deadline="next_prepare", max_defer=1, observes="latest_at_execution")], "piecewise"),
            ("movable in-capture under full", [_desc("f", "stack.end", mobility="free", deadline="generate_end", max_defer=3, observes="stale_ok")], "full"),
            ("movable out-of-capture under full is ok", [_desc("f", "execute_model.post", mobility="deferrable", deadline="next_prepare", max_defer=1, observes="latest_at_execution")], "full"),
        ]
        for name, flows, graph_mode in cases:
            with self.subTest(case=name):
                _assert_same(self, flows, graph_mode)
        with self.assertRaises(RuntimeError):
            compile_placement_schedule([], "cudagraphs-ish")

    def test_seeded_randomized_differential(self) -> None:
        rng = random.Random(20260818)
        vocab = {
            "hook": list(HOOK_POINTS) + ["nope"],
            "mobility": list(MOBILITIES) + ["wild"],
            "deadline": list(DEADLINES) + ["never"],
            "observes": list(OBSERVATIONS) + ["yesterday"],
        }
        for round_i in range(200):
            flows = [
                _desc(
                    f"f{i}",
                    rng.choice(vocab["hook"]),
                    mobility=rng.choice(vocab["mobility"]),
                    deadline=rng.choice(vocab["deadline"]),
                    max_defer=rng.randint(-1, 4),
                    observes=rng.choice(vocab["observes"]),
                )
                for i in range(rng.randint(0, 4))
            ]
            graph_mode = rng.choice(list(GRAPH_MODES) + ["bogus"])
            with self.subTest(round=round_i):
                _assert_same(self, flows, graph_mode)

    def test_seeded_full_hook_dag_and_cost_differential(self) -> None:
        rng = random.Random(20260819)
        ports = ("x", "y", "z")
        resources = (
            VersionedStateResourceDesc("model_parameters", "esamp.shared", "distiller", 0),
            VersionedStateResourceDesc("optimizer_state", "esamp.shared", "adamw", 0),
        )
        for round_i in range(300):
            flows = []
            for flow_i in range(rng.randint(0, 6)):
                mobility = rng.choice(MOBILITIES)
                declared = rng.choice(HOOK_POINTS)
                earliest = rng.choice(HOOK_POINTS)
                movable_late = mobility in ("deferrable", "free")
                reads = tuple(port for port in ports if rng.random() < 0.25)
                writes = tuple(
                    PlacementWriteDesc(
                        port=port,
                        order=rng.randint(0, 3),
                        mode=rng.choice(("replace", "add")),
                        fusion_candidate=rng.random() < 0.15,
                    )
                    for port in ports
                    if rng.random() < 0.2
                )
                allowed = tuple(hook for hook in HOOK_POINTS if rng.random() < 0.35)
                flows.append(
                    PlacementFlowDesc(
                        flow_id=f"r{round_i}.f{flow_i}",
                        consumer_id=rng.choice(("", "a", "b")),
                        declared_hook=declared,
                        mobility=mobility,
                        deadline=rng.choice(("next_prepare", "generate_end")) if movable_late else "step_end",
                        max_defer_steps=rng.randint(1, 3) if movable_late else 0,
                        observes_version=(
                            "at_declared_window"
                            if mobility == "pinned"
                            else rng.choice(("latest_at_execution", "stale_ok"))
                        ),
                        reads=reads,
                        writes=writes,
                        delivery="device_lease",
                        ownership="runtime_lease",
                        max_bundle_rows=16,
                        effect_hook=rng.choice(IN_CAPTURE_HOOKS),
                        earliest_hook=earliest,
                        allowed_hooks=allowed,
                        resource_reads=tuple(
                            replace(resource, version=rng.randint(0, 3))
                            for resource in resources
                            if rng.random() < 0.2
                        ),
                        resource_writes=tuple(
                            PlacementResourceWriteDesc(
                                replace(resource, version=rng.randint(0, 3))
                            )
                            for resource in resources
                            if rng.random() < 0.2
                        ),
                    )
                )
            hook_costs = [
                PlacementHookCost(hook=hook, bubble_score=rng.randint(-50, 250))
                for hook in rng.sample(list(HOOK_POINTS), rng.randint(0, 5))
            ]
            graph_mode = rng.choice(GRAPH_MODES)
            with self.subTest(round=round_i):
                _assert_same(self, flows, graph_mode, hook_costs)


if __name__ == "__main__":
    unittest.main()
