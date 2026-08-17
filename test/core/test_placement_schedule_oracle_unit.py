#!/usr/bin/env python3
"""Differential test: Rust placement compiler vs the Python oracle.

B1 scope pins the identity schedule plus the full validation matrix; error
messages must match the oracle verbatim. Randomized descriptors use a seeded
PRNG for reproducibility.
"""

from __future__ import annotations

import random
import unittest

from tllm.core import compile_placement_schedule
from test.core.oracles.placement_schedule import (
    DEADLINES,
    GRAPH_MODES,
    HOOK_POINTS,
    MOBILITIES,
    OBSERVATIONS,
    OraclePlacementError,
    oracle_compile_placement_schedule,
)


def _desc(flow_id: str, hook: str, mobility=None, deadline=None, max_defer=None, observes=None) -> dict:
    return {
        "flow_id": flow_id,
        "declared_hook": hook,
        "mobility": mobility,
        "deadline": deadline,
        "max_defer_steps": max_defer,
        "observes_version": observes,
    }


def _rust_outcome(flows, graph_mode):
    try:
        schedule = compile_placement_schedule(flows, graph_mode)
    except RuntimeError as exc:
        return ("err", str(exc))
    return ("ok", [tuple(p) for p in schedule.placements])


def _oracle_outcome(flows, graph_mode):
    try:
        placements = oracle_compile_placement_schedule(flows, graph_mode)
    except OraclePlacementError as exc:
        return ("err", str(exc))
    return ("ok", [tuple(p) for p in placements])


def _assert_same(test_case: unittest.TestCase, flows, graph_mode) -> None:
    rust = _rust_outcome(flows, graph_mode)
    oracle = _oracle_outcome(flows, graph_mode)
    # Rust's multi-line in-capture message uses a line-continuation; normalize
    # whitespace before comparing so the oracle stays readable.
    norm = lambda out: (out[0], " ".join(str(out[1]).split()) if isinstance(out[1], str) else out[1])
    test_case.assertEqual(norm(rust), norm(oracle))


class PlacementScheduleOracleUnitTest(unittest.TestCase):
    def test_identity_schedule_for_valid_combinations(self) -> None:
        valid = [
            _desc("pinned_default", "layer.post"),
            _desc("pinned_explicit", "execute_model.post", mobility="pinned", deadline="step_end", max_defer=0, observes="at_declared_window"),
            _desc("deferrable", "execute_model.post", mobility="deferrable", deadline="next_prepare", max_defer=2, observes="latest_at_execution"),
            _desc("free", "sample_tokens.pre", mobility="free", deadline="generate_end", max_defer=8, observes="stale_ok"),
            _desc("advanceable", "stack.end", mobility="advanceable", deadline="step_end", max_defer=0, observes="stale_ok"),
        ]
        for graph_mode in GRAPH_MODES:
            with self.subTest(graph_mode=graph_mode):
                flows = [d for d in valid if not (graph_mode != "eager" and d["flow_id"] == "advanceable")]
                _assert_same(self, flows, graph_mode)
                outcome, payload = _rust_outcome(flows, graph_mode)
                self.assertEqual(outcome, "ok")
                self.assertEqual(payload, [(d["flow_id"], d["declared_hook"]) for d in flows])

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


if __name__ == "__main__":
    unittest.main()
