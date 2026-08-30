#!/usr/bin/env python3
"""Disk-persisted workload decision cache: key stability and fail-closed load."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tllm.schedule import (
    CompileOptions,
    MainWorkloadTimeline,
    PlanEpochAllocator,
    TimelineAnchor,
    WorkloadCompilerCoordinator,
    WorkloadDecisionCacheError,
    workload_compilation_cache_key,
)
from tllm.workload import (
    Between,
    declare_workload,
    provided,
    sink,
    task,
    workload,
)


def _timeline() -> MainWorkloadTimeline:
    return MainWorkloadTimeline(
        backend_id="vllm",
        backend_version="0.27.1",
        runner_family="mrv2",
        graph_mode="eager",
        anchors=(
            TimelineAnchor("model.execute", 0),
            TimelineAnchor("step.end", 1),
        ),
        regions=(),
        static_headroom_bytes=64,
    )


def _declaration(workload_id: str = "cache.probe"):
    @workload(id=workload_id)
    class Probe:
        @task(
            inputs={"value": provided("value")},
            outputs={"result_value": sink("result")},
            timing=Between("model.execute", "step.end"),
        )
        def run(self, value: object) -> object:
            return value

    return declare_workload(Probe())


def _coordinator(
    cache_directory: str | None,
    epoch_allocator: PlanEpochAllocator | None = None,
) -> WorkloadCompilerCoordinator:
    return WorkloadCompilerCoordinator(
        (_declaration(),),
        options=CompileOptions(memory_budget_bytes=1 << 20),
        epoch_allocator=epoch_allocator,
        cache_directory=cache_directory,
    )


class DecisionCacheUnitTest(unittest.TestCase):
    def test_compile_persists_and_second_instance_loads_without_recompiling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            # One runtime-owned allocator shared across coordinator rebuilds,
            # mirroring TllmRuntime's epoch ownership.
            allocator = PlanEpochAllocator()
            first = _coordinator(directory, allocator).compile(_timeline())
            entries = [entry.name for entry in Path(directory).glob("*.json")]
            self.assertEqual(entries, [f"{first.cache_key}.json"])

            # A fresh coordinator must serve the same decision from disk
            # without invoking the compiler.
            second = _coordinator(directory, allocator)
            with mock.patch(
                "tllm.schedule.coordinator.compile_schedule",
                side_effect=AssertionError("disk hit recompiled the schedule"),
            ):
                loaded = second.compile(_timeline())
            self.assertEqual(loaded.cache_key, first.cache_key)
            self.assertEqual(loaded.decision.signature, first.decision.signature)
            # The reload re-lowers the physical plan under a fresh plan epoch.
            self.assertIsNotNone(loaded.physical)
            self.assertNotEqual(
                loaded.physical.plan_epoch, first.physical.plan_epoch  # type: ignore[union-attr]
            )

    def test_cache_key_is_unchanged_by_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coordinator = _coordinator(directory)
            expected = workload_compilation_cache_key(
                coordinator.graph, _timeline(), coordinator.options
            )
            self.assertEqual(coordinator.compile(_timeline()).cache_key, expected)

    def test_tampered_decision_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _coordinator(directory).compile(_timeline())
            path = Path(directory) / f"{first.cache_key}.json"
            payload = json.loads(path.read_text(encoding="ascii"))
            decision = json.loads(payload["decision"])
            decision["signature"] = "0" * 64
            payload["decision"] = json.dumps(
                decision, sort_keys=True, separators=(",", ":")
            )
            path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                encoding="ascii",
            )
            with self.assertRaises(WorkloadDecisionCacheError):
                _coordinator(directory).compile(_timeline())

    def test_mismatched_embedded_key_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _coordinator(directory).compile(_timeline())
            path = Path(directory) / f"{first.cache_key}.json"
            payload = json.loads(path.read_text(encoding="ascii"))
            payload["cache_key"] = "f" * 64
            path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                encoding="ascii",
            )
            with self.assertRaises(WorkloadDecisionCacheError):
                _coordinator(directory).compile(_timeline())

    def test_stale_persisted_schema_requires_a_fresh_cache_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _coordinator(directory).compile(_timeline())
            path = Path(directory) / f"{first.cache_key}.json"
            payload = json.loads(path.read_text(encoding="ascii"))
            payload["schema_version"] = 1
            path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                WorkloadDecisionCacheError,
                "unsupported workload decision cache schema",
            ):
                _coordinator(directory).compile(_timeline())

    def test_malformed_entry_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coordinator = _coordinator(directory)
            cache_key = workload_compilation_cache_key(
                coordinator.graph, _timeline(), coordinator.options
            )
            (Path(directory) / f"{cache_key}.json").write_text(
                "{not json", encoding="ascii"
            )
            with self.assertRaises(WorkloadDecisionCacheError):
                coordinator.compile(_timeline())

    def test_unknown_timeline_is_a_miss_not_a_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coordinator = _coordinator(directory)
            coordinator.compile(_timeline())
            wider = MainWorkloadTimeline(
                backend_id="vllm",
                backend_version="0.27.1",
                runner_family="mrv2",
                graph_mode="eager",
                anchors=(
                    TimelineAnchor("model.execute", 0),
                    TimelineAnchor("step.end", 1),
                ),
                regions=(),
                static_headroom_bytes=128,
            )
            coordinator.compile(wider)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 2)

    def test_no_directory_keeps_cache_process_local(self) -> None:
        coordinator = _coordinator(None)
        compilation = coordinator.compile(_timeline())
        self.assertIs(coordinator.compile(_timeline()), compilation)

    def test_cache_directory_validation(self) -> None:
        with self.assertRaises(TypeError):
            _coordinator("")
        with self.assertRaises(TypeError):
            _coordinator(123)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
