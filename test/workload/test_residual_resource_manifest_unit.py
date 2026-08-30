from __future__ import annotations

import hashlib
import json
import unittest

from tllm import _core
from tllm.schedule import (
    CompileOptions,
    HiddenPhase,
    HiddenSite,
    ResidualResourceBinding,
    workload_compilation_cache_key,
)


class ResidualResourceManifestContractTest(unittest.TestCase):
    """Indexed hidden resources are signed scheduling inputs, not worker state."""

    _GRAPH = {
        "workload_key": "residual-resource-manifest",
        "tasks": [
            {
                "task_id": "run",
                "timing": {"after": "model.execute", "before": "step.end"},
                "temporary_bytes": 0,
                "implementation_ids": ["python:run@1"],
                "required_capabilities": [],
                "capture_safe": False,
                "inputs": [
                    {
                        "name": "hidden",
                        "kind": "hidden",
                        "source": "hidden:0:block_output:decode",
                        "output": "",
                        "layer": "0",
                        "layer_kind": "index",
                        "site": "block_output",
                        "phase": "decode",
                    }
                ],
                "outputs": [],
            }
        ],
        "edges": [],
        "values": [],
        "persistent_state_bytes": 0,
        "schema_version": 1,
    }
    _TIMELINE = {
        "backend_id": "residual-resource-manifest",
        "backend_version": "0.27.1",
        "runner_family": "mrv2",
        "graph_mode": "eager",
        "anchors": [
            {
                "anchor_id": "model.execute",
                "ordinal": 0,
                "observable": True,
                "graph_boundary": True,
                "visible_values": [],
            },
            {
                "anchor_id": "step.end",
                "ordinal": 1,
                "observable": True,
                "graph_boundary": True,
                "visible_values": [],
            },
        ],
        "regions": [],
        "static_headroom_bytes": 64,
        "capabilities": ["hidden/block_output/decode"],
        "schema_version": 1,
    }

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @staticmethod
    def _options(path: str) -> CompileOptions:
        return CompileOptions(
            memory_budget_bytes=64,
            residual_resource_manifest=(
                ResidualResourceBinding(
                    layer_index=0,
                    site=HiddenSite.BLOCK_OUTPUT,
                    phase=HiddenPhase.DECODE,
                    module_path=path,
                ),
            ),
        )

    def test_indexed_hidden_requires_exact_signed_manifest(self) -> None:
        graph = self._json(self._GRAPH)
        timeline = self._json(self._TIMELINE)
        unsigned = CompileOptions(memory_budget_bytes=64).to_json()
        with self.assertRaisesRegex(RuntimeError, "residual resource manifest.*missing"):
            _core.compile_schedule(graph, timeline, unsigned)

        signed_options = self._options("model.layers.0").to_json()
        decision = _core.compile_schedule(graph, timeline, signed_options)
        _core.validate_schedule(graph, timeline, signed_options, decision)

    def test_module_path_is_canonical_signed_identity(self) -> None:
        first = self._options("model.layers.0")
        second = self._options("model.layers.alt.0")
        self.assertNotEqual(first.to_json(), second.to_json())
        self.assertNotEqual(
            workload_compilation_cache_key(
                self._schedule_graph(), self._timeline(), first
            ),
            workload_compilation_cache_key(
                self._schedule_graph(), self._timeline(), second
            ),
        )

    def test_worker_local_manifest_must_match_signed_manifest(self) -> None:
        graph = self._json(self._GRAPH)
        timeline = self._json(self._TIMELINE)
        options = self._options("model.layers.0").to_json()
        decision = _core.compile_schedule(graph, timeline, options)
        cache_key = hashlib.sha256(
            f"workload-compilation-v4\0{graph}\0{timeline}\0{options}".encode("ascii")
        ).hexdigest()
        local_manifest = self._json(
            [
                {
                    "layer_index": 0,
                    "site": "block_output",
                    "phase": "decode",
                    "module_path": "model.layers.drifted.0",
                }
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "residual resource manifest does not match"):
            _core.validate_worker_reconstruction_wire(
                cache_key,
                graph,
                timeline,
                options,
                decision,
                1,
                graph,
                "null",
                local_manifest,
                "[]",
            )

    @staticmethod
    def _schedule_graph():
        from tllm.schedule import (
            HiddenLayerKind,
            ScheduleGraph,
            ScheduleInput,
            ScheduleInputKind,
            ScheduleTask,
            TaskTiming,
        )

        return ScheduleGraph(
            workload_key="residual-resource-manifest",
            tasks=(
                ScheduleTask(
                    task_id="run",
                    timing=TaskTiming("model.execute", "step.end"),
                    implementation_ids=("python:run@1",),
                    inputs=(
                        ScheduleInput(
                            name="hidden",
                            kind=ScheduleInputKind.HIDDEN,
                            source="hidden:0:block_output:decode",
                            layer="0",
                            layer_kind=HiddenLayerKind.INDEX,
                            site=HiddenSite.BLOCK_OUTPUT,
                            phase=HiddenPhase.DECODE,
                        ),
                    ),
                ),
            ),
        )

    @staticmethod
    def _timeline():
        from tllm.schedule import MainWorkloadTimeline, TimelineAnchor

        return MainWorkloadTimeline(
            backend_id="residual-resource-manifest",
            backend_version="0.27.1",
            runner_family="mrv2",
            graph_mode="eager",
            anchors=(
                TimelineAnchor("model.execute", 0),
                TimelineAnchor("step.end", 1),
            ),
            regions=(),
            static_headroom_bytes=64,
            capabilities=("hidden/block_output/decode",),
        )


if __name__ == "__main__":
    unittest.main()
