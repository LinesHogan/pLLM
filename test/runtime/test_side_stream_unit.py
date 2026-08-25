from __future__ import annotations

import unittest

import torch

from tllm.runtime.workload_execution import _plan_resources
from tllm.schedule import (
    CompileOptions,
    MainWorkloadTimeline,
    TimelineAnchor,
    WorkloadCompilerCoordinator,
)
from tllm.workgraph.physical import StreamClass
from tllm.workload import (
    AdamWSpec,
    Before,
    Between,
    State,
    StateScope,
    declare_workload,
    provided,
    read,
    request_key,
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
            TimelineAnchor("sampler.pre_draw", 1),
            TimelineAnchor("step.end", 2),
        ),
        regions=(),
        static_headroom_bytes=64,
    )


def _effect_free_declaration():
    @workload(id="side.effect-free")
    class Pure:
        @task(
            inputs={"value": provided("value")},
            outputs={"result_value": sink("result")},
            timing=Between("model.execute", "step.end"),
        )
        def run(self, value: object) -> object:
            assert isinstance(value, torch.Tensor)
            return value * 2

    return declare_workload(Pure())


def _stateful_declaration():
    @workload(id="side.stateful")
    class Stateful:
        bank = State(
            scope=StateScope.REQUEST,
            key=request_key("request_id"),
            parameters="model",
            optimizer=AdamWSpec(lr=1e-3),
        )

        @task(
            inputs={"value": provided("value")},
            outputs={"result_value": sink("result")},
            effects=(read(bank),),
            timing=Between("model.execute", "step.end"),
        )
        def run(self, value: object) -> object:
            assert isinstance(value, torch.Tensor)
            return value * 2

    return declare_workload(Stateful())


class SideStreamDecisionUnitTest(unittest.TestCase):
    def _compile(self, declaration, **option_overrides):
        options = CompileOptions(
            memory_budget_bytes=1 << 20, **option_overrides
        )
        return WorkloadCompilerCoordinator((declaration,), options=options).compile(
            _timeline()
        )

    def test_effect_free_region_stays_on_default_stream_by_default(self) -> None:
        compilation = self._compile(_effect_free_declaration())
        region = compilation.decision.execution_regions[0]
        self.assertEqual(region.stream.value, "default")

    def test_effect_free_region_is_signed_side_forward_when_enabled(self) -> None:
        compilation = self._compile(
            _effect_free_declaration(), allow_side_stream=True
        )
        region = compilation.decision.execution_regions[0]
        self.assertEqual(region.stream.value, "side_forward")
        assert compilation.physical is not None
        self.assertIs(compilation.physical.nodes[0].stream, StreamClass.SIDE_FORWARD)

    def test_state_effects_keep_default_stream_even_when_enabled(self) -> None:
        compilation = self._compile(
            _stateful_declaration(), allow_side_stream=True
        )
        for region in compilation.decision.execution_regions:
            self.assertEqual(region.stream.value, "default")
        assert compilation.physical is not None
        for node in compilation.physical.nodes:
            self.assertIs(node.stream, StreamClass.DEFAULT)

    def test_plan_resources_binds_side_stream_without_cuda(self) -> None:
        compilation = self._compile(
            _effect_free_declaration(), allow_side_stream=True
        )
        resources = _plan_resources(compilation)
        stream_keys = tuple(resource.key for resource in resources.streams)
        self.assertIn(StreamClass.SIDE_FORWARD, stream_keys)
        if not torch.cuda.is_available():
            # CPU-only hosts keep serial tokens; signed order is unchanged.
            self.assertEqual(len(resources.streams), 1)


if __name__ == "__main__":
    unittest.main()
