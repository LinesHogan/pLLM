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
    result,
    sink,
    task,
    temporary,
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


def _materialized_declaration():
    """Pure chain whose consumer Region reads only a signed materialization.

    The tasks sit in disjoint windows so the planner cannot merge them: the
    consumer Region's only input is the fence-covered cross-region value.
    """

    @workload(id="side.materialized")
    class Pure:
        @task(
            outputs={"value": temporary()},
            timing=Between("model.execute", "sampler.pre_draw"),
        )
        def produce(self) -> object:
            return torch.ones(2)

        @task(
            inputs={"value": result("produce", "value")},
            outputs={"result_value": sink("result")},
            timing=Between("sampler.pre_draw", "step.end"),
        )
        def consume(self, value: object) -> object:
            assert isinstance(value, torch.Tensor)
            return value * 2

    return declare_workload(Pure())


def _provided_declaration():
    """Pure but provided-input: not a signed materialization, never side."""

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
        compilation = self._compile(_materialized_declaration())
        for region in compilation.decision.execution_regions:
            self.assertEqual(region.stream.value, "default")

    def test_materialized_region_is_signed_side_forward_when_enabled(self) -> None:
        compilation = self._compile(
            _materialized_declaration(), allow_side_stream=True
        )
        streams = {
            region.tasks[0]: region.stream.value
            for region in compilation.decision.execution_regions
        }
        # The consumer reads only a fence-covered materialization: side.  The
        # producer has no signed inputs (never capture-safe): default.
        self.assertEqual(streams.get("consume"), "side_forward")
        self.assertEqual(streams.get("produce"), "default")
        assert compilation.physical is not None
        side_nodes = [
            node
            for node in compilation.physical.nodes
            if node.stream is StreamClass.SIDE_FORWARD
        ]
        self.assertEqual(len(side_nodes), 1)

    def test_provided_inputs_keep_default_stream_even_when_enabled(self) -> None:
        # A provided value is host-staged, not a signed materialization: the
        # model cannot prove the main snapshot covers its production.
        compilation = self._compile(
            _provided_declaration(), allow_side_stream=True
        )
        for region in compilation.decision.execution_regions:
            self.assertEqual(region.stream.value, "default")
        assert compilation.physical is not None
        for node in compilation.physical.nodes:
            self.assertIs(node.stream, StreamClass.DEFAULT)

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
            _materialized_declaration(), allow_side_stream=True
        )
        resources = _plan_resources(compilation)
        stream_keys = tuple(resource.key for resource in resources.streams)
        self.assertIn(StreamClass.SIDE_FORWARD, stream_keys)
        if not torch.cuda.is_available():
            # CPU-only hosts keep serial tokens; signed order is unchanged.
            self.assertEqual(len(resources.streams), 1)


if __name__ == "__main__":
    unittest.main()
