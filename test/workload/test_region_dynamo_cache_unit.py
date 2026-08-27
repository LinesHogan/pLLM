#!/usr/bin/env python3
"""Dynamo cache isolation: every materialized Region owns its code identity.

Regression test for the recompile-limit risk: all composed Region callables
are closures over one module-level code object, and Dynamo's recompile cache
is per code object.  Without a fresh code identity per materialization,
materializing more Regions than ``cache_size_limit`` (default 8) in one
process raises ``FailOnRecompileLimitHit`` (or silently degrades Regions to
eager).  The fix gives each materialization its own code object; global
knobs are never touched so the main model's compilation state is safe.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch
import torch._dynamo

from tllm.schedule import (
    CompileOptions,
    MainWorkloadTimeline,
    TimelineAnchor,
    WorkloadCompilerCoordinator,
)
from tllm.workload import (
    Between,
    declare_workload,
    materialize_region_artifacts,
    provided,
    sink,
    task,
    workload,
)

_REAL_TORCH_COMPILE = torch.compile


def _eager_backend_compile(function: object, **kwargs: object) -> object:
    return _REAL_TORCH_COMPILE(function, backend="eager", **kwargs)  # type: ignore[arg-type]


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


def _declaration(index: int):
    @workload(id=f"stress.region.{index}")
    class Probe:
        @task(
            inputs={"value": provided("value")},
            outputs={"result_value": sink("result")},
            timing=Between("model.execute", "step.end"),
        )
        def run(self, value: object) -> object:
            assert isinstance(value, torch.Tensor)
            return value * (index + 1)

    return declare_workload(Probe())


class RegionDynamoCacheUnitTest(unittest.TestCase):
    def test_many_materializations_do_not_share_one_dynamo_cache(self) -> None:
        previous = torch._dynamo.config.fail_on_recompile_limit_hit
        torch._dynamo.config.fail_on_recompile_limit_hit = True
        try:
            with mock.patch("torch.compile", side_effect=_eager_backend_compile):
                for index in range(torch._dynamo.config.cache_size_limit + 4):
                    declaration = _declaration(index)
                    compilation = WorkloadCompilerCoordinator(
                        (declaration,),
                        options=CompileOptions(memory_budget_bytes=1 << 20),
                    ).compile(_timeline())
                    (artifact,) = materialize_region_artifacts(
                        (declaration,), compilation.decision, compilation.options
                    )
                    outputs = artifact.execute(
                        {artifact.input_keys[0]: torch.tensor([2.0])}
                    )
                    torch.testing.assert_close(
                        outputs[artifact.output_keys[0]],
                        torch.tensor([2.0 * (index + 1)]),
                    )
        finally:
            torch._dynamo.config.fail_on_recompile_limit_hit = previous


if __name__ == "__main__":
    unittest.main()
