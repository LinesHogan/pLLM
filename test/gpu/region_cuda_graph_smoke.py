#!/usr/bin/env python3
"""Smoke test for lazy CUDA Graph capture of one compiled ExecutionRegion."""

from __future__ import annotations

import json
from typing import cast
from unittest import mock

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.contracts.hidden_locator import (
    HiddenLocatorCapability,
    HiddenPhase,
    HiddenSite,
)
from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime import residual_runtime
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_execution import WorkloadExecutionBindings
from tllm.schedule import (
    CompileOptions,
    ResidualResourceBinding,
    WorkloadCompilerCoordinator,
)
from tllm.workload import Between, declare_workload, hidden, sink, task, workload


@workload(id="gpu.region-cudagraph-smoke")
class CapturedRegion:
    @task(
        inputs={"value": hidden(0, "block_output", "decode")},
        outputs={"output": sink("answer")},
        timing=Between("model.execute", "step.end"),
    )
    def run(self, value: torch.Tensor) -> torch.Tensor:
        return value * 2 + 1


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    declaration = declare_workload(CapturedRegion())
    timeline = export_main_workload_timeline(
        BackendCapabilities(
            backend_id="vllm-0.27-mrv2",
            runner_family="mrv2",
            sampler_stages=frozenset(),
            native_min_p=False,
            tuple_sampler_return=False,
            split_execute_and_sample=True,
            graph_safety=GraphSafety.CAPTURE_SAFE,
            hidden_locators=frozenset(
                {HiddenLocatorCapability.BLOCK_OUTPUT_DECODE}
            ),
        ),
        graph_mode="piecewise",
        static_headroom_bytes=1 << 30,
    )
    compilation = WorkloadCompilerCoordinator(
        (declaration,),
        options=CompileOptions(
            memory_budget_bytes=1 << 30,
            compile_policy="torch_compile",
            residual_resource_manifest=(
                ResidualResourceBinding(
                    layer_index=0,
                    site=HiddenSite.BLOCK_OUTPUT,
                    phase=HiddenPhase.DECODE,
                    module_path="model.layers.0",
                ),
            ),
        ),
    ).compile(timeline)
    region = compilation.decision.execution_regions[0]
    if not region.capture_safe:
        raise AssertionError("Rust planner did not mark the typed Tensor Region capture-safe")

    state = ResidualRuntimeState()
    locator = ResidualLocator(layer=0, site="block_output", phase="decode")
    path = "model.layers.0"
    state.residual_bindings = {path: ResidualPathBinding(locator, path)}
    outputs: list[torch.Tensor] = []
    bindings = WorkloadExecutionBindings(
        declarations=(declaration,),
        output_sinks={
            "answer": lambda value, _step_id: outputs.append(
                cast(torch.Tensor, value).detach().clone()
            )
        },
        residual_paths={locator: path},
    )

    with mock.patch.object(residual_runtime, "RUNTIME", state):
        residual_runtime.publish_workload_compilation(compilation, bindings=bindings)
        execution = state.workload_execution
        assert execution is not None
        try:
            first = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4)
            state.tap_decode_hidden[path] = first
            state.decode_count = 3
            residual_runtime.prepare_physical_workgraph_step(1)
            residual_runtime.advance_workload_anchor(region.placement_anchor)
            torch.cuda.synchronize(device)
            torch.testing.assert_close(outputs[-1], first * 2 + 1)

            second = first + 10
            state.tap_decode_hidden[path] = second
            residual_runtime.prepare_physical_workgraph_step(2)
            residual_runtime.advance_workload_anchor(region.placement_anchor)
            torch.cuda.synchronize(device)
            torch.testing.assert_close(outputs[-1], second * 2 + 1)

            state.tap_decode_hidden[path] = second[:2]
            state.decode_count = 2
            residual_runtime.prepare_physical_workgraph_step(3)
            try:
                residual_runtime.advance_workload_anchor(region.placement_anchor)
            except RuntimeError as error:
                if "shape specialization miss" not in str(error):
                    raise
            else:
                raise AssertionError("shape specialization miss did not fail closed")
        finally:
            execution.close()

    print(
        json.dumps(
            {
                "artifact_key": region.artifact_key,
                "outputs": len(outputs),
                "replay_count": execution.executor.stats.replay_count,
                "shape_miss_rejected": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
