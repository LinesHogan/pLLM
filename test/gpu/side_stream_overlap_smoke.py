#!/usr/bin/env python3
"""GPU smoke: a signed side-stream Region overlaps busy main-stream work.

Compiles a two-Region pure workload (producer on the default stream, consumer
signed ``side_forward``), then proves on hardware:

- the consumer Region executes on a real non-default CUDA stream;
- its kernels complete while the main stream is still busy with a large
  matmul chain enqueued *after* the consumer's advance (legal overlap: the
  side stream only waits the main snapshot recorded at its advance);
- the join at step retirement orders the main stream after side work, so the
  published sink is numerically correct after host synchronization.
"""

from __future__ import annotations

import time
from typing import cast
from unittest import mock

import torch

from tllm.runtime import residual_runtime
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_execution import WorkloadExecutionBindings
from tllm.runtime.workload_bindings import RuntimeOutputBinding, RuntimePortRegistry
from tllm.schedule import (
    CompileOptions,
    MainWorkloadTimeline,
    TimelineAnchor,
    WorkloadCompilerCoordinator,
)
from tllm.workload import (
    Between,
    declare_workload,
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


class _Observer:
    """Out-of-band recorder the tasks write stream/timing evidence into."""

    def __init__(self) -> None:
        self.producer_on_default: bool | None = None
        self.consumer_on_side: bool | None = None
        self.side_done: torch.cuda.Event | None = None
        self.outputs: list[tuple[int, torch.Tensor]] = []


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    observer = _Observer()

    @workload(id="gpu.side-stream-overlap-smoke")
    class SideChain:
        @task(
            outputs={"value": temporary()},
            timing=Between("model.execute", "sampler.pre_draw"),
        )
        def produce(self) -> torch.Tensor:
            observer.producer_on_default = (
                torch.cuda.current_stream().cuda_stream
                == torch.cuda.default_stream().cuda_stream
            )
            return torch.full((1024, 1024), 3.0, device="cuda")

        @task(
            inputs={"value": result("produce", "value")},
            outputs={"output": sink("answer")},
            timing=Between("sampler.pre_draw", "step.end"),
        )
        def consume(self, value: torch.Tensor) -> torch.Tensor:
            observer.consumer_on_side = (
                torch.cuda.current_stream().cuda_stream
                != torch.cuda.default_stream().cuda_stream
            )
            output = value * 2
            done = torch.cuda.Event()
            done.record()
            observer.side_done = done
            return output

    declaration = declare_workload(SideChain())
    compilation = WorkloadCompilerCoordinator(
        (declaration,),
        options=CompileOptions(
            memory_budget_bytes=1 << 20,
            compile_policy="eager",
            allow_side_stream=True,
        ),
    ).compile(_timeline())
    regions = compilation.decision.execution_regions
    assert len(regions) == 2, f"expected two Regions, got {len(regions)}"
    by_task = {region.tasks[0]: region for region in regions}
    consume_region = by_task["consume"]
    assert consume_region.stream is not None
    assert consume_region.stream.value == "side_forward", (
        f"consume Region must be signed side_forward, got {consume_region.stream}"
    )

    state = ResidualRuntimeState()
    bindings = WorkloadExecutionBindings(
        declarations=(declaration,),
        ports=RuntimePortRegistry(
            outputs={
                "answer": RuntimeOutputBinding(
                    lambda value, step_id: observer.outputs.append(
                        (step_id, cast(torch.Tensor, value))
                    )
                )
            }
        ),
    )

    with mock.patch.object(residual_runtime, "RUNTIME", state):
        residual_runtime.publish_workload_compilation(compilation, bindings=bindings)
        residual_runtime.prepare_physical_workgraph_step(5)
        assert residual_runtime.advance_workload_anchor(
            by_task["produce"].placement_anchor
        )
        assert residual_runtime.advance_workload_anchor(consume_region.placement_anchor)

        # Main-path work enqueued *after* the side Region's advance must not
        # block it: the side stream only waits the snapshot taken at advance.
        busy = torch.randn(4096, 4096, device=device)
        for _ in range(200):
            busy = busy @ busy * 1e-6
        main_done = torch.cuda.Event()
        main_done.record()

        deadline = time.monotonic() + 5.0
        while not observer.side_done.query():  # type: ignore[union-attr]
            if time.monotonic() > deadline:
                raise SystemExit("side-stream Region never completed")
            time.sleep(0.001)
        assert not main_done.query(), (
            "main chain finished first; rerun with a longer chain to prove overlap"
        )

        execution = state.workload_execution
        assert execution is not None
        execution.finish_step(5)

    torch.cuda.synchronize()
    assert observer.producer_on_default is True, "producer must run on the default stream"
    assert observer.consumer_on_side is True, "consumer must run on a real side stream"
    assert len(observer.outputs) == 1
    step_id, value = observer.outputs[0]
    assert step_id == 5
    torch.testing.assert_close(
        value, torch.full((1024, 1024), 6.0, device=device)
    )
    print("side-stream overlap smoke OK: side Region completed under a busy main stream")


if __name__ == "__main__":
    main()
