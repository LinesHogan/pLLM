"""Public-entry vertical oracle for the clean-room Dummy workload."""

from __future__ import annotations

import torch

from tllm.canonical.workloads.dummy import DummyExported, DummySkipped, DummyStep
from tllm.consumers.dummy import DummyConfig, DummyWorkload
from tllm.contracts.semantic_graph import (
    AnchorSymbol,
    SchedulerTimeline,
    TimelineAnchor,
)


def test_public_dummy_runs_semantic_graph_rust_plan_and_clean_execution() -> None:
    workload = DummyWorkload(
        DummyConfig(
            export_every_n_steps=2,
            export_max_rows=2,
            export_max_cols=3,
        )
    )
    execution = workload.compile(
        SchedulerTimeline(
            anchors=(
                TimelineAnchor(AnchorSymbol("model.execute"), 0),
                TimelineAnchor(AnchorSymbol("step.end"), 1),
            ),
            capabilities=("host/cpu_export@1",),
        )
    )
    hidden = torch.arange(15, dtype=torch.float32).reshape(3, 5)

    assert isinstance(execution.run(DummyStep(hidden, 1)), DummySkipped)
    exported = execution.run(DummyStep(hidden, 2))
    assert isinstance(exported, DummyExported)
    torch.testing.assert_close(exported.rows, hidden[:2, :3])
    assert exported.rows.device.type == "cpu"
    assert execution.plan.fallback_count == 0
    assert workload.diagnostics().consumed_batches == 1
    assert workload.diagnostics().consumed_rows == 2
