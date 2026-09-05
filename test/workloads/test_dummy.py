"""Evidence 20: graph, Rust plan and owned hidden-row export."""

import torch

from tllm.canonical.workloads.dummy import (
    DummyConfig, DummyExported, DummySkipped, DummyStep, DummyWorkload,
)
from tllm.contracts.semantic_graph import (
    AnchorSymbol, SchedulerTimeline, TimelineAnchor,
)


def test_dummy_runs_graph_sign_plan_execute() -> None:
    workload = DummyWorkload(DummyConfig(
        export_every_n_steps=2, export_max_rows=2, export_max_cols=3,
    ))
    execution = workload.compile(SchedulerTimeline(
        anchors=(
            TimelineAnchor(AnchorSymbol("model.execute"), 0),
            TimelineAnchor(AnchorSymbol("step.end"), 1),
        ),
        capabilities=("host/cpu_export@1",),
    ))
    hidden = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    expected = hidden[:2, :3].clone()
    assert isinstance(execution.run(DummyStep(hidden, 1)), DummySkipped)
    exported = execution.run(DummyStep(hidden, 2))
    assert isinstance(exported, DummyExported)
    hidden.fill_(-1)
    torch.testing.assert_close(exported.rows, expected)
    assert exported.rows.device.type == "cpu"
    assert execution.plan.fallback_count == 0
    assert workload.diagnostics().consumed_batches == 1
    assert workload.diagnostics().consumed_rows == 2
