"""One Python-Rust/host-service lifecycle proof for canonical GTTA."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tllm.canonical.host_runtime import CanonicalHostRuntimeFacts
from tllm.canonical.workloads.gtta import (
    GTTACompileOptions,
    GTTAConfig,
    GTTADiagnostics,
    GTTAExecution,
    GTTAPromptPosition,
    GTTAQuery,
    GTTAWorkload,
    TrajectoryFeedback,
)
from tllm.canonical.workloads.gtta.environment import GTTAEnvironmentService
from tllm.canonical.workloads.gtta.session import GTTASession
from tllm.contracts.semantic_graph import AnchorId, StateScope


def test_gtta_runs_signed_environment_transaction_and_rolls_back_fifo() -> None:
    config = GTTAConfig(max_queue_size=3, score_momentum=0.25)
    workload = GTTAWorkload(config)
    graph = workload.semantic_graph()
    assert graph.states[0].scope is StateScope.INSTANCE
    compiled = workload.compile(
        GTTACompileOptions(
            CanonicalHostRuntimeFacts(torch.device("cpu")).timeline()
        )
    )
    assert compiled.execution.plan.fallback_count == 0
    assert {
        task.task_symbol: task.anchor_symbol
        for task in compiled.schedule_bundle.schedule_decision.tasks
    } == {
        "publish": "host.output.publish",
        "query": "host.compute.end",
        "update": "host.input.ready",
    }

    tampered_plan = replace(
        compiled.schedule_bundle.schedule_physical,
        tasks=(
            replace(
                compiled.schedule_bundle.schedule_physical.tasks[0],
                anchor_id=AnchorId(2),
            ),
            *compiled.schedule_bundle.schedule_physical.tasks[1:],
        ),
    )
    with pytest.raises(ValueError):
        GTTAExecution(
            config,
            replace(
                compiled.schedule_bundle,
                schedule_physical=tampered_plan,
            ),
        )

    environment = GTTAEnvironmentService(config)
    feedback = (
        TrajectoryFeedback(
            "maps",
            "open directions",
            "Map   Results",
            "directions panel",
            "Directions became available",
            1.0,
        ),
        TrajectoryFeedback(
            "maps",
            "open directions",
            "Map Results",
            "directions panel 2",
            "",
            0.5,
        ),
        TrajectoryFeedback(
            "maps",
            "clear",
            "Map Results",
            "empty page",
            "",
            -0.25,
        ),
    )
    assert all(environment.enqueue_feedback(entry) for entry in feedback)
    assert not environment.enqueue_feedback(
        TrajectoryFeedback(
            "maps",
            "dropped",
            "Map Results",
            "unreachable",
            "",
            9.0,
        )
    )
    assert environment.dropped_trajectories == 1

    interrupted = environment.begin(2)
    assert tuple(event.reward for event in interrupted.trajectories.entries) == (
        1.0,
        0.5,
    )
    environment.abort(interrupted.transaction_id)
    restored = environment.begin(0)
    assert tuple(event.reward for event in restored.trajectories.entries) == (
        1.0,
        0.5,
        -0.25,
    )
    environment.abort(restored.transaction_id)

    session = GTTASession(environment, compiled.execution)
    query = GTTAQuery(
        prompt_idx=4,
        task_id="maps",
        candidate_actions=("clear", "unknown", "open directions"),
        observation="map results",
        state_signature="",
        action="",
        max_records=0,
        position=GTTAPromptPosition.SUFFIX,
    )
    result = session.query(query, 0)
    assert result.update.processed == 3
    assert result.update.state_version == 1
    assert result.query.ranked_actions == (
        ("open directions", 0.5625),
        ("unknown", 0.0),
        ("clear", -0.1875),
    )
    assert tuple(record.reward for record in result.query.records) == (
        1.0,
        0.5,
        -0.25,
    )
    assert result.query.prompt_context.prompt_idx == 4
    assert result.query.prompt_context.position is GTTAPromptPosition.SUFFIX
    assert result.query.prompt_context.text == "\n".join(
        (
            '{"action_taken": "open directions", "environmental_dynamics": '
            '"Directions became available", "initial_state": "Map   Results", '
            '"reward": 1.0, "state_signature": "map results", "task_id": "maps"}',
            '{"action_taken": "open directions", "environmental_dynamics": '
            '"Next observation: directions panel 2", "initial_state": "Map Results", '
            '"reward": 0.5, "state_signature": "map results", "task_id": "maps"}',
            '{"action_taken": "clear", "environmental_dynamics": '
            '"Next observation: empty page", "initial_state": "Map Results", '
            '"reward": -0.25, "state_signature": "map results", "task_id": "maps"}',
        )
    )

    empty = session.query(query, 0)
    assert empty.update.processed == 0
    assert empty.update.state_version == 1
    assert compiled.execution.diagnostics() == GTTADiagnostics(2, 3, 1, 0)
