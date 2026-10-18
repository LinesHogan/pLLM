"""Evidence 22: environment scoring, version rollback and FIFO settlement."""

import json
from dataclasses import replace

import torch

from pllm import make_llm
from pllm.execution.host_runtime import HostRuntimeFacts
from pllm.workloads.gtta import (
    GTTAConfig,
    GTTAHostRequest,
    GTTAPromptPosition,
    GTTAQuery,
    GTTAWorkload,
    TrajectoryFeedback,
)
from pllm.workloads.gtta.environment import GTTAEnvironmentService


def test_gtta_updates_environment_and_rolls_back_fifo() -> None:
    config = GTTAConfig(max_queue_size=3, score_momentum=0.25)
    service = GTTAEnvironmentService(config)
    coordinator = make_llm(workload=GTTAWorkload(service), config=HostRuntimeFacts(torch.device("cpu")))
    execution = coordinator.execution
    feedback = (
        TrajectoryFeedback("maps", "open", "Map Results", "directions", "", 1.0),
        TrajectoryFeedback("maps", "open", "Map Results", "directions 2", "", 0.5),
        TrajectoryFeedback("maps", "clear", "Map Results", "empty", "", -0.25),
    )
    assert all(service.enqueue_feedback(event) for event in feedback)
    assert not service.enqueue_feedback(feedback[0])
    request = GTTAHostRequest(
        GTTAQuery(
            prompt_idx=4,
            task_id="maps",
            candidate_actions=("clear", "unknown", "open"),
            observation="map results",
            state_signature="",
            action="",
            max_records=0,
            position=GTTAPromptPosition.SUFFIX,
        )
    )
    interrupted = service.begin(GTTAHostRequest(request.query, limit=2))
    receipt = execution.execute(interrupted)
    execution.abort(receipt)
    service.abort(interrupted)
    assert execution.diagnostics().state_version == 0
    restored = service.begin(request)
    assert tuple(event.reward for event in restored.trajectories.entries) == (1.0, 0.5, -0.25)
    service.abort(restored)

    result = coordinator.run(request)
    assert result is not None
    assert result.update.processed == 3
    assert result.update.state_version == 1
    assert result.query.ranked_actions == (("open", 0.5625), ("unknown", 0.0), ("clear", -0.1875))
    assert tuple(record.reward for record in result.query.records) == (1.0, 0.5, -0.25)
    context = result.query.prompt_context
    assert context.prompt_idx == 4 and context.position is GTTAPromptPosition.SUFFIX
    assert [json.loads(line)["reward"] for line in context.text.splitlines()] == [1.0, 0.5, -0.25]
    assert service.queued_trajectories == 0
    empty = coordinator.run(request)
    assert empty is not None and empty.update.processed == 0
    assert empty.update.state_version == 1
    limited = coordinator.run(replace(request, query=replace(request.query, max_records=2)))
    assert limited.query.records == result.query.records[:2]
    tied = coordinator.run(replace(request, query=replace(request.query, candidate_actions=("z", "a"))))
    assert tied.query.ranked_actions == (("z", 0.0), ("a", 0.0))
    assert execution.plan.fallback_count == 0
