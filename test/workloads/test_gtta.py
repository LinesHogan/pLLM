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

FEEDBACK = (
    TrajectoryFeedback("maps", "open", "Map Results", "directions", "", 1.0),
    TrajectoryFeedback("maps", "open", "Map Results", "directions 2", "", 0.5),
    TrajectoryFeedback("maps", "clear", "Map Results", "empty", "", -0.25),
)


def _coordinator():
    service = GTTAEnvironmentService(GTTAConfig(max_queue_size=3, score_momentum=0.25))
    return service, make_llm(workload=GTTAWorkload(service), config=HostRuntimeFacts(torch.device("cpu")))


def _request(**changes) -> GTTAHostRequest:
    query = GTTAQuery(
        prompt_idx=4,
        task_id="maps",
        candidate_actions=("clear", "unknown", "open"),
        observation="map results",
        state_signature="",
        action="",
        max_records=0,
        position=GTTAPromptPosition.SUFFIX,
    )
    return GTTAHostRequest(replace(query, **changes))


def test_feedback_queue_admits_to_capacity() -> None:
    service, _ = _coordinator()
    assert all(service.enqueue_feedback(event) for event in FEEDBACK)
    assert not service.enqueue_feedback(FEEDBACK[0])


def test_abort_rolls_back_fifo_without_a_version() -> None:
    service, coordinator = _coordinator()
    for event in FEEDBACK:
        service.enqueue_feedback(event)
    execution = coordinator.execution
    interrupted = service.begin(GTTAHostRequest(_request().query, limit=2))
    execution.abort(execution.execute(interrupted))
    service.abort(interrupted)
    assert execution.diagnostics().state_version == 0
    restored = service.begin(_request())
    assert tuple(event.reward for event in restored.trajectories.entries) == (1.0, 0.5, -0.25)
    service.abort(restored)


def test_run_scores_actions_and_settles_the_queue() -> None:
    service, coordinator = _coordinator()
    for event in FEEDBACK:
        service.enqueue_feedback(event)
    result = coordinator.run(_request())
    assert result is not None
    assert (result.update.processed, result.update.state_version) == (3, 1)
    assert result.query.ranked_actions == (("open", 0.5625), ("unknown", 0.0), ("clear", -0.1875))
    assert tuple(record.reward for record in result.query.records) == (1.0, 0.5, -0.25)
    context = result.query.prompt_context
    assert context.prompt_idx == 4 and context.position is GTTAPromptPosition.SUFFIX
    assert [json.loads(line)["reward"] for line in context.text.splitlines()] == [1.0, 0.5, -0.25]
    assert service.queued_trajectories == 0
    empty = coordinator.run(_request())
    assert empty is not None
    assert (empty.update.processed, empty.update.state_version) == (0, 1)
    limited = coordinator.run(_request(max_records=2))
    assert limited.query.records == result.query.records[:2]
    tied = coordinator.run(_request(candidate_actions=("z", "a")))
    assert tied.query.ranked_actions == (("z", 0.0), ("a", 0.0))
    assert coordinator.execution.plan.fallback_count == 0
