"""Evidence 22: environment scoring, FIFO consumption and query failures after updates."""

import json
from dataclasses import replace

import pytest

from pllm.workloads.gtta import (
    GTTAConfig,
    GTTAHostRequest,
    GTTAPromptPosition,
    GTTAQuery,
    GTTAWorkload,
    TrajectoryFeedback,
)
from pllm.workloads.gtta.environment import GTTAEnvironment
from pllm.workloads.gtta.state import GTTADynamicsState

FEEDBACK = (
    TrajectoryFeedback("maps", "open", "Map Results", "directions", "", 1.0),
    TrajectoryFeedback("maps", "open", "Map Results", "directions 2", "", 0.5),
    TrajectoryFeedback("maps", "clear", "Map Results", "empty", "", -0.25),
)


QUERY = GTTAQuery(
    prompt_index=4,
    task_id="maps",
    candidate_actions=("clear", "unknown", "open"),
    observation="map results",
    state_signature="",
    action="",
    max_records=0,
    position=GTTAPromptPosition.SUFFIX,
)


def interrupted_query(self, query):
    raise RuntimeError("query interrupted after update")


def test_gtta_scores_actions_consumes_fifo_and_retains_updates_on_query_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = GTTAEnvironment(GTTAConfig(max_queue_size=3, score_momentum=0.25))
    workload = GTTAWorkload(service)
    assert all(service.enqueue_feedback(event) for event in FEEDBACK)
    assert not service.enqueue_feedback(FEEDBACK[0])
    with monkeypatch.context() as fault, pytest.raises(RuntimeError, match="interrupted"):
        fault.setattr(GTTADynamicsState, "query", interrupted_query)
        workload.run(GTTAHostRequest(QUERY, limit=2))
    assert service.queued_trajectories == 1

    result = workload.run(GTTAHostRequest(QUERY))
    assert result.processed == 1
    assert result.query.ranked_actions == (("open", 0.5625), ("unknown", 0.0), ("clear", -0.1875))
    assert tuple(record.reward for record in result.query.records) == (1.0, 0.5, -0.25)
    context = result.query.prompt_context
    assert context.prompt_index == 4 and context.position is GTTAPromptPosition.SUFFIX
    assert [json.loads(line)["reward"] for line in context.text.splitlines()] == [1.0, 0.5, -0.25]
    empty = workload.run(GTTAHostRequest(QUERY))
    assert empty.processed == 0
    limited = workload.run(GTTAHostRequest(replace(QUERY, max_records=2)))
    assert limited.query.records == result.query.records[:2]
