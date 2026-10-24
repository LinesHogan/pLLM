"""Evidence 22: environment scoring, version rollback and FIFO settlement."""

import json
from dataclasses import replace

import torch

from pllm import make_llm
from pllm.host import HostRuntimeFacts
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


QUERY = GTTAQuery(
    prompt_idx=4,
    task_id="maps",
    candidate_actions=("clear", "unknown", "open"),
    observation="map results",
    state_signature="",
    action="",
    max_records=0,
    position=GTTAPromptPosition.SUFFIX,
)


def test_gtta_scores_actions_settles_fifo_and_rolls_back_aborts() -> None:
    service = GTTAEnvironmentService(GTTAConfig(max_queue_size=3, score_momentum=0.25))
    coordinator = make_llm(workload=GTTAWorkload(service), config=HostRuntimeFacts(torch.device("cpu")))
    execution = coordinator.execution
    assert all(service.enqueue_feedback(event) for event in FEEDBACK)
    assert not service.enqueue_feedback(FEEDBACK[0])
    interrupted = service.begin(GTTAHostRequest(QUERY, limit=2))
    execution.abort(execution.execute(interrupted))
    service.abort(interrupted)
    assert execution.diagnostics().state_version == 0

    result = coordinator.run(GTTAHostRequest(QUERY))
    assert result is not None
    assert (result.update.processed, result.update.state_version) == (3, 1)
    assert result.query.ranked_actions == (("open", 0.5625), ("unknown", 0.0), ("clear", -0.1875))
    assert tuple(record.reward for record in result.query.records) == (1.0, 0.5, -0.25)
    context = result.query.prompt_context
    assert context.prompt_idx == 4 and context.position is GTTAPromptPosition.SUFFIX
    assert [json.loads(line)["reward"] for line in context.text.splitlines()] == [1.0, 0.5, -0.25]
    empty = coordinator.run(GTTAHostRequest(QUERY))
    assert empty is not None and (empty.update.processed, empty.update.state_version) == (0, 1)
    limited = coordinator.run(GTTAHostRequest(replace(QUERY, max_records=2)))
    assert limited.query.records == result.query.records[:2]
