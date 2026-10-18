"""FTTT dataflow and mathematics over the original host transaction."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial

import pllm
from pllm.contracts.semantic_graph import SchedulerTimeline, TaskResourceFact, TaskSymbol, ValueResourceFact
from pllm.workloads.fttt import tasks
from pllm.workloads.fttt.host_service import FTTTFeedbackService
from pllm.workloads.fttt.judge import FTTTDefaultJudge
from pllm.workloads.fttt.types import FTTTConfig, FTTTHostTransaction, FTTTResult


@dataclass(frozen=True, slots=True)
class FTTTWorkload(
    pllm.HostWorkload[
        pllm.HostTransactionCoordinator[
            pllm.CompletionInput | pllm.NoHostRequest,
            FTTTHostTransaction,
            FTTTResult,
        ]
    ]
):
    feedback: FTTTFeedbackService = field(default_factory=lambda: FTTTFeedbackService(FTTTConfig(), FTTTDefaultJudge()))

    @property
    def config(self) -> FTTTConfig:
        return self.feedback.config

    def resource_facts(self, facts: pllm.HostRuntimeFacts) -> SchedulerTimeline:
        timeline = facts.timeline()
        if facts.memory_budget_bytes is None:
            return timeline
        if pllm.Workload.native_methods({"update": tasks.train, "query": tasks.predict}) != _NATIVE:
            raise ValueError("modified FTTT mathematics require their own resource_facts declaration")
        rows, hidden, rank, features = (
            facts.max_input_rows,
            facts.hidden_size,
            self.config.rank,
            self.config.feedback_feature_size,
        )
        if rows is None:
            raise ValueError("finite FTTT costs require max_input_rows")
        inputs = hidden + features + 1
        parameters = rank * (inputs + hidden)
        # Parameters/two moments, their rollback snapshot, and the original active hidden input.
        reserved = 4 * (6 * parameters + rows * hidden)
        workspace = (
            4 * (4 * parameters + rows * (3 * inputs + 6 * hidden + 4 * rank + 8) + 4 * features) + 8 * rows + 64
        )
        return replace(
            timeline,
            reserved_bytes=(reserved,) * len(timeline.anchors),
            task_resources=(
                TaskResourceFact(TaskSymbol("publish"), 0),
                TaskResourceFact(TaskSymbol("query"), 4 * rows * (rank + hidden)),
                TaskResourceFact(TaskSymbol("update"), workspace, input_shapes=((rows, hidden),)),
            ),
            value_resources=(
                ValueResourceFact(TaskSymbol("publish"), "result", 4 * (rows * hidden + 1)),
                ValueResourceFact(TaskSymbol("query"), "context", 4 * (rows * (inputs + hidden) + 1)),
                ValueResourceFact(TaskSymbol("update"), "context", 4 * (rows * inputs + 1)),
            ),
        )

    def declare_host(
        self, facts: pllm.HostRuntimeFacts
    ) -> pllm.TransactionHostBinding[
        pllm.CompletionInput | pllm.NoHostRequest,
        FTTTHostTransaction,
        tasks.FTTTStepContext,
        tasks.FTTTEditState,
        FTTTResult,
    ]:
        if not 0 < self.config.rank <= facts.hidden_size:
            raise ValueError("FTTT rank exceeds the specialized hidden size")
        return pllm.transaction(
            "fttt.feedback",
            facts,
            config=self.config,
            state_factory=partial(tasks.initialize_edit_state, self.config, facts),
            service=self.feedback,
            update=partial(tasks.train, self.config),
            query=tasks.predict,
            publish=tasks.publish,
            timeline=self.resource_facts(facts),
        )


_NATIVE = pllm.Workload.native_methods({"update": tasks.train, "query": tasks.predict})
