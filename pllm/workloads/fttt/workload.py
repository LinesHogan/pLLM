"""FTTT dataflow and mathematics over the original host transaction."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import pllm
from pllm.workloads.fttt import tasks
from pllm.workloads.fttt.host_service import FTTTFeedbackService
from pllm.workloads.fttt.judge import FTTTDefaultJudge
from pllm.workloads.fttt.types import FTTTConfig, FTTTHostTransaction, FTTTResult


@dataclass(frozen=True, slots=True)
class FTTTWorkload(
    pllm.host.HostWorkload[
        pllm.host.HostTransactionCoordinator[
            pllm.host.CompletionInput | pllm.host.NoHostRequest,
            FTTTHostTransaction,
            tasks.FTTTStepContext,
            tasks.FTTTEditState,
            FTTTResult,
        ]
    ]
):
    feedback: FTTTFeedbackService = field(default_factory=lambda: FTTTFeedbackService(FTTTConfig(), FTTTDefaultJudge()))

    @property
    def config(self) -> FTTTConfig:
        return self.feedback.config

    def declare_host(
        self, facts: pllm.host.HostRuntimeFacts
    ) -> pllm.host.TransactionHostBinding[
        pllm.host.CompletionInput | pllm.host.NoHostRequest,
        FTTTHostTransaction,
        tasks.FTTTStepContext,
        tasks.FTTTEditState,
        FTTTResult,
    ]:
        if not 0 < self.config.rank <= facts.hidden_size:
            raise ValueError("FTTT rank exceeds the specialized hidden size")
        return pllm.host.transaction(
            "fttt.feedback",
            facts,
            config=self.config,
            state_factory=partial(tasks.initialize_edit_state, self.config, facts),
            service=self.feedback,
            update=partial(tasks.train, self.config),
            query=tasks.predict,
            publish=tasks.publish,
            example=partial(tasks.training_example, self.config),
        )
