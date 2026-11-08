"""FTTT on the host: judge each completion, train a low-rank edit on the failed ones, and return the edit a retry
applies.

FTTTWorkload is the edit as an ordinary pllm.Workload state trained by AdamW. FTTTFeedback compiles it with
pllm.host.make_host and trains it: pass `FTTTFeedback` as generate_feedback's `feedback=` and
`FTTTResult.state_update` as its `edit=`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch
from torch.nn import functional as F

import pllm
from pllm.execution.cuda_stream import upload
from pllm.host import CompletionInput, HostConfig, WeightedRowBatch, make_host
from pllm.workloads.fttt.feedback_queue import FTTTFeedbackQueue
from pllm.workloads.fttt.math import feedback_rows
from pllm.workloads.fttt.types import FTTTConfig, FTTTQueuedFeedback, FTTTResult


def edit_rows(hidden: torch.Tensor, feedback: torch.Tensor, down: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """The edit each row gets: `up @ down` applied to the row's hidden state, feedback features and reward."""
    return F.linear(F.linear(torch.cat((hidden, feedback), dim=1), down), up)


def edit_targets(feedback: torch.Tensor, hidden_size: int, edit_scale: float) -> torch.Tensor:
    """What each row's edit should be: feedback feature c added to column c % hidden_size, averaged over
    features // hidden_size + 1, and scaled by edit_scale and how far the reward is below 1."""
    features, rewards = feedback[:, :-1], feedback[:, -1:]
    width = features.shape[1]
    folded = F.pad(features, (0, -width % hidden_size)).view(len(features), -1, hidden_size).sum(dim=1)
    return folded / (width // hidden_size + 1) * ((1.0 - rewards).clamp(min=0.0) * edit_scale)


@dataclass(frozen=True)
class FTTTWorkload(pllm.Workload):
    """A rank-r edit from (hidden, feedback features, reward) rows to hidden-size deltas, fit by AdamW."""

    config: FTTTConfig = FTTTConfig()

    @property
    def edit(self) -> pllm.InstanceState:
        rank, feedback = self.config.rank, self.config.feedback_width
        return pllm.InstanceState(
            {"down": lambda m: (rank, m.hidden_size + feedback), "up": lambda m: (m.hidden_size, rank)},
            optimizer=pllm.AdamW(self.config.learning_rate, self.config.betas, self.config.eps),
            init=pllm.Computed(self.initialize),
        )

    def initialize(self, parameters: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
        """A random down factor and a zero up factor: the first edit is zero and the first step learns `up`."""
        down = parameters["down"]
        down.normal_(std=down.shape[1] ** -0.5, generator=generator)

    @pllm.train(state="edit")
    def update(
        self,
        hidden: torch.Tensor = pllm.batch("hidden"),
        feedback: torch.Tensor = pllm.batch("targets"),
        down: torch.Tensor = pllm.parameter("edit", "down"),
        up: torch.Tensor = pllm.parameter("edit", "up"),
    ) -> torch.Tensor:
        """Mean squared error of each row's edit against its target; FTTT batches hold only the failed rows."""
        target = edit_targets(feedback, hidden.shape[1], self.config.edit_scale)
        return F.mse_loss(edit_rows(hidden, feedback, down, up), target)


class FTTTFeedback:
    """The `generate_feedback` callable: a queue of judged feedback and the edit it trains on `host` (whose
    target_shape FTTT sets to its feedback width); steps run on the caller's current CUDA stream."""

    def __init__(self, queue: FTTTFeedbackQueue, host: HostConfig) -> None:
        self.queue = queue
        self.edit = make_host(replace(host, target_shape=(queue.config.feedback_width,)), FTTTWorkload(queue.config))

    def __call__(self, batch: CompletionInput) -> FTTTResult | None:
        """Judge `batch`, queue its feedback, and run one edit step on the oldest queued feedback."""
        self.queue.enqueue(batch)
        return self.step()

    def step(self) -> FTTTResult | None:
        """Train the oldest feedback, or return None for an empty queue. Failure drops the queue;
        any edit already published remains available."""
        queued = self.queue.begin()
        if queued is None:
            return None
        try:
            if queued.selected_rows:
                delta, losses = self._train(queued)
            else:
                delta = queued.batch.hidden.new_empty((0, queued.batch.hidden.shape[1]))
                losses = ()
            version = self.edit.version
            self.queue.commit(queued, version)
        except BaseException:
            self.queue.abort()
            raise
        return FTTTResult(
            delta,
            queued.reward_outcomes,
            queued.retry_controls,
            queued.update_barriers,
            queued.feedback_records,
            losses,
            len(losses),
            version,
        )

    def _train(self, queued: FTTTQueuedFeedback) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Step the edit on the failed rows, then apply the updated edit to the same rows."""
        hidden = queued.batch.hidden
        rows = hidden.index_select(0, upload(queued.selected_rows, hidden.device))
        records = queued.feedback_records
        feedback = upload(
            feedback_rows(records, feature_size=self.queue.config.feedback_feature_size), hidden.device, torch.float32
        )
        step = self.edit.train(WeightedRowBatch(rows, feedback, torch.ones(len(records), device=hidden.device)))
        assert step.loss is not None  # FTTTWorkload.update always returns a loss
        return edit_rows(rows, feedback, step.parameters["down"], step.parameters["up"]), (step.loss,)


__all__ = ["FTTTFeedback", "FTTTWorkload", "edit_rows", "edit_targets"]
