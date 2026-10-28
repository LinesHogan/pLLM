"""FTTT tensor mathematics; the shared host executor owns ordering and rollback."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt

import torch
from torch.nn import functional as F

from pllm.host import CompletionInput, HostRuntimeFacts
from pllm.workloads.fttt.math import edit_training_tensors
from pllm.workloads.fttt.types import (
    FTTTClosedLoopIntent,
    FTTTConfig,
    FTTTHostTransaction,
    FTTTResult,
)


@dataclass(slots=True)
class FTTTEditState:
    down: torch.Tensor
    up: torch.Tensor
    down_exp_avg: torch.Tensor
    down_exp_avg_sq: torch.Tensor
    up_exp_avg: torch.Tensor
    up_exp_avg_sq: torch.Tensor
    specialization: HostRuntimeFacts
    step: int = 0
    version: int = 0

    def prepare(self, transaction: FTTTHostTransaction) -> FTTTStepContext:
        if (
            transaction.batch.hidden.shape[1] != self.specialization.hidden_size
            or transaction.batch.hidden.dtype != torch.float32
            or transaction.batch.hidden.device != self.specialization.device
            or self.specialization.max_input_rows is not None
            and len(transaction.batch.hidden) > self.specialization.max_input_rows
        ):
            raise ValueError(
                "FTTT batch hidden tensor must match specialized device, float32 dtype, width and row limit"
            )
        if transaction.batch.readiness is not None:
            transaction.batch.readiness.order_on(torch.cuda.current_stream(self.specialization.device))
        return FTTTStepContext(transaction, self)

    def snapshot(self) -> FTTTEditState:
        return replace(
            self,
            down=self.down.detach().clone(),
            up=self.up.detach().clone(),
            down_exp_avg=self.down_exp_avg.clone(),
            down_exp_avg_sq=self.down_exp_avg_sq.clone(),
            up_exp_avg=self.up_exp_avg.clone(),
            up_exp_avg_sq=self.up_exp_avg_sq.clone(),
        )

    def restore(self, snapshot: FTTTEditState) -> None:
        self.down, self.up = snapshot.down, snapshot.up
        self.down_exp_avg, self.down_exp_avg_sq = snapshot.down_exp_avg, snapshot.down_exp_avg_sq
        self.up_exp_avg, self.up_exp_avg_sq = snapshot.up_exp_avg, snapshot.up_exp_avg_sq
        self.step, self.version = snapshot.step, snapshot.version

    def output(self, context: FTTTStepContext) -> FTTTResult:
        assert context.delta is not None and context.intent is not None
        return FTTTResult(
            context.transaction.intent.transaction_id,
            context.delta,
            context.intent,
            () if context.loss is None else (context.loss,),
            int(context.loss is not None),
            self.version,
        )


def initialize_edit_state(config: FTTTConfig, specialization: HostRuntimeFacts) -> FTTTEditState:
    input_size = specialization.hidden_size + config.feedback_feature_size + 1
    generator = torch.Generator(device=specialization.device).manual_seed(specialization.initialization_seed)
    down = torch.empty((config.rank, input_size), dtype=torch.float32, device=specialization.device)
    down.normal_(std=1.0 / sqrt(input_size), generator=generator)
    up = torch.zeros((specialization.hidden_size, config.rank), dtype=torch.float32, device=specialization.device)
    return FTTTEditState(
        down,
        up,
        torch.zeros_like(down),
        torch.zeros_like(down),
        torch.zeros_like(up),
        torch.zeros_like(up),
        specialization,
    )


@dataclass(slots=True)
class FTTTStepContext:
    transaction: FTTTHostTransaction
    edit_state: FTTTEditState
    edit_input: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    delta: torch.Tensor | None = None
    intent: FTTTClosedLoopIntent | None = None


def train(config: FTTTConfig, context: FTTTStepContext) -> None:
    transaction, hidden, state = context.transaction, context.transaction.batch.hidden, context.edit_state
    if not transaction.selected_rows:
        context.edit_input = hidden.new_empty((0, hidden.shape[1] + config.feedback_feature_size + 1))
        return
    selected = hidden.detach().index_select(
        0, torch.tensor(transaction.selected_rows, dtype=torch.long, device=hidden.device)
    )
    edit_input, target = edit_training_tensors(
        selected,
        transaction.intent.feedback_records,
        feature_size=config.feedback_feature_size,
        edit_scale=config.edit_scale,
    )
    down, up = state.down.detach().requires_grad_(True), state.up.detach().requires_grad_(True)
    loss = F.mse_loss(F.linear(F.linear(edit_input, down), up), target)
    down_gradient, up_gradient = torch.autograd.grad(loss, (down, up))
    state.step += 1
    beta_one, beta_two = config.betas
    with torch.no_grad():
        for parameter, gradient, exp_avg, exp_avg_sq in (
            (state.down, down_gradient, state.down_exp_avg, state.down_exp_avg_sq),
            (state.up, up_gradient, state.up_exp_avg, state.up_exp_avg_sq),
        ):
            exp_avg.mul_(beta_one).add_(gradient, alpha=1.0 - beta_one)
            exp_avg_sq.mul_(beta_two).addcmul_(gradient, gradient, value=1.0 - beta_two)
            denominator = exp_avg_sq.sqrt().div_((1.0 - beta_two**state.step) ** 0.5).add_(config.eps)
            parameter.addcdiv_(exp_avg, denominator, value=-config.learning_rate / (1.0 - beta_one**state.step))
    state.version += 1
    context.edit_input, context.loss = edit_input, loss.detach()


def predict(context: FTTTStepContext) -> None:
    assert context.edit_input is not None
    context.delta = F.linear(F.linear(context.edit_input, context.edit_state.down), context.edit_state.up).detach()


def publish(context: FTTTStepContext) -> None:
    context.intent = replace(context.transaction.intent, adapter_version=context.edit_state.version)


def training_example(config: FTTTConfig, facts: HostRuntimeFacts) -> tuple[FTTTStepContext, FTTTEditState]:
    """Use the host service's trainable path; feedback contents do not change tensor shapes."""
    from pllm.workloads.fttt.host_service import FTTTFeedbackService
    from pllm.workloads.fttt.judge import FTTTDefaultJudge

    batch = CompletionInput.example(facts, torch.device("cpu"))
    service = FTTTFeedbackService(
        replace(config, max_queue_size=len(batch.hidden), positive_reward_threshold=1.0), FTTTDefaultJudge()
    )
    transaction = service.begin(batch)
    assert transaction is not None
    state = initialize_edit_state(config, replace(facts, device=torch.device("cpu")))
    return state.prepare(transaction), state.snapshot()
