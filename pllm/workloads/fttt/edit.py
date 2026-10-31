"""FTTT's low-rank edit: its tensors, one Adam step on judged feedback, and the edit delta a retry applies.

FTTTWorkload owns one FTTTEdit and calls `train` then `delta` for each queued feedback batch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt

import torch
from torch.nn import functional as F

from pllm.workloads.fttt.math import edit_training_tensors
from pllm.workloads.fttt.types import FTTTConfig, FTTTFeedbackRecord


@dataclass(slots=True)
class FTTTEdit:
    """Down/up factors mapping (hidden, feedback features, reward) rows to a hidden-size edit, with Adam moments."""

    down: torch.Tensor
    up: torch.Tensor
    down_exp_avg: torch.Tensor
    down_exp_avg_sq: torch.Tensor
    up_exp_avg: torch.Tensor
    up_exp_avg_sq: torch.Tensor
    step: int = 0
    version: int = 0

    @classmethod
    def initial(cls, config: FTTTConfig, hidden_size: int, device: torch.device, seed: int) -> FTTTEdit:
        """Random down factor and zero up factor, so the first edit is zero and the first step can learn `up`."""
        input_size = hidden_size + config.feedback_feature_size + 1
        generator = torch.Generator(device=device).manual_seed(seed)
        down = torch.empty((config.rank, input_size), dtype=torch.float32, device=device)
        down.normal_(std=1.0 / sqrt(input_size), generator=generator)
        up = torch.zeros((hidden_size, config.rank), dtype=torch.float32, device=device)
        return cls(down, up, torch.zeros_like(down), torch.zeros_like(down), torch.zeros_like(up), torch.zeros_like(up))

    def snapshot(self) -> FTTTEdit:
        """An independent copy to restore when a step fails."""
        return replace(
            self,
            down=self.down.detach().clone(),
            up=self.up.detach().clone(),
            down_exp_avg=self.down_exp_avg.clone(),
            down_exp_avg_sq=self.down_exp_avg_sq.clone(),
            up_exp_avg=self.up_exp_avg.clone(),
            up_exp_avg_sq=self.up_exp_avg_sq.clone(),
        )

    def train(
        self, config: FTTTConfig, selected: torch.Tensor, records: tuple[FTTTFeedbackRecord, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One Adam step on the selected hidden rows and their feedback; returns the edit input rows and the loss."""
        edit_input, target = edit_training_tensors(
            selected, records, feature_size=config.feedback_feature_size, edit_scale=config.edit_scale
        )
        down, up = self.down.detach().requires_grad_(True), self.up.detach().requires_grad_(True)
        loss = F.mse_loss(F.linear(F.linear(edit_input, down), up), target)
        down_gradient, up_gradient = torch.autograd.grad(loss, (down, up))
        self.step += 1
        beta_one, beta_two = config.betas
        with torch.no_grad():
            for parameter, gradient, exp_avg, exp_avg_sq in (
                (self.down, down_gradient, self.down_exp_avg, self.down_exp_avg_sq),
                (self.up, up_gradient, self.up_exp_avg, self.up_exp_avg_sq),
            ):
                exp_avg.mul_(beta_one).add_(gradient, alpha=1.0 - beta_one)
                exp_avg_sq.mul_(beta_two).addcmul_(gradient, gradient, value=1.0 - beta_two)
                denominator = exp_avg_sq.sqrt().div_((1.0 - beta_two**self.step) ** 0.5).add_(config.eps)
                parameter.addcdiv_(exp_avg, denominator, value=-config.learning_rate / (1.0 - beta_one**self.step))
        self.version += 1
        return edit_input, loss.detach()

    def delta(self, edit_input: torch.Tensor) -> torch.Tensor:
        """The edit the current factors give each input row."""
        return F.linear(F.linear(edit_input, self.down), self.up).detach()


__all__ = ["FTTTEdit"]
