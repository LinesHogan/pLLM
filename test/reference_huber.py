"""Frozen independent reference for the low-rank Huber adaptation experiment.

Algorithm: independent request AdamW over A[hidden, rank], B[rank, hidden].
Prediction is x + (x @ A) @ B, using the pre-update request parameters. Guidance
is enabled only for input versions > 0, with beta 0.2. All decode rows train
after sampling; loss is Huber(delta=0.25), averaged over rows and hidden elements
within each request, never averaged across requests. No reset between steps.

Initialize A then B with normal(0, 0.02), advancing the supplied generator across
request admissions. AdamW uses lr=0.03, betas=(0.8, 0.95), eps=1e-8, decay=0.1.
The implementation under adaptation must not import these helpers. This file is
the fixed oracle and must not be modified to accommodate framework behavior.
"""

import torch
from torch.nn import functional as F


def initialize(
    hidden: int, rank: int, generator: torch.Generator, device: torch.device
) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    return (
        torch.nn.Parameter(torch.empty(hidden, rank, device=device).normal_(0.0, 0.02, generator=generator)),
        torch.nn.Parameter(torch.empty(rank, hidden, device=device).normal_(0.0, 0.02, generator=generator)),
    )


def optimizer(parameters: tuple[torch.nn.Parameter, torch.nn.Parameter]) -> torch.optim.AdamW:
    return torch.optim.AdamW(parameters, lr=0.03, betas=(0.8, 0.95), eps=1e-8, weight_decay=0.1)


def predict(source: torch.Tensor, parameters: tuple[torch.nn.Parameter, torch.nn.Parameter]) -> torch.Tensor:
    a, b = parameters
    return source + (source @ a) @ b


def step(
    source: torch.Tensor,
    target: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, torch.nn.Parameter],
    update: torch.optim.AdamW,
) -> torch.Tensor:
    update.zero_grad(set_to_none=True)
    loss = F.huber_loss(predict(source, parameters), target, delta=0.25, reduction="mean")
    loss.backward()
    update.step()
    return loss.detach()
