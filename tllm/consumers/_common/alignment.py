#!/usr/bin/env python3
"""Common building blocks shared by the test-time-training consumers."""

from __future__ import annotations

import math
import re

import torch
from torch import nn
from torch.nn import functional as F


class LowRankAdapter(nn.Module):
    """Small LoRA-style adapter with zero-initialized output projection."""

    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        rank: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if int(in_dim) <= 0:
            raise ValueError("LowRankAdapter requires positive in_dim")
        if int(out_dim) <= 0:
            raise ValueError("LowRankAdapter requires positive out_dim")
        if int(rank) <= 0:
            raise ValueError("LowRankAdapter requires positive rank")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.rank = int(rank)
        self.down = nn.Linear(self.in_dim, self.rank, bias=False, dtype=dtype, device=device)
        self.up = nn.Linear(self.rank, self.out_dim, bias=False, dtype=dtype, device=device)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.down.weight, mean=0.0, std=1.0 / math.sqrt(float(self.in_dim)))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


def detached_train_input(tensor: torch.Tensor) -> torch.Tensor:
    """Return a trainable detached view, cloning PyTorch inference tensors."""
    out = tensor.detach()
    if torch.is_inference(out):
        out = out.clone()
    return out


def sentence_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = -100) -> torch.Tensor:
    """Return per-row mean token cross entropy for 2D or 3D logits."""
    if logits.ndim == 2:
        logits_3d = logits.unsqueeze(1)
        labels_2d = labels.reshape(-1, 1)
    elif logits.ndim == 3:
        logits_3d = logits
        labels_2d = labels
    else:
        raise RuntimeError("sentence_cross_entropy expects 2D or 3D logits")
    if labels_2d.ndim != 2:
        raise RuntimeError("sentence_cross_entropy expects labels aligned as rows x tokens")
    if int(logits_3d.shape[0]) != int(labels_2d.shape[0]) or int(logits_3d.shape[1]) != int(labels_2d.shape[1]):
        raise RuntimeError("sentence_cross_entropy logits and labels must align on rows and tokens")
    vocab = int(logits_3d.shape[-1])
    losses = F.cross_entropy(
        logits_3d.reshape(-1, vocab),
        labels_2d.reshape(-1).to(device=logits.device, dtype=torch.long),
        reduction="none",
        ignore_index=int(ignore_index),
    ).reshape(labels_2d.shape)
    valid = labels_2d.to(device=logits.device).ne(int(ignore_index))
    denom = valid.sum(dim=1).clamp_min(1).to(dtype=losses.dtype)
    return (losses * valid.to(dtype=losses.dtype)).sum(dim=1) / denom


def select_rows_above_threshold(values: torch.Tensor, *, threshold: float) -> list[int]:
    """Select row ids with value strictly above threshold."""
    flat = values.reshape(-1).detach().cpu()
    return [idx for idx, value in enumerate(flat.tolist()) if float(value) > float(threshold)]


def weighted_one_hot_kl_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    sentence_ce: torch.Tensor,
    threshold: float,
    lamb: float,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Upstream-TTL-shaped selected-row weighted one-hot KL/CE loss."""
    ce = sentence_cross_entropy(logits, labels, ignore_index=ignore_index)
    ce_for_weights = sentence_ce.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    if int(ce.shape[0]) != int(ce_for_weights.shape[0]):
        raise RuntimeError("weighted_one_hot_kl_loss requires one CE score per row")
    selected = ce_for_weights > float(threshold)
    if not bool(selected.any()):
        return logits.sum() * 0.0
    weights = float(lamb) * torch.exp(ce_for_weights[selected] - float(threshold))
    return (ce[selected] * weights).mean()


_BOXED_PATTERN = re.compile(r"\\boxed\s*\{([^{}]*)\}")


def normalize_answer_text(text: str) -> str:
    """Extract and normalize an answer string, preferring the last boxed answer."""
    raw = str(text).strip()
    matches = _BOXED_PATTERN.findall(raw)
    if matches:
        raw = matches[-1]
    raw = raw.strip().lower()
    raw = re.sub(r"\s+", " ", raw)
    raw = raw.replace(" ", "")
    return raw


def select_top_k_per_prompt(*, num_prompts: int, votes_per_prompt: int, samples_per_prompt: int) -> list[int]:
    """Return upstream-style first-k sample indices within each prompt vote block."""
    prompts = max(0, int(num_prompts))
    votes = max(0, int(votes_per_prompt))
    keep = max(0, min(int(samples_per_prompt), votes))
    selected: list[int] = []
    for prompt_idx in range(prompts):
        start = prompt_idx * votes
        selected.extend(range(start, start + keep))
    return selected


__all__ = [
    "LowRankAdapter",
    "detached_train_input",
    "normalize_answer_text",
    "select_rows_above_threshold",
    "select_top_k_per_prompt",
    "sentence_cross_entropy",
    "weighted_one_hot_kl_loss",
]
