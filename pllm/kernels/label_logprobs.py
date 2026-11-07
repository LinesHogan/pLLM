"""Log-probabilities of labels under a frozen LM head, one block of rows at a time; used by
pllm.model.CausalModel.token_logprobs.

Both directions are opaque custom ops so that torch.compile cannot schedule several blocks' [rows, vocab] logits at
once: only one block's logits exist at a time, and backward recomputes them instead of keeping them from forward.
"""

from typing import Protocol

import torch
from torch.nn import functional as F

# FP32 logits for 256 rows over a 152k vocabulary take about 150 MiB.
_BLOCK_ROWS = 256


def head_logits(hidden: torch.Tensor, weight: torch.Tensor, vocab_size: int, scale: float) -> torch.Tensor:
    """The LM head as vLLM's logits processor applies it, in the model dtype: drop padded vocabulary rows, scale."""
    return F.linear(hidden, weight)[..., :vocab_size] * scale


def _wide(logits: torch.Tensor) -> torch.Tensor:
    return logits.to(torch.promote_types(logits.dtype, torch.float32))


@torch.library.custom_op("pllm::label_logprobs", mutates_args=())
def label_logprobs(
    hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, vocab_size: int, scale: float, ignore_index: int
) -> torch.Tensor:
    """[rows] log-probabilities, in at least FP32, of each row's label; rows labelled ignore_index read 0."""
    return torch.cat(
        [
            -F.cross_entropy(
                _wide(head_logits(hidden[start : start + _BLOCK_ROWS], weight, vocab_size, scale)),
                labels[start : start + _BLOCK_ROWS],
                ignore_index=ignore_index,
                reduction="none",
            )
            for start in range(0, len(labels), _BLOCK_ROWS)
        ]
    )


def _label_logprobs_meta(
    hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, vocab_size: int, scale: float, ignore_index: int
) -> torch.Tensor:
    return hidden.new_empty(labels.shape, dtype=torch.promote_types(hidden.dtype, torch.float32))


@torch.library.custom_op("pllm::label_logprobs_backward", mutates_args=())
def label_logprobs_backward(
    grad: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    scale: float,
    ignore_index: int,
) -> torch.Tensor:
    """The hidden gradient: grad * (onehot(label) - softmax), rounded to the logits dtype as autograd would."""
    blocks: list[torch.Tensor] = []
    for start in range(0, len(labels), _BLOCK_ROWS):
        rows, targets = hidden[start : start + _BLOCK_ROWS], labels[start : start + _BLOCK_ROWS]
        logits = head_logits(rows, weight, vocab_size, scale)
        upstream = grad[start : start + _BLOCK_ROWS].where(targets != ignore_index, 0)[:, None]
        delta = _wide(logits).softmax(-1).mul_(-upstream)
        delta.scatter_add_(-1, targets.clamp_min(0)[:, None], upstream)
        blocks.append((delta.to(logits.dtype) * scale) @ weight[:vocab_size])
    return torch.cat(blocks)


def _label_logprobs_backward_meta(
    grad: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    scale: float,
    ignore_index: int,
) -> torch.Tensor:
    return torch.empty_like(hidden)


class _Context(Protocol):
    saved_tensors: tuple[torch.Tensor, ...]
    constants: tuple[int, float, int]

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


def _save(
    ctx: _Context, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float, int], output: torch.Tensor
) -> None:
    ctx.save_for_backward(*inputs[:3])
    ctx.constants = inputs[3:]


def _backward(ctx: _Context, grad: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
    hidden, weight, labels = ctx.saved_tensors
    return label_logprobs_backward(grad, hidden, weight, labels, *ctx.constants), None, None, None, None, None


label_logprobs.register_fake(_label_logprobs_meta)  # pyright: ignore[reportUnknownMemberType]
label_logprobs_backward.register_fake(_label_logprobs_backward_meta)  # pyright: ignore[reportUnknownMemberType]
label_logprobs.register_autograd(_backward, setup_context=_save)  # pyright: ignore[reportUnknownMemberType]
