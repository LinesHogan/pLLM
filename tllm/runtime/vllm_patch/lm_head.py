"""Resolve a third-party model's output projection at one runtime boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import torch


class _ModelNode(Protocol):
    model: object


class _OutputEmbeddingOwner(Protocol):
    def get_output_embeddings(self) -> object: ...


class _RawLMHead(Protocol):
    weight: object
    bias: object


@dataclass(frozen=True, slots=True)
class ResolvedLMHead:
    weight: torch.Tensor
    bias: torch.Tensor | None


def _resolve_candidate(candidate: object) -> ResolvedLMHead | None:
    try:
        head = cast(_RawLMHead, candidate)
        weight = head.weight
    except AttributeError:
        return None
    if not isinstance(weight, torch.Tensor):
        return None
    bias = getattr(head, "bias", None)
    if bias is not None and not isinstance(bias, torch.Tensor):
        return None
    return ResolvedLMHead(weight=weight, bias=bias)


def resolve_lm_head(model: object) -> ResolvedLMHead:
    candidates: list[object] = []
    node = model
    for _ in range(3):
        candidates.append(getattr(node, "lm_head", None))
        try:
            node = cast(_ModelNode, node).model
        except AttributeError:
            break
    try:
        candidates.append(
            cast(_OutputEmbeddingOwner, model).get_output_embeddings()
        )
    except AttributeError:
        pass
    for candidate in candidates:
        resolved = _resolve_candidate(candidate)
        if resolved is not None:
            return resolved
    raise RuntimeError("sampler output could not resolve the model output projection")


__all__ = ["ResolvedLMHead", "resolve_lm_head"]
