#!/usr/bin/env python3
"""lm_head resolution for sampler modifiers (model-structure introspection).

The sampler bridge needs the output projection to score a predicted hidden
state; model classes nest it differently (``model.lm_head``,
``model.model.lm_head``, ``get_output_embeddings()``), so resolution walks
the known shapes once, here, instead of in every workload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import torch


class _LMHeadOwner(Protocol):
    lm_head: object
    model: object


class _OutputEmbeddingOwner(Protocol):
    def get_output_embeddings(self) -> object:
        ...


class _RawLMHead(Protocol):
    weight: object
    bias: object


@dataclass(frozen=True, slots=True)
class ResolvedLMHead:
    """Tensor parameters needed by the sampler bridge."""

    weight: torch.Tensor
    bias: torch.Tensor | None


def _with_weight(candidate: object) -> ResolvedLMHead | None:
    try:
        raw_head = cast(_RawLMHead, candidate)
        weight = raw_head.weight
    except AttributeError:
        return None
    if not isinstance(weight, torch.Tensor):
        return None
    try:
        bias = raw_head.bias
    except AttributeError:
        bias = None
    if bias is not None and not isinstance(bias, torch.Tensor):
        return None
    return ResolvedLMHead(weight=weight, bias=bias)


def resolve_lm_head(model: object) -> ResolvedLMHead:
    candidates: list[object] = []
    owner: object = model
    for _ in range(3):  # model, model.model, model.model.model
        try:
            candidates.append(cast(_LMHeadOwner, owner).lm_head)
        except AttributeError:
            pass
        try:
            owner = cast(_LMHeadOwner, owner).model
        except AttributeError:
            break
    try:
        candidates.append(cast(_OutputEmbeddingOwner, model).get_output_embeddings())
    except AttributeError:
        pass
    for candidate in candidates:
        resolved = _with_weight(candidate)
        if resolved is not None:
            return resolved
    raise RuntimeError("sampler modifier could not resolve lm_head from model")


__all__ = ["ResolvedLMHead", "resolve_lm_head"]
