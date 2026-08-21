#!/usr/bin/env python3
"""lm_head resolution shared by ESamp sampler providers (legacy and program)."""

from __future__ import annotations

from typing import Any


def _with_weight(candidate: Any) -> Any | None:
    try:
        candidate.weight
    except AttributeError:
        return None
    return candidate


def resolve_lm_head(model: Any) -> Any:
    candidates: list[Any] = []
    owner = model
    for _ in range(3):  # model, model.model, model.model.model
        try:
            candidates.append(owner.lm_head)
        except AttributeError:
            pass
        try:
            owner = owner.model
        except AttributeError:
            break
    try:
        candidates.append(model.get_output_embeddings())
    except AttributeError:
        pass
    for candidate in candidates:
        resolved = _with_weight(candidate)
        if resolved is not None:
            return resolved
    raise RuntimeError("ESamp sampler provider could not resolve lm_head from model")


__all__ = ["resolve_lm_head"]
