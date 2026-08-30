"""Explicit per-step values exposed to signed sampler input ports."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.runtime.sampler_bridge.types import SamplerStepView
from tllm.workload.runtime_values import SKIPPED, RuntimeValueMarker


@dataclass(frozen=True, slots=True)
class _Waiting:
    pass


@dataclass(frozen=True, slots=True)
class _BeforeDraw:
    view: SamplerStepView


@dataclass(frozen=True, slots=True)
class _AfterDraw:
    view: SamplerStepView
    tokens: torch.Tensor


@dataclass(frozen=True, slots=True)
class _IdentityWaiting:
    pass


@dataclass(frozen=True, slots=True)
class _PreparedIdentity:
    step_id: int
    request_ids: tuple[str, ...]
    sample_indices: tuple[int, ...]


class SamplerStepContext:
    """Own early row identity separately from draw-time tensor values.

    Decode localization publishes identity before the scheduler may run a
    Region placed ahead of ``sampler.pre_draw``. Logits and tokens become
    available later from the exact sampler view for the same signed step.
    """

    def __init__(self) -> None:
        self._state: _Waiting | _BeforeDraw | _AfterDraw = _Waiting()
        self._identity: _IdentityWaiting | _PreparedIdentity = (
            _IdentityWaiting()
        )

    def prepare_identity(
        self,
        step_id: int,
        request_ids: tuple[str, ...],
        sample_indices: tuple[int, ...],
    ) -> None:
        if len(request_ids) != len(sample_indices):
            raise ValueError(
                "sampler request and sample identity rows must match"
            )
        self._identity = _PreparedIdentity(
            step_id, request_ids, sample_indices
        )

    def prepare(self, view: SamplerStepView) -> None:
        self._state = _BeforeDraw(view)

    def sampled(self, view: SamplerStepView, tokens: torch.Tensor) -> None:
        self._state = _AfterDraw(view, tokens.reshape(-1).to(torch.long))

    def logits(self, step_id: int) -> torch.Tensor | RuntimeValueMarker:
        match self._state:
            case _BeforeDraw(view=view) | _AfterDraw(view=view) if (
                view.engine_step_id == step_id
            ):
                return view.logits
            case _:
                return SKIPPED

    def tokens(self, step_id: int) -> torch.Tensor | RuntimeValueMarker:
        match self._state:
            case _AfterDraw(view=view, tokens=tokens) if (
                view.engine_step_id == step_id
            ):
                return tokens
            case _:
                return SKIPPED

    def request_ids(
        self, step_id: int
    ) -> tuple[str, ...] | RuntimeValueMarker:
        match self._identity:
            case _PreparedIdentity(
                step_id=prepared_step,
                request_ids=request_ids,
            ) if prepared_step == step_id and request_ids:
                return request_ids
            case _:
                return SKIPPED

    def sample_indices(
        self, step_id: int
    ) -> tuple[int, ...] | RuntimeValueMarker:
        match self._identity:
            case _PreparedIdentity(
                step_id=prepared_step,
                sample_indices=sample_indices,
            ) if prepared_step == step_id and sample_indices:
                return sample_indices
            case _:
                return SKIPPED


__all__ = ["SamplerStepContext"]
