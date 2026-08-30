#!/usr/bin/env python3
"""Provider protocol for same-step sampler intervention."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from tllm.runtime.sampler_bridge.types import CandidateModifierState, SamplerStepView


class SamplerInterventionStage(StrEnum):
    PRE_FILTER = "pre_filter"
    POST_FILTER = "post_filter"


@runtime_checkable
class SamplerModifierProvider(Protocol):
    def is_active(self) -> bool:
        ...

    def intervention_stage(self) -> SamplerInterventionStage:
        ...

    def prepare_step(self, view: SamplerStepView) -> CandidateModifierState | None:
        ...


def validate_prepared_intervention(
    provider: SamplerModifierProvider,
    state: CandidateModifierState,
) -> None:
    expected = (
        SamplerInterventionStage.PRE_FILTER
        if state.backend == "pre_filter_dense"
        else SamplerInterventionStage.POST_FILTER
    )
    if provider.intervention_stage() is not expected:
        raise RuntimeError(
            "sampler provider stage disagrees with its prepared backend"
        )


__all__ = [
    "SamplerInterventionStage",
    "SamplerModifierProvider",
    "validate_prepared_intervention",
]
