#!/usr/bin/env python3
"""Formal sampler-step port and candidate-modifier contract vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, runtime_checkable

import torch

from tllm.ports.base import Locator, PortKind, PortRead

SamplerStepScope = Literal["current"]

CanonicalSamplerBackend = Literal["post_filter_exact", "pre_filter_dense", "post_filter_dense_cache"]
SamplerBackend = Literal[
    "post_filter_exact",
    "post_filter_exact_minp",
    "post_filter_exact_torch",
    "pre_filter_dense",
    "post_filter_dense_cache",
]
SAMPLER_BACKEND_CHOICES: tuple[str, ...] = (
    "post_filter_exact",
    "post_filter_exact_minp",
    "post_filter_exact_torch",
    "pre_filter_dense",
    "post_filter_dense_cache",
)
TemperatureMode = Literal["all_random", "all_greedy", "mixed"]
CandidateSamplingMode = Literal["random", "greedy", "mixed"]


@dataclass(frozen=True)
class SamplerFilterSpec:
    top_k: torch.Tensor | None
    top_p: torch.Tensor | None
    min_p: torch.Tensor | None
    temperature_mode: TemperatureMode
    has_generators: bool = False

    @property
    def has_top_k(self) -> bool:
        return self.top_k is not None

    @property
    def has_top_p(self) -> bool:
        return self.top_p is not None

    @property
    def has_min_p(self) -> bool:
        return self.min_p is not None


@dataclass(frozen=True)
class CandidateModifierRequest:
    logits: torch.Tensor
    affected_row_ids: torch.Tensor
    filter_spec: SamplerFilterSpec
    sampling_mode: CandidateSamplingMode


@dataclass(frozen=True)
class CandidateModifierState:
    beta: float
    backend: SamplerBackend
    affected_row_ids: torch.Tensor
    pred_hidden: torch.Tensor
    lm_head_weight: torch.Tensor
    lm_head_bias: torch.Tensor | None
    precomputed_dense_logits: torch.Tensor | None = None
    pred_hidden_row_map: torch.Tensor | None = None


@dataclass(frozen=True)
class CandidateSampleResult:
    sampled_token_ids: torch.Tensor
    debug_stats: Mapping[str, int | float | str] | None = None


SamplerModifierState = CandidateModifierState


@dataclass(frozen=True)
class SamplerLocator(Locator):
    step_scope: SamplerStepScope

    def __post_init__(self) -> None:
        if self.step_scope != "current":
            raise ValueError(f"unsupported sampler step_scope: {self.step_scope}")


class Sampler:
    KIND = PortKind.SAMPLER
    READABLE = True
    WRITABLE = False
    SUPPORTED_PHASES = ("decode",)
    BACKING_VLLM_STRUCT = (
        "same-step decode sampler view including aligned logits, request metadata, "
        "and source hidden rows immediately before final sampling."
    )

    @staticmethod
    def read(*, step_scope: SamplerStepScope = "current") -> PortRead:
        return PortRead(kind=Sampler.KIND, locator=SamplerLocator(step_scope=step_scope))


@runtime_checkable
class CandidateModifierProvider(Protocol):
    """Public sampler-port provider contract for candidate-level modifiers."""

    def is_active(self) -> bool:
        ...

    def prepare_candidate_state(
        self,
        request: CandidateModifierRequest,
    ) -> CandidateModifierState | None:
        ...
