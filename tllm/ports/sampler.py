#!/usr/bin/env python3
"""Formal sampler-step port and candidate-modifier contract vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, NamedTuple

import torch

from tllm.contracts.packed_rows import PackedSamplerRowBatch
from tllm.ports.base import Locator, PortKind, PortRead

SamplerStepScope = Literal["current"]
SamplerPhase = Literal["decode", "packed"]

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


@dataclass(frozen=True)
class SamplerFilterSpec:
    top_k: torch.Tensor | None
    top_p: torch.Tensor | None
    min_p: torch.Tensor | None
    temperature_mode: TemperatureMode
    has_generators: bool = False


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


class SamplerHiddenReweightBatch(NamedTuple):
    """Compact hidden predictions for a sorted subset of one full step."""

    pred_hidden: torch.Tensor
    affected_row_ids: tuple[int, ...]
    total_rows: int


@dataclass(frozen=True, slots=True)
class PackedSamplerGuidanceBatch:
    """Hidden predictions indexed in one explicit packed sampler row space."""

    sampler_rows: PackedSamplerRowBatch
    pred_hidden: torch.Tensor
    affected_row_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.pred_hidden.ndim != 2
            or int(self.pred_hidden.shape[0]) != len(self.affected_row_ids)
        ):
            raise ValueError(
                "packed sampler guidance must align predictions with affected rows"
            )
        if self.affected_row_ids != tuple(
            sorted(set(self.affected_row_ids))
        ):
            raise ValueError(
                "packed sampler guidance rows must be sorted and unique"
            )
        if any(
            row_id < 0 or row_id >= len(self.sampler_rows.links)
            for row_id in self.affected_row_ids
        ):
            raise ValueError("packed sampler guidance row is out of bounds")

    @property
    def total_rows(self) -> int:
        return len(self.sampler_rows.links)


@dataclass(frozen=True)
class CandidateSampleResult:
    sampled_token_ids: torch.Tensor
    debug_stats: Mapping[str, int | float | str] | None = None


SamplerModifierState = CandidateModifierState
SamplerOutputValue = (
    torch.Tensor | SamplerHiddenReweightBatch | PackedSamplerGuidanceBatch
)


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
    SUPPORTED_PHASES: tuple[SamplerPhase, ...] = ("decode", "packed")
    BACKING_VLLM_STRUCT = (
        "same-step sampler view containing either aligned legacy decode rows or "
        "explicit packed prefill/decode links immediately before final sampling."
    )

    @staticmethod
    def read(*, step_scope: SamplerStepScope = "current") -> PortRead:
        return PortRead(kind=Sampler.KIND, locator=SamplerLocator(step_scope=step_scope))
