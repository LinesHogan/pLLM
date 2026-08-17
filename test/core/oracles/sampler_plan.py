#!/usr/bin/env python3
"""LEGACY oracle — 仅作 Rust core 对照基准,禁止运行时引用。

Verbatim transcription of the sampler-bridge decision tree that used to live
inside ``sample_with_optional_modifier`` (branch structure + mask-based row
partition). The Rust ``plan_sampler_step``/``partition_rows`` must match this
oracle over the whole matrix before the oracle can be deleted (sunset plan:
two development cycles with zero divergence + dual-runner parity + no ratio
regression).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OracleSamplerPlan:
    path: str  # full_dense_greedy|full_dense_random|full_exact_greedy|full_exact_random|partitioned
    use_candidate_kernel: bool
    dense_inplace_applied: bool
    base: str  # greedy_sampled|argmax|empty|""


@dataclass(frozen=True)
class OracleRowPartition:
    affected_greedy: tuple = ()
    affected_random: tuple = ()
    unaffected_greedy: tuple = ()
    unaffected_random: tuple = ()


def oracle_plan_sampler_step(
    *,
    backend: str,
    temperature_mode: str,
    n_rows: int,
    n_affected: int,
    n_greedy: int,
    greedy_sampled_available: bool,
    candidate_eligible: bool,
) -> OracleSamplerPlan:
    if backend == "pre_filter_dense" and n_affected == n_rows:
        if temperature_mode == "all_greedy" or n_greedy == n_rows:
            return OracleSamplerPlan("full_dense_greedy", False, False, "")
        if temperature_mode == "all_random" or n_greedy == 0:
            return OracleSamplerPlan("full_dense_random", False, False, "")
        # Legacy fallthrough: in-place dense modify already happened.
        return OracleSamplerPlan("partitioned", False, True, "empty")
    if backend in {"post_filter_exact", "post_filter_dense_cache"} and n_affected == n_rows:
        if temperature_mode == "all_greedy" or n_greedy == n_rows:
            return OracleSamplerPlan("full_exact_greedy", False, False, "")
        if temperature_mode == "all_random" or n_greedy == 0:
            return OracleSamplerPlan("full_exact_random", candidate_eligible, False, "")
        return OracleSamplerPlan("partitioned", False, False, "empty")
    if temperature_mode == "all_greedy":
        base = "greedy_sampled" if greedy_sampled_available else "argmax"
        return OracleSamplerPlan("partitioned", False, False, base)
    return OracleSamplerPlan("partitioned", False, False, "empty")


def oracle_partition_rows(
    *,
    n_rows: int,
    affected_rows: tuple,
    greedy_rows: tuple,
    greedy_sampled_available: bool,
) -> OracleRowPartition:
    """Mirrors the legacy mask + nonzero (ascending-order) partition."""
    affected = sorted(affected_rows)
    if not greedy_sampled_available:
        affected_set = set(affected)
        return OracleRowPartition(
            affected_random=tuple(affected),
            unaffected_random=tuple(r for r in range(n_rows) if r not in affected_set),
        )
    greedy = sorted(greedy_rows)
    affected_set = set(affected)
    greedy_set = set(greedy)
    return OracleRowPartition(
        affected_greedy=tuple(r for r in affected if r in greedy_set),
        affected_random=tuple(r for r in affected if r not in greedy_set),
        unaffected_greedy=tuple(r for r in greedy if r not in affected_set),
        unaffected_random=tuple(
            r for r in range(n_rows) if r not in affected_set and r not in greedy_set
        ),
    )
