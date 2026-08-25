#!/usr/bin/env python3
"""Reusable Jacobi fixed-point execution with explicit approximation stats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from tllm.workgraph.recurrence import JacobiPolicy


StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class JacobiResult(Generic[StateT]):
    state: StateT
    rounds: int
    residual: float
    converged: bool


def run_jacobi(
    initial: StateT,
    *,
    step: Callable[[StateT], StateT],
    residual: Callable[[StateT, StateT], float],
    policy: JacobiPolicy,
) -> JacobiResult[StateT]:
    """Run a snapshot-to-snapshot iteration; never claim finite-round exactness."""

    current = initial
    last_residual = float("inf")
    for round_idx in range(1, int(policy.max_rounds) + 1):
        candidate = step(current)
        last_residual = float(residual(current, candidate))
        if last_residual < 0.0:
            raise ValueError("Jacobi residual must be non-negative")
        current = candidate
        if last_residual <= float(policy.tolerance):
            return JacobiResult(current, round_idx, last_residual, True)
    if policy.fail_if_not_converged:
        raise RuntimeError(
            f"Jacobi iteration did not converge after {policy.max_rounds} rounds; "
            f"residual={last_residual} tolerance={policy.tolerance}"
        )
    return JacobiResult(current, int(policy.max_rounds), last_residual, False)


__all__ = ["JacobiResult", "run_jacobi"]
