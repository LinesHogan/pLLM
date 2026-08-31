#!/usr/bin/env python3
"""Explicit step-scoped session shared across one logical engine step.

vLLM 0.27 splits one logical step across ``_prepare_inputs`` /
``execute_model`` / ``sample_tokens`` (MRv1) or across
``execute_model`` -> ``execute_model_state`` -> ``sample_tokens`` (MRv2).
``StepSession`` is the single object that carries per-step state across
those stage boundaries so upper layers never care which runner family
produced the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import torch

from tllm import _core


class StepState(Enum):
    CREATED = auto()
    BACKEND_WARMUP = auto()
    INPUTS_READY = auto()
    FORWARD_DONE = auto()
    SAMPLING = auto()
    COMPLETED = auto()
    FAILED = auto()


_ALLOWED_TRANSITIONS: dict[StepState, frozenset[StepState]] = {
    StepState.CREATED: frozenset(
        {StepState.BACKEND_WARMUP, StepState.INPUTS_READY, StepState.FAILED}
    ),
    StepState.BACKEND_WARMUP: frozenset(),
    StepState.INPUTS_READY: frozenset({StepState.FORWARD_DONE, StepState.FAILED}),
    StepState.FORWARD_DONE: frozenset({StepState.SAMPLING, StepState.COMPLETED, StepState.FAILED}),
    StepState.SAMPLING: frozenset({StepState.COMPLETED, StepState.FAILED}),
    StepState.COMPLETED: frozenset(),
    StepState.FAILED: frozenset(),
}


class StepSessionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SpecRowMapping:
    """Strict mapping of one logits row under speculative decoding.

    ``sampled_index`` is 0 for the target (first sampled) position and
    1..k for the draft/bonus positions of the same request in this step.
    ``hidden_row`` is the packed hidden-state row that produced the logits
    row; consumers must use it (not request order) to index hidden taps.
    """

    request_id: str
    prompt_idx: int
    sample_idx: int
    sampled_index: int
    logit_row: int
    hidden_row: int


@dataclass
class StepSession:
    step_id: int
    backend_id: str

    request_ids: tuple[str, ...] = ()
    request_index: dict[str, int] = field(default_factory=dict)
    sample_index: torch.Tensor | None = None

    num_scheduled_tokens: Any = None
    # PyO3 SchedulerBatchPhase derived from vLLM scheduler columns. It stays
    # opaque here so runtime code cannot reinterpret scheduler semantics.
    batch_phase: _core.SchedulerBatchPhase | None = None
    positions: torch.Tensor | None = None
    logits_indices: torch.Tensor | None = None
    expanded_idx_mapping: torch.Tensor | None = None

    hidden_states: torch.Tensor | None = None
    sample_hidden_states: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    sampled_tokens: torch.Tensor | None = None

    # Strict per-logits-row mapping under speculative decoding (empty when
    # no request produced more than one logits row this step).
    spec_rows: tuple[SpecRowMapping, ...] = ()
    # Effective CUDA-graph mode of the runner for this step, resolved via
    # tllm.runtime.graph_policy (None when not yet resolved).
    graph_mode: Any = None

    state: StepState = StepState.CREATED
    error: str = ""

    def transition(self, target: StepState) -> None:
        if target is StepState.FAILED:
            self.state = StepState.FAILED
            return
        if target not in _ALLOWED_TRANSITIONS[self.state]:
            raise StepSessionError(
                f"illegal step transition {self.state.name} -> {target.name} "
                f"(step_id={self.step_id}, backend={self.backend_id})"
            )
        self.state = target

    def fail(self, error: str) -> None:
        self.error = str(error)
        self.state = StepState.FAILED


def get_step_session(runner: Any) -> StepSession | None:
    session = getattr(runner, "_tllm_step_session", None)
    return session if isinstance(session, StepSession) else None


def set_step_session(runner: Any, session: StepSession | None) -> None:
    setattr(runner, "_tllm_step_session", session)
