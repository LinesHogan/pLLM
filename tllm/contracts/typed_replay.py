"""Typed model-region replay vocabulary shared by backend drivers and binders.

A backend driver owns the concrete replay machinery (hook installation,
forward-context slicing); a consumer-owned binder owns Region composition.
This module is the contract between them: the typed replay state, the
per-forward step plan, the handle protocol the driver implements, the bound
call resources a stager hands to its executor, and the signed primitive
keys.  Neither side may import the other; both import this module.

The primitive keys are signed artifact vocabulary.  Their string values are
part of cold artifact identity and must never change in place.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import torch

VLLM_LAYER_REPLAY_PRIMITIVE = "tllm.vllm:typed_layer_replay@1"
VLLM_FINAL_NORM_PRIMITIVE = "tllm.vllm:typed_final_norm@1"


@dataclass(frozen=True, slots=True)
class LayerReplayState:
    """One decoder-layer input/output, preserving the residual stream."""

    hidden: torch.Tensor
    residual: torch.Tensor | None

    def stream(self) -> torch.Tensor:
        return self.hidden if self.residual is None else self.hidden + self.residual


class TypedLayerReplayHandle(Protocol):
    """Backend-installed typed replay hook for one packed model forward.

    The concrete handle is driver-owned; consumers see only this surface.
    """

    def prepare(self, step: TypedLayerReplayStep) -> None:
        """Bind the workload-staged replay plan for the next forward."""
        ...

    def seed_state(self, packed_row: int) -> LayerReplayState:
        """Return the captured model input for one packed row."""
        ...

    def replay_layer(
        self,
        state: LayerReplayState,
        *,
        layer: int,
        request_ordinal: int,
        packed_row: int,
        token_position: int,
    ) -> LayerReplayState:
        """Run one decoder layer on one packed row inside the active forward."""
        ...

    def normalize(self, state: LayerReplayState) -> torch.Tensor:
        """Run the terminal norm on one packed row."""
        ...


@dataclass(frozen=True, slots=True)
class TypedLayerReplayRequest:
    """One request-local callback bound by a workload-owned stager."""

    request_id: str
    packed_rows: tuple[int, ...]
    replay: Callable[[TypedLayerReplayHandle], Mapping[int, torch.Tensor]]

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
            raise TypeError("typed layer replay request_id must be a non-empty str")
        if not self.packed_rows or any(
            isinstance(row, bool) or not isinstance(row, int) or row < 0  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
            for row in self.packed_rows
        ):
            raise ValueError("typed layer replay packed_rows must be non-empty int rows")
        if len(self.packed_rows) != len(set(self.packed_rows)):
            raise ValueError("typed layer replay packed_rows must be unique")
        if not callable(self.replay):
            raise TypeError("typed layer replay request callback must be callable")


@dataclass(frozen=True, slots=True)
class TypedLayerReplayStep:
    """Selected row callbacks for one packed model forward."""

    total_rows: int
    requests: tuple[TypedLayerReplayRequest, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.total_rows, bool)
            or not isinstance(self.total_rows, int)  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
            or self.total_rows < 1
        ):
            raise ValueError("typed layer replay total_rows must be a positive int")
        if not self.requests or not all(
            isinstance(request, TypedLayerReplayRequest) for request in self.requests  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
        ):
            raise TypeError("typed layer replay requests must be non-empty typed requests")
        rows = tuple(row for request in self.requests for row in request.packed_rows)
        if (
            len(rows) != len(set(rows))
            or any(row >= self.total_rows for row in rows)
        ):
            raise ValueError(
                "typed layer replay requests must select unique in-range packed rows"
            )


@dataclass(frozen=True, slots=True)
class LayerReplayCall:
    """One pre-bound layer invocation selected as a typed resource."""

    handle: TypedLayerReplayHandle
    layer: int
    request_ordinal: int
    packed_row: int
    token_position: int

    def execute(self, state: LayerReplayState) -> LayerReplayState:
        return self.handle.replay_layer(
            state,
            layer=self.layer,
            request_ordinal=self.request_ordinal,
            packed_row=self.packed_row,
            token_position=self.token_position,
        )


@dataclass(frozen=True, slots=True)
class FinalNormCall:
    """The pre-bound terminal norm resource for one typed model Region."""

    handle: TypedLayerReplayHandle

    def execute(self, state: LayerReplayState) -> torch.Tensor:
        return self.handle.normalize(state)


__all__ = [
    "FinalNormCall",
    "LayerReplayCall",
    "LayerReplayState",
    "TypedLayerReplayHandle",
    "TypedLayerReplayRequest",
    "TypedLayerReplayStep",
    "VLLM_FINAL_NORM_PRIMITIVE",
    "VLLM_LAYER_REPLAY_PRIMITIVE",
]
