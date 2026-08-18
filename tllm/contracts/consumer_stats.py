#!/usr/bin/env python3
"""Consumer-facing training statistics contract.

The runtime reports consumer training progress through this neutral type so
it never has to import a specific consumer package. Consumers may alias
their own stats name to :class:`TrainingStats` (ESamp does).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(slots=True, frozen=True)
class TrainingStats:
    loss_avg: float
    loss_count: int
    trace_losses: tuple[float, ...] = ()

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "TrainingStats":
        return cls(
            loss_avg=float(payload.get("loss_avg", 0.0)),
            loss_count=int(payload.get("loss_count", 0)),
            trace_losses=tuple(float(value) for value in payload.get("trace_losses", ())),
        )

    @classmethod
    def merge(cls, entries: Iterable["TrainingStats"]) -> "TrainingStats":
        total_loss = 0.0
        total_count = 0
        trace_losses: list[float] = []
        for entry in entries:
            count = int(entry.loss_count)
            total_loss += float(entry.loss_avg) * count
            total_count += count
            trace_losses.extend(float(value) for value in entry.trace_losses)
        return cls(
            loss_avg=total_loss / total_count if total_count > 0 else 0.0,
            loss_count=total_count,
            trace_losses=tuple(trace_losses),
        )


def training_stats_map_from_wire(payload: Mapping[str, Any]) -> dict[int, TrainingStats]:
    return {int(key): TrainingStats.from_wire(value) for key, value in payload.items()}


def merge_training_stats_maps(
    entries: Iterable[Mapping[int, TrainingStats]],
) -> dict[int, TrainingStats]:
    grouped: dict[int, list[TrainingStats]] = {}
    for entry in entries:
        for request_key, stats in entry.items():
            grouped.setdefault(int(request_key), []).append(stats)
    return {request_key: TrainingStats.merge(stats) for request_key, stats in grouped.items()}
