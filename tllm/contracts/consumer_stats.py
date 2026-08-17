#!/usr/bin/env python3
"""Consumer-facing training statistics contract.

The runtime reports consumer training progress through this neutral type so
it never has to import a specific consumer package. Consumers may alias
their own stats name to :class:`TrainingStats` (ESamp does).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TrainingStats:
    loss_avg: float
    loss_count: int
    trace_losses: tuple[float, ...] = ()
