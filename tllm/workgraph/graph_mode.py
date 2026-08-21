#!/usr/bin/env python3
"""CUDA-graph execution mode vocabulary for the workgraph IR.

Canonical home of :class:`GraphMode`. The runtime's ``graph_policy`` module
resolves the effective mode from a live runner and re-exports the enum.
"""

from __future__ import annotations

from enum import Enum, auto


class GraphMode(Enum):
    EAGER = auto()
    PIECEWISE = auto()  # attention eager, the rest captured/replayed
    FULL = auto()  # whole-step capture/replay
