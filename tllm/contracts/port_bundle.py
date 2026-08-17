#!/usr/bin/env python3
"""Stable consumer-facing input bundle contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

# A bundle entry is always a declared port payload. The concrete payload
# types (tensor rows, DeviceTensorLease, HiddenBatch, RowBatchMeta, the
# per-port batch/sink classes) are owned by port modules and runtime stores;
# this contracts module must not import upward, so the alias is ``object``
# and readers narrow via isinstance — no silent Any attribute access.
PortPayload = object


@dataclass(frozen=True)
class BundleKey:
    engine_step_id: int
    phase: str
    request_id: str
    sample_idx: int


@dataclass(frozen=True)
class PortBundle:
    key: BundleKey
    entries: Mapping[str, PortPayload] = field(default_factory=dict[str, PortPayload])
