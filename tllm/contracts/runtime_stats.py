#!/usr/bin/env python3
"""Wire-format runtime stats snapshot shared across process boundaries.

Pure data: no torch, no runtime objects. The physical executor owns the
counters; the control plane merges parent- and worker-side snapshots from
plain mappings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


def _wire_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key, 0)
    if not isinstance(value, (int, float, str)):
        raise TypeError(f"{key} must be an int-compatible wire value, got {type(value).__name__}")
    return int(value)


@dataclass(frozen=True, slots=True)
class WorkGraphRuntimeStats:
    active_plan_epoch: int = 0
    prepare_count: int = 0
    execute_count: int = 0
    finish_count: int = 0
    replay_count: int = 0
    eager_count: int = 0
    fallback_count: int = 0
    fence_wait_count: int = 0
    fence_record_count: int = 0
    bucket_miss_count: int = 0
    specialization_request_count: int = 0
    active_leases: int = 0
    retired_plans: int = 0
    deadline_flush_count: int = 0

    @classmethod
    def from_wire(cls, payload: Mapping[str, object]) -> "WorkGraphRuntimeStats":
        """Decode a worker-side numeric snapshot from the control plane."""

        return cls(
            active_plan_epoch=_wire_int(payload, "active_plan_epoch"),
            prepare_count=_wire_int(payload, "prepare_count"),
            execute_count=_wire_int(payload, "execute_count"),
            finish_count=_wire_int(payload, "finish_count"),
            replay_count=_wire_int(payload, "replay_count"),
            eager_count=_wire_int(payload, "eager_count"),
            fallback_count=_wire_int(payload, "fallback_count"),
            fence_wait_count=_wire_int(payload, "fence_wait_count"),
            fence_record_count=_wire_int(payload, "fence_record_count"),
            bucket_miss_count=_wire_int(payload, "bucket_miss_count"),
            specialization_request_count=_wire_int(payload, "specialization_request_count"),
            active_leases=_wire_int(payload, "active_leases"),
            retired_plans=_wire_int(payload, "retired_plans"),
            deadline_flush_count=_wire_int(payload, "deadline_flush_count"),
        )

    @classmethod
    def merge(cls, entries: tuple["WorkGraphRuntimeStats", ...]) -> "WorkGraphRuntimeStats":
        """Merge local and worker snapshots without exposing runtime objects."""

        if not entries:
            return cls()
        return cls(
            active_plan_epoch=max(entry.active_plan_epoch for entry in entries),
            prepare_count=sum(entry.prepare_count for entry in entries),
            execute_count=sum(entry.execute_count for entry in entries),
            finish_count=sum(entry.finish_count for entry in entries),
            replay_count=sum(entry.replay_count for entry in entries),
            eager_count=sum(entry.eager_count for entry in entries),
            fallback_count=sum(entry.fallback_count for entry in entries),
            fence_wait_count=sum(entry.fence_wait_count for entry in entries),
            fence_record_count=sum(entry.fence_record_count for entry in entries),
            bucket_miss_count=sum(entry.bucket_miss_count for entry in entries),
            specialization_request_count=sum(
                entry.specialization_request_count for entry in entries
            ),
            active_leases=sum(entry.active_leases for entry in entries),
            retired_plans=sum(entry.retired_plans for entry in entries),
            deadline_flush_count=sum(entry.deadline_flush_count for entry in entries),
        )
