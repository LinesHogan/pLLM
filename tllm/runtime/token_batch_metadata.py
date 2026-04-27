#!/usr/bin/env python3
"""Typed per-token metadata for model-forward runtime services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__test__ = False

RuntimeTokenPhase = Literal["prefill", "decode"]
RuntimeTokenRowLayout = Literal["packed", "localized", "compact_first_per_prompt"]


def _require_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _require_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    return value


@dataclass(frozen=True)
class RequestSampleKey:
    request_id: str
    prompt_idx: int
    sample_idx: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _require_str(self.request_id, field_name="request_id"))
        object.__setattr__(self, "prompt_idx", _require_int(self.prompt_idx, field_name="prompt_idx"))
        object.__setattr__(self, "sample_idx", _require_int(self.sample_idx, field_name="sample_idx"))


@dataclass(frozen=True)
class PackedTokenRow:
    packed_row_idx: int
    logical_row_idx: int
    token_offset: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "packed_row_idx", _require_int(self.packed_row_idx, field_name="packed_row_idx"))
        object.__setattr__(self, "logical_row_idx", _require_int(self.logical_row_idx, field_name="logical_row_idx"))
        if self.token_offset is not None:
            object.__setattr__(self, "token_offset", _require_int(self.token_offset, field_name="token_offset"))


@dataclass(frozen=True)
class RuntimeTokenRow:
    key: RequestSampleKey
    row: PackedTokenRow
    phase: RuntimeTokenPhase
    engine_step_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, RequestSampleKey):
            raise TypeError("key must be RequestSampleKey")
        if not isinstance(self.row, PackedTokenRow):
            raise TypeError("row must be PackedTokenRow")
        if self.phase not in ("prefill", "decode"):
            raise ValueError("phase must be 'prefill' or 'decode'")
        object.__setattr__(self, "engine_step_id", _require_int(self.engine_step_id, field_name="engine_step_id"))


@dataclass(frozen=True)
class RuntimeTokenBatchMeta:
    rows: tuple[RuntimeTokenRow, ...]
    row_layout: RuntimeTokenRowLayout = "packed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))
        for row in self.rows:
            if not isinstance(row, RuntimeTokenRow):
                raise TypeError("rows must contain RuntimeTokenRow values")
        if self.row_layout not in ("packed", "localized", "compact_first_per_prompt"):
            raise ValueError("row_layout must name an explicit coordinate system")


__all__ = [
    "PackedTokenRow",
    "RequestSampleKey",
    "RuntimeTokenBatchMeta",
    "RuntimeTokenPhase",
    "RuntimeTokenRow",
    "RuntimeTokenRowLayout",
]
