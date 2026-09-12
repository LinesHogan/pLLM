"""Closed slot instructions for structural control inside a scheduled statement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from pllm.contracts.region_program import DataType, Op


@dataclass(frozen=True, slots=True)
class ControlSlot:
    id: int
    type: DataType


@dataclass(frozen=True, slots=True)
class ControlPrimitive:
    operation: Op


@dataclass(frozen=True, slots=True)
class ControlIf:
    then_block: ControlBlock
    else_block: ControlBlock


@dataclass(frozen=True, slots=True)
class ControlWhile:
    limit: int
    condition: ControlBlock
    body: ControlBlock


@dataclass(frozen=True, slots=True)
class ControlScan:
    count: int | str
    body: ControlBlock
    invariants: int = 0


@dataclass(frozen=True, slots=True)
class ControlMap:
    count: int | str
    body: ControlBlock


ControlAction: TypeAlias = ControlPrimitive | ControlIf | ControlWhile | ControlScan | ControlMap


@dataclass(frozen=True, slots=True)
class ControlStep:
    inputs: tuple[int, ...]
    outputs: tuple[ControlSlot, ...]
    action: ControlAction
    releases: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ControlBlock:
    arguments: tuple[ControlSlot, ...]
    steps: tuple[ControlStep, ...]
    results: tuple[int, ...]
    discard_arguments: tuple[int, ...]
