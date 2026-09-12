"""Closed loop instructions lowered from one verified affine schedule."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from pllm.contracts.region_schedule import Statement
from pllm.contracts.control_instructions import ControlAction


class ScheduleOperator(StrEnum):
    AND = "and"
    AND_THEN = "and_then"
    OR = "or"
    OR_ELSE = "or_else"
    MAX = "max"
    MIN = "min"
    NEG = "neg"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    EXACT_DIV = "exact_div"
    FLOOR_DIV = "floor_div"
    POSITIVE_DIV = "positive_div"
    POSITIVE_MOD = "positive_mod"
    ZERO_MOD = "zero_mod"
    CONDITIONAL = "conditional"
    SELECT = "select"
    EQ = "eq"
    LE = "le"
    LT = "lt"
    GE = "ge"
    GT = "gt"


@dataclass(frozen=True, slots=True)
class ScheduleInteger:
    value: int


@dataclass(frozen=True, slots=True)
class ScheduleIterator:
    name: str


@dataclass(frozen=True, slots=True)
class ScheduleApply:
    operator: ScheduleOperator
    arguments: tuple[ScheduleExpression, ...]


ScheduleExpression: TypeAlias = ScheduleInteger | ScheduleIterator | ScheduleApply


@dataclass(frozen=True, slots=True)
class ScheduleInvoke:
    statement_id: int
    coordinates: tuple[ScheduleExpression, ...]


@dataclass(frozen=True, slots=True)
class ScheduleLoop:
    iterator: str
    initial: ScheduleExpression
    condition: ScheduleExpression
    step: ScheduleExpression
    body: ScheduleInstruction


@dataclass(frozen=True, slots=True)
class ScheduleSequence:
    children: tuple[ScheduleInstruction, ...]


@dataclass(frozen=True, slots=True)
class ScheduleGuard:
    condition: ScheduleExpression
    if_true: ScheduleInstruction
    if_false: ScheduleInstruction


ScheduleInstruction: TypeAlias = ScheduleInvoke | ScheduleLoop | ScheduleSequence | ScheduleGuard


@dataclass(frozen=True, slots=True)
class LoweredSchedule:
    version: int
    program_digest: str
    schedule_map: str
    statements: tuple[Statement, ...]
    body: ScheduleInstruction
    actions: tuple[ControlAction, ...]
    signature: str


@dataclass(frozen=True, slots=True)
class StatementInvocation:
    statement_id: int
    coordinates: tuple[int, ...]
