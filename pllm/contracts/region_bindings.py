"""Closed source accessors and schedule-relative mathematical value lifetimes."""

from dataclasses import dataclass

from pllm.contracts.region_schedule import ValueSite
from pllm.contracts.schedule_instructions import ScheduleExpression


@dataclass(frozen=True, slots=True)
class AccessCase:
    target_id: int
    condition: ScheduleExpression
    coordinates: tuple[ScheduleExpression, ...]


@dataclass(frozen=True, slots=True)
class AccessTable:
    input_id: int
    rank: int
    cases: tuple[AccessCase, ...]


@dataclass(frozen=True, slots=True)
class RegionBindings:
    version: int
    program_digest: str
    schedule_map: str
    values: tuple[ValueSite, ...]
    arguments: tuple[int, ...]
    results: tuple[int, ...]
    sources: tuple[AccessTable, ...]
    scalars: tuple[AccessTable, ...]
    times: tuple[AccessTable, ...]
    last_uses: tuple[AccessTable, ...]
    retained: tuple[AccessTable, ...]
    signature: str
