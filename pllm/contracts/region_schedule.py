"""Derived affine facts and candidates; the source Program remains authoritative."""

from dataclasses import dataclass

from pllm.contracts.region_program import DataType


@dataclass(frozen=True, slots=True)
class ValueSite:
    """One mathematical definition; resource order has no value-table entry."""

    id: int
    path: tuple[int, ...]
    name: str
    type: DataType
    extents: tuple[int | str, ...]
    bank_shape: tuple[int | str, ...]


@dataclass(frozen=True, slots=True)
class Statement:
    id: int
    path: tuple[int, ...]
    extents: tuple[int | str, ...]
    # Invariant inputs can refer directly to an enclosing scope's definition.
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    opaque: bool


@dataclass(frozen=True, slots=True)
class ScheduleCandidate:
    source: str
    schedule_map: str
    schedule_tree: str


@dataclass(frozen=True, slots=True)
class RegionTilingExploration:
    program_digest: str
    candidates: tuple[ScheduleCandidate, ...]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResourceDependencies:
    reads: str
    writes: str
    raw: str
    war: str
    waw: str


@dataclass(frozen=True, slots=True)
class RegionAnalysis:
    program_digest: str
    values: tuple[ValueSite, ...]
    statements: tuple[Statement, ...]
    domain: str
    aliases: str
    value_sources: str
    arguments: tuple[int, ...]
    results: tuple[int, ...]
    scalar_values: str
    value_consumers: str
    data_dependencies: str
    effect_dependencies: str
    resource_dependencies: ResourceDependencies
    dependencies: str
    reference_schedule: str
    candidates: tuple[ScheduleCandidate, ...]
    exact_provenance: bool
    diagnostics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScheduleInstance:
    statement_id: int
    coordinates: tuple[int, ...]
    time: tuple[int, ...]
