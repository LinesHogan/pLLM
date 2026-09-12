"""Native verification boundary for the structured source program."""

from pllm import _core
from pllm.contracts.region_program import Program, verify
from pllm.contracts.region_bindings import RegionBindings
from pllm.contracts.region_schedule import RegionAnalysis, RegionTilingExploration, ScheduleInstance
from pllm.contracts.schedule_instructions import LoweredSchedule
from pllm.contracts.region_retention import RegionRetention, RetainedAllocation
from pllm.execution.codec import decode_boundary, encode_boundary


def validate_region_program(program: Program) -> None:
    """Validate a Python source through the shared native authority."""
    verify(program)


def analyze_region_retention(program: Program, schedule_map: str, imports: tuple[RetainedAllocation, ...]) -> RegionRetention:
    return decode_boundary(_core.analyze_region_retention(encode_boundary(program), schedule_map, encode_boundary(imports)), RegionRetention)


def validate_region_retention(program: Program, imports: tuple[RetainedAllocation, ...], retention: RegionRetention) -> None:
    _core.validate_region_retention(encode_boundary(program), encode_boundary(imports), encode_boundary(retention))


def analyze_region_program(program: Program, *, optimize: bool = True) -> RegionAnalysis:
    """Derive dependencies; reference-only binding can omit candidate search."""
    return decode_boundary(_core.analyze_region_program(encode_boundary(program), optimize), RegionAnalysis)


def explore_region_tiling(program: Program, *, tile_sizes: tuple[int, ...] = (2, 4, 8, 16),
                          max_candidates: int = 8) -> RegionTilingExploration:
    """Tile source-derived permutable bands, then recheck full-domain legality."""
    return decode_boundary(_core.explore_region_tiling(encode_boundary(program), encode_boundary(tile_sizes), max_candidates),
                           RegionTilingExploration)


def explore_region_schedules(program: Program, *, max_candidates: int = 8) -> RegionTilingExploration:
    """Share global, window and tiled schedules, with the reference first.

    Native derivation checks legality and removes equivalent execution orders.
    Neither the catalog order nor its size is a performance or capacity claim.
    """
    return decode_boundary(_core.explore_region_schedules(encode_boundary(program), max_candidates), RegionTilingExploration)


def validate_region_schedule(program: Program, schedule_map: str) -> None:
    """Re-derive dependencies instead of trusting edges supplied by a candidate."""
    _core.validate_region_schedule(encode_boundary(program), schedule_map)


def explore_region_windows(program: Program, *, sizes: tuple[int, ...] = (1, 2, 4, 8), depth: int = 1,
                           max_candidates: int = 8) -> RegionTilingExploration:
    """Preserve reference loop windows while allowing ISL to reorder inside them."""
    return decode_boundary(_core.explore_region_windows(encode_boundary(program), encode_boundary(sizes), depth, max_candidates),
                           RegionTilingExploration)


def preview_region_schedule(program: Program, schedule_map: str, *, max_instances: int = 4096) -> tuple[ScheduleInstance, ...]:
    """Bounded diagnostic expansion, not the physical lowering or runtime format."""
    return decode_boundary(_core.preview_region_schedule(encode_boundary(program), schedule_map, max_instances),
                           tuple[ScheduleInstance, ...])


def lower_region_schedule(program: Program, schedule_map: str) -> LoweredSchedule:
    return decode_boundary(_core.lower_region_schedule(encode_boundary(program), schedule_map), LoweredSchedule)


def validate_lowered_region_schedule(program: Program, lowered: LoweredSchedule) -> None:
    _core.validate_lowered_region_schedule(encode_boundary(program), encode_boundary(lowered))


def lower_region_bindings(program: Program, schedule_map: str) -> RegionBindings:
    """Compile exact source routing and last uses without enumerating instances."""
    return decode_boundary(_core.lower_region_bindings(encode_boundary(program), schedule_map), RegionBindings)


def validate_region_bindings(program: Program, bindings: RegionBindings) -> None:
    _core.validate_region_bindings(encode_boundary(program), encode_boundary(bindings))
