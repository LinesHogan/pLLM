"""Compose tensor storage imports with source-derived symbolic value routes."""

from collections.abc import Mapping
from typing import TypeAlias

from pllm import _core
from pllm.contracts.region_capacity import RegionCapacity
from pllm.contracts.region_imports import (
    ExportedTensorImport,
    NativeTensorImport,
    RegionTensorImport,
    region_import_name,
)
from pllm.contracts.region_payload import RegionPayload
from pllm.contracts.region_program import Program
from pllm.contracts.region_retention import RetainedAllocation
from pllm.contracts.region_storage import RegionStorageAnalysis
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_native import BorrowedTensorPrimitive, CompiledPrimitive, RegionFunction
from pllm.execution.tensor_access import import_tensor_access_graph

RegionImportSource: TypeAlias = Mapping[str, RegionFunction] | tuple[RegionTensorImport, ...]


def analyze_region_payload(
    program: Program, schedule_map: str, cells: RegionImportSource, retained: tuple[RetainedAllocation, ...] = ()
) -> RegionPayload:
    return decode_boundary(
        _core.analyze_region_payload(
            encode_boundary(program),
            schedule_map,
            encode_boundary(import_region_functions(program, cells)),
            encode_boundary(retained),
        ),
        RegionPayload,
    )


def import_region_functions(
    program: Program, cells: RegionImportSource, *, catalog: tuple[RegionTensorImport, ...] = ()
) -> tuple[RegionTensorImport, ...]:
    """Normalize local implementations or closed, transportable descriptions.

    Planning FFI validates descriptor contents against Program. A descriptor is
    never an executable binding; execution still requires local implementations.
    """
    if isinstance(cells, tuple):
        if catalog:
            raise ValueError("closed region imports cannot be supplemented by another catalog")
        names = tuple(region_import_name(imported) for imported in cells)
        if len(set(names)) != len(names) or set(names) != {f.name for f in program.functions}:
            raise ValueError("region storage needs complete distinct original tensor imports")
        return tuple(sorted(cells, key=region_import_name))
    supplied = {region_import_name(imported): imported for imported in catalog}
    if len(supplied) != len(catalog):
        raise ValueError("backend source catalog requires distinct imports")
    names = {f.name for f in program.functions}
    if not set(cells) <= names or not names <= set(cells) | supplied.keys():
        raise ValueError("region storage needs complete original tensor imports")
    for declaration in program.functions:
        if declaration.name in cells and cells[declaration.name].declaration != declaration:
            raise ValueError("storage cell differs from its source function")
    result: list[RegionTensorImport] = []
    for name in sorted(cells):
        cell = cells[name]
        if isinstance(cell, ExportedCell):
            result.append(ExportedTensorImport(import_tensor_access_graph(cell)))
        elif isinstance(cell, (CompiledPrimitive, BorrowedTensorPrimitive)):
            result.append(cell.implementation)
        else:
            result.append(
                cell.implementation if cell.implementation is not None else NativeTensorImport(cell.declaration)
            )
        if name in supplied and supplied[name] != result[-1]:
            raise ValueError("local source differs from the backend import catalog")
    result.extend(supplied[name] for name in names - cells.keys())
    return tuple(sorted(result, key=region_import_name))


def analyze_region_storage(program: Program, schedule_map: str, cells: RegionImportSource) -> RegionStorageAnalysis:
    return decode_boundary(
        _core.analyze_region_storage(
            encode_boundary(program), schedule_map, encode_boundary(import_region_functions(program, cells))
        ),
        RegionStorageAnalysis,
    )


def validate_region_storage(program: Program, cells: RegionImportSource, analysis: RegionStorageAnalysis) -> None:
    _core.validate_region_storage(
        encode_boundary(program), encode_boundary(import_region_functions(program, cells)), encode_boundary(analysis)
    )


def analyze_region_capacity(program: Program, schedule_map: str, cells: RegionImportSource) -> RegionCapacity:
    """Bound tensor payloads for all declared sizes, not complete device memory."""
    return decode_boundary(
        _core.analyze_region_capacity(
            encode_boundary(program), schedule_map, encode_boundary(import_region_functions(program, cells))
        ),
        RegionCapacity,
    )


def validate_region_capacity(program: Program, cells: RegionImportSource, capacity: RegionCapacity) -> None:
    _core.validate_region_capacity(
        encode_boundary(program), encode_boundary(import_region_functions(program, cells)), encode_boundary(capacity)
    )
