"""Symbolic storage charges and lifetimes derived from a region and its imports.

Imported calls retain their proven alias/copy rules. Atomic structured results
receive a conservative backing charge: that attribution is a capacity bound,
not evidence that the result owns a distinct allocation or permits buffer reuse.
The source operation identifies which rule applies.
"""

from dataclasses import dataclass

from pllm.contracts.region_bindings import AccessTable


@dataclass(frozen=True, slots=True)
class RegionStorageRoot:
    value_id: int
    bytes: int
    device: str


@dataclass(frozen=True, slots=True)
class RegionStorageAnalysis:
    program_digest: str
    schedule_map: str
    imports_digest: str
    roots: tuple[RegionStorageRoot, ...]
    input_storages: str
    borrowed_storages: str
    live_storages: str
    allocation_times: str
    retained_storages: str
    value_storage: str
    storage_consumers: str
    routes: tuple[AccessTable, ...]
    births: tuple[AccessTable, ...]
    last_uses: tuple[AccessTable, ...]
    retained: tuple[AccessTable, ...]
    signature: str
