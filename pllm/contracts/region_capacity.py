"""Conservative tensor-payload bounds over the full declared parameter ranges.

These are bounds, not exact peak measurements or complete device reservations.
Backend scratch, allocator overhead and original owner residency are separate.
For native imports, only wrapper copies are modeled; allocations retained by
the kernel's original outputs also need backend bounds, even during return.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RegionStorageBound:
    value_id: int
    live_instances_upper_bound: int


@dataclass(frozen=True, slots=True)
class RegionDeviceCapacity:
    device: str
    entry_copy_bytes_upper_bound: int
    result_copy_bytes_upper_bound: int
    live_tensor_bytes_upper_bound: int
    call_scratch_bytes_upper_bound: int
    peak_tensor_bytes_upper_bound: int
    # Bounds for the modeled subtotal only if unknown_functions is nonempty.
    logical_read_bytes_upper_bound: int
    logical_write_bytes_upper_bound: int


@dataclass(frozen=True, slots=True)
class RegionCapacity:
    program_digest: str
    schedule_map: str
    imports_digest: str
    storage_signature: str
    roots: tuple[RegionStorageBound, ...]
    devices: tuple[RegionDeviceCapacity, ...]
    # Export operations with unknown accesses, or native kernels whose internal
    # allocations/traffic are unmodeled. Zero-trip calls do not contribute.
    unknown_functions: tuple[str, ...]
    signature: str
