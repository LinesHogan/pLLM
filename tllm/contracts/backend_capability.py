"""Closed capability names shared by workload declarations and backends."""

from enum import StrEnum


class BackendCapability(StrEnum):
    """Stable backend extensions admitted into compiler timelines."""

    BACKEND_REGION_BINDER = "backend/region_binder"
    TYPED_LAYER_REPLAY = "backend/typed_layer_replay"
    PACKED_PHASE_ROWS = "runtime/packed_phase_rows@1"
    PACKED_HIDDEN_BLOCK_OUTPUT = "hidden/block_output/packed@1"


__all__ = ["BackendCapability"]
