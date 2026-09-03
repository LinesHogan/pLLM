"""Closed capability names shared by workload declarations and backends."""

from enum import StrEnum


class BackendCapability(StrEnum):
    """Stable backend extensions admitted into compiler timelines."""

    TYPED_LAYER_REPLAY = "backend/typed_layer_replay"


__all__ = ["BackendCapability"]
