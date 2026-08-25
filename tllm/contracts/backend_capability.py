"""Closed capability names shared by workload declarations and backends."""

from enum import StrEnum


class BackendCapability(StrEnum):
    """Stable backend extensions admitted into compiler timelines."""

    BACKEND_REGION_BINDER = "backend/region_binder"
    RECIRCULATION_FINITE_REQUEST = "recirculation/finite_request"
    RECIRCULATION_PREFILL_BINDING = "recirculation/prefill_binding"


__all__ = ["BackendCapability"]
