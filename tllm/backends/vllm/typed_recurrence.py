"""Deprecated vLLM import path for generic typed recurrence execution.

The executor belongs to :mod:`tllm.runtime.typed_recurrence`; vLLM-specific
primitive adapters are constructed by their owning workload/backend code.
"""

from tllm.runtime.typed_recurrence import (
    BackendRegionAdapter,
    PrimitiveAdapter,
    PrimitiveAdapterRegistry,
    TypedRecurrenceExecutor,
)

__all__ = [
    "BackendRegionAdapter",
    "PrimitiveAdapter",
    "PrimitiveAdapterRegistry",
    "TypedRecurrenceExecutor",
]
