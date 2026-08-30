#!/usr/bin/env python3
"""Runtime layer public surface.

Submodules (``residual_runtime``, ``dispatch_plan``, ``scheduling.*`` ...) are
imported directly by their consumers; the package root exports only the
instance-scoped runtime API.
"""

from __future__ import annotations

from tllm.contracts.workload_ports import StagedPortTransactionResolution
from tllm.runtime.runtime import (
    StagedPortTransactionReceipt,
    TllmRuntime,
    default_runtime,
)
from tllm.runtime.backend_region_binders import BackendRegionBinderSlot
from tllm.runtime.workload_execution import StateVersionBinding
from tllm.runtime.workload_plugins import (
    NativeWorkload,
    WorkloadWorkerSpec,
)
from tllm.runtime.workload_bindings import WorkloadRuntimeBinding

__all__ = [
    "NativeWorkload",
    "BackendRegionBinderSlot",
    "StateVersionBinding",
    "StagedPortTransactionReceipt",
    "StagedPortTransactionResolution",
    "TllmRuntime",
    "WorkloadWorkerSpec",
    "WorkloadRuntimeBinding",
    "default_runtime",
]
