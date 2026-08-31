#!/usr/bin/env python3
"""Producer-consumer data contracts."""

from tllm.contracts.backend_capability import BackendCapability
from tllm.contracts.hidden_batch import HiddenBatch
from tllm.contracts.hidden_locator import (
    HiddenLayerKind,
    HiddenLocatorCapability,
    HiddenPhase,
    HiddenSite,
)
from tllm.contracts.packed_rows import (
    PackedPhaseRow,
    PackedPhaseRowBatch,
    PackedRowPlanInputs,
    PackedSamplerRowBatch,
    PackedSamplerRowLink,
)
from tllm.contracts.packed_hidden import (
    PackedHiddenResourceBinding,
    PackedHiddenSnapshot,
    PackedHiddenSnapshotBatch,
    PackedHiddenSnapshotSpecialization,
)
from tllm.contracts.gpu_stage import DeviceTensorLease
from tllm.contracts.request_meta_view import RowBatchMeta
from tllm.contracts.runtime_context import RuntimeContext
from tllm.contracts.shape_specialization import (
    RegionShapeSpecializationInput,
    RegionShapeSpecializationRequest,
)
from tllm.contracts.subscription import ConsumerSubscription

__all__ = [
    "ConsumerSubscription",
    "BackendCapability",
    "DeviceTensorLease",
    "HiddenBatch",
    "HiddenLayerKind",
    "HiddenLocatorCapability",
    "HiddenPhase",
    "HiddenSite",
    "PackedPhaseRow",
    "PackedPhaseRowBatch",
    "PackedRowPlanInputs",
    "PackedSamplerRowBatch",
    "PackedSamplerRowLink",
    "PackedHiddenResourceBinding",
    "PackedHiddenSnapshot",
    "PackedHiddenSnapshotBatch",
    "PackedHiddenSnapshotSpecialization",
    "RowBatchMeta",
    "RuntimeContext",
    "RegionShapeSpecializationInput",
    "RegionShapeSpecializationRequest",
]
