"""Clean-room, in-process vLLM 0.27.1 adapter surface.

This package is intentionally separate from ``tllm.backends`` and
``tllm.runtime``.  It contains only the signed eager MRv2 attachment pieces;
legacy runtime, backend-driver, and workgraph authority are forbidden here.
"""

from tllm.canonical_vllm.hidden_capture import Vllm027EagerPackedHiddenCapture
from tllm.canonical_vllm.packed_rows import (
    Mrv2PackedStepRows,
    Mrv2RequestIdBatch,
    Mrv2RequestIdBinding,
    Vllm027Mrv2PackedRowCodec,
)
from tllm.canonical_vllm.sampler import (
    CanonicalVllm027PostFilterSampler,
    SignedPostFilterGuidance,
)
from tllm.canonical_vllm.dense_overlay import (
    VllmDenseAddAdapter,
    VllmDenseAddTarget,
    VllmDenseAddTargetSpec,
    inspect_dense_add_targets,
)
from tllm.canonical_vllm.runtime import (
    CanonicalVllmConfig,
    CanonicalVllmDType,
    CanonicalVllmLlm,
    create_canonical_vllm_llm,
)

__all__ = [
    "CanonicalVllm027PostFilterSampler",
    "CanonicalVllmConfig",
    "CanonicalVllmDType",
    "CanonicalVllmLlm",
    "Mrv2PackedStepRows",
    "Mrv2RequestIdBatch",
    "Mrv2RequestIdBinding",
    "SignedPostFilterGuidance",
    "VllmDenseAddAdapter",
    "VllmDenseAddTarget",
    "VllmDenseAddTargetSpec",
    "inspect_dense_add_targets",
    "Vllm027EagerPackedHiddenCapture",
    "Vllm027Mrv2PackedRowCodec",
    "create_canonical_vllm_llm",
]
