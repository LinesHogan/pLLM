"""Pure tensor kernels shared by clean-room workloads."""

from tllm.canonical.kernels.low_rank_gated import (
    LowRankGatedLayout,
    initialize_low_rank_gated,
    low_rank_gated_forward,
)
from tllm.canonical.kernels.dense_fast_weight import (
    DenseFastWeightTransition,
    transition_dense_fast_weight,
)

__all__ = [
    "LowRankGatedLayout",
    "initialize_low_rank_gated",
    "low_rank_gated_forward",
    "DenseFastWeightTransition",
    "transition_dense_fast_weight",
]
