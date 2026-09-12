"""Source evidence for tensor algebra and batching, separate from operator IR."""

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias


class MatmulAssociation(StrEnum):
    LEFT_TO_RIGHT = "(A@B)@C -> A@(B@C)"
    RIGHT_TO_LEFT = "A@(B@C) -> (A@B)@C"


@dataclass(frozen=True, slots=True)
class MatmulCost:
    # Conventional dense GEMM count (one multiply-add counts as two FLOPs).
    flops: int
    # Only the internal product of this two-node subgraph. This is neither
    # total graph peak memory nor backend workspace nor a latency estimate.
    intermediate_bytes: int


@dataclass(frozen=True, slots=True)
class MatmulReassociation:
    source_digest: str
    input_graph_digest: str
    output_graph_digest: str
    association: MatmulAssociation
    # Each pair is (internal product, final product). The internal product must
    # have exactly one user; shared or returned intermediates are not erased.
    source_nodes: tuple[str, str]
    replacement_nodes: tuple[str, str]
    operands: tuple[str, str, str]
    dimensions: tuple[int, int, int, int]  # A[m,k], B[k,n], C[n,p]
    dtype: str
    device: str
    before: MatmulCost
    after: MatmulCost


@dataclass(frozen=True, slots=True)
class MatmulBatching:
    source_digest: str
    input_graph_digest: str
    output_graph_digest: str
    source_nodes: tuple[str, ...]
    operands: tuple[tuple[str, str], ...]
    # Two operand packs, one batched product, and one select per source result.
    replacement_nodes: tuple[str, ...]
    dimensions: tuple[int, int, int, int]  # batch, m, k, n
    dtype: str
    device: str
    # Packing both operands adds real storage/traffic; this is not total peak.
    packed_operand_bytes: int


TensorRewrite: TypeAlias = MatmulReassociation | MatmulBatching
