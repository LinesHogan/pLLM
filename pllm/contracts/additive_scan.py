"""Source-derived additive scan evidence shared by the region planner and binder."""

from dataclasses import dataclass

from pllm.contracts.region_imports import OrderedScanImport
from pllm.contracts.region_program import Program


@dataclass(frozen=True, slots=True)
class AdditiveScanEvidence:
    """Kernel window and backend workspace; Rust reconstructs all mathematics."""

    window: int
    workspace: int


@dataclass(frozen=True, slots=True)
class AdditiveScanPlan:
    """One verified source rewrite, including its complete allocation bound."""

    program: Program
    implementation: OrderedScanImport
    scale: float
    window: int
