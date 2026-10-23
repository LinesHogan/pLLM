"""Source-derived affine scan evidence shared by the region planner and binder."""

from dataclasses import dataclass

from pllm.contracts.region_imports import AffineScanImport, ScanAction, ScanForm
from pllm.contracts.region_program import Program, TensorType


@dataclass(frozen=True, slots=True)
class AffineScanEvidence:
    """Kernel window and backend workspace; Rust reconstructs all mathematics."""

    window: int
    workspace: int
    form: ScanForm


@dataclass(frozen=True, slots=True)
class AffineScanPlan:
    """One verified source rewrite, including its complete allocation bound."""

    program: Program
    implementation: AffineScanImport
    scale: float
    decay: float
    window: int
    chunk: int

    def competitive(self) -> bool:
        """Bound composition work and dependency depth against the source chunks.

        Heads and d² cancel from the comparison of d³ and chunk*d² GEMMs.
        Ordered composition has window dependent multiplies; a tree has two
        per doubling level plus the application of the initial state. These
        are structural bounds, not predictions of measured wall-clock time.
        """
        if self.implementation.action is ScanAction.IDENTITY:
            return True
        carry = self.implementation.declaration.inputs[0]
        assert isinstance(carry, TensorType)
        width = carry.shape[-1]
        depth = self.window if self.implementation.form is ScanForm.ORDERED else 2 * (self.window.bit_length() - 1) + 1
        return width <= self.chunk and depth * width <= self.window * self.chunk
