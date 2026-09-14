"""Closed storage imports for exported, native and compiled tensor boundaries.

An exported graph describes its operations and aliases. A native declaration
describes only the wrapper's private input and independent output copies,
recursively for tensor leaves in immutable banks;
kernel internals require separate backend facts. The compiled import supplies
allocation layouts from its compiler, borrows read-only inputs, and copies outputs.
Native scalar operands and explicit effects do not create tensor storage; their
original resource owners and allocations retained across calls are separate.
"""

from dataclasses import dataclass
from typing import TypeAlias

from pllm.contracts.region_program import AccessMode, AffineIndex, Function, IndexTerm, ScalarType, TensorType
from pllm.contracts.tensor_access import TensorAccessGraph


@dataclass(frozen=True, slots=True)
class ExportedTensorImport:
    access: TensorAccessGraph


@dataclass(frozen=True, slots=True)
class NativeTensorImport:
    declaration: Function


@dataclass(frozen=True, slots=True)
class BorrowedTensorImport:
    """Read one tensor field of an original read-only resource cell.

    Inputs are the resource coordinates, in order. The contiguous tensor is
    borrowed through invocation completion, without allocating a new backing.
    Regions using this ABI return independent copies of their mathematical
    results, including views and banks that could otherwise expose the owner.
    """

    declaration: Function
    field: int

    def validate(self) -> None:
        function = self.declaration
        if (type(self.field) is not int or self.field < 0 or self.field >= 2**63
            or len(function.resources) != 1 or len(function.outputs) != 1
            or not isinstance(function.outputs[0], TensorType)
            or any(type_ is not ScalarType.INDEX for type_ in function.inputs)
            or len(function.accesses) != 1):
            raise ValueError("tensor borrowing requires one typed field and index coordinates")
        access, = function.accesses
        if (access.resource != function.resources[0] or access.mode is not AccessMode.READ
            or len(access.ranges) != len(function.inputs)
            or any(interval.lower != AffineIndex(0, (IndexTerm(i),))
                or interval.upper != AffineIndex(1, (IndexTerm(i),)) for i, interval in enumerate(access.ranges))):
            raise ValueError("tensor borrowing must read exactly its original resource cell")


@dataclass(frozen=True, slots=True)
class KernelAllocation:
    name: str
    device: str
    bytes: int


@dataclass(frozen=True, slots=True)
class NativeKernelImport:
    """Backend ABI for a native kernel completing on the issuing stream.

    The provider identifies its local implementation and covers all transient
    allocations, including raw outputs and library workspace. Wrapper input/
    output copies are accounted separately. Persistent resources remain with
    their original owners; all accesses must appear in the source declaration.
    Device reads/writes must respect waits queued on that issuing stream. Any
    host observation of device data must await its stream before inspecting it.
    Ordinary NativeTensorImport makes none of these execution guarantees.
    """

    declaration: Function
    implementation: str
    device: str
    allocations: tuple[KernelAllocation, ...]


@dataclass(frozen=True, slots=True)
class FunctionalKernelImport:
    """Functional graph ABI: readonly inputs and fresh, distinct dense results.

    The backend retains the original functionalized graph and proves that each
    returned tensor owns its complete backing, without aliases to inputs or
    other outputs. All allocations, including those result backings, are listed.
    Execution and workspace guarantees otherwise match NativeKernelImport.
    This import has no external resource effects or wrapper tensor copies.
    """

    declaration: Function
    implementation: str
    device: str
    allocations: tuple[KernelAllocation, ...]


@dataclass(frozen=True, slots=True)
class CompiledTensorImport:
    # Source tensor mathematics; this is NOT the compiled memory profile.
    source_access: TensorAccessGraph
    backend: str
    code_digest: str
    # Graph-derived capacity account: the layout span of every tensor value in
    # the normalized graph, an explicit charge for reduction temporaries and a
    # named slack fraction — never code-generation allocation reads. Summing
    # all entries is conservative even when their lifetimes do not overlap.
    allocations: tuple[KernelAllocation, ...]
    # Execution borrows read-only inputs and independently copies each output.
    # Compiler replay rejects input mutations and runtime alignment copies.
    # Backend tensor allocations include raw results retained during copying;
    # graphs that could lower to opaque external workspaces are rejected by
    # this compiler adapter's normalized-graph operator subset.


RegionTensorImport: TypeAlias = ExportedTensorImport | NativeTensorImport | NativeKernelImport | FunctionalKernelImport | CompiledTensorImport | BorrowedTensorImport


def region_import_name(imported: RegionTensorImport) -> str:
    if isinstance(imported, ExportedTensorImport):
        return imported.access.source_digest
    if isinstance(imported, CompiledTensorImport):
        return imported.source_access.source_digest
    return imported.declaration.name
