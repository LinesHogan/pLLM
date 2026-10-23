"""Explicit native imports at the region's tensor/kernel boundary.

These are trusted compiler/backend adapters, not a purity annotation on arbitrary
user Python. Their data and resource contracts must match the source declaration.
The resource view delegates to the original owner and expires after each call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

import torch

from pllm.contracts.region_imports import (
    AffineScanImport,
    BorrowedTensorImport,
    CompiledTensorImport,
    ExportedTensorImport,
    FunctionalKernelImport,
    NativeKernelImport,
    RegionTensorImport,
    ScanAction,
    ScanForm,
    region_import_name,
)
from pllm.contracts.region_program import AccessMode, Function, ScalarType, TensorType
from pllm.execution.codec import (
    boundary_digest,
    refresh_tensor_metadata,
    tensor_graph_outputs,
    tensor_graph_result_nodes,
    tensor_node_metadata,
)
from pllm.execution.exported_cell import CellFunction, ExportedCell
from pllm.execution.module_graph import functionalized_kernel_operators, tensor_fragment_allocations
from pllm.execution.region_resources import RegionResources
from pllm.execution.region_values import Data, check_data, copy_data


class PrimitiveResources:
    """A call-scoped view of exactly its declared resources; no storage ownership."""

    def __init__(
        self,
        declaration: Function,
        pages: RegionResources | None,
        inputs: tuple[Data, ...],
    ) -> None:
        if (declaration.resources or declaration.accesses) and pages is None:
            raise ValueError("effectful native import requires original resource bindings")
        self._declaration = declaration
        self._pages = pages
        self._active = True
        self._accesses = () if pages is None else pages.bind_accesses(declaration, inputs)

    def _binding(self, resource: str) -> RegionResources:
        if not self._active:
            raise RuntimeError("native resource view has expired")
        if resource not in self._declaration.resources or self._pages is None:
            raise ValueError("native import accessed an undeclared resource")
        return self._pages

    def snapshot(self, resource: str, indices: tuple[int, ...] = ()) -> tuple[Data, ...]:
        return self._permit(resource, indices, AccessMode.READ).snapshot(resource, indices)

    def publish(self, resource: str, indices: tuple[int, ...], values: tuple[Data, ...]) -> None:
        self._permit(resource, indices, AccessMode.WRITE).publish(resource, indices, values)

    def _permit(self, resource: str, indices: tuple[int, ...], mode: AccessMode) -> RegionResources:
        pages = self._binding(resource)
        accesses = tuple(access for access in self._accesses if access.resource == resource)
        if accesses and not any(
            access.mode is mode
            and len(indices) == len(access.ranges)
            and all(lower <= index < upper for index, (lower, upper) in zip(indices, access.ranges, strict=True))
            for access in accesses
        ):
            raise ValueError("native access exceeds its declared read/write region")
        return pages

    def external(self, inputs: tuple[Data, ...]) -> tuple[Data, ...]:
        if not self._active:
            raise RuntimeError("native resource view has expired")
        if self._pages is None:
            raise ValueError("external call requires original resource bindings")
        if self._pages.bind_accesses(self._declaration, inputs) != self._accesses:
            raise ValueError("external invocation changes its original indexed access")
        return self._pages.external(self._declaration, inputs)

    def close(self) -> None:
        self._active = False
        self._pages = None


NativeFunction: TypeAlias = Callable[[tuple[Data, ...], PrimitiveResources], tuple[Data, ...]]


@dataclass(frozen=True, slots=True)
class AffineScanPrimitive:
    """Bind a Rust-derived affine kernel with borrowed inputs and owned outputs."""

    implementation: AffineScanImport
    function: Callable[[tuple[Data, ...]], tuple[Data, ...]]

    @property
    def declaration(self) -> Function:
        return self.implementation.declaration

    @property
    def transformations(self) -> tuple[str, ...]:
        if self.implementation.action is ScanAction.RIGHT_MATRIX:
            return (
                f"right-matrix affine scan: SGD distributivity, fp32 batched GEMMs and {self.implementation.form.value} composition",
            )
        if self.implementation.form is ScanForm.ORDERED:
            return ("additive scan: batched GEMMs with ordered carry rounding",)
        return ("additive scan: batched GEMMs and Hillis–Steele fp32 prefix reassociation",)

    def invoke(
        self, inputs: tuple[Data, ...], pages: RegionResources | None, sizes: Mapping[str, int] = {}
    ) -> tuple[Data, ...]:
        # The frame validates entry tensors; Rust proves this replacement call
        # has exactly the source scan's ports. No second shape walk is needed.
        with torch.no_grad():
            return self.function(inputs)


@dataclass(frozen=True, slots=True)
class NativePrimitive:
    declaration: Function
    function: NativeFunction
    implementation: NativeKernelImport | None = None

    def __post_init__(self) -> None:
        if self.implementation is not None and type(self.implementation) is not NativeKernelImport:
            raise ValueError("native wrapper requires its copying kernel ABI")
        if self.implementation is not None and self.implementation.declaration != self.declaration:
            raise ValueError("native kernel implementation differs from its source declaration")

    def invoke(
        self, inputs: tuple[Data, ...], pages: RegionResources | None, sizes: Mapping[str, int] = {}
    ) -> tuple[Data, ...]:
        for value, type_ in zip(inputs, self.declaration.inputs, strict=True):
            check_data(value, type_, sizes)
        # Native tensor kernels may update their arguments. Private mathematical
        # inputs preserve all SSA versions, independently of physical COW pages.
        arguments = tuple(copy_data(value) for value in inputs)
        resources = PrimitiveResources(self.declaration, pages, inputs)
        try:
            result = self.function(arguments, resources)
            if pages is not None:
                pages.before_statement()
            for value, type_ in zip(result, self.declaration.outputs, strict=True):
                check_data(value, type_, sizes)
            # Native outputs may be views or reusable backend buffers. Copy before
            # returning them to the SSA frame; completion still belongs upstream.
            return tuple(copy_data(value) for value in result)
        finally:
            resources.close()


@dataclass(frozen=True, slots=True)
class FunctionalKernelPrimitive:
    """An original functionalized FX fragment, with owned result backings."""

    graph: torch.fx.GraphModule
    implementation: FunctionalKernelImport

    @classmethod
    def bind(cls, graph: torch.fx.GraphModule, implementation: FunctionalKernelImport) -> FunctionalKernelPrimitive:
        functionalized_kernel_operators(graph)
        return cls(refresh_tensor_metadata(graph), implementation)

    def __post_init__(self) -> None:
        if type(self.implementation) is not FunctionalKernelImport:
            raise ValueError("functional graph requires its owned-result kernel ABI")
        functionalized_kernel_operators(self.graph)
        if self.declaration.resources or self.declaration.accesses:
            raise ValueError("functional kernel cannot own external resource effects")
        inputs = tuple(node for node in self.graph.graph.nodes if node.op == "placeholder")
        outputs = tensor_graph_result_nodes(self.graph)
        backings: set[torch.UntypedStorage] = set()
        for nodes, types, result in (
            (inputs, self.declaration.inputs, False),
            (outputs, self.declaration.outputs, True),
        ):
            for node, type_ in zip(nodes, types, strict=True):
                tensor = tensor_node_metadata(node)
                if tensor is None or not isinstance(type_, TensorType) or not tensor.is_contiguous():
                    raise ValueError("functional kernel requires dense tensor ports")
                check_data(tensor, type_)
                if str(tensor.device) != self.implementation.device:
                    raise ValueError("functional kernel ports must use their execution device")
                storage = tensor.untyped_storage()
                if result and (
                    storage in backings
                    or tensor.storage_offset() != 0
                    or storage.nbytes() != tensor.numel() * tensor.element_size()
                ):
                    raise ValueError("functional kernel results must own distinct complete backings")
                backings.add(storage)
        required = sum(item.bytes for item in tensor_fragment_allocations(self.graph))
        if any(item.device != self.implementation.device for item in self.implementation.allocations) or required > sum(
            item.bytes for item in self.implementation.allocations
        ):
            raise ValueError("functional kernel allocations do not cover its original graph")

    @property
    def declaration(self) -> Function:
        return self.implementation.declaration

    def invoke(
        self, inputs: tuple[Data, ...], pages: RegionResources | None, sizes: Mapping[str, int] = {}
    ) -> tuple[Data, ...]:
        tensors: list[torch.Tensor] = []
        for value, type_ in zip(inputs, self.declaration.inputs, strict=True):
            check_data(value, type_, sizes)
            if not isinstance(value, torch.Tensor) or not value.is_contiguous():
                raise ValueError("functional kernel requires dense tensor operands")
            tensors.append(value)
        with torch.no_grad():
            result = tensor_graph_outputs(self.graph, tuple(tensors))
        if pages is not None:
            pages.before_statement()
        for value, type_ in zip(result, self.declaration.outputs, strict=True):
            check_data(value, type_, sizes)
            if not value.is_contiguous():
                raise ValueError("functional kernel returned a non-dense result")
        return result


@dataclass(frozen=True, slots=True)
class CompiledPrimitive:
    """Compiler-proven read-only inputs and independently copied SSA outputs."""

    source: ExportedCell
    implementation: CompiledTensorImport
    function: CellFunction

    @property
    def declaration(self) -> Function:
        return self.source.declaration

    @property
    def transformations(self) -> tuple[str, ...]:
        return (*self.source.transformations, "tensor kernel compilation")

    def invoke(
        self, inputs: tuple[Data, ...], pages: RegionResources | None, sizes: Mapping[str, int] = {}
    ) -> tuple[Data, ...]:
        tensors: list[torch.Tensor] = []
        for value, guard in zip(inputs, self.source.input_signatures, strict=True):
            if not isinstance(value, torch.Tensor):
                raise ValueError("compiled tensor block requires tensor inputs")
            guard.check(value)
            tensors.append(value.detach())
        # The compiler rejects input mutations, alignment copies and opaque
        # calls. Borrows stay live through the output copies; the original
        # frame/owner continues to determine when inputs may be released.
        with torch.no_grad():
            result = self.function(*tensors)
        if pages is not None:
            pages.before_statement()
        for value, guard in zip(result, self.source.output_signatures, strict=True):
            guard.check(value)
        return tuple(copy_data(value) for value in result)


@dataclass(frozen=True, slots=True)
class BorrowedTensorPrimitive:
    """A typed read of original storage, with no callback or publication power."""

    implementation: BorrowedTensorImport

    def __post_init__(self) -> None:
        self.implementation.validate()

    @property
    def declaration(self) -> Function:
        return self.implementation.declaration

    def invoke(
        self, inputs: tuple[Data, ...], pages: RegionResources | None, sizes: Mapping[str, int] = {}
    ) -> tuple[Data, ...]:
        if pages is None:
            raise ValueError("tensor borrowing requires its original resource owner")
        return (pages.borrow_tensor(self.implementation, inputs),)


RegionFunction: TypeAlias = (
    ExportedCell
    | NativePrimitive
    | FunctionalKernelPrimitive
    | CompiledPrimitive
    | BorrowedTensorPrimitive
    | AffineScanPrimitive
)
TensorSource: TypeAlias = ExportedCell | BorrowedTensorPrimitive | NativePrimitive | FunctionalKernelPrimitive


def is_original_tensor_source(cell: RegionFunction) -> bool:
    """Source graphs, readonly accessors, or audited kernels with no state effects."""
    if type(cell) not in (ExportedCell, BorrowedTensorPrimitive, NativePrimitive, FunctionalKernelPrimitive):
        return False
    if isinstance(cell, ExportedCell):
        return not cell.transformations
    if isinstance(cell, (BorrowedTensorPrimitive, FunctionalKernelPrimitive)):
        return True
    return isinstance(cell, NativePrimitive) and cell.implementation is not None and not cell.declaration.resources


def is_original_region_source(cell: RegionFunction) -> bool:
    """Original computation, including unchanged invocation-bound effects.

    This is a source-binding check, not permission to benchmark effects or to
    absorb native calls into a tensor graph. Those need their own boundaries.
    """
    return type(cell) is NativePrimitive or is_original_tensor_source(cell)


def original_region_sources(cells: Mapping[str, RegionFunction]) -> dict[str, TensorSource]:
    result: dict[str, TensorSource] = {}
    for name, cell in cells.items():
        if isinstance(cell, (CompiledPrimitive, AffineScanPrimitive)) or not is_original_region_source(cell):
            raise ValueError("region preparation requires original source implementations")
        result[name] = cell
    return result


def source_region_functions(
    declarations: tuple[Function, ...], builtins: Mapping[str, RegionFunction], imports: tuple[RegionTensorImport, ...]
) -> dict[str, RegionFunction]:
    """Bind resource adapters; leave remote tensor graphs explicitly unbound.

    Builtins are the caller's own computation. Export descriptors carry no
    callable, whereas native endpoints delegate to the original invocation.
    """
    catalog = {region_import_name(item): item for item in imports}
    if len(catalog) != len(imports):
        raise ValueError("backend source catalog requires distinct imports")
    result: dict[str, RegionFunction] = {}
    for declaration in declarations:
        imported = catalog.get(declaration.name)
        if isinstance(imported, CompiledTensorImport):
            raise ValueError("compiled alternatives require numerical selection, not a source catalog")
        if declaration.name in builtins:
            result[declaration.name] = builtins[declaration.name]
        elif isinstance(imported, ExportedTensorImport):
            continue
        elif isinstance(imported, BorrowedTensorImport):
            result[declaration.name] = BorrowedTensorPrimitive(imported)
        elif isinstance(imported, (NativeKernelImport, FunctionalKernelImport)) and not declaration.resources:
            if imported.declaration != declaration:
                raise ValueError("backend source import differs from its declaration")
            # The original graph fragment is local to the worker, like an
            # export graph. Its execution contract is sufficient for planning.
            continue
        else:
            if not declaration.resources:
                raise ValueError("pure backend functions require an original tensor graph binding")
            if imported is not None and imported.declaration != declaration:
                raise ValueError("backend source import differs from its declaration")
            result[declaration.name] = external_primitive(
                declaration, imported if isinstance(imported, NativeKernelImport) else None
            )
    return result


def external_primitive(declaration: Function, implementation: NativeKernelImport | None = None) -> NativePrimitive:
    """Cold import descriptor; the endpoint is bound by the original invocation."""
    if not declaration.resources:
        raise ValueError("external import must declare its owned resource effects")

    def invoke(inputs: tuple[Data, ...], resources: PrimitiveResources) -> tuple[Data, ...]:
        return resources.external(inputs)

    return NativePrimitive(declaration, invoke, implementation)


def tensor_item(type_: TensorType) -> NativePrimitive:
    """Import an explicit scalar tensor read, including its device synchronization."""
    if type_.shape != ():
        raise ValueError("tensor item import requires a rank-zero tensor")
    if type_.dtype == "torch.bool":
        result_type = ScalarType.BOOL
    elif type_.dtype in ("torch.int8", "torch.uint8", "torch.int16", "torch.int32", "torch.int64"):
        result_type = ScalarType.INT
    elif type_.dtype in ("torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"):
        result_type = ScalarType.FLOAT
    else:
        raise ValueError("tensor item import requires a supported real scalar dtype")
    declaration = Function("native.tensor_item@1:" + boundary_digest(type_), (type_,), (result_type,))

    def read(inputs: tuple[Data, ...], resources: PrimitiveResources) -> tuple[Data, ...]:
        (tensor,) = inputs
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("tensor item import received a non-tensor")
        if result_type is ScalarType.BOOL:
            return (bool(tensor.item()),)
        if result_type is ScalarType.INT:
            return (int(tensor.item()),)
        return (float(tensor.item()),)

    # The only device allocation is the native wrapper's private scalar copy.
    # Tensor.item completes that copy/read on the issuing stream before return.
    implementation = NativeKernelImport(
        declaration, "native.scalar_read.current_stream@1", type_.device, (), host_reads=True
    )
    return NativePrimitive(declaration, read, implementation)
