"""Bind region effects to the original PTTS and backend invocation capabilities."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from itertools import product
from typing import Protocol

import torch

from pllm.contracts.region_imports import BorrowedTensorImport, NativeKernelImport
from pllm.contracts.region_program import (
    AccessMode,
    AffineIndex,
    BankType,
    BorrowedResource,
    DataType,
    ExternalResource,
    Function,
    Program,
    Resource,
    TensorType,
    bind_sizes,
    resolve_extent,
)
from pllm.contracts.region_streams import RegionStateWindow
from pllm.contracts.state_ref import LineageId, StateRef
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.host_borrow import HostResourceView, validate_host_views
from pllm.execution.paged_access import PagedResourceView, validate_paged_views
from pllm.execution.ptts.page_bundle import PageBundleArena, PagePin, PagePublication, PageRelease, PageWrite
from pllm.execution.region_frame import Routes
from pllm.execution.region_values import Bank, Data, check_data
from pllm.execution.tensor_borrow import (
    MutableTensorView,
    ReadonlyTensorCell,
    ReadonlyTensorView,
    validate_readonly_views,
)


class RegionPageOwner(Protocol):
    """Implemented by the original StateTaskContext, including its failure path."""

    @property
    def arena(self) -> PageBundleArena: ...
    @property
    def pin(self) -> PagePin: ...
    @property
    def page_pins(self) -> Mapping[PagePin, PageRelease | None]: ...
    def current_owned_ref(self, original: StateRef) -> StateRef: ...
    def check_page_execution(self) -> None: ...
    def prepare_owned_write(
        self, sources: tuple[StateRef, ...], fields: tuple[tuple[LayoutFieldId, ...], ...]
    ) -> PageWrite: ...
    def publish_owned_write(self, token: PageWrite) -> PagePublication: ...
    def settle_owned_publication(self, receipt: PagePublication) -> None: ...
    def borrow_published(self, refs: Sequence[StateRef]) -> PagePin: ...
    def release_owned_pin(self, pin: PagePin) -> None: ...


@dataclass(frozen=True, slots=True)
class ResourcePageCell:
    resource: str
    indices: tuple[int, ...]
    original: StateRef
    fields: tuple[LayoutFieldId, ...]


@dataclass(frozen=True, slots=True)
class PagedResourceBinding:
    """Bind a verified view to the original invocation, without owning its KV."""

    view: PagedResourceView
    owner: RegionPageOwner


@dataclass(frozen=True, slots=True)
class ReadonlyResourceBinding:
    view: ReadonlyTensorView
    owner: RegionPageOwner


@dataclass(frozen=True, slots=True)
class TensorResourceBinding:
    """An invocation borrows distinct mutable backend tensors."""

    view: MutableTensorView
    owner: RegionPageOwner


@dataclass(frozen=True, slots=True)
class PinnedResourceCell:
    indices: tuple[int, ...]
    pin: PagePin
    ref: StateRef
    fields: tuple[LayoutFieldId, ...]


@dataclass(frozen=True, slots=True)
class _PinnedTensorView(ReadonlyTensorView):
    owner: RegionPageOwner
    proofs: tuple[PinnedResourceCell, ...]

    def validate(self) -> None:
        self.owner.check_page_execution()
        ReadonlyTensorView.validate(self)
        proofs = {cell.indices: cell for cell in self.proofs}
        if len(proofs) != len(self.proofs) or set(proofs) != {cell.indices for cell in self.cells if cell.identities}:
            raise ValueError("pinned view requires exact nonempty cell proofs")

        def metadata(
            tensor: torch.Tensor,
        ) -> tuple[int, int, int, int, torch.Size, tuple[int, ...], torch.dtype, torch.device, int]:
            storage = tensor.untyped_storage()
            return (
                tensor.data_ptr(),
                tensor.numel() * tensor.element_size(),
                storage.data_ptr(),
                storage.nbytes(),
                tensor.shape,
                tensor.stride(),
                tensor.dtype,
                tensor.device,
                int(tensor.storage_offset()),
            )

        for cell in self.cells:
            if not cell.identities:
                continue
            proof = proofs[cell.indices]
            # Published/admission versions use original invocation-held exact pins;
            # never replace the begin pin used by E1's borrowed-root binding.
            if (
                proof.pin not in self.owner.page_pins
                or self.owner.page_pins[proof.pin] is not None
                or proof.pin.stream != self.owner.pin.stream
                or proof.ref not in proof.pin.refs
                or len(proof.fields) != len(cell.identities)
                or len(set(proof.fields)) != len(proof.fields)
            ):
                raise ValueError("pinned view is outside its owner's live exact pin fields")
            for identity, field in zip(cell.identities, proof.fields, strict=True):
                original = self.owner.arena.borrow_field(proof.pin, proof.ref, field)
                if metadata(identity.tensor) != metadata(original):
                    raise ValueError("pinned view storage differs from its exact arena field")


@dataclass(frozen=True, slots=True)
class PinnedResourceBinding:
    """A read-only projection of exact pins held by the original invocation."""

    view: _PinnedTensorView
    owner: RegionPageOwner

    @classmethod
    def bind(
        cls, resource: BorrowedResource, owner: RegionPageOwner, cells: tuple[PinnedResourceCell, ...]
    ) -> "PinnedResourceBinding":
        selected = {cell.indices: cell for cell in cells}
        views: list[ReadonlyTensorCell] = []
        for indices in product(*(range(size) for size in resource.domain)):
            cell = selected.get(indices)
            tensors = (
                ()
                if cell is None
                else tuple(owner.arena.borrow_field(cell.pin, cell.ref, field) for field in cell.fields)
            )
            views.append(ReadonlyTensorCell.bind(indices, tensors))
        view = _PinnedTensorView(resource, tuple(views), owner, cells)
        view.validate()
        return cls(view, owner)

    def validate(self, program: Program, paged: tuple[PagedResourceView, ...]) -> None:
        if self.view.owner is not self.owner:
            raise ValueError("pinned view belongs to another invocation")
        name = self.view.resource.name
        for function in program.functions:
            if name in function.resources:
                accesses = tuple(access for access in function.accesses if access.resource == name)
                if not accesses or any(access.mode is not AccessMode.READ for access in accesses):
                    raise ValueError("pinned resources require explicit READ effects only")
        # The pin proves the arena overlap; the existing writable-alias check
        # remains mandatory. Ordinary readonly validation below is unchanged.
        validate_readonly_views((self.view,), paged, ())


@dataclass(frozen=True, slots=True)
class HostResourceBinding:
    view: HostResourceView
    owner: RegionPageOwner


def _tensor_types(type_: DataType, sizes: Mapping[str, int]) -> Iterator[TensorType]:
    if isinstance(type_, TensorType):
        yield type_
    elif isinstance(type_, BankType):
        for _ in range(resolve_extent(type_.size, sizes)):
            yield from _tensor_types(type_.element, sizes)
    else:
        raise ValueError("page payloads require tensors or banks of tensors")


def _tensors(value: Data) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Bank):
        for element in value.elements:
            yield from _tensors(element)
    else:
        raise ValueError("page payload contains a non-tensor value")


@dataclass(frozen=True, slots=True)
class ExternalFunction:
    """One invocation's imported endpoint, borrowing the backend's original owner."""

    declaration: Function
    function: Callable[[tuple[Data, ...]], tuple[Data, ...]]
    implementation: NativeKernelImport | None = None

    def __post_init__(self) -> None:
        if self.implementation is not None and self.implementation.declaration != self.declaration:
            raise ValueError("external kernel implementation differs from its source declaration")


@dataclass(frozen=True, slots=True)
class BoundResourceAccess:
    resource: str
    mode: AccessMode
    ranges: tuple[tuple[int, int], ...]


class RegionResources:
    """Invocation-bound page mappings, checked borrows and external endpoints.

    Binding names cannot prove independence: different logical cells must have
    distinct physical lineages. Multiple fields in one cell publish atomically.
    Access and tensor bindings are cached here for this owner's invocation only.
    """

    def __init__(
        self,
        program: Program,
        owner: RegionPageOwner,
        cells: tuple[ResourcePageCell, ...],
        externals: tuple[ExternalFunction, ...] = (),
        *,
        sizes: Mapping[str, int] = {},
        borrows: tuple[
            PagedResourceBinding
            | ReadonlyResourceBinding
            | PinnedResourceBinding
            | HostResourceBinding
            | TensorResourceBinding,
            ...,
        ] = (),
        tensor_borrows: tuple[BorrowedTensorImport, ...] = (),
    ) -> None:
        sizes = bind_sizes(program, sizes)
        self._sizes = sizes
        owner.check_page_execution()
        resources = {r.name: r for r in program.resources if isinstance(r, Resource)}
        mappings: dict[tuple[str, tuple[int, ...]], ResourcePageCell] = {}
        lineages: set[LineageId] = set()
        for cell in cells:
            if cell.resource not in resources:
                raise ValueError("page binding names an undeclared resource")
            spec = resources[cell.resource]
            if len(cell.indices) != len(spec.domain) or any(
                type(i) is not int or not 0 <= i < n for i, n in zip(cell.indices, spec.domain, strict=True)
            ):
                raise ValueError("page binding lies outside the logical resource domain")
            key = (cell.resource, cell.indices)
            if key in mappings or cell.original.lineage_id in lineages:
                raise ValueError("logical resource cells alias the same physical lineage")
            if cell.original not in owner.pin.refs:
                raise ValueError("page binding is outside the invocation's original pin")
            types = tuple(t for type_ in spec.payload for t in _tensor_types(type_, sizes))
            if (
                not cell.fields
                or len(types) != len(cell.fields)
                or cell.fields != tuple(sorted(set(cell.fields)))
                or any(
                    type(f) is not int or f < 0 or f >= len(owner.arena.root(cell.original).slots) for f in cell.fields
                )
            ):
                raise ValueError("page payload fields must be complete, nonempty and ordered")
            for field, type_ in zip(cell.fields, types, strict=True):
                # The issued pin validates membership; inspect only metadata.
                value = owner.arena.borrow_field(owner.pin, cell.original, field)
                check_data(value, type_, self._sizes)
            mappings[key] = cell
            lineages.add(cell.original.lineage_id)
        self._resources = program.resources
        self._domains = {resource.name: resource.domain for resource in program.resources}
        self._owner = owner
        self._cells = mappings
        self._types = {r.name: r.payload for r in resources.values()}
        borrowed = {r.name: r for r in program.resources if isinstance(r, BorrowedResource)}
        views = {binding.view.resource.name: binding.view for binding in borrows}
        if len(views) != len(borrows) or set(views) != set(borrowed):
            raise ValueError("borrowed resource proofs must be complete and exact")
        if any(
            binding.owner is not owner or binding.view.resource != borrowed[binding.view.resource.name]
            for binding in borrows
        ):
            raise ValueError("borrowed resource proof belongs to different source storage or invocation")
        if views:
            paged = tuple(binding.view for binding in borrows if isinstance(binding, PagedResourceBinding))
            readonly = tuple(binding.view for binding in borrows if isinstance(binding, ReadonlyResourceBinding))
            host = tuple(binding.view for binding in borrows if isinstance(binding, HostResourceBinding))
            mutable: tuple[tuple[str, int, int], ...] = ()
            for binding in borrows:
                if isinstance(binding, TensorResourceBinding):
                    mutable += binding.view.validate((*owner.arena.allocation_ranges, *mutable))
            validate_paged_views(paged, (*owner.arena.allocation_ranges, *mutable))
            validate_readonly_views(readonly, paged, (*owner.arena.allocation_ranges, *mutable))
            validate_host_views(host)
            for binding in borrows:
                if isinstance(binding, PinnedResourceBinding):
                    binding.validate(program, paged)
                    validate_readonly_views((binding.view,), paged, mutable)
        self._borrows = views
        self._readonly = tuple(view for view in views.values() if isinstance(view, ReadonlyTensorView))
        external_names = {r.name for r in program.resources if isinstance(r, (ExternalResource, BorrowedResource))}
        expected = {f.name: f for f in program.functions if external_names.intersection(f.resources)}
        self._external_declarations = expected
        self._tensor_borrows = {item.declaration.name: item for item in tensor_borrows}
        if len(self._tensor_borrows) != len(tensor_borrows):
            raise ValueError("tensor borrowing repeats a source declaration")
        for item in tensor_borrows:
            item.validate()
            if expected.get(item.declaration.name) != item.declaration or not isinstance(
                views.get(item.declaration.resources[0]), ReadonlyTensorView
            ):
                raise ValueError("tensor borrowing requires its declared original read-only resource")
        expected = {name: function for name, function in expected.items() if name not in self._tensor_borrows}
        functions = {f.declaration.name: f for f in externals}
        if len(functions) != len(externals) or set(functions) != set(expected):
            raise ValueError("external function bindings must be complete and exact")
        for name, function in functions.items():
            if function.declaration != expected[name]:
                raise ValueError("external function differs from its source declaration")
        self._externals = functions
        self._access_arguments = {
            function.name: tuple(
                sorted(
                    {
                        term.argument
                        for access in function.accesses
                        for interval in access.ranges
                        for expression in (interval.lower, interval.upper)
                        for term in expression.terms
                    }
                )
            )
            for function in program.functions
        }
        self._accesses: dict[tuple[str, tuple[Data, ...]], tuple[BoundResourceAccess, ...]] = {}
        self._tensors: dict[tuple[str, tuple[Data, ...]], torch.Tensor] = {}

    def validate(self, program: Program, sizes: Mapping[str, int] = {}) -> None:
        if self._sizes != bind_sizes(program, sizes):
            raise ValueError("resource bindings belong to different size parameters")
        if self._resources != program.resources:
            raise ValueError("resource bindings belong to different source resources")
        names = {r.name for r in program.resources if isinstance(r, (ExternalResource, BorrowedResource))}
        if self._external_declarations != {f.name: f for f in program.functions if names.intersection(f.resources)}:
            raise ValueError("external bindings belong to different source declarations")
        self._owner.check_page_execution()
        for view in self._readonly:
            view.validate()

    def window(self, facts: tuple[RegionStateWindow, ...]) -> AbstractContextManager[None]:
        """Reserve original-owner COW capacity for the compiler's whole window."""
        bounds = {fact.resource: fact.private_roots for fact in facts}
        cells = tuple(self._cells.values())
        return self._owner.arena.reserve_window(
            tuple(self._owner.current_owned_ref(cell.original) for cell in cells),
            tuple(cell.fields for cell in cells),
            tuple(bounds[cell.resource] for cell in cells),
            self._owner.pin.stream,
        )

    def validate_cells(self, uses: Routes) -> None:
        """Cover every cell reached at these sizes, before any statement issues."""
        for ordinal, indices in uses:
            resource = self._resources[ordinal]
            if isinstance(resource, Resource) and (resource.name, indices) not in self._cells:
                raise ValueError("bind every state cell reached by the admitted region sizes")

    def scope(self) -> AbstractContextManager[None]:
        stream = self._owner.pin.stream
        return nullcontext() if stream is None else torch.cuda.stream(stream)

    def validate_readonly_tuning(self, program: Program, sizes: Mapping[str, int]) -> None:
        """Repeated evaluation may borrow state, but cannot invoke owner effects."""
        self.validate(program, sizes)
        if self._externals or any(not isinstance(view, ReadonlyTensorView) for view in self._borrows.values()):
            raise ValueError("tensor tuning requires original readonly tensor resources without external callbacks")

    def validate_stream(self, stream: torch.cuda.Stream) -> None:
        if self._owner.pin.stream != stream:
            raise ValueError("physical lane zero must use the original PTTS pin stream")

    def validate_native_kernels(self, implementations: Mapping[str, NativeKernelImport | None]) -> None:
        for name, endpoint in self._externals.items():
            if name not in implementations or endpoint.implementation != implementations[name]:
                raise ValueError("external endpoint differs from the imported native kernel execution contract")

    def validate_tensor_borrows(self, implementations: Mapping[str, BorrowedTensorImport]) -> None:
        if implementations != self._tensor_borrows:
            raise ValueError("tensor borrows differ from their original source bindings")

    def borrow_tensor(self, implementation: BorrowedTensorImport, inputs: tuple[Data, ...]) -> torch.Tensor:
        self.before_statement()
        function = implementation.declaration
        key = function.name, inputs
        if key in self._tensors:
            return self._tensors[key]
        indices: list[int] = []
        for value in inputs:
            assert type(value) is int
            indices.append(value)
        self.bind_accesses(function, inputs)
        view = self._borrows[function.resources[0]]
        assert isinstance(view, ReadonlyTensorView)
        value = view.tensor(tuple(indices), implementation.field)
        check_data(value, function.outputs[0], self._sizes)
        if not value.is_contiguous():
            raise ValueError("tensor borrow requires the declared contiguous layout")
        self._tensors[key] = value
        return value

    def before_statement(self) -> None:
        self._owner.check_page_execution()

    def bind_accesses(self, declaration: Function, inputs: tuple[Data, ...]) -> tuple[BoundResourceAccess, ...]:
        if not declaration.accesses:
            return ()
        key = declaration.name, tuple(inputs[i] for i in self._access_arguments[declaration.name])
        if key in self._accesses:
            return self._accesses[key]

        def index(expression: AffineIndex) -> int:
            value = expression.constant
            for term in expression.terms:
                operand = inputs[term.argument]
                if type(operand) is not int:
                    raise ValueError("resource access requires its original integer operands")
                product = operand * term.coefficient
                value += product
                if not -(2**63) <= product < 2**63 or not -(2**63) <= value < 2**63:
                    raise ValueError("resource access arithmetic exceeds the signed index range")
            return value

        result: list[BoundResourceAccess] = []
        for access in declaration.accesses:
            ranges = tuple((index(interval.lower), index(interval.upper)) for interval in access.ranges)
            domain = self._domains[access.resource]
            if any(not 0 <= lower <= upper <= size for (lower, upper), size in zip(ranges, domain, strict=True)):
                raise ValueError("native resource access lies outside its declared logical domain")
            view = self._borrows.get(access.resource)
            if view is not None:
                view.check(access.mode, ranges)
            result.append(BoundResourceAccess(access.resource, access.mode, ranges))
        self._accesses[key] = tuple(result)
        return self._accesses[key]

    def external(self, declaration: Function, inputs: tuple[Data, ...]) -> tuple[Data, ...]:
        self.before_statement()
        function = self._externals[declaration.name]
        return function.function(inputs)

    def _cell(self, resource: str, indices: tuple[int, ...]) -> ResourcePageCell:
        try:
            return self._cells[(resource, indices)]
        except KeyError as error:
            raise ValueError("logical resource index is out of bounds") from error

    def snapshot(self, resource: str, indices: tuple[int, ...]) -> tuple[Data, ...]:
        self.before_statement()
        cell = self._cell(resource, indices)
        current = self._owner.current_owned_ref(cell.original)
        pin = self._owner.pin if current == cell.original else self._owner.borrow_published((current,))
        try:
            with self.scope():
                # A functional exported call can return an input view. Owned
                # copies keep such results valid after the page pin is released.
                tensors = iter(
                    tuple(self._owner.arena.borrow_field(pin, current, field).detach().clone() for field in cell.fields)
                )

            def assemble(type_: DataType) -> Data:
                if isinstance(type_, BankType):
                    return Bank(tuple(assemble(type_.element) for _ in range(resolve_extent(type_.size, self._sizes))))
                return next(tensors)

            return tuple(assemble(type_) for type_ in self._types[resource])
        finally:
            if pin is not self._owner.pin:
                self._owner.release_owned_pin(pin)

    def publish(self, resource: str, indices: tuple[int, ...], values: tuple[Data, ...]) -> None:
        self.before_statement()
        cell = self._cell(resource, indices)
        for value, type_ in zip(values, self._types[resource], strict=True):
            check_data(value, type_, self._sizes)
        current = self._owner.current_owned_ref(cell.original)
        token = self._owner.prepare_owned_write((current,), (cell.fields,))
        with self.scope():
            for field, tensor in zip(cell.fields, (t for value in values for t in _tensors(value)), strict=True):
                self._owner.arena.write_field(token, 0, field).copy_(tensor)
        receipt = self._owner.publish_owned_write(token)
        # The original owner records settlement before reclaim can proceed. If
        # settlement fails, its ordinary failure path retains this exact receipt.
        self._owner.settle_owned_publication(receipt)
