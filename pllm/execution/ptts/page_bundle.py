"""Enqueue tensor work and CUDA completions around Rust's PTTS ownership ledger.

A PageBundleArena is owned by the single thread that runs the model step and has no lock; see StateExecution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from itertools import chain
from math import isfinite, prod
from multiprocessing.connection import Connection
from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import torch

from pllm import _core
from pllm.contracts.codec import boundary_digest, encode_boundary
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.semantic_graph import StateEffectKind
from pllm.contracts.state_ref import LayoutId, LineageId, StateRef, VersionId
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.admission import WindowBackpressure
from pllm.execution.cuda_stream import upload
from pllm.execution.ptts.page_borrow import PageFieldTransfer
from pllm.execution.ptts.slot_pools import PageEvents, PageSlot, SlotPools, Slots
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.shared_tensor import CudaIpcStorage, RemoteCompletion

CopyFields = tuple[LayoutFieldId, ...] | Literal["defer"] | None


@dataclass(frozen=True, slots=True, eq=False)
class PageBundle:
    """One state version: the page slot and allocation generation holding each field."""

    ref: StateRef
    resource: str
    slots: Slots  # [fields]
    generations: Slots  # [fields]
    readiness: torch.cuda.Event | None

    @property
    def pages(self) -> tuple[PageSlot, ...]:
        return tuple(PageSlot(int(slot), int(generation)) for slot, generation in zip(self.slots, self.generations))


@dataclass(slots=True)
class DeferredCopies:
    """An overwrite may consume its incoming page directly; untouched destinations still inherit at publication."""

    resource_ids: Mapping[str, int]
    fields: dict[tuple[str, int], dict[int, int]]

    @classmethod
    def of(cls, copies: list[_core.PageCopy], resources: tuple[str, ...]) -> DeferredCopies:
        return cls(
            {name: index for index, name in enumerate(resources)},
            {
                (resources[resource], field): dict(zip(destinations.tolist(), sources.tolist(), strict=True))
                for resource, field, sources, destinations in copies
            },
        )

    def slots(self, resource: str, field: int, slot: int, writing: bool) -> tuple[int, int]:
        pages = self.fields.get((resource, field))
        if pages is None or slot not in pages:
            return slot, -1
        source = pages[slot]
        if writing:
            pages[slot] = slot
        return source, slot if writing else -1

    def materialize(self, fill: Callable[[list[_core.PageCopy]], None]) -> None:
        copies: list[_core.PageCopy] = []
        pending: list[tuple[dict[int, int], Slots]] = []
        for (resource, field), pages in self.fields.items():
            pairs = [(src, dst) for dst, src in pages.items() if src != dst]
            if pairs:
                sources, destinations = np.asarray(pairs, dtype=np.int64).T
                copies.append((self.resource_ids[resource], field, sources, destinations))
                pending.append((pages, destinations))
        if copies:
            fill(copies)
            for pages, destinations in pending:
                pages.update((slot, slot) for slot in destinations.tolist())


@dataclass(frozen=True, slots=True, eq=False)
class PageWrite:
    operation_id: int
    before_refs: StateRefs
    after_refs: StateRefs
    stream: torch.cuda.Stream | None
    deferred: DeferredCopies | None = None


@dataclass(frozen=True, slots=True, eq=False)
class PagePublication:
    operation_id: int
    before_refs: StateRefs
    after_refs: StateRefs
    completion: torch.cuda.Event | None


@dataclass(frozen=True, slots=True, eq=False)
class PagePin:
    pin_id: int
    refs: StateRefs
    stream: torch.cuda.Stream | None


@dataclass(frozen=True, slots=True, eq=False)
class PageRelease:
    release_id: int
    pin: PagePin
    completion: torch.cuda.Event | None
    remote_reads: tuple[RemoteCompletion, ...] = ()


@dataclass(frozen=True, slots=True)
class _Pages:
    """An inspection snapshot; all ownership changes belong to the native ledger."""

    resource: str
    refs: StateRefs
    slots: Slots
    generations: Slots
    readiness: tuple[torch.cuda.Event | None, ...]
    written: npt.NDArray[np.bool_] | None

    def position(self, ref: StateRef) -> int | None:
        position = int(self.refs.positions(StateRefs.of((ref,)))[0])
        return position if position >= 0 else None

    def bundle(self, index: int) -> PageBundle:
        return PageBundle(
            self.refs[index],
            self.resource,
            self.slots[index],
            self.generations[index],
            self.readiness[index] if self.readiness else None,
        )


def _ref_columns(refs: StateRefs) -> tuple[Slots, Slots, Slots, Slots | None]:
    return refs.layouts, refs.lineages, refs.versions, refs.rows


def _refs(columns: Slots) -> StateRefs:
    return StateRefs(columns[0], columns[1], columns[2], columns[3])


def _access_id(access: PagePin | PageWrite) -> int:
    return access.pin_id if isinstance(access, PagePin) else -access.operation_id


def _stream_id(stream: torch.cuda.Stream | None) -> int:
    return 0 if stream is None else stream.cuda_stream


class LazyRoots(Sequence[tuple[torch.Tensor, ...]]):
    """Batch slot access with tensor views and individual bundles materialized only on demand."""

    def __init__(
        self, arena: PageBundleArena, access: tuple[PagePin | PageWrite, ...], owners: Slots, refs: StateRefs
    ) -> None:
        self._arena, self._access, self._owners, self._refs = arena, access, owners, refs
        self._views: dict[int, tuple[torch.Tensor, ...]] = {}
        self._matrix: tuple[str, Slots] | None = None

    def __len__(self) -> int:
        return len(self._refs)

    def select(self, indices: Slots) -> LazyRoots:
        """Borrow a resource-specific subset under the same issued page access."""
        return LazyRoots(self._arena, self._access, self._owners[indices], self._refs.take(indices))

    def slot_matrix(self) -> tuple[str, Slots]:
        """One matrix for all fields, in selection order."""
        if self._matrix is None:
            self._matrix = self._arena.slots(self._access, self._owners, self._refs)
        return self._matrix

    def field_slots(self, field: int) -> tuple[torch.Tensor, tuple[int, ...]]:
        """The field pool and selected slots, without walking the refs again."""
        resource, slots = self.slot_matrix()
        return self._arena.field_pool(resource, LayoutFieldId(field)), tuple(slots[:, field].tolist())

    @overload
    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[tuple[torch.Tensor, ...], ...]: ...

    def __getitem__(self, index: int | slice) -> tuple[torch.Tensor, ...] | tuple[tuple[torch.Tensor, ...], ...]:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index not in self._views:
            self._views[index] = self._arena.borrow_roots(
                (self._access[int(self._owners[index])],), self._refs.take((index,))
            )[0]
        return self._views[index]


@dataclass(frozen=True, slots=True)
class FieldNumericHealth:
    """One live published field; undefined or overflowing norms/maxima are None."""

    resource: str
    ref: StateRef
    field: LayoutFieldId
    norm: float | None
    max_abs: float | None
    nonfinite_count: int

    @classmethod
    def read(cls, resource: str, ref: StateRef, field: LayoutFieldId, value: torch.Tensor) -> FieldNumericHealth:
        """Read the L2 norm, absolute maximum and nonfinite count, including nonfinite values.

        FP64 reductions use bounded chunks; NaN/Inf norms and maxima become None.
        Empty fields have zero norm/maximum. Chunking may change FP64 rounding.
        """
        value = value.detach().flatten()
        stats = torch.zeros(3, dtype=torch.float64, device=value.device)
        for offset in range(0, value.numel(), 1 << 20):
            chunk = value[offset : offset + (1 << 20)].double()
            stats[0] += chunk.square().sum()
            stats[1] = torch.maximum(stats[1], chunk.abs().amax())
            stats[2] += (~chunk.isfinite()).sum()
        stats[0] = stats[0].sqrt()
        stats = stats.cpu()
        norm, maximum, count = (float(stat.item()) for stat in stats)
        return cls(
            resource, ref, field, norm if isfinite(norm) else None, maximum if isfinite(maximum) else None, int(count)
        )


@dataclass(frozen=True, slots=True)
class PageBundleUsage:
    """Derived owner snapshot, not a reservation or another page/version ledger."""

    reserved_bytes: int
    logical_live_bytes: int
    allocated_payload_bytes: int
    reserved_padding_bytes: int
    free_field_pages: int
    live_field_pages: int
    private_field_pages: int
    retiring_field_pages: int
    private_versions: int
    retiring_versions: int
    pinned_versions: int
    pending_receipts: int
    admission_capacity_failures: int


class PageBundleArena:
    """CPU/CUDA share admission, COW and atomic multi-root publication.

    Tensor fields are physical pages: unchanged fields share slots, writable
    fields receive private slots. Only this arena chooses logical versions.
    """

    def __init__(
        self,
        plan: PlanBundle,
        *,
        device: torch.device | None = None,
        owned_resources: tuple[str, ...] | None = None,
        map_host: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.plan = plan
        self.plan_digest = boundary_digest(plan)
        device = device or torch.device(plan.states.plans[0].device_kind.value)
        self.device = torch.device("cuda", torch.cuda.current_device()) if device == torch.device("cuda") else device
        self.pools: dict[str, SlotPools] = {}
        layouts = {item.resource: item.layout for item in plan.tensor_layout_request.layouts}
        self._owned_resources = tuple(layouts) if owned_resources is None else owned_resources
        self._plans = {
            selected.resource: selected for selected in plan.states.plans if selected.resource in self._owned_resources
        }
        states = {str(state.resource): state for state in plan.semantic_graph.states}
        for resource, selected in self._plans.items():
            self.pools[resource] = SlotPools(
                layouts[resource],
                selected.page_capacities,
                self.device,
                host=bool(selected.host_bytes),
                map_host=map_host,
                writer_fields=states[resource].writer_fields,
            )
        self.resource_names = tuple(self._plans)
        self._resource_ids = {resource: index for index, resource in enumerate(self.resource_names)}
        mutable = {
            effect.resource
            for task in plan.semantic_graph.tasks
            for effect in task.effects
            if effect.kind not in (StateEffectKind.READ, StateEffectKind.PUBLISH)
        }
        specs = np.array(
            [
                (selected.layout_id, selected.max_admitted_lineages, resource in mutable)
                for resource, selected in self._plans.items()
            ],
            dtype=np.int64,
        ).reshape(-1, 3)
        pinned = [
            [field not in pools.writer_fields for field in range(len(pools.tensors))] for pools in self.pools.values()
        ]
        self._ledger = _core.PageLedger([pool.ledger for pool in self.pools.values()], specs, pinned)
        self._events = PageEvents()
        self._remote_reads: dict[int, list[RemoteCompletion]] = {}
        # Remote readers retain their native pin until every ACK arrives.
        self.pending_releases: dict[PageRelease, None] = {}
        self._storage_streams: dict[int, torch.cuda.Stream] = {}
        self._poisoned = False
        self._initialization = (
            self._record(torch.cuda.current_stream(self.device)) if self.device.type == "cuda" else None
        )

    @property
    def owned_resources(self) -> tuple[str, ...]:
        """Internal materialization partition; the original plan remains complete."""
        return self._owned_resources

    @property
    def active_states(self) -> int:
        self.collect_completed()
        return self._ledger.active_states

    @property
    def free_pages(self) -> int:
        self.collect_completed()
        return sum(min(map(pools.free_count, range(len(pools.tensors)))) for pools in self.pools.values())

    @property
    def pending_pin_releases(self) -> int:
        return len(self.pending_releases)

    @property
    def allocation_ranges(self) -> tuple[tuple[str, int, int], ...]:
        """Allocation metadata, including spare COW pages and alignment padding."""
        return tuple(
            (str(tensor.device), tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
            for pools in self.pools.values()
            for tensor in pools.tensors
        )

    def usage(self) -> PageBundleUsage:
        """Read original owners without waiting, collecting or retaining tensors.

        Field-page categories are disjoint: latest roots, then private targets,
        then old/retired roots. Shared fields count once physically, once per
        version logically. These are ownership states, not GPU activity states.
        """
        live = tuple(group.bundle(i) for group in self._inspect("live") for i in range(len(group.refs)))
        held: dict[StateRef, PageBundle] = {}
        for group in self._inspect("held"):
            held.update((ref, group.bundle(i)) for i, ref in enumerate(group.refs))
        latest = {root.ref for root in live}
        retiring = tuple(root for ref, root in held.items() if ref not in latest)
        private = tuple(group.bundle(i) for group in self._inspect("private") for i in range(len(group.refs)))

        def pages(roots: tuple[PageBundle, ...]) -> set[tuple[str, LayoutFieldId, int]]:
            return {
                (root.resource, LayoutFieldId(field), page.slot_id)
                for root in roots
                for field, page in enumerate(root.pages)
            }

        live_pages = pages(live)
        private_pages = pages(private) - live_pages
        retiring_pages = pages(retiring) - live_pages - private_pages
        sizes = {
            (resource, field.field_id): prod(field.shape) * field.dtype.itemsize
            for resource, pool in self.pools.items()
            for field in pool.layout.fields
        }
        reserved = sum(selected.reserved_bytes for selected in self._plans.values())
        return PageBundleUsage(
            reserved,
            sum(
                sizes[root.resource, field.field_id]
                for root in (*live, *retiring, *private)
                for field in self.pools[root.resource].layout.fields
            ),
            sum(sizes[resource, field] for resource, field, _ in live_pages | private_pages | retiring_pages),
            reserved
            - sum(
                selected.page_capacities[field.field_id] * sizes[resource, field.field_id]
                for resource, selected in self._plans.items()
                for field in self.pools[resource].layout.fields
            ),
            sum(pool.free_count(field) for pool in self.pools.values() for field in range(len(pool.tensors))),
            len(live_pages),
            len(private_pages),
            len(retiring_pages),
            len(private),
            len(retiring),
            len(held),
            len(self.pending_releases),
            self._ledger.admission_failures,
        )

    def _pages(self, groups: list[_core.PageSnapshot]) -> tuple[_Pages, ...]:
        return tuple(
            _Pages(
                self.resource_names[resource],
                _refs(refs),
                slots,
                generations,
                tuple(self._events.get(key) for key in readiness.tolist()),
                written,
            )
            for resource, refs, slots, generations, readiness, written in groups
        )

    def _inspect(self, category: str) -> tuple[_Pages, ...]:
        return self._pages(self._ledger.inspect(category))

    def _held_pages(self) -> Sequence[_Pages]:
        return self._inspect("held")

    def latest(self, lineage: LineageId) -> StateRef:
        self._require_live()
        layout, lineage_id, version = self._ledger.latest(lineage)
        return StateRef(LayoutId(layout), LineageId(lineage_id), VersionId(version))

    def resolve_lineages(self, resource: str, lineages: Sequence[LineageId]) -> StateRefs:
        """Resolve directory membership once; selections reuse these rows until membership changes."""
        return _refs(self._ledger.resolve_lineages(self._resource_ids[resource], np.asarray(lineages, dtype=np.int64)))

    def latest_batch(self, resource: str, columns: Slots) -> StateRefs:
        """Read latest versions using the native projection's layout, lineage and arena-row columns."""
        self._require_live()
        versions = self._ledger.latest_batch(self._resource_ids[resource], columns)
        return StateRefs(columns[0], columns[1], versions, columns[2])

    def root(self, ref: StateRef) -> PageBundle:
        self._require_live()
        for group in (*self._inspect("live"), *self._held_pages()):
            index = group.position(ref)
            if index is not None:
                return group.bundle(index)
        raise KeyError(ref)

    def prepare_admission(self, resources: tuple[str, ...], stream: torch.cuda.Stream | None = None) -> PageWrite:
        self._require_stream(stream)
        self.collect_completed()
        try:
            prepared = self._ledger.prepare_admission(
                np.fromiter((self._resource_ids[resource] for resource in resources), dtype=np.int64),
                self._events.completed(),
            )
        except _core.PageCapacity as error:
            raise WindowBackpressure(str(error)) from error
        return self._prepare(prepared, StateRefs.of(()), stream)

    def prepare_transition(
        self,
        sources: StateRefs,
        stream: torch.cuda.Stream | None = None,
        *,
        fields: tuple[tuple[LayoutFieldId, ...], ...] | None = None,
        copy_fields: CopyFields = None,
    ) -> PageWrite:
        """Unwritten fields share the source. Deferred writers claim each destination before overwriting it."""
        self._require_stream(stream)
        field_columns = None
        if fields is not None:
            offsets = np.cumsum([0, *(len(value) for value in fields)], dtype=np.int64)
            field_columns = offsets, np.fromiter(chain.from_iterable(fields), dtype=np.int64)
        args = (
            _ref_columns(sources),
            field_columns,
            None if copy_fields is None or copy_fields == "defer" else np.asarray(copy_fields, dtype=np.int64),
            self._events.completed(),
        )
        try:
            prepared = self._ledger.prepare_transition(*args)
        except _core.PageCapacity:
            self.collect_completed()
            try:
                prepared = self._ledger.prepare_transition(*args)
            except _core.PageCapacity as error:
                raise RuntimeError(str(error)) from error
        return self._prepare(prepared, sources, stream, defer_copies=copy_fields == "defer")

    def _prepare(
        self,
        prepared: _core.PagePrepared,
        sources: StateRefs,
        stream: torch.cuda.Stream | None,
        defer_copies: bool = False,
    ) -> PageWrite:
        operation, columns, copies, waits, retained = prepared
        token = PageWrite(operation, sources, _refs(columns), stream)
        try:
            self._events.order(waits, stream)
            self._events.retain(retained)
            if defer_copies:
                return replace(token, deferred=DeferredCopies.of(copies, self.resource_names))
            with self._scope(stream), torch.no_grad():
                self.fill_versions(copies)
        except Exception:
            self.abort(token)
            raise
        return token

    def fill_versions(self, copies: list[_core.PageCopy]) -> None:
        """Copy the planned state fields into their destination version slots."""
        for resource, field, source, destination in copies:
            pools = self.pools[self.resource_names[resource]]
            if not len(source):
                pools.tensors[field].index_fill_(0, upload(destination, self.device), 0)
            else:
                pairs = upload(np.stack((source, destination)), self.device)
                pools.copy_from(field, pools.tensors[field], pairs[0], pairs[1])

    def write_generations(self, token: PageWrite, resource: str) -> Slots:
        """Return allocation generations of `resource`'s target slots to validate cached page bounds."""
        return next(pages.generations for pages in self._access_pages(token) if pages.resource == resource)

    def write_field(self, token: PageWrite, root_index: int, field: LayoutFieldId) -> torch.Tensor:
        (root,) = self._held_fields(token, token.after_refs.take((root_index,)), (field,))
        if token.deferred is not None:
            self.materialize(token)
        return self.pools[root.resource].tensor(field, root.pages[field])

    def writable_roots(self, token: PageWrite, *, initialize: bool = False) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Borrow writable fields, completing GPU work before handing storage to a CPU initializer."""
        self.materialize(token)
        roots = tuple(
            self.pools[root.resource].views(root.slots, root.generations, host=initialize)
            for root, _ in self._held_roots((token,) * len(token.after_refs), token.after_refs)
        )
        if (
            initialize
            and token.stream is not None
            and any(field.device.type == "cpu" for root in roots for field in root)
        ):
            # Admission zeroing and prior GPU readers must finish before CPU writes to mapped pages.
            token.stream.synchronize()
        return roots

    def materialize(self, token: PageWrite) -> None:
        """Raw private readers and publication require a value even for unvisited targets."""
        self._issued_write(token)
        if token.deferred is not None:
            with self._scope(token.stream), torch.no_grad():
                token.deferred.materialize(self.fill_versions)

    def publish(self, token: PageWrite) -> PagePublication:
        self.materialize(token)
        completion = self._record(token.stream)
        completed = self._events.completed()
        retained = self._ledger.publish(token.operation_id, self._events.add(completion), completed, False)
        self._events.retain(retained)
        return PagePublication(token.operation_id, token.before_refs, token.after_refs, completion)

    def abort(self, token: PageWrite) -> None:
        """End logical ownership now; future physical reuse waits for the queued writes."""
        self._issued_write(token)
        completion = self._record(token.stream)
        completed = self._events.completed()
        retained = self._ledger.publish(token.operation_id, self._events.add(completion), completed, True)
        self._events.retain(retained)

    def pin_exact(self, refs: StateRefs, stream: torch.cuda.Stream | None = None) -> PagePin:
        self._require_stream(stream)
        pin_id, waits = self._ledger.pin_exact(_ref_columns(refs), _stream_id(stream))
        self._events.order(waits, stream)
        token = PagePin(pin_id, refs, stream)
        return token

    def move_pin(self, pin: PagePin, stream: torch.cuda.Stream) -> PagePin:
        """Transfer an exact hold to a stream ordered after the prior reader, retaining remote readers."""
        self._require_stream(stream)
        self._issued_pin(pin)
        if pin.stream is not None and pin.stream != stream:
            stream.wait_stream(pin.stream)
        moved = PagePin(pin.pin_id, pin.refs, stream)
        self._ledger.move_pin(pin.pin_id, _stream_id(pin.stream), _stream_id(stream))
        return moved

    def borrow_field(self, pin: PagePin, ref: StateRef, field: LayoutFieldId) -> torch.Tensor:
        """Read-only page view; use on pin.stream and discard before releasing pin.

        Native consumers must not mutate this alias. Private writes use write_field;
        owned exports use snapshot. The original pin owns completion and reclamation.
        """
        (root,) = self._held_fields(pin, StateRefs.of((ref,)), (field,))
        return self.pools[root.resource].tensor(field, root.pages[field]).detach()

    def _access_pages(self, access: PagePin | PageWrite) -> tuple[_Pages, ...]:
        if isinstance(access, PagePin):
            self._issued_pin(access)
        else:
            self._issued_write(access)
        return self._pages(self._ledger.access_pages(_access_id(access), _stream_id(access.stream)))

    def _held_roots(
        self, access: tuple[PagePin | PageWrite, ...], refs: StateRefs
    ) -> tuple[tuple[PageBundle, frozenset[int] | None], ...]:
        result: list[tuple[PageBundle, frozenset[int] | None]] = []
        groups = {token: self._access_pages(token) for token in dict.fromkeys(access)}
        for token, ref in zip(access, refs, strict=True):
            for group in groups[token]:
                index = group.position(ref)
                if index is not None:
                    writable = (
                        None if group.written is None else frozenset(np.flatnonzero(group.written[index]).tolist())
                    )
                    result.append((group.bundle(index), writable))
                    break
            else:
                raise KeyError(ref)
        return tuple(result)

    def ref_resource(self, refs: StateRefs) -> str:
        """Look up a homogeneous selection's resource by its arena-wide lineage identity."""
        return self.resource_names[self._ledger.resource(int(refs.lineages[0]))]

    def access_resource(self, access: PagePin | PageWrite) -> str:
        """Read the homogeneous resource of an issued access without inspecting state records."""
        return self.resource_names[self._ledger.access_resource(_access_id(access), _stream_id(access.stream))]

    def slot_matrix(self, access: PagePin | PageWrite, refs: StateRefs) -> tuple[str, Slots]:
        """Resolve a homogeneous array selection under its issued pin or private write."""
        return self.slots((access,), np.empty(0, dtype=np.int64), refs)

    def slots(self, access: tuple[PagePin | PageWrite, ...], owners: Slots, refs: StateRefs) -> tuple[str, Slots]:
        """Resolve numeric owner positions and exact slots in one native call."""
        self._require_live()
        resource, slots = self._ledger.slots(
            np.array(
                [[_access_id(token) for token in access], [_stream_id(token.stream) for token in access]],
                dtype=np.int64,
            ),
            owners,
            _ref_columns(refs),
        )
        return self.resource_names[resource], slots

    def lazy_roots(self, access: tuple[PagePin | PageWrite, ...], owners: Slots, refs: StateRefs) -> LazyRoots:
        """Defer individual tensor views; batch slots stay in their exact pin/write matrix."""
        return LazyRoots(self, access, owners, refs)

    def _root_views(self, root: PageBundle, writable: frozenset[int] | None) -> tuple[torch.Tensor, ...]:
        self._require_live()
        pools = self.pools[root.resource]
        assert writable is None or len(writable) == len(root.slots)
        if writable is None:
            return tuple(view.detach() for view in pools.views(root.slots, root.generations))
        return pools.views(root.slots, root.generations)

    def borrow_roots(
        self, access: tuple[PagePin | PageWrite, ...], refs: StateRefs
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Every field of every ref, read under its pin (read-only views) or write (writable)."""
        self._require_live()
        for token in access:
            if isinstance(token, PageWrite) and token.deferred is not None:
                self.materialize(token)
        return tuple(self._root_views(root, writable) for root, writable in self._held_roots(access, refs))

    def field_pool(self, resource: str, field: LayoutFieldId) -> torch.Tensor:
        """Bind stable graph storage; each invocation still supplies pin-authorized read slots."""
        return self.pools[resource].tensors[field].detach()

    def field_slots(
        self, access: tuple[PagePin | PageWrite, ...], refs: StateRefs, field: LayoutFieldId
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        """The field pool and each selected exact-version slot."""
        resource, slots = self.slots(access, np.arange(len(access), dtype=np.int64), refs)
        return self.pools[resource].tensors[field].detach(), tuple(slots[:, field].tolist())

    def _held_fields(
        self, access: PagePin | PageWrite, refs: StateRefs, fields: tuple[LayoutFieldId, ...]
    ) -> tuple[PageBundle, ...]:
        held = self._held_roots((access,) * len(refs), refs)
        assert all(writable is None or set(fields) <= writable for _, writable in held)
        roots = tuple(root for root, _ in held)
        return roots

    def send_field(
        self, pin: PagePin, ref: StateRef, field: LayoutFieldId, connection: Connection, peer_pid: int
    ) -> PageFieldTransfer:
        """Export over a dedicated channel; the original pin owns remote completion."""
        value = self.borrow_field(pin, ref, field)
        (root,) = self._held_fields(pin, StateRefs.of((ref,)), (field,))
        descriptor, completion = self._export(root, field, value, pin.pin_id, connection, pin.stream)
        self._remote_reads.setdefault(pin.pin_id, []).append(completion)
        # Retain completion before sending: even a partial send may expose the mapping.
        connection.send_bytes(encode_boundary(descriptor).encode())
        return descriptor

    def _export(
        self,
        root: PageBundle,
        field: LayoutFieldId,
        value: torch.Tensor,
        owner_id: int,
        connection: Connection,
        stream: torch.cuda.Stream | None,
    ) -> tuple[PageFieldTransfer, RemoteCompletion]:
        assert stream is not None
        descriptor = PageFieldTransfer(
            self.plan_digest,
            owner_id,
            root.ref,
            root.resource,
            field,
            root.pages[field],
            int(value.storage_offset()),
            CudaIpcStorage.export(value, stream) if value.numel() else None,
        )
        return descriptor, RemoteCompletion(connection, encode_boundary(descriptor).encode())

    def snapshot(self, pin: PagePin, ref: StateRef) -> tuple[torch.Tensor, ...]:
        """Clone a pinned version in its declared placement, ready on the caller's stream."""
        ((root, _),) = self._held_roots((pin,), StateRefs.of((ref,)))
        pools = self.pools[root.resource]
        with self._scope(pin.stream):
            copies = tuple(value.detach().clone() for value in pools.views(root.slots, root.generations, host=True))
        if self.device.type == "cuda":
            assert pin.stream is not None
            consumer = torch.cuda.current_stream(self.device)
            if consumer != pin.stream:
                consumer.wait_stream(pin.stream)
                for value in copies:
                    if value.is_cuda:
                        value.record_stream(consumer)
        return copies

    def numeric_health(self) -> tuple[FieldNumericHealth, ...]:
        """Inspect current published roots via their exact snapshots; no work runs on the publish/step path."""
        stream = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        result: list[FieldNumericHealth] = []
        for group in self._inspect("live"):
            for ref in group.refs:
                pin = self.pin_exact(StateRefs.of((ref,)), stream)
                try:
                    result.extend(
                        FieldNumericHealth.read(group.resource, ref, LayoutFieldId(field), value)
                        for field, value in enumerate(self.snapshot(pin, ref))
                    )
                finally:
                    self.wait_release(self.release_pin(pin))
        return tuple(result)

    def release_pin(self, pin: PagePin) -> PageRelease:
        self._issued_pin(pin)
        remote = tuple(self._remote_reads.get(pin.pin_id, ()))
        completion = self._record(pin.stream)
        completed = self._events.completed()
        release_id, retained = self._ledger.release_pin(
            pin.pin_id,
            self._events.add(completion),
            bool(remote),
            _stream_id(pin.stream),
            completed,
        )
        self._events.retain(retained)
        receipt = PageRelease(release_id, pin, completion, remote)
        self._remote_reads.pop(pin.pin_id, None)
        if remote:
            self.pending_releases[receipt] = None
        return receipt

    def wait_release(self, receipt: PageRelease) -> None:
        self._require_live()
        if receipt.completion is not None:
            receipt.completion.synchronize()
        for read in receipt.remote_reads:
            read.synchronize()
        self.collect_completed()

    def release_complete(self, receipt: PageRelease) -> bool:
        self._require_live()
        self.collect_completed()
        return receipt not in self.pending_releases

    def retire(self, refs: StateRefs) -> None:
        self._require_live()
        self._events.retain(self._ledger.retire(_ref_columns(refs)))
        self.collect_completed()

    def collect_completed(self, *, wait: bool = False) -> None:
        """Release pins whose remote readers have finished; local releases already happened behind their fence."""
        self._require_live()
        for receipt in tuple(self.pending_releases):
            if wait:
                for read in receipt.remote_reads:
                    read.synchronize()
            elif not all(read.query() for read in receipt.remote_reads):
                continue
            retained = self._ledger.complete_release(receipt.release_id, self._events.completed())
            self._events.retain(retained)
            del self.pending_releases[receipt]

    def _issued_write(self, token: PageWrite) -> None:
        self._require_live()
        assert self._ledger.valid_access(-token.operation_id, _stream_id(token.stream))

    def _issued_pin(self, pin: PagePin) -> None:
        self._require_live()
        assert self._ledger.valid_access(pin.pin_id, _stream_id(pin.stream))

    def _require_live(self) -> None:
        if self._poisoned:
            raise RuntimeError(
                "this engine's state storage stopped after an earlier error; create a new engine with pllm.make_llm"
            )

    def _require_stream(self, stream: torch.cuda.Stream | None) -> None:
        self._require_live()
        if stream is None and self.device.type == "cuda":
            raise ValueError("A CUDA PageBundleArena needs a stream; pass torch.cuda.current_stream()")
        self._order(self._initialization, stream)
        if stream is not None and stream.cuda_stream not in self._storage_streams:
            # The arena may be destroyed after asynchronous cancellation. CUDA's
            # allocator must retain its backing on every declared reader/writer lane.
            for pools in self.pools.values():
                for tensor in pools.tensors:
                    tensor.record_stream(stream)
            self._storage_streams[stream.cuda_stream] = stream

    def _record(self, stream: torch.cuda.Stream | None) -> torch.cuda.Event | None:
        if stream is None:
            return None
        try:
            event = torch.cuda.Event(blocking=False)
            event.record(stream)
            return event
        except Exception:
            self._poisoned = True
            raise

    @staticmethod
    def _scope(stream: torch.cuda.Stream | None) -> AbstractContextManager[None]:
        return nullcontext() if stream is None else torch.cuda.stream(stream)

    @staticmethod
    def _order(event: torch.cuda.Event | None, stream: torch.cuda.Stream | None) -> None:
        if event is not None and stream is not None:
            stream.wait_event(event)
