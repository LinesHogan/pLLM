"""Array-backed page ownership and GPU completion for PTTS execution, capture, and remote reads."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from math import prod
from multiprocessing.connection import Connection
from threading import RLock
from typing import overload
from weakref import WeakSet

import numpy as np
import numpy.typing as npt
import torch

from pllm.contracts.compilation import PlanBundle
from pllm.contracts.semantic_graph import StateEffectKind
from pllm.contracts.state_ref import LineageId, StateRef, VersionId
from pllm.contracts.state_representation import StateDeviceKind, StateProviderPlan
from pllm.contracts.tensor_layout import LayoutFieldId, TensorLayoutSpec
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.cuda_stream import upload
from pllm.execution.ptts.page_bundle_planning import DTYPE_BYTES, page_stride_bytes
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.scheduler import validate_plan_bundle
from pllm.execution.shared_tensor import CudaIpcStorage, RemoteCompletion, tensor_dtype


@dataclass(frozen=True, slots=True)
class PageSlot:
    slot_id: int
    generation: int


Slots = npt.NDArray[np.int64]


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


@dataclass(frozen=True, slots=True, eq=False)
class PageWrite:
    operation_id: int
    before_refs: StateRefs
    after_refs: StateRefs
    stream: torch.cuda.Stream | None


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


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class PageRelease:
    release_id: int
    pin: PagePin
    completion: torch.cuda.Event | None
    remote_reads: tuple[RemoteCompletion, ...] = ()


@dataclass(frozen=True, slots=True)
class PageFieldTransfer:
    compilation_signature: str
    owner_id: int  # Original PagePin.pin_id; all remote grants are read-only.
    ref: StateRef
    resource: str
    field_id: LayoutFieldId
    page: PageSlot
    storage_offset: int
    cuda_storage: CudaIpcStorage | None = None


@dataclass(slots=True)
class BorrowedPage:
    """Read-only native alias; discard all aliases before close (not OS-enforced)."""

    descriptor: PageFieldTransfer
    _tensor: torch.Tensor | None
    _connection: Connection
    stream: torch.cuda.Stream | None = None

    @property
    def tensor(self) -> torch.Tensor:
        if self._tensor is None:
            raise RuntimeError("remote page borrow is closed")
        if self.stream is not None and torch.cuda.current_stream(self.stream.device) != self.stream:
            raise RuntimeError("remote CUDA page access requires its original receiving stream")
        return self._tensor

    def close(self) -> None:
        if self._tensor is None:
            raise RuntimeError("remote page borrow is already closed")
        if self.stream is not None:
            self.stream.synchronize()
        self._tensor = None
        self._connection.send_bytes(encode_boundary(self.descriptor).encode())


class _Table:
    """Latest versions and lineage ownership for one resource; rows survive retirement until the last hold."""

    def __init__(self, resource: str, pools: _Pools) -> None:
        self.resource, self.pools = resource, pools
        capacity, width = pools.holders.shape[1], len(pools.tensors)
        self.row_of: dict[LineageId, int] = {}
        self.lineages = np.zeros(capacity, dtype=np.int64)
        self.slots = np.zeros((capacity, width), dtype=np.int64)
        self.generations = np.zeros_like(self.slots)
        self.versions = np.zeros(capacity, dtype=np.int64)
        self.readiness: npt.NDArray[np.object_] = np.full(capacity, None, dtype=object)
        self.status = np.zeros(capacity, dtype=np.int8)  # 0 free, 1 pending, 2 live, 3 retired
        # Row holds recycle metadata; only the pool's holders decide page lifetime.
        self.owners = np.zeros(capacity, dtype=np.int64)
        self.writing = np.zeros(capacity, dtype=np.bool_)

    def reserve(self, refs: StateRefs) -> Slots:
        rows = np.flatnonzero(self.status == 0)[: len(refs)]
        assert len(rows) == len(refs)
        self.lineages[rows] = refs.lineages
        self.status[rows] = 1
        self.row_of.update(zip(refs.lineages.tolist(), rows.tolist()))
        return rows

    def ref(self, row: int) -> StateRef:
        return StateRef(
            self.pools.selected.layout_id, LineageId(int(self.lineages[row])), VersionId(int(self.versions[row]))
        )

    def reclaim(self) -> None:
        rows = np.flatnonzero((self.owners == 0) & (self.status != 0))
        for lineage in self.lineages[rows].tolist():
            del self.row_of[lineage]
        self.status[rows] = 0
        self.readiness[rows] = None


@dataclass(slots=True)
class _Pages:
    """An exact batch of pages, owned by a pin, writer, or completion receipt."""

    table: _Table
    refs: StateRefs
    rows: Slots
    slots: Slots
    generations: Slots
    readiness: tuple[torch.cuda.Event | None, ...] = ()
    written: npt.NDArray[np.bool_] | None = None
    _positions: dict[StateRef, int] | None = None

    def position(self, ref: StateRef) -> int | None:
        if self._positions is None:
            self._positions = {ref: index for index, ref in enumerate(self.refs)}
        return self._positions.get(ref)

    def bundle(self, index: int) -> PageBundle:
        return PageBundle(
            self.refs[index],
            self.table.resource,
            self.slots[index],
            self.generations[index],
            self.readiness[index] if self.readiness else None,
        )

    def hold(self) -> None:
        self.table.pools.hold(self.slots)
        self.table.owners += np.bincount(self.rows, minlength=len(self.table.owners))

    def release(self) -> None:
        self.table.pools.release(self.slots)
        self.table.owners -= np.bincount(self.rows, minlength=len(self.table.owners))
        self.table.reclaim()


@dataclass(frozen=True, slots=True)
class _Write:
    targets: tuple[_Pages, ...]
    sources: tuple[_Pages, ...]


@dataclass(slots=True)
class _Publication:
    write: _Write
    settled: bool = False


@dataclass(slots=True)
class _Pin:
    pages: tuple[_Pages, ...]
    remote: list[RemoteCompletion] = dataclass_field(default_factory=lambda: list[RemoteCompletion]())


class LazyRoots(Sequence[tuple[torch.Tensor, ...]]):
    """Batch slot access with tensor views and individual bundles materialized only on demand."""

    def __init__(
        self, arena: PageBundleArena, access: tuple[PagePin | PageWrite, ...], refs: Sequence[StateRef]
    ) -> None:
        self._arena, self._access, self._refs = arena, access, refs
        self._views: dict[int, tuple[torch.Tensor, ...]] = {}
        self._matrix: tuple[str, Slots] | None = None

    def __len__(self) -> int:
        return len(self._refs)

    def slot_matrix(self) -> tuple[str, Slots]:
        """One matrix for all fields, in selection order."""
        if self._matrix is None:
            self._matrix = self._arena.slots(self._access, self._refs)
        return self._matrix

    def field_slots(self, field: int) -> tuple[torch.Tensor, tuple[int, ...]]:
        """The field pool and selected slots, without walking the refs again."""
        resource, slots = self.slot_matrix()
        pool = self._arena._pools[resource].tensors[field]  # pyright: ignore[reportPrivateUsage]
        return pool.detach(), tuple(slots[:, field].tolist())

    @overload
    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[tuple[torch.Tensor, ...], ...]: ...

    def __getitem__(self, index: int | slice) -> tuple[torch.Tensor, ...] | tuple[tuple[torch.Tensor, ...], ...]:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index not in self._views:
            self._views[index] = self._arena.borrow_roots((self._access[index],), (self._refs[index],))[0]
        return self._views[index]


class _Pools:
    def __init__(self, layout: TensorLayoutSpec, selected: StateProviderPlan, device: torch.device) -> None:
        self.layout, self.selected = layout, selected
        capacity = selected.page_capacity
        tensors: list[torch.Tensor] = []
        allocated_bytes = 0
        for field in layout.fields:
            size = DTYPE_BYTES[field.dtype]
            stride = page_stride_bytes(field) // size
            alignment = max(field.alignment_bytes // size, 1)
            count = capacity * stride + alignment - 1
            pool = torch.zeros(count, dtype=tensor_dtype(field.dtype), device=device)
            offset = (-pool.data_ptr() // size) % alignment
            tensors.append(
                pool[offset : offset + capacity * stride]
                .view(capacity, stride)[:, : prod(field.shape)]
                .view(capacity, *field.shape)
            )
            allocated_bytes += pool.numel() * size
        self.tensors = tuple(tensors)
        self.addresses = tuple(pool.data_ptr() for pool in self.tensors)
        # Per field and slot: how many versions share the page (0 = free), and how often it was reused.
        self.holders = np.zeros((len(layout.fields), capacity), dtype=np.int32)
        self.generations = np.zeros((len(layout.fields), capacity), dtype=np.int64)
        self._field_offsets = np.arange(len(layout.fields), dtype=np.int64) * capacity
        if allocated_bytes != selected.reserved_bytes:
            raise ValueError("PageBundle pool allocation differs from signed capacity")

    def free_count(self, field: int) -> int:
        return int(np.count_nonzero(self.holders[field] == 0))

    def allocate(self, field: int, count: int) -> tuple[Slots, Slots]:
        """`count` free pages of one field, smallest slots first, with their generations."""
        slots = np.flatnonzero(self.holders[field] == 0)[:count]
        if len(slots) < count:
            raise RuntimeError("PageBundle capacity is exhausted")
        self.holders[field, slots] = 1
        return slots, self.generations[field, slots]

    def share(self, field: int, slots: Slots) -> None:
        """Let more versions hold these pages (fields a transition does not write)."""
        np.add.at(self.holders[field], slots, 1)

    def tensor(self, field: int, page: PageSlot) -> torch.Tensor:
        if self.generations[field, page.slot_id] != page.generation or self.holders[field, page.slot_id] < 1:
            raise RuntimeError("PageBundle allocation generation was reclaimed")
        return self.tensors[field][page.slot_id]

    def views(self, slots: Slots, generations: Slots) -> tuple[torch.Tensor, ...]:
        """One tensor view per field of a version, after checking none of its pages was reclaimed."""
        fields = np.arange(len(slots))
        if (self.generations[fields, slots] != generations).any() or (self.holders[fields, slots] < 1).any():
            raise RuntimeError("PageBundle allocation generation was reclaimed")
        return tuple(tensor[slot] for tensor, slot in zip(self.tensors, slots.tolist()))

    def _counts(self, slots: Slots) -> Slots:
        return np.bincount((slots + self._field_offsets).ravel(), minlength=self.holders.size).reshape(
            self.holders.shape
        )

    def hold(self, slots: Slots) -> None:
        """Add a batch of exact-version holders, including repeated shared pages."""
        self.holders += self._counts(slots).astype(np.int32)

    def release(self, slots: Slots) -> None:
        """Drop a batch of holders; only the last holder advances the allocation generation."""
        counts = self._counts(slots).astype(np.int32)
        self.holders -= counts
        self.generations += (counts != 0) & (self.holders == 0)


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
        local_candidates: tuple[StateProviderPlan, ...] | None = None,
        *,
        device: torch.device | None = None,
        _owned_resources: tuple[str, ...] | None = None,
    ) -> None:
        validate_plan_bundle(plan, local_candidates=local_candidates)
        self.plan = plan
        placements = {(item.device_kind, item.device_ordinal) for item in plan.state_physical.representations}
        if not placements and not plan.semantic_graph.states and device is not None:
            if f"compute/{device.type}@1" not in plan.timeline.capabilities:
                raise ValueError("stateless execution device differs from the signed timeline")
            self.device = device
        elif len(placements) == 1:
            kind, ordinal = placements.pop()
            self.device = torch.device("cpu") if kind is StateDeviceKind.CPU else torch.device("cuda", ordinal)
            if device is not None and device != self.device:
                raise ValueError("explicit device differs from the signed state placement")
        else:
            raise ValueError("PageBundle arena requires one signed device placement")
        self._pools: dict[str, _Pools] = {}
        layouts = {item.resource: item.layout for item in plan.tensor_layout_request.layouts}
        self._owned_resources = tuple(layouts) if _owned_resources is None else _owned_resources
        if (
            len(set(self._owned_resources)) != len(self._owned_resources)
            or not set(self._owned_resources) <= layouts.keys()
        ):
            raise ValueError("arena ownership must name distinct original plan resources")
        for selected in plan.state_physical.representations:
            # Partial materialization owns no remote pool; test_video tests the partition.
            if selected.resource in self._owned_resources:
                self._pools[selected.resource] = _Pools(layouts[selected.resource], selected, self.device)
        self._tables = {resource: _Table(resource, pools) for resource, pools in self._pools.items()}
        self._writes: dict[PageWrite, _Write] = {}
        self._pins: dict[PagePin, _Pin] = {}
        self._publications: dict[PagePublication, _Publication] = {}
        self._releases: dict[PageRelease, _Pin] = {}
        self._issued_releases: WeakSet[PageRelease] = WeakSet()
        self._pending_events: dict[torch.cuda.Stream, deque[PagePublication | PageRelease]] = {}
        self._ready_receipts: dict[PagePublication | PageRelease, None] = {}
        self._next_lineage = 1
        self._next_operation = 1
        self._next_pin = 1
        self._next_release = 1
        self._poisoned = False
        self._admission_capacity_failures = 0
        self._lock = RLock()
        self._initialization = (
            self._record(torch.cuda.current_stream(self.device)) if self.device.type == "cuda" else None
        )

    @property
    def owned_resources(self) -> tuple[str, ...]:
        """Internal materialization partition; the original plan remains complete."""
        return self._owned_resources

    @property
    def active_lineages(self) -> int:
        with self._lock:
            self.collect_completed()
            return sum(int(np.count_nonzero(table.status >= 2)) for table in self._tables.values())

    @property
    def free_pages(self) -> int:
        with self._lock:
            self.collect_completed()
            return sum(min(map(pools.free_count, range(len(pools.tensors)))) for pools in self._pools.values())

    @property
    def pending_pin_releases(self) -> int:
        return len(self._releases)

    @property
    def pool_addresses(self) -> tuple[int, ...]:
        return tuple(address for pools in self._pools.values() for address in pools.addresses)

    @property
    def allocation_ranges(self) -> tuple[tuple[str, int, int], ...]:
        """Allocation metadata, including spare COW pages and alignment padding."""
        return tuple(
            (str(tensor.device), tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
            for pools in self._pools.values()
            for tensor in pools.tensors
        )

    def usage(self) -> PageBundleUsage:
        """Read original owners without waiting, collecting or retaining tensors/receipts.

        Field-page categories are disjoint: latest roots, then private targets,
        then old/retired roots. Shared fields count once physically, once per
        version logically. These are ownership states, not GPU activity states.
        """
        with self._lock:
            live = tuple(
                self._table_bundle(table, int(row))
                for table in self._tables.values()
                for row in np.flatnonzero(table.status == 2)
            )
            held: dict[StateRef, PageBundle] = {}
            for group in self._held_pages():
                held.update((ref, group.bundle(i)) for i, ref in enumerate(group.refs))
            latest = {root.ref for root in live}
            retiring = tuple(root for ref, root in held.items() if ref not in latest)
            private = tuple(
                group.bundle(i)
                for write in self._writes.values()
                for group in write.targets
                for i in range(len(group.refs))
            )

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
                (resource, field.field_id): prod(field.shape) * DTYPE_BYTES[field.dtype]
                for resource, pool in self._pools.items()
                for field in pool.layout.fields
            }
            reserved = sum(pool.selected.reserved_bytes for pool in self._pools.values())
            return PageBundleUsage(
                reserved,
                sum(
                    sizes[root.resource, field.field_id]
                    for root in (*live, *retiring, *private)
                    for field in self._pools[root.resource].layout.fields
                ),
                sum(sizes[resource, field] for resource, field, _ in live_pages | private_pages | retiring_pages),
                reserved
                - sum(
                    pool.selected.page_capacity * sizes[resource, field.field_id]
                    for resource, pool in self._pools.items()
                    for field in pool.layout.fields
                ),
                sum(pool.free_count(field) for pool in self._pools.values() for field in range(len(pool.tensors))),
                len(live_pages),
                len(private_pages),
                len(retiring_pages),
                len(private),
                len(retiring),
                len(held),
                len(self._publications) + len(self._releases),
                self._admission_capacity_failures,
            )

    def _held_pages(self) -> Sequence[_Pages]:
        return (
            tuple(group for pin in (*self._pins.values(), *self._releases.values()) for group in pin.pages)
            + tuple(
                group
                for publication in self._publications.values()
                for group in (*publication.write.sources, *publication.write.targets)
            )
            + tuple(group for write in self._writes.values() for group in write.sources)
        )

    @staticmethod
    def _table_bundle(table: _Table, row: int) -> PageBundle:
        return PageBundle(
            table.ref(row), table.resource, table.slots[row].copy(), table.generations[row].copy(), table.readiness[row]
        )

    def latest(self, lineage: LineageId) -> StateRef:
        with self._lock:
            self._require_live()
            for table in self._tables.values():
                row = table.row_of.get(lineage)
                if row is not None and table.status[row] == 2:
                    return table.ref(row)
            raise RuntimeError("PageBundle lineage is unadmitted or retiring")

    def resolve_lineages(self, resource: str, lineages: Sequence[LineageId]) -> StateRefs:
        """Resolve directory membership once; selections reuse these rows until membership changes."""
        with self._lock:
            table = self._tables[resource]
            rows = np.asarray([table.row_of.get(lineage, -1) for lineage in lineages], dtype=np.int64)
            if (rows < 0).any() or (table.status[rows] != 2).any():
                raise RuntimeError("PageBundle lineage is unadmitted or retiring")
            return StateRefs(
                np.full(len(rows), table.pools.selected.layout_id, dtype=np.int64),
                table.lineages[rows],
                table.versions[rows],
                rows,
            )

    def latest_refs(self, resource: str, lineages: StateRefs | Sequence[LineageId]) -> StateRefs:
        """Read the latest versions by cached arena row, without allocating individual refs."""
        with self._lock:
            self._require_live()
            refs = lineages if isinstance(lineages, StateRefs) else self.resolve_lineages(resource, lineages)
            table, rows = self._tables[resource], refs.rows
            assert rows is not None
            if (table.status[rows] != 2).any() or (table.lineages[rows] != refs.lineages).any():
                raise RuntimeError("PageBundle lineage is unadmitted or retiring")
            return StateRefs(refs.layouts, refs.lineages, table.versions[rows], rows)

    def root(self, ref: StateRef) -> PageBundle:
        with self._lock:
            self._require_live()
            for table in self._tables.values():
                row = table.row_of.get(ref.lineage_id)
                if row is not None and table.status[row] == 2 and table.versions[row] == ref.version_id:
                    return self._table_bundle(table, row)
            for group in self._held_pages():
                index = group.position(ref)
                if index is not None:
                    return group.bundle(index)
            raise RuntimeError("PageBundle root is unknown or reclaimed")

    def _select(self, refs: Sequence[StateRef], *, latest: bool = True) -> tuple[_Pages, ...]:
        refs = StateRefs.of(refs)
        groups: list[_Pages] = []
        for table in self._tables.values():
            mask = refs.layouts == table.pools.selected.layout_id
            if len(self._tables) > 1:
                # Different resources may share an interned layout ID.
                mask &= np.isin(refs.lineages, table.lineages[table.status != 0])
            selected = refs if mask.all() else refs.take(np.flatnonzero(mask))
            if not selected:
                continue
            rows = selected.rows
            if rows is None:
                rows = np.fromiter(
                    (table.row_of.get(lineage, -1) for lineage in selected.lineages.tolist()), np.int64, len(selected)
                )
                selected = selected.resolved(rows)
            if (rows < 0).any() or (table.lineages[rows] != selected.lineages).any():
                raise RuntimeError("PageBundle root is unknown or reclaimed")
            versions = selected.versions
            if (table.status[rows] != 2).any():
                raise RuntimeError("PageBundle lineage is retiring; no new pins or writes may start")
            if (table.versions[rows] == versions).all():
                groups.append(
                    _Pages(
                        table, selected, rows, table.slots[rows], table.generations[rows], tuple(table.readiness[rows])
                    )
                )
            elif latest:
                raise RuntimeError("PageBundle transition source is not latest")
            else:
                roots = tuple(self.root(ref) for ref in selected)
                groups.append(
                    _Pages(
                        table,
                        selected,
                        rows,
                        np.stack([root.slots for root in roots]),
                        np.stack([root.generations for root in roots]),
                        tuple(root.readiness for root in roots),
                    )
                )
        return tuple(groups)

    def prepare_admission(self, resources: tuple[str, ...], stream: torch.cuda.Stream | None = None) -> PageWrite:
        with self._lock:
            self._require_stream(stream)
            self.collect_completed()
            if not resources or any(resource not in self._pools for resource in resources):
                raise ValueError("PageBundle admission requires signed resources")
            for resource, table in self._tables.items():
                if (
                    np.count_nonzero((table.status == 1) | (table.status == 2)) + resources.count(resource)
                    > table.pools.selected.max_admitted_lineages
                ):
                    self._admission_capacity_failures += 1
                    raise RuntimeError("PageBundle admission capacity is exhausted")
            refs = StateRefs(
                np.asarray([self._pools[resource].selected.layout_id for resource in resources], dtype=np.int64),
                np.arange(self._next_lineage, self._next_lineage + len(resources), dtype=np.int64),
                np.zeros(len(resources), dtype=np.int64),
            )
            self._check_capacity(resources)
            self._next_lineage += len(refs)
            targets: list[_Pages] = []
            for resource in dict.fromkeys(resources):
                table = self._tables[resource]
                selected = refs.take(np.fromiter((i for i, name in enumerate(resources) if name == resource), np.int64))
                rows = table.reserve(selected)
                slots = np.empty((len(rows), len(table.pools.tensors)), dtype=np.int64)
                generations = np.empty_like(slots)
                for field in range(slots.shape[1]):
                    slots[:, field], generations[:, field] = table.pools.allocate(field, len(rows))
                table.owners[rows] += 1
                targets.append(
                    _Pages(table, selected, rows, slots, generations, written=np.ones(slots.shape, dtype=np.bool_))
                )
            return self._prepare((), refs, _Write(tuple(targets), ()), stream, None)

    def _check_capacity(
        self,
        resources: tuple[str, ...],
        sources: tuple[_Pages, ...] = (),
        masks: tuple[npt.NDArray[np.bool_], ...] = (),
    ) -> None:
        demand: dict[str, Slots] = {}
        if sources:
            demand = {group.table.resource: mask.sum(axis=0) for group, mask in zip(sources, masks)}
        else:
            mutable = {
                effect.resource
                for task in self.plan.semantic_graph.tasks
                for effect in task.effects
                if effect.kind not in (StateEffectKind.READ, StateEffectKind.PUBLISH)
            }
            for resource in set(resources):
                table = self._tables[resource]
                count = resources.count(resource)
                headroom = (
                    count + np.count_nonzero((table.status == 1) | ((table.status == 2) & ~table.writing))
                    if resource in mutable
                    else 0
                )
                demand[resource] = np.full(len(table.pools.tensors), count + headroom, dtype=np.int64)
        exhausted = {
            resource
            for resource, counts in demand.items()
            if any(self._pools[resource].free_count(field) < count for field, count in enumerate(counts))
        }
        if exhausted:
            self.collect_completed(wait=True, resources=exhausted)
        if any(
            self._pools[resource].free_count(field) < count
            for resource, counts in demand.items()
            for field, count in enumerate(counts)
        ):
            if not sources:
                self._admission_capacity_failures += 1
            raise RuntimeError("PageBundle capacity is exhausted by live owners")

    def prepare_transition(
        self,
        sources: Sequence[StateRef],
        stream: torch.cuda.Stream | None = None,
        *,
        fields: tuple[tuple[LayoutFieldId, ...], ...] | None = None,
        copy_fields: tuple[LayoutFieldId, ...] | None = None,
    ) -> PageWrite:
        """Allocate private writable fields; unchanged fields share the exact source pages."""
        with self._lock:
            self._require_stream(stream)
            sources = StateRefs.of(sources)
            groups = self._select(sources)
            if any(group.table.writing[group.rows].any() for group in groups):
                raise RuntimeError("PageBundle lineage already has a pending write")
            masks = self._write_masks(groups, sources, fields)
            self._check_capacity((), groups, masks)
            targets = sources.next_versions()
            allocated: list[_Pages] = []
            for group, mask in zip(groups, masks):
                table = group.table
                slots, generations = group.slots.copy(), group.generations.copy()
                for field in range(slots.shape[1]):
                    rows = mask[:, field]
                    slots[rows, field], generations[rows, field] = table.pools.allocate(field, int(rows.sum()))
                    table.pools.share(field, slots[~rows, field])
                refs = targets if len(groups) == 1 else group.refs.next_versions()
                allocated.append(_Pages(table, refs, group.rows, slots, generations, written=mask))
                table.owners[group.rows] += 1
                group.hold()
                table.writing[group.rows] = True
            return self._prepare(sources, targets, _Write(tuple(allocated), groups), stream, copy_fields)

    @staticmethod
    def _write_masks(
        groups: tuple[_Pages, ...], refs: Sequence[StateRef], fields: tuple[tuple[LayoutFieldId, ...], ...] | None
    ) -> tuple[npt.NDArray[np.bool_], ...]:
        masks: list[npt.NDArray[np.bool_]] = []
        for group in groups:
            mask = np.ones(group.slots.shape, dtype=np.bool_)
            if fields is not None:
                chosen = fields if len(groups) == 1 else tuple(fields[refs.index(ref)] for ref in group.refs)
                mask[:] = False
                if all(value == chosen[0] for value in chosen):
                    mask[:, chosen[0]] = True
                else:
                    for row, value in enumerate(chosen):
                        mask[row, value] = True
            masks.append(mask)
        return tuple(masks)

    def _prepare(
        self,
        sources: Sequence[StateRef],
        targets: Sequence[StateRef],
        write: _Write,
        stream: torch.cuda.Stream | None,
        copy_fields: tuple[LayoutFieldId, ...] | None,
    ) -> PageWrite:
        token = PageWrite(self._next_operation, StateRefs.of(sources), StateRefs.of(targets), stream)
        self._next_operation += 1
        self._writes[token] = write
        try:
            with self._scope(stream), torch.no_grad():
                for event in dict.fromkeys(event for group in write.sources for event in group.readiness):
                    self._order(event, stream)
                self._fill_versions(write, copy_fields)
        except Exception:
            self.abort(token)
            raise
        return token

    def _fill_versions(self, write: _Write, copy_fields: tuple[LayoutFieldId, ...] | None) -> None:
        if write.sources and copy_fields == ():
            return
        for index, target in enumerate(write.targets):
            assert target.written is not None
            for field, pool in enumerate(target.table.pools.tensors):
                if write.sources and copy_fields is not None and field not in copy_fields:
                    continue
                rows = target.written[:, field]
                if not rows.any():
                    continue
                destination = target.slots[rows, field].tolist()
                if not write.sources:
                    pool.index_fill_(0, upload(destination, self.device), 0)
                else:
                    origin = write.sources[index].slots[rows, field].tolist()
                    pairs = upload([origin, destination], self.device)
                    pool.index_copy_(0, pairs[1], pool.index_select(0, pairs[0]))

    def write_field(self, token: PageWrite, root_index: int, field: LayoutFieldId) -> torch.Tensor:
        with self._lock:
            (root,) = self._held_fields(token, (token.after_refs[root_index],), (field,))
            return self._pools[root.resource].tensor(field, root.pages[field])

    def writable_roots(self, token: PageWrite) -> tuple[tuple[torch.Tensor, ...], ...]:
        with self._lock:
            self._issued_write(token)
            return self.borrow_roots((token,) * len(token.after_refs), token.after_refs)

    def publish(self, token: PageWrite) -> PagePublication:
        with self._lock:
            write = self._issued_write(token)
            completion = self._record(token.stream)
            receipt = PagePublication(token.operation_id, token.before_refs, token.after_refs, completion)
            for target in write.targets:
                table, rows = target.table, target.rows
                target.hold()  # The table and the publication each own the new pages.
                table.slots[rows], table.generations[rows] = target.slots, target.generations
                table.versions[rows] = target.refs.versions
                table.readiness[rows] = completion
                table.status[rows] = 2
                table.writing[rows] = False
                target.readiness = (completion,) * len(rows)
            for source in write.sources:
                # The writer's source hold now protects the old table pages until completion.
                source.release()
            self._writes.pop(token)
            self._publications[receipt] = _Publication(write)
            self._queue_completion(receipt, token.stream)
            return receipt

    def settle(self, receipt: PagePublication) -> None:
        with self._lock:
            publication = self._publications.get(receipt)
            if publication is None or publication.settled:
                raise RuntimeError("PageBundle publication is not an issued receipt or is already settled")
            publication.settled = True

    def abort(self, token: PageWrite) -> None:
        with self._lock:
            write = self._issued_write(token)
            try:
                completion = self._record(token.stream)
                if completion is not None:
                    completion.synchronize()
            except Exception:
                self._poisoned = True
                raise
            for group in (*write.targets, *write.sources):
                group.table.writing[group.rows] = False
                group.release()
            self._writes.pop(token)

    def pin_exact(self, refs: Sequence[StateRef], stream: torch.cuda.Stream | None = None) -> PagePin:
        with self._lock:
            self._require_stream(stream)
            refs = StateRefs.of(refs)
            groups = self._select(refs, latest=False)
            for event in dict.fromkeys(event for group in groups for event in group.readiness):
                self._order(event, stream)
            for group in groups:
                group.hold()
            token = PagePin(self._next_pin, refs, stream)
            self._next_pin += 1
            self._pins[token] = _Pin(groups)
            return token

    def borrow_field(self, pin: PagePin, ref: StateRef, field: LayoutFieldId) -> torch.Tensor:
        """Read-only page view; use on pin.stream and discard before releasing pin.

        Native consumers must not mutate this alias. Private writes use write_field;
        owned exports use snapshot. The original pin owns completion and reclamation.
        """
        with self._lock:
            (root,) = self._held_fields(pin, (ref,), (field,))
            return self._pools[root.resource].tensor(field, root.pages[field]).detach()

    def _access_pages(self, access: PagePin | PageWrite) -> tuple[_Pages, ...]:
        return self._issued_pin(access).pages if isinstance(access, PagePin) else self._issued_write(access).targets

    def _held_roots(
        self, access: tuple[PagePin | PageWrite, ...], refs: Sequence[StateRef]
    ) -> tuple[tuple[PageBundle, frozenset[int] | None], ...]:
        result: list[tuple[PageBundle, frozenset[int] | None]] = []
        for token, ref in zip(access, refs, strict=True):
            for group in self._access_pages(token):
                index = group.position(ref)
                if index is not None:
                    writable = (
                        None if group.written is None else frozenset(np.flatnonzero(group.written[index]).tolist())
                    )
                    result.append((group.bundle(index), writable))
                    break
            else:
                raise RuntimeError("PageBundle root is not held or written by this transaction")
        return tuple(result)

    def slots(self, access: tuple[PagePin | PageWrite, ...], refs: Sequence[StateRef]) -> tuple[str, Slots]:
        """Resolve all field slots once for a homogeneous resource selection."""
        with self._lock:
            tokens = tuple(dict.fromkeys(access))
            if len(tokens) == 1:
                groups = self._access_pages(tokens[0])
                for group in groups:
                    if refs is group.refs or refs == group.refs:
                        return group.table.resource, group.slots
                positions = {ref: (group, i) for group in groups for i, ref in enumerate(group.refs)}
                selected = [positions[ref] for ref in refs]
            else:
                positions = {
                    token: {ref: (group, i) for group in self._access_pages(token) for i, ref in enumerate(group.refs)}
                    for token in tokens
                }
                selected = [positions[token][ref] for token, ref in zip(access, refs)]
            resource = selected[0][0].table.resource
            assert all(group.table.resource == resource for group, _ in selected)
            return resource, np.stack([group.slots[i] for group, i in selected])

    def lazy_roots(self, access: tuple[PagePin | PageWrite, ...], refs: Sequence[StateRef]) -> LazyRoots:
        """Defer individual tensor views; batch slots stay in their exact pin/write matrix."""
        return LazyRoots(self, access, refs)

    def _root_views(self, root: PageBundle, writable: frozenset[int] | None) -> tuple[torch.Tensor, ...]:
        with self._lock:
            self._require_live()
            pools = self._pools[root.resource]
            if writable is not None and len(writable) != len(root.slots):
                raise RuntimeError("PageBundle field is not writable in this transaction")
            if writable is None:
                return tuple(view.detach() for view in pools.views(root.slots, root.generations))
            return pools.views(root.slots, root.generations)

    def borrow_roots(
        self, access: tuple[PagePin | PageWrite, ...], refs: Sequence[StateRef]
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Every field of every ref, read under its pin (read-only views) or write (writable)."""
        with self._lock:
            self._require_live()
            return tuple(self._root_views(root, writable) for root, writable in self._held_roots(access, refs))

    def field_slots(
        self, access: tuple[PagePin | PageWrite, ...], refs: Sequence[StateRef], field: LayoutFieldId
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        """The field pool and each selected exact-version slot."""
        resource, slots = self.slots(access, refs)
        return self._pools[resource].tensors[field].detach(), tuple(slots[:, field].tolist())

    def _held_fields(
        self, access: PagePin | PageWrite, refs: Sequence[StateRef], fields: tuple[LayoutFieldId, ...]
    ) -> tuple[PageBundle, ...]:
        if not refs or len(set(refs)) != len(refs):
            raise ValueError("captured selection requires distinct nonempty original refs")
        held = self._held_roots((access,) * len(refs), refs)
        if any(writable is not None and not set(fields) <= writable for _, writable in held):
            raise RuntimeError("PageBundle field is not writable in this transaction")
        roots = tuple(root for root, _ in held)
        if len({root.resource for root in roots}) != 1:
            raise ValueError("captured selection must use one original resource layout")
        if any(not 0 <= field < len(roots[0].slots) for field in fields):
            raise ValueError("PageBundle field is outside the signed layout")
        return roots

    def send_field(
        self, pin: PagePin, ref: StateRef, field: LayoutFieldId, connection: Connection, peer_pid: int
    ) -> PageFieldTransfer:
        """Export over a dedicated channel; the original pin owns remote completion."""
        with self._lock:
            value = self.borrow_field(pin, ref, field)
            (root,) = self._held_fields(pin, (ref,), (field,))
            descriptor, completion = self._export(root, field, value, pin.pin_id, connection, pin.stream)
            self._issued_pin(pin).remote.append(completion)
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
        if (
            stream is None
            or value.device.type != "cuda"
            or "state/shared_cuda@1" not in self._pools[root.resource].selected.capabilities
        ):
            raise ValueError("page export requires a signed shared CUDA provider and stream")
        held_reads = (
            read
            for reads in (
                *(pin.remote for pin in self._pins.values()),
                *(release.remote_reads for release in self._releases),
            )
            for read in reads
        )
        if any(remote.connection is connection for remote in held_reads):
            raise ValueError("each remote page access requires its own completion channel")
        descriptor = PageFieldTransfer(
            self.plan.schedule_decision.signature,
            owner_id,
            root.ref,
            root.resource,
            field,
            root.pages[field],
            int(value.storage_offset()),
            CudaIpcStorage.export(value, stream) if value.numel() else None,
        )
        return descriptor, RemoteCompletion(connection, encode_boundary(descriptor).encode())

    @staticmethod
    def receive_field(
        plan: PlanBundle, connection: Connection, *, stream: torch.cuda.Stream | None = None
    ) -> BorrowedPage:
        descriptor = decode_boundary(connection.recv_bytes(65536).decode(), PageFieldTransfer)
        if (
            descriptor.compilation_signature != plan.schedule_decision.signature
            or descriptor.owner_id < 1
            or descriptor.ref.lineage_id < 1
            or descriptor.ref.version_id < 0
        ):
            raise ValueError("shared page belongs to another compilation or owner")
        entries = tuple(
            (entry, binding, provider)
            for entry, binding in zip(
                plan.tensor_layout_request.layouts, plan.tensor_layout_decision.bindings, strict=True
            )
            for provider in plan.state_physical.representations
            if entry.resource == binding.resource == provider.resource == descriptor.resource
        )
        if len(entries) != 1:
            raise ValueError("shared page resource is absent from the original plan")
        entry, binding, provider = entries[0]
        if (
            provider.device_kind is not StateDeviceKind.CUDA
            or "state/shared_cuda@1" not in provider.capabilities
            or binding.layout_id != descriptor.ref.layout_id
            or not 0 <= descriptor.field_id < len(entry.layout.fields)
        ):
            raise ValueError("shared page differs from the signed layout or placement")
        spec = entry.layout.fields[descriptor.field_id]
        capacity = provider.page_capacity
        size = DTYPE_BYTES[spec.dtype]
        stride = page_stride_bytes(spec) // size
        alignment = max(spec.alignment_bytes // size, 1)
        head = descriptor.storage_offset - descriptor.page.slot_id * stride
        if not 0 <= descriptor.page.slot_id < capacity or descriptor.page.generation < 0 or not 0 <= head < alignment:
            raise ValueError("shared page slot or offset is outside signed capacity")
        nbytes = (capacity * stride + alignment - 1) * size
        if stream is None or stream.device != torch.device("cuda", provider.device_ordinal):
            raise ValueError("remote CUDA page requires its signed receiving device and stream")
        if prod(spec.shape):
            if descriptor.cuda_storage is None:
                raise ValueError("remote CUDA page is missing its storage handle")
            tensor = descriptor.cuda_storage.map(
                nbytes, tensor_dtype(spec.dtype), descriptor.storage_offset, spec.shape, stream
            )
            if tensor.data_ptr() % spec.alignment_bytes:
                raise ValueError("CUDA IPC page mapping violates the signed alignment")
        else:
            if descriptor.cuda_storage is not None:
                raise ValueError("empty CUDA page must not carry an allocation handle")
            tensor = torch.empty(spec.shape, dtype=tensor_dtype(spec.dtype), device=stream.device)
        return BorrowedPage(descriptor, tensor, connection, stream)

    def snapshot(self, pin: PagePin, ref: StateRef) -> tuple[torch.Tensor, ...]:
        with self._lock:
            ((root, _),) = self._held_roots((pin,), (ref,))
            pools = self._pools[root.resource]
            with self._scope(pin.stream):
                return tuple(pools.tensor(field, page).detach().clone() for field, page in enumerate(root.pages))

    def release_pin(self, pin: PagePin) -> PageRelease:
        with self._lock:
            held = self._issued_pin(pin)
            receipt = PageRelease(self._next_release, pin, self._record(pin.stream), tuple(held.remote))
            self._next_release += 1
            self._pins.pop(pin)
            self._issued_releases.add(receipt)
            self._releases[receipt] = held
            self._queue_completion(receipt, pin.stream)
            return receipt

    def wait_release(self, receipt: PageRelease) -> None:
        with self._lock:
            self._issued_release(receipt)
            if receipt.completion is not None:
                receipt.completion.synchronize()
            for read in receipt.remote_reads:
                read.synchronize()
            self.collect_completed()

    def release_complete(self, receipt: PageRelease) -> bool:
        with self._lock:
            self._issued_release(receipt)
            self.collect_completed()
            return receipt not in self._releases

    def retire(self, refs: Sequence[StateRef]) -> None:
        with self._lock:
            self._require_live()
            groups = self._select(refs)
            if any(group.table.writing[group.rows].any() for group in groups):
                raise RuntimeError("PageBundle retirement overlaps a write or stale version")
            for group in groups:
                group.table.status[group.rows] = 3
                group.release()
            self.collect_completed()

    def _queue_completion(self, receipt: PagePublication | PageRelease, stream: torch.cuda.Stream | None) -> None:
        if stream is None:
            self._ready_receipts[receipt] = None
        else:
            self._pending_events.setdefault(stream, deque()).append(receipt)

    def _receipt_pages(self, receipt: PagePublication | PageRelease) -> tuple[_Pages, ...]:
        if isinstance(receipt, PageRelease):
            return self._releases[receipt].pages
        write = self._publications[receipt].write
        return (*write.sources, *write.targets)

    def _poll_completions(self, wait: bool, resources: set[str] | None) -> None:
        for stream, queue in tuple(self._pending_events.items()):
            # Under allocation pressure, an earlier event on this stream must finish
            # before a relevant later receipt can release any pages.
            waiting = wait and (
                resources is None
                or any(group.table.resource in resources for receipt in queue for group in self._receipt_pages(receipt))
            )
            while queue:
                receipt = queue[0]
                assert receipt.completion is not None
                if not receipt.completion.query():
                    if not waiting:
                        break
                    receipt.completion.synchronize()
                queue.popleft()
                self._ready_receipts[receipt] = None
            if not queue:
                del self._pending_events[stream]

    def collect_completed(self, *, wait: bool = False, resources: set[str] | None = None) -> None:
        with self._lock:
            self._require_live()
            self._poll_completions(wait, resources)
            completed: list[_Pages] = []
            receipts: list[PagePublication | PageRelease] = []
            for receipt in self._ready_receipts:
                if isinstance(receipt, PagePublication):
                    if not self._publications[receipt].settled:
                        continue
                else:
                    groups = self._releases[receipt].pages
                    relevant = resources is None or any(group.table.resource in resources for group in groups)
                    if wait and relevant and (resources is None or all(read.query() for read in receipt.remote_reads)):
                        for read in receipt.remote_reads:
                            read.synchronize()
                    if not all(read.query() for read in receipt.remote_reads):
                        continue
                completed.extend(self._receipt_pages(receipt))
                receipts.append(receipt)
            # A failed remote proof leaves every owner recorded, including locally ready receipts.
            for table in self._tables.values():
                groups = [group for group in completed if group.table is table]
                if groups:
                    table.pools.release(np.concatenate([group.slots for group in groups]))
                    table.owners -= np.bincount(
                        np.concatenate([group.rows for group in groups]), minlength=len(table.owners)
                    )
                    table.reclaim()
            for receipt in receipts:
                del self._ready_receipts[receipt]
                if isinstance(receipt, PagePublication):
                    del self._publications[receipt]
                else:
                    del self._releases[receipt]

    def _issued_write(self, token: PageWrite) -> _Write:
        self._require_live()
        write = self._writes.get(token)
        if write is None:
            raise RuntimeError("PageBundle write is not an issued token or is already consumed")
        return write

    def _issued_pin(self, pin: PagePin) -> _Pin:
        self._require_live()
        issued = self._pins.get(pin)
        if issued is None:
            raise RuntimeError("PageBundle pin is not an issued token or is released; no new reads may start")
        return issued

    def _issued_release(self, receipt: PageRelease) -> None:
        self._require_live()
        if receipt not in self._issued_releases:
            raise RuntimeError("PageBundle release differs from the issued receipt")

    def _require_live(self) -> None:
        if self._poisoned:
            raise RuntimeError("PageBundle arena is fail-stopped with resources retained")

    def _require_stream(self, stream: torch.cuda.Stream | None) -> None:
        self._require_live()
        if (
            self.device.type == "cpu"
            and stream is not None
            or self.device.type == "cuda"
            and (stream is None or stream.device != self.device)
        ):
            raise RuntimeError("PageBundle stream differs from its signed device")
        self._order(self._initialization, stream)

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
