"""Array-backed page ownership and GPU completion for PTTS execution, capture, and remote reads.

A PageBundleArena is owned by the single thread that runs the model step and has no lock; see StateExecution.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field as dataclass_field
from math import isfinite, prod
from multiprocessing.connection import Connection
from typing import overload

import numpy as np
import numpy.typing as npt
import torch

from pllm.contracts.codec import boundary_digest, encode_boundary
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.semantic_graph import StateEffectKind
from pllm.contracts.state_ref import LineageId, StateRef, VersionId
from pllm.contracts.state_representation import StateProviderPlan
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.admission import WindowBackpressure
from pllm.execution.cuda_stream import upload
from pllm.execution.ptts.page_borrow import PageFieldTransfer
from pllm.execution.ptts.slot_pools import PageSlot, SlotPools, Slots
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.shared_tensor import CudaIpcStorage, RemoteCompletion


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


@dataclass(frozen=True, slots=True, eq=False)
class PageRelease:
    release_id: int
    pin: PagePin
    completion: torch.cuda.Event | None
    remote_reads: tuple[RemoteCompletion, ...] = ()


class _Table:
    """Latest versions and lineage ownership for one resource; rows survive retirement until the last hold."""

    def __init__(self, selected: StateProviderPlan, pools: SlotPools) -> None:
        self.resource, self.selected, self.pools = selected.resource, selected, pools
        capacity, width = pools.holders.shape[1], len(pools.tensors)
        self.row_of: dict[LineageId, int] = {}
        self.lineages = np.zeros(capacity, dtype=np.int64)
        self.slots = np.zeros((capacity, width), dtype=np.int64)
        self.generations = np.zeros_like(self.slots)
        self.versions = np.zeros(capacity, dtype=np.int64)
        self.readiness: npt.NDArray[np.object_] = np.full(capacity, None, dtype=np.object_)
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
        return StateRef(self.selected.layout_id, LineageId(int(self.lineages[row])), VersionId(int(self.versions[row])))

    def reclaim(self) -> None:
        rows = np.flatnonzero((self.owners == 0) & (self.status != 0))
        for lineage in self.lineages[rows].tolist():
            del self.row_of[lineage]
        self.status[rows] = 0
        self.readiness[rows] = None


@dataclass(slots=True)
class _Pages:
    """An exact batch of pages, owned by a pin or a writer."""

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

    def release(self, completion: torch.cuda.Event | None = None) -> None:
        self.table.pools.release(self.slots, completion)
        self.table.owners -= np.bincount(self.rows, minlength=len(self.table.owners))
        self.table.reclaim()


@dataclass(frozen=True, slots=True)
class _Write:
    targets: tuple[_Pages, ...]
    sources: tuple[_Pages, ...]


@dataclass(slots=True)
class _Pin:
    pages: tuple[_Pages, ...]
    remote: list[RemoteCompletion] = dataclass_field(default_factory=lambda: list[RemoteCompletion]())


class LazyRoots(Sequence[tuple[torch.Tensor, ...]]):
    """Batch slot access with tensor views and individual bundles materialized only on demand."""

    def __init__(self, arena: PageBundleArena, access: tuple[PagePin | PageWrite, ...], refs: StateRefs) -> None:
        self._arena, self._access, self._refs = arena, access, refs
        self._views: dict[int, tuple[torch.Tensor, ...]] = {}
        self._matrix: tuple[str, Slots] | None = None

    def __len__(self) -> int:
        return len(self._refs)

    def select(self, indices: tuple[int, ...]) -> LazyRoots:
        """Borrow a resource-specific subset under the same issued page access."""
        return LazyRoots(self._arena, tuple(self._access[i] for i in indices), self._refs.take(indices))

    def slot_matrix(self) -> tuple[str, Slots]:
        """One matrix for all fields, in selection order."""
        if self._matrix is None:
            self._matrix = self._arena.slots(self._access, self._refs)
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
            self._views[index] = self._arena.borrow_roots((self._access[index],), self._refs.take((index,)))[0]
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
        _owned_resources: tuple[str, ...] | None = None,
        map_host: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.plan = plan
        self.plan_digest = boundary_digest(plan)
        device = device or torch.device(plan.states.plans[0].device_kind.value)
        self.device = torch.device("cuda", torch.cuda.current_device()) if device == torch.device("cuda") else device
        self._pools: dict[str, SlotPools] = {}
        self._tables: dict[str, _Table] = {}
        layouts = {item.resource: item.layout for item in plan.tensor_layout_request.layouts}
        self._owned_resources = tuple(layouts) if _owned_resources is None else _owned_resources
        for selected in plan.states.plans:
            # Partial materialization owns no remote pool; test_video tests the partition.
            if selected.resource in self._owned_resources:
                self._pools[selected.resource] = SlotPools(
                    layouts[selected.resource],
                    selected.page_capacity,
                    self.device,
                    host=bool(selected.host_bytes),
                    map_host=map_host,
                )
                self._tables[selected.resource] = _Table(selected, self._pools[selected.resource])
        self._writes: dict[PageWrite, _Write] = {}
        self._pins: dict[PagePin, _Pin] = {}
        # Pins with remote readers keep their pages until every remote read is acknowledged.
        self._releases: dict[PageRelease, _Pin] = {}
        self._storage_streams: set[int] = set()
        self._next_lineage = 1
        self._next_operation = 1
        self._next_pin = 1
        self._next_release = 1
        self._poisoned = False
        self._admission_capacity_failures = 0
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
        return sum(int(np.count_nonzero(table.status >= 2)) for table in self._tables.values())

    @property
    def free_pages(self) -> int:
        self.collect_completed()
        return sum(min(map(pools.free_count, range(len(pools.tensors)))) for pools in self._pools.values())

    @property
    def pending_pin_releases(self) -> int:
        return len(self._releases)

    @property
    def allocation_ranges(self) -> tuple[tuple[str, int, int], ...]:
        """Allocation metadata, including spare COW pages and alignment padding."""
        return tuple(
            (str(tensor.device), tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
            for pools in self._pools.values()
            for tensor in pools.tensors
        )

    def usage(self) -> PageBundleUsage:
        """Read original owners without waiting, collecting or retaining tensors.

        Field-page categories are disjoint: latest roots, then private targets,
        then old/retired roots. Shared fields count once physically, once per
        version logically. These are ownership states, not GPU activity states.
        """
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
            (resource, field.field_id): prod(field.shape) * field.dtype.itemsize
            for resource, pool in self._pools.items()
            for field in pool.layout.fields
        }
        reserved = sum(table.selected.reserved_bytes for table in self._tables.values())
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
                table.selected.page_capacity * sizes[resource, field.field_id]
                for resource, table in self._tables.items()
                for field in table.pools.layout.fields
            ),
            sum(pool.free_count(field) for pool in self._pools.values() for field in range(len(pool.tensors))),
            len(live_pages),
            len(private_pages),
            len(retiring_pages),
            len(private),
            len(retiring),
            len(held),
            len(self._releases),
            self._admission_capacity_failures,
        )

    def _held_pages(self) -> Sequence[_Pages]:
        return (
            tuple(group for pin in self._pins.values() for group in pin.pages)
            + tuple(group for pin in self._releases.values() for group in pin.pages)
            + tuple(group for write in self._writes.values() for group in write.sources)
        )

    @staticmethod
    def _table_bundle(table: _Table, row: int) -> PageBundle:
        return PageBundle(
            table.ref(row), table.resource, table.slots[row].copy(), table.generations[row].copy(), table.readiness[row]
        )

    def latest(self, lineage: LineageId) -> StateRef:
        self._require_live()
        for table in self._tables.values():
            row = table.row_of.get(lineage)
            if row is not None and table.status[row] == 2:
                return table.ref(row)
        raise KeyError(lineage)

    def resolve_lineages(self, resource: str, lineages: Sequence[LineageId]) -> StateRefs:
        """Resolve directory membership once; selections reuse these rows until membership changes."""
        table = self._tables[resource]
        rows = np.asarray([table.row_of.get(lineage, -1) for lineage in lineages], dtype=np.int64)
        assert (rows >= 0).all() and (table.status[rows] == 2).all()
        return StateRefs(
            np.full(len(rows), table.selected.layout_id, dtype=np.int64),
            table.lineages[rows],
            table.versions[rows],
            rows,
        )

    def latest_batch(self, resource: str, refs: StateRefs) -> StateRefs:
        """Read the latest versions by cached arena row, without allocating individual refs."""
        self._require_live()
        table, rows = self._tables[resource], refs.rows
        assert rows is not None
        assert (table.status[rows] == 2).all() and (table.lineages[rows] == refs.lineages).all()
        return StateRefs(refs.layouts, refs.lineages, table.versions[rows], rows)

    def root(self, ref: StateRef) -> PageBundle:
        self._require_live()
        for table in self._tables.values():
            row = table.row_of.get(ref.lineage_id)
            if row is not None and table.status[row] == 2 and table.versions[row] == ref.version_id:
                return self._table_bundle(table, row)
        for group in self._held_pages():
            index = group.position(ref)
            if index is not None:
                return group.bundle(index)
        raise KeyError(ref)

    def _select(self, refs: StateRefs, *, latest: bool = True) -> tuple[_Pages, ...]:
        groups: list[_Pages] = []
        for table in self._tables.values():
            mask = refs.layouts == table.selected.layout_id
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
            assert (rows >= 0).all() and (table.lineages[rows] == selected.lineages).all()
            versions = selected.versions
            assert (table.status[rows] == 2).all()
            if (table.versions[rows] == versions).all():
                groups.append(
                    _Pages(
                        table, selected, rows, table.slots[rows], table.generations[rows], tuple(table.readiness[rows])
                    )
                )
            else:
                assert not latest
                # Lagged readers usually select an entire retained step. Resolve batches
                # against the issued holds instead of scanning every hold for every row.
                slots, generations = table.slots[rows].copy(), table.generations[rows].copy()
                readiness = table.readiness[rows].copy()
                missing = table.versions[rows] != versions
                for held in self._held_pages():
                    if held.table is not table or not missing.any():
                        continue
                    positions = held.refs.positions(selected)
                    found = missing & (positions >= 0)
                    indices = positions[found]
                    slots[found], generations[found] = held.slots[indices], held.generations[indices]
                    if held.readiness:
                        readiness[found] = np.asarray(held.readiness, dtype=np.object_)[indices]
                    else:
                        readiness[found] = None
                    missing[found] = False
                assert not missing.any()
                groups.append(
                    _Pages(
                        table,
                        selected,
                        rows,
                        slots,
                        generations,
                        tuple(readiness),
                    )
                )
        return tuple(groups)

    def prepare_admission(self, resources: tuple[str, ...], stream: torch.cuda.Stream | None = None) -> PageWrite:
        self._require_stream(stream)
        self.collect_completed()
        for resource, table in self._tables.items():
            if (
                np.count_nonzero((table.status == 1) | (table.status == 2)) + resources.count(resource)
                > table.selected.max_admitted_lineages
            ):
                self._admission_capacity_failures += 1
                raise RuntimeError("PageBundle admission capacity is exhausted")
        refs = StateRefs(
            np.asarray([self._tables[resource].selected.layout_id for resource in resources], dtype=np.int64),
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
                slots[:, field], generations[:, field] = table.pools.allocate(field, len(rows), stream)
            table.owners[rows] += 1
            targets.append(
                _Pages(table, selected, rows, slots, generations, written=np.ones(slots.shape, dtype=np.bool_))
            )
        return self._prepare(StateRefs.of(()), refs, _Write(tuple(targets), ()), stream, None)

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
            self.collect_completed()
        if any(
            self._pools[resource].free_count(field) < count
            for resource, counts in demand.items()
            for field, count in enumerate(counts)
        ):
            if not sources:
                self._admission_capacity_failures += 1
                raise WindowBackpressure("page capacity is held by live owners; retry admission asynchronously")
            raise RuntimeError("page capacity is unavailable without waiting")

    def prepare_transition(
        self,
        sources: StateRefs,
        stream: torch.cuda.Stream | None = None,
        *,
        fields: tuple[tuple[LayoutFieldId, ...], ...] | None = None,
        copy_fields: tuple[LayoutFieldId, ...] | None = None,
    ) -> PageWrite:
        """Allocate private writable fields; unchanged fields share the exact source pages."""
        self._require_stream(stream)
        groups = self._select(sources)
        assert not any((group.table.writing[group.rows].any() for group in groups))
        masks = self._write_masks(groups, sources, fields)
        self._check_capacity((), groups, masks)
        targets = sources.next_versions()
        allocated: list[_Pages] = []
        for group, mask in zip(groups, masks):
            table = group.table
            slots, generations = group.slots.copy(), group.generations.copy()
            for field in range(slots.shape[1]):
                rows = mask[:, field]
                slots[rows, field], generations[rows, field] = table.pools.allocate(field, int(rows.sum()), stream)
                table.pools.share(field, slots[~rows, field])
            refs = targets if len(groups) == 1 else group.refs.next_versions()
            allocated.append(_Pages(table, refs, group.rows, slots, generations, written=mask))
            table.owners[group.rows] += 1
            group.hold()
            table.writing[group.rows] = True
        return self._prepare(sources, targets, _Write(tuple(allocated), groups), stream, copy_fields)

    @staticmethod
    def _write_masks(
        groups: tuple[_Pages, ...], refs: StateRefs, fields: tuple[tuple[LayoutFieldId, ...], ...] | None
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
        sources: StateRefs,
        targets: StateRefs,
        write: _Write,
        stream: torch.cuda.Stream | None,
        copy_fields: tuple[LayoutFieldId, ...] | None,
    ) -> PageWrite:
        token = PageWrite(self._next_operation, sources, targets, stream)
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
                    target.table.pools.copy_from(field, pool, pairs[0], pairs[1])

    def write_field(self, token: PageWrite, root_index: int, field: LayoutFieldId) -> torch.Tensor:
        (root,) = self._held_fields(token, token.after_refs.take((root_index,)), (field,))
        return self._pools[root.resource].tensor(field, root.pages[field])

    def writable_roots(self, token: PageWrite, *, initialize: bool = False) -> tuple[tuple[torch.Tensor, ...], ...]:
        self._issued_write(token)
        return tuple(
            self._pools[root.resource].views(root.slots, root.generations, host=initialize)
            for root, _ in self._held_roots((token,) * len(token.after_refs), token.after_refs)
        )

    def publish(self, token: PageWrite) -> PagePublication:
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
        self._writes.pop(token)
        # Each source page loses the writer's hold and the table's hold on the replaced version.
        for group in (*write.targets, *write.sources, *write.sources):
            group.release(completion)
        return receipt

    def abort(self, token: PageWrite) -> None:
        """End logical ownership now; future physical reuse waits for the queued writes."""
        write = self._issued_write(token)
        completion = self._record(token.stream)
        for group in (*write.targets, *write.sources):
            group.table.writing[group.rows] = False
            group.release(completion)
        self._writes.pop(token)

    def pin_exact(self, refs: StateRefs, stream: torch.cuda.Stream | None = None) -> PagePin:
        self._require_stream(stream)
        groups = self._select(refs, latest=False)
        for event in dict.fromkeys(event for group in groups for event in group.readiness):
            self._order(event, stream)
        for group in groups:
            group.hold()
        token = PagePin(self._next_pin, refs, stream)
        self._next_pin += 1
        self._pins[token] = _Pin(groups)
        return token

    def move_pin(self, pin: PagePin, stream: torch.cuda.Stream) -> PagePin:
        """Transfer an exact hold to a stream ordered after the prior reader, retaining remote readers."""
        self._require_stream(stream)
        self._issued_pin(pin)
        if pin.stream is not None and pin.stream != stream:
            stream.wait_stream(pin.stream)
        moved = PagePin(pin.pin_id, pin.refs, stream)
        self._pins[moved] = self._pins.pop(pin)
        return moved

    def borrow_field(self, pin: PagePin, ref: StateRef, field: LayoutFieldId) -> torch.Tensor:
        """Read-only page view; use on pin.stream and discard before releasing pin.

        Native consumers must not mutate this alias. Private writes use write_field;
        owned exports use snapshot. The original pin owns completion and reclamation.
        """
        (root,) = self._held_fields(pin, StateRefs.of((ref,)), (field,))
        return self._pools[root.resource].tensor(field, root.pages[field]).detach()

    def _access_pages(self, access: PagePin | PageWrite) -> tuple[_Pages, ...]:
        return self._issued_pin(access).pages if isinstance(access, PagePin) else self._issued_write(access).targets

    def _held_roots(
        self, access: tuple[PagePin | PageWrite, ...], refs: StateRefs
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
                raise KeyError(ref)
        return tuple(result)

    def slot_matrix(self, access: PagePin | PageWrite, refs: StateRefs) -> tuple[str, Slots]:
        """Resolve a homogeneous array selection under its issued pin or private write."""
        for group in self._access_pages(access):
            if refs is group.refs or refs == group.refs:
                return group.table.resource, group.slots
            if len(refs) and np.all(refs.layouts == group.refs.layouts[0]):
                positions = group.refs.positions(refs)
                if np.all(positions >= 0):
                    return group.table.resource, group.slots[positions]
        raise KeyError(refs)

    def slots(self, access: tuple[PagePin | PageWrite, ...], refs: StateRefs) -> tuple[str, Slots]:
        """Resolve all field slots once for a homogeneous resource selection."""
        tokens = tuple(dict.fromkeys(access))
        if len(tokens) == 1:
            return self.slot_matrix(tokens[0], refs)
        else:
            positions = {
                token: {ref: (group, i) for group in self._access_pages(token) for i, ref in enumerate(group.refs)}
                for token in tokens
            }
            selected = [positions[token][ref] for token, ref in zip(access, refs)]
        resource = selected[0][0].table.resource
        assert all(group.table.resource == resource for group, _ in selected)
        return resource, np.stack([group.slots[i] for group, i in selected])

    def lazy_roots(self, access: tuple[PagePin | PageWrite, ...], refs: StateRefs) -> LazyRoots:
        """Defer individual tensor views; batch slots stay in their exact pin/write matrix."""
        return LazyRoots(self, access, refs)

    def _root_views(self, root: PageBundle, writable: frozenset[int] | None) -> tuple[torch.Tensor, ...]:
        self._require_live()
        pools = self._pools[root.resource]
        assert writable is None or len(writable) == len(root.slots)
        if writable is None:
            return tuple(view.detach() for view in pools.views(root.slots, root.generations))
        return pools.views(root.slots, root.generations)

    def borrow_roots(
        self, access: tuple[PagePin | PageWrite, ...], refs: StateRefs
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Every field of every ref, read under its pin (read-only views) or write (writable)."""
        self._require_live()
        return tuple(self._root_views(root, writable) for root, writable in self._held_roots(access, refs))

    def field_pool(self, resource: str, field: LayoutFieldId) -> torch.Tensor:
        """Bind stable graph storage; each invocation still supplies pin-authorized read slots."""
        return self._pools[resource].tensors[field].detach()

    def field_slots(
        self, access: tuple[PagePin | PageWrite, ...], refs: StateRefs, field: LayoutFieldId
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        """The field pool and each selected exact-version slot."""
        resource, slots = self.slots(access, refs)
        return self._pools[resource].tensors[field].detach(), tuple(slots[:, field].tolist())

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
        pools = self._pools[root.resource]
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
        for table in self._tables.values():
            for row in np.flatnonzero(table.status == 2):
                ref = table.ref(int(row))
                pin = self.pin_exact(StateRefs.of((ref,)), stream)
                try:
                    result.extend(
                        FieldNumericHealth.read(table.resource, ref, LayoutFieldId(field), value)
                        for field, value in enumerate(self.snapshot(pin, ref))
                    )
                finally:
                    self.wait_release(self.release_pin(pin))
        return tuple(result)

    def release_pin(self, pin: PagePin) -> PageRelease:
        held = self._issued_pin(pin)
        receipt = PageRelease(self._next_release, pin, self._record(pin.stream), tuple(held.remote))
        self._next_release += 1
        self._pins.pop(pin)
        if receipt.remote_reads:
            self._releases[receipt] = held
        else:
            self._release_pages(held, receipt.completion)
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
        return receipt not in self._releases

    def retire(self, refs: StateRefs) -> None:
        self._require_live()
        groups = self._select(refs)
        assert not any((group.table.writing[group.rows].any() for group in groups))
        for group in groups:
            group.table.status[group.rows] = 3
            group.release()
        self.collect_completed()

    def _release_pages(self, pin: _Pin, completion: torch.cuda.Event | None) -> None:
        for group in pin.pages:
            group.release(completion)

    def collect_completed(self, *, wait: bool = False) -> None:
        """Release pins whose remote readers have finished; local releases already happened behind their fence."""
        self._require_live()
        for receipt, held in tuple(self._releases.items()):
            if wait:
                for read in receipt.remote_reads:
                    read.synchronize()
            elif not all(read.query() for read in receipt.remote_reads):
                continue
            self._release_pages(held, receipt.completion)
            del self._releases[receipt]

    def _issued_write(self, token: PageWrite) -> _Write:
        self._require_live()
        write = self._writes.get(token)
        assert write is not None
        return write

    def _issued_pin(self, pin: PagePin) -> _Pin:
        self._require_live()
        issued = self._pins.get(pin)
        assert issued is not None
        return issued

    def _require_live(self) -> None:
        if self._poisoned:
            raise RuntimeError("PageBundle arena is fail-stopped with resources retained")

    def _require_stream(self, stream: torch.cuda.Stream | None) -> None:
        self._require_live()
        if stream is None and self.device.type == "cuda":
            raise ValueError("A CUDA PageBundleArena needs a stream; pass torch.cuda.current_stream()")
        self._order(self._initialization, stream)
        if stream is not None and stream.cuda_stream not in self._storage_streams:
            # The arena may be destroyed after asynchronous cancellation. CUDA's
            # allocator must retain its backing on every declared reader/writer lane.
            for pools in self._pools.values():
                for tensor in pools.tensors:
                    tensor.record_stream(stream)
            self._storage_streams.add(stream.cuda_stream)

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
