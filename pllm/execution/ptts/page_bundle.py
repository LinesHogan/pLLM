"""One version, generation and completion owner for signed tensor PageBundles."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from itertools import accumulate
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
from pllm.execution.ptts.page_capture import CapturedPageFields, PagedKernel
from pllm.execution.scheduler import validate_plan_bundle
from pllm.execution.shared_tensor import CudaIpcStorage, RemoteCompletion, tensor_dtype
from pllm.kernels.adamw import AdamWConfig, paged_adamw
from pllm.kernels.paged_linear import PagedRows


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
    before_refs: tuple[StateRef, ...]
    after_refs: tuple[StateRef, ...]
    stream: torch.cuda.Stream | None


@dataclass(frozen=True, slots=True, eq=False)
class PagePublication:
    operation_id: int
    before_refs: tuple[StateRef, ...]
    after_refs: tuple[StateRef, ...]
    completion: torch.cuda.Event | None


@dataclass(frozen=True, slots=True, eq=False)
class PagePin:
    pin_id: int
    refs: tuple[StateRef, ...]
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


@dataclass(slots=True)
class _Root:
    bundle: PageBundle
    pins: int = 0


@dataclass(frozen=True, slots=True)
class _Write:
    targets: tuple[PageBundle, ...]
    fields: tuple[tuple[LayoutFieldId, ...], ...]
    # after_ref -> (private root, writable field ids), built once per write.
    by_ref: dict[StateRef, tuple[PageBundle, frozenset[int]]]


@dataclass(slots=True)
class _Pin:
    """An issued pin: the exact roots it holds, and remote reads that must finish before release."""

    roots: dict[StateRef, PageBundle]
    remote: list[RemoteCompletion] = dataclass_field(default_factory=lambda: list[RemoteCompletion]())


class LazyRoots(Sequence[tuple[torch.Tensor, ...]]):
    """Field views of each selected state, built on first read.

    Access was validated when the sequence was created; each view still checks that its
    page has not been reclaimed, so a stale read fails instead of seeing reused memory.
    """

    __slots__ = ("_arena", "_roots", "_views", "_slots")

    def __init__(self, arena: "PageBundleArena", roots: tuple[tuple[PageBundle, frozenset[int] | None], ...]) -> None:
        self._arena = arena
        self._roots = roots
        self._views: list[tuple[torch.Tensor, ...] | None] = [None] * len(roots)
        self._slots: dict[int, tuple[torch.Tensor, tuple[int, ...]]] = {}

    def __len__(self) -> int:
        return len(self._roots)

    def field_slots(self, field: int) -> tuple[torch.Tensor, tuple[int, ...]]:
        """The field's page pool and each state's slot in it, from the access validated at creation."""
        if field not in self._slots:
            self._slots[field] = self._arena._slots_of(self._roots, LayoutFieldId(field))  # pyright: ignore[reportPrivateUsage]
        return self._slots[field]

    @overload
    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[tuple[torch.Tensor, ...], ...]: ...

    def __getitem__(self, index: int | slice) -> tuple[torch.Tensor, ...] | tuple[tuple[torch.Tensor, ...], ...]:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        views = self._views[index]
        if views is None:
            views = self._views[index] = self._arena._root_views(*self._roots[index])  # pyright: ignore[reportPrivateUsage]
        return views


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

    def release(self, slots: Slots, generations: Slots) -> None:
        """Drop one hold on each page of `slots` `[versions, fields]` (or one version `[fields]`)."""
        slots, generations = np.atleast_2d(slots), np.atleast_2d(generations)
        fields: Slots = np.broadcast_to(np.arange(slots.shape[1], dtype=np.int64), slots.shape)
        if (self.generations[fields, slots] != generations).any() or (self.holders[fields, slots] < 1).any():
            raise RuntimeError("PageBundle allocation generation was reclaimed")
        np.subtract.at(self.holders, (fields, slots), 1)
        touched = np.zeros_like(self.holders, dtype=np.bool_)
        touched[fields, slots] = True
        self.generations[touched & (self.holders == 0)] += 1


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
        self._roots: dict[StateRef, _Root] = {}
        self._lineages: dict[LineageId, StateRef | None] = {}
        self._writes: dict[PageWrite, _Write] = {}
        # Bookkeeping that keeps per-step work proportional to what changed:
        self._stale: set[StateRef] = set()  # roots that are no longer latest; reclamation candidates
        self._retiring: set[LineageId] = set()  # retired lineages that still own roots
        self._root_count: Counter[LineageId] = Counter()  # roots per lineage
        self._writing: set[LineageId] = set()  # lineages with a pending private write
        self._pins: dict[PagePin, _Pin] = {}
        self._publications: dict[PagePublication, bool] = {}
        self._releases: set[PageRelease] = set()
        self._issued_releases: WeakSet[PageRelease] = WeakSet()
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
            return len(self._lineages)

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
            latest = set(self._lineages.values())
            live = tuple(root.bundle for ref, root in self._roots.items() if ref in latest)
            retiring = tuple(root.bundle for ref, root in self._roots.items() if ref not in latest)
            private = tuple(root for write in self._writes.values() for root in write.targets)

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
                sum(root.pins > 0 for root in self._roots.values()),
                len(self._publications) + len(self._releases),
                self._admission_capacity_failures,
            )

    def latest(self, lineage: LineageId) -> StateRef:
        with self._lock:
            self._require_live()
            latest = self._lineages.get(lineage)
            if latest is None:
                raise RuntimeError("PageBundle lineage is unadmitted or retiring")
            return latest

    def root(self, ref: StateRef) -> PageBundle:
        with self._lock:
            self._require_live()
            try:
                return self._roots[ref].bundle
            except KeyError as exc:
                raise RuntimeError("PageBundle root is unknown or reclaimed") from exc

    def prepare_admission(self, resources: tuple[str, ...], stream: torch.cuda.Stream | None = None) -> PageWrite:
        with self._lock:
            self._require_stream(stream)
            self.collect_completed()
            if not resources or any(resource not in self._pools for resource in resources):
                raise ValueError("PageBundle admission requires signed resources")
            for resource, pools in self._pools.items():
                live = sum(ref is not None and self.root(ref).resource == resource for ref in self._lineages.values())
                pending = sum(
                    root.resource == resource
                    for token, write in self._writes.items()
                    if not token.before_refs
                    for root in write.targets
                )
                if live + pending + resources.count(resource) > pools.selected.max_admitted_lineages:
                    self._admission_capacity_failures += 1
                    raise RuntimeError("PageBundle admission capacity is exhausted")
            refs = tuple(
                StateRef(self._pools[resource].selected.layout_id, LineageId(self._next_lineage + index), VersionId(0))
                for index, resource in enumerate(resources)
            )
            self._next_lineage += len(refs)
            fields = tuple(
                tuple(field.field_id for field in self._pools[resource].layout.fields) for resource in resources
            )
            return self._prepare((), refs, resources, fields, stream)

    def prepare_transition(
        self,
        sources: tuple[StateRef, ...],
        stream: torch.cuda.Stream | None = None,
        *,
        fields: tuple[tuple[LayoutFieldId, ...], ...] | None = None,
        copy_fields: tuple[LayoutFieldId, ...] | None = None,
    ) -> PageWrite:
        """Copy chosen fields by default; an explicit subset requires overwriting the rest."""
        with self._lock:
            self._require_stream(stream)
            self.collect_completed()
            if not sources or len({ref.lineage_id for ref in sources}) != len(sources):
                raise ValueError("PageBundle transition requires distinct lineages")
            resources = tuple(self.root(ref).resource for ref in sources)
            for ref in sources:
                if self.latest(ref.lineage_id) != ref:
                    raise RuntimeError("PageBundle transition source is not latest")
                if ref.lineage_id in self._writing:
                    raise RuntimeError("PageBundle lineage already has a pending write")
            fields = (
                fields
                if fields is not None
                else tuple(
                    tuple(field.field_id for field in self._pools[resource].layout.fields) for resource in resources
                )
            )
            if len(fields) != len(sources) or any(
                not chosen
                or chosen != tuple(sorted(set(chosen)))
                or any(field < 0 or field >= len(self._pools[resource].layout.fields) for field in chosen)
                for chosen, resource in zip(fields, resources, strict=True)
            ):
                raise ValueError("PageBundle writable fields differ from the signed layout")
            if copy_fields is not None and (
                copy_fields != tuple(sorted(set(copy_fields)))
                or any(not set(copy_fields) <= set(chosen) for chosen in fields)
            ):
                raise ValueError("PageBundle copied fields must be an ordered subset of each private field selection")
            targets = tuple(StateRef(ref.layout_id, ref.lineage_id, VersionId(ref.version_id + 1)) for ref in sources)
            return self._prepare(sources, targets, resources, fields, stream, copy_fields=copy_fields)

    def _prepare(
        self,
        sources: tuple[StateRef, ...],
        targets: tuple[StateRef, ...],
        resources: tuple[str, ...],
        fields: tuple[tuple[LayoutFieldId, ...], ...],
        stream: torch.cuda.Stream | None,
        *,
        copy_fields: tuple[LayoutFieldId, ...] | None = None,
    ) -> PageWrite:
        demand: dict[tuple[str, LayoutFieldId], int] = {}
        for resource, chosen in zip(resources, fields, strict=True):
            for field in chosen:
                key = (resource, field)
                demand[key] = demand.get(key, 0) + 1
        if not sources:
            mutable = {
                effect.resource
                for task in self.plan.semantic_graph.tasks
                for effect in task.effects
                if effect.kind not in (StateEffectKind.READ, StateEffectKind.PUBLISH)
            }
            for resource in set(resources) & mutable:
                # Borrow original owners, including prepared admissions and existing private writers.
                headroom = resources.count(resource) + sum(
                    ref is not None and self.root(ref).resource == resource for ref in self._lineages.values()
                )
                for token, write in self._writes.items():
                    headroom += sum(root.resource == resource for root in write.targets) * (
                        -1 if token.before_refs else 1
                    )
                for field in self._pools[resource].layout.fields:
                    demand[resource, field.field_id] += headroom
        exhausted = {
            resource for (resource, field), count in demand.items() if self._pools[resource].free_count(field) < count
        }
        if exhausted:
            # Only completed owners can return pages; active pins are never evicted.
            self.collect_completed(wait=True, resources=exhausted)
        if any(self._pools[resource].free_count(field) < count for (resource, field), count in demand.items()):
            if not sources:
                self._admission_capacity_failures += 1
            raise RuntimeError("PageBundle capacity is exhausted by live owners")
        allocated = self._allocate_versions(sources, targets, resources, fields)
        token = PageWrite(self._next_operation, sources, targets, stream)
        self._next_operation += 1
        self._writes[token] = _Write(
            tuple(allocated),
            fields,
            {
                root.ref: (root, frozenset(int(field) for field in chosen))
                for root, chosen in zip(allocated, fields, strict=True)
            },
        )
        self._writing.update(ref.lineage_id for ref in sources)
        for ref in sources:
            self._roots[ref].pins += 1
        try:
            with self._scope(stream), torch.no_grad():
                for event in {id(root.readiness): root.readiness for root in map(self.root, sources)}.values():
                    self._order(event, stream)
                self._fill_versions(sources, allocated, fields, copy_fields)
        except Exception:
            self.abort(token)
            raise
        return token

    def _allocate_versions(
        self,
        sources: tuple[StateRef, ...],
        targets: tuple[StateRef, ...],
        resources: tuple[str, ...],
        fields: tuple[tuple[LayoutFieldId, ...], ...],
    ) -> list[PageBundle]:
        """Pages for new versions: fresh pages for written fields, shared pages for the rest.

        An admission (no sources) writes every field. Capacity was checked by the caller.
        """
        allocated: list[PageBundle | None] = [None] * len(targets)
        for resource in dict.fromkeys(resources):
            pools = self._pools[resource]
            group = [index for index, name in enumerate(resources) if name == resource]
            width = len(pools.tensors)
            written = np.ones((len(group), width), dtype=np.bool_)
            slots = np.zeros((len(group), width), dtype=np.int64)
            generations = np.zeros_like(slots)
            if sources:
                origins = [self.root(sources[index]) for index in group]
                slots[:] = [root.slots for root in origins]
                generations[:] = [root.generations for root in origins]
                written[:] = False
                for row, index in enumerate(group):
                    written[row, list(fields[index])] = True
            for field in range(width):
                rows = written[:, field]
                if rows.any():
                    slots[rows, field], generations[rows, field] = pools.allocate(field, int(rows.sum()))
                if not rows.all():
                    pools.share(field, slots[~rows, field])
            for row, index in enumerate(group):
                allocated[index] = PageBundle(targets[index], resource, slots[row], generations[row], None)
        return [root for root in allocated if root is not None]

    def _fill_versions(
        self,
        sources: tuple[StateRef, ...],
        targets: list[PageBundle],
        fields: tuple[tuple[LayoutFieldId, ...], ...],
        copy_fields: tuple[LayoutFieldId, ...] | None,
    ) -> None:
        """Zero an admission's pages, or copy a transition's written fields from the source version.

        One batched copy per (resource, field), however many states the write covers.
        """
        if sources and copy_fields == ():
            return
        for resource in dict.fromkeys(root.resource for root in targets):
            pools = self._pools[resource]
            group = [index for index, root in enumerate(targets) if root.resource == resource]
            width = len(pools.tensors)
            destination: Slots = np.stack([targets[index].slots for index in group])
            origin: Slots = np.stack([self.root(sources[index]).slots for index in group]) if sources else destination
            written: npt.NDArray[np.bool_] = np.zeros((len(group), width), dtype=np.bool_)
            for row, index in enumerate(group):
                written[row, list(fields[index])] = True
            if sources and copy_fields is not None:
                written[:, [field for field in range(width) if field not in copy_fields]] = False
            for field, pool in enumerate(pools.tensors):
                rows = written[:, field]
                if not rows.any():
                    continue
                if not sources:
                    pool.index_fill_(0, upload(destination[rows, field].tolist(), self.device), 0)
                    continue
                pairs = upload([origin[rows, field].tolist(), destination[rows, field].tolist()], self.device)
                pool.index_copy_(0, pairs[1], pool.index_select(0, pairs[0]))

    def write_field(self, token: PageWrite, root_index: int, field: LayoutFieldId) -> torch.Tensor:
        with self._lock:
            (root,) = self._capture_roots(token, (token.after_refs[root_index],), (field,))
            return self._pools[root.resource].tensor(field, root.pages[field])

    def writable_roots(self, token: PageWrite) -> tuple[tuple[torch.Tensor, ...], ...]:
        with self._lock:
            self._issued_write(token)
            return self.borrow_roots((token,) * len(token.after_refs), token.after_refs)

    def publish(self, token: PageWrite) -> PagePublication:
        with self._lock:
            write = self._issued_write(token)
            for source in token.before_refs:
                if self.latest(source.lineage_id) != source:
                    raise RuntimeError("PageBundle publication source is no longer latest")
            completion = self._record(token.stream)
            receipt = PagePublication(token.operation_id, token.before_refs, token.after_refs, completion)
            for target in write.targets:
                ref = target.ref
                self._roots[ref] = _Root(
                    PageBundle(ref, target.resource, target.slots, target.generations, completion), pins=1
                )
                self._lineages[ref.lineage_id] = ref
                self._root_count[ref.lineage_id] += 1
            self._stale.update(token.before_refs)
            self._writing.difference_update(ref.lineage_id for ref in token.before_refs)
            self._writes.pop(token)
            self._publications[receipt] = False
            return receipt

    def settle(self, receipt: PagePublication) -> None:
        with self._lock:
            self.collect_completed()
            if self._publications.get(receipt) is not False:
                raise RuntimeError("PageBundle publication is not an issued receipt or is already settled")
            self._publications[receipt] = True

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
            for root in write.targets:
                self._pools[root.resource].release(root.slots, root.generations)
            for ref in token.before_refs:
                self._roots[ref].pins -= 1
            self._writing.difference_update(ref.lineage_id for ref in token.before_refs)
            self._writes.pop(token)
            self._reclaim()

    def pin_exact(self, refs: tuple[StateRef, ...], stream: torch.cuda.Stream | None = None) -> PagePin:
        with self._lock:
            self._require_stream(stream)
            if (not refs and self._pools) or len(refs) != len(set(refs)):
                raise ValueError("PageBundle pin requires distinct exact roots")
            roots = tuple(self.root(ref) for ref in refs)
            for root in roots:
                if self._lineages[root.ref.lineage_id] is None:
                    raise RuntimeError("PageBundle lineage is retiring; no new pins may start")
                self._order(root.readiness, stream)
            token = PagePin(self._next_pin, refs, stream)
            self._next_pin += 1
            for ref in refs:
                self._roots[ref].pins += 1
            self._pins[token] = _Pin(dict(zip(refs, roots, strict=True)))
            return token

    def borrow_field(self, pin: PagePin, ref: StateRef, field: LayoutFieldId) -> torch.Tensor:
        """Read-only page view; use on pin.stream and discard before releasing pin.

        Native consumers must not mutate this alias. Private writes use write_field;
        owned exports use snapshot. The original pin owns completion and reclamation.
        """
        with self._lock:
            (root,) = self._capture_roots(pin, (ref,), (field,))
            return self._pools[root.resource].tensor(field, root.pages[field]).detach()

    def _held_roots(
        self, access: tuple[PagePin | PageWrite, ...], refs: tuple[StateRef, ...]
    ) -> tuple[tuple[PageBundle, frozenset[int] | None], ...]:
        """Each ref's root under the pin or write that grants it, with the write's writable fields.

        Every distinct token is validated once, so a batch over many states costs O(states).
        Pinned roots carry None: all their fields are readable.
        """
        result: list[tuple[PageBundle, frozenset[int] | None]] = []
        for token, ref in zip(access, refs, strict=True):
            match token:
                case PagePin():
                    root = self._issued_pin(token).roots.get(ref)
                    if root is None:
                        raise RuntimeError("PageBundle root is not held by this pin")
                    result.append((root, None))
                case PageWrite():
                    entry = self._issued_write(token).by_ref.get(ref)
                    if entry is None:
                        raise RuntimeError("PageBundle root is not written by this transaction")
                    result.append(entry)
        return tuple(result)

    def lazy_roots(self, access: tuple[PagePin | PageWrite, ...], refs: tuple[StateRef, ...]) -> "LazyRoots":
        """Like borrow_roots, but builds each state's field views only when it is read."""
        with self._lock:
            self._require_live()
            return LazyRoots(self, self._held_roots(access, refs))

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
        self, access: tuple[PagePin | PageWrite, ...], refs: tuple[StateRef, ...]
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Every field of every ref, read under its pin (read-only views) or write (writable)."""
        with self._lock:
            self._require_live()
            return tuple(self._root_views(root, writable) for root, writable in self._held_roots(access, refs))

    def field_slots(
        self, access: tuple[PagePin | PageWrite, ...], refs: tuple[StateRef, ...], field: LayoutFieldId
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        """A field's whole page pool `[capacity, *shape]` and the slot of each ref's page in it.

        Each ref is checked against the pin or write that grants access to it, so indexing the
        pool at the returned slots reads exactly those versions.
        """
        with self._lock:
            self._require_live()
            return self._slots_of(self._held_roots(access, refs), field)

    def _slots_of(
        self, roots: tuple[tuple[PageBundle, frozenset[int] | None], ...], field: LayoutFieldId
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        with self._lock:
            self._require_live()
            if any(writable is not None and int(field) not in writable for _, writable in roots):
                raise RuntimeError("PageBundle field is not writable in this transaction")
            if len({root.resource for root, _ in roots}) != 1:
                raise ValueError("field slots require one resource layout")
            pool = self._pools[roots[0][0].resource].tensors[field].detach()
            return pool, tuple(int(root.slots[field]) for root, _ in roots)

    def paged_rows(
        self,
        access: tuple[PagePin | PageWrite, ...],
        refs: tuple[StateRef, ...],
        fields: tuple[LayoutFieldId, ...],
        rows: tuple[int, ...],
    ) -> tuple[PagedRows, ...]:
        """Derive immutable row mappings from issued exact-version capabilities.

        No mapping is reused or overwritten by a later invocation. Allocations
        and consumption use the original stream and its release/publication fence.
        """
        with self._lock:
            if not refs or not fields or len(access) != len(refs) or any(i < 0 or i >= len(refs) for i in rows):
                raise ValueError("paged rows require original state indices and access")
            roots = tuple(
                self._capture_roots(token, (ref,), fields)[0] for token, ref in zip(access, refs, strict=True)
            )
            stream = access[0].stream
            if (
                self.device.type != "cuda"
                or stream is None
                or any(token.stream != stream for token in access)
                or len({root.resource for root in roots}) != 1
            ):
                raise ValueError("paged rows require one CUDA resource and consumer stream")

            def validate() -> None:
                with self._lock:
                    if torch.cuda.current_stream(self.device) != stream:
                        raise RuntimeError("paged computation must use its original consumer stream")
                    for token, ref in zip(access, refs, strict=True):
                        self._capture_roots(token, (ref,), fields)

            def validate_write() -> None:
                validate()
                if not any(isinstance(token, PageWrite) for token in access):
                    raise RuntimeError("paged writes require an original private PageWrite")

            validate()
            order = tuple(sorted(range(len(rows)), key=rows.__getitem__))
            counts = [0] * len(refs)
            for state in rows:
                counts[state] += 1

            def index(values: tuple[int, ...]) -> torch.Tensor:
                return upload(values, self.device)

            row_order, row_states = index(order), index(rows)
            offsets = upload(tuple(accumulate(counts)), self.device, torch.int32)
            active = torch.ones(len(rows), device=self.device, dtype=torch.bool)
            pools = self._pools[roots[0].resource]
            slots = torch.tensor(
                tuple(tuple(int(root.slots[field]) for root in roots) for field in fields),
                device=self.device,
                dtype=torch.long,
            )
            write_slots = (
                slots
                if all(isinstance(token, PageWrite) for token in access)
                else torch.tensor(
                    tuple(
                        tuple(
                            int(root.slots[field]) if isinstance(token, PageWrite) else -1
                            for root, token in zip(roots, access, strict=True)
                        )
                        for field in fields
                    ),
                    device=self.device,
                    dtype=torch.long,
                )
            )
            return tuple(
                PagedRows(
                    pools.tensors[field].reshape(pools.tensors[field].shape[0], -1).detach(),
                    slots[index],
                    row_order,
                    offsets,
                    row_states,
                    validate,
                    active,
                    write_slots[index],
                    validate_write=validate_write,
                )
                for index, field in enumerate(fields)
            )

    def capture_fields(
        self,
        access: PagePin | PageWrite,
        refs: tuple[StateRef, ...],
        fields: tuple[LayoutFieldId, ...],
        indices: torch.Tensor,
        buffers: tuple[torch.Tensor, ...],
        graph: torch.cuda.CUDAGraph,
        *,
        paged: bool = False,
    ) -> CapturedPageFields:
        """Bind cold scratch to a signed pool. Budget scratch as AER workspace.

        Capture read_captured/write_captured on access.stream; select_captured and
        replay_captured gate subsequent launches with the original pin/write.
        Queue replay before releasing its pin or publishing/aborting its write.
        The bound graph may access only these fields and independent workspace;
        retain the arena and scratch until the original completion has finished.
        """
        with self._lock:
            if access.stream is None or not fields or fields != tuple(sorted(set(fields))):
                raise ValueError("captured fields require a CUDA stream and distinct ordered fields")
            root = self._capture_roots(access, refs, fields)[0]
            if not self._pools[root.resource].selected.capture_stable:
                raise ValueError("signed provider does not support captured page selection")
            view = CapturedPageFields(
                self,
                root.resource,
                fields,
                indices,
                buffers,
                tuple(value.data_ptr() for value in (indices, *buffers)),
                access.stream,
                graph,
                access,
                refs,
                paged=paged,
            )
            self.select_captured(view, access, refs)
            return view

    def capture_kernel(
        self,
        access: PagePin | PageWrite,
        refs: tuple[StateRef, ...],
        fields: tuple[LayoutFieldId, ...],
        inputs: tuple[torch.Tensor, ...],
        kernel: Callable[[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]] | PagedKernel,
        *,
        preserve_fields: bool = True,
    ) -> CapturedPageFields:
        """Cold capture of explicit tensor mathematics; no kernel selection or fallback.

        Fields have a fixed leading capacity axis, with current refs in prefix
        order and zero-filled inactive rows. Inputs and result
        buffers stay fixed across replays. Private fields are restored after
        warmup unless the caller supplies disposable admission pages and disables
        preservation. The caller must abort those pages after capture.
        The callable is consumed here, never stored in a runtime or wire contract.
        """
        if access.stream is None:
            raise ValueError("kernel capture requires its original CUDA stream")
        with self._lock, torch.cuda.stream(access.stream), torch.inference_mode(False), torch.enable_grad():
            root = self._capture_roots(access, refs, fields)[0]
            if not preserve_fields and (not isinstance(access, PageWrite) or access.before_refs):
                raise ValueError("disposable capture requires private admission pages")
            specs = self._pools[root.resource].layout.fields
            indices = torch.empty((len(fields), len(refs)), device=self.device, dtype=torch.int64)
            buffers = (
                ()
                if isinstance(kernel, PagedKernel)
                else tuple(
                    torch.empty(
                        (len(refs), *specs[field].shape), device=self.device, dtype=tensor_dtype(specs[field].dtype)
                    )
                    for field in fields
                )
            )
            view = self.capture_fields(
                access, refs, fields, indices, buffers, torch.cuda.CUDAGraph(), paged=isinstance(kernel, PagedKernel)
            )
            view.inputs = inputs

            def compute() -> tuple[torch.Tensor, ...]:
                if isinstance(kernel, PagedKernel):
                    return kernel.compute(view, inputs)
                return kernel(self.read_captured(view), inputs)

            originals = (
                tuple(
                    (value, value.clone())
                    for ref in refs
                    for field in fields
                    for value in (self.write_field(access, access.after_refs.index(ref), field),)
                )
                if preserve_fields and isinstance(access, PageWrite) and view.paged
                else ()
            )
            try:
                for _ in range(3):
                    compute()
                access.stream.synchronize()
                with torch.cuda.graph(view.graph, stream=access.stream):
                    view.results = compute()
                    if isinstance(access, PageWrite) and not view.paged:
                        self.write_captured(view)
            finally:
                with torch.no_grad():
                    for destination, original in originals:
                        destination.copy_(original)
            return view

    def _capture_roots(
        self, access: PagePin | PageWrite, refs: tuple[StateRef, ...], fields: tuple[LayoutFieldId, ...]
    ) -> tuple[PageBundle, ...]:
        if not refs or len(set(refs)) != len(refs):
            raise ValueError("captured selection requires distinct nonempty original refs")
        match access:
            case PagePin():
                held = self._issued_pin(access).roots
                if any(ref not in held for ref in refs):
                    raise RuntimeError("PageBundle root is not held by this pin")
                roots = tuple(held[ref] for ref in refs)
            case PageWrite():
                write = self._issued_write(access)
                positions = {ref: index for index, ref in enumerate(access.after_refs)}
                if any(ref not in positions for ref in refs):
                    raise RuntimeError("PageBundle root is not written by this transaction")
                indices = tuple(positions[ref] for ref in refs)
                for index in indices:
                    for field in fields:
                        if field not in write.fields[index]:
                            raise RuntimeError("PageBundle field is not writable in this transaction")
                roots = tuple(write.targets[index] for index in indices)
        if len({root.resource for root in roots}) != 1:
            raise ValueError("captured selection must use one original resource layout")
        if any(not 0 <= field < len(roots[0].slots) for field in fields):
            raise ValueError("PageBundle field is outside the signed layout")
        return roots

    def _captured_pool(self, view: CapturedPageFields) -> _Pools:
        if (
            view.arena is not self
            or view.indices.device != self.device
            or view.indices.dtype != torch.int64
            or view.indices.ndim != 2
            or view.indices.shape[0] != len(view.fields)
            or not 0 < len(view.refs) <= view.indices.shape[1]
            or not view.indices.is_contiguous()
            or len(view.buffers) != (0 if view.paged else len(view.fields))
            or tuple(value.data_ptr() for value in (view.indices, *view.buffers)) != view.addresses
        ):
            raise ValueError("captured page scratch address or layout changed")
        pools = self._pools[view.resource]
        if any(pools.tensors[field].data_ptr() != pools.addresses[field] for field in view.fields):
            raise ValueError("captured pool address changed")
        ranges = tuple(
            (value.data_ptr(), value.data_ptr() + value.numel() * value.element_size())
            for value in (view.indices, *view.buffers)
        )
        bank_ranges = tuple(
            (value.data_ptr(), value.data_ptr() + value.shape[0] * value.stride(0) * value.element_size())
            for value in pools.tensors
        )
        if any(
            max(start, other_start) < min(end, other_end)
            for index, (start, end) in enumerate(ranges)
            for other_start, other_end in (*ranges[index + 1 :], *bank_ranges)
        ):
            raise ValueError("captured scratch must not alias another buffer or arena page")
        for field, buffer in zip(() if view.paged else view.fields, view.buffers, strict=True):
            spec = pools.layout.fields[field]
            if (
                pools.tensors[field].data_ptr() != pools.addresses[field]
                or buffer.shape != (view.indices.shape[1], *spec.shape)
                or buffer.dtype != tensor_dtype(spec.dtype)
                or buffer.device != self.device
                or not buffer.is_contiguous()
                or buffer.requires_grad
            ):
                raise ValueError("captured field scratch differs from the signed layout")
        return pools

    def select_captured(
        self, view: CapturedPageFields, access: PagePin | PageWrite, refs: tuple[StateRef, ...]
    ) -> None:
        with self._lock, torch.no_grad():
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("page selection must happen outside graph capture")
            self._captured_pool(view)
            roots = self._capture_roots(access, refs, view.fields)
            if (
                access.stream is None
                or roots[0].resource != view.resource
                or type(access) is not type(view.access)
                or len(refs) > view.indices.shape[1]
            ):
                raise ValueError("captured selection changed its original stream, resource or access mode")
            if access.stream != view.stream:
                access.stream.wait_stream(view.stream)
            view.stream = access.stream
            with torch.cuda.stream(view.stream):
                view.indices.fill_(-1)
                for index, field in enumerate(view.fields):
                    for row, root in enumerate(roots):
                        view.indices[index, row].fill_(int(root.slots[field]))
            view.access, view.refs = access, refs

    def read_captured(self, view: CapturedPageFields) -> tuple[torch.Tensor, ...]:
        with self._lock:
            if view.paged:
                raise ValueError("paged captures do not own gathered field buffers")
            pools = self._captured_pool(view)
            self._capture_roots(view.access, view.refs, view.fields)
            if torch.cuda.current_stream(self.device) != view.stream:
                raise RuntimeError("capture must use the original consumer stream")
            for index, (field, output) in enumerate(zip(view.fields, view.buffers, strict=True)):
                slots = view.indices[index]
                torch.index_select(pools.tensors[field], 0, torch.where(slots >= 0, slots, slots[0]), out=output)
                output.masked_fill_((slots < 0).reshape(-1, *(1 for _ in output.shape[1:])), 0)
            return view.buffers

    def read_paged_captured(self, view: CapturedPageFields, rows: torch.Tensor) -> tuple[PagedRows, ...]:
        with self._lock:
            pools = self._captured_pool(view)
            if not view.paged or rows.ndim != 1 or rows.dtype != torch.long or rows.device != self.device:
                raise ValueError("paged capture requires its signed device row indices")

            def validate() -> None:
                with self._lock:
                    self._capture_roots(view.access, view.refs, view.fields)
                    if torch.cuda.current_stream(self.device) != view.stream:
                        raise RuntimeError("paged capture must use the original consumer stream")

            validate()

            def validate_write() -> None:
                validate()
                if not isinstance(view.access, PageWrite):
                    raise RuntimeError("paged writes require an original private PageWrite")

            return PagedRows.from_fields(
                tuple(
                    pools.tensors[field].reshape(pools.tensors[field].shape[0], -1).detach() for field in view.fields
                ),
                view.indices,
                rows,
                validate,
                validate_write,
            )

    def adamw_captured(self, view: CapturedPageFields, gradients: torch.Tensor, config: AdamWConfig) -> None:
        with self._lock:
            if not isinstance(view.access, PageWrite) or not view.paged:
                raise RuntimeError("paged AdamW requires an original private PageWrite")
            pools = self._captured_pool(view)
            self._capture_roots(view.access, view.refs, view.fields)
            if torch.cuda.current_stream(self.device) != view.stream:
                raise RuntimeError("paged AdamW must use the original writer stream")
            paged_adamw(tuple(pools.tensors[field] for field in view.fields), view.indices, gradients, config)

    def write_captured(self, view: CapturedPageFields) -> None:
        with self._lock:
            if not isinstance(view.access, PageWrite) or view.paged:
                raise RuntimeError("captured writes require an original private PageWrite")
            pools = self._captured_pool(view)
            self._capture_roots(view.access, view.refs, view.fields)
            if torch.cuda.current_stream(self.device) != view.stream:
                raise RuntimeError("capture must use the original writer stream")
            for index, (field, value) in enumerate(zip(view.fields, view.buffers, strict=True)):
                slots = view.indices[index]
                shape = (-1, *(1 for _ in value.shape[1:]))
                destination = pools.tensors[field]
                if value.dtype is torch.bool:
                    destination, value = destination.view(torch.uint8), value.view(torch.uint8)
                neutral = float("-inf") if value.is_floating_point() else torch.iinfo(value.dtype).min
                # Padding reduces into the first private row with the neutral value;
                # each real slot has exactly one writer, including NaN/Inf values.
                value.masked_fill_((slots < 0).reshape(shape), neutral)
                destination.scatter_reduce_(
                    0,
                    torch.where(slots >= 0, slots, slots[0]).reshape(shape).expand_as(value),
                    value,
                    reduce="amax",
                    include_self=False,
                )

    def replay_captured(self, view: CapturedPageFields) -> None:
        with self._lock, torch.cuda.stream(view.stream):
            self._captured_pool(view)
            self._capture_roots(view.access, view.refs, view.fields)
            view.graph.replay()

    def send_field(
        self, pin: PagePin, ref: StateRef, field: LayoutFieldId, connection: Connection, peer_pid: int
    ) -> PageFieldTransfer:
        """Export over a dedicated channel; the original pin owns remote completion."""
        with self._lock:
            value = self.borrow_field(pin, ref, field)
            root = self.root(ref)
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
            self._issued_pin(pin)
            if ref not in pin.refs:
                raise RuntimeError("PageBundle root is not held by this pin")
            root = self.root(ref)
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
            self._releases.add(receipt)
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

    def retire(self, refs: tuple[StateRef, ...]) -> None:
        with self._lock:
            self._require_live()
            if not refs or len(set(refs)) != len(refs):
                raise ValueError("PageBundle retirement requires distinct latest roots")
            for ref in refs:
                if self.latest(ref.lineage_id) != ref or ref.lineage_id in self._writing:
                    raise RuntimeError("PageBundle retirement overlaps a write or stale version")
            for ref in refs:
                self._lineages[ref.lineage_id] = None
                self._retiring.add(ref.lineage_id)
            self._stale.update(refs)
            self.collect_completed()

    def collect_completed(self, *, wait: bool = False, resources: set[str] | None = None) -> None:
        with self._lock:
            self._require_live()
            if wait:
                for receipt, settled in self._publications.items():
                    if (
                        settled
                        and receipt.completion is not None
                        and (
                            resources is None
                            or any(
                                self.root(ref).resource in resources
                                for ref in (*receipt.before_refs, *receipt.after_refs)
                            )
                        )
                    ):
                        receipt.completion.synchronize()
                for release in self._releases:
                    if resources is not None and (
                        not any(self.root(ref).resource in resources for ref in release.pin.refs)
                        or any(not read.query() for read in release.remote_reads)
                    ):
                        continue
                    if release.completion is not None:
                        release.completion.synchronize()
                    for read in release.remote_reads:
                        read.synchronize()
            for receipt, settled in tuple(self._publications.items()):
                if settled and (receipt.completion is None or receipt.completion.query()):
                    for ref in (*receipt.before_refs, *receipt.after_refs):
                        self._roots[ref].pins -= 1
                    self._publications.pop(receipt)
            for receipt in tuple(self._releases):
                if (receipt.completion is None or receipt.completion.query()) and all(
                    read.query() for read in receipt.remote_reads
                ):
                    for ref in receipt.pin.refs:
                        self._roots[ref].pins -= 1
                    self._releases.remove(receipt)
            self._reclaim()

    def _reclaim(self) -> None:
        """Release stale roots that no pin holds and whose last GPU writer has finished."""
        released: dict[str, list[PageBundle]] = {}
        for ref in tuple(self._stale):
            stored = self._roots[ref]
            bundle = stored.bundle
            if stored.pins or (bundle.readiness is not None and not bundle.readiness.query()):
                continue
            released.setdefault(bundle.resource, []).append(bundle)
            del self._roots[ref]
            self._stale.discard(ref)
            self._root_count[ref.lineage_id] -= 1
            if not self._root_count[ref.lineage_id]:
                del self._root_count[ref.lineage_id]
                if ref.lineage_id in self._retiring:
                    self._retiring.discard(ref.lineage_id)
                    del self._lineages[ref.lineage_id]
        for resource, bundles in released.items():
            self._pools[resource].release(
                np.stack([bundle.slots for bundle in bundles]), np.stack([bundle.generations for bundle in bundles])
            )

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
