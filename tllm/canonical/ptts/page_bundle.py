"""One version, generation and completion owner for signed tensor PageBundles."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from math import prod
from threading import RLock
from weakref import WeakValueDictionary

import torch

from tllm.canonical.ptts.page_bundle_planning import DTYPE_BYTES, page_stride_bytes
from tllm.canonical.scheduler import validate_plan_bundle
from tllm.canonical.state_representation import validate_state_representation_reconstruction
from tllm.contracts.canonical_compilation import CanonicalPlanBundle
from tllm.contracts.state_ref import LineageId, StateRef, VersionId
from tllm.contracts.state_representation import StateDeviceKind, StateProviderPlan, StateRepresentation
from tllm.contracts.tensor_layout import LayoutFieldId, TensorDType, TensorLayoutSpec


def tensor_dtype(dtype: TensorDType) -> torch.dtype:
    return {
        TensorDType.BOOL: torch.bool, TensorDType.UINT8: torch.uint8,
        TensorDType.INT32: torch.int32, TensorDType.INT64: torch.int64,
        TensorDType.FLOAT16: torch.float16, TensorDType.BFLOAT16: torch.bfloat16,
        TensorDType.FLOAT32: torch.float32, TensorDType.FLOAT64: torch.float64,
    }[dtype]


@dataclass(frozen=True, slots=True)
class PageSlot:
    slot_id: int
    generation: int


@dataclass(frozen=True, slots=True)
class PageBundle:
    ref: StateRef
    resource: str
    pages: tuple[PageSlot, ...]
    readiness: torch.cuda.Event | None


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


@dataclass(slots=True)
class _Root:
    bundle: PageBundle
    pins: int = 0


@dataclass(frozen=True, slots=True)
class _Lineage:
    resource: str
    latest: StateRef | None


@dataclass(frozen=True, slots=True)
class _Write:
    token: PageWrite
    targets: tuple[PageBundle, ...]
    fields: tuple[tuple[LayoutFieldId, ...], ...]


@dataclass(slots=True)
class _Publication:
    receipt: PagePublication
    settled: bool = False


class _Pools:
    def __init__(self, layout: TensorLayoutSpec, selected: StateProviderPlan, device: torch.device) -> None:
        self.layout, self.selected = layout, selected
        capacity = selected.max_admitted_lineages * selected.max_live_versions_per_lineage
        tensors: list[torch.Tensor] = []
        allocated_bytes = 0
        for field in layout.fields:
            size = DTYPE_BYTES[field.dtype]
            stride = page_stride_bytes(field) // size
            alignment = max(field.alignment_bytes // size, 1)
            pool = torch.zeros(capacity * stride + alignment - 1, dtype=tensor_dtype(field.dtype), device=device)
            offset = (-pool.data_ptr() // size) % alignment
            tensors.append(pool[offset:offset + capacity * stride].view(capacity, stride)
                [:, :prod(field.shape)].view(capacity, *field.shape))
            allocated_bytes += pool.numel() * size
        self.tensors = tuple(tensors)
        self.addresses = tuple(pool.data_ptr() for pool in self.tensors)
        self.free = [list(range(capacity)) for _ in layout.fields]
        self.generations = [[0] * capacity for _ in layout.fields]
        self.users = [[0] * capacity for _ in layout.fields]
        if allocated_bytes != selected.reserved_bytes:
            raise ValueError("PageBundle pool allocation differs from signed capacity")

    def allocate(self, field: int) -> PageSlot:
        if not self.free[field]:
            raise RuntimeError("PageBundle capacity is exhausted")
        slot = self.free[field].pop(0)
        self.users[field][slot] = 1
        return PageSlot(slot, self.generations[field][slot])

    def tensor(self, field: int, page: PageSlot) -> torch.Tensor:
        if self.generations[field][page.slot_id] != page.generation or self.users[field][page.slot_id] < 1:
            raise RuntimeError("PageBundle allocation generation was reclaimed")
        return self.tensors[field][page.slot_id]

    def release(self, pages: tuple[PageSlot, ...]) -> None:
        for field, page in enumerate(pages):
            self.tensor(field, page)
            self.users[field][page.slot_id] -= 1
            if self.users[field][page.slot_id] == 0:
                self.generations[field][page.slot_id] += 1
                self.free[field].append(page.slot_id)
                self.free[field].sort()


class PageBundleArena:
    """CPU/CUDA share admission, COW and atomic multi-root publication.

    Tensor fields are physical pages: unchanged fields share slots, writable
    fields receive private slots. Only this arena chooses logical versions.
    """

    def __init__(self, plan: CanonicalPlanBundle,
                 local_candidates: tuple[StateProviderPlan, ...] | None = None) -> None:
        validate_plan_bundle(plan)
        validate_state_representation_reconstruction(plan.state_request, plan.state_decision,
            plan.state_request.candidates if local_candidates is None else local_candidates)
        self.plan = plan
        placements = {(item.device_kind, item.device_ordinal) for item in plan.state_physical.representations}
        if len(placements) != 1:
            raise ValueError("PageBundle arena requires one signed device placement")
        kind, ordinal = placements.pop()
        self.device = torch.device("cpu") if kind is StateDeviceKind.CPU else torch.device("cuda", ordinal)
        self._pools: dict[str, _Pools] = {}
        layouts = {item.resource: item.layout for item in plan.tensor_layout_request.layouts}
        for selected in plan.state_physical.representations:
            if selected.representation is not StateRepresentation.DENSE_PAGE_ARENA:
                raise ValueError("PageBundle arena cannot reconstruct the signed representation")
            self._pools[selected.resource] = _Pools(layouts[selected.resource], selected, self.device)
        self._roots: dict[StateRef, _Root] = {}
        self._lineages: dict[LineageId, _Lineage] = {}
        self._writes: dict[int, _Write] = {}
        self._pins: dict[int, PagePin] = {}
        self._publications: dict[int, _Publication] = {}
        self._releases: dict[int, PageRelease] = {}
        self._issued_releases: WeakValueDictionary[int, PageRelease] = WeakValueDictionary()
        self._next_lineage = 1
        self._next_operation = 1
        self._next_pin = 1
        self._next_release = 1
        self._poisoned = False
        self._lock = RLock()
        self._initialization = self._record(torch.cuda.current_stream(self.device)) if self.device.type == "cuda" else None

    @property
    def active_lineages(self) -> int:
        with self._lock:
            self.collect_completed()
            return len(self._lineages)

    @property
    def free_pages(self) -> int:
        with self._lock:
            self.collect_completed()
            return sum(min(map(len, pools.free)) for pools in self._pools.values())

    @property
    def pending_pin_releases(self) -> int:
        return len(self._releases)

    @property
    def pool_addresses(self) -> tuple[int, ...]:
        return tuple(address for pools in self._pools.values() for address in pools.addresses)

    def require_stable(self) -> None:
        for pools in self._pools.values():
            for field, pool, address in zip(pools.layout.fields, pools.tensors, pools.addresses, strict=True):
                if pool.data_ptr() != address or pool.shape[1:] != field.shape or pool.dtype != tensor_dtype(field.dtype):
                    raise RuntimeError("PageBundle capture address or layout changed")

    def latest(self, lineage: LineageId) -> StateRef:
        with self._lock:
            self._require_live()
            state = self._lineages.get(lineage)
            if state is None or state.latest is None:
                raise RuntimeError("PageBundle lineage is unadmitted or retiring")
            return state.latest

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
                live = sum(lineage.resource == resource for lineage in self._lineages.values())
                pending = sum(root.resource == resource for write in self._writes.values()
                    if not write.token.before_refs for root in write.targets)
                if live + pending + resources.count(resource) > pools.selected.max_admitted_lineages:
                    raise RuntimeError("PageBundle admission capacity is exhausted")
            refs = tuple(StateRef(self._pools[resource].selected.layout_id,
                LineageId(self._next_lineage + index), VersionId(0)) for index, resource in enumerate(resources))
            self._next_lineage += len(refs)
            fields = tuple(tuple(field.field_id for field in self._pools[resource].layout.fields) for resource in resources)
            return self._prepare((), refs, resources, fields, stream)

    def prepare_transition(self, sources: tuple[StateRef, ...], stream: torch.cuda.Stream | None = None,
                           *, fields: tuple[tuple[LayoutFieldId, ...], ...] | None = None) -> PageWrite:
        with self._lock:
            self._require_stream(stream)
            self.collect_completed()
            if not sources or len({ref.lineage_id for ref in sources}) != len(sources):
                raise ValueError("PageBundle transition requires distinct lineages")
            resources = tuple(self.root(ref).resource for ref in sources)
            for ref, resource in zip(sources, resources, strict=True):
                if self.latest(ref.lineage_id) != ref:
                    raise RuntimeError("PageBundle transition source is not latest")
                if any(ref in write.token.before_refs for write in self._writes.values()):
                    raise RuntimeError("PageBundle lineage already has a pending write")
                versions = sum(known.lineage_id == ref.lineage_id for known in self._roots)
                if versions >= self._pools[resource].selected.max_live_versions_per_lineage:
                    raise RuntimeError("PageBundle version capacity is exhausted")
            fields = fields if fields is not None else tuple(
                tuple(field.field_id for field in self._pools[resource].layout.fields) for resource in resources)
            if len(fields) != len(sources) or any(not chosen or chosen != tuple(sorted(set(chosen)))
                or any(field < 0 or field >= len(self._pools[resource].layout.fields) for field in chosen)
                for chosen, resource in zip(fields, resources, strict=True)):
                raise ValueError("PageBundle writable fields differ from the signed layout")
            targets = tuple(StateRef(ref.layout_id, ref.lineage_id, VersionId(ref.version_id + 1)) for ref in sources)
            return self._prepare(sources, targets, resources, fields, stream)

    def _prepare(self, sources: tuple[StateRef, ...], targets: tuple[StateRef, ...],
                 resources: tuple[str, ...], fields: tuple[tuple[LayoutFieldId, ...], ...],
                 stream: torch.cuda.Stream | None) -> PageWrite:
        allocated: list[PageBundle] = []
        try:
            for index, (ref, resource, chosen) in enumerate(zip(targets, resources, fields, strict=True)):
                pools = self._pools[resource]
                source = self.root(sources[index]) if sources else None
                pages: list[PageSlot] = []
                try:
                    for field in range(len(pools.layout.fields)):
                        if source is not None and field not in chosen:
                            page = source.pages[field]
                            pools.users[field][page.slot_id] += 1
                        else:
                            page = pools.allocate(field)
                        pages.append(page)
                except Exception:
                    pools.release(tuple(pages))
                    raise
                allocated.append(PageBundle(ref, resource, tuple(pages), None))
        except Exception:
            for root in allocated:
                self._pools[root.resource].release(root.pages)
            raise
        token = PageWrite(self._next_operation, sources, targets, stream)
        self._next_operation += 1
        self._writes[token.operation_id] = _Write(token, tuple(allocated), fields)
        for ref in sources:
            self._roots[ref].pins += 1
        try:
            with self._scope(stream), torch.no_grad():
                for index, (target, chosen) in enumerate(zip(allocated, fields, strict=True)):
                    pools = self._pools[target.resource]
                    source = self.root(sources[index]) if sources else None
                    if source is not None:
                        self._order(source.readiness, stream)
                    for field in chosen:
                        tensor = pools.tensor(field, target.pages[field])
                        if source is None:
                            tensor.zero_()
                        else:
                            tensor.copy_(pools.tensor(field, source.pages[field]))
        except Exception:
            self.abort(token)
            raise
        return token

    def write_field(self, token: PageWrite, root_index: int, field: LayoutFieldId) -> torch.Tensor:
        with self._lock:
            write = self._issued_write(token)
            if field not in write.fields[root_index]:
                raise RuntimeError("PageBundle field is not writable in this transaction")
            root = write.targets[root_index]
            return self._pools[root.resource].tensor(field, root.pages[field])

    def writable_roots(self, token: PageWrite) -> tuple[tuple[torch.Tensor, ...], ...]:
        with self._lock:
            write = self._issued_write(token)
            return tuple(tuple(self.write_field(token, index, field.field_id)
                for field in self._pools[root.resource].layout.fields) for index, root in enumerate(write.targets))

    def copy_roots(self, token: PageWrite, values: tuple[tuple[torch.Tensor, ...], ...]) -> None:
        with self._lock:
            self._issued_write(token)
            targets = self.writable_roots(token)
            if len(values) != len(targets) or any(len(root) != len(target) or any(
                value.shape != expected.shape or value.dtype != expected.dtype or value.device != self.device
                for value, expected in zip(root, target)) for root, target in zip(values, targets)):
                raise ValueError("PageBundle input tensors differ from the signed layout")
            with self._scope(token.stream), torch.no_grad():
                for root, target in zip(values, targets, strict=True):
                    for value, destination in zip(root, target, strict=True):
                        destination.copy_(value)

    def publish(self, token: PageWrite) -> PagePublication:
        with self._lock:
            write = self._issued_write(token)
            for source in token.before_refs:
                if self.latest(source.lineage_id) != source:
                    raise RuntimeError("PageBundle publication source is no longer latest")
            completion = self._record(token.stream)
            receipt = PagePublication(token.operation_id, token.before_refs, token.after_refs, completion)
            roots = dict(self._roots)
            lineages = dict(self._lineages)
            for target in write.targets:
                bundle = PageBundle(target.ref, target.resource, target.pages, completion)
                roots[target.ref] = _Root(bundle, pins=1)
                lineages[target.ref.lineage_id] = _Lineage(target.resource, target.ref)
            # No tensor work or user code is allowed between the visibility swap and receipt issuance.
            self._roots, self._lineages = roots, lineages
            self._writes.pop(token.operation_id)
            self._publications[receipt.operation_id] = _Publication(receipt)
            return receipt

    def settle(self, receipt: PagePublication) -> None:
        with self._lock:
            issued = self._publications.get(receipt.operation_id)
            if issued is None or issued.settled:
                raise RuntimeError("PageBundle publication is unknown or already settled")
            if issued.receipt is not receipt:
                raise RuntimeError("PageBundle publication differs from the issued receipt")
            issued.settled = True
            self.collect_completed()

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
                self._pools[root.resource].release(root.pages)
            for ref in token.before_refs:
                self._roots[ref].pins -= 1
            self._writes.pop(token.operation_id)
            self._reclaim()

    def pin_exact(self, refs: tuple[StateRef, ...], stream: torch.cuda.Stream | None = None) -> PagePin:
        with self._lock:
            self._require_stream(stream)
            if not refs or len(refs) != len(set(refs)):
                raise ValueError("PageBundle pin requires distinct exact roots")
            roots = tuple(self.root(ref) for ref in refs)
            for root in roots:
                if self._lineages[root.ref.lineage_id].latest is None:
                    raise RuntimeError("PageBundle lineage is retiring; no new pins may start")
                self._order(root.readiness, stream)
            token = PagePin(self._next_pin, refs, stream)
            self._next_pin += 1
            for ref in refs:
                self._roots[ref].pins += 1
            self._pins[token.pin_id] = token
            return token

    def snapshot(self, pin: PagePin, ref: StateRef) -> tuple[torch.Tensor, ...]:
        with self._lock:
            self._issued_pin(pin)
            if ref not in pin.refs:
                raise RuntimeError("PageBundle root is not held by this pin")
            root = self.root(ref)
            pools = self._pools[root.resource]
            with self._scope(pin.stream):
                return tuple(pools.tensor(field, page).detach().clone() for field, page in enumerate(root.pages))

    def field_rows(self, pin: PagePin, field: LayoutFieldId) -> torch.Tensor:
        with self._lock:
            self._issued_pin(pin)
            roots = tuple(self.root(ref) for ref in pin.refs)
            if len({root.ref.layout_id for root in roots}) != 1:
                raise ValueError("PageBundle batched reads require one signed layout")
            with self._scope(pin.stream):
                return torch.stack(tuple(self._pools[root.resource].tensor(field, root.pages[field]) for root in roots))

    def release_pin(self, pin: PagePin) -> PageRelease:
        with self._lock:
            self._issued_pin(pin)
            receipt = PageRelease(self._next_release, pin, self._record(pin.stream))
            self._next_release += 1
            self._pins.pop(pin.pin_id)
            self._issued_releases[receipt.release_id] = receipt
            self._releases[receipt.release_id] = receipt
            return receipt

    def wait_release(self, receipt: PageRelease) -> None:
        with self._lock:
            self._issued_release(receipt)
            if receipt.completion is not None:
                receipt.completion.synchronize()
            self.collect_completed()

    def release_complete(self, receipt: PageRelease) -> bool:
        with self._lock:
            self._issued_release(receipt)
            self.collect_completed()
            return receipt.release_id not in self._releases

    def retire(self, refs: tuple[StateRef, ...]) -> None:
        with self._lock:
            self._require_live()
            if not refs or len(set(refs)) != len(refs):
                raise ValueError("PageBundle retirement requires distinct latest roots")
            for ref in refs:
                if self.latest(ref.lineage_id) != ref or any(ref in write.token.before_refs for write in self._writes.values()):
                    raise RuntimeError("PageBundle retirement overlaps a write or stale version")
            lineages = dict(self._lineages)
            for ref in refs:
                lineages[ref.lineage_id] = _Lineage(self.root(ref).resource, None)
            self._lineages = lineages
            self.collect_completed()

    def collect_completed(self, *, wait: bool = False) -> None:
        with self._lock:
            self._require_live()
            if wait:
                for publication in self._publications.values():
                    if publication.receipt.completion is not None:
                        publication.receipt.completion.synchronize()
                for release in self._releases.values():
                    if release.completion is not None:
                        release.completion.synchronize()
            for publication in tuple(self._publications.values()):
                receipt = publication.receipt
                if publication.settled and (receipt.completion is None or receipt.completion.query()):
                    for ref in (*receipt.before_refs, *receipt.after_refs):
                        self._roots[ref].pins -= 1
                    self._publications.pop(receipt.operation_id)
            for receipt in tuple(self._releases.values()):
                if receipt.completion is None or receipt.completion.query():
                    for ref in receipt.pin.refs:
                        self._roots[ref].pins -= 1
                    self._releases.pop(receipt.release_id)
            self._reclaim()

    def _reclaim(self) -> None:
        for ref, stored in tuple(self._roots.items()):
            bundle = stored.bundle
            if (not stored.pins and self._lineages[ref.lineage_id].latest != ref
                and (bundle.readiness is None or bundle.readiness.query())):
                self._pools[bundle.resource].release(bundle.pages)
                self._roots.pop(ref)
        retained = {ref.lineage_id for ref in self._roots}
        for lineage, state in tuple(self._lineages.items()):
            if state.latest is None and lineage not in retained:
                self._lineages.pop(lineage)

    def _issued_write(self, token: PageWrite) -> _Write:
        self._require_live()
        write = self._writes.get(token.operation_id)
        if write is None:
            raise RuntimeError("PageBundle write is unknown or consumed")
        if write.token is not token:
            raise RuntimeError("PageBundle write differs from the issued token")
        return write

    def _issued_pin(self, pin: PagePin) -> None:
        self._require_live()
        issued = self._pins.get(pin.pin_id)
        if issued is None:
            raise RuntimeError("PageBundle pin is unknown or released; no new reads may start")
        if issued is not pin:
            raise RuntimeError("PageBundle pin differs from the issued token")

    def _issued_release(self, receipt: PageRelease) -> None:
        self._require_live()
        if self._issued_releases.get(receipt.release_id) is not receipt:
            raise RuntimeError("PageBundle release differs from the issued receipt")

    def _require_live(self) -> None:
        if self._poisoned:
            raise RuntimeError("PageBundle arena is fail-stopped with resources retained")

    def _require_stream(self, stream: torch.cuda.Stream | None) -> None:
        self._require_live()
        if (self.device.type == "cpu" and stream is not None
            or self.device.type == "cuda" and (stream is None or stream.device != self.device)):
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
