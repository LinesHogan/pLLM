"""Minimal immutable-root core for paged test-time tensor state.

The core owns physical pages, immutable root publication, root pins, and
completion-gated reclaim. Request keys, logical versions, receipts, and
scheduling remain owned by the request-slot and scheduler authorities.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import NewType

import torch

from tllm.contracts.request_state import StateSlotRef
from tllm.runtime.request_slot_adapter import (
    CompletionFence,
    CudaCompletionFence,
    NoCompletionFence,
)


StateLayoutId = NewType("StateLayoutId", int)
StateResourceId = NewType("StateResourceId", str)
TensorFieldId = NewType("TensorFieldId", int)


@dataclass(frozen=True, slots=True)
class FrozenTensorField:
    field_id: TensorFieldId
    shape: tuple[int, ...]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if int(self.field_id) < 0:
            raise ValueError("tensor field id must be non-negative")
        if any(size < 1 for size in self.shape):
            raise ValueError("tensor field dimensions must be positive")


@dataclass(frozen=True, slots=True)
class FrozenTensorLayout:
    layout_id: StateLayoutId
    root_page_count: int
    fields: tuple[FrozenTensorField, ...]

    def __post_init__(self) -> None:
        if int(self.layout_id) < 0:
            raise ValueError("state layout id must be non-negative")
        if self.root_page_count < 1:
            raise ValueError("state layout root page count must be positive")
        if not self.fields:
            raise ValueError("state layout requires at least one tensor field")
        if tuple(int(field.field_id) for field in self.fields) != tuple(
            range(len(self.fields))
        ):
            raise ValueError("state tensor field ids must be dense and ordered")


PageTensorRows = tuple[tuple[torch.Tensor, ...], ...]


class PageBundle:
    """Cold-allocated global pools sharing a dense physical page index."""

    __slots__ = (
        "_layout",
        "_capacity",
        "_device",
        "_pools",
        "_addresses",
    )

    def __init__(
        self,
        layout: FrozenTensorLayout,
        *,
        capacity: int,
        device: torch.device,
    ) -> None:
        if capacity < layout.root_page_count:
            raise ValueError(
                "page bundle capacity must fit one complete version root"
            )
        if device.type not in ("cpu", "cuda"):
            raise ValueError("page bundle requires a CPU or CUDA device")
        self._layout = layout
        self._capacity = capacity
        self._device = device
        self._pools = tuple(
            torch.zeros(
                (capacity, *field.shape),
                dtype=field.dtype,
                device=device,
            )
            for field in layout.fields
        )
        self._addresses = tuple(pool.data_ptr() for pool in self._pools)

    @classmethod
    def from_pools(
        cls,
        layout: FrozenTensorLayout,
        pools: tuple[torch.Tensor, ...],
    ) -> "PageBundle":
        if len(pools) != len(layout.fields):
            raise ValueError("page bundle pools differ from their layout")
        capacity = int(pools[0].shape[0])
        device = pools[0].device
        if capacity < layout.root_page_count:
            raise ValueError(
                "page bundle capacity must fit one complete version root"
            )
        if device.type not in ("cpu", "cuda"):
            raise ValueError("page bundle requires a CPU or CUDA device")
        for field, pool in zip(layout.fields, pools, strict=True):
            if (
                tuple(pool.shape) != (capacity, *field.shape)
                or pool.dtype is not field.dtype
                or pool.device != device
            ):
                raise ValueError("page bundle pool signature differs from layout")
        bundle = cls.__new__(cls)
        bundle._layout = layout
        bundle._capacity = capacity
        bundle._device = device
        bundle._pools = pools
        bundle._addresses = tuple(pool.data_ptr() for pool in pools)
        return bundle

    @property
    def layout(self) -> FrozenTensorLayout:
        return self._layout

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def capture_operands(self) -> tuple[torch.Tensor, ...]:
        """Global fixed-address operands used by every representation."""

        return self._pools

    def require_stable(self) -> None:
        """Validate the cold tensor signature at a specialization boundary."""

        for field, pool, address in zip(
            self._layout.fields,
            self._pools,
            self._addresses,
            strict=True,
        ):
            if (
                pool.data_ptr() != address
                or tuple(pool.shape) != (self._capacity, *field.shape)
                or pool.dtype is not field.dtype
                or pool.device != self._device
            ):
                raise RuntimeError("page bundle tensor signature changed")

@dataclass(frozen=True, slots=True)
class StateVersionLineage:
    resource: StateResourceId
    epoch: int
    slot: StateSlotRef

    def __post_init__(self) -> None:
        if not str(self.resource):
            raise ValueError("state version lineage requires a resource")
        if self.epoch < 1:
            raise ValueError("state version lineage epoch must be positive")


@dataclass(frozen=True, slots=True)
class StateVersionRef:
    lineage: StateVersionLineage
    version: int

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError("state version must be non-negative")


@dataclass(frozen=True, slots=True)
class ImmutableStateVersionRoot:
    root_id: int
    ref: StateVersionRef
    layout_id: StateLayoutId
    bundle_epoch: int
    page_ids: tuple[int, ...]
    readiness: CompletionFence


@dataclass(frozen=True, slots=True)
class StateVersionRootPin:
    pin_id: int
    roots: tuple[ImmutableStateVersionRoot, ...]


@dataclass(frozen=True, slots=True)
class AdmissionPageWrite:
    """Pending capability held only by a trusted physical producer."""

    transaction_id: int
    target: StateVersionRef
    target_page_ids: tuple[int, ...]
    target_pages: PageTensorRows


@dataclass(frozen=True, slots=True)
class TransitionPageWrite:
    """Pending COW capability held only by a trusted physical producer."""

    transaction_id: int
    source: StateVersionRef
    target: StateVersionRef
    source_page_ids: tuple[int, ...]
    target_page_ids: tuple[int, ...]
    source_pages: PageTensorRows
    target_pages: PageTensorRows


PageWrite = AdmissionPageWrite | TransitionPageWrite


@dataclass(frozen=True, slots=True)
class _FreePage:
    pass


@dataclass(frozen=True, slots=True)
class _StagedPage:
    transaction_id: int


@dataclass(frozen=True, slots=True)
class _RootPage:
    ref: StateVersionRef


_PageOwner = _FreePage | _StagedPage | _RootPage


@dataclass(slots=True)
class _StoredRoot:
    root: ImmutableStateVersionRoot
    pins: int = 0


@dataclass(frozen=True, slots=True)
class _NoPendingWrite:
    pass


@dataclass(frozen=True, slots=True)
class _PendingAdmission:
    write: AdmissionPageWrite


@dataclass(frozen=True, slots=True)
class _PendingTransition:
    write: TransitionPageWrite


_PendingWrite = _NoPendingWrite | _PendingAdmission | _PendingTransition
_NO_PENDING_WRITE = _NoPendingWrite()


@dataclass(frozen=True, slots=True)
class _PendingPinRelease:
    pin: StateVersionRootPin
    event: torch.cuda.Event


@dataclass(frozen=True, slots=True)
class _PendingTransitionRelease:
    source: StateVersionRef
    event: torch.cuda.Event


class PagedStateVersionArena:
    """Physical root authority driven by exact caller-issued versions.

    A request-state bridge publishes the physical root before committing the
    matching logical transaction. If logical commit then fails, that bridge
    must poison both authorities; physical latest state is never used to
    reconstruct a logical current version.
    """

    __slots__ = (
        "_bundle",
        "_bundle_epoch",
        "_page_owners",
        "_roots",
        "_latest_published_root",
        "_pins",
        "_pending_pin_releases",
        "_pending_transition_releases",
        "_pending_write",
        "_next_root_id",
        "_next_pin_id",
        "_next_transaction_id",
        "_poisoned",
        "_lock",
    )

    def __init__(self, bundle: PageBundle, *, bundle_epoch: int) -> None:
        if bundle_epoch < 1:
            raise ValueError("page bundle epoch must be positive")
        bundle.require_stable()
        self._bundle = bundle
        self._bundle_epoch = bundle_epoch
        self._page_owners: list[_PageOwner] = [
            _FreePage() for _ in range(bundle.capacity)
        ]
        self._roots: dict[StateVersionRef, _StoredRoot] = {}
        self._latest_published_root: dict[
            StateVersionLineage, StateVersionRef
        ] = {}
        self._pins: dict[int, StateVersionRootPin] = {}
        self._pending_pin_releases: dict[int, _PendingPinRelease] = {}
        self._pending_transition_releases: list[
            _PendingTransitionRelease
        ] = []
        self._pending_write: _PendingWrite = _NO_PENDING_WRITE
        self._next_root_id = 1
        self._next_pin_id = 1
        self._next_transaction_id = 1
        self._poisoned = False
        self._lock = RLock()

    @property
    def bundle(self) -> PageBundle:
        return self._bundle

    @property
    def bundle_epoch(self) -> int:
        return self._bundle_epoch

    @property
    def free_pages(self) -> int:
        with self._lock:
            self._require_live()
            count = 0
            for owner in self._page_owners:
                match owner:
                    case _FreePage():
                        count += 1
                    case _StagedPage() | _RootPage():
                        pass
            return count

    def poison(self) -> None:
        with self._lock:
            self._poisoned = True

    def prepare_admission(
        self,
        target: StateVersionRef,
    ) -> AdmissionPageWrite:
        with self._lock:
            self._require_no_pending_write()
            self._collect_completed_locked()
            if target.version != 0:
                raise ValueError("admission state root must begin at version zero")
            if (
                target.lineage in self._latest_published_root
                or target in self._roots
            ):
                raise RuntimeError("state lineage is already admitted")
            transaction_id = self._next_transaction_id
            target_page_ids = self._reserve_pages(transaction_id)
            write = AdmissionPageWrite(
                transaction_id,
                target,
                target_page_ids,
                _producer_page_rows(self._bundle, target_page_ids),
            )
            self._pending_write = _PendingAdmission(write)
            self._next_transaction_id += 1
            return write

    def prepare_transition_synchronous(
        self,
        source: StateVersionRef,
        target: StateVersionRef,
    ) -> TransitionPageWrite:
        with self._lock:
            self._require_cpu()
            write = self._reserve_transition(source, target)
            match self._stored_root(source).root.readiness:
                case NoCompletionFence():
                    return write
                case CudaCompletionFence():
                    raise RuntimeError("CPU state root has a CUDA readiness event")

    def prepare_transition_cuda(
        self,
        source: StateVersionRef,
        target: StateVersionRef,
        stream: torch.cuda.Stream,
    ) -> TransitionPageWrite:
        with self._lock:
            self._require_cuda_stream(stream)
            write = self._reserve_transition(source, target)
            match self._stored_root(source).root.readiness:
                case CudaCompletionFence(event=event):
                    stream.wait_event(event)
                    return write
                case NoCompletionFence():
                    raise RuntimeError("CUDA state root has no readiness event")

    def publish(
        self,
        write: PageWrite,
        readiness: CompletionFence,
    ) -> ImmutableStateVersionRoot:
        with self._lock:
            self._require_fence_domain(readiness)
            match self._pending_write, write:
                case (
                    _PendingAdmission(write=issued),
                    AdmissionPageWrite(),
                ) if issued is write:
                    pass
                case (
                    _PendingTransition(write=issued),
                    TransitionPageWrite(),
                ) if issued is write:
                    pass
                case _:
                    raise RuntimeError("page write is not the pending transaction")
            self._bundle.require_stable()
            target, target_page_ids = self._write_target(write)
            for page_id in target_page_ids:
                match self._page_owners[page_id]:
                    case _StagedPage(transaction_id=transaction_id) if (
                        transaction_id == write.transaction_id
                    ):
                        pass
                    case _:
                        raise RuntimeError("page write lost one of its target pages")
            root = ImmutableStateVersionRoot(
                root_id=self._next_root_id,
                ref=target,
                layout_id=self._bundle.layout.layout_id,
                bundle_epoch=self._bundle_epoch,
                page_ids=target_page_ids,
                readiness=readiness,
            )
            self._roots[target] = _StoredRoot(root)
            self._latest_published_root[target.lineage] = target
            for page_id in target_page_ids:
                self._page_owners[page_id] = _RootPage(target)
            match write, readiness:
                case AdmissionPageWrite(), NoCompletionFence():
                    pass
                case AdmissionPageWrite(), CudaCompletionFence():
                    pass
                case TransitionPageWrite(source=source), NoCompletionFence():
                    self._stored_root(source).pins -= 1
                case TransitionPageWrite(source=source), CudaCompletionFence(event=event):
                    self._pending_transition_releases.append(
                        _PendingTransitionRelease(source, event)
                    )
            self._pending_write = _NO_PENDING_WRITE
            self._next_root_id += 1
            self._collect_completed_locked()
            return root

    def abort_unsubmitted(self, write: PageWrite) -> None:
        """Discard a reserved write before its producer submits device work."""

        with self._lock:
            self._require_live()
            match self._pending_write, write:
                case (
                    _PendingAdmission(write=issued),
                    AdmissionPageWrite(),
                ) if issued is write:
                    pass
                case (
                    _PendingTransition(write=issued),
                    TransitionPageWrite(),
                ) if issued is write:
                    self._stored_root(write.source).pins -= 1
                case _:
                    raise RuntimeError("page write is not the pending transaction")
            _target, target_page_ids = self._write_target(write)
            for page_id in target_page_ids:
                match self._page_owners[page_id]:
                    case _StagedPage(transaction_id=transaction_id) if (
                        transaction_id == write.transaction_id
                    ):
                        self._page_owners[page_id] = _FreePage()
                    case _:
                        raise RuntimeError("aborted write lost one of its pages")
            self._pending_write = _NO_PENDING_WRITE
            self._collect_unpinned_roots()

    def pin_exact_synchronous(
        self,
        refs: tuple[StateVersionRef, ...],
    ) -> StateVersionRootPin:
        with self._lock:
            self._require_cpu()
            roots = self._roots_for_pin(refs)
            for root in roots:
                match root.readiness:
                    case NoCompletionFence():
                        pass
                    case CudaCompletionFence():
                        raise RuntimeError("CPU state root has a CUDA readiness event")
            return self._commit_pin(roots)

    def pin_exact_cuda(
        self,
        refs: tuple[StateVersionRef, ...],
        stream: torch.cuda.Stream,
    ) -> StateVersionRootPin:
        with self._lock:
            self._require_cuda_stream(stream)
            roots = self._roots_for_pin(refs)
            for root in roots:
                match root.readiness:
                    case CudaCompletionFence(event=event):
                        stream.wait_event(event)
                    case NoCompletionFence():
                        raise RuntimeError("CUDA state root has no readiness event")
            return self._commit_pin(roots)

    def release_pin(
        self,
        pin: StateVersionRootPin,
        completion: CompletionFence,
    ) -> None:
        with self._lock:
            self._require_fence_domain(completion)
            self._active_pin(pin)
            if pin.pin_id in self._pending_pin_releases:
                raise RuntimeError("state root pin release is already pending")
            match completion:
                case NoCompletionFence():
                    self._finalize_pin_release(pin)
                case CudaCompletionFence(event=event):
                    self._pending_pin_releases[pin.pin_id] = _PendingPinRelease(
                        pin,
                        event,
                    )

    def release_completed_pin(
        self,
        pin: StateVersionRootPin,
        completion: CompletionFence,
    ) -> None:
        """Atomically release a pin after its exact fence has completed."""

        with self._lock:
            self._require_fence_domain(completion)
            self._active_pin(pin)
            if pin.pin_id in self._pending_pin_releases:
                raise RuntimeError("state root pin already has another releaser")
            if not self._fence_complete(completion):
                raise RuntimeError("state root completion fence is still pending")
            self._finalize_pin_release(pin)

    def pin_release_complete(self, pin: StateVersionRootPin) -> bool:
        with self._lock:
            self._active_pin(pin)
            if pin.pin_id not in self._pending_pin_releases:
                return False
            pending = self._pending_pin_releases[pin.pin_id]
            if not pending.event.query():
                return False
            self._finalize_pin_release(pin)
            return True

    def wait_pin_release(self, pin: StateVersionRootPin) -> None:
        with self._lock:
            self._active_pin(pin)
            try:
                pending = self._pending_pin_releases[pin.pin_id]
            except KeyError as exc:
                raise RuntimeError(
                    "state root pin has no pending CUDA release"
                ) from exc
            pending.event.synchronize()
            self._finalize_pin_release(pin)

    def collect_completed(self) -> None:
        with self._lock:
            self._collect_completed_locked()

    def root(self, ref: StateVersionRef) -> ImmutableStateVersionRoot:
        with self._lock:
            return self._stored_root(ref).root

    def retire(self, ref: StateVersionRef) -> None:
        """Retire an exact latest root after logical RETIRE is authorized."""

        with self._lock:
            self._require_no_pending_write()
            self._collect_completed_locked()
            stored = self._stored_root(ref)
            if (
                ref.lineage not in self._latest_published_root
                or self._latest_published_root[ref.lineage] != ref
            ):
                raise RuntimeError("only the latest root may retire its lineage")
            if stored.pins:
                raise RuntimeError("cannot retire a pinned state root")
            if not self._fence_complete(stored.root.readiness):
                raise RuntimeError("cannot retire a root before its writer completes")
            self._latest_published_root.pop(ref.lineage)
            self._roots.pop(ref)
            self._free_root_pages(stored.root)

    def _reserve_transition(
        self,
        source: StateVersionRef,
        target: StateVersionRef,
    ) -> TransitionPageWrite:
        self._require_no_pending_write()
        self._collect_completed_locked()
        if (
            target.lineage != source.lineage
            or target.version != source.version + 1
        ):
            raise ValueError("transition target must be the next lineage version")
        if source.lineage not in self._latest_published_root:
            raise RuntimeError("state transition lineage is not admitted")
        if (
            self._latest_published_root[source.lineage] != source
            or target in self._roots
        ):
            raise RuntimeError("state transition source is not the latest root")
        stored = self._stored_root(source)
        transaction_id = self._next_transaction_id
        target_page_ids = self._reserve_pages(transaction_id)
        write = TransitionPageWrite(
            transaction_id,
            source,
            target,
            stored.root.page_ids,
            target_page_ids,
            _producer_page_rows(self._bundle, stored.root.page_ids),
            _producer_page_rows(self._bundle, target_page_ids),
        )
        stored.pins += 1
        self._pending_write = _PendingTransition(write)
        self._next_transaction_id += 1
        return write

    def _reserve_pages(self, transaction_id: int) -> tuple[int, ...]:
        selected: list[int] = []
        required = self._bundle.layout.root_page_count
        for page_id, owner in enumerate(self._page_owners):
            match owner:
                case _FreePage():
                    selected.append(page_id)
                    if len(selected) == required:
                        break
                case _StagedPage() | _RootPage():
                    pass
        if len(selected) != required:
            raise RuntimeError("paged state capacity is exhausted by live roots")
        page_ids = tuple(selected)
        for page_id in page_ids:
            self._page_owners[page_id] = _StagedPage(transaction_id)
        return page_ids

    @staticmethod
    def _write_target(
        write: PageWrite,
    ) -> tuple[StateVersionRef, tuple[int, ...]]:
        match write:
            case AdmissionPageWrite(target=target, target_page_ids=page_ids):
                return target, page_ids
            case TransitionPageWrite(target=target, target_page_ids=page_ids):
                return target, page_ids

    def _roots_for_pin(
        self,
        refs: tuple[StateVersionRef, ...],
    ) -> tuple[ImmutableStateVersionRoot, ...]:
        self._require_live()
        if not refs or len(refs) != len(set(refs)):
            raise ValueError("state root pin requires unique version references")
        return tuple(self._stored_root(ref).root for ref in refs)

    def _commit_pin(
        self,
        roots: tuple[ImmutableStateVersionRoot, ...],
    ) -> StateVersionRootPin:
        pin = StateVersionRootPin(self._next_pin_id, roots)
        for root in roots:
            self._stored_root(root.ref).pins += 1
        self._pins[pin.pin_id] = pin
        self._next_pin_id += 1
        return pin

    def _active_pin(self, pin: StateVersionRootPin) -> StateVersionRootPin:
        self._require_live()
        try:
            issued = self._pins[pin.pin_id]
        except KeyError as exc:
            raise RuntimeError("state root pin is unknown or released") from exc
        if issued is not pin:
            raise RuntimeError("state root pin differs from the issued pin")
        return issued

    def _finalize_pin_release(self, pin: StateVersionRootPin) -> None:
        issued = self._active_pin(pin)
        for root in issued.roots:
            self._stored_root(root.ref).pins -= 1
        self._pins.pop(pin.pin_id)
        if pin.pin_id in self._pending_pin_releases:
            self._pending_pin_releases.pop(pin.pin_id)
        self._collect_unpinned_roots()

    def _collect_completed_locked(self) -> None:
        self._require_live()
        completed_pin_ids = tuple(
            pin_id
            for pin_id, pending in self._pending_pin_releases.items()
            if pending.event.query()
        )
        for pin_id in completed_pin_ids:
            self._finalize_pin_release(self._pins[pin_id])
        remaining_transitions: list[_PendingTransitionRelease] = []
        for pending in self._pending_transition_releases:
            if pending.event.query():
                self._stored_root(pending.source).pins -= 1
            else:
                remaining_transitions.append(pending)
        self._pending_transition_releases = remaining_transitions
        self._collect_unpinned_roots()

    def _collect_unpinned_roots(self) -> None:
        latest_refs = frozenset(self._latest_published_root.values())
        removable = tuple(
            ref
            for ref, stored in self._roots.items()
            if ref not in latest_refs and stored.pins == 0
        )
        for ref in removable:
            stored = self._roots.pop(ref)
            self._free_root_pages(stored.root)

    def _free_root_pages(self, root: ImmutableStateVersionRoot) -> None:
        for page_id in root.page_ids:
            match self._page_owners[page_id]:
                case _RootPage(ref=owner_ref) if owner_ref == root.ref:
                    self._page_owners[page_id] = _FreePage()
                case _:
                    raise RuntimeError("state root lost one of its physical pages")

    def _stored_root(self, ref: StateVersionRef) -> _StoredRoot:
        self._require_live()
        try:
            return self._roots[ref]
        except KeyError as exc:
            raise RuntimeError("state version root is unknown or reclaimed") from exc

    def _require_no_pending_write(self) -> None:
        self._require_live()
        match self._pending_write:
            case _NoPendingWrite():
                return
            case _PendingAdmission() | _PendingTransition():
                raise RuntimeError("paged state already has a pending write")

    def _require_fence_domain(self, fence: CompletionFence) -> None:
        self._require_live()
        match self._bundle.device.type, fence:
            case "cpu", NoCompletionFence():
                return
            case "cuda", CudaCompletionFence():
                return
            case "cpu", CudaCompletionFence():
                raise RuntimeError("CPU state cannot publish a CUDA fence")
            case "cuda", NoCompletionFence():
                raise RuntimeError("CUDA state requires a completion event")
            case _:
                raise RuntimeError("paged state has an unsupported device")

    @staticmethod
    def _fence_complete(fence: CompletionFence) -> bool:
        match fence:
            case NoCompletionFence():
                return True
            case CudaCompletionFence(event=event):
                return event.query()

    def _require_cpu(self) -> None:
        self._require_live()
        if self._bundle.device.type != "cpu":
            raise RuntimeError("synchronous paged-state operation requires CPU")

    def _require_cuda_stream(self, stream: torch.cuda.Stream) -> None:
        self._require_live()
        if self._bundle.device.type != "cuda":
            raise RuntimeError("CUDA paged-state operation requires CUDA storage")
        if stream.device != self._bundle.device:
            raise RuntimeError("paged-state stream belongs to another device")

    def _require_live(self) -> None:
        if self._poisoned:
            raise RuntimeError("paged state version arena is poisoned")


def _producer_page_rows(
    bundle: PageBundle,
    page_ids: tuple[int, ...],
) -> PageTensorRows:
    """Resolve rows only while the physical arena owns a pending write."""

    if len(page_ids) != bundle.layout.root_page_count:
        raise ValueError("version root page map differs from its layout")
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("one version root cannot alias its physical pages")
    if any(page_id < 0 or page_id >= bundle.capacity for page_id in page_ids):
        raise ValueError("physical page id is outside the page bundle")
    return tuple(
        tuple(pool[page_id] for pool in bundle.capture_operands)
        for page_id in page_ids
    )


__all__ = [
    "AdmissionPageWrite",
    "FrozenTensorField",
    "FrozenTensorLayout",
    "ImmutableStateVersionRoot",
    "PageBundle",
    "PagedStateVersionArena",
    "PageWrite",
    "StateLayoutId",
    "StateResourceId",
    "StateVersionLineage",
    "StateVersionRef",
    "StateVersionRootPin",
    "TensorFieldId",
    "TransitionPageWrite",
]
