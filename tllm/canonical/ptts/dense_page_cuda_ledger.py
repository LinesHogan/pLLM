"""CUDA event-gated logical roots for the canonical dense-page provider."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from weakref import WeakValueDictionary

import torch

from tllm.canonical.ptts.dense_page_cuda_types import CudaDensePageVersionPin
from tllm.contracts.state_ref import LayoutId, LineageId, StateRef, VersionId


@dataclass(frozen=True, slots=True)
class CudaDensePageFence:
    fence_id: int
    device_ordinal: int
    stream: torch.cuda.Stream
    event: torch.cuda.Event


@dataclass(frozen=True, slots=True)
class CudaDensePageRoot:
    ref: StateRef
    page_id: int
    readiness: CudaDensePageFence


@dataclass(frozen=True, slots=True)
class CudaDensePageAdmissionBatch:
    transaction_id: int
    targets: tuple[StateRef, ...]
    target_page_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CudaDensePageTransitionBatch:
    transaction_id: int
    sources: tuple[StateRef, ...]
    targets: tuple[StateRef, ...]
    source_page_ids: tuple[int, ...]
    target_page_ids: tuple[int, ...]


CudaDensePageWriteBatch = (
    CudaDensePageAdmissionBatch | CudaDensePageTransitionBatch
)


@dataclass(slots=True)
class _StoredCudaRoot:
    root: CudaDensePageRoot
    pins: int = 0


@dataclass(frozen=True, slots=True)
class _NoPendingWrite:
    pass


@dataclass(frozen=True, slots=True)
class _ReservedAdmission:
    batch: CudaDensePageAdmissionBatch


@dataclass(frozen=True, slots=True)
class _SubmittedAdmission:
    batch: CudaDensePageAdmissionBatch


@dataclass(frozen=True, slots=True)
class _ReservedTransition:
    batch: CudaDensePageTransitionBatch


@dataclass(frozen=True, slots=True)
class _SubmittedTransition:
    batch: CudaDensePageTransitionBatch


_PendingWrite = (
    _NoPendingWrite
    | _ReservedAdmission
    | _SubmittedAdmission
    | _ReservedTransition
    | _SubmittedTransition
)
_NO_PENDING_WRITE = _NoPendingWrite()


@dataclass(frozen=True, slots=True)
class _PendingPinRelease:
    pin: CudaDensePageVersionPin
    completion: CudaDensePageFence


@dataclass(frozen=True, slots=True)
class _PendingWriterRelease:
    source: StateRef
    completion: CudaDensePageFence


class CudaDensePageLedger:
    """Own page IDs and immutable StateRef roots; tensor pools stay external."""

    __slots__ = (
        "_layout_id",
        "_device_ordinal",
        "_max_admitted_lineages",
        "_max_live_versions_per_lineage",
        "_free_page_ids",
        "_roots",
        "_latest_by_lineage",
        "_admitted_lineages",
        "_pins",
        "_completed_pins",
        "_pending_pin_releases",
        "_pending_writer_releases",
        "_pending_write",
        "_next_pin_id",
        "_next_transaction_id",
        "_poisoned",
        "_lock",
    )

    def __init__(
        self,
        *,
        layout_id: LayoutId,
        device_ordinal: int,
        max_admitted_lineages: int,
        max_live_versions_per_lineage: int,
    ) -> None:
        if (
            type(layout_id) is not int
            or type(device_ordinal) is not int
            or type(max_admitted_lineages) is not int
            or type(max_live_versions_per_lineage) is not int
            or int(layout_id) < 0
            or device_ordinal < 0
            or max_admitted_lineages < 1
            or max_live_versions_per_lineage < 2
        ):
            raise ValueError("CUDA dense-page ledger capacity facts are invalid")
        physical_rows = max_admitted_lineages * max_live_versions_per_lineage
        self._layout_id = layout_id
        self._device_ordinal = device_ordinal
        self._max_admitted_lineages = max_admitted_lineages
        self._max_live_versions_per_lineage = max_live_versions_per_lineage
        self._free_page_ids = list(range(physical_rows))
        self._roots: dict[StateRef, _StoredCudaRoot] = {}
        self._latest_by_lineage: dict[LineageId, StateRef] = {}
        self._admitted_lineages: set[LineageId] = set()
        self._pins: dict[int, CudaDensePageVersionPin] = {}
        self._completed_pins: WeakValueDictionary[
            int,
            CudaDensePageVersionPin,
        ] = WeakValueDictionary()
        self._pending_pin_releases: dict[int, _PendingPinRelease] = {}
        self._pending_writer_releases: list[_PendingWriterRelease] = []
        self._pending_write: _PendingWrite = _NO_PENDING_WRITE
        self._next_pin_id = 1
        self._next_transaction_id = 1
        self._poisoned = False
        self._lock = RLock()

    @property
    def free_pages(self) -> int:
        with self._lock:
            self._require_live()
            self._collect_completed_locked()
            return len(self._free_page_ids)

    @property
    def poisoned(self) -> bool:
        with self._lock:
            return self._poisoned

    @property
    def pending_pin_releases(self) -> int:
        with self._lock:
            self._require_live()
            return len(self._pending_pin_releases)

    def prepare_admission_batch(
        self,
        targets: tuple[StateRef, ...],
    ) -> CudaDensePageAdmissionBatch:
        with self._lock:
            self._require_no_pending_write()
            self._collect_completed_locked()
            if (
                type(targets) is not tuple
                or not targets
                or any(type(target) is not StateRef for target in targets)
                or len(targets) != len(set(targets))
            ):
                raise ValueError("CUDA admission requires unique references")
            lineages = tuple(target.lineage_id for target in targets)
            if len(lineages) != len(set(lineages)):
                raise ValueError("CUDA admission requires unique lineages")
            if (
                len(self._admitted_lineages) + len(lineages)
                > self._max_admitted_lineages
            ):
                raise RuntimeError("CUDA admission entitlement is exhausted")
            for target in targets:
                if (
                    target.layout_id != self._layout_id
                    or target.version_id != VersionId(0)
                ):
                    raise ValueError("CUDA admission requires this layout at V0")
                if (
                    target.lineage_id in self._admitted_lineages
                    or target in self._roots
                ):
                    raise RuntimeError("CUDA lineage is already admitted")
            page_ids = self._reserve_pages(len(targets))
            batch = CudaDensePageAdmissionBatch(
                self._take_transaction_id(),
                targets,
                page_ids,
            )
            self._pending_write = _ReservedAdmission(batch)
            return batch

    def prepare_transition_batch(
        self,
        sources: tuple[StateRef, ...],
        targets: tuple[StateRef, ...],
        stream: torch.cuda.Stream,
    ) -> CudaDensePageTransitionBatch:
        with self._lock:
            self._require_stream(stream)
            self._require_no_pending_write()
            self._collect_completed_locked()
            if (
                type(sources) is not tuple
                or type(targets) is not tuple
                or not sources
                or any(type(source) is not StateRef for source in sources)
                or any(type(target) is not StateRef for target in targets)
                or len(sources) != len(targets)
                or len(sources) != len(set(sources))
                or len(targets) != len(set(targets))
            ):
                raise ValueError("CUDA transition requires aligned unique roots")
            lineages = tuple(source.lineage_id for source in sources)
            if len(lineages) != len(set(lineages)):
                raise ValueError("CUDA transition requires unique lineages")
            stored_roots: list[_StoredCudaRoot] = []
            for source, target in zip(sources, targets, strict=True):
                self._require_transition_refs(source, target)
                stored = self._stored_root(source)
                stream.wait_event(stored.root.readiness.event)
                stored_roots.append(stored)
            target_page_ids = self._reserve_pages(len(targets))
            for stored in stored_roots:
                stored.pins += 1
            batch = CudaDensePageTransitionBatch(
                self._take_transaction_id(),
                sources,
                targets,
                tuple(stored.root.page_id for stored in stored_roots),
                target_page_ids,
            )
            self._pending_write = _ReservedTransition(batch)
            return batch

    def mark_submitted(self, batch: CudaDensePageWriteBatch) -> None:
        with self._lock:
            match self._pending_write, batch:
                case (
                    _ReservedAdmission(batch=issued),
                    CudaDensePageAdmissionBatch(),
                ) if issued is batch:
                    self._pending_write = _SubmittedAdmission(batch)
                case (
                    _ReservedTransition(batch=issued),
                    CudaDensePageTransitionBatch(),
                ) if issued is batch:
                    self._pending_write = _SubmittedTransition(batch)
                case _:
                    raise RuntimeError("CUDA page batch is not reserved")

    def publish_batch(
        self,
        batch: CudaDensePageWriteBatch,
        readiness: CudaDensePageFence,
    ) -> tuple[CudaDensePageRoot, ...]:
        with self._lock:
            self._require_fence(readiness)
            match self._pending_write, batch:
                case (
                    _SubmittedAdmission(batch=issued),
                    CudaDensePageAdmissionBatch(),
                ) if issued is batch:
                    pass
                case (
                    _SubmittedTransition(batch=issued),
                    CudaDensePageTransitionBatch(),
                ) if issued is batch:
                    pass
                case _:
                    raise RuntimeError("CUDA page batch is not submitted")
            roots = tuple(
                CudaDensePageRoot(target, page_id, readiness)
                for target, page_id in zip(
                    batch.targets,
                    batch.target_page_ids,
                    strict=True,
                )
            )
            match batch:
                case CudaDensePageAdmissionBatch():
                    self._admitted_lineages.update(
                        root.ref.lineage_id for root in roots
                    )
                case CudaDensePageTransitionBatch():
                    pass
            for root in roots:
                self._roots[root.ref] = _StoredCudaRoot(root)
                self._latest_by_lineage[root.ref.lineage_id] = root.ref
            match batch:
                case CudaDensePageTransitionBatch(sources=sources):
                    self._pending_writer_releases.extend(
                        _PendingWriterRelease(source, readiness)
                        for source in sources
                    )
                case CudaDensePageAdmissionBatch():
                    pass
            self._pending_write = _NO_PENDING_WRITE
            self._collect_completed_locked()
            return roots

    def abort_unsubmitted_batch(self, batch: CudaDensePageWriteBatch) -> None:
        with self._lock:
            match self._pending_write, batch:
                case (
                    _ReservedAdmission(batch=issued),
                    CudaDensePageAdmissionBatch(),
                ) if issued is batch:
                    pass
                case (
                    _ReservedTransition(batch=issued),
                    CudaDensePageTransitionBatch(),
                ) if issued is batch:
                    for source in batch.sources:
                        self._stored_root(source).pins -= 1
                case _:
                    raise RuntimeError("only an unsubmitted CUDA batch can abort")
            self._return_pages(batch.target_page_ids)
            self._pending_write = _NO_PENDING_WRITE
            self._reclaim_unpinned_roots()

    def pin_exact(
        self,
        refs: tuple[StateRef, ...],
        stream: torch.cuda.Stream,
    ) -> CudaDensePageVersionPin:
        with self._lock:
            self._require_stream(stream)
            self._collect_completed_locked()
            if (
                type(refs) is not tuple
                or not refs
                or any(type(ref) is not StateRef for ref in refs)
                or len(refs) != len(set(refs))
            ):
                raise ValueError("CUDA pin requires unique exact roots")
            roots = tuple(self._stored_root(ref).root for ref in refs)
            for root in roots:
                stream.wait_event(root.readiness.event)
            pin = CudaDensePageVersionPin(
                self._next_pin_id,
                refs,
                stream,
            )
            for ref in refs:
                self._stored_root(ref).pins += 1
            self._pins[pin.pin_id] = pin
            self._next_pin_id += 1
            return pin

    def release_pin(
        self,
        pin: CudaDensePageVersionPin,
        completion: CudaDensePageFence,
    ) -> None:
        with self._lock:
            self._require_fence(completion)
            self._active_pin(pin)
            if pin.stream.cuda_stream != completion.stream.cuda_stream:
                raise RuntimeError(
                    "CUDA pin completion belongs to another consumer stream"
                )
            if pin.pin_id in self._pending_pin_releases:
                raise RuntimeError("CUDA pin release is already pending")
            self._pending_pin_releases[pin.pin_id] = _PendingPinRelease(
                pin,
                completion,
            )

    def pinned_page_ids(
        self,
        pin: CudaDensePageVersionPin,
        stream: torch.cuda.Stream,
    ) -> tuple[int, ...]:
        with self._lock:
            self._require_stream(stream)
            issued = self._readable_pin(pin)
            if stream.cuda_stream != issued.stream.cuda_stream:
                raise RuntimeError("CUDA pin read uses another consumer stream")
            roots = tuple(
                self._stored_root(ref).root for ref in issued.refs
            )
            for root in roots:
                stream.wait_event(root.readiness.event)
            return tuple(root.page_id for root in roots)

    def wait_ready(self, refs: tuple[StateRef, ...]) -> None:
        with self._lock:
            if (
                type(refs) is not tuple
                or not refs
                or any(type(ref) is not StateRef for ref in refs)
            ):
                raise ValueError("CUDA readiness wait requires exact roots")
            events = tuple(
                self._stored_root(ref).root.readiness.event for ref in refs
            )
            try:
                for event in events:
                    event.synchronize()
            except Exception:
                self._poisoned = True
                raise

    def wait_pin_release(self, pin: CudaDensePageVersionPin) -> None:
        with self._lock:
            if self._completed_pin(pin):
                return
            pending = self._pending_pin_release(pin)
            try:
                pending.completion.event.synchronize()
            except Exception:
                self._poisoned = True
                raise
            self._finalize_pin_release(pending)

    def pin_release_complete(self, pin: CudaDensePageVersionPin) -> bool:
        with self._lock:
            if self._completed_pin(pin):
                return True
            pending = self._pending_pin_release(pin)
            if not self._query_event(pending.completion.event):
                return False
            self._finalize_pin_release(pending)
            return True

    def retire_batch(self, refs: tuple[StateRef, ...]) -> None:
        with self._lock:
            self._require_no_pending_write()
            self._collect_completed_locked()
            if (
                type(refs) is not tuple
                or not refs
                or any(type(ref) is not StateRef for ref in refs)
                or len(refs) != len(set(refs))
            ):
                raise ValueError("CUDA retirement requires unique roots")
            lineages = tuple(ref.lineage_id for ref in refs)
            if len(lineages) != len(set(lineages)):
                raise ValueError("CUDA retirement requires unique lineages")
            stored_roots = tuple(self._stored_root(ref) for ref in refs)
            for ref, stored in zip(refs, stored_roots, strict=True):
                try:
                    latest = self._latest_by_lineage[ref.lineage_id]
                except KeyError as exc:
                    raise RuntimeError(
                        "CUDA retirement requires latest roots"
                    ) from exc
                if latest != ref:
                    raise RuntimeError("CUDA retirement requires latest roots")
                if stored.pins:
                    raise RuntimeError("cannot retire a fenced CUDA root")
                if not self._query_event(stored.root.readiness.event):
                    raise RuntimeError("CUDA root writer is still pending")
                if any(
                    candidate.lineage_id == ref.lineage_id and candidate != ref
                    for candidate in self._roots
                ):
                    raise RuntimeError(
                        "cannot retire a lineage with retained CUDA versions"
                    )
            page_ids = tuple(stored.root.page_id for stored in stored_roots)
            for ref in refs:
                self._roots.pop(ref)
                self._latest_by_lineage.pop(ref.lineage_id)
                self._admitted_lineages.remove(ref.lineage_id)
            self._return_pages(page_ids)

    def collect_completed(self) -> None:
        with self._lock:
            self._require_live()
            self._collect_completed_locked()

    def poison(self) -> None:
        with self._lock:
            self._poisoned = True

    def _require_transition_refs(self, source: StateRef, target: StateRef) -> None:
        if (
            target.layout_id != source.layout_id
            or target.lineage_id != source.lineage_id
            or int(target.version_id) != int(source.version_id) + 1
        ):
            raise ValueError("CUDA transition target must be the next version")
        if source.layout_id != self._layout_id:
            raise RuntimeError("CUDA transition uses another layout")
        try:
            latest = self._latest_by_lineage[source.lineage_id]
        except KeyError as exc:
            raise RuntimeError("CUDA transition lineage is not admitted") from exc
        if latest != source or target in self._roots:
            raise RuntimeError("CUDA transition source is not latest")
        live_versions = sum(
            ref.lineage_id == source.lineage_id for ref in self._roots
        )
        if live_versions + 1 > self._max_live_versions_per_lineage:
            raise RuntimeError("CUDA version entitlement is exhausted")

    def _collect_completed_locked(self) -> None:
        pin_completion = tuple(
            (pin_id, self._query_event(pending.completion.event))
            for pin_id, pending in self._pending_pin_releases.items()
        )
        writer_completion = tuple(
            self._query_event(pending.completion.event)
            for pending in self._pending_writer_releases
        )
        completed_pins = tuple(
            pin_id for pin_id, complete in pin_completion if complete
        )
        for pin_id in completed_pins:
            pending = self._pending_pin_releases[pin_id]
            self._finalize_pin_release(pending)
        retained_writers: list[_PendingWriterRelease] = []
        for pending, complete in zip(
            self._pending_writer_releases,
            writer_completion,
            strict=True,
        ):
            if complete:
                self._stored_root(pending.source).pins -= 1
            else:
                retained_writers.append(pending)
        self._pending_writer_releases = retained_writers
        self._reclaim_unpinned_roots()

    def _reclaim_unpinned_roots(self) -> None:
        latest = frozenset(self._latest_by_lineage.values())
        reclaimable = tuple(
            ref
            for ref, stored in self._roots.items()
            if (
                ref not in latest
                and stored.pins == 0
                and self._query_event(stored.root.readiness.event)
            )
        )
        pages = tuple(self._roots.pop(ref).root.page_id for ref in reclaimable)
        self._return_pages(pages)

    def _query_event(self, event: torch.cuda.Event) -> bool:
        try:
            return event.query()
        except Exception:
            self._poisoned = True
            raise

    def _stored_root(self, ref: StateRef) -> _StoredCudaRoot:
        self._require_live()
        try:
            return self._roots[ref]
        except KeyError as exc:
            raise RuntimeError("CUDA root is unknown or reclaimed") from exc

    def _active_pin(
        self,
        pin: CudaDensePageVersionPin,
    ) -> CudaDensePageVersionPin:
        try:
            issued = self._pins[pin.pin_id]
        except KeyError as exc:
            raise RuntimeError("CUDA pin is unknown or released") from exc
        if issued is not pin:
            raise RuntimeError("CUDA pin differs from the issued pin")
        return issued

    def _pending_pin_release(
        self,
        pin: CudaDensePageVersionPin,
    ) -> _PendingPinRelease:
        self._active_pin(pin)
        try:
            pending = self._pending_pin_releases[pin.pin_id]
        except KeyError as exc:
            raise RuntimeError("CUDA pin has no pending release") from exc
        if pending.pin is not pin:
            raise RuntimeError("CUDA pending release belongs to another pin")
        return pending

    def _completed_pin(self, pin: CudaDensePageVersionPin) -> bool:
        try:
            completed = self._completed_pins[pin.pin_id]
        except KeyError:
            return False
        if completed is not pin:
            raise RuntimeError("CUDA completed pin differs from the issued pin")
        return True

    def _finalize_pin_release(
        self,
        pending: _PendingPinRelease,
    ) -> None:
        issued = self._active_pin(pending.pin)
        for ref in issued.refs:
            self._stored_root(ref).pins -= 1
        self._completed_pins[issued.pin_id] = issued
        self._pins.pop(issued.pin_id)
        self._pending_pin_releases.pop(issued.pin_id)
        self._reclaim_unpinned_roots()

    def _readable_pin(
        self,
        pin: CudaDensePageVersionPin,
    ) -> CudaDensePageVersionPin:
        issued = self._active_pin(pin)
        if pin.pin_id in self._pending_pin_releases:
            raise RuntimeError("CUDA pin release is pending; no new reads may start")
        return issued

    def _reserve_pages(self, count: int) -> tuple[int, ...]:
        if len(self._free_page_ids) < count:
            raise AssertionError("CUDA entitlement lost its physical pages")
        pages = tuple(self._free_page_ids[:count])
        del self._free_page_ids[:count]
        return pages

    def _return_pages(self, page_ids: tuple[int, ...]) -> None:
        self._free_page_ids.extend(page_ids)
        self._free_page_ids.sort()

    def _take_transaction_id(self) -> int:
        transaction_id = self._next_transaction_id
        self._next_transaction_id += 1
        return transaction_id

    def _require_stream(self, stream: torch.cuda.Stream) -> None:
        self._require_live()
        if (
            type(stream) is not torch.cuda.Stream
            or stream.device != torch.device("cuda", self._device_ordinal)
        ):
            raise RuntimeError("CUDA dense-page stream uses another device")

    def _require_fence(self, fence: CudaDensePageFence) -> None:
        self._require_live()
        if fence.device_ordinal != self._device_ordinal:
            raise RuntimeError("CUDA dense-page fence uses another device")

    def _require_no_pending_write(self) -> None:
        self._require_live()
        match self._pending_write:
            case _NoPendingWrite():
                return
            case _:
                raise RuntimeError("CUDA dense-page write is pending")

    def _require_live(self) -> None:
        if self._poisoned:
            raise RuntimeError("CUDA dense-page ledger is poisoned")


__all__ = [
    "CudaDensePageAdmissionBatch",
    "CudaDensePageLedger",
    "CudaDensePageRoot",
    "CudaDensePageTransitionBatch",
    "CudaDensePageWriteBatch",
]
