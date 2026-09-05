"""Request-key selection and exclusive transition leases over one PageBundle arena."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import torch

from tllm.canonical.ptts.page_bundle import PageBundleArena, PagePublication, PageWrite
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.request_state import IndexedRequestIdentityBatch, RequestStateKey
from tllm.contracts.semantic_graph import SemanticState
from tllm.contracts.state_ref import LineageId, StateRef


@dataclass(frozen=True, slots=True)
class RequestStateAdmissionBatch:
    keys: tuple[RequestStateKey, ...]
    write: PageWrite


@dataclass(frozen=True, slots=True)
class RequestStateSelectedRow:
    canonical_row_index: int
    identity: RequestIdentity
    key: RequestStateKey
    state_index: int


@dataclass(frozen=True, slots=True)
class RequestStateSelection:
    rows: tuple[RequestStateSelectedRow, ...]
    keys: tuple[RequestStateKey, ...]
    refs: tuple[StateRef, ...]


@dataclass(frozen=True, slots=True)
class RequestStateTransitionLease:
    lease_id: int
    selection: RequestStateSelection


class RequestStateDirectory:
    """Own key-to-lineage relations, never another latest-version map."""

    def __init__(self, arena: PageBundleArena, state: SemanticState) -> None:
        self.arena = arena
        self.state = state
        self._lineages: dict[RequestStateKey, LineageId] = {}
        self._leases: dict[int, RequestStateTransitionLease] = {}
        self._leased_keys: set[RequestStateKey] = set()
        self._pending: RequestStateAdmissionBatch | None = None
        self._next_lease = 1
        self._lock = RLock()

    @property
    def active_count(self) -> int:
        return len(self._lineages)

    def prepare_admission(self, identities: tuple[RequestIdentity, ...],
                          stream: torch.cuda.Stream | None = None) -> RequestStateAdmissionBatch:
        with self._lock:
            if self._pending is not None:
                raise RuntimeError("request admission is already pending")
            keys = self._unique_keys(identities)
            if not keys or any(key in self._lineages for key in keys):
                raise RuntimeError("request admission requires unadmitted keys")
            batch = RequestStateAdmissionBatch(keys,
                self.arena.prepare_admission((str(self.state.resource),) * len(keys), stream))
            self._pending = batch
            return batch

    def commit_admission(self, batch: RequestStateAdmissionBatch) -> PagePublication:
        with self._lock:
            self._require_admission(batch)
            lineages = dict(self._lineages)
            for key, ref in zip(batch.keys, batch.write.after_refs, strict=True):
                lineages[key] = ref.lineage_id
            receipt = self.arena.publish(batch.write)
            self._lineages = lineages
            self._pending = None
            return receipt

    def abort_admission(self, batch: RequestStateAdmissionBatch) -> None:
        with self._lock:
            self._require_admission(batch)
            self.arena.abort(batch.write)
            self._pending = None

    def select(self, indexed_rows: IndexedRequestIdentityBatch) -> RequestStateSelection:
        with self._lock:
            keys: list[RequestStateKey] = []
            refs: list[StateRef] = []
            rows: list[RequestStateSelectedRow] = []
            indices: dict[RequestStateKey, int] = {}
            for indexed in indexed_rows.rows:
                key = RequestStateKey.project(indexed.identity, self.state.key_fields)
                if key not in indices:
                    try:
                        ref = self.arena.latest(self._lineages[key])
                    except KeyError as exc:
                        raise RuntimeError("request selection references an unadmitted key") from exc
                    indices[key] = len(keys)
                    keys.append(key)
                    refs.append(ref)
                rows.append(RequestStateSelectedRow(indexed.canonical_row_index, indexed.identity, key, indices[key]))
            if not rows:
                raise ValueError("request selection requires rows")
            return RequestStateSelection(tuple(rows), tuple(keys), tuple(refs))

    def acquire_transition_lease(self, selection: RequestStateSelection) -> RequestStateTransitionLease:
        with self._lock:
            self._require_latest(selection)
            if self._leased_keys.intersection(selection.keys):
                raise RuntimeError("request transition lease overlaps an active step")
            lease = RequestStateTransitionLease(self._next_lease, selection)
            self._next_lease += 1
            self._leases[lease.lease_id] = lease
            self._leased_keys.update(selection.keys)
            return lease

    def release_transition_lease(self, lease: RequestStateTransitionLease) -> None:
        with self._lock:
            self._require_lease(lease)
            self._leases.pop(lease.lease_id)
            self._leased_keys.difference_update(lease.selection.keys)

    def retire(self, identities: tuple[RequestIdentity, ...]) -> tuple[StateRef, ...]:
        with self._lock:
            keys = self._unique_keys(identities)
            if not keys or self._leased_keys.intersection(keys):
                raise RuntimeError("request retirement requires keys without an active step")
            try:
                refs = tuple(self.arena.latest(self._lineages[key]) for key in keys)
            except KeyError as exc:
                raise RuntimeError("request retirement references an unadmitted key") from exc
            lineages = {key: lineage for key, lineage in self._lineages.items() if key not in keys}
            self.arena.retire(refs)
            self._lineages = lineages
            return refs

    def _unique_keys(self, identities: tuple[RequestIdentity, ...]) -> tuple[RequestStateKey, ...]:
        return tuple(dict.fromkeys(RequestStateKey.project(identity, self.state.key_fields) for identity in identities))

    def _require_latest(self, selection: RequestStateSelection) -> None:
        for key, ref in zip(selection.keys, selection.refs, strict=True):
            if self._lineages.get(key) != ref.lineage_id or self.arena.latest(ref.lineage_id) != ref:
                raise RuntimeError("request transition selection is stale")

    def _require_lease(self, lease: RequestStateTransitionLease) -> None:
        if self._leases.get(lease.lease_id) is not lease:
            raise RuntimeError("request transition lease is unknown, replaced or released")

    def _require_admission(self, batch: RequestStateAdmissionBatch) -> None:
        if self._pending is not batch:
            raise RuntimeError("request admission is not the issued pending batch")
