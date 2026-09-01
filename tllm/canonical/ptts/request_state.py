"""Representation-independent request ownership for canonical PTTS roots."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from tllm.contracts.request_identity import (
    RequestIdentity,
    RequestKeyField,
    canonical_request_key_fields,
)
from tllm.contracts.request_state import (
    IndexedRequestIdentityBatch,
    RequestStateKey,
)
from tllm.contracts.state_ref import LayoutId, LineageId, StateRef, VersionId


@dataclass(frozen=True, slots=True)
class RequestStateAdmission:
    key: RequestStateKey
    ref: StateRef


@dataclass(frozen=True, slots=True)
class RequestStateAdmissionBatch:
    operation_id: int
    admissions: tuple[RequestStateAdmission, ...]


@dataclass(frozen=True, slots=True)
class RequestStateSelectedRow:
    canonical_row_index: int
    identity: RequestIdentity
    key: RequestStateKey
    state_index: int

    def __post_init__(self) -> None:
        if self.canonical_row_index < 0 or self.state_index < 0:
            raise ValueError("request-state row indices must be non-negative")


@dataclass(frozen=True, slots=True)
class RequestStateSelection:
    selection_id: int
    rows: tuple[RequestStateSelectedRow, ...]
    keys: tuple[RequestStateKey, ...]
    refs: tuple[StateRef, ...]

    def __post_init__(self) -> None:
        if (
            self.selection_id < 1
            or not self.rows
            or len(self.keys) != len(self.refs)
            or not self.keys
            or len(self.keys) != len(set(self.keys))
            or tuple(row.canonical_row_index for row in self.rows)
            != tuple(sorted({row.canonical_row_index for row in self.rows}))
            or any(row.state_index >= len(self.keys) for row in self.rows)
            or any(
                self.keys[row.state_index] != row.key for row in self.rows
            )
        ):
            raise ValueError("request-state selection is not canonical")


@dataclass(frozen=True, slots=True)
class RequestStateTransitionLease:
    lease_id: int
    selection: RequestStateSelection


@dataclass(frozen=True, slots=True)
class RequestStateTransition:
    key: RequestStateKey
    before: StateRef
    after: StateRef


@dataclass(frozen=True, slots=True)
class RequestStateTransitionBatch:
    operation_id: int
    selection_id: int
    transitions: tuple[RequestStateTransition, ...]

    @property
    def before_refs(self) -> tuple[StateRef, ...]:
        return tuple(transition.before for transition in self.transitions)

    @property
    def after_refs(self) -> tuple[StateRef, ...]:
        return tuple(transition.after for transition in self.transitions)


@dataclass(frozen=True, slots=True)
class RequestStateRetirement:
    key: RequestStateKey
    ref: StateRef


@dataclass(frozen=True, slots=True)
class RequestStateRetirementBatch:
    operation_id: int
    retirements: tuple[RequestStateRetirement, ...]

    @property
    def refs(self) -> tuple[StateRef, ...]:
        return tuple(retirement.ref for retirement in self.retirements)


@dataclass(frozen=True, slots=True)
class _NoPendingOperation:
    pass


@dataclass(frozen=True, slots=True)
class _PendingAdmission:
    batch: RequestStateAdmissionBatch


@dataclass(frozen=True, slots=True)
class _PendingTransition:
    batch: RequestStateTransitionBatch


@dataclass(frozen=True, slots=True)
class _PendingRetirement:
    batch: RequestStateRetirementBatch


_PendingOperation = (
    _NoPendingOperation
    | _PendingAdmission
    | _PendingTransition
    | _PendingRetirement
)
_NO_PENDING_OPERATION = _NoPendingOperation()


class RequestStateDirectory:
    """Own request keys and exact latest roots without physical storage policy."""

    __slots__ = (
        "_layout_id",
        "_key_fields",
        "_latest",
        "_transition_leases",
        "_leased_keys",
        "_pending",
        "_next_lineage_id",
        "_next_lease_id",
        "_next_operation_id",
        "_next_selection_id",
        "_lock",
    )

    def __init__(
        self,
        layout_id: LayoutId,
        key_fields: tuple[RequestKeyField, ...],
    ) -> None:
        canonical_request_key_fields(key_fields)
        if type(layout_id) is not int or int(layout_id) < 0:
            raise ValueError("request-state directory requires canonical key fields")
        self._layout_id = layout_id
        self._key_fields = key_fields
        self._latest: dict[RequestStateKey, StateRef] = {}
        self._transition_leases: dict[int, RequestStateTransitionLease] = {}
        self._leased_keys: set[RequestStateKey] = set()
        self._pending: _PendingOperation = _NO_PENDING_OPERATION
        self._next_lineage_id = 1
        self._next_lease_id = 1
        self._next_operation_id = 1
        self._next_selection_id = 1
        self._lock = RLock()

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._latest)

    def prepare_admission(
        self,
        identities: tuple[RequestIdentity, ...],
    ) -> RequestStateAdmissionBatch:
        with self._lock:
            self._require_no_pending()
            keys = self._unique_keys(identities)
            if not keys:
                raise ValueError("request-state admission requires identities")
            if any(key in self._latest for key in keys):
                raise RuntimeError("request-state key is already admitted")
            admissions = tuple(
                RequestStateAdmission(
                    key,
                    StateRef(
                        self._layout_id,
                        LineageId(self._next_lineage_id + offset),
                        VersionId(0),
                    ),
                )
                for offset, key in enumerate(keys)
            )
            self._next_lineage_id += len(admissions)
            batch = RequestStateAdmissionBatch(
                self._take_operation_id(),
                admissions,
            )
            self._pending = _PendingAdmission(batch)
            return batch

    def commit_admission(self, batch: RequestStateAdmissionBatch) -> None:
        with self._lock:
            self._require_admission(batch)
            for admission in batch.admissions:
                self._latest[admission.key] = admission.ref
            self._pending = _NO_PENDING_OPERATION

    def abort_admission(self, batch: RequestStateAdmissionBatch) -> None:
        with self._lock:
            self._require_admission(batch)
            self._pending = _NO_PENDING_OPERATION

    def select(
        self,
        indexed_rows: IndexedRequestIdentityBatch,
    ) -> RequestStateSelection:
        with self._lock:
            self._require_no_pending()
            keys: list[RequestStateKey] = []
            refs: list[StateRef] = []
            rows: list[RequestStateSelectedRow] = []
            state_index_by_key: dict[RequestStateKey, int] = {}
            for indexed in indexed_rows.rows:
                key = RequestStateKey.project(indexed.identity, self._key_fields)
                try:
                    state_index = state_index_by_key[key]
                except KeyError:
                    try:
                        ref = self._latest[key]
                    except KeyError as exc:
                        raise RuntimeError(
                            "request-state selection references an unadmitted key"
                        ) from exc
                    state_index = len(keys)
                    state_index_by_key[key] = state_index
                    keys.append(key)
                    refs.append(ref)
                rows.append(
                    RequestStateSelectedRow(
                        indexed.canonical_row_index,
                        indexed.identity,
                        key,
                        state_index,
                    )
                )
            selection = RequestStateSelection(
                self._next_selection_id,
                tuple(rows),
                tuple(keys),
                tuple(refs),
            )
            self._next_selection_id += 1
            return selection

    def acquire_transition_lease(
        self,
        selection: RequestStateSelection,
    ) -> RequestStateTransitionLease:
        with self._lock:
            self._require_no_pending()
            self._require_latest_selection(selection)
            if any(key in self._leased_keys for key in selection.keys):
                raise RuntimeError(
                    "request-state transition lease overlaps an active step"
                )
            lease = RequestStateTransitionLease(
                self._next_lease_id,
                selection,
            )
            self._next_lease_id += 1
            self._transition_leases[lease.lease_id] = lease
            self._leased_keys.update(selection.keys)
            return lease

    def release_transition_lease(
        self,
        lease: RequestStateTransitionLease,
    ) -> None:
        with self._lock:
            self._require_no_pending()
            self._require_transition_lease(lease)
            self._transition_leases.pop(lease.lease_id)
            self._leased_keys.difference_update(lease.selection.keys)

    def prepare_transition(
        self,
        selection: RequestStateSelection,
        lease: RequestStateTransitionLease,
    ) -> RequestStateTransitionBatch:
        with self._lock:
            self._require_no_pending()
            self._require_transition_lease(lease)
            if lease.selection is not selection:
                raise RuntimeError(
                    "request-state transition uses another leased selection"
                )
            self._require_latest_selection(selection)
            batch = RequestStateTransitionBatch(
                self._take_operation_id(),
                selection.selection_id,
                tuple(
                    RequestStateTransition(
                        key,
                        ref,
                        StateRef(
                            ref.layout_id,
                            ref.lineage_id,
                            VersionId(int(ref.version_id) + 1),
                        ),
                    )
                    for key, ref in zip(
                        selection.keys,
                        selection.refs,
                        strict=True,
                    )
                ),
            )
            self._pending = _PendingTransition(batch)
            return batch

    def commit_transition(self, batch: RequestStateTransitionBatch) -> None:
        with self._lock:
            self._require_transition(batch)
            for transition in batch.transitions:
                self._latest[transition.key] = transition.after
            self._pending = _NO_PENDING_OPERATION

    def validate_transition_publication(
        self,
        batch: RequestStateTransitionBatch,
    ) -> None:
        """Prevalidate the logical half of a no-fail publication commit."""

        with self._lock:
            self._require_transition(batch)

    def abort_transition(self, batch: RequestStateTransitionBatch) -> None:
        with self._lock:
            self._require_transition(batch)
            self._pending = _NO_PENDING_OPERATION

    def prepare_retirement(
        self,
        identities: tuple[RequestIdentity, ...],
    ) -> RequestStateRetirementBatch:
        with self._lock:
            self._require_no_pending()
            keys = self._unique_keys(identities)
            if not keys:
                raise ValueError("request-state retirement requires identities")
            if any(key in self._leased_keys for key in keys):
                raise RuntimeError(
                    "request-state retirement overlaps an active step"
                )
            try:
                retirements = tuple(
                    RequestStateRetirement(key, self._latest[key]) for key in keys
                )
            except KeyError as exc:
                raise RuntimeError(
                    "request-state retirement references an unadmitted key"
                ) from exc
            batch = RequestStateRetirementBatch(
                self._take_operation_id(),
                retirements,
            )
            self._pending = _PendingRetirement(batch)
            return batch

    def commit_retirement(self, batch: RequestStateRetirementBatch) -> None:
        with self._lock:
            self._require_retirement(batch)
            for retirement in batch.retirements:
                self._latest.pop(retirement.key)
            self._pending = _NO_PENDING_OPERATION

    def abort_retirement(self, batch: RequestStateRetirementBatch) -> None:
        with self._lock:
            self._require_retirement(batch)
            self._pending = _NO_PENDING_OPERATION

    def _unique_keys(
        self,
        identities: tuple[RequestIdentity, ...],
    ) -> tuple[RequestStateKey, ...]:
        keys: list[RequestStateKey] = []
        seen: set[RequestStateKey] = set()
        for identity in identities:
            key = RequestStateKey.project(identity, self._key_fields)
            if key not in seen:
                keys.append(key)
                seen.add(key)
        return tuple(keys)

    def _take_operation_id(self) -> int:
        operation_id = self._next_operation_id
        self._next_operation_id += 1
        return operation_id

    def _require_latest_selection(
        self,
        selection: RequestStateSelection,
    ) -> None:
        for key, ref in zip(
            selection.keys,
            selection.refs,
            strict=True,
        ):
            try:
                latest = self._latest[key]
            except KeyError as exc:
                raise RuntimeError(
                    "request-state transition selection is stale"
                ) from exc
            if latest != ref:
                raise RuntimeError(
                    "request-state transition selection is stale"
                )

    def _require_transition_lease(
        self,
        lease: RequestStateTransitionLease,
    ) -> None:
        try:
            issued = self._transition_leases[lease.lease_id]
        except KeyError as exc:
            raise RuntimeError(
                "request-state transition lease is unknown or released"
            ) from exc
        if issued is not lease:
            raise RuntimeError(
                "request-state transition lease differs from the issued lease"
            )

    def _require_no_pending(self) -> None:
        match self._pending:
            case _NoPendingOperation():
                return
            case _:
                raise RuntimeError("request-state directory operation is pending")

    def _require_admission(self, batch: RequestStateAdmissionBatch) -> None:
        match self._pending:
            case _PendingAdmission(batch=issued) if issued is batch:
                return
            case _:
                raise RuntimeError("request-state admission is not pending")

    def _require_transition(self, batch: RequestStateTransitionBatch) -> None:
        match self._pending:
            case _PendingTransition(batch=issued) if issued is batch:
                return
            case _:
                raise RuntimeError("request-state transition is not pending")

    def _require_retirement(self, batch: RequestStateRetirementBatch) -> None:
        match self._pending:
            case _PendingRetirement(batch=issued) if issued is batch:
                return
            case _:
                raise RuntimeError("request-state retirement is not pending")


__all__ = [
    "RequestStateAdmission",
    "RequestStateAdmissionBatch",
    "RequestStateDirectory",
    "RequestStateRetirement",
    "RequestStateRetirementBatch",
    "RequestStateSelectedRow",
    "RequestStateSelection",
    "RequestStateTransition",
    "RequestStateTransitionBatch",
    "RequestStateTransitionLease",
]
