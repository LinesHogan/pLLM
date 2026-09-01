"""Representation-independent global lineage and exact request leases."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.state_ref import LayoutId, LineageId, StateRef, VersionId


@dataclass(frozen=True, slots=True)
class GlobalStateAdmission:
    operation_id: int
    ref: StateRef


@dataclass(frozen=True, slots=True)
class GlobalStateTransition:
    operation_id: int
    before: StateRef
    after: StateRef


@dataclass(frozen=True, slots=True)
class GlobalVersionLease:
    lease_id: int
    identity: RequestIdentity
    ref: StateRef


@dataclass(frozen=True, slots=True)
class _Empty:
    pass


@dataclass(frozen=True, slots=True)
class _Ready:
    latest: StateRef


@dataclass(frozen=True, slots=True)
class _AdmissionPending:
    admission: GlobalStateAdmission


@dataclass(frozen=True, slots=True)
class _TransitionPending:
    transition: GlobalStateTransition


_GlobalState = _Empty | _Ready | _AdmissionPending | _TransitionPending


class GlobalVersionDirectory:
    """Own one logical lineage while requests retain immutable exact versions."""

    __slots__ = (
        "_layout_id",
        "_leases",
        "_lock",
        "_next_lease_id",
        "_next_operation_id",
        "_state",
    )

    def __init__(self, layout_id: LayoutId) -> None:
        self._layout_id = layout_id
        self._state: _GlobalState = _Empty()
        self._leases: dict[RequestIdentity, GlobalVersionLease] = {}
        self._next_operation_id = 1
        self._next_lease_id = 1
        self._lock = RLock()

    @property
    def active_lease_count(self) -> int:
        with self._lock:
            return len(self._leases)

    def prepare_admission(self) -> GlobalStateAdmission:
        with self._lock:
            match self._state:
                case _Empty():
                    pass
                case _Ready() | _AdmissionPending() | _TransitionPending():
                    raise RuntimeError("global state is already admitted")
            admission = GlobalStateAdmission(
                self._take_operation_id(),
                StateRef(self._layout_id, LineageId(1), VersionId(0)),
            )
            self._state = _AdmissionPending(admission)
            return admission

    def commit_admission(self, admission: GlobalStateAdmission) -> None:
        with self._lock:
            match self._state:
                case _AdmissionPending(admission=issued) if issued is admission:
                    self._state = _Ready(admission.ref)
                case _:
                    raise RuntimeError("global admission is not pending")

    def abort_admission(self, admission: GlobalStateAdmission) -> None:
        with self._lock:
            match self._state:
                case _AdmissionPending(admission=issued) if issued is admission:
                    self._state = _Empty()
                case _:
                    raise RuntimeError("global admission is not pending")

    def latest(self) -> StateRef:
        with self._lock:
            match self._state:
                case _Ready(latest=latest):
                    return latest
                case _Empty() | _AdmissionPending() | _TransitionPending():
                    raise RuntimeError("global state has no available latest version")

    def pin_latest(self, identity: RequestIdentity) -> GlobalVersionLease:
        with self._lock:
            try:
                return self._leases[identity]
            except KeyError:
                pass
            latest = self.latest()
            lease = GlobalVersionLease(self._next_lease_id, identity, latest)
            self._next_lease_id += 1
            self._leases[identity] = lease
            return lease

    def release(self, lease: GlobalVersionLease) -> None:
        with self._lock:
            try:
                issued = self._leases[lease.identity]
            except KeyError as exc:
                raise RuntimeError("global version lease is unknown or released") from exc
            if issued is not lease:
                raise RuntimeError("global version lease differs from issued lease")
            self._leases.pop(lease.identity)

    def prepare_transition(self) -> GlobalStateTransition:
        with self._lock:
            before = self.latest()
            transition = GlobalStateTransition(
                self._take_operation_id(),
                before,
                StateRef(
                    before.layout_id,
                    before.lineage_id,
                    VersionId(int(before.version_id) + 1),
                ),
            )
            self._state = _TransitionPending(transition)
            return transition

    def validate_transition_publication(
        self,
        transition: GlobalStateTransition,
    ) -> None:
        with self._lock:
            match self._state:
                case _TransitionPending(transition=issued) if issued is transition:
                    pass
                case _:
                    raise RuntimeError("global transition is not pending")

    def commit_transition(self, transition: GlobalStateTransition) -> None:
        with self._lock:
            self.validate_transition_publication(transition)
            self._state = _Ready(transition.after)

    def abort_transition(self, transition: GlobalStateTransition) -> None:
        with self._lock:
            self.validate_transition_publication(transition)
            self._state = _Ready(transition.before)

    def _take_operation_id(self) -> int:
        operation_id = self._next_operation_id
        self._next_operation_id += 1
        return operation_id


__all__ = [
    "GlobalStateAdmission",
    "GlobalStateTransition",
    "GlobalVersionDirectory",
    "GlobalVersionLease",
]
