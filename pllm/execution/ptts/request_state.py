"""Qualified request keys and atomic leases over the original PageBundle arena."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property
from threading import RLock
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
import torch

from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.semantic_graph import SemanticState, StateResource, StateScope
from pllm.contracts.state_ref import LineageId, StateRef
from pllm.execution.ptts.page_bundle import PageBundleArena, PagePin, PagePublication, PageWrite
from pllm.execution.ptts.state_refs import Indices, StateRefs


@dataclass(frozen=True, slots=True)
class StateAddress:
    resource: StateResource
    key: RequestStateKey | None


class _Projection(NamedTuple):
    """One state's resource, its keys and their lineages, and each row's position in the selection."""

    resource: StateResource
    keys: tuple[RequestStateKey | None, ...]
    lineages: StateRefs
    positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RequestStateAdmissionBatch:
    keys: tuple[StateAddress, ...]
    holds: tuple[tuple[RequestIdentity, StateAddress], ...]
    write: PageWrite | None
    inference_pins: tuple[tuple[RequestIdentity, PagePin], ...]


@dataclass(frozen=True, slots=True)
class ResourceSelection:
    resource: StateResource
    keys: tuple[RequestStateKey | None, ...]
    refs: StateRefs
    # Indices into the containing selection's flat refs, aligned to its rows.
    state_indices: tuple[int, ...]


@dataclass(frozen=True)
class RequestStateSelection:
    rows: tuple[PackedPhaseRow, ...]
    groups: tuple[ResourceSelection, ...]

    @cached_property
    def refs(self) -> StateRefs:
        return StateRefs.join(tuple(group.refs for group in self.groups))

    @cached_property
    def addresses(self) -> tuple[StateAddress, ...]:
        return tuple(StateAddress(group.resource, key) for group in self.groups for key in group.keys)

    def for_resource(self, resource: StateResource) -> ResourceSelection:
        for group in self.groups:
            if group.resource == resource:
                return group
        raise KeyError(resource)

    def _sole_group(self) -> ResourceSelection:
        if len(self.groups) != 1:
            raise ValueError("state access must name a resource in a multi-state selection")
        return self.groups[0]

    @property
    def keys(self) -> tuple[RequestStateKey | None, ...]:
        return self._sole_group().keys if self.groups else ()

    @property
    def state_indices(self) -> tuple[int, ...]:
        return self._sole_group().state_indices if self.groups else ()

    @cached_property
    def row_states(self) -> Mapping[RequestIdentity, int]:
        return MappingProxyType({row.identity: index for row, index in zip(self.rows, self.state_indices, strict=True)})

    def indices(self, rows: tuple[PackedPhaseRow, ...]) -> tuple[int, ...]:
        """Reuse the directory's projection when a tensor invocation covers the selected rows."""
        if rows == self.rows:
            return self.state_indices
        return tuple(self.row_states[row.identity] for row in rows)

    def ref_for(self, identity: RequestIdentity, resource: StateResource | None = None) -> StateRef:
        group = self._sole_group() if resource is None else self.for_resource(resource)
        for row, index in zip(self.rows, group.state_indices, strict=True):
            if row.identity == identity:
                return self.refs[index]
        raise KeyError(identity)


class RequestStateDirectory:
    """One resource-qualified key-to-lineage map; versions remain in the arena."""

    def __init__(self, arena: PageBundleArena, state: SemanticState | tuple[SemanticState, ...] | None) -> None:
        states = state if isinstance(state, tuple) else () if state is None else (state,)
        if (
            len(set(s.resource for s in states)) != len(states)
            or any(s not in arena.plan.semantic_graph.states for s in states)
            or not states
            and arena.plan.semantic_graph.states
        ):
            raise ValueError("directory requires its original semantic states, or a stateless graph")
        self.arena, self.states = arena, states
        self._lineages: dict[StateAddress, LineageId] = {}
        self._holders: dict[StateAddress, set[RequestIdentity]] = {}
        self._identity_holds: dict[tuple[StateResource, RequestIdentity], StateAddress] = {}
        self._inference_pins: dict[RequestIdentity, PagePin] = {}
        self._leases: dict[int, RequestStateSelection] = {}
        self._pending: RequestStateAdmissionBatch | None = None
        self._projection: tuple[tuple[RequestIdentity, ...], tuple[_Projection, ...]] | None = None
        self._lock = RLock()

    @property
    def state(self) -> SemanticState | None:
        if len(self.states) > 1:
            raise ValueError("directory state access requires a resource name")
        return self.states[0] if self.states else None

    @property
    def scope(self) -> StateScope | None:
        scopes = {s.scope for s in self.states}
        if len(scopes) > 1:
            raise ValueError("mixed state scopes require resource-specific access")
        return next(iter(scopes), None)

    @property
    def request_states(self) -> tuple[SemanticState, ...]:
        return tuple(s for s in self.states if s.scope is StateScope.REQUEST)

    @property
    def instance_states(self) -> tuple[SemanticState, ...]:
        return tuple(s for s in self.states if s.scope is StateScope.INSTANCE)

    @property
    def active_count(self) -> int:
        return len(self._lineages)

    @property
    def inference_pin_count(self) -> int:
        return len(self._inference_pins)

    def inference_pin(self, identity: RequestIdentity, stream: torch.cuda.Stream | None = None) -> PagePin:
        with self._lock:
            pin = self._inference_pins[identity]
            if pin.stream != stream:
                raise ValueError("inference must use its original admission pin stream")
            return pin

    def has_inference_pin(self, identity: RequestIdentity) -> bool:
        return identity in self._inference_pins

    def _addresses(self, identities: tuple[RequestIdentity, ...]) -> tuple[StateAddress, ...]:
        return tuple(
            StateAddress(state.resource, key)
            for state in self.states
            for key in (
                (None,)
                if state.scope is StateScope.INSTANCE
                else tuple(
                    dict.fromkeys(RequestStateKey.project(identity, state.key_fields) for identity in identities)
                )
            )
        )

    def prepare_admission(
        self, identities: tuple[RequestIdentity, ...], stream: torch.cuda.Stream | None = None
    ) -> RequestStateAdmissionBatch:
        with self._lock:
            if not self.states:
                raise RuntimeError("stateless graph has no state to admit")
            if self._pending is not None:
                raise RuntimeError("request admission is already pending")
            if len(set(identities)) != len(identities):
                raise RuntimeError("request admission requires distinct unadmitted requests")
            addresses = self._addresses(identities)
            if not identities:
                new = tuple(key for key in addresses if key.key is None)
                if not new or any(key in self._lineages for key in new):
                    raise RuntimeError("instance admission requires unadmitted keys")
                holds: tuple[tuple[RequestIdentity, StateAddress], ...] = ()
                instances: tuple[StateRef, ...] = ()
            else:
                if any(
                    (s.resource, identity) in self._identity_holds
                    for s in self.request_states
                    for identity in identities
                ):
                    raise RuntimeError("request admission requires distinct unadmitted requests")
                new = tuple(key for key in addresses if key.key is not None and key not in self._lineages)
                holds = tuple(
                    (identity, StateAddress(s.resource, RequestStateKey.project(identity, s.key_fields)))
                    for s in self.request_states
                    for identity in identities
                )
                instances = tuple(self._latest(key) for key in addresses if key.key is None)
                if instances and any(identity in self._inference_pins for identity in identities):
                    raise ValueError("instance inference admission requires distinct new requests")
            write = self.arena.prepare_admission(tuple(str(key.resource) for key in new), stream) if new else None
            pins: list[tuple[RequestIdentity, PagePin]] = []
            try:
                for identity in identities if instances else ():
                    pins.append((identity, self.arena.pin_exact(instances, stream)))
            except Exception:
                for _, pin in pins:
                    self.arena.release_pin(pin)
                if write is not None:
                    self.arena.abort(write)
                raise
            batch = RequestStateAdmissionBatch(new, holds, write, tuple(pins))
            self._pending = batch
            return batch

    def commit_admission(self, batch: RequestStateAdmissionBatch) -> PagePublication | None:
        with self._lock:
            self._require_admission(batch)
            receipt = None
            if batch.write is not None:
                lineages = dict(self._lineages)
                for key, ref in zip(batch.keys, batch.write.after_refs, strict=True):
                    lineages[key] = ref.lineage_id
                receipt = self.arena.publish(batch.write)
                self._lineages = lineages
                self._projection = None
            for identity, key in batch.holds:
                self._identity_holds[key.resource, identity] = key
                self._holders.setdefault(key, set()).add(identity)
            self._inference_pins.update(batch.inference_pins)
            self._pending = None
            return receipt

    def abort_admission(self, batch: RequestStateAdmissionBatch) -> None:
        with self._lock:
            self._require_admission(batch)
            if batch.write is not None:
                self.arena.abort(batch.write)
            for _, pin in batch.inference_pins:
                self.arena.release_pin(pin)
            self._pending = None

    def _latest(self, key: StateAddress) -> StateRef:
        return self.arena.latest(self._lineage(key))

    def _lineage(self, key: StateAddress) -> LineageId:
        try:
            return self._lineages[key]
        except KeyError as exc:
            message = (
                "instance state is not admitted"
                if key.key is None
                else "request selection references an unadmitted key"
            )
            raise RuntimeError(message) from exc

    def latest_refs(self, identities: tuple[RequestIdentity, ...]) -> tuple[StateRef, ...]:
        with self._lock:
            return tuple(self._latest(key) for key in self._addresses(identities))

    def select(self, rows: tuple[PackedPhaseRow, ...], history: tuple[StateRefs, ...] = ()) -> RequestStateSelection:
        """Select exact versions, rewinding recent step inputs newest first when lag is requested."""
        with self._lock:
            if not rows and self.request_states and not self.instance_states:
                raise ValueError("request selection requires rows")
            identities = tuple(row.identity for row in rows)
            if self._projection is None or self._projection[0] != identities:
                self._projection = identities, self._project(rows)
            groups = tuple(
                ResourceSelection(resource, keys, self.arena.latest_batch(resource, lineages), positions)
                for resource, keys, lineages, positions in self._projection[1]
            )
            if history:
                groups = tuple(replace(group, refs=self._rewind(group.refs, history)) for group in groups)
            return RequestStateSelection(rows if self.states else (), groups)

    @staticmethod
    def _rewind(refs: StateRefs, history: tuple[StateRefs, ...]) -> StateRefs:
        versions = refs.versions.copy()
        for before in reversed(history):
            _, selected, previous = np.intersect1d(refs.lineages, before.lineages, return_indices=True)
            versions[selected] = before.versions[previous]
        return StateRefs(refs.layouts, refs.lineages, versions, refs.rows)

    def _project(self, rows: tuple[PackedPhaseRow, ...]) -> tuple[_Projection, ...]:
        """Each state's keys, their lineages, and every row's position in the selection.

        Decode batches repeat the same rows step after step, so `select` reuses this until the rows
        or the admitted lineages change; only the latest versions are read again.
        """
        projections: list[_Projection] = []
        offset = 0
        for state in self.states:
            if state.scope is StateScope.INSTANCE:
                keys: tuple[RequestStateKey | None, ...] = (None,)
                positions = (offset,) * len(rows)
            else:
                projected = tuple(RequestStateKey.project(row.identity, state.key_fields) for row in rows)
                keys = tuple(dict.fromkeys(projected))
                indices = {key: offset + i for i, key in enumerate(keys)}
                positions = tuple(indices[key] for key in projected)
            lineages = tuple(self._lineage(StateAddress(state.resource, key)) for key in keys)
            projections.append(
                _Projection(state.resource, keys, self.arena.resolve_lineages(state.resource, lineages), positions)
            )
            offset += len(keys)
        return tuple(projections)

    def select_and_lease(
        self, rows: tuple[PackedPhaseRow, ...], history: tuple[StateRefs, ...] = ()
    ) -> RequestStateSelection:
        """Lease lineages to one step, optionally selecting retained exact reader versions."""
        with self._lock:
            selection = self.select(rows, history)
            if self._overlaps(selection.refs.lineages):
                raise RuntimeError("request transition lease overlaps an active step")
            self._leases[id(selection)] = selection
            return selection

    def refresh_lease(self, selection: RequestStateSelection) -> RequestStateSelection:
        """Keep the same lineage lease while moving from lagged readers to latest writers."""
        with self._lock:
            latest = self.select(selection.rows)
            self.release_transition_lease(selection)
            self._leases[id(latest)] = latest
            return latest

    def release_transition_lease(self, selection: RequestStateSelection) -> None:
        with self._lock:
            if self._leases.get(id(selection)) is not selection:
                raise RuntimeError("request transition selection is not the active lease")
            del self._leases[id(selection)]

    def _overlaps(self, lineages: Indices) -> bool:
        return any(np.isin(lineages, lease.refs.lineages).any() for lease in self._leases.values())

    def _leased(self, addresses: tuple[StateAddress, ...]) -> bool:
        return self._overlaps(np.asarray([self._lineage(key) for key in addresses], dtype=np.int64))

    def retire(self, identities: tuple[RequestIdentity, ...]) -> tuple[StateRef, ...]:
        with self._lock:
            if len(set(identities)) != len(identities):
                raise ValueError("retirement requires distinct requests")
            addresses = self._addresses(identities)
            if not identities:
                if self._inference_pins:
                    raise RuntimeError("instance retirement requires all inference requests to finish")
                releasing = tuple(key for key in addresses if key.key is None)
                if not releasing:
                    raise RuntimeError("request retirement requires keys without an active step")
                pins: tuple[PagePin, ...] = ()
                leaving: tuple[StateAddress, ...] = ()
            else:
                if any(
                    (s.resource, identity) not in self._identity_holds
                    for s in self.request_states
                    for identity in identities
                ):
                    raise RuntimeError("request retirement references an unadmitted key")
                pins = tuple(self._inference_pins[identity] for identity in identities) if self.instance_states else ()
                leaving = tuple(key for key in addresses if key.key is not None)
                members = {
                    key: {identity for identity in identities if self._identity_holds[key.resource, identity] == key}
                    for key in leaving
                }
                releasing = tuple(
                    key
                    for key in leaving
                    if members[key] == self._holders.get(key, set())
                    and key.key is not None
                    and RequestKeyField.ROUTE not in key.key.fields
                )
            if self._leased(releasing):
                raise RuntimeError("request retirement requires keys without an active step")
            refs = tuple(self._latest(key) for key in releasing)
            if refs:
                self.arena.retire(refs)
            if pins:
                for identity, pin in zip(identities, pins, strict=True):
                    self.arena.release_pin(pin)
                    self._inference_pins.pop(identity)
            for state in self.request_states:
                for identity in identities:
                    key = self._identity_holds.pop((state.resource, identity))
                    self._holders[key].discard(identity)
            for key in releasing:
                self._lineages.pop(key)
                self._projection = None
                self._holders.pop(key, None)
            return tuple(dict.fromkeys((*refs, *(ref for pin in pins for ref in pin.refs))))

    def retire_dormant(self) -> tuple[StateRef, ...]:
        with self._lock:
            dormant = tuple(key for key, holders in self._holders.items() if not holders)
            if self._leased(dormant):
                raise RuntimeError("dormant retirement requires keys without an active step")
            refs = tuple(self._latest(key) for key in dormant)
            if refs:
                self.arena.retire(refs)
            for key in dormant:
                self._lineages.pop(key)
                self._projection = None
                self._holders.pop(key)
            return refs

    def _require_admission(self, batch: RequestStateAdmissionBatch) -> None:
        if self._pending is not batch:
            raise RuntimeError("request admission is not the issued pending batch")
