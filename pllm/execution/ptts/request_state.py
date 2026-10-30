"""Map session request keys to resource-qualified lineages in the PageBundle arena.

Request state is touched only by the thread that owns the arena and has no lock; see StateExecution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import cached_property
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
import torch

from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.semantic_graph import SemanticState, StateResource, StateScope
from pllm.contracts.state_ref import LineageId, StateRef
from pllm.execution.ptts.page_bundle import PageBundleArena, PagePin, PagePublication
from pllm.execution.ptts.state_refs import StateRefs


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

    def scoped(self, resource: StateResource) -> RequestStateSelection:
        """Project one resource, renumbering rows into that resource's roots."""
        group = self.for_resource(resource)
        return RequestStateSelection(
            self.rows,
            (replace(group, state_indices=tuple(group.refs.index(self.refs[index]) for index in group.state_indices)),),
        )

    def _sole_group(self) -> ResourceSelection:
        if len(self.groups) != 1:
            raise ValueError("name a resource when accessing multiple states")
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

    def __init__(self, arena: PageBundleArena, states: tuple[SemanticState, ...]) -> None:
        self.arena, self.states = arena, states
        self._lineages: dict[StateAddress, LineageId] = {}
        self._identity_holds: dict[tuple[StateResource, RequestIdentity], StateAddress] = {}
        self._inference_pins: dict[RequestIdentity, PagePin] = {}
        self._projection: tuple[tuple[RequestIdentity, ...], tuple[_Projection, ...]] | None = None

    @property
    def state(self) -> SemanticState | None:
        if len(self.states) > 1:
            raise ValueError("name a resource when accessing multiple states")
        return self.states[0] if self.states else None

    @property
    def scope(self) -> StateScope | None:
        scopes = {s.scope for s in self.states}
        if len(scopes) > 1:
            raise ValueError("name a resource when accessing multiple states")
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

    def inference_pin(self, identity: RequestIdentity) -> PagePin:
        return self._inference_pins[identity]

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

    def admit(
        self,
        identities: tuple[RequestIdentity, ...],
        stream: torch.cuda.Stream | None,
        initialize: Callable[[tuple[tuple[torch.Tensor, ...], ...], tuple[StateAddress, ...]], None],
    ) -> PagePublication | None:
        """Initialize and publish new lineages, then install their holders and inference pins."""
        addresses = self._addresses(identities)
        if not identities:
            new = tuple(key for key in addresses if key.key is None)
            holds: tuple[tuple[RequestIdentity, StateAddress], ...] = ()
            instances = StateRefs.of(())
        else:
            new = tuple(key for key in addresses if key.key is not None and key not in self._lineages)
            holds = tuple(
                (identity, StateAddress(s.resource, RequestStateKey.project(identity, s.key_fields)))
                for s in self.request_states
                for identity in identities
            )
            instances = StateRefs.of(
                tuple(self.arena.latest(self._lineages[key]) for key in addresses if key.key is None)
            )
        write = self.arena.prepare_admission(tuple(str(key.resource) for key in new), stream) if new else None
        pins: list[tuple[RequestIdentity, PagePin]] = []
        receipt = None
        try:
            for identity in identities if instances else ():
                pins.append((identity, self.arena.pin_exact(instances, stream)))
            if write is not None:
                with nullcontext() if stream is None else torch.cuda.stream(stream):
                    initialize(self.arena.writable_roots(write), new)
                lineages = dict(self._lineages)
                for key, ref in zip(new, write.after_refs, strict=True):
                    lineages[key] = ref.lineage_id
                receipt = self.arena.publish(write)
                self._lineages = lineages
                self._projection = None
        except Exception:
            for _, pin in pins:
                self.arena.release_pin(pin)
            if write is not None:
                self.arena.abort(write)
            raise
        for identity, key in holds:
            self._identity_holds[key.resource, identity] = key
        self._inference_pins.update(pins)
        if receipt is not None:
            self.arena.settle(receipt)
        return receipt

    def latest_refs(self, identities: tuple[RequestIdentity, ...]) -> tuple[StateRef, ...]:
        return tuple(self.arena.latest(self._lineages[key]) for key in self._addresses(identities))

    def select(self, rows: tuple[PackedPhaseRow, ...], history: tuple[StateRefs, ...] = ()) -> RequestStateSelection:
        """Select exact versions, rewinding recent step inputs newest first when lag is requested."""
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
            lineages = tuple(self._lineages[StateAddress(state.resource, key)] for key in keys)
            projections.append(
                _Projection(state.resource, keys, self.arena.resolve_lineages(state.resource, lineages), positions)
            )
            offset += len(keys)
        return tuple(projections)

    def retire(self, identities: tuple[RequestIdentity, ...]) -> tuple[StateRef, ...]:
        addresses = self._addresses(identities)
        if not identities:
            releasing = tuple(key for key in addresses if key.key is None)
            pins: tuple[PagePin, ...] = ()
            leaving: tuple[StateAddress, ...] = ()
        else:
            pins = tuple(self._inference_pins[identity] for identity in identities) if self.instance_states else ()
            leaving = tuple(key for key in addresses if key.key is not None)
            releasing = tuple(
                key
                for key in leaving
                if all(identity in identities for (_, identity), held in self._identity_holds.items() if held == key)
                and key.key is not None
                and RequestKeyField.ROUTE not in key.key.fields
            )
        refs = tuple(self.arena.latest(self._lineages[key]) for key in releasing)
        if refs:
            self.arena.retire(StateRefs.of(refs))
        if pins:
            for identity, pin in zip(identities, pins, strict=True):
                self.arena.release_pin(pin)
                self._inference_pins.pop(identity)
        for state in self.request_states:
            for identity in identities:
                self._identity_holds.pop((state.resource, identity))
        for key in releasing:
            self._lineages.pop(key)
            self._projection = None
        return tuple(dict.fromkeys((*refs, *(ref for pin in pins for ref in pin.refs))))

    def retire_dormant(self) -> tuple[StateRef, ...]:
        held = set(self._identity_holds.values())
        dormant = tuple(key for key in self._lineages if key.key is not None and key not in held)
        refs = tuple(self.arena.latest(self._lineages[key]) for key in dormant)
        if refs:
            self.arena.retire(StateRefs.of(refs))
        for key in dormant:
            self._lineages.pop(key)
            self._projection = None
        return refs
