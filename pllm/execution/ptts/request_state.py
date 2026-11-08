"""Map session request keys to resource-qualified lineages in the PageBundle arena.

Request state is touched only by the thread that owns the arena and has no lock; see StateExecution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import cached_property
from types import MappingProxyType

import numpy as np
import torch

from pllm import _core
from pllm.contracts.packed_rows import PackedRows, identity_words
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.semantic_graph import SemanticState, StateResource, StateScope
from pllm.contracts.state_ref import LineageId, StateRef
from pllm.execution.ptts.page_bundle import PageBundleArena, PagePin, PagePublication
from pllm.execution.ptts.state_refs import Indices, StateRefs


@dataclass(frozen=True, slots=True)
class StateAddress:
    resource: StateResource
    key: RequestStateKey | None


@dataclass(frozen=True, slots=True, eq=False)
class ResourceSelection:
    """One resource's exact refs and numeric row-to-state projection."""

    resource: StateResource
    key_table: Mapping[int, RequestStateKey | None]
    refs: StateRefs
    # Indices into the containing selection's flat refs, aligned to its rows.
    state_indices: Indices

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ResourceSelection)
            and self.resource == other.resource
            and self.refs == other.refs
            and self.keys == other.keys
            and np.array_equal(self.state_indices, other.state_indices)
        )

    @property
    def instance(self) -> bool:
        """Whether admission bound this resource to instance scope."""
        return len(self.key_table) == 1 and next(iter(self.key_table.values())) is None

    @property
    def keys(self) -> tuple[RequestStateKey | None, ...]:
        """Materialize resource keys only for inspection and explicit state imports."""
        return tuple(self.key_table[int(lineage)] for lineage in self.refs.lineages)


@dataclass(frozen=True)
class RequestStateSelection:
    rows: PackedRows
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
            (replace(group, state_indices=group.refs.positions(self.refs.take(group.state_indices))),),
        )

    def _sole_group(self) -> ResourceSelection:
        if len(self.groups) != 1:
            raise ValueError("name a resource when accessing multiple states")
        return self.groups[0]

    @property
    def keys(self) -> tuple[RequestStateKey | None, ...]:
        return self._sole_group().keys if self.groups else ()

    @property
    def state_indices(self) -> Indices:
        return self._sole_group().state_indices if self.groups else np.empty(0, dtype=np.int64)

    @property
    def instance(self) -> bool:
        """Whether the sole resource has one instance-scoped state."""
        return bool(self.groups) and self._sole_group().instance

    @cached_property
    def row_states(self) -> Mapping[RequestIdentity, int]:
        return MappingProxyType({row.identity: index for row, index in zip(self.rows, self.state_indices, strict=True)})

    def ref_for(self, identity: RequestIdentity, resource: StateResource | None = None) -> StateRef:
        group = self._sole_group() if resource is None else self.for_resource(resource)
        for row, index in zip(self.rows, group.state_indices, strict=True):
            if row.identity == identity:
                return self.refs[int(index)]
        raise KeyError(identity)


class RequestStateDirectory:
    """One resource-qualified key-to-lineage map; versions remain in the arena."""

    def __init__(self, arena: PageBundleArena, states: tuple[SemanticState, ...]) -> None:
        self.arena, self.states = arena, states
        self._lineages: dict[StateAddress, LineageId] = {}
        self._identity_holds: dict[tuple[StateResource, RequestIdentity], StateAddress] = {}
        self._inference_pins: dict[RequestIdentity, PagePin] = {}
        fields = {
            RequestKeyField.REQUEST_ID: 1,
            RequestKeyField.PROMPT_INDEX: 2,
            RequestKeyField.SAMPLE_INDEX: 4,
            RequestKeyField.ROUTE: 8,
        }
        self._projection = _core.ResourceProjection(
            np.array([sum(fields[field] for field in state.key_fields) for state in states], dtype=np.int64)
        )
        self._keys: list[dict[int, RequestStateKey | None]] = [{} for _ in states]

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
                    initialize(self.arena.writable_roots(write, initialize=True), new)
                lineages = dict(self._lineages)
                for key, ref in zip(new, write.after_refs, strict=True):
                    lineages[key] = ref.lineage_id
                receipt = self.arena.publish(write)
                self._lineages = lineages
        except Exception:
            for _, pin in pins:
                self.arena.release_pin(pin)
            if write is not None:
                self.arena.abort(write)
            raise
        for identity, key in holds:
            self._identity_holds[key.resource, identity] = key
        self._inference_pins.update(pins)
        self._bind(identities)
        return receipt

    def _bind(self, identities: tuple[RequestIdentity, ...]) -> None:
        """Install numeric projection bindings at admission, including shared route keys."""
        for resource, state in enumerate(self.states):
            keys: dict[int, RequestStateKey | None] = (
                {} if state.scope is StateScope.INSTANCE else self._keys[resource].copy()
            )
            for identity in identities if state.scope is StateScope.REQUEST else (None,):
                key = None if identity is None else RequestStateKey.project(identity, state.key_fields)
                address = StateAddress(state.resource, key)
                ref = self.arena.resolve_lineages(state.resource, (self._lineages[address],))
                assert ref.rows is not None
                words = np.empty(0, dtype=np.int64) if identity is None else identity_words(identity)
                lineage = int(ref.lineages[0])
                keys[lineage] = key
                self._projection.bind(words, resource, int(ref.layouts[0]), lineage, int(ref.rows[0]))
            self._keys[resource] = keys

    def latest_refs(self, identities: tuple[RequestIdentity, ...]) -> tuple[StateRef, ...]:
        return tuple(self.arena.latest(self._lineages[key]) for key in self._addresses(identities))

    def select(self, rows: PackedRows, history: tuple[StateRefs, ...] = ()) -> RequestStateSelection:
        """Project numeric rows and select exact versions, rewinding inputs for declared lag."""
        batch = rows.batch
        projected = self._projection.project(batch.identity_data, batch.identity_offsets, rows.identity_indices)
        groups: list[ResourceSelection] = []
        for state, keys, (columns, positions) in zip(self.states, self._keys, projected, strict=True):
            refs = self.arena.latest_batch(state.resource, columns)
            if history:
                refs = self.rewind_versions(refs, history)
            groups.append(ResourceSelection(state.resource, keys, refs, positions))
        return RequestStateSelection(rows if self.states else rows[:0], tuple(groups))

    @staticmethod
    def rewind_versions(refs: StateRefs, history: tuple[StateRefs, ...]) -> StateRefs:
        """Select the historical state versions visible at the requested publication lag."""
        versions = _core.rewind_versions(
            refs.lineages,
            refs.versions,
            [before.lineages for before in history],
            [before.versions for before in history],
        )
        return StateRefs(refs.layouts, refs.lineages, versions, refs.rows)

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
            self._forget_key(key)
        return tuple(dict.fromkeys((*refs, *(ref for pin in pins for ref in pin.refs))))

    def _forget_key(self, address: StateAddress) -> None:
        lineage = self._lineages.pop(address)
        for resource, state in enumerate(self.states):
            if state.resource == address.resource:
                self._projection.retire(resource, int(lineage))
                keys = self._keys[resource].copy()
                del keys[lineage]
                self._keys[resource] = keys
                break

    def retire_dormant(self) -> tuple[StateRef, ...]:
        held = set(self._identity_holds.values())
        dormant = tuple(key for key in self._lineages if key.key is not None and key not in held)
        refs = tuple(self.arena.latest(self._lineages[key]) for key in dormant)
        if refs:
            self.arena.retire(StateRefs.of(refs))
        for key in dormant:
            self._forget_key(key)
        return refs
