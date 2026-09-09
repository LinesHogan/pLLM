"""Request-key selection and exclusive transition leases over one PageBundle arena."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from collections.abc import Mapping
from threading import RLock

import torch

from pllm.execution.ptts.page_bundle import PageBundleArena, PagePin, PagePublication, PageWrite
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.semantic_graph import SemanticState, StateScope
from pllm.contracts.state_ref import LineageId, StateRef


@dataclass(frozen=True, slots=True)
class RequestStateAdmissionBatch:
    keys: tuple[RequestStateKey | None, ...]
    write: PageWrite


@dataclass(frozen=True)
class RequestStateSelection:
    rows: tuple[PackedPhaseRow, ...]
    keys: tuple[RequestStateKey | None, ...]
    refs: tuple[StateRef, ...]
    state_indices: tuple[int, ...]

    @cached_property
    def row_states(self) -> Mapping[RequestIdentity, int]:
        return MappingProxyType({row.identity: index for row, index in zip(self.rows, self.state_indices, strict=True)})

    def ref_for(self, identity: RequestIdentity) -> StateRef:
        return self.refs[self.row_states[identity]]


class RequestStateDirectory:
    """Own key-to-lineage relations, never another latest-version map."""

    def __init__(self, arena: PageBundleArena, state: SemanticState | None) -> None:
        if state not in arena.plan.semantic_graph.states and (state is not None or arena.plan.semantic_graph.states):
            raise ValueError("directory requires its original semantic state, or a stateless graph")
        self.arena = arena
        self.state = state
        self._lineages: dict[RequestStateKey | None, LineageId] = {}
        self._inference_pins: dict[RequestIdentity, PagePin] = {}
        self._leases: dict[RequestStateKey | None, RequestStateSelection] = {}
        self._pending: RequestStateAdmissionBatch | None = None
        self._lock = RLock()

    @property
    def scope(self) -> StateScope | None:
        return None if self.state is None else self.state.scope

    @property
    def active_count(self) -> int:
        return len(self._lineages)

    @property
    def inference_pin_count(self) -> int:
        return len(self._inference_pins)

    def admit_inference(self, identities: tuple[RequestIdentity, ...],
                        stream: torch.cuda.Stream | None = None) -> tuple[StateRef, ...]:
        """Fix each request to the original instance root visible at admission."""
        with self._lock:
            if (self.scope is not StateScope.INSTANCE or not identities
                or len(set(identities)) != len(identities)
                or any(identity in self._inference_pins for identity in identities)):
                raise ValueError("instance inference admission requires distinct new requests")
            refs = self.select(()).refs
            pending: dict[RequestIdentity, PagePin] = {}
            try:
                for identity in identities:
                    pending[identity] = self.arena.pin_exact(refs, stream)
            except Exception:
                for pin in pending.values():
                    self.arena.release_pin(pin)
                raise
            self._inference_pins.update(pending)
            return refs

    def inference_pin(self, identity: RequestIdentity, stream: torch.cuda.Stream | None = None) -> PagePin:
        with self._lock:
            pin = self._inference_pins[identity]
            if pin.stream != stream:
                raise ValueError("inference must use its original admission pin stream")
            return pin

    def has_inference_pin(self, identity: RequestIdentity) -> bool:
        return identity in self._inference_pins

    def retire_inference(self, identities: tuple[RequestIdentity, ...]) -> tuple[StateRef, ...]:
        with self._lock:
            if not identities or len(set(identities)) != len(identities):
                raise ValueError("inference retirement requires distinct requests")
            pins = tuple(self._inference_pins[identity] for identity in identities)
            for identity, pin in zip(identities, pins, strict=True):
                self.arena.release_pin(pin)
                self._inference_pins.pop(identity)
            return tuple(dict.fromkeys(ref for pin in pins for ref in pin.refs))

    def prepare_admission(self, identities: tuple[RequestIdentity, ...],
                          stream: torch.cuda.Stream | None = None) -> RequestStateAdmissionBatch:
        with self._lock:
            if self.state is None:
                raise RuntimeError("stateless graph has no state to admit")
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

    def select(self, rows: tuple[PackedPhaseRow, ...]) -> RequestStateSelection:
        with self._lock:
            if self.state is None:
                return RequestStateSelection((), (), (), ())
            if self.scope is StateScope.INSTANCE:
                try:
                    ref = self.arena.latest(self._lineages[None])
                except KeyError as exc:
                    raise RuntimeError("instance state is not admitted") from exc
                return RequestStateSelection(rows, (None,), (ref,), (0,) * len(rows))
            keys: list[RequestStateKey | None] = []
            refs: list[StateRef] = []
            state_indices: list[int] = []
            indices: dict[RequestStateKey, int] = {}
            for row in rows:
                key = RequestStateKey.project(row.identity, self.state.key_fields)
                if key not in indices:
                    try:
                        ref = self.arena.latest(self._lineages[key])
                    except KeyError as exc:
                        raise RuntimeError("request selection references an unadmitted key") from exc
                    indices[key] = len(keys)
                    keys.append(key)
                    refs.append(ref)
                state_indices.append(indices[key])
            if not rows:
                raise ValueError("request selection requires rows")
            return RequestStateSelection(rows, tuple(keys), tuple(refs), tuple(state_indices))

    def acquire_transition_lease(self, selection: RequestStateSelection) -> None:
        with self._lock:
            self._require_latest(selection)
            if self._leases.keys() & selection.keys:
                raise RuntimeError("request transition lease overlaps an active step")
            self._leases.update((key, selection) for key in selection.keys)

    def release_transition_lease(self, selection: RequestStateSelection) -> None:
        with self._lock:
            if any(self._leases.get(key) is not selection for key in selection.keys):
                raise RuntimeError("request transition selection is not the active lease")
            for key in selection.keys:
                self._leases.pop(key)

    def retire(self, identities: tuple[RequestIdentity, ...]) -> tuple[StateRef, ...]:
        with self._lock:
            if self.scope is StateScope.INSTANCE and self._inference_pins:
                raise RuntimeError("instance retirement requires all inference requests to finish")
            keys = self._unique_keys(identities)
            if not keys or self._leases.keys() & keys:
                raise RuntimeError("request retirement requires keys without an active step")
            try:
                refs = tuple(self.arena.latest(self._lineages[key]) for key in keys)
            except KeyError as exc:
                raise RuntimeError("request retirement references an unadmitted key") from exc
            lineages = {key: lineage for key, lineage in self._lineages.items() if key not in keys}
            self.arena.retire(refs)
            self._lineages = lineages
            return refs

    def _unique_keys(self, identities: tuple[RequestIdentity, ...]) -> tuple[RequestStateKey | None, ...]:
        if self.state is None:
            return ()
        if self.scope is StateScope.INSTANCE:
            if identities:
                raise ValueError("instance admission/retirement is not a request lifecycle action")
            return (None,)
        return tuple(dict.fromkeys(RequestStateKey.project(identity, self.state.key_fields) for identity in identities))

    def _require_latest(self, selection: RequestStateSelection) -> None:
        for key, ref in zip(selection.keys, selection.refs, strict=True):
            if self._lineages.get(key) != ref.lineage_id or self.arena.latest(ref.lineage_id) != ref:
                raise RuntimeError("request transition selection is stale")

    def _require_admission(self, batch: RequestStateAdmissionBatch) -> None:
        if self._pending is not batch:
            raise RuntimeError("request admission is not the issued pending batch")
