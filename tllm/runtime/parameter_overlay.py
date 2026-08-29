"""Immutable versioned parameter overlays with transactional publication."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from types import MappingProxyType

import torch

from tllm.contracts.request_identity import RequestIdentity


class OverlayRepresentation(StrEnum):
    DENSE_REPLACE = "dense_replace"
    LOW_RANK_ADD = "low_rank_add"


class OverlayScope(StrEnum):
    GLOBAL = "global"
    REQUEST = "request"


@dataclass(frozen=True, slots=True)
class OverlayChannel:
    namespace: str
    name: str


@dataclass(frozen=True, slots=True)
class OverlayLineage:
    channel: OverlayChannel
    request: RequestIdentity | None = None


@dataclass(frozen=True, slots=True)
class OverlayVersionRef:
    lineage: OverlayLineage
    epoch: int
    version: int


@dataclass(frozen=True, slots=True)
class OverlayManifest:
    representation: OverlayRepresentation
    tensor_names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]


class OverlayPayload:
    """Privately owned tensors; every external materialization is a clone."""

    __slots__ = ("representation", "_tensors")

    def __init__(
        self,
        representation: OverlayRepresentation,
        tensors: Mapping[str, torch.Tensor],
    ) -> None:
        self.representation = representation
        self._tensors = {
            name: tensor.detach().to(device="cpu", copy=True).contiguous()
            for name, tensor in sorted(tensors.items())
        }

    @property
    def manifest(self) -> OverlayManifest:
        names = tuple(self._tensors)
        return OverlayManifest(
            representation=self.representation,
            tensor_names=names,
            shapes=tuple(tuple(self._tensors[name].shape) for name in names),
            dtypes=tuple(str(self._tensors[name].dtype) for name in names),
        )

    @property
    def byte_size(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self._tensors.values()
        )

    def materialize(self) -> Mapping[str, torch.Tensor]:
        return MappingProxyType(
            {
                name: tensor.clone(memory_format=torch.contiguous_format)
                for name, tensor in self._tensors.items()
            }
        )


@dataclass(frozen=True, slots=True)
class OverlayPublication:
    lineage: OverlayLineage
    epoch: int
    payload: OverlayPayload


@dataclass(frozen=True, slots=True)
class OverlayVersion:
    ref: OverlayVersionRef
    manifest: OverlayManifest
    byte_size: int


@dataclass(frozen=True, slots=True)
class OverlayLease:
    request: RequestIdentity
    ref: OverlayVersionRef


@dataclass(frozen=True, slots=True)
class _PendingPublication:
    transaction: str
    publication: OverlayPublication


@dataclass(slots=True)
class _StoredVersion:
    version: OverlayVersion
    payload: OverlayPayload
    leases: int = 0


class OverlayVersionStore:
    """Atomic multi-tensor publication and request-stable version leases."""

    def __init__(self, *, max_versions: int = 0, max_bytes: int = 0) -> None:
        self.max_versions = max_versions
        self.max_bytes = max_bytes
        self._pending: dict[str, _PendingPublication] = {}
        self._versions: dict[OverlayVersionRef, _StoredVersion] = {}
        self._latest: dict[OverlayLineage, OverlayVersionRef] = {}
        self._next_versions: dict[OverlayLineage, int] = {}
        self._leases: dict[RequestIdentity, OverlayLease] = {}
        self._bytes = 0
        self._lock = RLock()

    @property
    def byte_size(self) -> int:
        with self._lock:
            return self._bytes

    def stage(self, transaction: str, publication: OverlayPublication) -> None:
        with self._lock:
            if transaction in self._pending:
                raise RuntimeError("overlay publication transaction is already staged")
            self._pending[transaction] = _PendingPublication(transaction, publication)

    def commit(self, transaction: str) -> OverlayVersion:
        with self._lock:
            pending = self._pending[transaction]
            publication = pending.publication
            next_version = self._next_versions.get(publication.lineage, 0) + 1
            ref = OverlayVersionRef(
                publication.lineage, publication.epoch, next_version
            )
            payload = OverlayPayload(
                publication.payload.representation,
                publication.payload.materialize(),
            )
            version = OverlayVersion(ref, payload.manifest, payload.byte_size)
            prior_latest = self._latest.get(publication.lineage)
            removable = (
                ()
                if prior_latest is None or self._versions[prior_latest].leases
                else (prior_latest,)
            )
            prospective_count = len(self._versions) + 1 - len(removable)
            prospective_bytes = (
                self._bytes
                + payload.byte_size
                - sum(self._versions[item].version.byte_size for item in removable)
            )
            if self.max_versions and prospective_count > self.max_versions:
                raise RuntimeError(
                    "overlay version capacity is exhausted by live versions"
                )
            if self.max_bytes and prospective_bytes > self.max_bytes:
                raise RuntimeError(
                    "overlay byte capacity is exhausted by live versions"
                )
            self._pending.pop(transaction)
            for removable_ref in removable:
                self._versions.pop(removable_ref)
            self._versions[ref] = _StoredVersion(version, payload)
            self._latest[publication.lineage] = ref
            self._next_versions[publication.lineage] = next_version
            self._bytes = prospective_bytes
            self._collect_unleased()
            return version

    def abort(self, transaction: str) -> None:
        with self._lock:
            self._pending.pop(transaction)

    def latest(self, lineage: OverlayLineage) -> OverlayVersion | None:
        with self._lock:
            ref = self._latest.get(lineage)
            return None if ref is None else self._versions[ref].version

    def version(self, ref: OverlayVersionRef) -> OverlayVersion:
        with self._lock:
            return self._versions[ref].version

    def materialize(self, ref: OverlayVersionRef) -> Mapping[str, torch.Tensor]:
        with self._lock:
            return self._versions[ref].payload.materialize()

    def pin_latest(
        self, request: RequestIdentity, lineage: OverlayLineage
    ) -> OverlayLease | None:
        with self._lock:
            existing = self._leases.get(request)
            if existing is not None:
                return existing
            ref = self._latest.get(lineage)
            if ref is None:
                return None
            return self._pin(request, ref)

    def pin_exact(
        self, request: RequestIdentity, ref: OverlayVersionRef
    ) -> OverlayLease:
        with self._lock:
            existing = self._leases.get(request)
            if existing is not None:
                if existing.ref != ref:
                    raise RuntimeError("request already pins another overlay version")
                return existing
            return self._pin(request, ref)

    def _pin(
        self, request: RequestIdentity, ref: OverlayVersionRef
    ) -> OverlayLease:
        stored = self._versions[ref]
        lease = OverlayLease(request, ref)
        stored.leases += 1
        self._leases[request] = lease
        return lease

    def release(self, request: RequestIdentity) -> bool:
        with self._lock:
            lease = self._leases.pop(request, None)
            if lease is None:
                return False
            self._versions[lease.ref].leases -= 1
            self._collect_unleased()
            return True

    def advance_request(
        self, request: RequestIdentity, ref: OverlayVersionRef
    ) -> OverlayLease:
        """Move an explicit request lineage to a newly committed version."""

        with self._lock:
            existing = self._leases.get(request)
            if existing is not None:
                self._versions[existing.ref].leases -= 1
            lease = OverlayLease(request, ref)
            self._versions[ref].leases += 1
            self._leases[request] = lease
            self._collect_unleased()
            return lease

    def pinned(self, request: RequestIdentity) -> OverlayLease | None:
        with self._lock:
            return self._leases.get(request)

    def versions(self, lineage: OverlayLineage) -> tuple[OverlayVersion, ...]:
        with self._lock:
            return tuple(
                stored.version
                for ref, stored in sorted(
                    self._versions.items(), key=lambda item: item[0].version
                )
                if ref.lineage == lineage
            )

    def lineages(self) -> tuple[OverlayLineage, ...]:
        with self._lock:
            return tuple(
                sorted(
                    self._latest,
                    key=lambda lineage: (
                        lineage.channel.namespace,
                        lineage.channel.name,
                        "" if lineage.request is None else lineage.request.request_id,
                        -1 if lineage.request is None else lineage.request.sample_idx,
                    ),
                )
            )

    def _collect_unleased(self) -> None:
        latest_refs = frozenset(self._latest.values())
        removable = tuple(
            ref
            for ref, stored in self._versions.items()
            if ref not in latest_refs and stored.leases == 0
        )
        for ref in removable:
            stored = self._versions.pop(ref)
            self._bytes -= stored.version.byte_size

__all__ = [
    "OverlayChannel",
    "OverlayLease",
    "OverlayLineage",
    "OverlayManifest",
    "OverlayPayload",
    "OverlayPublication",
    "OverlayRepresentation",
    "OverlayScope",
    "OverlayVersion",
    "OverlayVersionRef",
    "OverlayVersionStore",
]
