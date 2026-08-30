"""Adapter-neutral request alias ownership and finished-batch expansion.

Adapters resolve their private request identifiers before calling this ledger.
The ledger then retains every observed raw alias until its request family is
finished, so a parent or child completion always produces the same closed,
typed cleanup batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Protocol
from uuid import uuid4

from tllm.contracts.request_identity import (
    RequestFinishedBatch,
    RequestIdentity,
    resolve_request_identity,
)


class RequestLifecycleError(RuntimeError):
    """A request lifecycle event violates the registered alias contract."""


class UnknownRequestAliasError(RequestLifecycleError):
    """A completion cannot be associated with a registered raw alias."""


class DuplicateRequestFinishError(RequestLifecycleError):
    """A raw completion has already been consumed by this ledger."""


def new_request_finish_id() -> int:
    """Create one positive, wire-safe distributed transaction identity."""

    return max(1, uuid4().int >> 65)


@dataclass(frozen=True, slots=True)
class RequestAlias:
    """One adapter-observed raw id and its already-resolved identity."""

    raw_request_id: str
    identity: RequestIdentity


@dataclass(frozen=True, slots=True)
class RequestFinishTrigger:
    """One terminal raw id and the registered alias it resolved through."""

    raw_request_id: str
    alias: RequestAlias


@dataclass(frozen=True, slots=True)
class RequestFinishReceipt:
    """Auditable expansion of raw terminal events into typed cleanup state."""

    finish_id: int
    triggers: tuple[RequestFinishTrigger, ...]
    aliases: tuple[RequestAlias, ...]
    batch: RequestFinishedBatch


@dataclass(frozen=True, slots=True)
class RequestFinishOutcome:
    """The one committed typed cleanup batch and its signed-port count."""

    receipt: RequestFinishReceipt
    staged_ports: int


class RequestLifecycleRuntime(Protocol):
    """The closed runtime state needed by adapter-neutral lifecycle wiring."""

    request_lifecycle: "RequestLifecycleLedger"
    reqid_to_promptidx: dict[str, int]
    reqid_to_sampleidx: dict[str, int]


class RequestFinishCleanupService(Protocol):
    """Closed signed-cleanup service supplied by the backend boundary."""

    def stage(self, batch: RequestFinishedBatch) -> int:
        """Stage the typed batch on every signed request-finished port."""
        ...

    def confirm(self, batch: RequestFinishedBatch, staged_ports: int) -> None:
        """Drain, run, and confirm the corresponding signed cleanup."""
        ...


class RequestLifecycleLedger:
    """Own raw request aliases until their typed request family finishes.

    A request family is the closed ``RequestIdentity.request_id`` supplied by
    the adapter.  Thus a parent completion and any parallel-sample child
    completion expand over the same family without this class knowing an
    adapter's request-id grammar.  A transient EngineCore-style ``-<8 alnum>``
    suffix is accepted only when its unsuffixed raw alias was registered.
    """

    def __init__(self) -> None:
        self._active_by_raw_id: dict[str, RequestAlias] = {}
        self._finished_raw_ids: set[str] = set()
        self._pending: RequestFinishReceipt | None = None
        self._next_finish_id = 1
        self._lock = RLock()

    @property
    def active_aliases(self) -> tuple[RequestAlias, ...]:
        """Aliases still owned by live request families, in registration order."""

        with self._lock:
            return tuple(self._active_by_raw_id.values())

    @property
    def pending_finish(self) -> RequestFinishReceipt | None:
        """The one unfinalized finish receipt, if signed cleanup has begun."""

        with self._lock:
            return self._pending

    def register_alias(
        self,
        *,
        raw_request_id: str,
        identity: RequestIdentity,
    ) -> RequestAlias:
        """Record a raw alias after its adapter-local identity has been resolved.

        Re-observing the same live mapping is idempotent.  A terminal raw id
        cannot be registered again, preventing raw-id ABA reuse within one
        ledger lifetime.
        """

        with self._lock:
            self._validate_identity(identity)
            if not raw_request_id:
                raise ValueError("request alias raw_request_id must be non-empty")
            if raw_request_id in self._finished_raw_ids:
                raise DuplicateRequestFinishError(
                    f"request alias {raw_request_id!r} was already finished"
                )

            registered = self._active_by_raw_id.get(raw_request_id)
            if registered is None:
                alias = RequestAlias(raw_request_id=raw_request_id, identity=identity)
                self._active_by_raw_id[raw_request_id] = alias
                return alias
            if registered.identity != identity:
                raise RequestLifecycleError(
                    f"request alias {raw_request_id!r} was resolved inconsistently"
                )
            return registered

    def prepare_finish(
        self,
        raw_request_ids: tuple[str, ...],
        *,
        finish_id: int | None = None,
    ) -> RequestFinishReceipt:
        """Prepare a typed cleanup batch without releasing any alias.

        The caller must complete signed state/overlay cleanup and then call
        :meth:`commit_finish`.  Until that acknowledgement, every mapping
        remains active and can still prove the cleanup transaction.
        """

        with self._lock:
            if self._pending is not None:
                raise RequestLifecycleError(
                    "a request finish transaction is already pending"
                )
            self._validate_distinct_raw_ids(raw_request_ids)
            triggers = tuple(
                RequestFinishTrigger(
                    raw_request_id=raw_request_id,
                    alias=self._resolve_finish_alias(raw_request_id),
                )
                for raw_request_id in raw_request_ids
            )
            family_ids = tuple(
                dict.fromkeys(
                    trigger.alias.identity.request_id for trigger in triggers
                )
            )
            aliases = tuple(
                alias
                for alias in self._active_by_raw_id.values()
                if alias.identity.request_id in family_ids
            )
            batch = RequestFinishedBatch(
                tuple(dict.fromkeys(alias.identity for alias in aliases))
            )
            assigned_finish_id = self._next_finish_id if finish_id is None else finish_id
            if type(assigned_finish_id) is not int or assigned_finish_id < 1:
                raise ValueError("request finish_id must be a positive integer")
            receipt = RequestFinishReceipt(
                finish_id=assigned_finish_id,
                triggers=triggers,
                aliases=aliases,
                batch=batch,
            )
            if finish_id is None:
                self._next_finish_id += 1
            self._pending = receipt
            return receipt

    def prepare_canonical_finish(
        self,
        raw_request_ids: tuple[str, ...],
        batch: RequestFinishedBatch,
        *,
        finish_id: int,
    ) -> RequestFinishReceipt:
        """Prepare a parent mirror from worker-authorized parallel identities."""

        with self._lock:
            if self._pending is not None:
                raise RequestLifecycleError(
                    "a request finish transaction is already pending"
                )
            self._validate_distinct_raw_ids(raw_request_ids)
            if len(batch.requests) != len(set(batch.requests)) or not batch.requests:
                raise ValueError(
                    "canonical request-finished batch must be non-empty and unique"
                )
            if type(finish_id) is not int or finish_id < 1:
                raise ValueError("request finish_id must be a positive integer")
            triggers = tuple(
                RequestFinishTrigger(
                    raw_request_id=raw_request_id,
                    alias=self._resolve_finish_alias(raw_request_id),
                )
                for raw_request_id in raw_request_ids
            )
            prompt_by_family = {
                trigger.alias.identity.request_id: trigger.alias.identity.prompt_idx
                for trigger in triggers
            }
            if {
                identity.request_id for identity in batch.requests
            } != set(prompt_by_family) or any(
                identity.prompt_idx != prompt_by_family[identity.request_id]
                for identity in batch.requests
            ):
                raise RuntimeError(
                    "canonical worker identities differ from parent request families"
                )
            aliases = tuple(
                alias
                for alias in self._active_by_raw_id.values()
                if alias.identity.request_id in prompt_by_family
            )
            if any(alias.identity not in batch.requests for alias in aliases):
                raise RuntimeError(
                    "canonical worker batch omits a parent-observed request identity"
                )
            receipt = RequestFinishReceipt(
                finish_id=finish_id,
                triggers=triggers,
                aliases=aliases,
                batch=batch,
            )
            self._pending = receipt
            return receipt

    def commit_finish(self, receipt: RequestFinishReceipt) -> None:
        """Release aliases only after signed cleanup has acknowledged receipt."""

        with self._lock:
            if self._pending != receipt:
                raise RequestLifecycleError(
                    "request finish acknowledgement differs from the pending receipt"
                )
            for alias in receipt.aliases:
                del self._active_by_raw_id[alias.raw_request_id]
                self._finished_raw_ids.add(alias.raw_request_id)
            self._finished_raw_ids.update(
                trigger.raw_request_id for trigger in receipt.triggers
            )
            self._pending = None

    def abort_finish(self, receipt: RequestFinishReceipt) -> None:
        """Discard a failed cleanup attempt while retaining every alias."""

        with self._lock:
            if self._pending != receipt:
                raise RequestLifecycleError(
                    "request finish abort differs from the pending receipt"
                )
            self._pending = None

    def _resolve_finish_alias(self, raw_request_id: str) -> RequestAlias:
        if raw_request_id in self._finished_raw_ids:
            raise DuplicateRequestFinishError(
                f"request completion {raw_request_id!r} was already consumed"
            )
        exact = self._active_by_raw_id.get(raw_request_id)
        if exact is not None:
            return exact

        unsuffixed = self._without_enginecore_suffix(raw_request_id)
        if unsuffixed is not None:
            if unsuffixed in self._finished_raw_ids:
                raise DuplicateRequestFinishError(
                    f"request completion {raw_request_id!r} was already consumed"
                )
            suffixed = self._active_by_raw_id.get(unsuffixed)
            if suffixed is not None:
                return suffixed
        raise UnknownRequestAliasError(
            f"request completion {raw_request_id!r} has no registered alias"
        )

    @staticmethod
    def _without_enginecore_suffix(raw_request_id: str) -> str | None:
        prefix, separator, suffix = raw_request_id.rpartition("-")
        if separator and len(suffix) == 8 and suffix.isalnum() and prefix:
            return prefix
        return None

    @staticmethod
    def _validate_identity(identity: RequestIdentity) -> None:
        if not identity.request_id:
            raise ValueError("request aliases require a resolved request_id")
        if identity.prompt_idx < 0:
            raise ValueError("request aliases require a resolved prompt_idx")
        if identity.sample_idx < 0:
            raise ValueError("request aliases require a resolved sample_idx")

    @staticmethod
    def _validate_distinct_raw_ids(raw_request_ids: tuple[str, ...]) -> None:
        if not raw_request_ids:
            raise ValueError("request completion batch must not be empty")
        if len(raw_request_ids) != len(set(raw_request_ids)):
            raise DuplicateRequestFinishError(
                "request completion batch contains duplicate raw request ids"
            )


def register_request_alias(
    runtime: RequestLifecycleRuntime,
    raw_request_id: str,
) -> RequestAlias:
    """Resolve and retain one adapter-observed request id without fallback.

    Adapters call this at admission and whenever their private request-id
    grammar exposes a child or EngineCore suffix.  Resolution is deliberately
    strict: callers may *observe* an unrelated vLLM id, but a lifecycle
    completion may never manufacture an identity for it.
    """

    identity = resolve_request_identity(
        raw_request_id,
        prompt_indices=runtime.reqid_to_promptidx,
        sample_indices=runtime.reqid_to_sampleidx,
    )
    if identity is None:
        raise UnknownRequestAliasError(
            f"request alias {raw_request_id!r} has no resolved identity"
        )
    return runtime.request_lifecycle.register_alias(
        raw_request_id=raw_request_id,
        identity=identity,
    )


def observe_request_aliases(
    runtime: RequestLifecycleRuntime,
    raw_request_ids: Sequence[str],
) -> tuple[RequestAlias, ...]:
    """Register only observed aliases that resolve through the current map.

    vLLM runner batches can contain engine-private housekeeping ids.  They do
    not become lifecycle aliases simply because the adapter observed them;
    completions remain strict through :func:`register_request_alias`.
    """

    aliases: list[RequestAlias] = []
    seen: set[str] = set()
    for raw_request_id in raw_request_ids:
        if raw_request_id in seen:
            continue
        seen.add(raw_request_id)
        identity = resolve_request_identity(
            raw_request_id,
            prompt_indices=runtime.reqid_to_promptidx,
            sample_indices=runtime.reqid_to_sampleidx,
        )
        if identity is None:
            continue
        aliases.append(
            runtime.request_lifecycle.register_alias(
                raw_request_id=raw_request_id,
                identity=identity,
            )
        )
    return tuple(aliases)


def remove_finished_request_mappings(
    runtime: RequestLifecycleRuntime,
    receipt: RequestFinishReceipt,
) -> None:
    """Drop raw mapping rows only after the matching ledger commit.

    ``dict.pop(..., None)`` makes the terminal mapping removal idempotent for
    an EngineCore suffix that was never itself published as a mapping row.
    """

    raw_ids = tuple(
        dict.fromkeys(
            (
                *(alias.raw_request_id for alias in receipt.aliases),
                *(trigger.raw_request_id for trigger in receipt.triggers),
            )
        )
    )
    for raw_request_id in raw_ids:
        runtime.reqid_to_promptidx.pop(raw_request_id, None)
        runtime.reqid_to_sampleidx.pop(raw_request_id, None)


def finish_registered_requests(
    *,
    runtime: RequestLifecycleRuntime,
    raw_request_ids: tuple[str, ...],
    cleanup: RequestFinishCleanupService,
) -> RequestFinishOutcome:
    """Run the closed request-finish transaction in its required order.

    The adapter owns the concrete signed cleanup runner; this runtime helper
    owns no workload callback or backend policy.  A stage/cleanup failure
    leaves the receipt pending and mappings live because cleanup may already
    have changed signed state; callers must fail-stop rather than retry.
    """

    receipt = prepare_registered_request_finish(
        runtime=runtime,
        raw_request_ids=raw_request_ids,
    )
    outcome = apply_prepared_request_finish(
        runtime=runtime,
        receipt=receipt,
        cleanup=cleanup,
    )
    finalize_prepared_request_finish(runtime=runtime, receipt=receipt)
    return outcome


def prepare_registered_request_finish(
    *,
    runtime: RequestLifecycleRuntime,
    raw_request_ids: tuple[str, ...],
    finish_id: int | None = None,
) -> RequestFinishReceipt:
    """Prepare the closed typed batch and retain every raw alias."""

    return runtime.request_lifecycle.prepare_finish(
        raw_request_ids,
        finish_id=finish_id,
    )


def apply_prepared_request_finish(
    *,
    runtime: RequestLifecycleRuntime,
    receipt: RequestFinishReceipt,
    cleanup: RequestFinishCleanupService,
) -> RequestFinishOutcome:
    """Run signed cleanup while deliberately retaining the pending receipt.

    A cleanup service may have staged a port, retired typed state, or removed
    an overlay before it raises.  It is therefore unsafe to abort and retry;
    callers must fail-stop on an exception.  Only a later successful
    :func:`finalize_prepared_request_finish` releases raw mappings.
    """

    if runtime.request_lifecycle.pending_finish != receipt:
        raise RequestLifecycleError(
            "request finish apply differs from the pending receipt"
        )
    staged_ports = cleanup.stage(receipt.batch)
    if type(staged_ports) is not int or staged_ports < 0:
        raise TypeError("request-finished stage must return a non-negative integer")
    cleanup.confirm(receipt.batch, staged_ports)
    return RequestFinishOutcome(receipt=receipt, staged_ports=staged_ports)


def finalize_prepared_request_finish(
    *,
    runtime: RequestLifecycleRuntime,
    receipt: RequestFinishReceipt,
) -> None:
    """Commit one applied receipt and, only then, remove raw mapping rows."""

    runtime.request_lifecycle.commit_finish(receipt)
    remove_finished_request_mappings(runtime, receipt)
