"""Adapter-neutral request alias ownership and finished-batch expansion.

Adapters resolve their private request identifiers before calling this ledger.
The ledger then retains every observed raw alias until its request family is
finished, so a parent or child completion always produces the same closed,
typed cleanup batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from tllm.contracts.request_identity import RequestFinishedBatch, RequestIdentity


class RequestLifecycleError(RuntimeError):
    """A request lifecycle event violates the registered alias contract."""


class UnknownRequestAliasError(RequestLifecycleError):
    """A completion cannot be associated with a registered raw alias."""


class DuplicateRequestFinishError(RequestLifecycleError):
    """A raw completion has already been consumed by this ledger."""


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
        self, raw_request_ids: tuple[str, ...]
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
            receipt = RequestFinishReceipt(
                finish_id=self._next_finish_id,
                triggers=triggers,
                aliases=aliases,
                batch=batch,
            )
            self._next_finish_id += 1
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
