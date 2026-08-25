#!/usr/bin/env python3
"""Parent-side control plane over vLLM's collective_rpc.

The plane invokes the string-named methods of ``TllmWorkerExtension`` on the
workers that own the model runner — in-process under mp=0 (InprocClient) and
in the EngineCore child under mp=1 (SyncMPClient). Using one mechanism for
both modes keeps the four parity cells on identical code paths; only string
method names cross the wire, so ``VLLM_ALLOW_INSECURE_SERIALIZATION`` is never
required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from tllm.contracts.consumer_stats import (
    TrainingStats,
    training_stats_map_from_wire,
)
from tllm.runtime.stats import DistillerTimingStats, PathHotspotStats, PlacementStats
from tllm.runtime.scheduling.workgraph_executor import WorkGraphRuntimeStats

if TYPE_CHECKING:
    from tllm.contracts.shape_specialization import RegionShapeSpecializationRequest

TLLM_WORKER_EXTENSION_QUALNAME = "tllm.runtime.mp.worker_extension.TllmWorkerExtension"

_SHAPE_SPECIALIZATION_REPLY_FIELDS = frozenset(
    {"signature", "plan_epoch", "requests"}
)
_SHAPE_SPECIALIZATION_ACK_REPLY_FIELDS = frozenset(
    {"signature", "plan_epoch", "acknowledged"}
)
_WORKLOAD_ACTIVATION_REPLY_FIELDS = frozenset(
    {"signature", "plan_epoch", "activated"}
)
_WORKLOAD_STATUS_REPLY_FIELDS = frozenset(
    {"pid", "signature", "plan_epoch", "artifacts", "state_versions", "diagnostics"}
)
_WORKLOAD_INSTALL_REPLY_FIELDS = frozenset(
    {"signature", "plan_epoch", "artifacts", "native", "committed"}
)


def _shape_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise TypeError(f"{label} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_wire_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise TypeError(f"{label} must be a positive integer")
    return value


def _nonnegative_wire_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"{label} must be a non-negative integer")
    return value


def _exact_wire_object(
    payload: object, label: str, fields: frozenset[str]
) -> Mapping[str, object]:
    """Accept only an exact JSON object envelope from collective_rpc."""

    if type(payload) is not dict:
        raise TypeError(f"{label} must be a JSON object")
    wire = payload
    observed = frozenset(wire)
    if observed != fields:
        missing = sorted(fields - observed)
        unexpected = sorted(observed - fields)
        raise ValueError(
            f"{label} has invalid fields: missing={missing}, unexpected={unexpected}"
        )
    if any(type(key) is not str for key in wire):
        raise TypeError(f"{label} fields must be strings")
    return wire


def _workload_status_from_wire(payload: object) -> dict[str, object]:
    """Validate the closed, observational worker-status wire contract."""

    fields = _exact_wire_object(
        payload,
        "workload-status reply",
        _WORKLOAD_STATUS_REPLY_FIELDS,
    )
    pid = _positive_wire_int(fields["pid"], "status.pid")
    signature = _shape_digest(fields["signature"], "status.signature")
    plan_epoch = _positive_wire_int(fields["plan_epoch"], "status.plan_epoch")

    raw_artifacts = fields["artifacts"]
    if type(raw_artifacts) is not list:
        raise TypeError("status.artifacts must be a JSON array")
    artifacts: list[dict[str, str]] = []
    artifact_ids: set[str] = set()
    for raw in raw_artifacts:
        artifact = _exact_wire_object(
            raw, "status artifact", frozenset({"region_id", "artifact_key"})
        )
        region_id = artifact["region_id"]
        if not isinstance(region_id, str) or not region_id:
            raise TypeError("status artifact region_id must be a non-empty string")
        if region_id in artifact_ids:
            raise ValueError("status.artifacts contains duplicate region IDs")
        artifact_ids.add(region_id)
        artifacts.append(
            {
                "region_id": region_id,
                "artifact_key": _shape_digest(
                    artifact["artifact_key"], "status artifact_key"
                ),
            }
        )

    raw_versions = fields["state_versions"]
    if type(raw_versions) is not list:
        raise TypeError("status.state_versions must be a JSON array")
    state_versions: list[dict[str, object]] = []
    for raw in raw_versions:
        entry = _exact_wire_object(
            raw, "status state version", frozenset({"resource", "key", "version"})
        )
        resource = entry["resource"]
        key = entry["key"]
        if not isinstance(resource, str) or not resource:
            raise TypeError("status state-version resource must be a non-empty string")
        if key is not None and (
            isinstance(key, bool) or not isinstance(key, (str, int))
        ):
            raise TypeError("status state-version key must be None, str, or non-bool int")
        version = _nonnegative_wire_int(entry["version"], "status state-version version")
        state_versions.append({"resource": resource, "key": key, "version": version})

    raw_diagnostics = fields["diagnostics"]
    if type(raw_diagnostics) is not dict:
        raise TypeError("status.diagnostics must be a JSON object")
    diagnostics: dict[str, dict[str, int | float]] = {}
    for workload_id, raw_metrics in raw_diagnostics.items():
        if type(workload_id) is not str or not workload_id:
            raise TypeError("status diagnostic workload IDs must be non-empty strings")
        if type(raw_metrics) is not dict:
            raise TypeError("status diagnostic metrics must be JSON objects")
        metrics: dict[str, int | float] = {}
        for name, value in raw_metrics.items():
            if type(name) is not str or not name:
                raise TypeError("status diagnostic names must be non-empty strings")
            if type(value) not in (int, float):
                raise TypeError("status diagnostic values must be int or float")
            if type(value) is float and not isfinite(value):
                raise ValueError("status diagnostic float values must be finite")
            metrics[name] = value
        diagnostics[workload_id] = metrics
    return {
        "pid": pid,
        "signature": signature,
        "plan_epoch": plan_epoch,
        "artifacts": artifacts,
        "state_versions": state_versions,
        "diagnostics": diagnostics,
    }


def _workload_install_report_from_wire(
    payload: object,
    *,
    committed: bool,
) -> dict[str, object]:
    """Decode one prepare/commit report without semantic host coercions."""

    fields = _exact_wire_object(
        payload,
        "workload-install reply",
        _WORKLOAD_INSTALL_REPLY_FIELDS,
    )
    signature = _shape_digest(fields["signature"], "workload-install signature")
    plan_epoch = _positive_wire_int(
        fields["plan_epoch"], "workload-install plan_epoch"
    )
    native = fields["native"]
    if type(native) is not bool:
        raise TypeError("workload-install native must be a JSON boolean")
    observed_committed = fields["committed"]
    if type(observed_committed) is not bool:
        raise TypeError("workload-install committed must be a JSON boolean")
    if observed_committed is not committed:
        raise ValueError("workload-install committed phase does not match RPC")
    raw_artifacts = fields["artifacts"]
    if type(raw_artifacts) is not list:
        raise TypeError("workload-install artifacts must be a JSON array")
    artifacts: list[dict[str, str]] = []
    artifact_ids: set[str] = set()
    for raw in raw_artifacts:
        artifact = _exact_wire_object(
            raw,
            "workload-install artifact",
            frozenset({"region_id", "artifact_key"}),
        )
        region_id = artifact["region_id"]
        if type(region_id) is not str or not region_id:
            raise TypeError(
                "workload-install artifact region_id must be a non-empty string"
            )
        if region_id in artifact_ids:
            raise ValueError("workload-install artifacts contain duplicate region IDs")
        artifact_ids.add(region_id)
        artifacts.append(
            {
                "region_id": region_id,
                "artifact_key": _shape_digest(
                    artifact["artifact_key"], "workload-install artifact_key"
                ),
            }
        )
    return {
        "signature": signature,
        "plan_epoch": plan_epoch,
        "artifacts": artifacts,
        "native": native,
        "committed": observed_committed,
    }


@dataclass(frozen=True, slots=True)
class WorkerShapeSpecializationReply:
    """One worker's immutable epoch identity and pending typed requests."""

    signature: str
    plan_epoch: int
    requests: tuple["RegionShapeSpecializationRequest", ...]

    @classmethod
    def from_wire(cls, payload: object) -> "WorkerShapeSpecializationReply":
        from tllm.contracts.shape_specialization import (
            RegionShapeSpecializationRequest,
        )

        fields = _exact_wire_object(
            payload,
            "shape-specialization reply",
            _SHAPE_SPECIALIZATION_REPLY_FIELDS,
        )
        signature = _shape_digest(fields["signature"], "reply.signature")
        plan_epoch = _positive_wire_int(fields["plan_epoch"], "reply.plan_epoch")
        raw_requests = fields["requests"]
        if type(raw_requests) is not list:
            raise TypeError("reply.requests must be a JSON array")
        requests = tuple(
            RegionShapeSpecializationRequest.from_wire(item)
            for item in raw_requests
        )
        dedup_keys = tuple(request.dedup_key for request in requests)
        if len(dedup_keys) != len(set(dedup_keys)):
            raise ValueError("reply.requests contains duplicate specialization requests")
        if any(
            request.base_signature != signature
            or request.observed_plan_epoch != plan_epoch
            for request in requests
        ):
            raise ValueError(
                "shape-specialization reply request identity does not match active epoch"
            )
        return cls(signature=signature, plan_epoch=plan_epoch, requests=requests)


@dataclass(frozen=True, slots=True)
class WorkerWorkloadActivationReply:
    """Proof that one worker has opened a committed epoch for safe-point use."""

    signature: str
    plan_epoch: int
    activated: bool

    @classmethod
    def from_wire(cls, payload: object) -> "WorkerWorkloadActivationReply":
        fields = _exact_wire_object(
            payload,
            "workload activation reply",
            _WORKLOAD_ACTIVATION_REPLY_FIELDS,
        )
        activated = fields["activated"]
        if type(activated) is not bool:
            raise TypeError("workload activation reply.activated must be a boolean")
        return cls(
            signature=_shape_digest(fields["signature"], "activation.signature"),
            plan_epoch=_positive_wire_int(fields["plan_epoch"], "activation.plan_epoch"),
            activated=activated,
        )


@dataclass(frozen=True, slots=True)
class WorkerStatsReply:
    esamp: TrainingStats
    esamp_per_request: Mapping[int, TrainingStats]
    distiller_timing: DistillerTimingStats
    path_hotspot: PathHotspotStats
    placement: PlacementStats
    workgraph: WorkGraphRuntimeStats = field(default_factory=WorkGraphRuntimeStats)

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "WorkerStatsReply":
        return cls(
            esamp=TrainingStats.from_wire(payload["esamp"]),
            esamp_per_request=training_stats_map_from_wire(payload["esamp_per_request"]),
            distiller_timing=DistillerTimingStats.from_wire(payload["distiller_timing"]),
            path_hotspot=PathHotspotStats.from_wire(payload["path_hotspot"]),
            placement=PlacementStats.from_wire(payload["placement"]),
            workgraph=WorkGraphRuntimeStats.from_wire(payload.get("workgraph", {})),
        )


_ACTIVE_CONTROL_PLANE: "ControlPlane | None" = None


def set_active_control_plane(plane: "ControlPlane | None") -> None:
    global _ACTIVE_CONTROL_PLANE
    _ACTIVE_CONTROL_PLANE = plane


def active_control_plane() -> "ControlPlane | None":
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "0") != "1":
        return None
    return _ACTIVE_CONTROL_PLANE


class ControlPlane:
    """Thin facade: one method per worker-extension entry point."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    @staticmethod
    def try_bind(llm: Any) -> "ControlPlane | None":
        """Bind when the LLM exposes a collective_rpc-capable engine core.

        Test doubles and non-vLLM runners may lack one; callers then fall
        back to process-local behavior (mp=0 semantics).
        """
        try:
            core_client = llm.llm_engine.engine_core
            if not callable(core_client.collective_rpc):
                return None
        except AttributeError:
            return None
        return ControlPlane(llm)

    def call(self, method: str, *args: Any, timeout: float | None = None) -> list[Any]:
        try:
            core_client = self._llm.llm_engine.engine_core
        except AttributeError as exc:
            raise ValueError("control plane requires an LLM with llm_engine") from exc
        return list(core_client.collective_rpc(str(method), timeout=timeout, args=tuple(args)))

    def ping(self, *, timeout: float | None = None) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self.call("tllm_ping", timeout=timeout)]

    def push_request_mapping(
        self, mappings: Iterable[tuple[str, int, int]], *, timeout: float | None = None
    ) -> None:
        rows = [[str(req_id), int(prompt_idx), int(sample_idx)] for req_id, prompt_idx, sample_idx in mappings]
        if not rows:
            return
        self.call("tllm_push_request_mapping", rows, timeout=timeout)

    def pull_stats(
        self,
        category: str = "all",
        *,
        sync: bool = True,
        timeout: float | None = None,
    ) -> list[WorkerStatsReply]:
        return [
            WorkerStatsReply.from_wire(dict(entry))
            for entry in self.call("tllm_pull_stats", str(category), bool(sync), timeout=timeout)
        ]

    def synchronize(self, *, timeout: float | None = None) -> None:
        self.call("tllm_synchronize", timeout=timeout)

    def apply_enable_flags(self, flags: Mapping[str, bool], *, timeout: float | None = None) -> None:
        self.call("tllm_apply_enable_flags", {str(k): bool(v) for k, v in flags.items()}, timeout=timeout)

    def register_consumer(
        self, name: str, config: Mapping[str, Any], *, primary: bool = False, timeout: float | None = None
    ) -> None:
        self.call("tllm_register_consumer", str(name), dict(config), bool(primary), timeout=timeout)

    def clear_consumers(self, *, timeout: float | None = None) -> None:
        self.call("tllm_clear_consumers", timeout=timeout)

    def workload_timelines(
        self,
        *,
        configured_budget: int = 0,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        return [
            dict(entry)
            for entry in self.call(
                "tllm_workload_timeline",
                int(configured_budget),
                timeout=timeout,
            )
        ]

    def workload_status(
        self,
        *,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Read immutable native Region identity and worker execution evidence."""

        return [
            _workload_status_from_wire(entry)
            for entry in self.call("tllm_workload_status", timeout=timeout)
        ]

    def install_workload_compilation(
        self,
        compilation: Any,
        *,
        reconstruction_specs: Iterable[Any] = (),
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        from tllm.schedule import WorkloadCompilation
        from tllm.runtime.workload_plugins import WorkloadWorkerSpec

        if not isinstance(compilation, WorkloadCompilation):
            raise TypeError("compilation must be WorkloadCompilation")
        specs = tuple(reconstruction_specs)
        if any(not isinstance(spec, WorkloadWorkerSpec) for spec in specs):
            raise TypeError(
                "reconstruction_specs must contain WorkloadWorkerSpec values"
            )
        return [
            dict(report)
            for report in self.call(
                "tllm_install_workload_decision",
                compilation.cache_key,
                compilation.graph.to_json(),
                compilation.timeline.to_json(),
                compilation.options.to_json(),
                compilation.decision.to_json(),
                compilation.physical.plan_epoch
                if compilation.physical is not None
                else 1,
                [spec.to_wire() for spec in specs],
                timeout=timeout,
            )
        ]

    def prepare_workload_compilation(
        self,
        compilation: Any,
        *,
        reconstruction_specs: Iterable[Any] = (),
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Stage a signed Region plan on every worker without publishing it.

        The returned artifact report is the worker's proof that its local
        reconstruction agrees with the parent-signed decision.  Callers must
        validate every reply before calling :meth:`commit_workload_compilation`.
        """
        from tllm.schedule import WorkloadCompilation
        from tllm.runtime.workload_plugins import WorkloadWorkerSpec

        if not isinstance(compilation, WorkloadCompilation):
            raise TypeError("compilation must be WorkloadCompilation")
        specs = tuple(reconstruction_specs)
        if any(not isinstance(spec, WorkloadWorkerSpec) for spec in specs):
            raise TypeError(
                "reconstruction_specs must contain WorkloadWorkerSpec values"
            )
        return [
            dict(report)
            for report in self.call(
                "tllm_prepare_workload_decision",
                compilation.cache_key,
                compilation.graph.to_json(),
                compilation.timeline.to_json(),
                compilation.options.to_json(),
                compilation.decision.to_json(),
                compilation.physical.plan_epoch
                if compilation.physical is not None
                else 1,
                [spec.to_wire() for spec in specs],
                timeout=timeout,
            )
        ]

    def commit_workload_compilation(
        self,
        compilation: Any,
        *,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Publish a previously prepared signed Region plan on every worker."""
        from tllm.schedule import WorkloadCompilation

        if not isinstance(compilation, WorkloadCompilation):
            raise TypeError("compilation must be WorkloadCompilation")
        epoch = compilation.physical.plan_epoch if compilation.physical is not None else 1
        return [
            dict(report)
            for report in self.call(
                "tllm_commit_workload_decision",
                compilation.decision.signature,
                int(epoch),
                timeout=timeout,
            )
        ]

    def activate_workload_compilation(
        self,
        compilation: Any,
        *,
        timeout: float | None = None,
    ) -> list[WorkerWorkloadActivationReply]:
        """Open a globally committed immutable epoch on every worker.

        This is deliberately separate from commit: callers must validate all
        commit votes before they can make a staged epoch eligible for a worker
        safe-point promotion.
        """

        from tllm.schedule import WorkloadCompilation

        if not isinstance(compilation, WorkloadCompilation):
            raise TypeError("compilation must be WorkloadCompilation")
        signature = _shape_digest(compilation.decision.signature, "signature")
        epoch = _positive_wire_int(
            compilation.physical.plan_epoch if compilation.physical is not None else 1,
            "plan_epoch",
        )
        return [
            WorkerWorkloadActivationReply.from_wire(reply)
            for reply in self.call(
                "tllm_activate_workload_decision",
                signature,
                epoch,
                timeout=timeout,
            )
        ]

    def abort_workload_compilation(
        self,
        compilation: Any,
        *,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Discard or roll back a prepared Region epoch on every worker."""
        from tllm.schedule import WorkloadCompilation

        if not isinstance(compilation, WorkloadCompilation):
            raise TypeError("compilation must be WorkloadCompilation")
        epoch = compilation.physical.plan_epoch if compilation.physical is not None else 1
        return [
            dict(report)
            for report in self.call(
                "tllm_abort_workload_decision",
                compilation.decision.signature,
                int(epoch),
                timeout=timeout,
            )
        ]

    def finalize_workload_compilation(
        self,
        compilation: Any,
        *,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Release retained prior epochs after a successful global commit."""
        from tllm.schedule import WorkloadCompilation

        if not isinstance(compilation, WorkloadCompilation):
            raise TypeError("compilation must be WorkloadCompilation")
        epoch = compilation.physical.plan_epoch if compilation.physical is not None else 1
        return [
            dict(report)
            for report in self.call(
                "tllm_finalize_workload_decision",
                compilation.decision.signature,
                int(epoch),
                timeout=timeout,
            )
        ]

    def drain_workload_epochs(
        self,
        signature: str,
        plan_epoch: int,
        *,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Collect worker epochs already retired at a runtime safe point.

        This deliberately accepts only scalar JSON wire values.  It does not
        synchronize CUDA or force a current step to retire; workers merely
        release executions that their local ``NativeWorkloadExecutionStore``
        has already marked retired.
        """

        if not isinstance(signature, str):
            raise TypeError("signature must be a string")
        if isinstance(plan_epoch, bool) or not isinstance(plan_epoch, int):
            raise TypeError("plan_epoch must be an integer")
        return [
            dict(report)
            for report in self.call(
                "tllm_drain_workload_epochs",
                signature,
                int(plan_epoch),
                timeout=timeout,
            )
        ]

    def peek_shape_specialization_requests(
        self, *, timeout: float | None = None
    ) -> list[WorkerShapeSpecializationReply]:
        """Read pending worker shape misses without consuming them.

        The returned requests are revalidated through the closed Python/Rust
        contract before a coordinator may compile a replacement epoch.
        """

        return [
            WorkerShapeSpecializationReply.from_wire(entry)
            for entry in self.call(
                "tllm_peek_shape_specialization_requests", timeout=timeout
            )
        ]

    def ack_shape_specialization_requests(
        self,
        signature: str,
        plan_epoch: int,
        dedup_keys: Iterable[str],
        *,
        timeout: float | None = None,
    ) -> list[tuple[str, ...]]:
        """Confirm a published replacement epoch and remove exact worker misses.

        Call this only after every worker has prepared and committed the new
        immutable epoch.  Empty acknowledgement sets never issue an RPC.
        """

        normalized_signature = _shape_digest(signature, "signature")
        normalized_epoch = _positive_wire_int(plan_epoch, "plan_epoch")
        keys = tuple(_shape_digest(key, "dedup_key") for key in dedup_keys)
        if not keys:
            return []
        if len(keys) != len(set(keys)):
            raise ValueError("dedup_keys must be unique")
        replies = self.call(
            "tllm_ack_shape_specialization_requests",
            normalized_signature,
            normalized_epoch,
            list(keys),
            timeout=timeout,
        )
        acknowledged: list[tuple[str, ...]] = []
        for reply in replies:
            fields = _exact_wire_object(
                reply,
                "shape-specialization acknowledgement reply",
                _SHAPE_SPECIALIZATION_ACK_REPLY_FIELDS,
            )
            reply_signature = _shape_digest(
                fields["signature"], "acknowledgement.signature"
            )
            reply_epoch = _positive_wire_int(
                fields["plan_epoch"], "acknowledgement.plan_epoch"
            )
            raw_acknowledged = fields["acknowledged"]
            if type(raw_acknowledged) is not list:
                raise TypeError("acknowledgement.acknowledged must be a JSON array")
            reply_keys = tuple(
                _shape_digest(key, "acknowledgement.dedup_key")
                for key in raw_acknowledged
            )
            if (
                reply_signature != normalized_signature
                or reply_epoch != normalized_epoch
                or reply_keys != keys
            ):
                raise ValueError("shape-specialization acknowledgement identity mismatch")
            acknowledged.append(reply_keys)
        return acknowledged

    def set_fast_weight_enabled(
        self,
        enabled: bool,
        *,
        layers: Iterable[int] | None = None,
        chunk_size: int | None = None,
        learning_rate: float | None = None,
        mode: str | None = None,
        decode_update_enabled: bool | None = None,
        key_mode: str | None = None,
        max_slots: int | None = None,
        max_bytes: int | None = None,
        fail_on_unsupported: bool | None = None,
        timeout: float | None = None,
    ) -> None:
        self.call(
            "tllm_set_fast_weight_enabled",
            bool(enabled),
            None if layers is None else [int(layer) for layer in layers],
            chunk_size,
            learning_rate,
            mode,
            decode_update_enabled,
            key_mode,
            max_slots,
            max_bytes,
            fail_on_unsupported,
            timeout=timeout,
        )
