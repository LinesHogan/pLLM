#!/usr/bin/env python3
"""Worker extension exposing tLLM control-plane methods to collective_rpc.

vLLM mixes this class into every worker (``worker_extension_cls`` in
ParallelConfig, resolved by qualified name inside the worker process, so it
is spawn-safe). All methods have plain string names and take/return only
msgpack-safe data (str/int/float/bool/list/dict/None): string method names
cross the client→EngineCore boundary WITHOUT
``VLLM_ALLOW_INSECURE_SERIALIZATION``; callables would not.

Every method is a thin delegate to the process-local runtime
(``tllm.runtime.residual_runtime``): in the parent under mp=0, in the
EngineCore child under mp=1. No torch objects cross the wire; stats that
involve CUDA events are resolved to floats child-side before returning.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, is_dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Mapping,
    Protocol,
    cast,
)

if TYPE_CHECKING:
    import torch

    from tllm.contracts.runtime_context import RunnerLike
    from tllm.ports.residual_stream import ResidualLocator
    from tllm.runtime.scheduling.workgraph_executor import WorkGraphRuntimeStatsReader
    from tllm.runtime.workload_execution import (
        NativeWorkloadExecution,
        NativeWorkloadExecutionStore,
        WorkloadExecutionBindings,
    )
    from tllm.runtime.vllm_patch.port_runtime_hooks import RuntimeHookTransaction
    from tllm.schedule import ScheduleGraph, WorkloadCompilation
    from tllm.schedule.contracts import ScheduleDecision


class _WorkerView(Protocol):
    """The one vLLM Worker attribute this extension touches."""

    model_runner: object | None  # None only while v1 Worker is constructing


class _ExternalRunner(Protocol):
    """vLLM runner fields needed at the process boundary."""

    model_config: object
    device: object
    max_num_reqs: object
    max_num_tokens: object
    max_num_batched_tokens: object
    model: object


class _ExternalModelConfig(Protocol):
    get_hidden_size: object
    dtype: object


WireValue = None | str | int | float | bool | list["WireValue"] | dict[str, "WireValue"]


@dataclass(frozen=True, slots=True)
class _RunnerSpecializationFacts:
    rows: int
    hidden_size: int
    dtype: object
    device: object
    model_layer_count: int | None
    max_prefill_tokens: int | None


@dataclass(slots=True)
class _StagedWorkloadDecision:
    """Worker-local two-phase install state.

    ``execution`` is fully reconstructed during prepare but deliberately not
    assigned to ``RUNTIME``.  The previous runtime references stay retained
    through commit so an abort can deterministically restore them if another
    worker rejects the epoch.
    """

    signature: str
    plan_epoch: int
    compilation: "WorkloadCompilation"
    bindings: "WorkloadExecutionBindings | None"
    execution: "NativeWorkloadExecution | None"
    native_schedule_graph: "ScheduleGraph | None"
    residual_raw_paths: Mapping["ResidualLocator", str]
    native_workloads_pending: bool
    workload_compilation: "WorkloadCompilation | None"
    workload_execution: "NativeWorkloadExecution | None"
    workload_execution_store: "NativeWorkloadExecutionStore | None"
    physical_executor: "WorkGraphRuntimeStatsReader | None"
    hook_transaction: "RuntimeHookTransaction | None" = None
    publication_store: "NativeWorkloadExecutionStore | None" = None
    execution_staged_in_store: bool = False
    committed: bool = False
    activated: bool = False


def _runner_of(worker: object) -> object | None:
    return cast(_WorkerView, worker).model_runner


def _runner_specialization_facts(
    runner: object | None,
) -> _RunnerSpecializationFacts | None:
    """Read optional shape facts from an external vLLM runner fail-closed."""

    if runner is None:
        return None
    raw_runner = cast(_ExternalRunner, runner)
    try:
        model_config = raw_runner.model_config
        device = raw_runner.device
        max_num_reqs = raw_runner.max_num_reqs
    except AttributeError:
        return None
    if model_config is None or device is None:
        return None
    raw_config = cast(_ExternalModelConfig, model_config)
    try:
        get_hidden_size = raw_config.get_hidden_size
        dtype = raw_config.dtype
    except AttributeError:
        return None
    if not callable(get_hidden_size) or dtype is None:
        return None
    try:
        hidden_size = cast(Callable[[], object], get_hidden_size)()
    except (AttributeError, OverflowError, TypeError, ValueError):
        return None
    from tllm.backends.protocol import runner_binding_for

    binding = runner_binding_for(raw_runner)
    model_layer_count: int | None = None
    if binding is not None:
        raw_layer_count = binding.driver.runner_model_layer_count(raw_runner)
        if type(raw_layer_count) is not int or raw_layer_count < 1:
            raise TypeError("runner model layer count must be an exact positive integer")
        model_layer_count = raw_layer_count
    if type(max_num_reqs) is not int or type(hidden_size) is not int:
        raise TypeError(
            "runner specialization rows and hidden_size must be exact integers"
        )
    if max_num_reqs < 1 or hidden_size < 1:
        raise ValueError(
            "runner specialization rows and hidden_size must be positive"
        )
    for name in ("max_num_tokens", "max_num_batched_tokens"):
        try:
            max_prefill_tokens = getattr(raw_runner, name)
        except AttributeError:
            continue
        if type(max_prefill_tokens) is not int or max_prefill_tokens < 1:
            raise TypeError(
                "runner specialization max_prefill_tokens must be an exact positive integer"
            )
        break
    else:
        raise RuntimeError(
            "runner must expose max_num_tokens or max_num_batched_tokens for "
            "runtime specialization"
        )
    return _RunnerSpecializationFacts(
        rows=max_num_reqs,
        hidden_size=hidden_size,
        dtype=dtype,
        device=device,
        model_layer_count=model_layer_count,
        max_prefill_tokens=max_prefill_tokens,
    )


def _runner_device_fact(runner: object | None) -> object | None:
    """Read only the device fact without requiring shape specialization."""

    if runner is None:
        return None
    try:
        return cast(_ExternalRunner, runner).device
    except AttributeError:
        return None


def _runner_has_loaded_model(runner: object | None) -> bool:
    if runner is None:
        return False
    try:
        return cast(_ExternalRunner, runner).model is not None
    except AttributeError:
        return False


def _hook_runner(runner: object | None) -> "RunnerLike":
    """Narrow the foreign vLLM runner only at the hook-install boundary."""

    if runner is None:
        raise RuntimeError("worker has no model runner")
    return cast("RunnerLike", runner)


def _to_plain(value: object, *, _depth: int = 0) -> WireValue:
    """Convert stats payloads to msgpack-safe data; reject anything else."""
    if _depth > 8:
        raise TypeError(f"stats payload too deep: {type(value).__name__}")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _to_plain(asdict(value), _depth=_depth + 1)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            str(key): _to_plain(item, _depth=_depth + 1)
            for key, item in mapping.items()
        }
    if isinstance(value, (list, tuple)):
        items = cast(list[object] | tuple[object, ...], value)
        return [_to_plain(item, _depth=_depth + 1) for item in items]
    raise TypeError(
        f"stats payload value of type {type(value).__name__} is not wire-safe; "
        "resolve CUDA events/tensors to floats before returning"
    )


def _decision_requires_hidden_taps(decision: "ScheduleDecision") -> bool:
    """Read hidden-input requirements from the signed Region partition."""

    from tllm.schedule.contracts import RegionInputKind

    return any(
        binding.kind is RegionInputKind.HIDDEN
        for region in decision.execution_regions
        for binding in region.input_bindings
    )


def _wire_sha256(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise TypeError(f"{label} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _wire_text(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    return value


def _wire_positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise TypeError(f"{label} must be a positive integer")
    return value


def _wire_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{label} must be a non-negative integer")
    return value


def _workload_identity(signature: object, plan_epoch: object) -> tuple[str, int]:
    return (
        _wire_sha256(signature, "signature"),
        _wire_positive_int(plan_epoch, "plan_epoch"),
    )


class TllmWorkerExtension:
    """Methods are invoked by qualified string name via collective_rpc."""

    # These fields are intentionally explicit rather than discovered through
    # runtime probes: a worker owns at most one staged immutable epoch.
    model_runner: object | None = None
    _tllm_staged_workload: _StagedWorkloadDecision | None = None
    _tllm_last_finalized_workload: tuple[str, int] | None = None
    _tllm_last_drained_workload: tuple[str, int] | None = None

    def _tllm_worker_id(self) -> str:
        return f"{os.getpid()}:{id(self):x}"

    def tllm_ping(self) -> dict[str, WireValue]:
        """Liveness/provenance probe: which process is answering."""
        from tllm.runtime import residual_runtime as core

        return {
            "pid": os.getpid(),
            "worker_id": self._tllm_worker_id(),
            "patch_installed": core.runtime_patch_installed(),
            "consumer": core.active_consumer_id(),
            "mapping_size": len(core.RUNTIME.reqid_to_promptidx),
        }

    def tllm_push_request_mapping(self, mappings: list[list[object]]) -> int:
        """Install request_id -> (prompt_idx, sample_idx) rows.

        Each entry is ``[request_id, prompt_idx, sample_idx]``. Called by the
        parent BEFORE the corresponding add_request so the FIFO input socket
        guarantees the mapping lands before the request is schedulable.
        """
        from tllm.runtime import residual_runtime as core

        count = 0
        for entry in mappings:
            request_id = _wire_text(entry[0], "request_id")
            prompt_idx = _wire_nonnegative_int(entry[1], "prompt_idx")
            sample_idx = _wire_nonnegative_int(entry[2], "sample_idx")
            core.RUNTIME.reqid_to_promptidx[request_id] = prompt_idx
            core.RUNTIME.reqid_to_sampleidx[request_id] = sample_idx
            count += 1
        return count

    def tllm_pull_stats(self, category: str = "all", sync: bool = True) -> dict[str, WireValue]:
        """Read and reset runtime stats, resolved to plain data child-side."""
        from tllm.runtime import residual_runtime as core

        supported = {
            "all",
            "esamp",
            "esamp_per_request",
            "distiller_timing",
            "path_hotspot",
            "placement",
            "workgraph",
        }
        if category not in supported:
            raise ValueError(f"unsupported stats category {category!r}; supported: {sorted(supported)}")
        payload: dict[str, WireValue] = {
            "esamp": {"loss_avg": 0.0, "loss_count": 0, "trace_losses": []},
            "esamp_per_request": {},
            "distiller_timing": {},
            "path_hotspot": {},
            "placement": {},
            "workgraph": {},
        }
        if category in ("all", "esamp"):
            payload["esamp"] = _to_plain(core.read_and_reset_esamp_stats(sync=sync))
        if category in ("all", "esamp_per_request"):
            payload["esamp_per_request"] = _to_plain(core.read_and_reset_esamp_per_request_stats(sync=sync))
        if category in ("all", "distiller_timing"):
            payload["distiller_timing"] = _to_plain(core.read_and_reset_distiller_timing_stats(sync=sync))
        if category in ("all", "path_hotspot"):
            payload["path_hotspot"] = _to_plain(core.read_and_reset_path_hotspot_stats(sync=sync))
        if category in ("all", "placement"):
            payload["placement"] = _to_plain(core.read_and_reset_placement_stats())
        if category in ("all", "workgraph"):
            payload["workgraph"] = _to_plain(core.read_workgraph_runtime_stats())
        return payload

    def tllm_synchronize(self) -> None:
        """Drain outstanding async consumer work in this process."""
        from tllm.runtime import residual_runtime as core

        core.synchronize_esamp()

    def tllm_prepare_workload_port(
        self, key: str, transaction: str, wire: dict[str, object]
    ) -> dict[str, object]:
        from tllm.runtime import residual_runtime as core
        from tllm.runtime.mp.port_wire import decode_port_payload

        store = core.RUNTIME.workload_execution_store
        execution = (
            store.next_execution
            if store is not None
            else core.RUNTIME.workload_execution
        )
        if execution is None:
            raise RuntimeError("worker has no active workload execution")
        execution.prepare_port(key, transaction, decode_port_payload(wire))
        return {
            "worker_id": self._tllm_worker_id(),
            "key": key,
            "transaction": transaction,
            "prepared": True,
        }

    def tllm_commit_workload_port(
        self, key: str, transaction: str
    ) -> dict[str, object]:
        from tllm.runtime import residual_runtime as core

        store = core.RUNTIME.workload_execution_store
        execution = (
            store.next_execution
            if store is not None
            else core.RUNTIME.workload_execution
        )
        if execution is None:
            raise RuntimeError("worker has no active workload execution")
        sequence = execution.commit_port(key, transaction)
        return {
            "worker_id": self._tllm_worker_id(),
            "key": key,
            "transaction": transaction,
            "committed": True,
            "sequence": sequence,
        }

    def tllm_abort_workload_port(
        self, key: str, transaction: str
    ) -> dict[str, object]:
        from tllm.runtime import residual_runtime as core

        store = core.RUNTIME.workload_execution_store
        execution = (
            store.next_execution
            if store is not None
            else core.RUNTIME.workload_execution
        )
        if execution is not None:
            execution.abort_port(key, transaction)
        return {
            "worker_id": self._tllm_worker_id(),
            "key": key,
            "transaction": transaction,
            "aborted": True,
        }

    def tllm_drain_workload_output(self, key: str) -> dict[str, object]:
        from tllm.runtime import residual_runtime as core
        from tllm.runtime.mp.port_wire import encode_inline_port_payload

        store = core.RUNTIME.workload_execution_store
        execution = store.current if store is not None else core.RUNTIME.workload_execution
        if execution is None:
            raise RuntimeError("worker has no active workload execution")
        return {
            "worker_id": self._tllm_worker_id(),
            "key": key,
            "values": [encode_inline_port_payload(value) for value in execution.drain_output(key)],
        }

    def tllm_apply_enable_flags(self, flags: dict[str, bool]) -> None:
        """Apply supported enable flags to this process's runtime."""
        from tllm.runtime import residual_runtime as core

        supported = ("esamp_training_enabled",)
        unknown = sorted(set(flags) - set(supported))
        if unknown:
            raise ValueError(f"unsupported enable flag(s) {unknown}; supported: {list(supported)}")
        if "esamp_training_enabled" in flags:
            if (
                core.RUNTIME.workload_execution is not None
                or core.RUNTIME.workload_execution_store is not None
            ):
                raise RuntimeError(
                    "native ESamp enable changes require a coordinator-signed "
                    "cold workload epoch"
                )
            core.set_esamp_training_enabled(bool(flags["esamp_training_enabled"]))

    def tllm_register_consumer(self, name: str, config: dict[str, object], primary: bool = False) -> None:
        """Rebuild a consumer from its spec and activate it in this process."""
        from tllm.runtime import consumer_plugins as consumer_registry
        from tllm.runtime.mp.bootstrap import ConsumerSpec

        consumer = consumer_registry.build_consumer_from_spec(ConsumerSpec(name=str(name), config=dict(config)))
        consumer_registry.activate_consumer(consumer, primary=bool(primary))

    def tllm_clear_consumers(self) -> None:
        from tllm.runtime import residual_runtime as core

        core.clear_dispatch_consumers()

    def tllm_bind_runner_runtime(self) -> None:
        """Record this worker's runner in the runner->runtime routing table.

        String-method replacement for the old closure RPC (closures need
        VLLM_ALLOW_INSECURE_SERIALIZATION and cannot cross to the child).
        """
        from tllm.runtime.runtime import active_runtime, bind_runner_runtime

        runner = _runner_of(self)
        if runner is not None:
            bind_runner_runtime(runner, active_runtime())

    def tllm_runner_name(self) -> str:
        """Report the model runner class actually serving this worker."""
        runner = _runner_of(self)
        return "unknown" if runner is None else f"{type(runner).__module__}.{type(runner).__name__}"

    def tllm_workload_status(self) -> dict[str, object]:
        """Report evidence from the currently active signed native Region epoch."""

        from tllm.runtime import residual_runtime as core

        compilation = core.RUNTIME.workload_compilation
        execution = core.RUNTIME.workload_execution
        if compilation is None or execution is None or compilation.physical is None:
            raise RuntimeError("worker has no active native workload epoch")
        return {
            "pid": os.getpid(),
            "signature": compilation.decision.signature,
            "plan_epoch": compilation.physical.plan_epoch,
            "artifacts": [
                {"region_id": region_id, "artifact_key": artifact_key}
                for region_id, artifact_key in execution.artifact_report
            ],
            "state_versions": [
                {"resource": resource, "key": key, "version": version}
                for resource, key, version in execution.state_version_report()
            ],
            "diagnostics": {
                workload_id: dict(metrics)
                for workload_id, metrics in execution.read_resource_diagnostics().items()
            },
        }

    def tllm_peek_shape_specialization_requests(self) -> dict[str, object]:
        """Expose pending cold-path Region misses without consuming them.

        The active signed plan identity is carried beside every request so a
        coordinator cannot mistakenly compile an old worker epoch.  Requests
        remain queued until a later exact acknowledgement.
        """

        from tllm.runtime import residual_runtime as core

        compilation = core.RUNTIME.workload_compilation
        execution = core.RUNTIME.workload_execution
        if compilation is None or execution is None or compilation.physical is None:
            raise RuntimeError("worker has no active native workload epoch")
        signature = _wire_sha256(
            compilation.decision.signature, "active workload signature"
        )
        plan_epoch = _wire_positive_int(
            compilation.physical.plan_epoch, "active workload plan_epoch"
        )
        requests = execution.peek_shape_specialization_requests()
        for request in requests:
            if (
                request.base_signature != signature
                or request.observed_plan_epoch != plan_epoch
            ):
                raise RuntimeError(
                    "worker pending shape-specialization request identity mismatch"
                )
        return {
            "signature": signature,
            "plan_epoch": plan_epoch,
            "requests": [request.to_wire() for request in requests],
        }

    def tllm_ack_shape_specialization_requests(
        self,
        signature: str,
        plan_epoch: int,
        dedup_keys: list[str],
    ) -> dict[str, object]:
        """Acknowledge exactly the misses consumed by a published new epoch."""

        from tllm.runtime import residual_runtime as core

        requested_signature = _wire_sha256(signature, "signature")
        requested_epoch = _wire_positive_int(plan_epoch, "plan_epoch")
        if type(dedup_keys) is not list:
            raise TypeError("dedup_keys must be a JSON array")
        keys = tuple(_wire_sha256(key, "dedup_key") for key in dedup_keys)
        if not keys:
            raise ValueError("dedup_keys cannot be empty")
        if len(keys) != len(set(keys)):
            raise ValueError("dedup_keys must be unique")

        compilation = core.RUNTIME.workload_compilation
        execution = core.RUNTIME.workload_execution
        if compilation is None or execution is None or compilation.physical is None:
            raise RuntimeError("worker has no active native workload epoch")
        active_signature = _wire_sha256(
            compilation.decision.signature, "active workload signature"
        )
        active_epoch = _wire_positive_int(
            compilation.physical.plan_epoch, "active workload plan_epoch"
        )
        if (
            requested_signature != active_signature
            or requested_epoch != active_epoch
        ):
            raise RuntimeError("shape-specialization acknowledgement identity mismatch")
        acknowledged = execution.acknowledge_shape_specialization_requests(keys)
        return {
            "signature": active_signature,
            "plan_epoch": active_epoch,
            "acknowledged": list(acknowledged),
        }

    def tllm_workload_timeline(self, configured_budget: int = 0) -> dict[str, object]:
        """Report immutable timeline and shape-specialization facts."""

        import json

        from tllm.runtime import residual_runtime as core
        from tllm.runtime.static_resources import static_device_headroom_bytes
        from tllm.schedule.timeline import export_main_workload_timeline

        resolution = core.RUNTIME.workgraph_backend_resolution
        if resolution is None:
            raise RuntimeError("worker has not resolved its workload backend timeline")
        runner = _runner_of(self)
        if runner is None:
            raise RuntimeError("worker has no model runner")
        facts = _runner_specialization_facts(runner)
        import torch

        raw_device = _runner_device_fact(runner)
        if raw_device is None:
            if configured_budget <= 0:
                raise RuntimeError("worker cannot resolve its static device fact")
            headroom = int(configured_budget)
            device = None
        else:
            device = torch.device(cast("torch.device | str | int", raw_device))
            headroom = static_device_headroom_bytes(
                device, configured_budget=int(configured_budget)
            )
        timeline = export_main_workload_timeline(
            resolution.capabilities,
            graph_mode=resolution.graph_mode.name.lower(),
            static_headroom_bytes=headroom,
            backend_version=resolution.backend_version,
        )
        specialization: dict[str, object] | None = None
        if facts is not None:
            specialization = {
                "device": str(facts.device),
                "dtype": str(facts.dtype),
                "hidden_size": facts.hidden_size,
                "rows": facts.rows,
                "max_prefill_tokens": facts.max_prefill_tokens,
                **(
                    {"model_layer_count": facts.model_layer_count}
                    if facts.model_layer_count is not None
                    else {}
                ),
            }
        return {
            "timeline": dict(json.loads(timeline.to_json())),
            "specialization": specialization,
        }

    def tllm_install_workload_decision(
        self,
        cache_key: str,
        graph_json: str,
        timeline_json: str,
        options_json: str,
        decision_json: str,
        plan_epoch: int = 1,
        reconstruction_specs: list[Mapping[str, object]] | None = None,
    ) -> dict[str, object]:
        """Compatibility one-shot install for older control-plane callers.

        New callers must use the explicit prepare/commit/abort RPCs below so
        no worker exposes a new epoch before every worker has reconstructed
        and reported the exact signed Region artifacts.
        """

        prepared = self.tllm_prepare_workload_decision(
            cache_key,
            graph_json,
            timeline_json,
            options_json,
            decision_json,
            plan_epoch,
            reconstruction_specs,
        )
        activated = False
        identity = _workload_identity(prepared["signature"], plan_epoch)
        try:
            committed = self.tllm_commit_workload_decision(
                identity[0], identity[1]
            )
            self.tllm_activate_workload_decision(
                identity[0], identity[1]
            )
            activated = True
            # The compatibility RPC has no distributed coordinator to roll
            # back against, so retain the historical one-shot lifetime.
            self.tllm_finalize_workload_decision(
                identity[0], identity[1]
            )
            return committed
        except Exception:
            if not activated:
                try:
                    self.tllm_abort_workload_decision(
                        identity[0], identity[1]
                    )
                except Exception:
                    # Preserve the primary prepare/commit/activation failure.
                    pass
            raise

    @staticmethod
    def _workload_install_report(
        staged: _StagedWorkloadDecision,
        *,
        committed: bool,
    ) -> dict[str, object]:
        execution = staged.execution
        artifacts = () if execution is None else execution.artifact_report
        return {
            "signature": staged.signature,
            "plan_epoch": staged.plan_epoch,
            "artifacts": [
                {"region_id": region_id, "artifact_key": artifact_key}
                for region_id, artifact_key in artifacts
            ],
            "native": execution is not None,
            "committed": committed,
        }

    @staticmethod
    def _close_staged_execution(staged: _StagedWorkloadDecision) -> None:
        execution = staged.execution
        if execution is not None:
            execution.close()

    def _staged_workload(self) -> _StagedWorkloadDecision | None:
        return self._tllm_staged_workload

    def _require_staged_workload(
        self,
        signature: object,
        plan_epoch: object,
    ) -> _StagedWorkloadDecision:
        identity = _workload_identity(signature, plan_epoch)
        staged = self._staged_workload()
        if staged is None:
            raise RuntimeError("worker has no prepared workload decision")
        if (staged.signature, staged.plan_epoch) != identity:
            raise RuntimeError("worker staged workload decision identity mismatch")
        return staged

    def tllm_prepare_workload_decision(
        self,
        cache_key: str,
        graph_json: str,
        timeline_json: str,
        options_json: str,
        decision_json: str,
        plan_epoch: int = 1,
        reconstruction_specs: list[Mapping[str, object]] | None = None,
    ) -> dict[str, object]:
        """Validate and materialize an immutable plan without publishing it."""

        cache_key = _wire_text(cache_key, "cache_key")
        graph_json = _wire_text(graph_json, "graph_json")
        timeline_json = _wire_text(timeline_json, "timeline_json")
        options_json = _wire_text(options_json, "options_json")
        decision_json = _wire_text(decision_json, "decision_json")
        plan_epoch = _wire_positive_int(plan_epoch, "plan_epoch")
        if self._staged_workload() is not None:
            raise RuntimeError(
                "worker already has a prepared workload decision; commit, abort, "
                "or finalize it before preparing another epoch"
            )

        from tllm.runtime import residual_runtime as core
        from tllm.runtime.workload_execution import build_native_workload_execution
        from tllm.schedule import install_workload_compilation
        from tllm.runtime.workload_plugins import (
            WorkloadWorkerSpec,
            reconstruct_worker_bindings,
        )

        compilation = install_workload_compilation(
            cache_key=cache_key,
            graph_json=graph_json,
            timeline_json=timeline_json,
            options_json=options_json,
            decision_json=decision_json,
            plan_epoch=plan_epoch,
        )
        specs = tuple(
            WorkloadWorkerSpec.from_wire(raw)
            for raw in (reconstruction_specs or ())
        )
        specialization = None
        if specs:
            runner = _runner_of(self)
            facts = _runner_specialization_facts(runner)
            if facts is not None:
                import torch

                from tllm.runtime.specialization import RuntimeSpecializationSpec

                specialization = RuntimeSpecializationSpec(
                    rows=facts.rows,
                    hidden_size=facts.hidden_size,
                    dtype=cast("torch.dtype", facts.dtype),
                    device=torch.device(
                        cast("torch.device | str | int", facts.device)
                    ),
                    model_layer_count=facts.model_layer_count,
                    max_prefill_tokens=facts.max_prefill_tokens,
                )
        bindings = (
            reconstruct_worker_bindings(
                compilation,
                specs,
                specialization=specialization,
            )
            if specs
            else None
        )
        needs_hidden_taps = (
            bindings is not None and _decision_requires_hidden_taps(compilation.decision)
        )
        runner = _runner_of(self)
        if needs_hidden_taps and not _runner_has_loaded_model(runner):
            raise RuntimeError(
                "native hidden-input workload requires a loaded worker model"
            )

        previous_native_graph = core.RUNTIME.native_schedule_graph
        previous_residual_paths = dict(core.RUNTIME.residual_raw_paths)
        previous_pending = core.RUNTIME.native_workloads_pending
        previous_compilation = core.RUNTIME.workload_compilation
        previous_execution = core.RUNTIME.workload_execution
        previous_execution_store = core.RUNTIME.workload_execution_store
        previous_physical = core.RUNTIME.physical_executor
        execution = None
        try:
            if bindings is not None:
                core.RUNTIME.native_schedule_graph = compilation.graph
                core.RUNTIME.native_workloads_pending = True
                core.RUNTIME.residual_raw_paths = dict(bindings.residual_paths)
            execution = build_native_workload_execution(
                compilation,
                runtime=core.RUNTIME,
                consumers=(),
                bindings=bindings,
            )
        except Exception:
            if execution is not None:
                execution.close()
            raise
        finally:
            # Prepare must leave the active runtime exactly untouched.
            core.RUNTIME.native_schedule_graph = previous_native_graph
            core.RUNTIME.residual_raw_paths = previous_residual_paths
            core.RUNTIME.native_workloads_pending = previous_pending

        staged = _StagedWorkloadDecision(
            signature=compilation.decision.signature,
            plan_epoch=int(plan_epoch),
            compilation=compilation,
            bindings=bindings,
            execution=execution,
            native_schedule_graph=previous_native_graph,
            residual_raw_paths=previous_residual_paths,
            native_workloads_pending=previous_pending,
            workload_compilation=previous_compilation,
            workload_execution=previous_execution,
            workload_execution_store=previous_execution_store,
            physical_executor=previous_physical,
        )
        self._tllm_staged_workload = staged
        return self._workload_install_report(staged, committed=False)

    def tllm_commit_workload_decision(
        self,
        signature: str,
        plan_epoch: int,
    ) -> dict[str, object]:
        """Vote to commit a prepared epoch without making it executable.

        ``tllm_activate_workload_decision`` is the only operation which makes
        the vote eligible for runtime publication.  This keeps both initial
        installs and replacements invisible until every rank has committed.
        """

        from tllm.runtime import residual_runtime as core
        from tllm.runtime.workload_execution import (
            NativeWorkloadExecution,
            NativeWorkloadExecutionStore,
        )

        staged = self._require_staged_workload(signature, plan_epoch)
        if staged.committed:
            return self._workload_install_report(staged, committed=True)
        compilation = staged.compilation
        bindings = staged.bindings
        execution = staged.execution
        previous = staged.workload_execution
        previous_store = staged.workload_execution_store
        stage_replacement = (
            isinstance(execution, NativeWorkloadExecution)
            and (
                previous_store is not None
                or isinstance(previous, NativeWorkloadExecution)
            )
        )
        if stage_replacement and bindings is not None:
            assert isinstance(previous, NativeWorkloadExecution)
            replacement_paths = dict(bindings.residual_paths)
            topology_changed = (
                replacement_paths != dict(staged.residual_raw_paths)
                or _decision_requires_hidden_taps(compilation.decision)
                != previous.requires_decode_hidden
            )
            if topology_changed:
                raise RuntimeError(
                    "native hook topology cannot be deferred to the epoch safe point"
                )
        try:
            if stage_replacement:
                if previous_store is not None:
                    if previous_store.current is not previous:
                        raise RuntimeError(
                            "worker execution store does not match active workload epoch"
                        )
                    publication_store = previous_store
                else:
                    assert isinstance(previous, NativeWorkloadExecution)
                    publication_store = NativeWorkloadExecutionStore(previous)
                    core.RUNTIME.workload_execution_store = publication_store
                assert isinstance(execution, NativeWorkloadExecution)
                publication_store.stage(execution, activation_required=True)
                staged.publication_store = publication_store
                staged.execution_staged_in_store = True
            else:
                if isinstance(execution, NativeWorkloadExecution):
                    staged.publication_store = NativeWorkloadExecutionStore(execution)
            staged.committed = True
        except Exception:
            hook_transaction = staged.hook_transaction
            if hook_transaction is not None:
                hook_transaction.rollback()
                staged.hook_transaction = None
            core.RUNTIME.native_schedule_graph = staged.native_schedule_graph
            core.RUNTIME.residual_raw_paths = dict(staged.residual_raw_paths)
            core.RUNTIME.native_workloads_pending = staged.native_workloads_pending
            core.RUNTIME.workload_compilation = staged.workload_compilation
            core.RUNTIME.workload_execution = staged.workload_execution
            publication_store = staged.publication_store
            if staged.execution_staged_in_store:
                if publication_store is not None:
                    publication_store.abort_staged()
            elif publication_store is not None:
                publication_store.close()
            core.RUNTIME.workload_execution_store = staged.workload_execution_store
            core.RUNTIME.physical_executor = staged.physical_executor
            staged.publication_store = None
            staged.execution_staged_in_store = False
            raise
        return self._workload_install_report(staged, committed=True)

    def tllm_activate_workload_decision(
        self,
        signature: str,
        plan_epoch: int,
    ) -> dict[str, object]:
        """Open a committed epoch only after the parent has all commit votes."""

        from tllm.runtime import residual_runtime as core
        from tllm.runtime.workload_execution import (
            NativeWorkloadExecution,
            StagedExecution,
        )

        identity = (
            _wire_sha256(signature, "signature"),
            _wire_positive_int(plan_epoch, "plan_epoch"),
        )
        staged = self._require_staged_workload(*identity)
        if not staged.committed:
            raise RuntimeError("cannot activate a workload decision before commit")
        if staged.activated:
            return {
                "signature": staged.signature,
                "plan_epoch": staged.plan_epoch,
                "activated": True,
            }

        execution = staged.execution
        store = staged.publication_store
        if staged.execution_staged_in_store:
            if store is None:
                raise RuntimeError("worker lost its staged workload execution store")
            match store.staged:
                case StagedExecution(execution=staged_execution) if staged_execution is execution:
                    pass
                case _:
                    raise RuntimeError("worker staged workload execution store diverged")
            store.activate_staged(identity[1])
        else:
            try:
                bindings = staged.bindings
                if bindings is not None:
                    core.RUNTIME.native_schedule_graph = staged.compilation.graph
                    core.RUNTIME.native_workloads_pending = True
                    core.RUNTIME.residual_raw_paths = dict(bindings.residual_paths)
                    if _decision_requires_hidden_taps(staged.compilation.decision):
                        runner = _runner_of(self)
                        if not _runner_has_loaded_model(runner):
                            raise RuntimeError(
                                "native hidden-input workload requires a loaded worker model"
                            )
                        from tllm.runtime.vllm_patch.port_runtime_hooks import (
                            setup_runtime_hooks_transactionally,
                        )

                        staged.hook_transaction = setup_runtime_hooks_transactionally(
                            core=core,
                            runner=_hook_runner(runner),
                        )
                if isinstance(execution, NativeWorkloadExecution):
                    if store is None or store.current is not execution:
                        raise RuntimeError("worker initial workload execution store diverged")
                    core.RUNTIME.workload_execution_store = store
                core.RUNTIME.workload_compilation = staged.compilation
                core.RUNTIME.workload_execution = execution
                core.RUNTIME.physical_executor = execution
                core.RUNTIME.native_workloads_pending = False
            except Exception:
                hook_transaction = staged.hook_transaction
                if hook_transaction is not None:
                    hook_transaction.rollback()
                    staged.hook_transaction = None
                core.RUNTIME.native_schedule_graph = staged.native_schedule_graph
                core.RUNTIME.residual_raw_paths = dict(staged.residual_raw_paths)
                core.RUNTIME.native_workloads_pending = staged.native_workloads_pending
                core.RUNTIME.workload_compilation = staged.workload_compilation
                core.RUNTIME.workload_execution = staged.workload_execution
                core.RUNTIME.workload_execution_store = staged.workload_execution_store
                core.RUNTIME.physical_executor = staged.physical_executor
                raise
        staged.activated = True
        return {
            "signature": staged.signature,
            "plan_epoch": staged.plan_epoch,
            "activated": True,
        }

    def tllm_abort_workload_decision(
        self,
        signature: str,
        plan_epoch: int,
    ) -> dict[str, object]:
        """Discard a prepared epoch or restore the retained prior active epoch."""

        from tllm.runtime import residual_runtime as core

        identity = _workload_identity(signature, plan_epoch)
        staged = self._staged_workload()
        if staged is None:
            # A collective prepare can fail after only a prefix of workers
            # stage.  Abort is deliberately idempotent for the remaining
            # workers so the coordinator can issue one broadcast cleanup.
            return {
                "signature": identity[0],
                "plan_epoch": identity[1],
                "aborted": False,
            }
        if (staged.signature, staged.plan_epoch) != identity:
            raise RuntimeError("worker staged workload decision identity mismatch")
        if staged.activated:
            raise RuntimeError("cannot abort a workload epoch after global activation")
        if staged.committed and staged.execution_staged_in_store:
            from tllm.runtime.workload_execution import StagedExecution

            publication_store = staged.publication_store
            if publication_store is None:
                raise RuntimeError("worker lost its staged workload execution store")
            match publication_store.staged:
                case StagedExecution(execution=execution) if execution is staged.execution:
                    pass
                case _:
                    if publication_store.current is staged.execution:
                        raise RuntimeError(
                            "cannot abort a workload epoch after safe-point promotion"
                        )
                    raise RuntimeError(
                        "worker workload execution store diverged before abort"
                    )
        if staged.committed:
            hook_transaction = staged.hook_transaction
            if hook_transaction is not None:
                hook_transaction.rollback()
                staged.hook_transaction = None
            core.RUNTIME.native_schedule_graph = staged.native_schedule_graph
            core.RUNTIME.residual_raw_paths = dict(staged.residual_raw_paths)
            core.RUNTIME.native_workloads_pending = staged.native_workloads_pending
            core.RUNTIME.workload_compilation = staged.workload_compilation
            core.RUNTIME.workload_execution = staged.workload_execution
            publication_store = staged.publication_store
            if staged.execution_staged_in_store:
                if publication_store is not None:
                    publication_store.abort_staged()
            elif publication_store is not None:
                publication_store.close()
            core.RUNTIME.workload_execution_store = staged.workload_execution_store
            core.RUNTIME.physical_executor = staged.physical_executor
        if not staged.execution_staged_in_store and staged.publication_store is None:
            self._close_staged_execution(staged)
        self._tllm_staged_workload = None
        return {
            "signature": identity[0],
            "plan_epoch": identity[1],
            "aborted": True,
        }

    def tllm_finalize_workload_decision(
        self,
        signature: str,
        plan_epoch: int,
    ) -> dict[str, object]:
        """Finalize only after the Store has promoted and drained this epoch."""

        identity = _workload_identity(signature, plan_epoch)
        staged = self._staged_workload()
        if staged is None:
            if self._tllm_last_finalized_workload == identity:
                return {
                    "signature": identity[0],
                    "plan_epoch": identity[1],
                    "finalized": True,
                    "drained": True,
                }
            if self._tllm_last_finalized_workload is not None:
                raise RuntimeError("worker finalized workload decision identity mismatch")
            raise RuntimeError("worker has no prepared workload decision")
        if staged.signature != identity[0] or staged.plan_epoch != identity[1]:
            raise RuntimeError("worker staged workload decision identity mismatch")
        if not staged.committed:
            raise RuntimeError("cannot finalize a workload decision before commit")
        if not staged.activated:
            raise RuntimeError("cannot finalize a workload decision before activation")
        store = staged.publication_store
        if store is not None:
            from tllm.runtime.workload_execution import NoStagedExecution

            match store.staged:
                case NoStagedExecution():
                    no_staged_execution = True
                case _:
                    no_staged_execution = False
            drained = (
                no_staged_execution
                and store.current.plan_epoch == identity[1]
                and store.retired_count == 0
            )
            if not drained:
                return {
                    "signature": staged.signature,
                    "plan_epoch": staged.plan_epoch,
                    "finalized": False,
                    "drained": False,
                }
        else:
            previous = staged.workload_execution
            if previous is not None and previous is not staged.execution:
                previous.close()
        hook_transaction = staged.hook_transaction
        if hook_transaction is not None:
            hook_transaction.finalize()
            staged.hook_transaction = None
        self._tllm_staged_workload = None
        self._tllm_last_finalized_workload = identity
        return {
            "signature": staged.signature,
            "plan_epoch": staged.plan_epoch,
            "finalized": True,
            "drained": True,
        }

    def tllm_drain_workload_epochs(
        self,
        signature: str,
        plan_epoch: int,
    ) -> dict[str, object]:
        """Release only Store epochs already retired by a runtime safe point."""

        identity = _workload_identity(signature, plan_epoch)
        staged = self._staged_workload()
        if staged is None:
            if (
                self._tllm_last_finalized_workload == identity
                or self._tllm_last_drained_workload == identity
            ):
                return {
                    "signature": identity[0],
                    "plan_epoch": identity[1],
                    "drained": True,
                    "closed_epochs": 0,
                }
            if (
                self._tllm_last_finalized_workload is not None
                or self._tllm_last_drained_workload is not None
            ):
                raise RuntimeError("worker drained workload decision identity mismatch")
            raise RuntimeError("worker has no prepared workload decision")
        if staged.signature != identity[0] or staged.plan_epoch != identity[1]:
            raise RuntimeError("worker staged workload decision identity mismatch")
        if not staged.committed:
            raise RuntimeError("cannot drain a workload decision before commit")
        if not staged.activated:
            raise RuntimeError("cannot drain a workload decision before activation")
        store = staged.publication_store
        from tllm.runtime.workload_execution import NoStagedExecution

        no_staged_execution = False
        if store is not None:
            match store.staged:
                case NoStagedExecution():
                    no_staged_execution = True
                case _:
                    pass
        if (
            store is None
            or not no_staged_execution
            or store.current.plan_epoch != identity[1]
        ):
            return {
                "signature": identity[0],
                "plan_epoch": identity[1],
                "drained": False,
                "closed_epochs": 0,
            }
        closed = store.collect_retired()
        drained = store.retired_count == 0
        if drained:
            self._tllm_last_drained_workload = identity
        return {
            "signature": identity[0],
            "plan_epoch": identity[1],
            "drained": drained,
            "closed_epochs": closed,
        }
