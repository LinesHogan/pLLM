"""vLLM adapter for generic immutable parameter overlays.

It translates generic versions, row assignments, and leases into vLLM model
state without owning workload computation, ordering, or scheduling policy.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING, Protocol

from tllm.backends.protocol import OverlayBackendState
from tllm.backends.vllm.dense_parameter_overlay import (
    DenseOverlayRunner,
    install_dense_replacement_target,
    validate_dense_replacement_targets,
)
from tllm.backends.vllm.runner_adapter import PackedRowPlanInputs
from tllm.contracts.request_identity import (
    RequestFinishedBatch,
    RequestIdentity,
    resolve_request_identity,
)
from tllm.runtime.parameter_overlay import (
    OverlayChannel,
    OverlayLeaseKey,
    OverlayLineage,
    OverlayRepresentation,
    OverlayRowAssignment,
    OverlayRowPhase,
    OverlayStepPlan,
    OverlayTarget,
    OverlayVersion,
    OverlayVersionRef,
    OverlayVersionStore,
)

if TYPE_CHECKING:
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.lora_weights import LoRALayerWeights
    from vllm.lora.request import LoRARequest


_IN_MEMORY_PATH = "__tllm_in_memory_overlay__"
_LORA_ID_BASE = 1_000_000_000


class _OverlayExecution(Protocol):
    @property
    def parameter_overlays(
        self,
    ) -> Mapping[OverlayChannel, OverlayVersionStore]: ...

    def stage_request_finished(self, batch: RequestFinishedBatch) -> int: ...


class _OverlayRuntime(Protocol):
    @property
    def workload_execution(self) -> _OverlayExecution | None: ...

    @property
    def reqid_to_promptidx(self) -> Mapping[str, int]: ...

    @property
    def reqid_to_sampleidx(self) -> Mapping[str, int]: ...


class _AdapterManager(Protocol):
    def list_adapters(self) -> Mapping[int, LoRAModel]: ...

    def add_adapter(self, model: LoRAModel) -> bool: ...

    def remove_adapter(self, lora_id: int) -> bool: ...


class _WorkerLoRAManager(Protocol):
    @property
    def _adapter_manager(self) -> _AdapterManager: ...


class _LoRAConfig(Protocol):
    @property
    def max_lora_rank(self) -> int: ...


class _RequestStates(Protocol):
    @property
    def req_id_to_index(self) -> Mapping[str, int]: ...


class _LoRAState(Protocol):
    def add_request(
        self, request_id: str, request_index: int, request: LoRARequest
    ) -> None: ...


class _IndexedIntBuffer(Protocol):
    def __setitem__(self, index: int, value: int) -> None: ...


class _MRV1InputBatch(Protocol):
    @property
    def req_id_to_index(self) -> Mapping[str, int]: ...

    @property
    def request_lora_mapping(self) -> _IndexedIntBuffer: ...

    @property
    def lora_id_to_request_ids(self) -> MutableMapping[int, set[str]]: ...

    @property
    def lora_id_to_lora_request(
        self,
    ) -> MutableMapping[int, LoRARequest]: ...


class _MRV2InputBatch(Protocol):
    @property
    def req_ids(self) -> Sequence[str]: ...

    @property
    def num_reqs(self) -> int: ...


class VLLMParameterOverlayRunner(DenseOverlayRunner, Protocol):
    @property
    def lora_manager(self) -> _WorkerLoRAManager: ...

    @property
    def lora_config(self) -> _LoRAConfig: ...

    @property
    def req_states(self) -> _RequestStates: ...

    @property
    def lora_state(self) -> _LoRAState: ...

    @property
    def input_batch(self) -> _MRV1InputBatch: ...

class VLLMParameterOverlayController:
    def __init__(
        self,
        *,
        runtime: _OverlayRuntime,
        runner: VLLMParameterOverlayRunner,
        state: OverlayBackendState,
    ) -> None:
        self.runtime = runtime
        self.runner = runner
        self.state = state
        self.stores = state.stores

    @property
    def adapter_manager(self) -> _AdapterManager:
        return self.runner.lora_manager._adapter_manager  # pyright: ignore[reportPrivateUsage] -- vLLM exposes in-memory model installation only on its internal manager

    @property
    def max_rank(self) -> int:
        return self.runner.lora_config.max_lora_rank

    def _global_lora_lineages(
        self,
    ) -> tuple[tuple[OverlayVersionStore, OverlayLineage], ...]:
        result: list[tuple[OverlayVersionStore, OverlayLineage]] = []
        for _channel, store in sorted(
            self.stores.items(), key=lambda item: (item[0].namespace, item[0].name)
        ):
            for lineage in store.lineages():
                latest = store.latest(lineage)
                if (
                    lineage.channel == _channel
                    and
                    lineage.request is None
                    and latest is not None
                    and latest.manifest.representation
                    is OverlayRepresentation.LOW_RANK_ADD
                ):
                    result.append((store, lineage))
        return tuple(result)

    def owns_stores(
        self, stores: Mapping[OverlayChannel, OverlayVersionStore]
    ) -> bool:
        return self.state.owns_stores(stores)

    def validate_dense_targets(self) -> None:
        if self.state.dense_targets_validated:
            return
        targets = tuple(
            channel.target
            for channel in self.stores
            if channel.target is not None
        )
        if targets:
            validate_dense_replacement_targets(self.runner, targets)
        self.state.dense_targets_validated = True

    def close(self) -> None:
        if self.state.request_identities:
            raise RuntimeError(
                "cannot rebind parameter overlays while requests are active"
            )
        failures: list[Exception] = []
        for handle in self.state.dense_targets.values():
            try:
                handle.restore()
            except Exception as error:
                failures.append(error)
        for lora_id in self.state.version_handles.values():
            try:
                self.adapter_manager.remove_adapter(lora_id)
            except Exception as error:
                failures.append(error)
        if failures:
            raise RuntimeError(
                "could not remove all rebound parameter overlays"
            ) from failures[0]

    def _lora_id(self, ref: OverlayVersionRef) -> int:
        existing = self.state.version_handles.get(ref)
        if existing is not None:
            return existing
        value = _LORA_ID_BASE + self.state.allocate_handle_ordinal()
        self.state.version_handles[ref] = value
        return value

    @staticmethod
    def _canonical_module_name(parameter_name: str) -> str:
        prefix = "base_model.model."
        name = (
            parameter_name[len(prefix) :]
            if parameter_name.startswith(prefix)
            else parameter_name
        )
        return name.removesuffix(".lora_A.default.weight")

    def _build_model(
        self,
        store: OverlayVersionStore,
        version: OverlayVersion,
    ) -> LoRAModel:
        from vllm.lora.lora_model import LoRAModel
        from vllm.lora.lora_weights import LoRALayerWeights

        tensors = store.materialize(version.ref)
        alpha = int(tensors["__tllm_lora_alpha__"].item())
        a_suffix = ".lora_A.default.weight"
        loras: dict[str, LoRALayerWeights] = {}
        rank = 0
        for name, lora_a in tensors.items():
            if not name.endswith(a_suffix):
                continue
            lora_b = tensors[name.removesuffix(a_suffix) + ".lora_B.default.weight"]
            module_name = self._canonical_module_name(name)
            rank = int(lora_a.shape[0])
            loras[module_name] = LoRALayerWeights(
                module_name=module_name,
                rank=rank,
                lora_alpha=alpha,
                lora_a=lora_a,
                lora_b=lora_b,
                scaling=alpha / rank,
            )
        if rank > self.max_rank:
            raise RuntimeError(
                f"overlay LoRA rank {rank} exceeds vLLM max_lora_rank={self.max_rank}"
            )
        return LoRAModel(self._lora_id(version.ref), rank, loras)

    def _request_for(self, version: OverlayVersion) -> LoRARequest:
        from vllm.lora.request import LoRARequest

        return LoRARequest(
            lora_name=(
                f"{version.ref.lineage.channel.namespace}/"
                f"{version.ref.lineage.channel.name}:v{version.ref.version}"
            ),
            lora_int_id=self._lora_id(version.ref),
            lora_path=_IN_MEMORY_PATH,
        )

    def _materialize(
        self, store: OverlayVersionStore, version: OverlayVersion
    ) -> None:
        lora_id = self._lora_id(version.ref)
        if lora_id not in self.adapter_manager.list_adapters():
            self.adapter_manager.add_adapter(self._build_model(store, version))

    def sync_versions(self) -> int:
        added = 0
        active_lora_refs: set[OverlayVersionRef] = set()
        for store, lineage in self._global_lora_lineages():
            for version in store.versions(lineage):
                active_lora_refs.add(version.ref)
                lora_id = self._lora_id(version.ref)
                if lora_id not in self.adapter_manager.list_adapters():
                    self._materialize(store, version)
                    added += 1
        for ref, lora_id in tuple(self.state.version_handles.items()):
            if ref not in active_lora_refs:
                self.adapter_manager.remove_adapter(lora_id)
                self.state.version_handles.pop(ref)

        active_dense_refs: dict[OverlayTarget, set[OverlayVersionRef]] = {}
        for channel, store in sorted(
            self.stores.items(),
            key=lambda item: (item[0].namespace, item[0].name),
        ):
            target = channel.target
            if target is None:
                continue
            for lineage in store.lineages():
                if lineage.channel != channel:
                    continue
                latest = store.latest(lineage)
                if (
                    latest is None
                    or latest.manifest.representation
                    is not OverlayRepresentation.DENSE_REPLACE
                ):
                    continue
                handle = self.state.dense_targets.get(target)
                if handle is None:
                    handle = install_dense_replacement_target(
                        self.runner, target
                    )
                    self.state.dense_targets[target] = handle
                target_refs = active_dense_refs.setdefault(target, set())
                for version in store.versions(lineage):
                    target_refs.add(version.ref)
                    if not handle.has_version(version.ref):
                        handle.install_version(
                            version, store.materialize(version.ref)
                        )
                        added += 1
        for target, handle in tuple(self.state.dense_targets.items()):
            refs = frozenset(active_dense_refs.get(target, set()))
            if refs:
                handle.retain_versions(refs)
            else:
                handle.restore()
                self.state.dense_targets.pop(target)
                self.state.dense_reported_rows.pop(target, None)
        return added

    def has_dense_versions(self) -> bool:
        return bool(self.state.dense_targets)

    def prepare_dense_step(
        self,
        *,
        step_id: int,
        columns: PackedRowPlanInputs,
    ) -> OverlayStepPlan:
        from tllm import _core

        row_plan = _core.plan_packed_prefill_rows(
            list(columns.scheduled_tokens),
            list(columns.prompt_tokens),
            list(columns.computed_tokens),
        )
        rows: dict[tuple[OverlayTarget, OverlayVersionRef], list[int]] = {}
        phases: dict[
            tuple[OverlayTarget, OverlayVersionRef], list[OverlayRowPhase]
        ] = {}
        dense_channels = tuple(
            (channel, store)
            for channel, store in self.stores.items()
            if channel.target is not None
        )
        for request_ordinal, packed_row, phase in zip(
            row_plan.request_ordinals,
            row_plan.packed_row_indices,
            row_plan.phases,
            strict=True,
        ):
            request_id = columns.request_ids[request_ordinal]
            identity = self._identity(request_id)
            for channel, store in dense_channels:
                target = channel.target
                if target is None:
                    continue
                lineage = OverlayLineage(channel, identity)
                lease = store.pinned(OverlayLeaseKey(identity, lineage))
                if lease is None:
                    continue
                key = (target, lease.ref)
                rows.setdefault(key, []).append(packed_row)
                phases.setdefault(key, []).append(OverlayRowPhase(phase))
                self.state.dense_request_ids.add(request_id)
        assignments = tuple(
            OverlayRowAssignment(
                target=target,
                version=ref,
                packed_rows=tuple(rows[(target, ref)]),
                phases=tuple(phases[(target, ref)]),
            )
            for target, ref in sorted(
                rows,
                key=lambda item: (
                    item[0].layer,
                    item[0].module,
                    item[0].parameter,
                    item[1].epoch,
                    item[1].version,
                ),
            )
        )
        plan = OverlayStepPlan(step_id, row_plan.total_rows, assignments)
        for handle in self.state.dense_targets.values():
            handle.prepare_step(plan)
        return plan

    def clear_dense_step(self) -> None:
        for target, handle in self.state.dense_targets.items():
            applied = handle.apply_count
            reported = self.state.dense_reported_rows.get(target, 0)
            delta = applied - reported
            if delta:
                for channel, store in self.stores.items():
                    if channel.target == target:
                        store.record_backend_application(delta)
                self.state.dense_reported_rows[target] = applied
            handle.clear_step()

    def _identity(self, request_id: str) -> RequestIdentity:
        identity = resolve_request_identity(
            request_id,
            prompt_indices=self.runtime.reqid_to_promptidx,
            sample_indices=self.runtime.reqid_to_sampleidx,
        )
        resolved = (
            RequestIdentity(request_id, -1, 0) if identity is None else identity
        )
        self.state.request_identities[request_id] = resolved
        return resolved

    def pin_request(self, request_id: str) -> LoRARequest | None:
        lineages = self._global_lora_lineages()
        if not lineages:
            return None
        if len(lineages) != 1:
            raise RuntimeError("vLLM supports one global low-rank overlay per request")
        store, lineage = lineages[0]
        identity = self._identity(request_id)
        lease = store.pin_latest(OverlayLeaseKey(identity, lineage))
        if lease is None:
            return None
        version = store.version(lease.ref)
        self._materialize(store, version)
        return self._request_for(version)

    def release_finished(
        self, request_ids: tuple[str, ...]
    ) -> RequestFinishedBatch:
        finished: list[RequestIdentity] = []
        observed: set[RequestIdentity] = set()
        for request_id in request_ids:
            identity = self.state.request_identities.get(request_id)
            if identity is None:
                identity = self._identity(request_id)
            aliases = tuple(
                raw_id
                for raw_id, raw_identity in self.state.request_identities.items()
                if raw_identity == identity
            )
            for raw_id in aliases:
                self.state.request_identities.pop(raw_id)
                self.state.attached_requests.discard(raw_id)
                self.state.dense_request_ids.discard(raw_id)
            self.state.attached_requests.discard(request_id)
            self.state.dense_request_ids.discard(request_id)
            if identity in observed:
                continue
            observed.add(identity)
            finished.append(identity)
            stores = set(self.stores.values())
            for store in stores:
                for lineage in store.lineages():
                    if lineage.request is None:
                        store.release(OverlayLeaseKey(identity, lineage))
        return RequestFinishedBatch(tuple(finished))

    def finish_requests(
        self, request_ids: tuple[str, ...]
    ) -> RequestFinishedBatch:
        batch = self.release_finished(request_ids)
        execution = self.runtime.workload_execution
        if execution is not None and batch.requests:
            requires_signed_cleanup = any(
                lineage.request in batch.requests
                for store in set(self.stores.values())
                for lineage in store.lineages()
            )
            if requires_signed_cleanup:
                staged = execution.stage_request_finished(batch)
                if staged < 1:
                    raise RuntimeError(
                        "request-scoped overlays require a signed request-finished port"
                    )
        return batch

    def apply_mrv1_input_batch(self, input_batch: _MRV1InputBatch) -> int:
        applied = 0
        for request_id, request_index in input_batch.req_id_to_index.items():
            request = self.pin_request(request_id)
            if request is None:
                continue
            lora_id = request.lora_int_id
            input_batch.request_lora_mapping[request_index] = lora_id
            input_batch.lora_id_to_request_ids.setdefault(lora_id, set()).add(
                request_id
            )
            input_batch.lora_id_to_lora_request[lora_id] = request
            applied += 1
        return applied

    def apply_mrv2_input_batch(self, input_batch: _MRV2InputBatch) -> int:
        applied = 0
        request_ids = tuple(input_batch.req_ids[: input_batch.num_reqs])
        active = frozenset(request_ids)
        self.finish_requests(
            tuple(
                request_id
                for request_id in (
                    self.state.attached_requests
                    | self.state.dense_request_ids
                )
                if request_id not in active
            )
        )
        for request_id in request_ids:
            if request_id in self.state.attached_requests:
                continue
            request = self.pin_request(request_id)
            if request is None:
                continue
            request_index = self.runner.req_states.req_id_to_index[request_id]
            self.runner.lora_state.add_request(request_id, request_index, request)
            self.state.attached_requests.add(request_id)
            applied += 1
        return applied

def maybe_bind_publisher(
    *,
    runtime: _OverlayRuntime,
    runner: VLLMParameterOverlayRunner,
    state: OverlayBackendState | None,
) -> VLLMParameterOverlayController | None:
    execution = runtime.workload_execution
    stores: Mapping[OverlayChannel, OverlayVersionStore] = (
        {} if execution is None else execution.parameter_overlays
    )
    if state is None:
        if stores:
            raise RuntimeError(
                "parameter overlays require an active runner binding"
            )
        return None
    if not stores and not state.stores:
        return None
    controller = VLLMParameterOverlayController(
        runtime=runtime,
        runner=runner,
        state=state,
    )
    if not controller.owns_stores(stores):
        controller.close()
        state.rebind(stores)
        if not stores:
            return None
        controller = VLLMParameterOverlayController(
            runtime=runtime,
            runner=runner,
            state=state,
        )
    controller.validate_dense_targets()
    return controller


def unbind_parameter_overlays(
    *, runner: VLLMParameterOverlayRunner, state: OverlayBackendState
) -> None:
    failures: list[Exception] = []
    for handle in state.dense_targets.values():
        try:
            handle.restore()
        except Exception as error:
            failures.append(error)
    if state.version_handles:
        adapter_manager = runner.lora_manager._adapter_manager  # pyright: ignore[reportPrivateUsage] -- vLLM exposes in-memory model removal only on its internal manager
        for lora_id in state.version_handles.values():
            try:
                adapter_manager.remove_adapter(lora_id)
            except Exception as error:
                failures.append(error)
    if failures:
        raise RuntimeError(
            "could not release all parameter overlays during runner unbind"
        ) from failures[0]
    state.rebind({})


def finish_parameter_overlay_requests(
    *,
    runtime: _OverlayRuntime,
    runner: VLLMParameterOverlayRunner,
    state: OverlayBackendState,
    request_ids: tuple[str, ...],
) -> RequestFinishedBatch:
    controller = maybe_bind_publisher(
        runtime=runtime, runner=runner, state=state
    )
    if controller is None:
        return RequestFinishedBatch(())
    return controller.finish_requests(request_ids)


def synchronize_parameter_overlays(
    *,
    runtime: _OverlayRuntime,
    runner: VLLMParameterOverlayRunner,
    state: OverlayBackendState,
) -> int:
    controller = maybe_bind_publisher(
        runtime=runtime, runner=runner, state=state
    )
    return 0 if controller is None else controller.sync_versions()

__all__ = [
    "VLLMParameterOverlayRunner",
    "VLLMParameterOverlayController",
    "finish_parameter_overlay_requests",
    "maybe_bind_publisher",
    "synchronize_parameter_overlays",
    "unbind_parameter_overlays",
]
