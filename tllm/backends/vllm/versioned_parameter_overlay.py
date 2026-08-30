"""vLLM adapter for generic immutable low-rank parameter overlays.

This module is backend quarantine: it translates generic overlay versions and
leases into vLLM's native LoRA manager without owning training or ordering.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING, Protocol

from tllm.backends.protocol import OverlayBackendState
from tllm.contracts.request_identity import RequestIdentity, resolve_request_identity
from tllm.runtime.parameter_overlay import (
    OverlayChannel,
    OverlayLeaseKey,
    OverlayLineage,
    OverlayRepresentation,
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


class _SchedulerOutput(Protocol):
    @property
    def finished_req_ids(self) -> Sequence[str]: ...


class _OverlayRunner(Protocol):
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
        runner: _OverlayRunner,
        state: OverlayBackendState,
    ) -> None:
        worker_manager = runner.lora_manager
        self.runtime = runtime
        self.runner = runner
        self.state = state
        self.stores = state.stores
        self.adapter_manager = worker_manager._adapter_manager  # pyright: ignore[reportPrivateUsage] -- vLLM exposes in-memory model installation only on its internal manager
        self.max_rank = runner.lora_config.max_lora_rank

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

    def close(self) -> None:
        self.release_finished(tuple(self.state.request_identities))
        failures: list[Exception] = []
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
        for store, lineage in self._global_lora_lineages():
            for version in store.versions(lineage):
                lora_id = self._lora_id(version.ref)
                if lora_id not in self.adapter_manager.list_adapters():
                    self._materialize(store, version)
                    added += 1
        return added

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

    def release_finished(self, request_ids: tuple[str, ...]) -> int:
        released = 0
        for request_id in request_ids:
            identity = self.state.request_identities.pop(request_id, None)
            self.state.attached_requests.discard(request_id)
            if identity is None:
                continue
            stores = set(self.stores.values())
            for store in stores:
                released += store.retire_request(identity).released_leases
        return released

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
        self.release_finished(
            tuple(
                request_id
                for request_id in self.state.attached_requests
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
    runner: _OverlayRunner,
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
    return controller


def sync_mrv1_before_prepare(
    *,
    runtime: _OverlayRuntime,
    runner: _OverlayRunner,
    state: OverlayBackendState | None,
    scheduler_output: _SchedulerOutput,
) -> int:
    controller = maybe_bind_publisher(
        runtime=runtime, runner=runner, state=state
    )
    if controller is None:
        return 0
    controller.release_finished(tuple(scheduler_output.finished_req_ids))
    controller.sync_versions()
    return controller.apply_mrv1_input_batch(runner.input_batch)


__all__ = [
    "VLLMParameterOverlayController",
    "maybe_bind_publisher",
    "sync_mrv1_before_prepare",
]
