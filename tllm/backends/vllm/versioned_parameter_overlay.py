"""vLLM adapter for generic immutable low-rank parameter overlays.

This module is backend quarantine: it translates generic overlay versions and
leases into vLLM's native LoRA manager without owning training or ordering.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from tllm.backends.protocol import runner_binding_for
from tllm.contracts.request_identity import RequestIdentity, resolve_request_identity
from tllm.runtime.parameter_overlay import (
    OverlayChannel,
    OverlayLineage,
    OverlayRepresentation,
    OverlayVersion,
    OverlayVersionRef,
    OverlayVersionStore,
)


_IN_MEMORY_PATH = "__tllm_in_memory_overlay__"
_LORA_ID_BASE = 1_000_000_000
_CONTROLLER_KEY = "tllm.versioned_parameter_overlay"


class VLLMParameterOverlayController:
    def __init__(
        self,
        *,
        runtime: object,
        runner: object,
        stores: Mapping[OverlayChannel, OverlayVersionStore],
    ) -> None:
        worker_manager = runner.lora_manager
        self.runtime = runtime
        self.runner = runner
        self.stores = stores
        self.adapter_manager = worker_manager._adapter_manager
        self.max_rank = int(runner.lora_config.max_lora_rank)
        self._ids: dict[OverlayVersionRef, int] = {}
        self._requests: dict[OverlayVersionRef, object] = {}
        self._request_identities: dict[str, RequestIdentity] = {}
        self._attached_requests: set[str] = set()

    def _global_lora_lineages(
        self,
    ) -> tuple[tuple[OverlayVersionStore, OverlayLineage], ...]:
        result: list[tuple[OverlayVersionStore, OverlayLineage]] = []
        for channel, store in sorted(
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
        return self.stores.keys() == stores.keys() and all(
            self.stores[channel] is store for channel, store in stores.items()
        )

    def _lora_id(self, ref: OverlayVersionRef) -> int:
        existing = self._ids.get(ref)
        if existing is not None:
            return existing
        value = _LORA_ID_BASE + len(self._ids) + 1
        self._ids[ref] = value
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
    ) -> object:
        from vllm.lora.lora_model import LoRAModel
        from vllm.lora.lora_weights import LoRALayerWeights

        tensors = store.materialize(version.ref)
        alpha = int(tensors["__tllm_lora_alpha__"].item())
        a_suffix = ".lora_A.default.weight"
        loras: dict[str, object] = {}
        rank = 0
        for name, lora_a in tensors.items():
            if not name.endswith(a_suffix):
                continue
            lora_b = tensors[name.removesuffix(a_suffix) + ".lora_B.default.weight"]
            module_name = self._canonical_module_name(name)
            rank = lora_a.shape[0]
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

    def _request_for(self, version: OverlayVersion) -> object:
        request = self._requests.get(version.ref)
        if request is not None:
            return request
        from vllm.lora.request import LoRARequest

        request = LoRARequest(
            lora_name=(
                f"{version.ref.lineage.channel.namespace}/"
                f"{version.ref.lineage.channel.name}:v{version.ref.version}"
            ),
            lora_int_id=self._lora_id(version.ref),
            lora_path=_IN_MEMORY_PATH,
        )
        self._requests[version.ref] = request
        return request

    def _materialize(
        self, store: OverlayVersionStore, version: OverlayVersion
    ) -> None:
        lora_id = self._lora_id(version.ref)
        if lora_id not in self.adapter_manager.list_adapters():
            self.adapter_manager.add_adapter(self._build_model(store, version))
        self._request_for(version)

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
        self._request_identities[request_id] = resolved
        return resolved

    def pin_request(self, request_id: str) -> object | None:
        lineages = self._global_lora_lineages()
        if not lineages:
            return None
        if len(lineages) != 1:
            raise RuntimeError("vLLM supports one global low-rank overlay per request")
        store, lineage = lineages[0]
        identity = self._identity(request_id)
        lease = store.pin_latest(identity, lineage)
        if lease is None:
            return None
        version = store.version(lease.ref)
        self._materialize(store, version)
        return self._request_for(version)

    def release_finished(self, request_ids: tuple[str, ...]) -> int:
        released = 0
        for request_id in request_ids:
            identity = self._request_identities.pop(request_id, None)
            self._attached_requests.discard(request_id)
            if identity is None:
                continue
            for store in self.stores.values():
                released += int(store.release(identity))
        return released

    def apply_mrv1_input_batch(self, input_batch: object) -> int:
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

    def apply_mrv2_input_batch(self, input_batch: object) -> int:
        applied = 0
        request_ids = tuple(input_batch.req_ids[: input_batch.num_reqs])
        active = frozenset(request_ids)
        self.release_finished(
            tuple(request_id for request_id in self._attached_requests if request_id not in active)
        )
        for request_id in request_ids:
            if request_id in self._attached_requests:
                continue
            request = self.pin_request(request_id)
            if request is None:
                continue
            request_index = self.runner.req_states.req_id_to_index[request_id]
            self.runner.lora_state.add_request(request_id, request_index, request)
            self._attached_requests.add(request_id)
            applied += 1
        return applied

def maybe_bind_publisher(
    *, runtime: object, runner: object
) -> VLLMParameterOverlayController | None:
    execution = runtime.workload_execution
    if execution is None or not execution.parameter_overlays:
        return None
    binding = runner_binding_for(runner)
    if binding is None:
        raise RuntimeError("parameter overlays require an active runner binding")
    existing = binding.parameter_overlay_handles.get(_CONTROLLER_KEY)
    if existing is not None and existing.owns_stores(execution.parameter_overlays):
        return existing
    controller = VLLMParameterOverlayController(
        runtime=runtime,
        runner=runner,
        stores=execution.parameter_overlays,
    )
    binding.parameter_overlay_handles[_CONTROLLER_KEY] = controller
    return controller


def sync_mrv1_before_prepare(
    *, runtime: object, runner: object, scheduler_output: object
) -> int:
    controller = maybe_bind_publisher(
        runtime=runtime, runner=runner
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
