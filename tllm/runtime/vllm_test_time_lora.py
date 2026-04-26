#!/usr/bin/env python3
"""vLLM-facing helpers for in-memory test-time LoRA updates."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from tllm.contracts.test_time_lora import TestTimeLoRAUpdate, TestTimeLoRAWeights
from tllm.runtime.test_time_lora import TestTimeLoRAService

__test__ = False

_IN_MEMORY_LORA_PATH = "__tllm_in_memory_lora__"


class VLLMInMemoryLoRAManager:
    """Adapter around vLLM's worker LoRA manager.

    vLLM routes requests by LoRA adapter id, then maps that id to an internal
    GPU slot. This adapter registers a LoRAModel built from already-materialized
    tensors, so vLLM's normal mapping path can run without loading PEFT files.
    """

    def __init__(self, worker_manager: object) -> None:
        adapter_manager = getattr(worker_manager, "_adapter_manager", None)
        if adapter_manager is None:
            raise RuntimeError("vLLM LoRA worker manager does not expose _adapter_manager")
        self.worker_manager = worker_manager
        self.adapter_manager = adapter_manager
        self._requests: dict[int, object] = {}
        self._adapter_keys: dict[int, str] = {}
        self._pending_models: dict[int, object] = {}

    @property
    def modules(self) -> Mapping[str, object]:
        modules = getattr(self.adapter_manager, "modules", None)
        if not isinstance(modules, Mapping):
            raise RuntimeError("vLLM LoRA adapter manager does not expose modules")
        return modules

    def request_for_lora_id(self, lora_id: int) -> object:
        try:
            return self._requests[lora_id]
        except KeyError as exc:
            raise RuntimeError(f"unknown in-memory LoRA id {lora_id}") from exc

    def commit_update(self, update: TestTimeLoRAUpdate) -> None:
        if not isinstance(update, TestTimeLoRAUpdate):
            raise TypeError("update must be TestTimeLoRAUpdate")
        if update.scope != "request":
            raise RuntimeError("vLLM in-memory LoRA currently requires request-scoped updates")
        lora_model = self._build_lora_model(update)
        self._pending_models[update.lora_int_id] = lora_model
        self._requests[update.lora_int_id] = self._make_lora_request(update)
        self._adapter_keys[update.lora_int_id] = update.adapter_key

    def materialize_adapter(self, lora_id: int) -> None:
        list_adapters = getattr(self.adapter_manager, "list_adapters", None)
        if not callable(list_adapters):
            raise RuntimeError("vLLM LoRA adapter manager lacks list_adapters")
        lora_model = self._pending_models.get(lora_id)
        if lora_model is None and lora_id in list_adapters():
            return
        if lora_model is None:
            raise RuntimeError(f"in-memory LoRA id {lora_id} has no pending model")
        self._replace_registered_adapter(lora_model)
        self._pending_models.pop(lora_id, None)

    def _replace_registered_adapter(self, lora_model: object) -> None:
        adapter_id = int(getattr(lora_model, "id"))
        list_adapters = getattr(self.adapter_manager, "list_adapters", None)
        remove_adapter = getattr(self.adapter_manager, "remove_adapter", None)
        add_adapter = getattr(self.adapter_manager, "add_adapter", None)
        if not callable(list_adapters) or not callable(add_adapter):
            raise RuntimeError("vLLM LoRA adapter manager lacks adapter registration methods")
        if adapter_id in list_adapters():
            if not callable(remove_adapter):
                raise RuntimeError("vLLM LoRA adapter manager cannot replace existing adapters")
            remove_adapter(adapter_id)
        add_adapter(lora_model)

    def _build_lora_model(self, update: TestTimeLoRAUpdate) -> object:
        from vllm.lora.lora import LoRALayerWeights
        from vllm.lora.models import LoRAModel

        lora_model = LoRAModel(update.lora_int_id, update.rank, {})
        for module_name, weights in update.target_modules.items():
            if module_name in getattr(self.adapter_manager, "packed_modules", {}):
                raise RuntimeError("packed vLLM LoRA modules are not supported by this in-memory path yet")
            lora_model.loras[module_name] = _to_vllm_layer_weights(module_name, update.rank, weights, LoRALayerWeights)
        return lora_model

    def _make_lora_request(self, update: TestTimeLoRAUpdate) -> object:
        from vllm.lora.request import LoRARequest

        return LoRARequest(
            lora_name=update.adapter_key,
            lora_int_id=update.lora_int_id,
            lora_path=_IN_MEMORY_LORA_PATH,
        )


def _to_vllm_layer_weights(module_name: str, rank: int, weights: TestTimeLoRAWeights, cls: type[object]) -> object:
    lora_b = weights.lora_b.detach().clone()
    if weights.scaling != 1.0:
        lora_b.mul_(weights.scaling)
    return cls(
        module_name=module_name,
        rank=rank,
        lora_alpha=rank,
        lora_a=weights.lora_a.detach().clone(),
        lora_b=lora_b,
        bias=None if weights.bias is None else weights.bias.detach().clone(),
        embeddings_tensor=None if weights.embeddings_tensor is None else weights.embeddings_tensor.detach().clone(),
        scaling=1.0,
    )


def maybe_bind_test_time_lora_service(*, runtime: object, runner: object) -> TestTimeLoRAService | None:
    if getattr(runtime, "test_time_lora_service", None) is not None:
        return getattr(runtime, "test_time_lora_service")
    worker_manager = getattr(runner, "lora_manager", None)
    if worker_manager is None:
        return None
    adapter_manager = getattr(worker_manager, "_adapter_manager", None)
    lora_config = getattr(worker_manager, "lora_config", None)
    if adapter_manager is None or lora_config is None:
        return None
    service = TestTimeLoRAService(
        manager=VLLMInMemoryLoRAManager(worker_manager),
        max_rank=getattr(lora_config, "max_lora_rank"),
    )
    setattr(runtime, "test_time_lora_service", service)
    return service


def maybe_commit_and_apply_request_mapping(*, runtime: object, runner: object) -> int:
    service = getattr(runtime, "test_time_lora_service", None)
    if not isinstance(service, TestTimeLoRAService):
        return 0
    committed = service.commit_staged_updates()
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        return committed
    return committed + apply_request_lora_mapping(service=service, input_batch=input_batch)


def apply_request_lora_mapping(*, service: TestTimeLoRAService, input_batch: object) -> int:
    if not isinstance(service.manager, VLLMInMemoryLoRAManager):
        raise TypeError("service manager must be VLLMInMemoryLoRAManager")
    req_id_to_index = getattr(input_batch, "req_id_to_index", None)
    request_lora_mapping = getattr(input_batch, "request_lora_mapping", None)
    lora_id_to_request_ids = getattr(input_batch, "lora_id_to_request_ids", None)
    lora_id_to_lora_request = getattr(input_batch, "lora_id_to_lora_request", None)
    if not isinstance(req_id_to_index, Mapping) or request_lora_mapping is None:
        raise RuntimeError("vLLM input batch does not expose LoRA request mapping")
    if not isinstance(lora_id_to_request_ids, dict) or not isinstance(lora_id_to_lora_request, dict):
        raise RuntimeError("vLLM input batch does not expose LoRA request registries")

    applied = 0
    for request_id, lora_id in service.request_assignments().items():
        req_index = req_id_to_index.get(request_id)
        if req_index is None:
            continue
        previous_lora_id = int(request_lora_mapping[int(req_index)])
        if previous_lora_id != 0 and previous_lora_id != int(lora_id):
            previous_request_ids = lora_id_to_request_ids.get(previous_lora_id)
            if isinstance(previous_request_ids, set):
                previous_request_ids.discard(request_id)
                if not previous_request_ids:
                    lora_id_to_request_ids.pop(previous_lora_id, None)
                    lora_id_to_lora_request.pop(previous_lora_id, None)
        service.manager.materialize_adapter(int(lora_id))
        request_lora_mapping[int(req_index)] = int(lora_id)
        lora_id_to_request_ids.setdefault(int(lora_id), set()).add(request_id)
        lora_id_to_lora_request[int(lora_id)] = service.manager.request_for_lora_id(int(lora_id))
        applied += 1
    return applied
