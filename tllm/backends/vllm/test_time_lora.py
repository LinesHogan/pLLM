#!/usr/bin/env python3
"""vLLM 0.27 materialization for immutable test-time LoRA versions."""

from __future__ import annotations

from collections.abc import Mapping
from types import MethodType

from tllm.runtime.test_time_lora import LoRAVersion, LoRAVersionStore


_IN_MEMORY_PATH = "__tllm_in_memory_lora__"
_LORA_ID_BASE = 1_000_000_000


class VLLMVersionedLoRAPublisher:
    def __init__(self, *, runner: object, store: LoRAVersionStore) -> None:
        worker_manager = getattr(runner, "lora_manager", None)
        lora_config = getattr(runner, "lora_config", None)
        adapter_manager = getattr(worker_manager, "_adapter_manager", None)
        if worker_manager is None or lora_config is None or adapter_manager is None:
            raise RuntimeError(
                "active TLM requires vLLM native LoRA; construct the engine with enable_lora=True"
            )
        self.runner = runner
        self.store = store
        self.worker_manager = worker_manager
        self.adapter_manager = adapter_manager
        self.max_rank = int(getattr(lora_config, "max_lora_rank"))
        self._published_version = 0
        self._requests: dict[int, object] = {}

    @staticmethod
    def _lora_id(version: int) -> int:
        return _LORA_ID_BASE + int(version)

    def request_for_version(self, version: int) -> object:
        try:
            return self._requests[int(version)]
        except KeyError as exc:
            raise RuntimeError(f"LoRA version {version} has not been materialized") from exc

    def _build_model(self, version: LoRAVersion) -> object:
        from vllm.lora.lora_model import LoRAModel
        from vllm.lora.lora_weights import LoRALayerWeights

        if version.rank > self.max_rank:
            raise RuntimeError(
                f"TLM LoRA rank {version.rank} exceeds vLLM max_lora_rank={self.max_rank}"
            )
        loras = {
            name: LoRALayerWeights(
                module_name=name,
                rank=weights.rank,
                lora_alpha=int(weights.alpha),
                lora_a=weights.lora_a,
                lora_b=weights.lora_b,
                scaling=float(weights.alpha) / float(weights.rank),
            )
            for name, weights in version.target_modules.items()
        }
        return LoRAModel(self._lora_id(version.version), version.rank, loras)

    def _make_request(self, version: LoRAVersion) -> object:
        from vllm.lora.request import LoRARequest

        return LoRARequest(
            lora_name=f"{version.adapter_key}:v{version.version}",
            lora_int_id=self._lora_id(version.version),
            lora_path=_IN_MEMORY_PATH,
        )

    def _make_capacity(self) -> None:
        capacity = int(getattr(self.adapter_manager, "capacity", 0) or 0)
        list_adapters = getattr(self.adapter_manager, "list_adapters", None)
        remove_adapter = getattr(self.adapter_manager, "remove_adapter", None)
        if capacity <= 0 or not callable(list_adapters):
            return
        registered = list_adapters()
        if len(registered) < capacity:
            return
        pinned_ids = {
            self._lora_id(version) for version in self.store.pinned_versions()
        }
        candidates = sorted(
            lora_id
            for lora_id in registered
            if lora_id >= _LORA_ID_BASE and lora_id not in pinned_ids
        )
        if not candidates or not callable(remove_adapter):
            raise RuntimeError(
                "vLLM LoRA CPU capacity is exhausted by active or external adapters"
            )
        remove_adapter(candidates[0])

    def _materialize(self, version: LoRAVersion) -> bool:
        lora_id = self._lora_id(version.version)
        list_adapters = getattr(self.adapter_manager, "list_adapters", None)
        add_adapter = getattr(self.adapter_manager, "add_adapter", None)
        if not callable(list_adapters) or not callable(add_adapter):
            raise RuntimeError("vLLM LoRA manager lacks adapter registration methods")
        if lora_id in list_adapters():
            if version.version not in self._requests:
                raise RuntimeError(
                    f"vLLM LoRA id collision for TLM version {version.version}"
                )
            return False
        self._make_capacity()
        add_adapter(self._build_model(version))
        return True

    def sync_versions(self) -> int:
        added = 0
        for version in self.store.versions_after(self._published_version):
            added += int(self._materialize(version))
            self._requests[version.version] = self._make_request(version)
            self._published_version = version.version
        return added

    def pin_request(self, request_id: str) -> object | None:
        version = self.store.pin_request(request_id)
        if version is None:
            return None
        if version > self._published_version:
            self.sync_versions()
        self._materialize(self.store.get(version))
        return self.request_for_version(version)

    def release_finished(self, request_ids: object) -> int:
        released = 0
        for request_id in tuple(request_ids or ()):
            released += int(self.store.release_request(str(request_id)))
        return released

    def apply_mrv1_input_batch(self, input_batch: object) -> int:
        req_id_to_index = getattr(input_batch, "req_id_to_index", None)
        request_mapping = getattr(input_batch, "request_lora_mapping", None)
        ids_to_requests = getattr(input_batch, "lora_id_to_request_ids", None)
        id_to_request = getattr(input_batch, "lora_id_to_lora_request", None)
        if not isinstance(req_id_to_index, Mapping) or request_mapping is None:
            raise RuntimeError("MRv1 input batch lacks LoRA request mapping")
        if not isinstance(ids_to_requests, dict) or not isinstance(id_to_request, dict):
            raise RuntimeError("MRv1 input batch lacks LoRA request registries")
        applied = 0
        for request_id, req_index in req_id_to_index.items():
            request = self.pin_request(str(request_id))
            if request is None:
                continue
            lora_id = int(getattr(request, "lora_int_id"))
            previous = int(request_mapping[int(req_index)])
            if previous not in (0, lora_id) and previous < _LORA_ID_BASE:
                raise RuntimeError(
                    f"request {request_id!r} already carries a non-TLM LoRA adapter"
                )
            if previous != lora_id and previous >= _LORA_ID_BASE:
                raise RuntimeError(
                    f"request {request_id!r} attempted to change pinned TLM version"
                )
            request_mapping[int(req_index)] = lora_id
            ids_to_requests.setdefault(lora_id, set()).add(str(request_id))
            id_to_request[lora_id] = request
            applied += 1
        return applied

    def bind_mrv2_lifecycle(self) -> None:
        runner = self.runner
        if getattr(runner, "_tllm_tlm_lora_bound", False):
            return
        original_add = runner.add_requests
        original_finish = runner.finish_requests
        publisher = self

        def add_requests(this: object, scheduler_output: object) -> None:
            publisher.sync_versions()
            original_add(scheduler_output)
            for new_request in tuple(
                getattr(scheduler_output, "scheduled_new_reqs", ()) or ()
            ):
                req_id = str(getattr(new_request, "req_id"))
                # MRv2 profiles LoRA kernels with internal requests after
                # load_model. They are not scheduler admissions and must not
                # consume a persistent test-time adapter version.
                if req_id.startswith("_warmup_"):
                    continue
                existing = getattr(new_request, "lora_request", None)
                if existing is not None:
                    raise RuntimeError(
                        f"request {req_id!r} already carries a non-TLM LoRA adapter"
                    )
                request = publisher.pin_request(req_id)
                if request is None:
                    continue
                req_index = this.req_states.req_id_to_index[req_id]
                this.lora_state.add_request(req_id, req_index, request)

        def finish_requests(this: object, scheduler_output: object) -> None:
            finished = tuple(getattr(scheduler_output, "finished_req_ids", ()) or ())
            original_finish(scheduler_output)
            publisher.release_finished(finished)

        runner.add_requests = MethodType(add_requests, runner)
        runner.finish_requests = MethodType(finish_requests, runner)
        setattr(runner, "_tllm_tlm_lora_bound", True)


def maybe_bind_publisher(*, runtime: object, runner: object, runner_family: str) -> object | None:
    store = getattr(runtime, "test_time_lora_store", None)
    if store is None:
        return None
    if not isinstance(store, LoRAVersionStore):
        raise TypeError("runtime adapter store must be LoRAVersionStore")
    existing = getattr(runtime, "test_time_lora_publisher", None)
    if existing is not None:
        return existing
    publisher = VLLMVersionedLoRAPublisher(runner=runner, store=store)
    if runner_family == "mrv2":
        publisher.bind_mrv2_lifecycle()
    elif runner_family != "mrv1":
        raise RuntimeError(f"unsupported vLLM runner family {runner_family!r}")
    setattr(runtime, "test_time_lora_publisher", publisher)
    return publisher


def sync_mrv1_before_prepare(
    *, runtime: object, runner: object, scheduler_output: object
) -> int:
    publisher = maybe_bind_publisher(
        runtime=runtime,
        runner=runner,
        runner_family="mrv1",
    )
    if not isinstance(publisher, VLLMVersionedLoRAPublisher):
        return 0
    publisher.release_finished(
        getattr(scheduler_output, "finished_req_ids", ())
    )
    publisher.sync_versions()
    return publisher.apply_mrv1_input_batch(runner.input_batch)


__all__ = [
    "VLLMVersionedLoRAPublisher",
    "maybe_bind_publisher",
    "sync_mrv1_before_prepare",
]
