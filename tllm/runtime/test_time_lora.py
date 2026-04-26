#!/usr/bin/env python3
"""Runtime service for dependency-light test-time LoRA updates."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

from tllm.contracts.test_time_lora import TestTimeLoRAUpdate

__test__ = False


def _require_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    return value


class TestTimeLoRAService:
    """Stage and commit in-memory LoRA slot updates.

    The service intentionally owns only stable runtime semantics: integer LoRA
    slot ids, canonical request-id assignments, and per-target tensor updates.
    vLLM-specific managers are adapted behind a tiny duck-typed surface.
    """

    def __init__(self, *, manager: object, max_rank: int, max_adapters: int) -> None:
        self.manager = manager
        self.max_rank = _require_int(max_rank, field_name="max_rank")
        self.max_adapters = _require_int(max_adapters, field_name="max_adapters")
        if self.max_rank <= 0:
            raise ValueError("TestTimeLoRAService requires positive max_rank")
        if self.max_adapters <= 0:
            raise ValueError("TestTimeLoRAService requires positive max_adapters")
        self._staged_updates: deque[TestTimeLoRAUpdate] = deque()
        self._request_to_lora_id: dict[str, int] = {}
        self._adapter_keys: dict[str, int] = {}

    @property
    def modules(self) -> Mapping[str, object]:
        modules = getattr(self.manager, "modules", None)
        if isinstance(modules, Mapping):
            return modules
        adapter_manager = getattr(self.manager, "_adapter_manager", None)
        modules = getattr(adapter_manager, "modules", None)
        if isinstance(modules, Mapping):
            return modules
        raise RuntimeError("TestTimeLoRAService manager does not expose LoRA modules")

    def adapter_for_request(self, request_id: str) -> int | None:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be str")
        return self._request_to_lora_id.get(request_id)

    def stage_update(self, update: TestTimeLoRAUpdate) -> None:
        if not isinstance(update, TestTimeLoRAUpdate):
            raise TypeError("update must be TestTimeLoRAUpdate")
        self._validate_update(update)
        self._staged_updates.append(update)

    def commit_staged_updates(self) -> int:
        committed = 0
        while self._staged_updates:
            update = self._staged_updates.popleft()
            self._commit_update(update)
            committed += 1
        return committed

    def _validate_update(self, update: TestTimeLoRAUpdate) -> None:
        if update.rank > self.max_rank:
            raise RuntimeError("test-time LoRA update rank exceeds service max_rank")
        if update.lora_int_id >= self.max_adapters:
            raise RuntimeError("test-time LoRA update lora_int_id exceeds service max_adapters")
        modules = self.modules
        for module_name in update.target_modules:
            if module_name not in modules:
                raise RuntimeError(f"unknown LoRA target module `{module_name}`")

    def _commit_update(self, update: TestTimeLoRAUpdate) -> None:
        modules = self.modules
        slot = update.lora_int_id
        for module_name, weights in update.target_modules.items():
            module = modules[module_name]
            set_lora = getattr(module, "set_lora", None)
            if not callable(set_lora):
                raise RuntimeError(f"LoRA target module `{module_name}` does not expose set_lora")
            set_lora(
                slot,
                weights.lora_a,
                weights.lora_b,
                bias=weights.bias,
                scaling=weights.scaling,
            )
        self._adapter_keys[update.adapter_key] = slot
        if update.scope == "request":
            for request_id in update.request_ids:
                self._request_to_lora_id[request_id] = slot
        activate = getattr(self.manager, "activate_adapter", None)
        if callable(activate):
            activate(slot, update)
