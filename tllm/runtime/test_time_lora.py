#!/usr/bin/env python3
"""Runtime service for dependency-light test-time LoRA updates."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol

from tllm.contracts.test_time_lora import TestTimeLoRAUpdate, TestTimeLoRAWeights

__test__ = False


def _require_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    return value


class TestTimeLoRABackend(Protocol):
    @property
    def modules(self) -> Mapping[str, object]:
        ...

    def commit_update(self, update: TestTimeLoRAUpdate) -> None:
        ...


class TestTimeLoRAService:
    """Stage and commit in-memory LoRA adapter updates.

    The service intentionally owns only stable runtime semantics: positive LoRA
    adapter ids, canonical request-id assignments, and per-target tensor updates.
    Backend-specific managers own physical slot policy behind commit_update().
    """

    def __init__(
        self,
        *,
        manager: object,
        max_rank: int,
    ) -> None:
        self.manager = _coerce_backend(manager)
        self.max_rank = _require_int(max_rank, field_name="max_rank")
        if self.max_rank <= 0:
            raise ValueError("TestTimeLoRAService requires positive max_rank")
        self._staged_updates: deque[TestTimeLoRAUpdate] = deque()
        self._request_to_lora_id: dict[str, int] = {}
        self._adapter_keys: dict[str, int] = {}

    @property
    def modules(self) -> Mapping[str, object]:
        return self.manager.modules

    def adapter_for_request(self, request_id: str) -> int | None:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be str")
        return self._request_to_lora_id.get(request_id)

    def request_assignments(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._request_to_lora_id))

    def unbind_request(self, request_id: str) -> bool:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be str")
        return self._request_to_lora_id.pop(request_id, None) is not None

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
        modules = self.modules
        for module_name in update.target_modules:
            if module_name not in modules:
                raise RuntimeError(f"unknown LoRA target module `{module_name}`")

    def _commit_update(self, update: TestTimeLoRAUpdate) -> None:
        self.manager.commit_update(update)
        self._adapter_keys[update.adapter_key] = update.lora_int_id
        if update.scope == "request":
            for request_id in update.request_ids:
                self._request_to_lora_id[request_id] = update.lora_int_id


class _DirectModuleBackend:
    def __init__(self, manager: object) -> None:
        self.manager = manager

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

    def commit_update(self, update: TestTimeLoRAUpdate) -> None:
        for module_name, weights in update.target_modules.items():
            _write_direct_module_lora(self.modules[module_name], update.lora_int_id, weights)
        activate = getattr(self.manager, "activate_adapter", None)
        if callable(activate):
            activate(update.lora_int_id, update)


def _coerce_backend(manager: object) -> TestTimeLoRABackend:
    modules = getattr(manager, "modules", None)
    commit_update = getattr(manager, "commit_update", None)
    if isinstance(modules, Mapping) and callable(commit_update):
        return manager
    return _DirectModuleBackend(manager)


def _write_direct_module_lora(module: object, adapter_id: int, weights: TestTimeLoRAWeights) -> None:
    set_lora = getattr(module, "set_lora", None)
    if not callable(set_lora):
        raise RuntimeError("LoRA target module does not expose set_lora")
    set_lora(
        adapter_id,
        weights.lora_a,
        weights.lora_b,
        bias=weights.bias,
        scaling=weights.scaling,
    )
