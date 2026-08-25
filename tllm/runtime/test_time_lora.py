#!/usr/bin/env python3
"""Backend-neutral immutable LoRA versions and request pinning."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from threading import RLock
from types import MappingProxyType
from typing import Mapping

import torch


@dataclass(frozen=True, slots=True)
class LoRAAdapterWeights:
    lora_a: torch.Tensor
    lora_b: torch.Tensor
    alpha: float

    def __post_init__(self) -> None:
        if not isinstance(self.lora_a, torch.Tensor) or not isinstance(
            self.lora_b, torch.Tensor
        ):
            raise TypeError("LoRA weights must be tensors")
        if self.lora_a.ndim != 2 or self.lora_b.ndim != 2:
            raise ValueError("LoRA A and B must be rank-2 tensors")
        if int(self.lora_a.shape[0]) != int(self.lora_b.shape[1]):
            raise ValueError("LoRA A/B rank dimensions do not match")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real):
            raise TypeError("LoRA alpha must be real")
        if float(self.alpha) <= 0.0:
            raise ValueError("LoRA alpha must be positive")
        object.__setattr__(self, "alpha", float(self.alpha))

    @property
    def rank(self) -> int:
        return int(self.lora_a.shape[0])


@dataclass(frozen=True, slots=True)
class LoRAVersion:
    version: int
    adapter_key: str
    target_modules: Mapping[str, LoRAAdapterWeights]

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("LoRA version must be int")
        if self.version <= 0:
            raise ValueError("LoRA version must be positive")
        if not isinstance(self.adapter_key, str) or not self.adapter_key:
            raise ValueError("LoRA adapter_key must be non-empty")
        if not self.target_modules:
            raise ValueError("LoRA version requires target modules")
        modules: dict[str, LoRAAdapterWeights] = {}
        ranks: set[int] = set()
        for name, weights in self.target_modules.items():
            if not isinstance(name, str) or not name:
                raise ValueError("LoRA target module names must be non-empty")
            if not isinstance(weights, LoRAAdapterWeights):
                raise TypeError("LoRA target modules must contain LoRAAdapterWeights")
            modules[name] = weights
            ranks.add(weights.rank)
        if len(ranks) != 1:
            raise ValueError("all LoRA target modules must use the same rank")
        object.__setattr__(self, "target_modules", MappingProxyType(modules))

    @property
    def rank(self) -> int:
        return next(iter(self.target_modules.values())).rank


class LoRAVersionStore:
    """Atomic adapter publication with stable versions for in-flight requests."""

    def __init__(self, adapter_key: str = "tlm") -> None:
        if not isinstance(adapter_key, str) or not adapter_key:
            raise ValueError("adapter_key must be non-empty")
        self.adapter_key = adapter_key
        self._versions: dict[int, LoRAVersion] = {}
        self._request_versions: dict[str, int] = {}
        self._assignment_history: dict[str, int] = {}
        self._latest = 0
        self._lock = RLock()

    @property
    def latest_version(self) -> int:
        with self._lock:
            return self._latest

    def publish(
        self, target_modules: Mapping[str, LoRAAdapterWeights]
    ) -> LoRAVersion:
        owned: dict[str, LoRAAdapterWeights] = {}
        for name, weights in target_modules.items():
            if not isinstance(weights, LoRAAdapterWeights):
                raise TypeError("publish expects LoRAAdapterWeights values")
            owned[name] = LoRAAdapterWeights(
                lora_a=weights.lora_a.detach().to(device="cpu", copy=True).contiguous(),
                lora_b=weights.lora_b.detach().to(device="cpu", copy=True).contiguous(),
                alpha=weights.alpha,
            )
        with self._lock:
            version_id = self._latest + 1
            version = LoRAVersion(version_id, self.adapter_key, owned)
            self._versions[version_id] = version
            self._latest = version_id
            return version

    def get(self, version: int) -> LoRAVersion:
        with self._lock:
            try:
                return self._versions[int(version)]
            except KeyError as exc:
                raise RuntimeError(f"unknown LoRA version {version}") from exc

    def versions_after(self, version: int) -> tuple[LoRAVersion, ...]:
        with self._lock:
            return tuple(
                self._versions[key]
                for key in sorted(self._versions)
                if key > int(version)
            )

    def pin_request(self, request_id: str) -> int | None:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be non-empty")
        with self._lock:
            existing = self._request_versions.get(request_id)
            if existing is not None:
                return existing
            if self._latest == 0:
                return None
            self._request_versions[request_id] = self._latest
            self._assignment_history[request_id] = self._latest
            return self._latest

    def request_version(self, request_id: str) -> int | None:
        with self._lock:
            return self._request_versions.get(request_id)

    def pinned_versions(self) -> frozenset[int]:
        with self._lock:
            return frozenset(self._request_versions.values())

    def assignment_history(self) -> Mapping[str, int]:
        with self._lock:
            return MappingProxyType(dict(self._assignment_history))

    def release_request(self, request_id: str) -> bool:
        with self._lock:
            return self._request_versions.pop(request_id, None) is not None


__all__ = ["LoRAAdapterWeights", "LoRAVersion", "LoRAVersionStore"]
