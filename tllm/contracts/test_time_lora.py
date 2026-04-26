#!/usr/bin/env python3
"""Contracts for runtime-owned test-time LoRA updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Literal, Mapping

import torch

LoRAScope = Literal["global", "request"]

__test__ = False


def _require_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    return value


def _require_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    return value


@dataclass(frozen=True)
class TestTimeLoRATarget:
    module_name: str
    packed_members: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        module_name = _require_str(self.module_name, field_name="module_name")
        packed_members: list[str] = []
        for member in self.packed_members:
            if not isinstance(member, str):
                raise TypeError("packed_members must contain str")
            packed_members.append(_require_str(member, field_name="packed_members member"))
        object.__setattr__(self, "module_name", module_name)
        object.__setattr__(self, "packed_members", tuple(packed_members))


@dataclass(frozen=True)
class TestTimeLoRAWeights:
    lora_a: torch.Tensor
    lora_b: torch.Tensor
    embeddings_tensor: torch.Tensor | None = None
    bias: torch.Tensor | None = None
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.lora_a, torch.Tensor) or not isinstance(self.lora_b, torch.Tensor):
            raise ValueError("TestTimeLoRAWeights requires tensor lora_a and lora_b")
        if self.lora_a.ndim != 2 or self.lora_b.ndim != 2:
            raise ValueError("TestTimeLoRAWeights expects rank-2 LoRA tensors")
        if int(self.lora_a.shape[1]) != int(self.lora_b.shape[0]):
            raise ValueError("TestTimeLoRAWeights rank dimension mismatch")
        if self.embeddings_tensor is not None and not isinstance(self.embeddings_tensor, torch.Tensor):
            raise ValueError("TestTimeLoRAWeights embeddings_tensor must be a tensor when provided")
        if self.bias is not None and not isinstance(self.bias, torch.Tensor):
            raise ValueError("TestTimeLoRAWeights bias must be a tensor when provided")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real):
            raise TypeError("TestTimeLoRAWeights alpha must be real")
        object.__setattr__(self, "alpha", float(self.alpha))

    @property
    def rank(self) -> int:
        return int(self.lora_a.shape[1])

    @property
    def scaling(self) -> float:
        return float(self.alpha) / float(max(1, self.rank))


@dataclass(frozen=True)
class TestTimeLoRAUpdate:
    adapter_key: str
    lora_int_id: int
    rank: int
    scope: LoRAScope
    target_modules: Mapping[str, TestTimeLoRAWeights]
    request_ids: tuple[str, ...] = field(default_factory=tuple)
    step_id: int = -1

    def __post_init__(self) -> None:
        adapter_key = _require_str(self.adapter_key, field_name="adapter_key")
        lora_int_id = _require_int(self.lora_int_id, field_name="lora_int_id")
        rank = _require_int(self.rank, field_name="rank")
        step_id = _require_int(self.step_id, field_name="step_id")
        if lora_int_id <= 0:
            raise ValueError("TestTimeLoRAUpdate requires positive lora_int_id")
        if rank <= 0:
            raise ValueError("TestTimeLoRAUpdate requires positive rank")
        if self.scope not in ("global", "request"):
            raise ValueError("TestTimeLoRAUpdate scope must be 'global' or 'request'")
        request_ids: list[str] = []
        for request_id in self.request_ids:
            if not isinstance(request_id, str):
                raise TypeError("request_ids must contain str")
            request_ids.append(_require_str(request_id, field_name="request_id"))
        if self.scope == "request" and not request_ids:
            raise ValueError("request-scoped LoRA update requires request_ids")
        if not self.target_modules:
            raise ValueError("TestTimeLoRAUpdate requires target_modules")
        target_modules: dict[str, TestTimeLoRAWeights] = {}
        for module_name, weights in self.target_modules.items():
            module_key = _require_str(module_name, field_name="target module name")
            if not isinstance(weights, TestTimeLoRAWeights):
                raise ValueError("target_modules values must be TestTimeLoRAWeights")
            if int(weights.rank) != rank:
                raise ValueError("target module LoRA rank does not match update rank")
            target_modules[module_key] = weights
        object.__setattr__(self, "adapter_key", adapter_key)
        object.__setattr__(self, "lora_int_id", lora_int_id)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "request_ids", tuple(request_ids))
        object.__setattr__(self, "target_modules", MappingProxyType(target_modules))
        object.__setattr__(self, "step_id", step_id)
