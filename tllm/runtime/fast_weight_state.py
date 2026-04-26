#!/usr/bin/env python3
"""Runtime state for In-Place TTT fast-weight overlays."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Callable

import torch
from torch.nn import functional as F

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


def _require_2d_tensor(value: object, *, field_name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field_name} must be torch.Tensor")
    if value.ndim != 2:
        raise RuntimeError(f"{field_name} must be 2D")
    return value


@dataclass(frozen=True)
class FastWeightKey:
    request_id: str
    sample_idx: int
    layer: int
    target_module: str = "down_proj"

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _require_str(self.request_id, field_name="request_id"))
        object.__setattr__(self, "sample_idx", _require_int(self.sample_idx, field_name="sample_idx"))
        object.__setattr__(self, "layer", _require_int(self.layer, field_name="layer"))
        object.__setattr__(
            self,
            "target_module",
            _require_str(self.target_module, field_name="target_module"),
        )


@dataclass(frozen=True)
class FastWeightSlot:
    past_h: torch.Tensor
    past_t: torch.Tensor
    past_w: torch.Tensor


class FastWeightState:
    def __init__(self) -> None:
        self._slots: dict[FastWeightKey, FastWeightSlot] = {}

    def write(
        self,
        *,
        key: FastWeightKey,
        past_h: torch.Tensor,
        past_t: torch.Tensor,
        past_w: torch.Tensor,
    ) -> None:
        if not isinstance(key, FastWeightKey):
            raise TypeError("key must be FastWeightKey")
        h = _require_2d_tensor(past_h, field_name="past_h")
        t = _require_2d_tensor(past_t, field_name="past_t")
        w = _require_2d_tensor(past_w, field_name="past_w")
        if int(h.shape[0]) != int(t.shape[0]):
            raise RuntimeError("past_h and past_t must have matching row count")
        if int(h.shape[1]) != int(w.shape[1]):
            raise RuntimeError("past_h width must match past_w input width")
        if int(t.shape[1]) != int(w.shape[0]):
            raise RuntimeError("past_t width must match past_w output width")
        self._slots[key] = FastWeightSlot(
            past_h=h.detach().clone(),
            past_t=t.detach().clone(),
            past_w=w.detach().clone(),
        )

    def read(self, key: FastWeightKey) -> FastWeightSlot:
        if not isinstance(key, FastWeightKey):
            raise TypeError("key must be FastWeightKey")
        slot = self._slots.get(key)
        if slot is None:
            raise KeyError(key)
        return FastWeightSlot(
            past_h=slot.past_h.detach().clone(),
            past_t=slot.past_t.detach().clone(),
            past_w=slot.past_w.detach().clone(),
        )

    def get(self, key: FastWeightKey) -> FastWeightSlot | None:
        if not isinstance(key, FastWeightKey):
            raise TypeError("key must be FastWeightKey")
        slot = self._slots.get(key)
        if slot is None:
            return None
        return FastWeightSlot(
            past_h=slot.past_h.detach().clone(),
            past_t=slot.past_t.detach().clone(),
            past_w=slot.past_w.detach().clone(),
        )

    def clear(self, key: FastWeightKey) -> None:
        if not isinstance(key, FastWeightKey):
            raise TypeError("key must be FastWeightKey")
        self._slots.pop(key, None)


@dataclass(frozen=True)
class DownProjFastWeightOverlayHandle:
    module: object
    original_forward: Callable[..., object]

    def restore(self) -> None:
        setattr(self.module, "forward", self.original_forward)


def module_weight_view(module: object) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        base_layer = getattr(module, "base_layer", None)
        weight = getattr(base_layer, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise RuntimeError("module does not expose a tensor weight view")
    if weight.ndim != 2:
        raise RuntimeError("module weight view must be 2D")
    return weight


def _require_single_tensor_parallel() -> None:
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size
    except Exception:
        return
    try:
        tp_size = int(get_tensor_model_parallel_world_size())
    except Exception:
        return
    if tp_size != 1:
        raise RuntimeError("In-Place TTT fast-weight overlay currently requires tensor_parallel_size=1")


def _replace_forward_output(base_output: object, replacement: torch.Tensor) -> object:
    if isinstance(base_output, torch.Tensor):
        return replacement.to(device=base_output.device, dtype=base_output.dtype)
    if isinstance(base_output, tuple) and base_output and isinstance(base_output[0], torch.Tensor):
        first = replacement.to(device=base_output[0].device, dtype=base_output[0].dtype)
        return (first, *base_output[1:])
    raise RuntimeError("down_proj forward output must be a tensor or tuple starting with a tensor")


def install_down_proj_fast_weight_overlay(
    *,
    module: object,
    state: FastWeightState,
    key: FastWeightKey,
    require_single_tp: bool = True,
) -> DownProjFastWeightOverlayHandle:
    if not isinstance(state, FastWeightState):
        raise TypeError("state must be FastWeightState")
    if not isinstance(key, FastWeightKey):
        raise TypeError("key must be FastWeightKey")
    if require_single_tp:
        _require_single_tensor_parallel()
    weight = module_weight_view(module)
    original_forward = getattr(module, "forward", None)
    if not callable(original_forward):
        raise RuntimeError("module must expose callable forward")

    def _forward(self, hidden: torch.Tensor, *args: object, **kwargs: object) -> object:
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("down_proj hidden input must be torch.Tensor")
        base_output = original_forward(hidden, *args, **kwargs)
        slot = state.get(key)
        if slot is None:
            return base_output
        if tuple(slot.past_w.shape) != tuple(weight.shape):
            raise RuntimeError("fast-weight overlay shape must match module weight")
        replacement = F.linear(hidden, slot.past_w.to(device=hidden.device, dtype=hidden.dtype))
        return _replace_forward_output(base_output, replacement)

    bound_forward = MethodType(_forward, module)
    setattr(module, "forward", bound_forward)
    return DownProjFastWeightOverlayHandle(module=module, original_forward=original_forward)


def apply_down_proj_fast_weight(
    *,
    hidden: torch.Tensor,
    base_weight: torch.Tensor,
    state: FastWeightState,
    key: FastWeightKey,
) -> torch.Tensor:
    h = _require_2d_tensor(hidden, field_name="hidden")
    w = _require_2d_tensor(base_weight, field_name="base_weight")
    if int(h.shape[1]) != int(w.shape[1]):
        raise RuntimeError("hidden width must match base_weight input width")
    slot = state.get(key)
    effective_weight = w
    if slot is not None:
        if tuple(slot.past_w.shape) != tuple(w.shape):
            raise RuntimeError("fast-weight overlay shape must match base_weight")
        effective_weight = slot.past_w.to(device=w.device, dtype=w.dtype)
    return torch.nn.functional.linear(h, effective_weight)
