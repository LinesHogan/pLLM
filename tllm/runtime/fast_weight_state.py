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


@dataclass(frozen=True)
class FastWeightStats:
    update_count: int = 0
    apply_count: int = 0
    trained_rows: int = 0
    tail_rows: int = 0


@dataclass(frozen=True)
class FastWeightStepResult:
    output: torch.Tensor
    update_count: int = 0
    trained_rows: int = 0
    tail_rows: int = 0
    applied_weight: bool = False


class FastWeightState:
    def __init__(self) -> None:
        self._slots: dict[FastWeightKey, FastWeightSlot] = {}
        self._stats: dict[FastWeightKey, FastWeightStats] = {}

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
        self._stats.pop(key, None)

    def stats(self, key: FastWeightKey) -> FastWeightStats:
        if not isinstance(key, FastWeightKey):
            raise TypeError("key must be FastWeightKey")
        return self._stats.get(key, FastWeightStats())

    def record_step(self, key: FastWeightKey, result: FastWeightStepResult) -> None:
        if not isinstance(key, FastWeightKey):
            raise TypeError("key must be FastWeightKey")
        if not isinstance(result, FastWeightStepResult):
            raise TypeError("result must be FastWeightStepResult")
        current = self.stats(key)
        self._stats[key] = FastWeightStats(
            update_count=int(current.update_count) + int(result.update_count),
            apply_count=int(current.apply_count) + (1 if bool(result.applied_weight) else 0),
            trained_rows=int(current.trained_rows) + int(result.trained_rows),
            tail_rows=int(result.tail_rows),
        )


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


def _require_supported_down_proj(module: object, *, require_single_tp: bool) -> None:
    tp_size = getattr(module, "tp_size", None)
    if require_single_tp and tp_size is not None and int(tp_size) != 1:
        raise RuntimeError("In-Place TTT fast-weight overlay currently requires tensor_parallel_size=1")
    quant_method = getattr(module, "quant_method", None)
    if quant_method is not None and quant_method.__class__.__name__ != "UnquantizedLinearMethod":
        raise RuntimeError("In-Place TTT fast-weight overlay currently requires non-quantized down_proj")
    if getattr(module, "bias", None) is not None:
        raise RuntimeError("In-Place TTT fast-weight overlay currently requires biasless down_proj")


def _replace_forward_output(base_output: object, replacement: torch.Tensor) -> object:
    if isinstance(base_output, torch.Tensor):
        return replacement.to(device=base_output.device, dtype=base_output.dtype)
    if isinstance(base_output, tuple) and base_output and isinstance(base_output[0], torch.Tensor):
        first = replacement.to(device=base_output[0].device, dtype=base_output[0].dtype)
        return (first, *base_output[1:])
    raise RuntimeError("down_proj forward output must be a tensor or tuple starting with a tensor")


def _first_forward_tensor(output: object, *, field_name: str) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise RuntimeError(f"{field_name} output must be a tensor or tuple starting with a tensor")


def run_down_proj_fast_weight_step(
    *,
    hidden: torch.Tensor,
    target: torch.Tensor,
    base_weight: torch.Tensor,
    state: FastWeightState,
    key: FastWeightKey,
    chunk_size: int,
    learning_rate: float,
    update_enabled: bool = True,
    cache_tail: bool = True,
) -> FastWeightStepResult:
    h = _require_2d_tensor(hidden, field_name="hidden")
    t = _require_2d_tensor(target, field_name="target")
    w = _require_2d_tensor(base_weight, field_name="base_weight")
    if not isinstance(state, FastWeightState):
        raise TypeError("state must be FastWeightState")
    if not isinstance(key, FastWeightKey):
        raise TypeError("key must be FastWeightKey")
    if int(h.shape[0]) != int(t.shape[0]):
        raise RuntimeError("hidden and target must have matching row count")
    if int(h.shape[1]) != int(w.shape[1]):
        raise RuntimeError("hidden width must match base_weight input width")
    if int(t.shape[1]) != int(w.shape[0]):
        raise RuntimeError("target width must match base_weight output width")

    slot = state.get(key)
    fast_w = w.detach().clone()
    applied_weight = False
    h_work = h
    t_work = t
    if slot is not None:
        if tuple(slot.past_w.shape) != tuple(w.shape):
            raise RuntimeError("fast-weight state past_w shape must match base_weight")
        fast_w = slot.past_w.to(device=w.device, dtype=w.dtype).clone()
        applied_weight = True
        if update_enabled and int(slot.past_h.shape[0]) > 0:
            past_h = slot.past_h.to(device=h.device, dtype=h.dtype)
            past_t = slot.past_t.to(device=t.device, dtype=t.dtype)
            if int(past_h.shape[1]) != int(h.shape[1]) or int(past_t.shape[1]) != int(t.shape[1]):
                raise RuntimeError("fast-weight cached tail shape must match current step")
            h_work = torch.cat([past_h, h], dim=0)
            t_work = torch.cat([past_t, t], dim=0)

    if not update_enabled:
        output = F.linear(h, fast_w.to(device=h.device, dtype=h.dtype))
        result = FastWeightStepResult(
            output=output,
            update_count=0,
            trained_rows=0,
            tail_rows=int(slot.past_h.shape[0]) if slot is not None else 0,
            applied_weight=applied_weight,
        )
        state.record_step(key, result)
        return result

    chunk = max(1, int(chunk_size))
    total_rows = int(h_work.shape[0])
    full_rows = (total_rows // chunk) * chunk
    outputs: list[torch.Tensor] = []
    update_count = 0
    for start in range(0, full_rows, chunk):
        end = start + chunk
        h_chunk = h_work[start:end]
        t_chunk = t_work[start:end]
        residual = t_chunk - h_chunk.matmul(fast_w.to(device=h_chunk.device, dtype=h_chunk.dtype).t())
        step_w = float(learning_rate) * residual.t().matmul(h_chunk) / float(chunk)
        fast_w = fast_w + step_w.to(device=fast_w.device, dtype=fast_w.dtype)
        outputs.append(h_chunk.matmul(fast_w.to(device=h_chunk.device, dtype=h_chunk.dtype).t()))
        update_count += 1

    tail_h = h_work[full_rows:]
    tail_t = t_work[full_rows:]
    tail_rows = int(tail_h.shape[0])
    if tail_rows > 0:
        outputs.append(tail_h.matmul(fast_w.to(device=tail_h.device, dtype=tail_h.dtype).t()))
    output_all = torch.cat(outputs, dim=0) if outputs else h_work.new_empty((0, int(w.shape[0])))
    output = output_all[-int(h.shape[0]) :] if int(h.shape[0]) > 0 else output_all[:0]

    if cache_tail:
        state.write(
            key=key,
            past_h=tail_h.detach().clone(),
            past_t=tail_t.detach().clone(),
            past_w=fast_w.detach().clone(),
        )
    result = FastWeightStepResult(
        output=output,
        update_count=update_count,
        trained_rows=full_rows,
        tail_rows=tail_rows,
        applied_weight=applied_weight or update_count > 0,
    )
    state.record_step(key, result)
    return result


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
    _require_supported_down_proj(module, require_single_tp=require_single_tp)
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


def install_mlp_fast_weight_wrapper(
    *,
    module: object,
    state: FastWeightState,
    key: FastWeightKey,
    chunk_size: int,
    learning_rate: float,
    phase_getter: Callable[[], str] | None = None,
    decode_update_enabled: bool = False,
    require_single_tp: bool = True,
) -> DownProjFastWeightOverlayHandle:
    if not isinstance(state, FastWeightState):
        raise TypeError("state must be FastWeightState")
    if not isinstance(key, FastWeightKey):
        raise TypeError("key must be FastWeightKey")
    if require_single_tp:
        _require_single_tensor_parallel()
    gate_up_proj = getattr(module, "gate_up_proj", None)
    act_fn = getattr(module, "act_fn", None)
    down_proj = getattr(module, "down_proj", None)
    if gate_up_proj is None or not callable(getattr(gate_up_proj, "forward", None)):
        raise RuntimeError("MLP fast-weight wrapper requires gate_up_proj.forward")
    if not callable(act_fn):
        raise RuntimeError("MLP fast-weight wrapper requires callable act_fn")
    if down_proj is None or not callable(getattr(down_proj, "forward", None)):
        raise RuntimeError("MLP fast-weight wrapper requires down_proj.forward")
    _require_supported_down_proj(down_proj, require_single_tp=require_single_tp)
    weight = module_weight_view(down_proj)
    original_forward = getattr(module, "forward", None)
    if not callable(original_forward):
        raise RuntimeError("module must expose callable forward")

    def _forward(self, hidden: torch.Tensor, *args: object, **kwargs: object) -> object:
        if args or kwargs:
            return original_forward(hidden, *args, **kwargs)
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("MLP hidden input must be torch.Tensor")
        gate_up = _first_forward_tensor(gate_up_proj.forward(hidden), field_name="gate_up_proj")
        activated = act_fn(gate_up)
        if not isinstance(activated, torch.Tensor):
            raise RuntimeError("MLP act_fn output must be torch.Tensor")
        phase = phase_getter() if phase_getter is not None else "prefill"
        update_enabled = str(phase) != "decode" or bool(decode_update_enabled)
        step = run_down_proj_fast_weight_step(
            hidden=activated,
            target=hidden,
            base_weight=weight,
            state=state,
            key=key,
            chunk_size=chunk_size,
            learning_rate=learning_rate,
            update_enabled=update_enabled,
            cache_tail=update_enabled,
        )
        return step.output

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
