#!/usr/bin/env python3
"""Runtime state for In-Place TTT fast-weight overlays."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MethodType
from typing import Callable

from opt_einsum import contract
import torch
from torch.nn import functional as F

__test__ = False

FastWeightMode = str


def normalize_fast_weight_mode(mode: object) -> FastWeightMode:
    normalized = _require_str(mode, field_name="mode")
    if normalized not in {"exact", "low_rank"}:
        raise ValueError("fast_weight_mode must be one of: exact, low_rank")
    return normalized


def _require_non_negative_int(value: object, *, field_name: str) -> int:
    parsed = _require_int(value, field_name=field_name)
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed




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
    def __init__(self, *, max_slots: int | None = None, max_bytes: int | None = None, mode: str = "exact") -> None:
        self.mode = normalize_fast_weight_mode(mode)
        if self.mode == "low_rank":
            raise NotImplementedError(
                "In-Place TTT fast-weight low_rank mode is not implemented yet; "
                "exact mode keeps the past_h/past_t/past_w state contract."
            )
        self._slots: dict[FastWeightKey, FastWeightSlot] = {}
        self._stats: dict[FastWeightKey, FastWeightStats] = {}
        self._max_slots = None if max_slots is None else max(0, int(max_slots))
        self._max_bytes = None if max_bytes is None else max(0, int(max_bytes))

    def set_limits(self, *, max_slots: int | None = None, max_bytes: int | None = None) -> None:
        self._max_slots = None if max_slots is None else max(0, int(max_slots))
        self._max_bytes = None if max_bytes is None else max(0, int(max_bytes))

    @staticmethod
    def _slot_bytes(slot: FastWeightSlot) -> int:
        return int(slot.past_h.numel() * slot.past_h.element_size()) + int(
            slot.past_t.numel() * slot.past_t.element_size()
        ) + int(slot.past_w.numel() * slot.past_w.element_size())

    @staticmethod
    def _tensor_bytes(*tensors: torch.Tensor) -> int:
        return sum(int(tensor.numel() * tensor.element_size()) for tensor in tensors)

    @property
    def slot_count(self) -> int:
        return len(self._slots)

    @property
    def estimated_bytes(self) -> int:
        return sum(self._slot_bytes(slot) for slot in self._slots.values())

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
        is_new_slot = key not in self._slots
        if is_new_slot and self._max_slots is not None and len(self._slots) >= int(self._max_slots):
            raise RuntimeError(
                f"fast-weight slot limit exceeded: slots={len(self._slots) + 1} max_slots={int(self._max_slots)}"
            )
        old_bytes = self._slot_bytes(self._slots[key]) if key in self._slots else 0
        next_bytes = self.estimated_bytes - old_bytes + self._tensor_bytes(h, t, w)
        if self._max_bytes is not None and next_bytes > int(self._max_bytes):
            raise RuntimeError(f"fast-weight byte limit exceeded: bytes={next_bytes} max_bytes={int(self._max_bytes)}")
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


def _validate_ttt_proj_weight(ttt_proj_weight: torch.Tensor | None, base_weight: torch.Tensor) -> None:
    if ttt_proj_weight is None:
        return
    if not isinstance(ttt_proj_weight, torch.Tensor):
        raise TypeError("ttt_proj_weight must be torch.Tensor")
    if ttt_proj_weight.ndim != 2:
        raise RuntimeError("ttt_proj_weight must be 2D")
    hidden = int(base_weight.shape[0])
    if tuple(ttt_proj_weight.shape) != (hidden, hidden):
        raise RuntimeError("ttt_proj_weight must be hidden_size x hidden_size")


def _validate_ttt_conv_weight(ttt_conv_weight: torch.Tensor | None, base_weight: torch.Tensor) -> None:
    if ttt_conv_weight is None:
        return
    if not isinstance(ttt_conv_weight, torch.Tensor):
        raise TypeError("ttt_conv_weight must be torch.Tensor")
    if ttt_conv_weight.ndim != 3:
        raise RuntimeError("ttt_conv_weight must be 3D")
    hidden = int(base_weight.shape[0])
    if int(ttt_conv_weight.shape[0]) != hidden or int(ttt_conv_weight.shape[1]) != 1:
        raise RuntimeError("ttt_conv_weight must be depthwise hidden_size x 1 x kernel")


def _apply_ttt_conv_to_rows(target: torch.Tensor, *, conv_weight: torch.Tensor | None, chunk_size: int) -> torch.Tensor:
    if conv_weight is None:
        return target
    if int(target.shape[0]) <= 0:
        return target
    hidden = int(target.shape[1])
    weight = conv_weight.to(device=target.device, dtype=target.dtype)
    kernel = int(weight.shape[2])
    padding = kernel // 2
    conv_input = target.view(1, int(target.shape[0]), hidden).transpose(1, 2)
    conv = F.conv1d(conv_input, weight, bias=None, stride=1, padding=padding, groups=hidden).transpose(1, 2).reshape(
        int(target.shape[0]), hidden
    )
    if int(conv.shape[0]) != int(target.shape[0]):
        conv = conv[: int(target.shape[0])]
    return conv


def _require_fast_weight_row_keys(row_keys: object, *, rows: int) -> tuple[FastWeightKey, ...]:
    if not isinstance(row_keys, Sequence):
        raise TypeError("row_key_provider must return a sequence of FastWeightKey")
    normalized = tuple(row_keys)
    if len(normalized) != int(rows):
        raise RuntimeError(f"row_key_provider returned {len(normalized)} keys for {int(rows)} rows")
    for key in normalized:
        if not isinstance(key, FastWeightKey):
            raise TypeError("row_key_provider must return only FastWeightKey values")
    return normalized


def _run_grouped_down_proj_fast_weight_step(
    *,
    hidden: torch.Tensor,
    target: torch.Tensor,
    base_weight: torch.Tensor,
    state: FastWeightState,
    row_keys: tuple[FastWeightKey, ...],
    chunk_size: int,
    learning_rate: float,
    update_enabled: bool,
    cache_tail: bool,
    ttt_proj_weight: torch.Tensor | None = None,
    ttt_conv_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if int(hidden.shape[0]) != len(row_keys):
        raise RuntimeError("row key count must match hidden row count")
    output = hidden.new_empty((int(hidden.shape[0]), int(base_weight.shape[0])))
    grouped_rows: dict[FastWeightKey, list[int]] = {}
    for row_idx, key in enumerate(row_keys):
        grouped_rows.setdefault(key, []).append(int(row_idx))
    for group_key, rows in grouped_rows.items():
        row_tensor = torch.as_tensor(rows, device=hidden.device, dtype=torch.long)
        step = run_down_proj_fast_weight_step(
            hidden=hidden.index_select(0, row_tensor),
            target=target.index_select(0, row_tensor),
            base_weight=base_weight,
            state=state,
            key=group_key,
            chunk_size=chunk_size,
            learning_rate=learning_rate,
            update_enabled=update_enabled,
            cache_tail=cache_tail,
            ttt_proj_weight=ttt_proj_weight,
            ttt_conv_weight=ttt_conv_weight,
        )
        output.index_copy_(0, row_tensor, step.output.to(device=output.device, dtype=output.dtype))
    return output


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
    ttt_proj_weight: torch.Tensor | None = None,
    ttt_conv_weight: torch.Tensor | None = None,
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
    _validate_ttt_proj_weight(ttt_proj_weight, w)
    _validate_ttt_conv_weight(ttt_conv_weight, w)

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
        outputs.append(h_chunk.matmul(fast_w.to(device=h_chunk.device, dtype=h_chunk.dtype).t()))
        t_chunk = _apply_ttt_conv_to_rows(t_work[start:end], conv_weight=ttt_conv_weight, chunk_size=chunk)
        if ttt_proj_weight is None:
            step_w = contract("c h, c d -> d h", h_chunk, t_chunk)
        else:
            proj = ttt_proj_weight.to(device=t_chunk.device, dtype=t_chunk.dtype)
            step_w = contract("c h, c d, d e -> e h", h_chunk, t_chunk, proj)
        fast_w = fast_w + (step_w * float(learning_rate)).to(device=fast_w.device, dtype=fast_w.dtype)
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



def install_mlp_fast_weight_wrapper(
    *,
    module: object,
    state: FastWeightState,
    key: FastWeightKey,
    row_key_provider: Callable[[int], Sequence[FastWeightKey] | None] | None = None,
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
    ttt_proj = getattr(module, "ttt_proj", None)
    ttt_conv = getattr(module, "ttt_conv", None)
    if gate_up_proj is None or not callable(getattr(gate_up_proj, "forward", None)):
        raise RuntimeError("MLP fast-weight wrapper requires gate_up_proj.forward")
    if not callable(act_fn):
        raise RuntimeError("MLP fast-weight wrapper requires callable act_fn")
    if down_proj is None or not callable(getattr(down_proj, "forward", None)):
        raise RuntimeError("MLP fast-weight wrapper requires down_proj.forward")
    _require_supported_down_proj(down_proj, require_single_tp=require_single_tp)
    weight = module_weight_view(down_proj)
    ttt_proj_weight = module_weight_view(ttt_proj) if ttt_proj is not None else None
    ttt_conv_weight = getattr(ttt_conv, "weight", None) if ttt_conv is not None else None
    _validate_ttt_proj_weight(ttt_proj_weight, weight)
    _validate_ttt_conv_weight(ttt_conv_weight, weight)
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
        if row_key_provider is not None:
            provided_row_keys = row_key_provider(int(activated.shape[0]))
            if provided_row_keys is None:
                return original_forward(hidden)
            row_keys = _require_fast_weight_row_keys(provided_row_keys, rows=int(activated.shape[0]))
            return _run_grouped_down_proj_fast_weight_step(
                hidden=activated,
                target=hidden,
                base_weight=weight,
                state=state,
                row_keys=row_keys,
                chunk_size=chunk_size,
                learning_rate=learning_rate,
                update_enabled=update_enabled,
                cache_tail=update_enabled,
                ttt_proj_weight=ttt_proj_weight,
                ttt_conv_weight=ttt_conv_weight,
            )
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
            ttt_proj_weight=ttt_proj_weight,
            ttt_conv_weight=ttt_conv_weight,
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
