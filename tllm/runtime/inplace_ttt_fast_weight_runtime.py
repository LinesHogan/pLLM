#!/usr/bin/env python3
"""Runtime lifecycle helpers for In-Place TTT fast-weight MLP wrappers."""

from __future__ import annotations

import os
from typing import Iterable, Sequence

from tllm.runtime.fast_weight_state import (
    DownProjFastWeightOverlayHandle,
    FastWeightKey,
    FastWeightState,
    install_mlp_fast_weight_wrapper,
)

STATIC_REQUEST_ID = "__tllm_inplace_ttt_static__"


def fast_weight_runtime_enabled(runtime: object) -> bool:
    cfg = getattr(runtime, "config", None)
    return bool(getattr(cfg, "inplace_ttt_fast_weight_enabled", False)) or os.getenv("TLLM_INPLACE_TTT_FAST_WEIGHT") == "1"


def runtime_fast_weight_phase(runtime: object) -> str:
    phase = str(getattr(runtime, "inplace_ttt_fast_weight_phase", "prefill") or "prefill")
    return "decode" if phase == "decode" else "prefill"


def restore_inplace_ttt_fast_weight_wrappers(runtime: object) -> int:
    handles = getattr(runtime, "inplace_ttt_fast_weight_handles", None)
    if not isinstance(handles, dict):
        return 0
    restored = 0
    for handle in list(handles.values()):
        if isinstance(handle, DownProjFastWeightOverlayHandle):
            handle.restore()
            restored += 1
    handles.clear()
    setattr(runtime, "inplace_ttt_fast_weight_hook_spec", None)
    return restored


def _configured_layers(runtime: object, layer_count: int) -> tuple[int, ...]:
    cfg = getattr(runtime, "config", None)
    raw_layers = getattr(cfg, "inplace_ttt_fast_weight_layers", (0,))
    if raw_layers is None:
        return tuple(range(int(layer_count)))
    layers: list[int] = []
    for layer in raw_layers:
        idx = int(layer)
        if idx < 0:
            idx = int(layer_count) + idx
        if idx < 0 or idx >= int(layer_count):
            raise RuntimeError(f"In-Place TTT fast-weight layer index out of range: {layer}")
        if idx not in layers:
            layers.append(idx)
    return tuple(layers)


def _model_layers(model: object) -> Sequence[object]:
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        layers = getattr(model, "layers", None)
    if layers is None:
        raise RuntimeError("In-Place TTT fast-weight runtime requires model.layers")
    return layers


def iter_configured_mlps(model: object, layers: Iterable[int]) -> Iterable[tuple[int, object]]:
    model_layers = _model_layers(model)
    for layer_idx in layers:
        layer = model_layers[int(layer_idx)]
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise RuntimeError(f"In-Place TTT fast-weight layer {layer_idx} does not expose mlp")
        yield int(layer_idx), mlp


def _hook_spec(runtime: object, runner: object, layers: tuple[int, ...]) -> tuple[object, ...]:
    cfg = getattr(runtime, "config", None)
    model = getattr(runner, "model", None)
    return (
        id(model),
        layers,
        int(getattr(cfg, "inplace_ttt_fast_weight_chunk_size", 4096)),
        float(getattr(cfg, "inplace_ttt_fast_weight_lr", 1e-3)),
        bool(getattr(cfg, "inplace_ttt_fast_weight_decode_update_enabled", False)),
        str(getattr(cfg, "inplace_ttt_fast_weight_key_mode", "static") or "static"),
        str(getattr(cfg, "inplace_ttt_fast_weight_static_request_id", STATIC_REQUEST_ID) or STATIC_REQUEST_ID),
    )


def _active_request_count(runner: object) -> int:
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        return 0
    return sum(1 for req_id in list(getattr(input_batch, "req_ids", ()) or ())[: int(getattr(input_batch, "num_reqs", 0) or 0)] if req_id is not None)


def require_static_key_single_request(runtime: object, runner: object) -> None:
    if not fast_weight_runtime_enabled(runtime):
        return
    cfg = getattr(runtime, "config", None)
    key_mode = str(getattr(cfg, "inplace_ttt_fast_weight_key_mode", "static") or "static")
    if key_mode != "static":
        return
    active = _active_request_count(runner)
    if active > 1:
        raise RuntimeError("In-Place TTT fast-weight static key mode requires a single active request")


def maybe_install_inplace_ttt_fast_weight_wrappers(runtime: object, runner: object) -> int:
    if not fast_weight_runtime_enabled(runtime):
        return 0
    cfg = getattr(runtime, "config", None)
    key_mode = str(getattr(cfg, "inplace_ttt_fast_weight_key_mode", "static") or "static")
    if key_mode != "static":
        raise RuntimeError("In-Place TTT fast-weight runtime currently supports only key_mode='static'")

    model = getattr(runner, "model", None)
    if model is None:
        raise RuntimeError("In-Place TTT fast-weight runtime requires runner.model")
    model_layers = _model_layers(model)
    layers = _configured_layers(runtime, len(model_layers))
    spec = _hook_spec(runtime, runner, layers)
    if getattr(runtime, "inplace_ttt_fast_weight_hook_spec", None) == spec:
        handles = getattr(runtime, "inplace_ttt_fast_weight_handles", None)
        if isinstance(handles, dict) and handles:
            return 0

    restore_inplace_ttt_fast_weight_wrappers(runtime)
    state = getattr(runtime, "inplace_ttt_fast_weight_state", None)
    if not isinstance(state, FastWeightState):
        state = FastWeightState()
        setattr(runtime, "inplace_ttt_fast_weight_state", state)
    handles: dict[str, DownProjFastWeightOverlayHandle] = {}
    fail_on_unsupported = bool(getattr(cfg, "inplace_ttt_fast_weight_fail_on_unsupported", False))
    installed = 0
    skipped = 0
    static_request_id = str(getattr(cfg, "inplace_ttt_fast_weight_static_request_id", STATIC_REQUEST_ID) or STATIC_REQUEST_ID)

    for layer_idx, mlp in iter_configured_mlps(model, layers):
        try:
            handle = install_mlp_fast_weight_wrapper(
                module=mlp,
                state=state,
                key=FastWeightKey(request_id=static_request_id, sample_idx=0, layer=layer_idx),
                chunk_size=int(getattr(cfg, "inplace_ttt_fast_weight_chunk_size", 4096)),
                learning_rate=float(getattr(cfg, "inplace_ttt_fast_weight_lr", 1e-3)),
                phase_getter=lambda runtime=runtime: runtime_fast_weight_phase(runtime),
                decode_update_enabled=bool(getattr(cfg, "inplace_ttt_fast_weight_decode_update_enabled", False)),
                require_single_tp=True,
            )
        except Exception:
            skipped += 1
            if fail_on_unsupported:
                raise
            continue
        handles[f"model.layers.{layer_idx}.mlp"] = handle
        installed += 1

    setattr(runtime, "inplace_ttt_fast_weight_handles", handles)
    setattr(runtime, "inplace_ttt_fast_weight_hook_spec", spec)
    setattr(runtime, "inplace_ttt_fast_weight_installed_count", installed)
    setattr(runtime, "inplace_ttt_fast_weight_skipped_count", skipped)
    return installed


def update_inplace_ttt_fast_weight_phase_from_runner(runtime: object, runner: object) -> str:
    if not fast_weight_runtime_enabled(runtime):
        return runtime_fast_weight_phase(runtime)
    require_static_key_single_request(runtime, runner)
    input_batch = getattr(runner, "input_batch", None)
    phase = "prefill"
    if input_batch is not None:
        req_ids = list(getattr(input_batch, "req_ids", ()) or ())[: int(getattr(input_batch, "num_reqs", 0) or 0)]
        req_id_to_index = getattr(input_batch, "req_id_to_index", {}) or {}
        num_prompt_tokens = getattr(input_batch, "num_prompt_tokens", {}) or {}
        num_computed_tokens = getattr(input_batch, "num_computed_tokens_cpu", {}) or {}
        for req_id in req_ids:
            if req_id is None:
                continue
            req_idx = req_id_to_index.get(req_id)
            if req_idx is None:
                continue
            if int(num_computed_tokens[req_idx]) >= int(num_prompt_tokens[req_idx]):
                phase = "decode"
                break
    setattr(runtime, "inplace_ttt_fast_weight_phase", phase)
    return phase
