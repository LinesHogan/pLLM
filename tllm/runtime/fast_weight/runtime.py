#!/usr/bin/env python3
"""Runtime lifecycle helpers for In-Place TTT fast-weight MLP wrappers."""

from __future__ import annotations

import os
from typing import Iterable, Sequence

from tllm.runtime.fast_weight.state import (
    DownProjFastWeightOverlayHandle,
    FastWeightKey,
    FastWeightState,
    install_mlp_fast_weight_wrapper,
    normalize_fast_weight_mode,
)
from tllm.runtime.token_batch_metadata import PackedTokenRow, RequestSampleKey, RuntimeTokenBatchMeta, RuntimeTokenRow

STATIC_REQUEST_ID = "__tllm_fast_weight_static__"


def fast_weight_runtime_enabled(runtime: object) -> bool:
    cfg = getattr(runtime, "config", None)
    return bool(getattr(cfg, "fast_weight_enabled", False)) or os.getenv("TLLM_INPLACE_TTT_FAST_WEIGHT") == "1"


def runtime_fast_weight_phase(runtime: object) -> str:
    phase = str(getattr(runtime, "fast_weight_phase", "prefill") or "prefill")
    return "decode" if phase == "decode" else "prefill"


def restore_fast_weight_wrappers(runtime: object) -> int:
    handles = getattr(runtime, "fast_weight_handles", None)
    if not isinstance(handles, dict):
        return 0
    restored = 0
    for handle in list(handles.values()):
        if isinstance(handle, DownProjFastWeightOverlayHandle):
            handle.restore()
            restored += 1
    handles.clear()
    setattr(runtime, "fast_weight_hook_spec", None)
    return restored


def _configured_layers(runtime: object, layer_count: int) -> tuple[int, ...]:
    cfg = getattr(runtime, "config", None)
    raw_layers = getattr(cfg, "fast_weight_layers", (0,))
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
        normalize_fast_weight_mode(getattr(cfg, "fast_weight_mode", "exact")),
        int(getattr(cfg, "fast_weight_chunk_size", 4096)),
        float(getattr(cfg, "fast_weight_lr", 1e-3)),
        bool(getattr(cfg, "fast_weight_decode_update_enabled", False)),
        str(getattr(cfg, "fast_weight_key_mode", "static") or "static"),
        str(getattr(cfg, "fast_weight_static_request_id", STATIC_REQUEST_ID) or STATIC_REQUEST_ID),
        int(getattr(cfg, "fast_weight_max_slots", 64) or 0),
        int(getattr(cfg, "fast_weight_max_bytes", 0) or 0),
    )


def _active_request_count(runner: object) -> int:
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        return 0
    return sum(1 for req_id in list(getattr(input_batch, "req_ids", ()) or ())[: int(getattr(input_batch, "num_reqs", 0) or 0)] if req_id is not None)


def _resolve_prompt_sample(runtime: object, req_id: object) -> tuple[int, int]:
    request_id = str(req_id)
    prompt_map = getattr(runtime, "reqid_to_promptidx", {}) or {}
    sample_map = getattr(runtime, "reqid_to_sampleidx", {}) or {}
    prompt_idx = prompt_map.get(request_id)
    if prompt_idx is not None:
        return int(prompt_idx), int(sample_map.get(request_id, 0))
    if "_" in request_id:
        maybe_sample_idx, parent_req_id = request_id.split("_", 1)
        if maybe_sample_idx.isdigit():
            parent_prompt_idx = prompt_map.get(parent_req_id)
            if parent_prompt_idx is not None:
                return int(parent_prompt_idx), int(maybe_sample_idx)
    return -1, 0


def _index_int(values: object, idx: int, *, default: int = 0) -> int:
    try:
        return int(values[idx])  # type: ignore[index]
    except Exception:
        return int(default)


def update_fast_weight_metadata_from_runner(
    runtime: object,
    runner: object,
    *,
    num_scheduled_tokens_np: object | None = None,
) -> str:
    if not fast_weight_runtime_enabled(runtime):
        return runtime_fast_weight_phase(runtime)
    input_batch = getattr(runner, "input_batch", None)
    rows: list[RuntimeTokenRow] = []
    phase = "prefill"
    if input_batch is not None:
        raw_req_ids = getattr(input_batch, "req_ids", ())
        req_ids = list(raw_req_ids if raw_req_ids is not None else ())[: int(getattr(input_batch, "num_reqs", 0) or 0)]
        req_id_to_index = getattr(input_batch, "req_id_to_index", None)
        if req_id_to_index is None:
            req_id_to_index = {}
        num_prompt_tokens = getattr(input_batch, "num_prompt_tokens", None)
        if num_prompt_tokens is None:
            num_prompt_tokens = {}
        num_computed_tokens = getattr(input_batch, "num_computed_tokens_cpu", None)
        if num_computed_tokens is None:
            num_computed_tokens = {}
        row_base = 0
        engine_step_id = int(getattr(runtime, "event_step_id", 0) or 0)
        for req_pos, req_id in enumerate(req_ids):
            if req_id is None:
                continue
            scheduled = _index_int(num_scheduled_tokens_np, req_pos, default=1) if num_scheduled_tokens_np is not None else 1
            if scheduled <= 0:
                continue
            req_idx = req_id_to_index.get(req_id)
            prompt_len = _index_int(num_prompt_tokens, int(req_idx), default=0) if req_idx is not None else 0
            computed = _index_int(num_computed_tokens, int(req_idx), default=0) if req_idx is not None else 0
            prefill_len = max(0, min(int(scheduled), int(prompt_len) - int(computed)))
            prompt_idx, sample_idx = _resolve_prompt_sample(runtime, req_id)
            request_key = RequestSampleKey(request_id=str(req_id), prompt_idx=int(prompt_idx), sample_idx=int(sample_idx))
            for local_i in range(int(scheduled)):
                row_phase = "prefill" if local_i < prefill_len else "decode"
                if row_phase == "decode":
                    phase = "decode"
                rows.append(
                    RuntimeTokenRow(
                        key=request_key,
                        row=PackedTokenRow(
                            packed_row_idx=int(row_base + local_i),
                            logical_row_idx=len(rows),
                            token_offset=int(computed + local_i),
                        ),
                        phase=row_phase,
                        engine_step_id=engine_step_id,
                    )
                )
            row_base += int(scheduled)
    setattr(runtime, "fast_weight_current_token_batch_meta", RuntimeTokenBatchMeta(rows=tuple(rows), row_layout="packed"))
    setattr(runtime, "fast_weight_phase", phase)
    return phase


def require_static_key_single_request(runtime: object, runner: object) -> None:
    if not fast_weight_runtime_enabled(runtime):
        return
    cfg = getattr(runtime, "config", None)
    key_mode = str(getattr(cfg, "fast_weight_key_mode", "static") or "static")
    if key_mode != "static":
        return
    active = _active_request_count(runner)
    if active > 1:
        raise RuntimeError("In-Place TTT fast-weight static key mode requires a single active request")


def current_token_batch_fast_weight_keys(
    runtime: object,
    *,
    layer: int,
    rows: int,
) -> tuple[FastWeightKey, ...] | None:
    meta = getattr(runtime, "fast_weight_current_token_batch_meta", None)
    if not isinstance(meta, RuntimeTokenBatchMeta):
        return None
    if meta.row_layout != "packed":
        raise RuntimeError("In-Place TTT request_sample key mode requires packed row metadata")
    if int(rows) != len(meta.rows):
        raise RuntimeError(f"In-Place TTT token metadata row count mismatch: hidden_rows={int(rows)} meta_rows={len(meta.rows)}")
    return tuple(
        FastWeightKey(
            request_id=row.key.request_id,
            sample_idx=int(row.key.sample_idx),
            layer=int(layer),
        )
        for row in meta.rows
    )


def maybe_install_fast_weight_wrappers(runtime: object, runner: object) -> int:
    if not fast_weight_runtime_enabled(runtime):
        return 0
    cfg = getattr(runtime, "config", None)
    mode = normalize_fast_weight_mode(getattr(cfg, "fast_weight_mode", "exact"))
    if mode == "low_rank":
        restore_fast_weight_wrappers(runtime)
        raise NotImplementedError(
            "In-Place TTT fast-weight low_rank mode is not implemented yet; "
            "exact mode is the only runtime state representation available."
        )
    key_mode = str(getattr(cfg, "fast_weight_key_mode", "static") or "static")
    if key_mode not in {"static", "request_sample"}:
        raise RuntimeError("In-Place TTT fast-weight key_mode must be 'static' or 'request_sample'")

    model = getattr(runner, "model", None)
    if model is None:
        raise RuntimeError("In-Place TTT fast-weight runtime requires runner.model")
    model_layers = _model_layers(model)
    layers = _configured_layers(runtime, len(model_layers))
    spec = _hook_spec(runtime, runner, layers)
    if getattr(runtime, "fast_weight_hook_spec", None) == spec:
        handles = getattr(runtime, "fast_weight_handles", None)
        if isinstance(handles, dict) and handles:
            return 0

    restore_fast_weight_wrappers(runtime)
    state = getattr(runtime, "fast_weight_state", None)
    if not isinstance(state, FastWeightState):
        state = FastWeightState(mode=mode)
        setattr(runtime, "fast_weight_state", state)
    max_slots = int(getattr(cfg, "fast_weight_max_slots", 64) or 0)
    max_bytes = int(getattr(cfg, "fast_weight_max_bytes", 0) or 0)
    state.set_limits(
        max_slots=max_slots if max_slots > 0 else None,
        max_bytes=max_bytes if max_bytes > 0 else None,
    )
    handles: dict[str, DownProjFastWeightOverlayHandle] = {}
    fail_on_unsupported = bool(getattr(cfg, "fast_weight_fail_on_unsupported", False))
    installed = 0
    skipped = 0
    static_request_id = str(getattr(cfg, "fast_weight_static_request_id", STATIC_REQUEST_ID) or STATIC_REQUEST_ID)

    for layer_idx, mlp in iter_configured_mlps(model, layers):
        static_key = FastWeightKey(request_id=static_request_id, sample_idx=0, layer=layer_idx)
        row_key_provider = None
        if key_mode == "request_sample":
            row_key_provider = lambda rows, runtime=runtime, layer_idx=layer_idx: current_token_batch_fast_weight_keys(
                runtime,
                layer=int(layer_idx),
                rows=int(rows),
            )
        try:
            handle = install_mlp_fast_weight_wrapper(
                module=mlp,
                state=state,
                key=static_key,
                row_key_provider=row_key_provider,
                chunk_size=int(getattr(cfg, "fast_weight_chunk_size", 4096)),
                learning_rate=float(getattr(cfg, "fast_weight_lr", 1e-3)),
                phase_getter=lambda runtime=runtime: runtime_fast_weight_phase(runtime),
                decode_update_enabled=bool(getattr(cfg, "fast_weight_decode_update_enabled", False)),
                require_single_tp=True,
            )
        except Exception:
            skipped += 1
            if fail_on_unsupported:
                raise
            continue
        handles[f"model.layers.{layer_idx}.mlp"] = handle
        installed += 1

    setattr(runtime, "fast_weight_handles", handles)
    setattr(runtime, "fast_weight_hook_spec", spec)
    setattr(runtime, "fast_weight_installed_count", installed)
    setattr(runtime, "fast_weight_skipped_count", skipped)
    return installed


def update_fast_weight_phase_from_runner(runtime: object, runner: object) -> str:
    if not fast_weight_runtime_enabled(runtime):
        return runtime_fast_weight_phase(runtime)
    require_static_key_single_request(runtime, runner)
    return update_fast_weight_metadata_from_runner(runtime, runner)
