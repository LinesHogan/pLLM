#!/usr/bin/env python3
"""Runtime lifecycle helpers for In-Place TTT fast-weight MLP wrappers."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Callable, Protocol

from tllm.backends.protocol import BackendDriver, GraphSafety, RunnerBinding, runner_binding_for

from tllm.contracts.request_identity import resolve_prompt_sample
from tllm.ports.base import PortWrite
from tllm.ports.module_weight_view import ModuleWeightViewLocator
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.fast_weight.state import (
    DownProjFastWeightOverlayHandle,
    FastWeightKey,
    FastWeightState,
    install_mlp_fast_weight_wrapper,
    normalize_fast_weight_mode,
)
from tllm.runtime.token_batch_metadata import (
    PackedTokenRow,
    RequestSampleKey,
    RuntimeTokenBatchMeta,
    RuntimeTokenPhase,
    RuntimeTokenRow,
)

STATIC_REQUEST_ID = "__tllm_fast_weight_static__"


class _FastWeightConfigLike(Protocol):
    @property
    def fast_weight_enabled(self) -> bool: ...

    @property
    def fast_weight_layers(self) -> Sequence[int] | None: ...

    @property
    def fast_weight_mode(self) -> str: ...

    @property
    def fast_weight_chunk_size(self) -> int: ...

    @property
    def fast_weight_lr(self) -> float: ...

    @property
    def fast_weight_decode_update_enabled(self) -> bool: ...

    @property
    def fast_weight_key_mode(self) -> str: ...

    @property
    def fast_weight_static_request_id(self) -> str: ...

    @property
    def fast_weight_max_slots(self) -> int: ...

    @property
    def fast_weight_max_bytes(self) -> int: ...


class _FastWeightRuntimeLike(Protocol):
    @property
    def config(self) -> _FastWeightConfigLike: ...

    dispatch_plan: DispatchPlan | None
    @property
    def reqid_to_promptidx(self) -> Mapping[str, int]: ...

    @property
    def reqid_to_sampleidx(self) -> Mapping[str, int]: ...
    event_step_id: int
    fast_weight_state: FastWeightState | None
    fast_weight_binding: RunnerBinding | None
    fast_weight_handles: dict[str, DownProjFastWeightOverlayHandle]
    fast_weight_hook_spec: tuple[object, ...] | None
    fast_weight_phase: RuntimeTokenPhase
    fast_weight_current_token_batch_meta: RuntimeTokenBatchMeta | None
    fast_weight_installed_count: int
    fast_weight_skipped_count: int


def fast_weight_runtime_enabled(runtime: _FastWeightRuntimeLike) -> bool:
    return bool(runtime.config.fast_weight_enabled) or os.getenv("TLLM_INPLACE_TTT_FAST_WEIGHT") == "1"


def runtime_fast_weight_phase(runtime: _FastWeightRuntimeLike) -> RuntimeTokenPhase:
    return runtime.fast_weight_phase


def restore_fast_weight_wrappers(runtime: _FastWeightRuntimeLike) -> int:
    binding = runtime.fast_weight_binding
    if isinstance(binding, RunnerBinding):
        restored = int(binding.driver.restore_module_weight_overlays(binding))
        runtime.fast_weight_binding = None
        runtime.fast_weight_handles.clear()
        runtime.fast_weight_hook_spec = None
        runtime.fast_weight_installed_count = 0
        runtime.fast_weight_skipped_count = 0
        return restored
    restored = 0
    for handle in list(runtime.fast_weight_handles.values()):
        if isinstance(handle, DownProjFastWeightOverlayHandle):
            handle.restore()
            restored += 1
    runtime.fast_weight_handles.clear()
    runtime.fast_weight_hook_spec = None
    runtime.fast_weight_installed_count = 0
    runtime.fast_weight_skipped_count = 0
    return restored


def _configured_layers(runtime: _FastWeightRuntimeLike, layer_count: int) -> tuple[int, ...]:
    raw_layers = runtime.config.fast_weight_layers
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


def _hook_spec(
    runtime: _FastWeightRuntimeLike,
    runner: object,
    targets: tuple[tuple[int, str], ...],
    binding: RunnerBinding,
) -> tuple[object, ...]:
    cfg = runtime.config
    return (
        binding.driver.runner_model_identity(runner),
        targets,
        id(binding),
        normalize_fast_weight_mode(cfg.fast_weight_mode),
        int(cfg.fast_weight_chunk_size),
        float(cfg.fast_weight_lr),
        bool(cfg.fast_weight_decode_update_enabled),
        str(cfg.fast_weight_key_mode or "static"),
        str(cfg.fast_weight_static_request_id or STATIC_REQUEST_ID),
        int(cfg.fast_weight_max_slots or 0),
        int(cfg.fast_weight_max_bytes or 0),
    )


def _active_request_count(driver: BackendDriver, runner: object) -> int:
    return int(driver.runner_active_request_count(runner))


def _resolve_prompt_sample(runtime: _FastWeightRuntimeLike, req_id: object) -> tuple[int, int]:
    return resolve_prompt_sample(
        req_id,
        prompt_indices=runtime.reqid_to_promptidx,
        sample_indices=runtime.reqid_to_sampleidx,
    )


def update_fast_weight_metadata_from_runner(
    runtime: _FastWeightRuntimeLike,
    runner: object,
    *,
    num_scheduled_tokens_np: object | None = None,
    input_batch: object | None = None,
    driver: BackendDriver | None = None,
) -> str:
    if not fast_weight_runtime_enabled(runtime):
        return runtime_fast_weight_phase(runtime)
    binding = runner_binding_for(runner)
    backend_driver = binding.driver if isinstance(binding, RunnerBinding) else driver
    if backend_driver is None:
        raise RuntimeError(
            "fast-weight metadata requires an active RunnerBinding or backend driver"
        )
    normalized = backend_driver.fast_weight_packed_row_plan_inputs(
        runner=runner,
        num_scheduled_tokens=num_scheduled_tokens_np,
        input_batch=input_batch,
    )
    rows: list[RuntimeTokenRow] = []
    phase: RuntimeTokenPhase = "prefill"
    if normalized.request_ids:
        from tllm.core import plan_packed_prefill_rows

        row_plan = plan_packed_prefill_rows(
            list(normalized.scheduled_tokens),
            list(normalized.prompt_tokens),
            list(normalized.computed_tokens),
            sum(normalized.scheduled_tokens),
        )
        for logical_row_idx, (request_ordinal, packed_row_idx, token_offset, row_phase) in enumerate(
            zip(
                row_plan.request_ordinals,
                row_plan.packed_row_indices,
                row_plan.token_offsets,
                row_plan.phases,
            )
        ):
            req_id = normalized.request_ids[int(request_ordinal)]
            prompt_idx, sample_idx = _resolve_prompt_sample(runtime, req_id)
            request_key = RequestSampleKey(request_id=str(req_id), prompt_idx=int(prompt_idx), sample_idx=int(sample_idx))
            if row_phase == "decode":
                phase = "decode"
            elif row_phase != "prefill":
                raise RuntimeError(f"packed prefill row planner returned invalid phase: {row_phase!r}")
            rows.append(
                RuntimeTokenRow(
                    key=request_key,
                    row=PackedTokenRow(
                        packed_row_idx=int(packed_row_idx),
                        logical_row_idx=int(logical_row_idx),
                        token_offset=int(token_offset),
                    ),
                    phase="decode" if row_phase == "decode" else "prefill",
                    engine_step_id=int(runtime.event_step_id),
                )
            )
    runtime.fast_weight_current_token_batch_meta = RuntimeTokenBatchMeta(rows=tuple(rows), row_layout="packed")
    runtime.fast_weight_phase = phase
    return phase


def require_static_key_single_request(runtime: _FastWeightRuntimeLike, runner: object) -> None:
    if not fast_weight_runtime_enabled(runtime):
        return
    key_mode = str(runtime.config.fast_weight_key_mode or "static")
    if key_mode != "static":
        return
    binding = _runner_binding(runner, None)
    active = _active_request_count(binding.driver, runner)
    if active > 1:
        raise RuntimeError("In-Place TTT fast-weight static key mode requires a single active request")


def current_token_batch_fast_weight_keys(
    runtime: _FastWeightRuntimeLike,
    *,
    layer: int,
    rows: int,
) -> tuple[FastWeightKey, ...] | None:
    meta = runtime.fast_weight_current_token_batch_meta
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


def current_token_batch_fast_weight_phases(
    runtime: _FastWeightRuntimeLike,
    *,
    rows: int,
) -> tuple[str, ...] | None:
    meta = runtime.fast_weight_current_token_batch_meta
    if not isinstance(meta, RuntimeTokenBatchMeta):
        return None
    if meta.row_layout != "packed":
        raise RuntimeError("In-Place TTT request_sample key mode requires packed row metadata")
    if int(rows) != len(meta.rows):
        raise RuntimeError(
            f"In-Place TTT token metadata row count mismatch: hidden_rows={int(rows)} "
            f"meta_rows={len(meta.rows)}"
        )
    return tuple(row.phase for row in meta.rows)


def _declared_module_weight_writes(runtime: _FastWeightRuntimeLike) -> tuple[PortWrite, ...]:
    plan = runtime.dispatch_plan
    if plan is None:
        return ()
    return tuple(write for _consumer_id, _flow, write in plan.module_weight_writes())


def _runner_binding(runner: object, binding: RunnerBinding | None) -> RunnerBinding:
    candidate = binding if binding is not None else runner_binding_for(runner)
    if not isinstance(candidate, RunnerBinding) or candidate.runner is not runner or not candidate.active:
        raise RuntimeError("ModuleWeightView overlay requires an active backend RunnerBinding")
    return candidate


def _overlay_targets(
    *,
    runtime: _FastWeightRuntimeLike,
    runner: object,
    binding: RunnerBinding,
    declarations: tuple[PortWrite, ...],
) -> tuple[ModuleWeightViewLocator, ...]:
    overlay = binding.capabilities.module_weight_overlay
    layer_count = int(binding.driver.runner_model_layer_count(runner))
    configured_layers = _configured_layers(runtime, layer_count)
    if declarations:
        if not fast_weight_runtime_enabled(runtime):
            raise RuntimeError("ModuleWeightView overlay provider is disabled for an active write")
        if overlay is None:
            first = declarations[0]
            locator = first.locator
            if not isinstance(locator, ModuleWeightViewLocator):
                raise TypeError("ModuleWeightView write must carry ModuleWeightViewLocator")
            raise RuntimeError(
                f"backend {binding.capabilities.backend_id!r} does not support "
                f"module_weight_view/{first.mode}/{locator.phase}"
            )
    elif not fast_weight_runtime_enabled(runtime):
        return ()
    if overlay is None:
        raise RuntimeError(
            f"backend {binding.capabilities.backend_id!r} does not support ModuleWeightView overlays"
        )
    if overlay.graph_safety != GraphSafety.EAGER_ONLY:
        raise RuntimeError("ModuleWeightView overlay backend capability must be EAGER_ONLY")
    from tllm.runtime.graph_policy import GraphMode, resolve_graph_mode

    if resolve_graph_mode(runner) != GraphMode.EAGER:
        raise RuntimeError("ModuleWeightView overlay requires eager graph mode")
    tp_size = int(binding.driver.runner_tensor_parallel_size(runner))
    if tp_size not in overlay.tensor_parallel_sizes:
        raise RuntimeError(
            "ModuleWeightView overlay currently requires tensor_parallel_size=1; "
            f"got {tp_size}"
        )
    key_mode = str(runtime.config.fast_weight_key_mode or "static")
    if key_mode == "request_sample" and not overlay.request_keyed:
        raise RuntimeError("backend ModuleWeightView overlay is not request-keyed")

    normalized: list[ModuleWeightViewLocator] = []
    declared_layers: list[int] = []
    for write in declarations:
        locator = write.locator
        if not isinstance(locator, ModuleWeightViewLocator):
            raise TypeError("ModuleWeightView write must carry ModuleWeightViewLocator")
        mode = str(write.mode or "")
        capability = f"module_weight_view/{mode}/{locator.phase}"
        if capability not in binding.capabilities.capability_names():
            raise RuntimeError(
                f"backend {binding.capabilities.backend_id!r} does not support {capability}"
            )
        if locator.module not in overlay.modules:
            raise RuntimeError(f"backend does not support ModuleWeightView module {locator.module!r}")
        layer = int(locator.layer)
        if layer < 0:
            layer += layer_count
        if layer < 0 or layer >= layer_count:
            raise RuntimeError(f"ModuleWeightView layer index out of range: {locator.layer}")
        declared_layers.append(layer)
        candidate = ModuleWeightViewLocator(layer=layer, module=locator.module, phase=locator.phase)
        if all((item.layer, item.module) != (candidate.layer, candidate.module) for item in normalized):
            normalized.append(candidate)
    if declarations and set(declared_layers) != set(configured_layers):
        raise RuntimeError(
            "ModuleWeightView configured layers do not match declared locators: "
            f"configured={configured_layers} declared={tuple(sorted(set(declared_layers)))}"
        )
    if not declarations:
        normalized = [
            ModuleWeightViewLocator(layer=layer, module="mlp", phase="prefill")
            for layer in configured_layers
        ]
    return tuple(normalized)


def maybe_install_fast_weight_wrappers(
    runtime: _FastWeightRuntimeLike,
    runner: object,
    *,
    binding: RunnerBinding | None = None,
) -> int:
    declarations = _declared_module_weight_writes(runtime)
    if not declarations and not fast_weight_runtime_enabled(runtime):
        return 0
    typed_binding = _runner_binding(runner, binding)
    targets = _overlay_targets(
        runtime=runtime,
        runner=runner,
        binding=typed_binding,
        declarations=declarations,
    )
    if not targets:
        return 0
    cfg = runtime.config
    mode = normalize_fast_weight_mode(cfg.fast_weight_mode)
    if mode == "low_rank":
        restore_fast_weight_wrappers(runtime)
        raise NotImplementedError(
            "In-Place TTT fast-weight low_rank mode is not implemented yet; "
            "exact mode is the only runtime state representation available."
        )
    key_mode = str(cfg.fast_weight_key_mode or "static")
    if key_mode not in {"static", "request_sample"}:
        raise RuntimeError("In-Place TTT fast-weight key_mode must be 'static' or 'request_sample'")

    spec_targets = tuple((int(locator.layer), locator.module) for locator in targets)
    spec = _hook_spec(runtime, runner, spec_targets, typed_binding)
    if runtime.fast_weight_hook_spec == spec:
        if runtime.fast_weight_handles:
            return 0

    restore_fast_weight_wrappers(runtime)
    state = runtime.fast_weight_state
    if not isinstance(state, FastWeightState):
        state = FastWeightState(mode=mode)
        runtime.fast_weight_state = state
    max_slots = int(cfg.fast_weight_max_slots or 0)
    max_bytes = int(cfg.fast_weight_max_bytes or 0)
    state.set_limits(
        max_slots=max_slots if max_slots > 0 else None,
        max_bytes=max_bytes if max_bytes > 0 else None,
    )
    handles: dict[str, DownProjFastWeightOverlayHandle] = {}
    installed = 0
    static_request_id = str(cfg.fast_weight_static_request_id or STATIC_REQUEST_ID)

    try:
        for locator in targets:
            layer_idx = int(locator.layer)
            static_key = FastWeightKey(request_id=static_request_id, sample_idx=0, layer=layer_idx)
            row_key_provider: Callable[[int], Sequence[FastWeightKey] | None] | None = None
            row_phase_provider: Callable[[int], Sequence[str] | None] | None = None
            if key_mode == "request_sample":
                row_key_provider = lambda rows, runtime=runtime, layer_idx=layer_idx: current_token_batch_fast_weight_keys(
                    runtime,
                    layer=int(layer_idx),
                    rows=int(rows),
                )
                row_phase_provider = lambda rows, runtime=runtime: current_token_batch_fast_weight_phases(
                    runtime,
                    rows=int(rows),
                )

            def _installer(
                module: object,
                *,
                static_key: FastWeightKey = static_key,
                row_key_provider: Callable[[int], Sequence[FastWeightKey] | None] | None = row_key_provider,
                row_phase_provider: Callable[[int], Sequence[str] | None] | None = row_phase_provider,
            ) -> DownProjFastWeightOverlayHandle:
                return install_mlp_fast_weight_wrapper(
                    module=module,
                    state=state,
                    key=static_key,
                    row_key_provider=row_key_provider,
                    row_phase_provider=row_phase_provider,
                    chunk_size=int(cfg.fast_weight_chunk_size),
                    learning_rate=float(cfg.fast_weight_lr),
                    phase_getter=lambda runtime=runtime: runtime_fast_weight_phase(runtime),
                    decode_update_enabled=bool(cfg.fast_weight_decode_update_enabled),
                    require_single_tp=True,
                )

            handle = typed_binding.driver.install_module_weight_overlay(
                binding=typed_binding,
                locator=locator,
                installer=_installer,
            )
            handles[f"model.layers.{layer_idx}.{locator.module}"] = handle
            installed += 1
    except Exception:
        typed_binding.driver.restore_module_weight_overlays(typed_binding)
        handles.clear()
        runtime.fast_weight_handles = {}
        runtime.fast_weight_binding = None
        runtime.fast_weight_hook_spec = None
        runtime.fast_weight_installed_count = 0
        runtime.fast_weight_skipped_count = 0
        raise

    runtime.fast_weight_handles = handles
    runtime.fast_weight_binding = typed_binding
    runtime.fast_weight_hook_spec = spec
    runtime.fast_weight_installed_count = installed
    runtime.fast_weight_skipped_count = 0
    return installed


def update_fast_weight_phase_from_runner(runtime: _FastWeightRuntimeLike, runner: object) -> str:
    if not fast_weight_runtime_enabled(runtime):
        return runtime_fast_weight_phase(runtime)
    require_static_key_single_request(runtime, runner)
    return update_fast_weight_metadata_from_runner(runtime, runner)
