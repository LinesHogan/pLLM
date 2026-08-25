#!/usr/bin/env python3
"""Shared vLLM target resolution and lifecycle for module-weight overlays."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from tllm.backends.protocol import (
    ResolvedModuleWeightView,
    RunnerBinding,
    _RUNNER_BINDING_ATTR,
    runner_binding_for,
)
from tllm.ports.module_weight_view import ModuleWeightViewLocator

_BINDING_ATTR = _RUNNER_BINDING_ATTR


@dataclass(frozen=True)
class PackedRowPlanInputs:
    """Backend-normalized inputs for the core packed-row planner."""

    request_ids: tuple[object, ...]
    scheduled_tokens: tuple[int, ...]
    prompt_tokens: tuple[int, ...]
    computed_tokens: tuple[int, ...]


def ensure_runner_binding(*, driver: object, runner: object) -> RunnerBinding:
    existing = getattr(runner, _BINDING_ATTR, None)
    if isinstance(existing, RunnerBinding) and existing.driver is driver and existing.active:
        return existing
    capabilities = driver.capabilities()  # type: ignore[attr-defined]
    binding = RunnerBinding(driver=driver, runner=runner, capabilities=capabilities)
    setattr(runner, _BINDING_ATTR, binding)
    return binding


def get_runner_binding(runner: object) -> RunnerBinding | None:
    return runner_binding_for(runner)


def _model_layers(model: object) -> Sequence[object]:
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        layers = getattr(model, "layers", None)
    if layers is None or not hasattr(layers, "__len__") or not hasattr(layers, "__getitem__"):
        raise RuntimeError("ModuleWeightView backend requires model.layers")
    return layers


def runner_model_identity(runner: object) -> int:
    return id(getattr(runner, "model", None))


def runner_model_layer_count(runner: object) -> int:
    model = getattr(runner, "model", None)
    if model is None:
        raise RuntimeError("ModuleWeightView backend requires runner.model")
    return len(_model_layers(model))


def runner_tensor_parallel_size(runner: object) -> int:
    parallel_config = getattr(runner, "parallel_config", None)
    if parallel_config is None:
        vllm_config = getattr(runner, "vllm_config", None)
        parallel_config = getattr(vllm_config, "parallel_config", None)
    return int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)


def runner_active_request_count(runner: object) -> int:
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        return 0
    req_ids = list(getattr(input_batch, "req_ids", ()) or ())
    num_reqs = int(getattr(input_batch, "num_reqs", 0) or 0)
    return sum(1 for req_id in req_ids[:num_reqs] if req_id is not None)


def _required_index_int(values: object, idx: int, *, field_name: str) -> int:
    if values is None:
        raise RuntimeError(f"fast-weight metadata requires {field_name}")
    try:
        value = values[idx]  # type: ignore[index]
    except Exception as exc:
        raise RuntimeError(f"fast-weight metadata {field_name}[{idx}] is unavailable") from exc
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"fast-weight metadata {field_name}[{idx}] must be an integer") from exc


def _active_request_ids(
    *,
    batch: object,
    family: str,
) -> tuple[object, ...]:
    raw_req_ids = getattr(batch, "req_ids", ())
    num_reqs = int(getattr(batch, "num_reqs", 0) or 0)
    request_ids = tuple(list(raw_req_ids if raw_req_ids is not None else ())[:num_reqs])
    if len(request_ids) != num_reqs:
        raise RuntimeError(
            f"{family} packed batch does not expose a complete active request prefix"
        )
    if any(request_id is None for request_id in request_ids):
        raise RuntimeError(f"{family} packed batch contains an empty active request slot")
    return request_ids


def _packed_row_plan_inputs_from_columns(
    *,
    request_ids: tuple[object, ...],
    num_scheduled_tokens: object | None,
    scheduled_field: str,
    prompt_tokens: object | None,
    prompt_field: str,
    computed_tokens: object | None,
    computed_field: str,
) -> PackedRowPlanInputs:
    scheduled_values: list[int] = []
    prompt_values: list[int] = []
    computed_values: list[int] = []
    for req_pos in range(len(request_ids)):
        scheduled_values.append(
            _required_index_int(
                num_scheduled_tokens,
                req_pos,
                field_name=scheduled_field,
            )
        )
        prompt_values.append(
            _required_index_int(
                prompt_tokens,
                req_pos,
                field_name=prompt_field,
            )
        )
        computed_values.append(
            _required_index_int(
                computed_tokens,
                req_pos,
                field_name=computed_field,
            )
        )

    return PackedRowPlanInputs(
        request_ids=request_ids,
        scheduled_tokens=tuple(scheduled_values),
        prompt_tokens=tuple(prompt_values),
        computed_tokens=tuple(computed_values),
    )


def _packed_batch(
    *, runner: object, input_batch: object | None
) -> object | None:
    return input_batch if input_batch is not None else getattr(runner, "input_batch", None)


def mrv1_packed_row_plan_inputs(
    *,
    runner: object,
    num_scheduled_tokens: object | None,
    input_batch: object | None = None,
) -> PackedRowPlanInputs:
    """Adapt the MRv1 request-indexed scheduler columns without guessing a layout."""

    batch = _packed_batch(runner=runner, input_batch=input_batch)
    if batch is None:
        return PackedRowPlanInputs((), (), (), ())
    request_ids = _active_request_ids(batch=batch, family="MRv1")
    if not request_ids:
        return PackedRowPlanInputs((), (), (), ())
    req_id_to_index = getattr(batch, "req_id_to_index", None)
    if req_id_to_index is None:
        raise RuntimeError("MRv1 packed batch requires req_id_to_index")
    expected_indices = {
        request_id: request_ordinal
        for request_ordinal, request_id in enumerate(request_ids)
    }
    actual_indices = {
        request_id: req_id_to_index.get(request_id) for request_id in request_ids
    }
    if actual_indices != expected_indices or len(req_id_to_index) != len(request_ids):
        raise RuntimeError(
            "vLLM packed batch request indices must be a compact request-ordered prefix"
        )
    prompt_tokens = getattr(batch, "num_prompt_tokens", None)
    if prompt_tokens is None:
        raise RuntimeError("MRv1 packed batch requires num_prompt_tokens")
    computed_tokens = getattr(batch, "num_computed_tokens_cpu", None)
    if computed_tokens is None:
        raise RuntimeError("MRv1 packed batch requires num_computed_tokens_cpu")
    return _packed_row_plan_inputs_from_columns(
        request_ids=request_ids,
        num_scheduled_tokens=num_scheduled_tokens,
        scheduled_field="MRv1 num_scheduled_tokens",
        prompt_tokens=prompt_tokens,
        prompt_field="num_prompt_tokens",
        computed_tokens=computed_tokens,
        computed_field="num_computed_tokens_cpu",
    )


def mrv2_packed_row_plan_inputs(
    *,
    runner: object,
    num_scheduled_tokens: object | None,
    input_batch: object | None = None,
) -> PackedRowPlanInputs:
    """Adapt the MRv2 array scheduler columns without guessing a layout."""

    batch = _packed_batch(runner=runner, input_batch=input_batch)
    if batch is None:
        return PackedRowPlanInputs((), (), (), ())
    request_ids = _active_request_ids(batch=batch, family="MRv2")
    if not request_ids:
        return PackedRowPlanInputs((), (), (), ())
    prompt_tokens = getattr(batch, "prefill_len_np", None)
    if prompt_tokens is None:
        raise RuntimeError("MRv2 packed batch requires prefill_len_np")
    computed_tokens = getattr(batch, "num_computed_tokens_np", None)
    if computed_tokens is None:
        raise RuntimeError("MRv2 packed batch requires num_computed_tokens_np")
    return _packed_row_plan_inputs_from_columns(
        request_ids=request_ids,
        num_scheduled_tokens=num_scheduled_tokens,
        scheduled_field="MRv2 num_scheduled_tokens",
        prompt_tokens=prompt_tokens,
        prompt_field="prefill_len_np",
        computed_tokens=computed_tokens,
        computed_field="num_computed_tokens_np",
    )


def resolve_module_weight_view(
    *,
    binding: RunnerBinding,
    locator: ModuleWeightViewLocator,
) -> ResolvedModuleWeightView:
    if not binding.active:
        raise RuntimeError("cannot resolve ModuleWeightView on an inactive runner binding")
    if not isinstance(locator, ModuleWeightViewLocator):
        raise TypeError("locator must be ModuleWeightViewLocator")
    overlay = binding.capabilities.module_weight_overlay
    if overlay is None or locator.module not in overlay.modules:
        raise RuntimeError(
            f"backend {binding.capabilities.backend_id!r} does not support "
            f"ModuleWeightView module {locator.module!r}"
        )
    model = getattr(binding.runner, "model", None)
    if model is None:
        raise RuntimeError("ModuleWeightView backend requires runner.model")
    layers = _model_layers(model)
    raw_layer = int(locator.layer)
    layer = raw_layer if raw_layer >= 0 else len(layers) + raw_layer
    if layer < 0 or layer >= len(layers):
        raise RuntimeError(f"ModuleWeightView layer index out of range: {raw_layer}")
    module = getattr(layers[layer], locator.module, None)
    if module is None:
        raise RuntimeError(
            f"ModuleWeightView target model.layers.{layer}.{locator.module} does not exist"
        )
    return ResolvedModuleWeightView(
        module=module,
        layer=layer,
        module_name=locator.module,
        path=f"model.layers.{layer}.{locator.module}",
    )


def _handle_key(target: ResolvedModuleWeightView, locator: ModuleWeightViewLocator) -> str:
    return f"{target.path}:{locator.phase}:wrapper"


def install_module_weight_overlay(
    *,
    binding: RunnerBinding,
    locator: ModuleWeightViewLocator,
    installer: Callable[[object], Any],
) -> Any:
    if not callable(installer):
        raise TypeError("installer must be callable")
    target = resolve_module_weight_view(binding=binding, locator=locator)
    key = _handle_key(target, locator)
    existing = binding.module_weight_overlay_handles.get(key)
    if existing is not None:
        return existing
    original_forward = getattr(target.module, "forward", None)
    if not callable(original_forward):
        raise RuntimeError(f"ModuleWeightView target {target.path} must expose callable forward")
    try:
        handle = installer(target.module)
        if not callable(getattr(handle, "restore", None)):
            raise TypeError("module weight overlay installer must return a restorable handle")
    except Exception:
        setattr(target.module, "forward", original_forward)
        raise
    binding.module_weight_overlay_handles[key] = handle
    return handle


def restore_module_weight_overlays(binding: RunnerBinding) -> int:
    restored = 0
    for key, handle in list(binding.module_weight_overlay_handles.items()):
        try:
            handle.restore()
        finally:
            binding.module_weight_overlay_handles.pop(key, None)
        restored += 1
    return restored


def unbind_runner(binding: RunnerBinding) -> None:
    if not binding.active:
        return
    restore_module_weight_overlays(binding)
    binding.active = False
    if getattr(binding.runner, _BINDING_ATTR, None) is binding:
        delattr(binding.runner, _BINDING_ATTR)


__all__ = [
    "PackedRowPlanInputs",
    "ensure_runner_binding",
    "get_runner_binding",
    "install_module_weight_overlay",
    "mrv1_packed_row_plan_inputs",
    "mrv2_packed_row_plan_inputs",
    "resolve_module_weight_view",
    "restore_module_weight_overlays",
    "runner_active_request_count",
    "runner_model_identity",
    "runner_model_layer_count",
    "runner_tensor_parallel_size",
    "unbind_runner",
]
