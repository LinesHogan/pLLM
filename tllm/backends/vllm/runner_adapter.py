#!/usr/bin/env python3
"""Shared vLLM runner binding and packed-row normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, SupportsInt

from tllm.backends.protocol import (
    RUNNER_BINDING_ATTR,
    RunnerBinding,
    RunnerBindingDriver,
    runner_device,
    runner_binding_for,
    runner_model_module,
)
from tllm.contracts.packed_rows import PackedRowPlanInputs

_BINDING_ATTR = RUNNER_BINDING_ATTR


def ensure_runner_binding(
    *, driver: RunnerBindingDriver, runner: object
) -> RunnerBinding:
    existing = getattr(runner, _BINDING_ATTR, None)
    if isinstance(existing, RunnerBinding) and existing.driver is driver and existing.active:
        return existing
    capabilities = driver.capabilities()
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
        raise RuntimeError("vLLM runner adapter requires model.layers")
    return layers


def runner_model_identity(runner: object) -> int:
    return id(runner_model_module(runner))


def runner_model_layer_count(runner: object) -> int:
    return len(_model_layers(runner_model_module(runner)))


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


class _IntegerColumn(Protocol):
    def __getitem__(self, index: int) -> SupportsInt: ...


class _ActiveRequestBatch(Protocol):
    @property
    def req_ids(self) -> Sequence[str | None]: ...

    @property
    def num_reqs(self) -> int: ...


class _MRV1Batch(_ActiveRequestBatch, Protocol):
    @property
    def req_id_to_index(self) -> Mapping[str, int]: ...

    @property
    def num_prompt_tokens(self) -> _IntegerColumn: ...

    @property
    def num_computed_tokens_cpu(self) -> _IntegerColumn: ...


class _MRV2Batch(_ActiveRequestBatch, Protocol):
    @property
    def num_scheduled_tokens(self) -> _IntegerColumn: ...

    @property
    def prefill_len_np(self) -> _IntegerColumn: ...

    @property
    def num_computed_tokens_np(self) -> _IntegerColumn: ...


class _MRV1Runner(Protocol):
    @property
    def input_batch(self) -> _MRV1Batch | None: ...


class _MRV2Runner(Protocol):
    @property
    def input_batch(self) -> _MRV2Batch | None: ...


class _MRV1SchedulerOutput(Protocol):
    @property
    def finished_req_ids(self) -> Sequence[str]: ...


def _required_index_int(
    values: _IntegerColumn | None, idx: int, *, field_name: str
) -> int:
    if values is None:
        raise RuntimeError(f"fast-weight metadata requires {field_name}")
    return int(values[idx])


def _active_request_ids(
    *,
    batch: _ActiveRequestBatch,
    family: str,
) -> tuple[str, ...]:
    num_reqs = batch.num_reqs
    request_ids = tuple(batch.req_ids[:num_reqs])
    if len(request_ids) != num_reqs:
        raise RuntimeError(
            f"{family} packed batch does not expose a complete active request prefix"
        )
    if any(request_id is None or not request_id for request_id in request_ids):
        raise RuntimeError(f"{family} packed batch contains an empty active request slot")
    return tuple(
        request_id for request_id in request_ids if request_id is not None
    )


def _packed_row_plan_inputs_from_columns(
    *,
    request_ids: tuple[str, ...],
    num_scheduled_tokens: _IntegerColumn | None,
    scheduled_field: str,
    prompt_tokens: _IntegerColumn | None,
    prompt_field: str,
    computed_tokens: _IntegerColumn | None,
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


def mrv1_packed_row_plan_inputs(
    *,
    runner: _MRV1Runner,
    num_scheduled_tokens: _IntegerColumn | None,
    input_batch: _MRV1Batch | None = None,
) -> PackedRowPlanInputs:
    """Adapt the MRv1 request-indexed scheduler columns without guessing a layout."""

    batch = input_batch if input_batch is not None else runner.input_batch
    if batch is None:
        return PackedRowPlanInputs((), (), (), ())
    request_ids = _active_request_ids(batch=batch, family="MRv1")
    if not request_ids:
        return PackedRowPlanInputs((), (), (), ())
    req_id_to_index = batch.req_id_to_index
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
    prompt_tokens = batch.num_prompt_tokens
    computed_tokens = batch.num_computed_tokens_cpu
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
    runner: _MRV2Runner,
    num_scheduled_tokens: _IntegerColumn | None,
    input_batch: _MRV2Batch | None = None,
) -> PackedRowPlanInputs:
    """Adapt the MRv2 array scheduler columns without guessing a layout."""

    batch = input_batch if input_batch is not None else runner.input_batch
    if batch is None:
        return PackedRowPlanInputs((), (), (), ())
    request_ids = _active_request_ids(batch=batch, family="MRv2")
    if not request_ids:
        return PackedRowPlanInputs((), (), (), ())
    prompt_tokens = batch.prefill_len_np
    computed_tokens = batch.num_computed_tokens_np
    scheduled_tokens = (
        batch.num_scheduled_tokens
        if num_scheduled_tokens is None
        else num_scheduled_tokens
    )
    return _packed_row_plan_inputs_from_columns(
        request_ids=request_ids,
        num_scheduled_tokens=scheduled_tokens,
        scheduled_field="MRv2 num_scheduled_tokens",
        prompt_tokens=prompt_tokens,
        prompt_field="prefill_len_np",
        computed_tokens=computed_tokens,
        computed_field="num_computed_tokens_np",
    )


def mrv1_finished_request_ids(
    scheduler_output: _MRV1SchedulerOutput,
) -> tuple[str, ...]:
    """Read the optional MRv1 terminal-id column without layout guessing."""

    try:
        finished = scheduler_output.finished_req_ids
    except AttributeError:
        return ()
    if any(type(request_id) is not str or not request_id for request_id in finished):
        raise RuntimeError("MRv1 finished_req_ids must contain non-empty strings")
    return tuple(finished)


def mrv1_active_request_ids(runner: _MRV1Runner) -> tuple[str, ...]:
    """Expose the complete MRv1 active-prefix ids for lifecycle admission."""

    batch = runner.input_batch
    return () if batch is None else _active_request_ids(batch=batch, family="MRv1")


def mrv2_active_request_ids(
    runner: _MRV2Runner,
    input_batch: _MRV2Batch | None = None,
) -> tuple[str, ...]:
    """Expose the complete MRv2 active-prefix ids for lifecycle admission."""

    batch = input_batch if input_batch is not None else runner.input_batch
    return () if batch is None else _active_request_ids(batch=batch, family="MRv2")


def unbind_runner(binding: RunnerBinding) -> None:
    if not binding.active:
        return
    binding.active = False
    if getattr(binding.runner, _BINDING_ATTR, None) is binding:
        delattr(binding.runner, _BINDING_ATTR)


__all__ = [
    "PackedRowPlanInputs",
    "ensure_runner_binding",
    "get_runner_binding",
    "mrv1_packed_row_plan_inputs",
    "mrv1_finished_request_ids",
    "mrv1_active_request_ids",
    "mrv2_active_request_ids",
    "mrv2_packed_row_plan_inputs",
    "runner_active_request_count",
    "runner_device",
    "runner_model_identity",
    "runner_model_module",
    "runner_model_layer_count",
    "runner_tensor_parallel_size",
    "unbind_runner",
]
