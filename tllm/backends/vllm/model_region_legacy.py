#!/usr/bin/env python3
"""vLLM eager model-region replay driven by compiled recurrence schedules.

This module is the quarantine boundary for vLLM model objects, forward
contexts, attention metadata, and physical KV slots. Consumers provide only
their non-model recurrence operations; the backend interprets model-layer ops
and the Rust-selected schedule.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import os
from typing import Any, cast

import torch
import tllm._core as _core

from tllm.backends.protocol import (
    ORDERED_ROW_FUSION_POLICY_PREFIX,
    GraphSafety,
    RunnerBinding,
    ordered_row_fusion_policy_fingerprint,
)
from tllm.backends.vllm import module_weight_overlay as _overlay
from tllm.contracts.recirculation import (
    RECIRCULATION_PREFILL_BINDER_IMPLEMENTATION_ID,
    RecirculationPrefillBindingConfig,
)
from tllm.runtime.recirculation import mix_recirculation_tensors
from tllm.workgraph.recurrence import (
    CompiledRecurrence,
    DependenceKind,
    RecurrenceOp,
)

_HANDLE_KEY = "compiled-recurrence"


@dataclass(frozen=True, slots=True)
class PreparedRegionRequest:
    request_id: str
    request_ordinal: int
    packed_rows: tuple[int, ...]
    token_positions: tuple[int, ...]
    compiled: CompiledRecurrence
    adapter: object


@dataclass(frozen=True, slots=True)
class NativePreparedRegionRequest:
    """One native request bound to Rust's closed recurrence descriptor."""

    request_id: str
    request_ordinal: int
    packed_rows: tuple[int, ...]
    token_positions: tuple[int, ...]
    schedule: _core.RecirculationPrefillSchedule
    config: RecirculationPrefillBindingConfig


@dataclass(frozen=True, slots=True)
class PreparedRegionStep:
    total_rows: int
    requests: tuple[PreparedRegionRequest | NativePreparedRegionRequest, ...]
    replay_request: (
        Callable[[object, object], dict[int, torch.Tensor]]
        | None
    ) = field(default=None, compare=False, repr=False)


@dataclass(slots=True)
class ModelRegionStats:
    request_count: int = 0
    normal_row_count: int = 0
    recurrent_row_count: int = 0
    fused_group_count: int = 0
    replay_count: int = 0
    request_replays: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _LayerState:
    hidden: torch.Tensor
    residual: torch.Tensor | None

    def stream(self) -> torch.Tensor:
        return self.hidden if self.residual is None else self.hidden + self.residual


def _active_recurrence_targets(runtime: object) -> tuple[Any, ...]:
    plan = getattr(runtime, "dispatch_plan", None)
    if plan is None:
        return ()
    return tuple(plan.recurrence_targets())


def _resolve_qwen_like_model(runner: object) -> tuple[object, object, Any, object]:
    outer = getattr(runner, "model", None)
    inner = getattr(outer, "model", None)
    layers = getattr(inner, "layers", None)
    norm = getattr(inner, "norm", None)
    if outer is None or inner is None or layers is None or norm is None:
        raise RuntimeError(
            "vLLM model-region replay currently requires a Qwen-like "
            "model.model.layers plus model.model.norm structure"
        )
    if not hasattr(layers, "__len__") or not hasattr(layers, "__getitem__"):
        raise RuntimeError("vLLM model-region layers must be indexable")
    if len(layers) < 1 or not callable(getattr(norm, "forward", None)):
        raise RuntimeError("vLLM model-region requires decoder layers and a final norm")
    for index, layer in enumerate(layers):
        if not callable(getattr(layer, "forward", None)):
            raise RuntimeError(f"vLLM model-region layer {index} has no callable forward")
        attention = getattr(getattr(layer, "self_attn", None), "attn", None)
        if attention is None:
            raise RuntimeError(
                "vLLM model-region currently supports decoder layers exposing "
                f"self_attn.attn; layer {index} does not"
            )
    return outer, inner, layers, norm


def _model_layer_index(op: RecurrenceOp, layer_count: int) -> int | None:
    prefix = "model_layer:"
    if not op.executor.startswith(prefix):
        return None
    try:
        layer = int(op.executor[len(prefix) :])
    except ValueError as exc:
        raise RuntimeError(f"invalid model-layer executor {op.executor!r}") from exc
    if layer != op.point.layer or layer < 0 or layer >= layer_count:
        raise RuntimeError(
            f"recurrence op {op.op_id!r} has inconsistent model layer {layer}"
        )
    return layer


def _metadata_replace_kwargs(
    metadata: object,
    *,
    request_ordinal: int,
    packed_rows: tuple[int, ...],
    token_positions: tuple[int, ...],
) -> dict[str, object]:
    first_position = token_positions[0]
    if any(
        position != first_position + index
        for index, position in enumerate(token_positions)
    ):
        raise RuntimeError(
            "ordered model-region rows must be consecutive token positions"
        )
    rows = list(packed_rows)
    slot_mapping = getattr(metadata, "slot_mapping", None)
    block_table = getattr(metadata, "block_table", None)
    query_start_loc = getattr(metadata, "query_start_loc", None)
    seq_lens = getattr(metadata, "seq_lens", None)
    if not all(isinstance(value, torch.Tensor) for value in (
        slot_mapping,
        block_table,
        query_start_loc,
        seq_lens,
    )):
        raise RuntimeError(
            "vLLM model-region requires tensor slot_mapping, block_table, "
            "query_start_loc, and seq_lens metadata"
        )
    assert isinstance(slot_mapping, torch.Tensor)
    assert isinstance(block_table, torch.Tensor)
    assert isinstance(query_start_loc, torch.Tensor)
    assert isinstance(seq_lens, torch.Tensor)
    if request_ordinal < 0 or request_ordinal >= int(block_table.shape[0]):
        raise RuntimeError(
            f"model-region request ordinal {request_ordinal} is outside the block table"
        )
    kwargs: dict[str, object] = {
        "num_actual_tokens": len(rows),
        "max_query_len": len(rows),
        "query_start_loc": query_start_loc.new_tensor([0, len(rows)]),
        "max_seq_len": token_positions[-1] + 1,
        "seq_lens": seq_lens.new_tensor([token_positions[-1] + 1]),
        "block_table": block_table[request_ordinal : request_ordinal + 1],
        "slot_mapping": slot_mapping[rows],
    }
    optional = {
        "use_cascade": False,
        "common_prefix_len": 0,
        "cu_prefix_query_lens": None,
        "prefix_kv_lens": None,
        "suffix_kv_lens": None,
        "scheduler_metadata": None,
        "prefix_scheduler_metadata": None,
        "max_num_splits": 0,
        # These fields describe DCP context rows, not ordinary causal prefill.
        "num_decode_reqs": 0,
        "num_prefill_reqs": 0,
        "num_decode_tokens": 0,
        "num_prefill_tokens": 0,
    }
    kwargs.update(
        (name, value) for name, value in optional.items() if hasattr(metadata, name)
    )
    return kwargs


def _slice_forward_context(
    context: object,
    *,
    request_ordinal: int,
    packed_rows: tuple[int, ...],
    token_positions: tuple[int, ...],
) -> _core.RecirculationPrefillPlan:
    attention = getattr(context, "attn_metadata", None)
    slots = getattr(context, "slot_mapping", None)
    if not isinstance(attention, dict) or not isinstance(slots, dict):
        raise RuntimeError(
            "vLLM model-region does not support DBO/list forward metadata"
        )
    sliced_attention = {
        name: replace(
            metadata,
            **_metadata_replace_kwargs(
                metadata,
                request_ordinal=request_ordinal,
                packed_rows=packed_rows,
                token_positions=token_positions,
            ),
        )
        for name, metadata in attention.items()
    }
    rows = list(packed_rows)
    sliced_slots = {name: value[rows] for name, value in slots.items()}
    batch_descriptor = getattr(context, "batch_descriptor", None)
    if batch_descriptor is not None:
        batch_descriptor = replace(batch_descriptor, num_tokens=len(rows))
    return replace(
        context,
        attn_metadata=sliced_attention,
        slot_mapping=sliced_slots,
        batch_descriptor=batch_descriptor,
        is_padding=None,
    )


class VllmModelRegionHandle:
    """Restorable instance-scoped interpreter for one Qwen-like runner."""

    def __init__(self, *, binding: RunnerBinding, runtime: object) -> None:
        self.binding = binding
        self.runner = binding.runner
        self.runtime = runtime
        _outer, self.inner_model, self.layers, self.norm = _resolve_qwen_like_model(
            self.runner
        )
        self.original_model_forward = self.inner_model.forward
        self.original_first_layer_forward = self.layers[0].forward
        self.prepared: PreparedRegionStep | None = None
        self.seed_hidden: torch.Tensor | None = None
        self.seed_residual: torch.Tensor | None = None
        self.seed_positions: torch.Tensor | None = None
        self.replaying = False
        self.restored = False
        self.stats = ModelRegionStats()
        try:
            self.layers[0].forward = self._first_layer_forward
            self.inner_model.forward = self._model_forward
        except Exception:
            self.layers[0].forward = self.original_first_layer_forward
            self.inner_model.forward = self.original_model_forward
            raise

    def prepare(self, step: PreparedRegionStep) -> None:
        if self.restored:
            raise RuntimeError("cannot prepare a restored model-region handle")
        if step.replay_request is None:
            step = replace(step, replay_request=self._execute_legacy_request)
        self.prepared = step

    def abort(self) -> None:
        """Drop all step-local state without uninstalling the reusable hook."""

        self.prepared = None
        self.seed_hidden = None
        self.seed_residual = None
        self.seed_positions = None
        self.replaying = False

    @staticmethod
    def _layer_arguments(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> tuple:
        positions = args[0] if len(args) > 0 else kwargs.get("positions")
        hidden = args[1] if len(args) > 1 else kwargs.get("hidden_states")
        residual = args[2] if len(args) > 2 else kwargs.get("residual")
        if not isinstance(positions, torch.Tensor) or not isinstance(hidden, torch.Tensor):
            raise RuntimeError(
                "vLLM model-region requires layer.forward(positions, hidden_states, residual)"
            )
        if residual is not None and not isinstance(residual, torch.Tensor):
            raise RuntimeError("vLLM model-region residual must be a tensor or None")
        return positions, hidden, residual

    def _first_layer_forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.prepared is not None and not self.replaying:
            positions, hidden, residual = self._layer_arguments(args, kwargs)
            if hidden.ndim < 2 or int(hidden.shape[0]) != self.prepared.total_rows:
                raise RuntimeError(
                    "model-region packed row mismatch before model execution: "
                    f"model={int(hidden.shape[0])} planned={self.prepared.total_rows}"
                )
            self.seed_positions = positions
            self.seed_hidden = hidden
            self.seed_residual = residual
        return self.original_first_layer_forward(*args, **kwargs)

    def _input_state(
        self,
        *,
        op: RecurrenceOp,
        incoming_data: Mapping[str, tuple[str, ...]],
        states: Mapping[str, _LayerState],
        by_id: Mapping[str, RecurrenceOp],
        request: PreparedRegionRequest,
    ) -> _LayerState:
        sources = incoming_data.get(op.op_id, ())
        available = [source for source in sources if source in states]
        if available:
            if len(available) != 1:
                raise RuntimeError(
                    f"model recurrence op {op.op_id!r} requires exactly one data input"
                )
            return states[available[0]]
        if op.point.pass_name != "normal" or op.point.layer != 0:
            missing = ", ".join(sources) or "<none>"
            raise RuntimeError(
                f"model recurrence op {op.op_id!r} has no materialized input; "
                f"declared data predecessors: {missing}"
            )
        if self.seed_hidden is None:
            raise RuntimeError("model-region did not capture the model input")
        token = op.point.token
        row = request.packed_rows[token]
        residual = (
            None
            if self.seed_residual is None
            else self.seed_residual[row : row + 1]
        )
        return _LayerState(self.seed_hidden[row : row + 1], residual)

    @staticmethod
    def _batch_states(inputs: tuple[_LayerState, ...]) -> _LayerState:
        hidden = torch.cat(tuple(state.hidden for state in inputs), dim=0)
        if all(state.residual is None for state in inputs):
            return _LayerState(hidden, None)
        residual = torch.cat(
            tuple(
                torch.zeros_like(state.hidden)
                if state.residual is None
                else state.residual
                for state in inputs
            ),
            dim=0,
        )
        return _LayerState(hidden, residual)

    def _execute_model_group(
        self,
        *,
        ops: tuple[RecurrenceOp, ...],
        layer_index: int,
        request: PreparedRegionRequest,
        context: object,
        incoming_data: Mapping[str, tuple[str, ...]],
        states: dict[str, _LayerState],
        values: dict[str, torch.Tensor],
        normal_outputs: dict[int, torch.Tensor],
        by_id: Mapping[str, RecurrenceOp],
    ) -> None:
        inputs = tuple(
            self._input_state(
                op=op,
                incoming_data=incoming_data,
                states=states,
                by_id=by_id,
                request=request,
            )
            for op in ops
        )
        rows = tuple(request.packed_rows[op.point.token] for op in ops)
        positions = tuple(request.token_positions[op.point.token] for op in ops)
        if tuple(sorted(positions)) != positions:
            raise RuntimeError("model-region fused rows must be in causal order")
        micro_context = _slice_forward_context(
            context,
            request_ordinal=request.request_ordinal,
            packed_rows=rows,
            token_positions=positions,
        )
        position_tensor = self.seed_positions
        if position_tensor is None or position_tensor.ndim != 1:
            raise RuntimeError("vLLM model-region currently requires 1-D positions")
        model_input = self._batch_states(inputs)
        from vllm.forward_context import override_forward_context

        with override_forward_context(micro_context):
            output = self.layers[layer_index](
                position_tensor[list(rows)],
                model_input.hidden,
                model_input.residual,
            )
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(
                "vLLM model-region requires decoder layers returning "
                "(hidden_states, residual)"
            )
        hidden, residual = output
        if not isinstance(hidden, torch.Tensor) or not isinstance(residual, torch.Tensor):
            raise RuntimeError("vLLM model-region layer outputs must be tensors")
        for index, op in enumerate(ops):
            state = _LayerState(
                hidden[index : index + 1], residual[index : index + 1]
            )
            states[op.op_id] = state
            values[op.op_id] = state.stream()
            if op.point.pass_name == "normal":
                self.stats.normal_row_count += 1
                if layer_index == len(self.layers) - 1:
                    normalized = self.norm(state.hidden, state.residual)
                    if not isinstance(normalized, tuple) or not isinstance(
                        normalized[0], torch.Tensor
                    ):
                        raise RuntimeError(
                            "vLLM model-region final norm must return (hidden, residual)"
                        )
                    normal_outputs[op.point.token] = normalized[0]
            elif op.point.pass_name == "recurrent":
                self.stats.recurrent_row_count += 1
        if len(ops) > 1:
            self.stats.fused_group_count += 1

    def _execute_non_model_op(
        self,
        *,
        op: RecurrenceOp,
        request: PreparedRegionRequest,
        incoming_data: Mapping[str, tuple[str, ...]],
        values: dict[str, torch.Tensor],
        states: dict[str, _LayerState],
        by_id: Mapping[str, RecurrenceOp],
    ) -> None:
        sources = incoming_data.get(op.op_id, ())
        predecessors = tuple(
            (by_id[source], values[source]) for source in sources if source in values
        )
        if len(predecessors) != len(sources):
            raise RuntimeError(
                f"recurrence op {op.op_id!r} ran before all data inputs were materialized"
            )
        execute = getattr(request.adapter, "execute", None)
        if not callable(execute):
            raise RuntimeError("recurrence runtime adapter must expose execute()")
        token = op.point.token
        position = request.token_positions[token]
        position_tensor = self.seed_positions
        assert position_tensor is not None
        result = execute(
            op,
            predecessors,
            position_tensor.new_tensor([position]),
        )
        if not isinstance(result, torch.Tensor) or result.ndim < 2 or result.shape[0] != 1:
            raise RuntimeError(
                f"recurrence executor {op.executor!r} must return one hidden row"
            )
        values[op.op_id] = result
        states[op.op_id] = _LayerState(result, None)

    def _execute_legacy_request(
        self,
        request: PreparedRegionRequest,
        context: object,
    ) -> dict[int, torch.Tensor]:
        graph = request.compiled.graph
        by_id = {op.op_id: op for op in graph.ops}
        incoming_data: dict[str, list[str]] = {}
        for edge in graph.edges:
            if edge.kind is DependenceKind.DATA:
                incoming_data.setdefault(edge.target, []).append(edge.source)
        frozen_incoming = {
            op_id: tuple(sources) for op_id, sources in incoming_data.items()
        }
        states: dict[str, _LayerState] = {}
        values: dict[str, torch.Tensor] = {}
        normal_outputs: dict[int, torch.Tensor] = {}
        for group_ids in request.compiled.schedule.groups:
            ops = tuple(by_id[op_id] for op_id in group_ids)
            layer_indices = tuple(
                _model_layer_index(op, len(self.layers)) for op in ops
            )
            if all(layer is not None for layer in layer_indices):
                if len(set(layer_indices)) != 1:
                    raise RuntimeError(
                        "compiled model-region group contains different model layers"
                    )
                self._execute_model_group(
                    ops=ops,
                    layer_index=int(layer_indices[0]),
                    request=request,
                    context=context,
                    incoming_data=frozen_incoming,
                    states=states,
                    values=values,
                    normal_outputs=normal_outputs,
                    by_id=by_id,
                )
                continue
            if len(ops) != 1 or layer_indices[0] is not None:
                raise RuntimeError(
                    "compiled model-region can only fuse model-layer ops"
                )
            self._execute_non_model_op(
                op=ops[0],
                request=request,
                incoming_data=frozen_incoming,
                values=values,
                states=states,
                by_id=by_id,
            )
        if set(normal_outputs) != set(range(len(request.packed_rows))):
            raise RuntimeError("compiled recurrence did not produce every normal logits row")
        return normal_outputs

    def _native_input_state(
        self,
        *,
        operation: _core.RecirculationPrefillOp,
        request: NativePreparedRegionRequest,
        states: Mapping[int, _LayerState],
    ) -> _LayerState:
        if operation.model_input_kind == _core.RecirculationInputKind.Ordinal:
            raw_ordinal = operation.model_input_ordinal
            if raw_ordinal is None:
                raise RuntimeError("Rust Recirculation ordinal input is missing its ordinal")
            state = states.get(int(raw_ordinal))
            if state is None:
                raise RuntimeError(
                    "Rust Recirculation schedule executed before its data input"
                )
            return state
        if operation.model_input_kind != _core.RecirculationInputKind.Seed:
            raise RuntimeError("Rust Recirculation model operation has no typed input")
        if self.seed_hidden is None:
            raise RuntimeError("model-region did not capture the model input")
        row = request.packed_rows[int(operation.token)]
        residual = (
            None
            if self.seed_residual is None
            else self.seed_residual[row : row + 1]
        )
        return _LayerState(self.seed_hidden[row : row + 1], residual)

    def _execute_native_model_group(
        self,
        *,
        group: _core.RecirculationPrefillGroup,
        request: NativePreparedRegionRequest,
        context: object,
        states: dict[int, _LayerState],
        values: dict[int, torch.Tensor],
    ) -> None:
        raw_layer = group.model_layer
        if raw_layer is None:
            raise RuntimeError("Rust model-layer group does not name its layer")
        layer_index = int(raw_layer)
        operations = tuple(group.operations)
        inputs = tuple(
            self._native_input_state(
                operation=operation,
                request=request,
                states=states,
            )
            for operation in operations
        )
        rows = tuple(
            request.packed_rows[int(operation.token)] for operation in operations
        )
        positions = tuple(
            request.token_positions[int(operation.token)] for operation in operations
        )
        micro_context = _slice_forward_context(
            context,
            request_ordinal=request.request_ordinal,
            packed_rows=rows,
            token_positions=positions,
        )
        position_tensor = self.seed_positions
        if position_tensor is None or position_tensor.ndim != 1:
            raise RuntimeError("vLLM model-region currently requires 1-D positions")
        model_input = self._batch_states(inputs)
        from vllm.forward_context import override_forward_context

        with override_forward_context(micro_context):
            output = self.layers[layer_index](
                position_tensor[list(rows)],
                model_input.hidden,
                model_input.residual,
            )
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(
                "vLLM model-region requires decoder layers returning "
                "(hidden_states, residual)"
            )
        hidden, residual = output
        if not isinstance(hidden, torch.Tensor) or not isinstance(residual, torch.Tensor):
            raise RuntimeError("vLLM model-region layer outputs must be tensors")
        for index, operation in enumerate(operations):
            state = _LayerState(
                hidden[index : index + 1], residual[index : index + 1]
            )
            ordinal = int(operation.ordinal)
            states[ordinal] = state
            values[ordinal] = state.stream()
            if operation.kind == _core.RecirculationPrefillOpKind.Normal:
                self.stats.normal_row_count += 1
            elif operation.kind == _core.RecirculationPrefillOpKind.Recurrent:
                self.stats.recurrent_row_count += 1
        if len(operations) > 1:
            self.stats.fused_group_count += 1

    def _execute_native_mix(
        self,
        *,
        operation: _core.RecirculationPrefillOp,
        request: NativePreparedRegionRequest,
        states: dict[int, _LayerState],
        values: dict[int, torch.Tensor],
    ) -> None:
        source_ordinal = operation.mix_source_ordinal
        destination_ordinal = operation.mix_destination_ordinal
        if source_ordinal is None or destination_ordinal is None:
            raise RuntimeError("Rust Recirculation mix operation has no typed inputs")
        source = values.get(int(source_ordinal))
        destination = values.get(int(destination_ordinal))
        if source is None or destination is None:
            raise RuntimeError("Rust Recirculation mix inputs are not materialized")
        position_tensor = self.seed_positions
        if position_tensor is None:
            raise RuntimeError("model-region did not capture model positions")
        position = request.token_positions[int(operation.token)]
        result = mix_recirculation_tensors(
            request.config,
            source,
            destination,
            position_tensor.new_tensor([position]),
        )
        if result.ndim < 2 or int(result.shape[0]) != 1:
            raise RuntimeError("native Recirculation mix must return one hidden row")
        ordinal = int(operation.ordinal)
        values[ordinal] = result
        states[ordinal] = _LayerState(result, None)

    def _execute_native_request(
        self,
        request: NativePreparedRegionRequest,
        context: object,
    ) -> dict[int, torch.Tensor]:
        # This callable is installed only by native preparation. The typed
        # schedule and its operation tags were validated in Rust.
        if int(request.schedule.token_count) != len(request.packed_rows):
            raise RuntimeError("native Recirculation schedule does not match packed rows")
        states: dict[int, _LayerState] = {}
        values: dict[int, torch.Tensor] = {}
        for group in request.schedule.groups:
            if group.kind == _core.RecirculationPrefillGroupKind.Mix:
                operations = tuple(group.operations)
                self._execute_native_mix(
                    operation=operations[0],
                    request=request,
                    states=states,
                    values=values,
                )
            elif group.kind == _core.RecirculationPrefillGroupKind.ModelLayer:
                self._execute_native_model_group(
                    group=group,
                    request=request,
                    context=context,
                    states=states,
                    values=values,
                )
            else:
                raise RuntimeError("Rust Recirculation group has an unknown primitive")
            for raw_ordinal in group.release_after:
                ordinal = int(raw_ordinal)
                states.pop(ordinal, None)
                values.pop(ordinal, None)
        retained_ordinals = {
            int(ordinal) for ordinal in request.schedule.retained_ordinals
        }
        if set(states) != retained_ordinals or set(values) != retained_ordinals:
            raise RuntimeError(
                "Rust Recirculation signed liveness does not match native Region values"
            )
        normal_outputs: dict[int, torch.Tensor] = {}
        for token, raw_ordinal in enumerate(request.schedule.final_normal_ordinals):
            state = states.get(int(raw_ordinal))
            if state is None:
                raise RuntimeError("Rust Recirculation final normal output is missing")
            normalized = self.norm(state.hidden, state.residual)
            if not isinstance(normalized, tuple) or not isinstance(
                normalized[0], torch.Tensor
            ):
                raise RuntimeError(
                    "vLLM model-region final norm must return (hidden, residual)"
                )
            normal_outputs[token] = normalized[0]
        if len(normal_outputs) != len(request.packed_rows):
            raise RuntimeError("Rust Recirculation did not produce every logits row")
        return normal_outputs

    def _model_forward(self, *args: Any, **kwargs: Any) -> Any:
        prepared = self.prepared
        try:
            output = self.original_model_forward(*args, **kwargs)
            if prepared is None or not prepared.requests:
                return output
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(
                    "vLLM model-region currently requires a last-PP-rank tensor output"
                )
            if self.seed_hidden is None or self.seed_positions is None:
                raise RuntimeError("model-region failed to capture model inputs")
            if output.ndim < 2 or int(output.shape[0]) != prepared.total_rows:
                raise RuntimeError(
                    "model-region packed row mismatch: "
                    f"model={int(output.shape[0])} planned={prepared.total_rows}"
                )
            from vllm.forward_context import get_forward_context

            context = get_forward_context()
            result = output.clone()
            self.replaying = True
            replay_request = prepared.replay_request
            if replay_request is None:
                raise RuntimeError("model-region step has no bound replay strategy")
            for request in prepared.requests:
                normal_outputs = replay_request(request, context)
                for token, packed_row in enumerate(request.packed_rows):
                    result[packed_row : packed_row + 1] = normal_outputs[token]
                self.stats.request_count += 1
                self.stats.replay_count += 1
                self.stats.request_replays[request.request_id] = (
                    self.stats.request_replays.get(request.request_id, 0) + 1
                )
            return result
        finally:
            self.replaying = False
            self.prepared = None
            self.seed_hidden = None
            self.seed_residual = None
            self.seed_positions = None

    def restore(self) -> None:
        if self.restored:
            return
        self.inner_model.forward = self.original_model_forward
        self.layers[0].forward = self.original_first_layer_forward
        self.prepared = None
        self.restored = True


def install_model_region_executor(
    *,
    binding: RunnerBinding,
    runtime: object,
) -> VllmModelRegionHandle:
    if not binding.active:
        raise RuntimeError("cannot install model-region executor on an inactive binding")
    capability = binding.capabilities.model_region
    if capability is None or not capability.replay or not capability.kv_overwrite:
        raise RuntimeError("backend does not provide model-region replay with KV overwrite")
    existing = binding.model_region_handles.get(_HANDLE_KEY)
    if isinstance(existing, VllmModelRegionHandle) and not existing.restored:
        return existing
    handle = VllmModelRegionHandle(binding=binding, runtime=runtime)
    binding.model_region_handles[_HANDLE_KEY] = handle
    setattr(runtime, "model_region_binding", binding)
    return handle


def restore_model_region_executors(binding: RunnerBinding) -> int:
    restored = 0
    for key, handle in list(binding.model_region_handles.items()):
        try:
            restore = getattr(handle, "restore", None)
            if callable(restore):
                restore()
        finally:
            binding.model_region_handles.pop(key, None)
        restored += 1
    return restored


def restore_runtime_model_region(runtime: object) -> int:
    binding = getattr(runtime, "model_region_binding", None)
    if not isinstance(binding, RunnerBinding):
        return 0
    restored = int(binding.driver.restore_model_region_executors(binding))
    setattr(runtime, "model_region_binding", None)
    return restored


def abort_model_region_step(runtime: object, step_id: int) -> bool:
    """Cancel one failed backend step without carrying its Region state forward."""

    aborted = False
    execution = getattr(runtime, "workload_execution", None)
    if execution is not None:
        for binder in execution.backend_region_binders.values():
            aborted = bool(binder.abort(step_id)) or aborted
    binding = getattr(runtime, "model_region_binding", None)
    if isinstance(binding, RunnerBinding):
        handle = binding.model_region_handles.get(_HANDLE_KEY)
        if isinstance(handle, VllmModelRegionHandle):
            handle.abort()
            aborted = True
    return aborted


def _require_model_region_binding(*, runtime: object, runner: object) -> RunnerBinding:
    binding = _overlay.get_runner_binding(runner)
    if not isinstance(binding, RunnerBinding) or not binding.active:
        raise RuntimeError("active recurrence requires an active backend RunnerBinding")
    capability = binding.capabilities.model_region
    if capability is None:
        raise RuntimeError("active recurrence requires backend model-region support")
    if capability.graph_safety is GraphSafety.EAGER_ONLY:
        from tllm.runtime.graph_policy import GraphMode, resolve_graph_mode

        if resolve_graph_mode(runner) is not GraphMode.EAGER:
            raise RuntimeError("vLLM model-region replay is eager-only")
    cache_config = getattr(runner, "cache_config", None)
    if bool(getattr(cache_config, "enable_prefix_caching", False)):
        raise RuntimeError(
            "vLLM model-region replay does not support prefix caching"
        )
    try:
        from vllm import envs as vllm_envs

        multiprocessing_enabled = bool(vllm_envs.VLLM_ENABLE_V1_MULTIPROCESSING)
    except (AttributeError, ImportError):
        multiprocessing_enabled = os.getenv(
            "VLLM_ENABLE_V1_MULTIPROCESSING", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
    if multiprocessing_enabled:
        raise RuntimeError(
            "vLLM model-region replay is ongoing and currently requires "
            "VLLM_ENABLE_V1_MULTIPROCESSING=0"
        )
    parallel_config = getattr(runner, "parallel_config", None)
    if parallel_config is None:
        parallel_config = getattr(
            getattr(runner, "vllm_config", None), "parallel_config", None
        )
    pipeline_size = int(getattr(parallel_config, "pipeline_parallel_size", 1))
    data_size = int(getattr(parallel_config, "data_parallel_size", 1))
    if pipeline_size != 1 or data_size != 1:
        raise RuntimeError(
            "vLLM model-region replay currently requires PP=1 and DP=1"
        )
    vllm_config = getattr(runner, "vllm_config", None)
    if (
        getattr(runner, "speculator", None) is not None
        or getattr(vllm_config, "speculative_config", None) is not None
    ):
        raise RuntimeError(
            "vLLM model-region replay does not support speculative decoding"
        )
    return binding


def _native_prefill_plan(
    *,
    driver: object,
    runner: object,
    num_scheduled_tokens: object | None,
    input_batch: object | None,
    layer_count: int,
) -> _core.RecirculationPrefillPlan:
    normalized = driver.fast_weight_packed_row_plan_inputs(
        runner=runner,
        num_scheduled_tokens=num_scheduled_tokens,
        input_batch=input_batch,
    )
    try:
        return _core.build_recirculation_prefill_plan(
            list(normalized.request_ids),
            list(normalized.scheduled_tokens),
            list(normalized.prompt_tokens),
            list(normalized.computed_tokens),
            layer_count,
        )
    except TypeError as exc:
        raise RuntimeError(
            "Rust Recirculation prefill plan rejected vLLM request identifiers: "
            f"{exc}"
        ) from exc
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _compile_native_recirculation(
    *,
    config: RecirculationPrefillBindingConfig,
    token_count: int,
    layer_count: int,
    ordered_row_fusion_policy: str,
) -> _core.RecirculationPrefillSchedule:
    try:
        return _core.compile_recirculation_prefill_schedule(
            token_count,
            layer_count,
            config.source_layer,
            config.destination_layer,
            ordered_row_fusion_policy,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _signed_ordered_row_fusion_policy(
    *,
    runtime: object,
    binding: RunnerBinding,
) -> str:
    """Require the worker to match the policy signed into its Region epoch."""
    compilation = getattr(runtime, "workload_compilation", None)
    timeline = getattr(compilation, "timeline", None)
    capabilities = getattr(timeline, "capabilities", ())
    signed = tuple(
        capability[len(ORDERED_ROW_FUSION_POLICY_PREFIX) :]
        for capability in capabilities
        if capability.startswith(ORDERED_ROW_FUSION_POLICY_PREFIX)
    )
    if len(signed) != 1:
        raise RuntimeError(
            "signed native Recirculation Region must contain exactly one "
            "ordered-row fusion policy"
        )
    model_region = binding.capabilities.model_region
    local = ordered_row_fusion_policy_fingerprint(
        None if model_region is None else model_region.ordered_row_fusion
    )
    if signed[0] != local:
        raise RuntimeError(
            "worker ordered-row fusion policy disagrees with the signed "
            f"native Region epoch: signed={signed[0]!r}, local={local!r}"
        )
    return signed[0]


def _bind_native_recirculation_plan(
    *,
    plan: _core.RecirculationPrefillPlan,
    inputs: Mapping[str, object],
    binding: RunnerBinding,
    runner: object,
    runtime: object,
    ordered_row_fusion_policy: str,
) -> PreparedRegionStep | None:
    if int(plan.total_rows) == 0:
        return None
    handle = binding.driver.install_model_region_executor(
        binding=binding,
        runtime=runtime,
    )
    if not isinstance(handle, VllmModelRegionHandle):
        raise RuntimeError("backend returned an invalid model-region handle")
    try:
        layer_count = len(handle.layers)
        if int(plan.layer_count) != layer_count:
            raise RuntimeError(
                "native Recirculation plan layer count does not match the bound model"
            )
        config = cast(RecirculationPrefillBindingConfig, inputs["config"])
        compiled_by_token_count: dict[int, _core.RecirculationPrefillSchedule] = {}
        requests: list[NativePreparedRegionRequest] = []
        for request in plan.requests:
            rows = tuple(int(row) for row in request.packed_row_indices)
            token_count = len(rows)
            schedule = compiled_by_token_count.get(token_count)
            if schedule is None:
                schedule = _compile_native_recirculation(
                    config=config,
                    token_count=token_count,
                    layer_count=layer_count,
                    ordered_row_fusion_policy=ordered_row_fusion_policy,
                )
                compiled_by_token_count[token_count] = schedule
            requests.append(
                NativePreparedRegionRequest(
                    request_id=request.request_id,
                    request_ordinal=int(request.request_ordinal),
                    packed_rows=rows,
                    token_positions=tuple(
                        int(value) for value in request.token_positions
                    ),
                    schedule=schedule,
                    config=config,
                )
            )
        prepared = PreparedRegionStep(
            total_rows=int(plan.total_rows),
            requests=tuple(requests),
            replay_request=handle._execute_native_request,
        )
        handle.prepare(prepared)
        return prepared
    except Exception:
        restore_model_region_executors(binding)
        if getattr(runtime, "model_region_binding", None) is binding:
            setattr(runtime, "model_region_binding", None)
        raise


def _stage_native_recirculation(
    *,
    binder: object,
    runtime: object,
    runner: object,
    num_scheduled_tokens: object | None,
    input_batch: object | None,
    step_id: int,
) -> None:
    binding = _require_model_region_binding(runtime=runtime, runner=runner)
    _outer, _inner, layers, _norm = _resolve_qwen_like_model(runner)
    ordered_row_fusion_policy = _signed_ordered_row_fusion_policy(
        runtime=runtime,
        binding=binding,
    )
    plan = _native_prefill_plan(
        driver=binding.driver,
        runner=runner,
        num_scheduled_tokens=num_scheduled_tokens,
        input_batch=input_batch,
        layer_count=len(layers),
    )
    def bind(
        plan: _core.RecirculationPrefillPlan,
        inputs: Mapping[str, object],
        callback_step_id: int,
    ) -> PreparedRegionStep | None:
        return _bind_native_recirculation_plan(
            plan=plan,
            inputs=inputs,
            binding=binding,
            runner=runner,
            runtime=runtime,
            ordered_row_fusion_policy=ordered_row_fusion_policy,
        )

    binder.stage(plan, bind, step_id)


def _stage_native_recirculation_context(context: object) -> None:
    """Backend-owned adapter for the signed Recirculation binder identity."""

    from tllm.runtime.backend_region_binders import BackendRegionStageContext

    if not isinstance(context, BackendRegionStageContext):
        raise TypeError("vLLM backend Region stager requires BackendRegionStageContext")
    _stage_native_recirculation(
        binder=context.slot,
        runtime=context.runtime,
        runner=context.runner,
        num_scheduled_tokens=context.num_scheduled_tokens,
        input_batch=context.input_batch,
        step_id=context.step_id,
    )


def _signed_backend_region_stager_registry(runtime: object) -> object:
    """Return this worker's backend-owned registry, with explicit test injection."""

    from tllm.runtime.backend_region_binders import BackendRegionStagerRegistry

    registry = getattr(runtime, "backend_region_stager_registry", None)
    if registry is None:
        return BackendRegionStagerRegistry(
            {
                RECIRCULATION_PREFILL_BINDER_IMPLEMENTATION_ID: (
                    _stage_native_recirculation_context
                ),
            }
        )
    if not isinstance(registry, BackendRegionStagerRegistry):
        raise TypeError("backend_region_stager_registry must be a BackendRegionStagerRegistry")
    return registry


def stage_signed_backend_region_event(
    *,
    event: "BackendObservableEvent",
    runtime: object,
    runner: object,
    num_scheduled_tokens: object | None,
    input_batch: object | None = None,
    step_id: object | None = None,
) -> bool:
    """Stage only the backend-owned binders signed for ``event``.

    Drivers call this immediately before dispatching the matching observable
    event.  The stage record, rather than Python Region inspection, selects
    the worker-local adapter.
    """

    from tllm.runtime.backend_region_binders import BackendRegionStageContext
    from tllm.runtime.workload_execution import BackendObservableEvent

    if not isinstance(event, BackendObservableEvent):
        raise TypeError("backend Region staging event must be a BackendObservableEvent")
    execution = getattr(runtime, "workload_execution", None)
    if execution is None:
        return False
    stages = execution.backend_region_binder_stages(event)
    if not stages:
        return False
    registry = _signed_backend_region_stager_registry(runtime)
    exact_step_id = getattr(runtime, "event_step_id", 0) if step_id is None else step_id
    for stage in stages:
        try:
            slot = execution.backend_region_binders[stage.implementation_id]
        except KeyError as exc:
            raise RuntimeError(
                "signed backend Region has no local binder slot for "
                f"{stage.implementation_id!r}"
            ) from exc
        registry.stage(
            BackendRegionStageContext(
                implementation_id=stage.implementation_id,
                backend_event=stage.backend_event.value,
                region_id=stage.region_id,
                slot=slot,
                runtime=runtime,
                runner=runner,
                step_id=exact_step_id,
                num_scheduled_tokens=num_scheduled_tokens,
                input_batch=input_batch,
            )
        )
    return True


def prepare_model_region_step(
    *,
    runtime: object,
    runner: object,
    num_scheduled_tokens: object | None,
    input_batch: object | None = None,
) -> PreparedRegionStep | None:
    # Native Regions are staged at their signed observable event by the
    # backend driver.  This compatibility entry point must not stage them
    # early or recreate a legacy Region alongside the immutable native plan.
    execution = getattr(runtime, "workload_execution", None)
    if execution is not None:
        return None
    targets = _active_recurrence_targets(runtime)
    binding = _overlay.get_runner_binding(runner)
    if not targets:
        if isinstance(binding, RunnerBinding):
            binding.driver.restore_model_region_executors(binding)
        if getattr(runtime, "model_region_binding", None) is binding:
            setattr(runtime, "model_region_binding", None)
        return None
    if len(targets) != 1:
        raise RuntimeError("vLLM model-region currently supports one active recurrence")
    binding = _require_model_region_binding(runtime=runtime, runner=runner)
    handle = binding.driver.install_model_region_executor(
        binding=binding,
        runtime=runtime,
    )
    if not isinstance(handle, VllmModelRegionHandle):
        raise RuntimeError("backend returned an invalid model-region handle")
    normalized = binding.driver.fast_weight_packed_row_plan_inputs(
        runner=runner,
        num_scheduled_tokens=num_scheduled_tokens,
        input_batch=input_batch,
    )
    from tllm.core import plan_packed_prefill_rows

    row_plan = plan_packed_prefill_rows(
        normalized.scheduled_tokens,
        normalized.prompt_tokens,
        normalized.computed_tokens,
        sum(normalized.scheduled_tokens),
    )
    dispatch_plan = getattr(runtime, "dispatch_plan", None)
    assert dispatch_plan is not None
    from tllm.runtime.graph_policy import resolve_graph_mode

    requests: list[PreparedRegionRequest] = []
    target = targets[0]
    layer_count = len(handle.layers)
    for span in row_plan.request_spans:
        prefill_count = int(span.prefill_row_count)
        if prefill_count == 0:
            continue
        if int(span.decode_row_count) != 0:
            raise RuntimeError(
                "model-region recurrence cannot mix prefill and decode rows "
                "for one request in the same engine step"
            )
        row_start = int(span.packed_row_start)
        rows = tuple(range(row_start, row_start + prefill_count))
        positions = tuple(int(row_plan.token_offsets[row]) for row in rows)
        compiled_items = dispatch_plan.compile_recurrences(
            token_count=prefill_count,
            layer_count=layer_count,
            backend_capabilities=binding.capabilities,
            graph_mode=resolve_graph_mode(runner),
            tp_size=_overlay.runner_tensor_parallel_size(runner),
        )
        if len(compiled_items) != 1:
            raise RuntimeError("model-region compilation must produce one recurrence")
        consumer_id, compiled = compiled_items[0]
        adapter = target.consumer.recurrence_runtime_adapter(
            layer_count=layer_count
        )
        if adapter is None:
            raise RuntimeError(
                f"recurrence consumer {consumer_id!r} has no runtime adapter"
            )
        ordinal = int(span.request_ordinal)
        requests.append(
            PreparedRegionRequest(
                request_id=str(normalized.request_ids[ordinal]),
                request_ordinal=ordinal,
                packed_rows=rows,
                token_positions=positions,
                compiled=compiled,
                adapter=adapter,
            )
        )
    prepared = PreparedRegionStep(
        total_rows=int(row_plan.total_rows),
        requests=tuple(requests),
    )
    handle.prepare(prepared)
    return prepared


__all__ = [
    "ModelRegionStats",
    "PreparedRegionRequest",
    "PreparedRegionStep",
    "VllmModelRegionHandle",
    "abort_model_region_step",
    "install_model_region_executor",
    "prepare_model_region_step",
    "stage_signed_backend_region_event",
    "restore_model_region_executors",
    "restore_runtime_model_region",
]
