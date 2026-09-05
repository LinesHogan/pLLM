"""Typed vLLM layer-replay resources for native backend Regions.

This module owns only vLLM model interception and exact layer/norm resources.
It receives an already signed plan invocation from an owning workload and
does not inspect workload identities, operation IDs, or primitive strings.
The shared replay vocabulary (state, step plan, bound calls, primitive keys)
lives in ``tllm.contracts.typed_replay``; consumer-owned binders and this
driver meet only through that contract and the ``BackendDriver`` protocol.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, cast

import torch

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    from vllm.forward_context import ForwardContext

from tllm.backends.protocol import (
    GraphSafety,
    RunnerBinding,
)
from tllm.backends.vllm import runner_adapter as _runner_adapter
from tllm.contracts.typed_replay import (
    LayerReplayState,
    TypedLayerReplayStep,
)


class _ForwardCall(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


class _ForwardModule(Protocol):
    forward: _ForwardCall


class _CallableModule(_ForwardModule, Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


class _LayerCollection(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> _CallableModule: ...

    def __iter__(self) -> Iterator[_CallableModule]: ...


@dataclass(frozen=True, slots=True)
class TypedLayerReplayStats:
    prepare_count: int
    replay_count: int
    replay_row_count: int
    request_replays: tuple[str, ...]


def _resolve_qwen_like_model(
    runner: object,
) -> tuple[_ForwardModule, _LayerCollection, _CallableModule]:
    outer = getattr(runner, "model", None)
    inner = getattr(outer, "model", None)
    layers = getattr(inner, "layers", None)
    norm = getattr(inner, "norm", None)
    if outer is None or inner is None or layers is None or norm is None:
        raise RuntimeError(
            "vLLM typed layer replay requires a Qwen-like "
            "model.model.layers plus model.model.norm structure"
        )
    if not hasattr(layers, "__len__") or not hasattr(layers, "__getitem__"):
        raise RuntimeError("vLLM typed layer replay layers must be indexable")
    typed_layers = cast(_LayerCollection, layers)
    typed_norm = cast(_CallableModule, norm)
    if len(typed_layers) < 1 or not callable(getattr(norm, "forward", None)):
        raise RuntimeError("vLLM typed layer replay requires decoder layers and final norm")
    for index, layer in enumerate(typed_layers):
        if not callable(getattr(layer, "forward", None)):
            raise RuntimeError(f"vLLM typed layer replay layer {index} has no callable forward")
        attention = getattr(getattr(layer, "self_attn", None), "attn", None)
        if attention is None:
            raise RuntimeError(
                "vLLM typed layer replay requires decoder layers exposing "
                f"self_attn.attn; layer {index} does not"
            )
    return cast(_ForwardModule, inner), typed_layers, typed_norm


def _metadata_replace_kwargs(
    metadata: object,
    *,
    request_ordinal: int,
    packed_row: int,
    token_position: int,
) -> dict[str, object]:
    slot_mapping = getattr(metadata, "slot_mapping", None)
    block_table = getattr(metadata, "block_table", None)
    query_start_loc = getattr(metadata, "query_start_loc", None)
    seq_lens = getattr(metadata, "seq_lens", None)
    if not all(
        isinstance(value, torch.Tensor)
        for value in (slot_mapping, block_table, query_start_loc, seq_lens)
    ):
        raise RuntimeError(
            "vLLM typed layer replay requires tensor slot_mapping, block_table, "
            "query_start_loc, and seq_lens metadata"
        )
    assert isinstance(slot_mapping, torch.Tensor)
    assert isinstance(block_table, torch.Tensor)
    assert isinstance(query_start_loc, torch.Tensor)
    assert isinstance(seq_lens, torch.Tensor)
    if request_ordinal < 0 or request_ordinal >= int(block_table.shape[0]):
        raise RuntimeError("typed layer replay request ordinal is outside the block table")
    kwargs: dict[str, object] = {
        "num_actual_tokens": 1,
        "max_query_len": 1,
        "query_start_loc": query_start_loc.new_tensor([0, 1]),
        "max_seq_len": token_position + 1,
        "seq_lens": seq_lens.new_tensor([token_position + 1]),
        "block_table": block_table[request_ordinal : request_ordinal + 1],
        "slot_mapping": slot_mapping[[packed_row]],
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
        "num_decode_reqs": 0,
        "num_prefill_reqs": 0,
        "num_decode_tokens": 0,
        "num_prefill_tokens": 0,
    }
    kwargs.update((name, value) for name, value in optional.items() if hasattr(metadata, name))
    return kwargs


def _slice_forward_context(
    context: object,
    *,
    request_ordinal: int,
    packed_row: int,
    token_position: int,
) -> ForwardContext:
    typed_context = cast("ForwardContext", context)
    attention = typed_context.attn_metadata
    slots = typed_context.slot_mapping
    if not isinstance(attention, dict) or not isinstance(slots, dict):
        raise RuntimeError("vLLM typed layer replay does not support DBO/list metadata")
    typed_attention = cast(dict[str, object], attention)
    if any(type(name) is not str or not name for name in typed_attention):
        raise RuntimeError("vLLM typed layer replay attention names must be strings")
    typed_slots = cast(dict[str, object], slots)
    if any(
        type(name) is not str
        or not name
        or not isinstance(value, torch.Tensor)
        for name, value in typed_slots.items()
    ):
        raise RuntimeError("vLLM typed layer replay slot mappings must be tensors")
    sliced_attention = {
        name: replace(
            cast("DataclassInstance", metadata),
            **_metadata_replace_kwargs(
                metadata,
                request_ordinal=request_ordinal,
                packed_row=packed_row,
                token_position=token_position,
            ),
        )
        for name, metadata in typed_attention.items()
    }
    sliced_slots = {
        name: cast(torch.Tensor, value)[[packed_row]]
        for name, value in typed_slots.items()
    }
    batch_descriptor = typed_context.batch_descriptor
    if batch_descriptor is not None:
        batch_descriptor = replace(batch_descriptor, num_tokens=1)
    return replace(
        typed_context,
        attn_metadata=sliced_attention,
        slot_mapping=sliced_slots,
        batch_descriptor=batch_descriptor,
        is_padding=None,
    )


class VllmTypedLayerReplayHandle:
    """Restorable hook which exposes only exact typed vLLM resources."""

    binding: RunnerBinding
    runner: object
    inner_model: _ForwardModule
    layers: _LayerCollection
    norm: _CallableModule
    original_model_forward: _ForwardCall
    original_first_layer_forward: _ForwardCall
    forward_context: ForwardContext | None

    __slots__ = (
        "binding",
        "runner",
        "inner_model",
        "layers",
        "norm",
        "original_model_forward",
        "original_first_layer_forward",
        "prepared",
        "seed_hidden",
        "seed_residual",
        "seed_positions",
        "forward_context",
        "replaying",
        "restored",
        "_prepare_count",
        "_replay_count",
        "_replay_row_count",
        "_request_replays",
    )

    def __init__(self, *, binding: RunnerBinding) -> None:
        self.binding = binding
        self.runner = binding.runner
        self.inner_model, self.layers, self.norm = _resolve_qwen_like_model(self.runner)
        self.original_model_forward = self.inner_model.forward
        self.original_first_layer_forward = self.layers[0].forward
        self.prepared: TypedLayerReplayStep | None = None
        self.seed_hidden: torch.Tensor | None = None
        self.seed_residual: torch.Tensor | None = None
        self.seed_positions: torch.Tensor | None = None
        self.forward_context = None
        self.replaying = False
        self.restored = False
        self._prepare_count = 0
        self._replay_count = 0
        self._replay_row_count = 0
        self._request_replays: list[str] = []
        try:
            self.layers[0].forward = self._first_layer_forward
            self.inner_model.forward = self._model_forward
        except Exception:
            self.layers[0].forward = self.original_first_layer_forward
            self.inner_model.forward = self.original_model_forward
            raise

    @staticmethod
    def _layer_arguments(
        args: tuple[object, ...], kwargs: Mapping[str, object]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        positions = args[0] if args else kwargs.get("positions")
        hidden = args[1] if len(args) > 1 else kwargs.get("hidden_states")
        residual = args[2] if len(args) > 2 else kwargs.get("residual")
        if not isinstance(positions, torch.Tensor) or not isinstance(hidden, torch.Tensor):
            raise RuntimeError(
                "vLLM typed layer replay requires layer.forward(positions, hidden_states, residual)"
            )
        if residual is not None and not isinstance(residual, torch.Tensor):
            raise RuntimeError("vLLM typed layer replay residual must be tensor or None")
        return positions, hidden, residual

    def _first_layer_forward(self, *args: object, **kwargs: object) -> object:
        if self.prepared is not None and not self.replaying:
            positions, hidden, residual = self._layer_arguments(args, kwargs)
            if hidden.ndim < 2 or int(hidden.shape[0]) != self.prepared.total_rows:
                raise RuntimeError("typed layer replay packed row mismatch before model execution")
            self.seed_positions = positions
            self.seed_hidden = hidden
            self.seed_residual = residual
        return self.original_first_layer_forward(*args, **kwargs)

    def prepare(self, step: TypedLayerReplayStep) -> None:
        if self.restored:
            raise RuntimeError("cannot prepare a restored typed layer replay handle")
        if self.prepared is not None:
            raise RuntimeError("typed layer replay already has a prepared step")
        self.prepared = step
        self._prepare_count += 1

    @property
    def stats(self) -> TypedLayerReplayStats:
        return TypedLayerReplayStats(
            prepare_count=self._prepare_count,
            replay_count=self._replay_count,
            replay_row_count=self._replay_row_count,
            request_replays=tuple(self._request_replays),
        )

    def seed_state(self, packed_row: int) -> LayerReplayState:
        if packed_row < 0:
            raise ValueError("typed layer replay packed row must be a non-negative int")
        if self.seed_hidden is None:
            raise RuntimeError("typed layer replay did not capture model input")
        residual = (
            None
            if self.seed_residual is None
            else self.seed_residual[packed_row : packed_row + 1]
        )
        return LayerReplayState(self.seed_hidden[packed_row : packed_row + 1], residual)

    def replay_layer(
        self,
        state: LayerReplayState,
        *,
        layer: int,
        request_ordinal: int,
        packed_row: int,
        token_position: int,
    ) -> LayerReplayState:
        if layer < 0 or layer >= len(self.layers):
            raise RuntimeError("typed vLLM layer replay layer is outside the bound model")
        positions = self.seed_positions
        context = self.forward_context
        if positions is None or positions.ndim != 1 or context is None:
            raise RuntimeError("typed vLLM layer replay is outside an active model forward")
        micro_context = _slice_forward_context(
            context,
            request_ordinal=request_ordinal,
            packed_row=packed_row,
            token_position=token_position,
        )
        from vllm.forward_context import override_forward_context

        with override_forward_context(micro_context):
            output = self.layers[layer](
                positions[[packed_row]],
                state.hidden,
                state.residual,
            )
        if not isinstance(output, tuple):
            raise RuntimeError("vLLM typed layer replay requires (hidden_states, residual)")
        typed_output = cast(tuple[object, ...], output)
        if len(typed_output) != 2:
            raise RuntimeError("vLLM typed layer replay requires (hidden_states, residual)")
        hidden, residual = typed_output
        if not isinstance(hidden, torch.Tensor) or not isinstance(residual, torch.Tensor):
            raise RuntimeError("vLLM typed layer replay layer outputs must be tensors")
        return LayerReplayState(hidden, residual)

    def normalize(self, state: LayerReplayState) -> torch.Tensor:
        normalized = self.norm(state.hidden, state.residual)
        if not isinstance(normalized, tuple) or not isinstance(normalized[0], torch.Tensor):
            raise RuntimeError("vLLM typed layer replay final norm must return (hidden, residual)")
        return normalized[0]

    def _model_forward(self, *args: object, **kwargs: object) -> object:
        prepared = self.prepared
        try:
            output = self.original_model_forward(*args, **kwargs)
            if prepared is None:
                return output
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("vLLM typed layer replay requires a tensor model output")
            if self.seed_hidden is None or self.seed_positions is None:
                raise RuntimeError("typed layer replay failed to capture model inputs")
            if output.ndim < 2 or int(output.shape[0]) != prepared.total_rows:
                raise RuntimeError("typed layer replay packed row mismatch after model execution")
            from vllm.forward_context import get_forward_context

            self.forward_context = get_forward_context()
            self.replaying = True
            result = output.clone()
            for request in prepared.requests:
                replacements = request.replay(self)
                expected = set(range(len(request.packed_rows)))
                if set(replacements) != expected:
                    raise RuntimeError("typed layer replay callback did not return every request row")
                for token, packed_row in enumerate(request.packed_rows):
                    value = replacements[token]
                    if value.shape[0] != 1:
                        raise RuntimeError("typed layer replay callback must return one tensor row")
                    result[packed_row : packed_row + 1] = value
                self._replay_count += 1
                self._replay_row_count += len(request.packed_rows)
                self._request_replays.append(request.request_id)
            return result
        finally:
            self.replaying = False
            self.prepared = None
            self.seed_hidden = None
            self.seed_residual = None
            self.seed_positions = None
            self.forward_context = None

    def abort(self) -> None:
        self.prepared = None
        self.seed_hidden = None
        self.seed_residual = None
        self.seed_positions = None
        self.forward_context = None
        self.replaying = False

    def restore(self) -> None:
        if self.restored:
            return
        self.inner_model.forward = self.original_model_forward
        self.layers[0].forward = self.original_first_layer_forward
        self.abort()
        self.restored = True


def _admit_typed_layer_replay(*, runtime: object, runner: object) -> RunnerBinding:
    binding = _runner_adapter.get_runner_binding(runner)
    if not isinstance(binding, RunnerBinding) or not binding.active:
        raise RuntimeError("typed vLLM layer replay requires an active RunnerBinding")
    capability = binding.capabilities.model_region
    if capability is None or not capability.replay or not capability.kv_overwrite:
        raise RuntimeError("backend does not provide typed model-region replay with KV overwrite")
    if capability.graph_safety is GraphSafety.EAGER_ONLY:
        from tllm.backends.vllm.graph_mode import GraphMode, resolve_graph_mode

        if resolve_graph_mode(runner) is not GraphMode.EAGER:
            raise RuntimeError("vLLM typed layer replay is eager-only")
    if bool(getattr(getattr(runner, "cache_config", None), "enable_prefix_caching", False)):
        raise RuntimeError("vLLM typed layer replay does not support prefix caching")
    parallel_config = getattr(runner, "parallel_config", None)
    if parallel_config is None:
        parallel_config = getattr(getattr(runner, "vllm_config", None), "parallel_config", None)
    if parallel_config is None:
        raise RuntimeError("vLLM typed layer replay requires explicit parallel topology")
    tensor_parallel_size = getattr(parallel_config, "tensor_parallel_size", None)
    pipeline_parallel_size = getattr(parallel_config, "pipeline_parallel_size", None)
    data_parallel_size = getattr(parallel_config, "data_parallel_size", None)
    if (
        type(tensor_parallel_size) is not int
        or type(pipeline_parallel_size) is not int
        or type(data_parallel_size) is not int
    ):
        raise RuntimeError("vLLM typed layer replay parallel sizes must be exact integers")
    if (
        tensor_parallel_size != 1
        or pipeline_parallel_size != 1
        or data_parallel_size != 1
    ):
        raise RuntimeError(
            "vLLM typed layer replay currently requires TP=1, PP=1, and DP=1"
        )
    if (
        getattr(runner, "speculator", None) is not None
        or getattr(getattr(runner, "vllm_config", None), "speculative_config", None)
        is not None
    ):
        raise RuntimeError("vLLM typed layer replay does not support speculative decoding")
    return binding


def install_typed_layer_replay(*, runtime: object, runner: object) -> VllmTypedLayerReplayHandle:
    """Return the single restorable typed replay hook for an active runner."""

    binding = _admit_typed_layer_replay(runtime=runtime, runner=runner)
    existing = binding.typed_layer_replay_handle
    if isinstance(existing, VllmTypedLayerReplayHandle) and not existing.restored:
        return existing
    handle = VllmTypedLayerReplayHandle(binding=binding)
    binding.typed_layer_replay_handle = handle
    setattr(runtime, "typed_layer_replay_binding", binding)
    return handle


def abort_typed_layer_replay(runtime: object) -> bool:
    binding = getattr(runtime, "typed_layer_replay_binding", None)
    if not isinstance(binding, RunnerBinding):
        return False
    handle = binding.typed_layer_replay_handle
    if not isinstance(handle, VllmTypedLayerReplayHandle):
        return False
    handle.abort()
    return True


def restore_typed_layer_replay(runtime: object) -> int:
    binding = getattr(runtime, "typed_layer_replay_binding", None)
    if not isinstance(binding, RunnerBinding):
        return 0
    handle = binding.typed_layer_replay_handle
    if not isinstance(handle, VllmTypedLayerReplayHandle):
        return 0
    handle.restore()
    binding.typed_layer_replay_handle = None
    setattr(runtime, "typed_layer_replay_binding", None)
    return 1


__all__ = [
    "TypedLayerReplayStats",
    "VllmTypedLayerReplayHandle",
    "abort_typed_layer_replay",
    "install_typed_layer_replay",
    "restore_typed_layer_replay",
]
