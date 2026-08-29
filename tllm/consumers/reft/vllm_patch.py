"""LoReFT-specific vLLM integration boundary."""

from __future__ import annotations

from types import MethodType
from dataclasses import dataclass
from collections.abc import Callable
from typing import Protocol, cast

import torch

from tllm.consumers.reft.workload import ReftWorkload
from tllm.runtime.runtime import TllmRuntime


@dataclass(frozen=True, slots=True)
class _ReftPatchSpec:
    layers: tuple[int, ...]
    first_tokens: int
    last_tokens: int


class _AttentionMetadata(Protocol):
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor


class _PublicComputedTokens(Protocol):
    num_computed_tokens_cpu: torch.Tensor


class _PrivateComputedTokens(Protocol):
    _num_computed_tokens_cpu: torch.Tensor


class ReftPatchHandle:
    def __init__(self, layers: list[torch.nn.Module], forwards: list[object]) -> None:
        self._layers = layers
        self._forwards = forwards

    def restore(self) -> None:
        for layer, forward in zip(self._layers, self._forwards, strict=True):
            layer.forward = forward  # type: ignore[method-assign]


def _decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    for path in ("model.model.layers", "model.layers", "transformer.h"):
        value: object = model
        try:
            for part in path.split("."):
                value = getattr(value, part)
        except AttributeError:
            continue
        if isinstance(value, torch.nn.ModuleList):
            return value
    raise RuntimeError("LoReFT could not resolve the decoder layer stack")


def _prefill_metadata() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm.forward_context import get_forward_context

    raw_metadata: object = get_forward_context().attn_metadata
    if isinstance(raw_metadata, dict):
        raw_metadata = next(iter(raw_metadata.values()))
    metadata = cast(_AttentionMetadata, raw_metadata)
    query_start = metadata.query_start_loc
    sequence_lengths = metadata.seq_lens
    if hasattr(raw_metadata, "num_computed_tokens_cpu"):
        computed = cast(_PublicComputedTokens, raw_metadata).num_computed_tokens_cpu
    elif hasattr(raw_metadata, "_num_computed_tokens_cpu"):
        computed = cast(
            _PrivateComputedTokens, raw_metadata
        )._num_computed_tokens_cpu
    else:
        query_lengths = query_start[1:] - query_start[:-1]
        computed = sequence_lengths - query_lengths
    return query_start, sequence_lengths, torch.as_tensor(computed)


def _prompt_rows(
    positions: torch.Tensor,
    query_start: torch.Tensor,
    sequence_lengths: torch.Tensor,
    computed: torch.Tensor,
    *,
    first_tokens: int,
    last_tokens: int,
) -> torch.Tensor:
    request_count = query_start.shape[0] - 1
    query_lengths = query_start[1:] - query_start[:-1]
    request_of_row = torch.repeat_interleave(
        torch.arange(request_count, device=positions.device), query_lengths
    )
    prompt_lengths = sequence_lengths.to(positions.device)[request_of_row]
    prefill = (computed == 0).to(positions.device)[request_of_row]
    selected = (positions < first_tokens) | (
        positions >= prompt_lengths - last_tokens
    )
    return (prefill & selected).nonzero(as_tuple=False).reshape(-1)


def _install_reft_patch(
    model: torch.nn.Module,
    initial_workload: ReftWorkload,
    workload_provider: Callable[[], ReftWorkload],
) -> ReftPatchHandle:
    spec = _ReftPatchSpec(
        layers=initial_workload.config.layers,
        first_tokens=initial_workload.config.first_tokens,
        last_tokens=initial_workload.config.last_tokens,
    )
    layers = _decoder_layers(model)
    wrapped: list[torch.nn.Module] = []
    forwards: list[object] = []
    for layer_slot, layer_index in enumerate(spec.layers):
        layer = layers[layer_index]
        original = layer.forward

        def forward(
            self: torch.nn.Module,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor | None = None,
            *args: object,
            _original: object = original,
            _slot: int = layer_slot,
            **kwargs: object,
        ) -> object:
            output = _original(positions, hidden_states, residual, *args, **kwargs)  # type: ignore[operator]
            workload = workload_provider()
            if workload.state_version == 0:
                return output
            hidden = cast(tuple[torch.Tensor, ...], output)[0]
            query_start, sequence_lengths, computed = _prefill_metadata()
            rows = _prompt_rows(
                positions.reshape(-1),
                query_start.to(hidden.device),
                sequence_lengths.to(hidden.device),
                computed.to(hidden.device),
                first_tokens=spec.first_tokens,
                last_tokens=spec.last_tokens,
            )
            if rows.numel() > 0:
                hidden.index_add_(
                    0,
                    rows,
                    workload.loreft_delta(_slot, hidden[rows]).to(hidden.dtype),
                )
            return output

        layer.forward = MethodType(forward, layer)  # type: ignore[method-assign]
        wrapped.append(layer)
        forwards.append(original)
    return ReftPatchHandle(wrapped, forwards)


def install_reft_patch(
    model: torch.nn.Module, workload: ReftWorkload
) -> ReftPatchHandle:
    return _install_reft_patch(model, workload, lambda: workload)


def install_reft_patch_on_llm(
    llm: object, runtime: TllmRuntime[ReftWorkload]
) -> ReftPatchHandle:
    runner = llm.llm_engine.model_executor.driver_worker.model_runner  # type: ignore[attr-defined]
    workload = runtime.local_workload(runtime.workloads[0])
    return _install_reft_patch(
        runner.model,
        workload,
        lambda: runtime.local_workload(workload),
    )


__all__ = ["ReftPatchHandle", "install_reft_patch", "install_reft_patch_on_llm"]
