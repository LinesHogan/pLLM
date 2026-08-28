"""vLLM glue for the activation intervention primitive (quarantine zone).

Installs restorable forward wrappers on the declared decoder layers.  Each
wrapper runs the original forward, extracts per-request prefill metadata
from the vLLM forward context, selects prompt rows with the pure
:func:`prefill_row_mask` rule, and applies the LoReFT delta in place on the
returned hidden states — so every downstream layer and the sampler observe
the patched residual stream.

Eager-only in this round: the wrapper is a Python branch on host metadata
and is not CUDA-graph capture-safe.  Chunked-prefill continuation chunks
are excluded by the row rule (see ``prefill_row_mask``); demos should pin
``max_num_batched_tokens >= max_model_len``.
"""

from __future__ import annotations

from types import MethodType
from typing import Callable

import torch

from tllm.contracts.activation_patch import ActivationPatchSpec
from tllm.runtime.activation_patch import LoreftBank, loreft_delta, prefill_row_mask


class ActivationPatchHandle:
    """Restorable handle for one installed patch set."""

    def __init__(self, layers: list[torch.nn.Module], originals: list[object]) -> None:
        self._layers = layers
        self._originals = originals
        self._restored = False

    def restore(self) -> None:
        if self._restored:
            raise RuntimeError("activation patch handle already restored")
        for layer, original in zip(self._layers, self._originals, strict=True):
            layer.forward = original  # type: ignore[union-attr]
        self._restored = True


def _resolve_decoder_layers(model: torch.nn.Module) -> object:
    for path in ("model.model.layers", "model.layers", "transformer.h"):
        current: object = model
        try:
            for part in path.split("."):
                current = getattr(current, part)
        except AttributeError:
            continue
        if isinstance(current, (list, torch.nn.ModuleList)) and len(current) > 0:
            return current
    raise RuntimeError("activation patch could not resolve the decoder layer stack")


def _forward_metadata() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """(query_start_loc, seq_lens, num_computed) from the vLLM forward context."""
    from vllm.forward_context import get_forward_context

    context = get_forward_context()
    metadata = getattr(context, "attn_metadata", None)
    if isinstance(metadata, dict):
        # vLLM 0.27 keys attention metadata per attention layer; every value
        # shares the same CommonAttentionMetadata row layout, so any entry
        # carries the scheduling facts we need.
        metadata = next(iter(metadata.values()), None)
    if metadata is None:
        return None
    query_start_loc = getattr(metadata, "query_start_loc", None)
    seq_lens = getattr(metadata, "seq_lens", None)
    if query_start_loc is None or seq_lens is None:
        return None
    computed = getattr(metadata, "num_computed_tokens_cpu", None)
    if computed is None:
        computed = getattr(metadata, "_num_computed_tokens_cpu", None)
    if computed is None:
        # FlashAttentionMetadata (vLLM 0.27) carries no num_computed field;
        # derive it from its own context_len = seq_len - query_len identity.
        if query_start_loc.ndim != 1 or query_start_loc.shape[0] != seq_lens.shape[0] + 1:
            return None
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        computed = seq_lens - query_lens
    return query_start_loc, seq_lens, torch.as_tensor(computed)


def install_activation_patch(
    model: torch.nn.Module,
    spec: ActivationPatchSpec,
    bank_getter: Callable[[], LoreftBank | None],
) -> ActivationPatchHandle:
    """Wrap ``spec.layers`` forwards so prefill prompt rows get the patch."""
    if not callable(bank_getter):
        raise TypeError("install_activation_patch requires a callable bank_getter")
    layers = _resolve_decoder_layers(model)
    num_layers = len(layers)  # type: ignore[arg-type]
    for layer_index in spec.layers:
        if layer_index >= num_layers:
            raise ValueError(
                f"activation patch layer {layer_index} exceeds the model's "
                f"{num_layers} decoder layers"
            )

    wrapped_layers: list[torch.nn.Module] = []
    originals: list[object] = []
    for layer_slot, layer_index in enumerate(spec.layers):
        layer = layers[layer_index]  # type: ignore[index]
        original_forward = layer.forward

        def patched_forward(
            self: torch.nn.Module,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor | None = None,
            *args: object,
            _original: object = original_forward,
            _slot: int = layer_slot,
            **kwargs: object,
        ) -> object:
            output = _original(positions, hidden_states, residual, *args, **kwargs)  # type: ignore[operator]
            bank = bank_getter()
            if bank is None or not bank.active:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2:
                return output
            metadata = _forward_metadata()
            if metadata is None:
                return output
            query_start_loc, seq_lens, computed = metadata
            mask = prefill_row_mask(
                positions.reshape(-1),
                query_start_loc.to(hidden.device),
                seq_lens.to(hidden.device),
                computed.to(hidden.device),
                first_tokens=bank.spec.first_tokens,
                last_tokens=bank.spec.last_tokens,
            )
            rows = mask.nonzero(as_tuple=False).reshape(-1)
            if int(rows.shape[0]) == 0:
                return output
            delta = loreft_delta(bank, _slot, hidden[rows])
            hidden.index_add_(0, rows, delta.to(hidden.dtype))
            return output

        layer.forward = MethodType(patched_forward, layer)  # type: ignore[method-assign]
        wrapped_layers.append(layer)
        originals.append(original_forward)
    return ActivationPatchHandle(wrapped_layers, originals)


def install_activation_patch_on_llm(
    llm: object,
    spec: ActivationPatchSpec,
    bank_getter: Callable[[], LoreftBank | None],
) -> ActivationPatchHandle:
    """Resolve the runner model from a tLLM-made LLM and install the patch."""
    runner = llm.llm_engine.model_executor.driver_worker.model_runner  # type: ignore[attr-defined]
    model = getattr(runner, "model", None)
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("activation patch could not resolve the runner model")
    return install_activation_patch(model, spec, bank_getter)


__all__ = ["ActivationPatchHandle", "install_activation_patch", "install_activation_patch_on_llm"]
