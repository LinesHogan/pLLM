"""Activation intervention primitive: LoReFT parameter bank + patch math.

This module is the pure, backend-agnostic half of the activation
intervention capability.  It owns:

- :class:`LoreftBank` — the versioned parameter bank the installed forward
  wrappers read every step (one bank per workload, shared through the kit
  host channel so the host object and the reconstructed executing instance
  see the same tensors);
- :func:`loreft_delta` — the LoReFT patch ``(Wh + b − Rh) R^T`` for the
  selected rows of one layer, byte-identical to pyreft's
  ``LoreftIntervention.forward`` with the default linear activation
  (``~/pyreft/pyreft/interventions.py``);
- :func:`prefill_row_mask` — prompt-position selection (first/last N
  tokens) from explicit per-request metadata, so the backend glue never
  re-derives the rule.

The vLLM glue (module wrapping, forward-context metadata extraction) lives
in ``tllm/runtime/vllm_patch/activation_patch_hooks.py`` — the quarantine
boundary owns every vLLM import.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.contracts.activation_patch import ActivationPatchSpec


@dataclass(slots=True)
class LoreftBank:
    """Versioned LoReFT parameters for one workload's declared layer set.

    Tensor layout is stacked over ``spec.layers``: ``source_w`` [L, r, d],
    ``source_b`` [L, r], ``rotate`` [L, d, r] (pyreft's per-layer
    ``{"weight", "bias", "rotate_layer"}``).  Wrappers hold the bank, so
    in-place publication (``copy_``) is how new weights go live.
    """

    spec: ActivationPatchSpec
    source_w: torch.Tensor
    source_b: torch.Tensor
    rotate: torch.Tensor
    version: int = 0
    active: bool = True

    def __post_init__(self) -> None:
        num_layers = len(self.spec.layers)
        if self.source_w.ndim != 3 or self.source_w.shape[0] != num_layers:
            raise ValueError("LoreftBank.source_w must be [layers, rank, hidden]")
        rank, hidden = int(self.source_w.shape[1]), int(self.source_w.shape[2])
        if tuple(self.source_b.shape) != (num_layers, rank):
            raise ValueError("LoreftBank.source_b must be [layers, rank]")
        if tuple(self.rotate.shape) != (num_layers, hidden, rank):
            raise ValueError("LoreftBank.rotate must be [layers, hidden, rank]")

    @property
    def rank(self) -> int:
        return int(self.source_w.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.source_w.shape[2])

    def publish(
        self,
        source_w: torch.Tensor,
        source_b: torch.Tensor,
        rotate: torch.Tensor,
        *,
        version: int,
    ) -> None:
        """Land new parameters in place (wrapper references stay valid)."""
        if not isinstance(version, int) or isinstance(version, bool) or version <= self.version:
            raise ValueError("LoreftBank.publish requires a monotonically increasing version")
        for name, incoming in (
            ("source_w", source_w),
            ("source_b", source_b),
            ("rotate", rotate),
        ):
            current = getattr(self, name)
            if tuple(incoming.shape) != tuple(current.shape):
                raise ValueError(
                    f"LoreftBank.publish shape mismatch for {name}: "
                    f"{tuple(incoming.shape)} != {tuple(current.shape)}"
                )
            with torch.no_grad():
                current.copy_(incoming.to(device=current.device, dtype=current.dtype))
        self.version = version


def loreft_delta(bank: LoreftBank, layer_slot: int, rows: torch.Tensor) -> torch.Tensor:
    """LoReFT patch for ``rows`` [n, d] at one layer slot: (Wh + b − Rh) R^T."""
    weight = bank.source_w[layer_slot]
    rotate = bank.rotate[layer_slot]
    projected = torch.nn.functional.linear(rows, weight, bank.source_b[layer_slot])
    rotated = rows @ rotate
    return (projected - rotated) @ rotate.transpose(0, 1)


def prefill_row_mask(
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    *,
    first_tokens: int,
    last_tokens: int,
) -> torch.Tensor:
    """Boolean mask over packed rows selecting prefill prompt positions.

    A request contributes rows only while prefilling from position zero
    (``num_computed_tokens == 0``); chunked-prefill continuation chunks are
    excluded rather than mis-selected — ``last_tokens`` against a partial
    window would silently patch the wrong rows.  Within a prefill request a
    row is selected when its absolute position is in the first
    ``first_tokens`` or last ``last_tokens`` prompt positions.
    """
    if positions.ndim != 1:
        raise ValueError("prefill_row_mask expects a flat positions tensor")
    device = positions.device
    num_requests = int(query_start_loc.shape[0]) - 1
    if num_requests <= 0:
        return torch.zeros(positions.shape[0], dtype=torch.bool, device=device)
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    request_of_row = torch.repeat_interleave(
        torch.arange(num_requests, device=device), query_lens
    )
    if int(request_of_row.shape[0]) != int(positions.shape[0]):
        raise ValueError("query_start_loc does not cover the packed rows")
    is_prefill = num_computed_tokens == 0
    prompt_lens = seq_lens.to(device)
    per_row_prefill = is_prefill.to(device)[request_of_row]
    per_row_prompt_len = prompt_lens[request_of_row]
    selected = (positions < first_tokens) | (
        positions >= per_row_prompt_len - last_tokens
    )
    return per_row_prefill & selected


__all__ = ["LoreftBank", "loreft_delta", "prefill_row_mask"]
