"""Signed vocabulary for activation interventions (hidden write-back).

An activation intervention patches the residual stream in flight at
declared decoder layers and token positions — the engine-side capability
behind ReFT/LoReFT-style methods.  The spec is pure declaration: which
layers, which prompt positions, which phase.  The parameters of the patch
function live in the declaring workload's versioned State; this module
intentionally knows nothing about them.

Phase is pinned to ``"prefill"`` for now: that is the pyreft
``intervene_on_prompt=True`` semantics (patch during prefill, let the KV
carry the effect through decode).  Decode-phase intervention is a separate
design (per-step position identity) and must not sneak in via a free-form
string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast


def _require_layers(value: object) -> tuple[int, ...]:
    # Values arrive JSON-decoded; the object-typed parameter keeps the
    # runtime check honest (declared types are the contract, not evidence).
    if not isinstance(value, (tuple, list)) or not value:
        raise TypeError("ActivationPatchSpec.layers must be non-empty tuple[int, ...]")
    items = tuple(cast("tuple[object, ...] | list[object]", value))
    if any(
        not isinstance(layer, int) or isinstance(layer, bool) or layer < 0
        for layer in items
    ):
        raise TypeError("ActivationPatchSpec.layers must contain only int >= 0")
    if len(set(items)) != len(items):
        raise ValueError("ActivationPatchSpec.layers must not repeat a layer")
    return tuple(sorted(cast(tuple[int, ...], items)))


def _require_count(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"ActivationPatchSpec.{name} must be int >= 0")
    return value


@dataclass(frozen=True, slots=True)
class ActivationPatchSpec:
    """Where an activation intervention applies inside the model forward."""

    layers: tuple[int, ...]
    first_tokens: int = 0
    last_tokens: int = 0
    phase: str = "prefill"

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", _require_layers(self.layers))
        object.__setattr__(
            self, "first_tokens", _require_count(self.first_tokens, "first_tokens")
        )
        object.__setattr__(
            self, "last_tokens", _require_count(self.last_tokens, "last_tokens")
        )
        if self.first_tokens == 0 and self.last_tokens == 0:
            raise ValueError(
                "ActivationPatchSpec must select at least one prompt token "
                "(first_tokens or last_tokens > 0)"
            )
        if self.phase != "prefill":
            raise ValueError(
                "ActivationPatchSpec.phase supports only 'prefill' in this round; "
                "decode-phase intervention needs per-step position identity"
            )


__all__ = ["ActivationPatchSpec"]
