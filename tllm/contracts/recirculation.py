"""Typed contracts for exact Recirculation backend binding."""

from __future__ import annotations

from dataclasses import dataclass


RECIRCULATION_PREFILL_BINDER_IMPLEMENTATION_ID = (
    "tllm.recirculation:bind_prefill@1"
)


@dataclass(frozen=True, slots=True)
class RecirculationPrefillBindingConfig:
    """Immutable tensor-binding parameters carried outside signed JSON."""

    source_layer: int
    destination_layer: int
    alpha: float
    beta: float | None
    ramp_tokens: int

    def resolved_layers(self, layer_count: int) -> tuple[int, int]:
        source = self.source_layer if self.source_layer >= 0 else layer_count + self.source_layer
        destination = (
            self.destination_layer
            if self.destination_layer >= 0
            else layer_count + self.destination_layer
        )
        if not 0 <= destination < source < layer_count:
            raise ValueError(
                "Recirculation layers must satisfy "
                "0 <= destination_layer < source_layer < layer_count"
            )
        return source, destination


__all__ = [
    "RECIRCULATION_PREFILL_BINDER_IMPLEMENTATION_ID",
    "RecirculationPrefillBindingConfig",
]
