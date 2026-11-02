"""Closed hidden-activation locator vocabulary shared across compiler layers."""

from __future__ import annotations

from enum import StrEnum


class HiddenLayerKind(StrEnum):
    """Closed hidden-layer locator forms shared by author IR and schedules."""

    NONE = "none"
    INDEX = "index"
    PATH = "path"
    WILDCARD = "wildcard"


class HiddenSite(StrEnum):
    """Where in the forward a hidden input is read.

    BLOCK_OUTPUT is a decoder layer's fresh MLP output (without its residual), MODEL_OUTPUT the final normalized
    hidden. RESIDUAL is the residual stream after a layer, the sum the next sublayer normalizes, and
    ATTENTION_RESIDUAL the stream between a layer's attention and its MLP. LAYER_LOGITS is the LM head applied to
    the final norm of the residual stream after each layer a workload names (pllm.layer_logits). ``NONE`` is the
    only non-hidden wire value and serializes as the legacy empty string.
    """

    NONE = ""
    POST = "post"
    BLOCK_OUTPUT = "block_output"
    MODEL_OUTPUT = "model_output"
    RESIDUAL = "residual"
    ATTENTION_RESIDUAL = "attention_residual"
    LAYER_LOGITS = "layer_logits"


class HiddenPhase(StrEnum):
    """Signed hidden-activation execution phases."""

    NONE = ""
    PREFILL = "prefill"
    DECODE = "decode"


def logit_layers(layers: tuple[int, ...], layer_count: int) -> tuple[int, ...]:
    """The distinct absolute layers of pllm.layer_logits requests, in first-mention order; the last layer's logits
    are the model's own."""
    return tuple(dict.fromkeys(layer % layer_count for layer in layers))


__all__ = [
    "HiddenLayerKind",
    "HiddenPhase",
    "HiddenSite",
    "logit_layers",
]
