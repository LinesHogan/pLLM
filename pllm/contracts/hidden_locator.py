"""Closed hidden-activation locator vocabulary shared across compiler layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class LayerRouteSelector:
    """State field holding a runtime physical layer index (or -1 to skip) over candidate layers."""

    resource: str
    field: str
    over: tuple[int, ...]


def resolve_route_layers(over: tuple[int, ...], layer_count: int) -> tuple[int, ...]:
    """Normalize candidate layer indices into sorted unique 0..layer_count - 1 values."""
    if not over or any(not -layer_count <= layer < layer_count for layer in over):
        raise ValueError(f"LayerRoute(over={over!r}) must name layers within 0..{layer_count - 1}")
    resolved = tuple(layer % layer_count for layer in over)
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            f"LayerRoute(over={over!r}) names duplicate layers after resolving against {layer_count} layers"
        )
    return tuple(sorted(resolved))


def route_locator_tag(route: LayerRouteSelector) -> str:
    """Stable locator tag for a resolved LayerRoute tap."""
    return f"{route.resource}.{route.field}:{','.join(map(str, route.over))}"


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
    """Which rows a hidden input reads: each request's completed prompt ending (PREFILL), its latest decode token
    (DECODE), or every token computed fresh this step (ALL), prompt chunks included."""

    NONE = ""
    PREFILL = "prefill"
    DECODE = "decode"
    ALL = "all"


def logit_layers(layers: tuple[int, ...], layer_count: int) -> tuple[int, ...]:
    """The distinct absolute layers of pllm.layer_logits requests, in first-mention order; the last layer's logits
    are the model's own."""
    return tuple(dict.fromkeys(layer % layer_count for layer in layers))


__all__ = [
    "HiddenLayerKind",
    "HiddenPhase",
    "HiddenSite",
    "LayerRouteSelector",
    "logit_layers",
    "resolve_route_layers",
    "route_locator_tag",
]
