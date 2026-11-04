"""One closed wire codec shared by source validation and execution boundaries."""

from __future__ import annotations

import json
from enum import IntEnum
from hashlib import sha256
from typing import TypeVar

import torch
from cattrs import Converter
from cattrs.errors import ForbiddenExtraKeysError
from cattrs.strategies import configure_tagged_union

from pllm.contracts.semantic_graph import (
    HiddenInputSource,
    HostServiceInputSource,
    InputTokensInputSource,
    PackedRowsInputSource,
    PackedSamplerRowsInputSource,
    SamplerFeedbackInputSource,
    SemanticInputSource,
    StateSelectionInputSource,
    StepIdInputSource,
    TaskResultInputSource,
)

T = TypeVar("T")
_converter = Converter(forbid_extra_keys=True, detailed_validation=False)
_converter.register_unstructure_hook(torch.device, str)
_converter.register_structure_hook(torch.device, lambda value, _target: torch.device(str(value)))
_converter.register_unstructure_hook(IntEnum, lambda value: value.value)

_input_tags: dict[type[object], str] = {
    HiddenInputSource: "hidden",
    InputTokensInputSource: "input_tokens",
    StepIdInputSource: "step_id",
    PackedRowsInputSource: "packed_rows",
    PackedSamplerRowsInputSource: "packed_sampler_rows",
    SamplerFeedbackInputSource: "sampler_feedback",
    HostServiceInputSource: "host_service",
    StateSelectionInputSource: "state_selection",
    TaskResultInputSource: "result",
}
configure_tagged_union(SemanticInputSource, _converter, tag_generator=_input_tags.__getitem__, tag_name="kind")


def encode_boundary(value: object) -> str:
    try:
        return json.dumps(_converter.unstructure(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, AttributeError, KeyError, ValueError) as error:
        raise ValueError("invalid closed boundary value: " + str(error)) from error


def boundary_digest(*values: object) -> str:
    """Bind original typed parameters and cold facts with the same closed encoding."""
    return sha256(encode_boundary(values).encode("utf-8")).hexdigest()


def decode_boundary(payload: str, target: type[T]) -> T:
    """The caller supplies a statically bound type, never a type named by payload."""
    try:
        return _converter.structure(json.loads(payload), target)
    except (TypeError, AttributeError, KeyError, ValueError, ForbiddenExtraKeysError) as exc:
        raise ValueError("invalid closed boundary payload") from exc
