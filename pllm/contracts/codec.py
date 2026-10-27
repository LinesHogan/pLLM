"""One closed wire codec shared by source validation and execution boundaries."""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import IntEnum, StrEnum
from hashlib import sha256
from typing import Protocol, TypeVar, cast, get_args, get_origin

import torch
from cattrs import Converter, gen
from cattrs.errors import ForbiddenExtraKeysError
from cattrs.strategies import configure_tagged_union

from pllm.contracts.semantic_graph import (
    HiddenInputSource,
    HostServiceInputSource,
    PackedRowsInputSource,
    PackedSamplerRowsInputSource,
    SamplerFeedbackInputSource,
    SemanticInputSource,
    StateSelectionInputSource,
    StepIdInputSource,
    TaskResultInputSource,
)
from pllm.contracts.tensor_borrow import AccessMode

T = TypeVar("T")
E = TypeVar("E", bound=StrEnum)
_converter = Converter(forbid_extra_keys=True, detailed_validation=False)
_converter.register_unstructure_hook(torch.device, str)
_converter.register_structure_hook(torch.device, lambda value, _target: torch.device(boundary_scalar(value, str)))


def boundary_scalar(value: object, target: type[T]) -> T:
    if target is float and type(value) is int:
        return cast(T, float(value))
    if type(value) is not target:
        raise TypeError("boundary scalar has the wrong type")
    return cast(T, value)


for _type in (str, int, float, bool):
    _converter.register_structure_hook(_type, boundary_scalar)


def _enum(value: object, target: type[E]) -> E:
    return target(boundary_scalar(value, str))


_converter.register_structure_hook(StrEnum, _enum)


def _enum_encoder(target: type[E]) -> Callable[[E], str]:
    def encode(value: E) -> str:
        if type(value) is not target:
            raise TypeError("boundary enum has the wrong type")
        return value.value

    return encode


_converter.register_unstructure_hook(AccessMode, _enum_encoder(AccessMode))

# Source indices must not be objects that merely behave like Python integers.
_converter.register_unstructure_hook(int, lambda value: boundary_scalar(value, int))
_converter.register_unstructure_hook(bool, lambda value: boundary_scalar(value, bool))
_converter.register_unstructure_hook(IntEnum, lambda value: value.value)


class _TupleCodegen(Protocol):
    def make_iterable_unstructure_fn(
        self, target: object, converter: Converter, unstructure_to: type[tuple[object, ...]]
    ) -> Callable[[tuple[object, ...]], object]: ...

    def make_hetero_tuple_unstructure_fn(
        self, target: object, converter: Converter, unstructure_to: type[tuple[object, ...]]
    ) -> Callable[[tuple[object, ...]], object]: ...


def _tuple_encoder(target: object) -> Callable[[object], object]:
    args = get_args(target)
    variable = not args or args[-1] is Ellipsis
    codegen = cast(_TupleCodegen, gen)
    encode = (
        codegen.make_iterable_unstructure_fn(target, _converter, tuple)
        if variable
        else codegen.make_hetero_tuple_unstructure_fn(target, _converter, tuple)
    )

    def checked(value: object) -> object:
        if type(value) is not tuple:
            raise TypeError("boundary sequence must be a tuple")
        items = cast(tuple[object, ...], value)
        if not variable and len(items) != len(args):
            raise ValueError("boundary tuple has the wrong arity")
        return encode(items)

    return checked


_converter.register_unstructure_hook_factory(lambda target: get_origin(target) is tuple, _tuple_encoder)

# Keep cattrs' typed tuple handler, but reject iterable objects that are not JSON arrays.
_tuple_handler = _converter.get_structure_hook(tuple)


def _array(value: object, target: object) -> object:
    if type(value) is not list:
        raise TypeError("boundary sequence must be a JSON array")
    return _tuple_handler(value, target)


_converter.register_structure_hook_func(lambda target: get_origin(target) is tuple, _array)

_input_tags: dict[type[object], str] = {
    HiddenInputSource: "hidden",
    StepIdInputSource: "step_id",
    PackedRowsInputSource: "packed_rows",
    PackedSamplerRowsInputSource: "packed_sampler_rows",
    SamplerFeedbackInputSource: "sampler_feedback",
    HostServiceInputSource: "host_service",
    StateSelectionInputSource: "state_selection",
    TaskResultInputSource: "result",
}
configure_tagged_union(SemanticInputSource, _converter, tag_generator=_input_tags.__getitem__, tag_name="kind")


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate boundary field")
    return result


def _nonfinite(value: str) -> object:
    raise ValueError("non-finite boundary scalar: " + value)


def encode_boundary(value: object) -> str:
    try:
        return json.dumps(_converter.unstructure(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, AttributeError, KeyError) as error:
        raise ValueError("invalid closed boundary value: " + str(error)) from error


def boundary_digest(*values: object) -> str:
    """Bind original typed parameters and cold facts with the same closed encoding."""
    return sha256(encode_boundary(values).encode("utf-8")).hexdigest()


def decode_boundary(payload: str, target: type[T]) -> T:
    """The caller supplies a statically bound type, never a type named by payload."""
    try:
        return _converter.structure(json.loads(payload, object_pairs_hook=_object, parse_constant=_nonfinite), target)
    except (TypeError, AttributeError, KeyError, ValueError, ForbiddenExtraKeysError) as exc:
        raise ValueError("invalid closed boundary payload") from exc
