#!/usr/bin/env python3
"""Dataclass <-> JSON-safe dict conversion for cross-process bootstrap specs.

Only plain dataclasses of primitives / enums / Literals / nested dataclasses
are supported — exactly the shape of consumer configs. Anything else
(callables, tensors, modules) raises TypeError loudly instead of silently
degrading, because a bootstrap spec that cannot cross the process boundary
must fail at capture time in the parent, not in the child.
"""

from __future__ import annotations

import dataclasses
import types
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Literal, Mapping, TypeVar, Union, get_args, get_origin, get_type_hints

T = TypeVar("T")


def dataclass_to_jsonable(obj: Any) -> dict[str, Any]:
    if not is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(f"expected a dataclass instance, got {type(obj).__name__}")
    return {f.name: _to_jsonable(getattr(obj, f.name)) for f in fields(obj)}


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return dataclass_to_jsonable(value)
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(item) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not JSON-safe for bootstrap specs")


def dataclass_from_dict(cls: type[T], data: Mapping[str, Any]) -> T:
    if not (isinstance(cls, type) and is_dataclass(cls)):
        raise TypeError(f"expected a dataclass type, got {cls!r}")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue  # dataclass default applies
        kwargs[f.name] = _from_jsonable(hints.get(f.name, Any), data[f.name], field_name=f.name)
    return cls(**kwargs)  # type: ignore[arg-type, call-arg]


def _from_jsonable(hint: Any, value: Any, *, field_name: str) -> Any:
    if value is None:
        return None
    origin = get_origin(hint)
    if origin in (Union, types.UnionType):
        non_none = [arg for arg in get_args(hint) if arg is not type(None)]
        last_error: Exception | None = None
        for arg in non_none:
            try:
                return _from_jsonable(arg, value, field_name=field_name)
            except (TypeError, ValueError) as exc:
                last_error = exc
        raise TypeError(f"field `{field_name}`: value {value!r} matches no arm of {hint!r}") from last_error
    if origin is Literal:
        allowed = get_args(hint)
        if value not in allowed:
            raise ValueError(f"field `{field_name}`: {value!r} not in {allowed!r}")
        return value
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"field `{field_name}`: expected sequence for tuple, got {type(value).__name__}")
        args = get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_from_jsonable(args[0], item, field_name=field_name) for item in value)
        if args and len(args) == len(value):
            return tuple(_from_jsonable(arg, item, field_name=field_name) for arg, item in zip(args, value))
        return tuple(value)
    if isinstance(hint, type) and is_dataclass(hint):
        if not isinstance(value, Mapping):
            raise TypeError(f"field `{field_name}`: expected mapping for {hint.__name__}, got {type(value).__name__}")
        return dataclass_from_dict(hint, value)
    if isinstance(hint, type) and issubclass(hint, Enum):
        return hint(value)
    if hint is int:
        return int(value)
    if hint is float:
        return float(value)
    if hint is bool:
        return bool(value)
    return value
