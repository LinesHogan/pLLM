"""Statically typed workload configuration with Rust-owned normalization."""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Generic, Protocol, Self, TypeGuard, TypeVar, cast, dataclass_transform, overload

from tllm import _core

T = TypeVar("T")
ConfigAtom = bool | int | float | str | tuple[int, ...]


class _ConfigNormalizer(Protocol):
    def normalize_workload_config(self, schema_json: str, payload_json: str) -> str: ...


_NORMALIZER = cast(_ConfigNormalizer, _core)


class ConfigKind(StrEnum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STR = "str"
    INT_TUPLE = "int_tuple"


class NumericConstraint(StrEnum):
    ANY = "any"
    POSITIVE = "positive"
    NONNEGATIVE = "nonnegative"
    UNIT_INTERVAL = "unit_interval"


class ConfigField(Generic[T]):
    """A config value on instances and a typed declaration reference on classes."""

    def __init__(
        self, default: T, constraint: NumericConstraint = NumericConstraint.ANY
    ) -> None:
        self.default = default
        self.name = ""
        self.kind = _config_kind(default)
        self.constraint = constraint

    def __set_name__(self, owner: type[WorkloadConfig], name: str) -> None:
        self.name = name

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> ConfigField[T]: ...

    @overload
    def __get__(self, instance: WorkloadConfig, owner: type[object] | None = None) -> T: ...

    def __get__(
        self, instance: WorkloadConfig | None, owner: type[object] | None = None
    ) -> ConfigField[T] | T:
        if instance is None:
            return self
        return cast(T, instance.raw_value(self.name))

    def specialization_value(self, value: ConfigAtom) -> str:
        match self.kind:
            case ConfigKind.BOOL | ConfigKind.INT:
                return str(value)
            case ConfigKind.FLOAT:
                return cast(float, value).hex()
            case ConfigKind.STR:
                return cast(str, value)
            case ConfigKind.INT_TUPLE:
                return ",".join(str(item) for item in cast(tuple[int, ...], value))


def config_field(
    *, default: T, constraint: NumericConstraint = NumericConstraint.ANY
) -> T:
    """PEP 681 field specifier; runtime stores a typed descriptor."""

    return cast(T, ConfigField(default, constraint))


@dataclass_transform(field_specifiers=(config_field,), frozen_default=True)
class WorkloadConfig:
    """Immutable config base whose subclasses receive IDE-visible constructors."""

    __config_fields__: tuple[ConfigField[ConfigAtom], ...] = ()
    _config_values: dict[str, ConfigAtom]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        fields: dict[str, ConfigField[ConfigAtom]] = {}
        for base in reversed(cls.__mro__[1:-1]):
            inherited = cast(
                tuple[ConfigField[ConfigAtom], ...],
                base.__dict__.get("__config_fields__", ()),
            )
            fields.update((field.name, field) for field in inherited)
        fields.update(
            (
                name,
                cast(ConfigField[ConfigAtom], value),
            )
            for name, value in cls.__dict__.items()
            if isinstance(value, ConfigField)
        )
        cls.__config_fields__ = tuple(fields.values())

    def __init__(self, **values: ConfigAtom) -> None:
        fields = type(self).__config_fields__
        payload = {field.name: field.default for field in fields}
        payload.update(values)
        schema = {
            "fields": [
                {
                    "name": field.name,
                    "kind": field.kind.value,
                    "constraint": field.constraint.value,
                }
                for field in fields
            ]
        }
        normalized = _NORMALIZER.normalize_workload_config(
            json.dumps(schema, sort_keys=True, separators=(",", ":")),
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
        decoded = _decode_normalized_config(normalized, fields)
        object.__setattr__(
            self,
            "_config_values",
            {
                field.name: decoded[field.name]
                for field in fields
            },
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def to_wire(self) -> Mapping[str, ConfigAtom]:
        return dict(self._config_values)

    def raw_value(self, name: str) -> ConfigAtom:
        return self._config_values[name]

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(field.name for field in cls.__config_fields__)

    @classmethod
    def from_wire(cls, values: Mapping[str, ConfigAtom]) -> Self:
        return cls(**dict(values))

    def replace(self, **changes: ConfigAtom) -> Self:
        values = dict(self._config_values)
        values.update(changes)
        return type(self)(**values)

    def specialization_key(self) -> Mapping[str, str]:
        return {
            field.name: field.specialization_value(self._config_values[field.name])
            for field in type(self).__config_fields__
        }


def _config_kind(value: object) -> ConfigKind:
    """Encode trusted class-definition defaults for the Rust schema boundary."""

    if type(value) is bool:
        return ConfigKind.BOOL
    if type(value) is int:
        return ConfigKind.INT
    if type(value) is float:
        return ConfigKind.FLOAT
    if type(value) is str:
        return ConfigKind.STR
    if _is_int_tuple(value):
        return ConfigKind.INT_TUPLE
    raise TypeError(f"unsupported config default type: {type(value).__name__}")


def _is_string_object_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    if not isinstance(value, Mapping):
        return False
    mapping = cast(Mapping[object, object], value)
    return all(isinstance(key, str) for key in mapping)


def _is_int_tuple(value: object) -> TypeGuard[tuple[int, ...]]:
    if not isinstance(value, tuple):
        return False
    items = cast(tuple[object, ...], value)
    return all(type(item) is int for item in items)


def _int_tuple_from_json(value: object) -> tuple[int, ...] | None:
    if not isinstance(value, list):
        return None
    items = cast(list[object], value)
    if any(type(item) is not int for item in items):
        return None
    return tuple(cast(list[int], items))


def _decode_config_atom(value: object, field: ConfigField[ConfigAtom]) -> ConfigAtom:
    match field.kind:
        case ConfigKind.BOOL:
            if type(value) is bool:
                return value
        case ConfigKind.INT:
            if type(value) is int:
                return value
        case ConfigKind.FLOAT:
            if type(value) is float:
                return value
        case ConfigKind.STR:
            if type(value) is str:
                return value
        case ConfigKind.INT_TUPLE:
            if _is_int_tuple(value):
                return value
            decoded_tuple = _int_tuple_from_json(value)
            if decoded_tuple is not None:
                return decoded_tuple
    raise TypeError(
        f"Rust config normalization returned an invalid {field.kind.value} value "
        f"for {field.name!r}"
    )


def _decode_normalized_config(
    payload_json: str, fields: tuple[ConfigField[ConfigAtom], ...]
) -> dict[str, ConfigAtom]:
    decoded: object = json.loads(payload_json)
    if not _is_string_object_mapping(decoded):
        raise TypeError("Rust config normalization must return a JSON object")
    expected = {field.name for field in fields}
    if set(decoded) != expected:
        raise ValueError("Rust config normalization returned an unexpected field set")
    return {field.name: _decode_config_atom(decoded[field.name], field) for field in fields}


__all__ = [
    "ConfigField",
    "NumericConstraint",
    "WorkloadConfig",
    "config_field",
]
