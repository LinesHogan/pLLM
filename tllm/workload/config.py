"""Statically typed workload configuration with Rust-owned normalization."""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Generic, Literal, Protocol, Self, TypeVar, cast, dataclass_transform, overload

from tllm import _core

ConfigAtom = bool | int | float | str | tuple[int, ...]
T = TypeVar("T", bound=ConfigAtom)
ConfigStrT = TypeVar("ConfigStrT", bound=str)


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


class _ConfigFieldToken:
    pass


_CONFIG_FIELD_TOKEN = _ConfigFieldToken()


class ConfigField(Generic[T]):
    """A config value on instances and a typed declaration reference on classes."""

    def __init__(
        self,
        default: T,
        kind: ConfigKind,
        constraint: NumericConstraint = NumericConstraint.ANY,
        choices: tuple[str, ...] = (),
        *,
        _token: _ConfigFieldToken,
    ) -> None:
        if _token is not _CONFIG_FIELD_TOKEN:
            raise TypeError("ConfigField must use a nominal typed constructor")
        self.default = default
        self.name = ""
        self.kind = kind
        self.constraint = constraint
        self.choices = choices

    @classmethod
    def boolean(
        cls,
        default: bool,
        constraint: NumericConstraint = NumericConstraint.ANY,
    ) -> ConfigField[bool]:
        return ConfigField(
            default, ConfigKind.BOOL, constraint, _token=_CONFIG_FIELD_TOKEN
        )

    @classmethod
    def integer(
        cls,
        default: int,
        constraint: NumericConstraint = NumericConstraint.ANY,
    ) -> ConfigField[int]:
        return ConfigField(
            default, ConfigKind.INT, constraint, _token=_CONFIG_FIELD_TOKEN
        )

    @classmethod
    def floating(
        cls,
        default: float,
        constraint: NumericConstraint = NumericConstraint.ANY,
    ) -> ConfigField[float]:
        return ConfigField(
            default, ConfigKind.FLOAT, constraint, _token=_CONFIG_FIELD_TOKEN
        )

    @classmethod
    def text(
        cls,
        default: ConfigStrT,
        constraint: NumericConstraint = NumericConstraint.ANY,
    ) -> ConfigField[ConfigStrT]:
        return ConfigField(
            default, ConfigKind.STR, constraint, _token=_CONFIG_FIELD_TOKEN
        )

    @classmethod
    def choice(
        cls, default: ConfigStrT, choices: tuple[ConfigStrT, ...]
    ) -> ConfigField[ConfigStrT]:
        return ConfigField(
            default,
            ConfigKind.STR,
            NumericConstraint.ANY,
            tuple(choices),
            _token=_CONFIG_FIELD_TOKEN,
        )

    @classmethod
    def integer_tuple(
        cls,
        default: tuple[int, ...],
        constraint: NumericConstraint = NumericConstraint.ANY,
    ) -> ConfigField[tuple[int, ...]]:
        return ConfigField(
            default,
            ConfigKind.INT_TUPLE,
            constraint,
            _token=_CONFIG_FIELD_TOKEN,
        )

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


@overload
def config_field(
    *,
    default: bool,
    kind: Literal[ConfigKind.BOOL],
    constraint: NumericConstraint = NumericConstraint.ANY,
) -> bool: ...


@overload
def config_field(
    *,
    default: int,
    kind: Literal[ConfigKind.INT],
    constraint: NumericConstraint = NumericConstraint.ANY,
) -> int: ...


@overload
def config_field(
    *,
    default: float,
    kind: Literal[ConfigKind.FLOAT],
    constraint: NumericConstraint = NumericConstraint.ANY,
) -> float: ...


@overload
def config_field(
    *,
    default: ConfigStrT,
    kind: Literal[ConfigKind.STR],
    constraint: NumericConstraint = NumericConstraint.ANY,
) -> ConfigStrT: ...


@overload
def config_field(
    *,
    default: tuple[int, ...],
    kind: Literal[ConfigKind.INT_TUPLE],
    constraint: NumericConstraint = NumericConstraint.ANY,
) -> tuple[int, ...]: ...


def config_field(
    *,
    default: T,
    kind: ConfigKind,
    constraint: NumericConstraint = NumericConstraint.ANY,
) -> T:
    """PEP 681 field specifier; runtime stores a typed descriptor."""

    return cast(
        T,
        ConfigField(
            default,
            kind,
            constraint,
            _token=_CONFIG_FIELD_TOKEN,
        ),
    )


def declared_config_field(field: ConfigField[T]) -> T:
    """Install one reusable declaration field with its instance value type.

    A ``ConfigField`` is intentionally dual-faced: declarations consume the
    class-level descriptor while configured workload instances expose ``T``.
    Keep that static bridge here beside Rust-owned config normalization so
    workload authoring code never needs local casts.
    """

    return cast(T, field)


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
                    "choices": field.choices,
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
            if type(value) is list:
                list_items = cast(list[object], value)
                if all(type(item) is int for item in list_items):
                    return tuple(cast(list[int], list_items))
    raise TypeError(
        f"Rust config normalization returned an invalid {field.kind.value} value "
        f"for {field.name!r}"
    )


def _decode_normalized_config(
    payload_json: str, fields: tuple[ConfigField[ConfigAtom], ...]
) -> dict[str, ConfigAtom]:
    decoded: object = json.loads(payload_json)
    if not isinstance(decoded, Mapping):
        raise TypeError("Rust config normalization must return a JSON object")
    mapping = cast(Mapping[object, object], decoded)
    if not all(isinstance(key, str) for key in mapping):
        raise TypeError("Rust config normalization returned non-string field names")
    typed = cast(Mapping[str, object], mapping)
    expected = {field.name for field in fields}
    if set(typed) != expected:
        raise ValueError("Rust config normalization returned an unexpected field set")
    return {field.name: _decode_config_atom(typed[field.name], field) for field in fields}


__all__ = [
    "ConfigField",
    "ConfigKind",
    "NumericConstraint",
    "WorkloadConfig",
    "config_field",
    "declared_config_field",
]
