#!/usr/bin/env python3
"""Structural identities for compiler-visible port locations.

Runtime payloads never cross this boundary.  The descriptor is deliberately
plain, immutable data so the Rust compiler can use the full tagged structure
as a hash key instead of relying on an escaped string convention.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass, fields, is_dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import cast

from tllm.ports.base import PortRead, PortWrite


@dataclass(frozen=True, slots=True)
class LocatorFieldDesc:
    path: tuple[str, ...]
    value_type: str
    value: str

    def __post_init__(self) -> None:
        path = cast(object, self.path)
        if not isinstance(path, tuple):
            raise TypeError("LocatorFieldDesc.path must be tuple[str, ...]")
        typed_path = cast(tuple[object, ...], path)
        if not all(isinstance(segment, str) for segment in typed_path):
            raise TypeError("LocatorFieldDesc.path must be tuple[str, ...]")
        normalized_path = cast(tuple[str, ...], typed_path)
        if not normalized_path or any(not segment for segment in normalized_path):
            raise ValueError("LocatorFieldDesc.path must contain non-empty segments")
        value_type = cast(object, self.value_type)
        if not isinstance(value_type, str) or not value_type:
            raise TypeError("LocatorFieldDesc.value_type must be non-empty str")
        value = cast(object, self.value)
        if not isinstance(value, str):
            raise TypeError("LocatorFieldDesc.value must be str")
        object.__setattr__(self, "path", normalized_path)
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class PortAddressDesc:
    kind: str
    locator_type: str = ""
    locator_fields: tuple[LocatorFieldDesc, ...] = ()

    def __post_init__(self) -> None:
        kind = cast(object, self.kind)
        if not isinstance(kind, str) or not kind:
            raise TypeError("PortAddressDesc.kind must be non-empty str")
        locator_type = cast(object, self.locator_type)
        if not isinstance(locator_type, str):
            raise TypeError("PortAddressDesc.locator_type must be str")
        locator_fields = cast(object, self.locator_fields)
        if not isinstance(locator_fields, tuple):
            raise TypeError("PortAddressDesc.locator_fields must be tuple[LocatorFieldDesc, ...]")
        typed_fields = cast(tuple[object, ...], locator_fields)
        if not all(isinstance(field, LocatorFieldDesc) for field in typed_fields):
            raise TypeError("PortAddressDesc.locator_fields must be tuple[LocatorFieldDesc, ...]")
        normalized_fields = cast(tuple[LocatorFieldDesc, ...], typed_fields)
        if normalized_fields and not locator_type:
            raise ValueError("PortAddressDesc.locator_type is required when locator_fields are present")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "locator_type", locator_type)
        object.__setattr__(self, "locator_fields", normalized_fields)

    @classmethod
    def opaque(cls, name: str) -> "PortAddressDesc":
        """Build an address for compiler tests that do not model a real port."""
        return cls(kind=name)

    @property
    def label(self) -> str:
        """Human-readable display text, not a compiler identity."""

        if not self.locator_type:
            return self.kind
        values = ",".join(
            f"{'.'.join(field.path)}:{field.value_type}={field.value}"
            for field in self.locator_fields
        )
        return f"{self.kind}|{self.locator_type}|{values}"

    @property
    def identity(self) -> str:
        """Stable digest of the complete tagged address structure.

        ``label`` is intentionally concise for logs and can contain arbitrary
        locator text. Closed value identities use this digest instead, so
        delimiters in user values cannot merge distinct ports.
        """

        payload = {
            "kind": self.kind,
            "locator_type": self.locator_type,
            "locator_fields": [
                {
                    "path": list(field.path),
                    "value_type": field.value_type,
                    "value": field.value,
                }
                for field in self.locator_fields
            ],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return sha256(encoded).hexdigest()


def _append_locator_value(
    out: list[LocatorFieldDesc],
    *,
    path: tuple[str, ...],
    value: object,
) -> None:
    if isinstance(value, Enum):
        enum_type = f"{type(value).__module__}.{type(value).__qualname__}"
        enum_value = value.value
        if isinstance(enum_value, str):
            rendered = enum_value
        elif isinstance(enum_value, bool):
            rendered = "true" if enum_value else "false"
        elif isinstance(enum_value, int):
            rendered = f"{enum_value}"
        else:
            raise TypeError(f"unsupported enum locator value {type(enum_value).__name__}")
        out.append(LocatorFieldDesc(path=path, value_type=f"enum:{enum_type}", value=rendered))
        return
    if value is None:
        out.append(LocatorFieldDesc(path=path, value_type="none", value=""))
        return
    if isinstance(value, bool):
        out.append(LocatorFieldDesc(path=path, value_type="bool", value="true" if value else "false"))
        return
    if isinstance(value, int):
        out.append(LocatorFieldDesc(path=path, value_type="int", value=str(value)))
        return
    if isinstance(value, float):
        out.append(LocatorFieldDesc(path=path, value_type="float", value=value.hex()))
        return
    if isinstance(value, str):
        out.append(LocatorFieldDesc(path=path, value_type="str", value=value))
        return
    if isinstance(value, tuple):
        tuple_value = cast(tuple[object, ...], value)
        out.append(LocatorFieldDesc(path=path, value_type="tuple", value=f"{len(tuple_value)}"))
        for index, item in enumerate(tuple_value):
            _append_locator_value(out, path=(*path, str(index)), value=item)
        return
    raise TypeError(f"unsupported locator field value {type(value).__name__}")


def port_address_desc(port: PortRead | PortWrite) -> PortAddressDesc:
    """Lower one public port reference into a tagged structural address."""
    locator = port.locator
    if locator is None:
        return PortAddressDesc(kind=port.kind.value)
    if not is_dataclass(locator) or isinstance(locator, type):
        raise TypeError(f"port locator {type(locator).__name__} must be a dataclass")
    locator_type = f"{type(locator).__module__}.{type(locator).__qualname__}"
    values: list[LocatorFieldDesc] = []
    locator_fields = fields(locator)
    locator_values = astuple(locator)
    for locator_field, locator_value in zip(locator_fields, locator_values, strict=True):
        _append_locator_value(
            values,
            path=(locator_field.name,),
            value=locator_value,
        )
    return PortAddressDesc(
        kind=port.kind.value,
        locator_type=locator_type,
        locator_fields=tuple(values),
    )


def port_address_label(address: PortAddressDesc) -> str:
    return address.label


def port_phase(port: PortRead | PortWrite) -> str:
    """Return the optional phase dimension from a structured port address."""
    address = port_address_desc(port)
    for field in address.locator_fields:
        if field.path == ("phase",):
            return field.value
    return ""
