#!/usr/bin/env python3
"""Structural identities for compiler-visible port locations.

Runtime payloads never cross this boundary.  The descriptor is deliberately
plain, immutable data so the Rust compiler can use the full tagged structure
as a hash key instead of relying on an escaped string convention.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import cast

from tllm.ports.base import PortRead, PortWrite


@dataclass(frozen=True, slots=True)
class LocatorFieldDesc:
    path: tuple[str, ...]
    value_type: str
    value: str


@dataclass(frozen=True, slots=True)
class PortAddressDesc:
    kind: str
    locator_type: str = ""
    locator_fields: tuple[LocatorFieldDesc, ...] = ()

    @classmethod
    def opaque(cls, name: str) -> "PortAddressDesc":
        """Build an address for compiler tests that do not model a real port."""
        return cls(kind=str(name))

    @property
    def label(self) -> str:
        if not self.locator_type:
            return self.kind
        values = ",".join(
            f"{'.'.join(field.path)}:{field.value_type}={field.value}"
            for field in self.locator_fields
        )
        return f"{self.kind}|{self.locator_type}|{values}"


def _append_locator_value(
    out: list[LocatorFieldDesc],
    *,
    path: tuple[str, ...],
    value: object,
) -> None:
    if isinstance(value, Enum):
        enum_type = f"{type(value).__module__}.{type(value).__qualname__}"
        out.append(LocatorFieldDesc(path=path, value_type=f"enum:{enum_type}", value=str(value)))
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
        out.append(LocatorFieldDesc(path=path, value_type="tuple", value=str(len(tuple_value))))
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
    for locator_field in fields(locator):
        _append_locator_value(
            values,
            path=(locator_field.name,),
            value=getattr(locator, locator_field.name),
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


def port_address_phase(address: PortAddressDesc) -> str:
    """Return a locator's compiler-visible phase constraint, if declared."""
    for field in address.locator_fields:
        if field.path == ("phase",) and field.value_type == "str":
            return field.value
    return ""
