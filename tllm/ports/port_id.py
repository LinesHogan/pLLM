#!/usr/bin/env python3
"""Typed port identity.

``PortKind`` stays the coarse category; ``PortId`` is the unique identity of
a port so several producers can offer different ports of the same kind
(e.g. raw residual vs. normalized residual) without colliding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True)
class PortId:
    namespace: str
    name: str
    version: int = 1

    def __post_init__(self) -> None:
        namespace = cast(object, self.namespace)
        if not isinstance(namespace, str):
            raise TypeError("PortId.namespace must be str")
        name = cast(object, self.name)
        if not isinstance(name, str):
            raise TypeError("PortId.name must be str")
        version = cast(object, self.version)
        if not isinstance(version, int) or isinstance(version, bool):
            raise TypeError("PortId.version must be int")
        if not namespace or "." in namespace or ":" in namespace:
            raise ValueError(f"invalid port namespace {namespace!r}")
        if not name or ":" in name:
            raise ValueError(f"invalid port name {name!r}")
        if version < 1:
            raise ValueError(f"port version must be >= 1, got {version!r}")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)

    def __str__(self) -> str:
        return f"{self.namespace}:{self.name}@{self.version}"

    @classmethod
    def parse(cls, text: str) -> "PortId":
        try:
            ns, rest = text.split(":", 1)
            name, version = rest.rsplit("@", 1)
            return cls(namespace=ns, name=name, version=int(version))
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"cannot parse PortId from {text!r}") from exc
