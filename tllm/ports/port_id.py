#!/usr/bin/env python3
"""Typed port identity.

``PortKind`` stays the coarse category; ``PortId`` is the unique identity of
a port so several producers can offer different ports of the same kind
(e.g. raw residual vs. normalized residual) without colliding.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortId:
    namespace: str
    name: str
    version: int = 1

    def __post_init__(self) -> None:
        if not self.namespace or "." in self.namespace or ":" in self.namespace:
            raise ValueError(f"invalid port namespace {self.namespace!r}")
        if not self.name or ":" in self.name:
            raise ValueError(f"invalid port name {self.name!r}")
        if int(self.version) < 1:
            raise ValueError(f"port version must be >= 1, got {self.version!r}")

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
