"""Typed host-output ports with workload-instance-owned storage."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar, cast

T = TypeVar("T")


class RuntimeHostOutputStore:
    def __init__(self) -> None:
        self._values: dict[str, list[object]] = {}

    def publish(self, key: str, value: object) -> None:
        self._values.setdefault(key, []).append(value)

    def drain(self, key: str) -> tuple[object, ...]:
        return tuple(self._values.pop(key, ()))


class HostOutput(Generic[T]):
    def __init__(self) -> None:
        self.name = ""
        self.key = ""
        self.owner: type[object] = object

    def __set_name__(self, owner: type[object], name: str) -> None:
        self.name = name
        self.owner = owner
        self.key = f"{owner.__module__}.{owner.__qualname__}:{name}"

    def drain(self, store: RuntimeHostOutputStore) -> tuple[T, ...]:
        return cast(tuple[T, ...], store.drain(self.key))

    def runtime_publisher(
        self, store: RuntimeHostOutputStore
    ) -> Callable[[object, int], None]:
        def publish(value: object, _step_id: int) -> None:
            store.publish(self.key, value)

        return publish


__all__ = ["HostOutput", "RuntimeHostOutputStore"]
