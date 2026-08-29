"""Consumer-local structural contract shared by Recirculation implementations."""

from typing import Protocol


class RecirculationSettings(Protocol):
    @property
    def source_layer(self) -> int: ...

    @property
    def destination_layer(self) -> int: ...

    @property
    def alpha(self) -> float: ...

    @property
    def beta(self) -> float | None: ...

    @property
    def ramp_tokens(self) -> int: ...

    def resolved_layers(self, layer_count: int) -> tuple[int, int]: ...


__all__ = ["RecirculationSettings"]
