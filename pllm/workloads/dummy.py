"""Bounded hidden export expressed as an ordinary replaceable tensor function."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

import pllm


@dataclass(frozen=True, slots=True)
class DummyConfig:
    enabled: bool = True
    export_every_n_steps: int = 256
    export_max_rows: int = 1
    export_max_cols: int = 16
    layer: int = 0

    def __post_init__(self) -> None:
        if min(self.export_every_n_steps, self.export_max_rows, self.export_max_cols) < 1:
            raise ValueError("dummy export requires positive stride and dimensions")


def hidden_rows(hidden: torch.Tensor) -> torch.Tensor:
    return hidden


@dataclass(frozen=True, slots=True)
class DummyWorkload(pllm.Workload):
    config: DummyConfig = DummyConfig()
    transform: Callable[[torch.Tensor], torch.Tensor] = hidden_rows
    export_name: str = "hidden_rows"

    def name(self) -> str:
        return self.export_name

    def every(self) -> int:
        return self.config.export_every_n_steps

    def max_rows(self) -> int:
        return self.config.export_max_rows

    def max_cols(self) -> int:
        return self.config.export_max_cols

    def enabled(self) -> bool:
        return self.config.enabled

    def layer(self) -> int:
        return self.config.layer

    @pllm.export(name=name, every=every, max_rows=max_rows, max_cols=max_cols, enabled=enabled)
    def export_hidden(self, hidden: torch.Tensor = pllm.hidden(layer)) -> torch.Tensor:
        return self.transform(hidden)
