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
    name = "dummy"
    config: DummyConfig = DummyConfig()
    transform: Callable[[torch.Tensor], torch.Tensor] = hidden_rows
    export_name: str = "hidden_rows"

    @pllm.on(pllm.Phase.OBSERVE)
    @pllm.hook(pllm.Config("config.layer"))
    @pllm.export(
        name=pllm.Config("export_name"),
        every=pllm.Config("config.export_every_n_steps"),
        max_rows=pllm.Config("config.export_max_rows"),
        max_cols=pllm.Config("config.export_max_cols"),
        enabled=pllm.Config("config.enabled"),
    )
    def export_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.transform(hidden)
