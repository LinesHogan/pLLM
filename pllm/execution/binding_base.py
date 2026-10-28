"""Per-binding declaration state and the authoring values shared by model effects and row tasks.

TensorBindingBase owns the attributes every tensor binding accumulates while a workload declares its
effects; SGD and Buffer are the authoring values both binding halves accept.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import torch

from pllm.contracts.backend_attachment import (
    BackendPlacementProbe,
)
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import (
    FieldView,
    ParameterTarget,
)
from pllm.contracts.semantic_graph import (
    TaskSymbol,
)
from pllm.contracts.state_feed import FeedBinding
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.declaration import Builder, Dimensions, Field
from pllm.execution.tensor_frame import (
    TensorFrame,
)
from pllm.execution.tensor_task import TensorTask


@dataclass(frozen=True)
class SGD:
    lr: float

    def __post_init__(self) -> None:
        if not isfinite(self.lr) or self.lr <= 0:
            raise ValueError("SGD requires a positive finite learning rate")


@dataclass(frozen=True)
class Buffer:
    shape: tuple[int, ...] | Callable[[int, int], tuple[int, ...]]
    dtype: torch.dtype | None = None


def buffer_field(name: str, buffer: Buffer, dimensions: Callable[[Dimensions], tuple[int, int]]) -> Field:
    shape = buffer.shape

    def size(model: Dimensions) -> tuple[int, ...]:
        return shape(*dimensions(model)) if callable(shape) else shape

    return Field(
        name,
        size,
        (lambda model: model.dtype) if buffer.dtype is None else TensorDType(str(buffer.dtype).removeprefix("torch.")),
        8 if buffer.dtype is torch.int64 else 16,
    )


class TensorBindingBase(Builder[TensorFrame]):
    """State shared by the model-effect and row-task halves of one tensor binding."""

    def __init__(self, name: str, digest: str, dimensions: Dimensions | None = None, *, host: bool = False) -> None:
        super().__init__(name, digest, dimensions)
        self.host = host
        self.feeds: tuple[FeedBinding, ...] = ()
        self.exports: dict[TaskSymbol, tuple[int, int, bool]] = {}
        self.tensor_tasks: dict[TaskSymbol, TensorTask] = {}
        self.initializers: dict[str, Callable[[tuple[torch.Tensor, ...], torch.Generator], None]] = {}
        # Module states: their flat initial values, built once at bind from the model dimensions and seed.
        self.initial_values: dict[str, Callable[[Dimensions, int], torch.Tensor]] = {}
        self.weighted_training = False
        self.probe: BackendPlacementProbe | None = None
        self.max_requests = 1
        self.sample_limit: int | None = None
        self.overlay_targets: tuple[ParameterTarget, ...] = ()
        self.parameter_phases: tuple[HiddenPhase, ...] = ()
        self.field_views: dict[tuple[str, str], FieldView] = {}
        self.imports: dict[str, tuple[FieldView, ...]] = {}
