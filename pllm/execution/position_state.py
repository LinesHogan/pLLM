"""Positional cache declarations and ordered tensor ports used by token routes."""

from dataclasses import dataclass
from math import isfinite

import torch


@dataclass(frozen=True)
class PositionState:
    """One model-hidden vector per KV allocation block, initialized at token zero."""

    initial: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.initial):
            raise ValueError("initialize PositionState with a finite scalar")
        object.__setattr__(self, "initial", float(self.initial))


@dataclass(frozen=True)
class PositionAnnouncement:
    """PositionState names and initializers, sent to the worker before vLLM allocates KV."""

    states: tuple[tuple[str, float], ...] = ()


@torch.library.custom_op("pllm::position_save", mutates_args=())
def position_save(value: torch.Tensor, name: str) -> None:
    raise RuntimeError("use layers.save inside a @pllm.layer_route method")


def _fake_save(value: torch.Tensor, name: str) -> None:
    pass


position_save.register_fake(_fake_save)  # pyright: ignore[reportUnknownMemberType]


class PositionBuffers:
    """Fixed device indices and initializer tensors borrowing vLLM's complete pages."""

    def __init__(self, tables: tuple[torch.Tensor, ...], initial: tuple[float, ...], tensor: torch.Tensor) -> None:
        self.tables = tables
        self.initial = tuple(torch.full_like(tensor, value) for value in initial)
        self.indices = torch.empty((len(tables), 2, tensor.shape[0]), dtype=torch.int64, device=tensor.device)
        self.flags = torch.empty((2, tensor.shape[0], 1), dtype=torch.bool, device=tensor.device)
        self.saved: tuple[torch.Tensor, ...] = ()

    def entry(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Snapshot predecessor pages before any authored save or current-page write."""
        return (
            self.flags[1],
            *(
                torch.where(self.flags[0], initial, table[index[1]])
                for table, index, initial in zip(self.tables, self.indices, self.initial, strict=True)
            ),
        )

    def exit(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Store only the selected branch's final SSA value for each declaration."""
        for table, index, value in zip(self.tables, self.indices, inputs, strict=True):
            assert isinstance(value, torch.Tensor)
            table[index[0]] = value
        return ()

    def backup(self) -> None:
        """Save the entire writable page before any cold warmup or capture execution."""
        self.saved = tuple(table[index[0]] for table, index in zip(self.tables, self.indices, strict=True))

    def reset(self) -> None:
        """Restore the cold entry state before each attempt and after capture."""
        self.exit(self.saved)
