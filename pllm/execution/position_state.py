"""Positional cache declarations and ordered tensor ports used by token routes."""

from dataclasses import dataclass, replace
from math import isfinite

import torch

from pllm.contracts.region_program import (
    AccessMode,
    AccessRange,
    AffineIndex,
    BorrowedResource,
    Function,
    ResourceAccess,
    TensorType,
)
from pllm.execution.region_resources import ExternalFunction
from pllm.execution.region_values import Data
from pllm.execution.tensor_borrow import TensorResourceView


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
    """Versioned model-hidden/model-dtype schema sent before backend KV allocation."""

    states: tuple[tuple[str, float], ...] = ()
    revision: int = 1

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.states)
        if (
            self.revision != 1
            or names != tuple(sorted(set(names)))
            or any(not name or not isfinite(initial) for name, initial in self.states)
        ):
            raise ValueError("PositionState announcement needs revision 1, unique names and finite initializers")


def position_functions(tensor: TensorType, count: int) -> tuple[Function, Function]:
    """Conservatively order entry reads and final writes across the entire cache."""
    accesses = tuple(
        ResourceAccess("position", mode, (AccessRange(AffineIndex(), AffineIndex(1)),))
        for mode in (AccessMode.READ, AccessMode.WRITE)
    )
    return (
        Function(
            "position.entry@1",
            (),
            (TensorType((1,), "torch.bool", tensor.device), *((tensor,) * count)),
            ("position",),
            accesses,
        ),
        Function("position.exit@1", (tensor,) * count, (), ("position",), accesses),
    )


POSITION_RESOURCE = BorrowedResource("position", (1,))


@torch.library.custom_op("pllm::position_save", mutates_args=())
def position_save(value: torch.Tensor, name: str) -> None:
    raise RuntimeError("use layers.save inside a @pllm.layer_route method")


def _fake_save(value: torch.Tensor, name: str) -> None:
    pass


position_save.register_fake(_fake_save)  # pyright: ignore[reportUnknownMemberType]


class PositionBuffers:
    """Fixed device indices and initializer tensors borrowing vLLM's complete pages."""

    resource = POSITION_RESOURCE

    def __init__(self, tables: tuple[torch.Tensor, ...], initial: tuple[float, ...], tensor: torch.Tensor) -> None:
        self.tables = tables
        self.initial = tuple(torch.full_like(tensor, value) for value in initial)
        self.indices = torch.empty((len(tables), 2, 1), dtype=torch.int64, device=tensor.device)
        self.flags = torch.empty(2, dtype=torch.bool, device=tensor.device)
        self.saved: tuple[torch.Tensor, ...] = ()

    def entry(self, inputs: tuple[Data, ...]) -> tuple[Data, ...]:
        """Snapshot predecessor pages before any authored save or current-page write."""
        return (
            self.flags[1:2],
            *(
                torch.where(self.flags[0:1], initial, table[index[1]])
                for table, index, initial in zip(self.tables, self.indices, self.initial, strict=True)
            ),
        )

    def exit(self, inputs: tuple[Data, ...]) -> tuple[Data, ...]:
        """Store only the selected branch's final SSA value for each declaration."""
        for table, index, value in zip(self.tables, self.indices, inputs, strict=True):
            assert isinstance(value, torch.Tensor)
            table[index[0]] = value
        return ()

    def endpoints(self, tensor: TensorType) -> tuple[ExternalFunction, ...]:
        """Bind the signed entry/store ports to this invocation's cache owner."""
        return tuple(
            ExternalFunction(declaration, callback)
            for declaration, callback in zip(
                position_functions(tensor, len(self.tables)), (self.entry, self.exit), strict=True
            )
        )

    def backup(self) -> None:
        """Save the entire writable page before any cold warmup or capture execution."""
        self.saved = tuple(table[index[0]] for table, index in zip(self.tables, self.indices, strict=True))

    def reset(self) -> None:
        """Restore the cold entry state before each attempt and after capture."""
        self.exit(self.saved)

    def validate(self, protected: tuple[tuple[str, int, int], ...]) -> tuple[tuple[str, int, int], ...]:
        tensors = (*self.tables, *self.initial, self.indices, self.flags)
        return TensorResourceView(replace(self.resource, domain=(len(tensors),)), tensors).validate(protected)

    def check(self, mode: AccessMode, ranges: tuple[tuple[int, int], ...]) -> None:
        assert ranges == ((0, 1),)
