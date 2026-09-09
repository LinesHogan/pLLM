"""Logical-state linear algebra over explicit tensor packing or original page pools.

PagedRows uses arena-issued mappings and never gathers matrix parameters.
TensorRows retains an explicit ordinary-tensor implementation of the same math.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
from pllm.kernels.adamw import AdamWConfig, paged_adamw
from pllm.kernels.paged_gemm import grouped_linear, grouped_parameter_gradient, select_rows, split_linear, write_pages


def _readonly() -> None:
    raise RuntimeError("paged writes require an original private PageWrite")


def paged_workspace(rows: int, states: int, pages: int, activation_width: int, parameter_width: int) -> int:
    """Conservative live activation/gradient and two-field mapping envelope."""
    return 4 * (32 * rows * activation_width + 8 * states * parameter_width) + 8 * (12 * rows + 4 * states + 4 * pages)


@dataclass(frozen=True, slots=True)
class TensorRows:
    """Explicitly packed tensor implementation of logical-state linear algebra.

    This implementation may expand matrices per row; unlike PagedRows it makes no
    direct-pool or allocation-free claim. The caller supplies authorized states;
    an integer row count describes contiguous equal-sized state groups for BMM.
    """

    pool: torch.Tensor
    row_states: torch.Tensor | int
    writable: bool = False

    def __post_init__(self) -> None:
        if self.pool.ndim != 2 or self.pool.shape[0] < 1:
            raise ValueError("tensor rows require a state matrix")
        if isinstance(self.row_states, int):
            if self.row_states < 1:
                raise ValueError("packed state groups require a positive row count")
        elif self.row_states.ndim != 1 or self.row_states.dtype != torch.long or self.row_states.device != self.pool.device:
            raise ValueError("tensor rows require aligned device indices")

    @property
    def states(self) -> torch.Tensor:
        return (torch.arange(self.pool.shape[0], device=self.pool.device).repeat_interleave(self.row_states)
            if isinstance(self.row_states, int) else self.row_states.clamp_min(0))

    def _mask(self, value: torch.Tensor) -> torch.Tensor:
        return value if isinstance(self.row_states, int) else value.masked_fill(
            (self.row_states < 0).reshape(-1, *(1 for _ in value.shape[1:])), 0)

    def _matrix(self, start: int, shape: tuple[int, int]) -> torch.Tensor:
        width = shape[0] * shape[1]
        if min(shape) < 1 or start < 0 or start + width > self.pool.shape[1]:
            raise ValueError("tensor matrix slice exceeds its field")
        return self.pool[:, start:start + width].view(self.pool.shape[0], *shape)

    def _linear(self, inputs: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        rows = self.pool.shape[0] * self.row_states if isinstance(self.row_states, int) else self.row_states.numel()
        if inputs.shape != (rows, matrix.shape[1]):
            raise ValueError("tensor operand differs from its row mapping or matrix layout")
        if isinstance(self.row_states, int):
            return torch.bmm(inputs.reshape(self.pool.shape[0], self.row_states, matrix.shape[1]), matrix).flatten(0, 1)
        output = torch.bmm(inputs.unsqueeze(1), matrix.index_select(0, self.states)).squeeze(1)
        return self._mask(output)

    def linear(self, inputs: torch.Tensor, start: int, shape: tuple[int, int]) -> torch.Tensor:
        return self._linear(inputs, self._matrix(start, shape))

    def input_gradient(self, gradient: torch.Tensor, start: int, shape: tuple[int, int]) -> torch.Tensor:
        return self._linear(gradient, self._matrix(start, shape).transpose(1, 2))

    def parameter_gradient(self, inputs: torch.Tensor, gradient: torch.Tensor, selected: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(self.row_states, int):
            result = torch.bmm(inputs.reshape(self.pool.shape[0], self.row_states, -1).transpose(1, 2),
                gradient.reshape(self.pool.shape[0], self.row_states, -1))
        else:
            inputs, gradient = self._mask(inputs), self._mask(gradient)
            result = self.reduce(inputs.unsqueeze(2) * gradient.unsqueeze(1))
        return result if selected is None else result.index_select(0, selected)

    def values(self, start: int, width: int) -> torch.Tensor:
        return self._mask(self.broadcast(self.pool.narrow(1, start, width)))

    def broadcast(self, values: torch.Tensor) -> torch.Tensor:
        if isinstance(self.row_states, int):
            return values.unsqueeze(1).expand(-1, self.row_states, *values.shape[1:]).flatten(0, 1)
        return values.index_select(0, self.states)

    def reduce(self, values: torch.Tensor) -> torch.Tensor:
        if isinstance(self.row_states, int):
            return values.reshape(self.pool.shape[0], self.row_states, *values.shape[1:]).sum(1)
        return values.new_zeros((self.pool.shape[0], *values.shape[1:])).index_add_(0, self.states, self._mask(values))

    def write(self, values: torch.Tensor, selected: torch.Tensor, *, add: bool = False, scale: float = 1.0) -> None:
        if not self.writable:
            raise RuntimeError("tensor writes require caller-owned private fields")
        values = values.reshape(selected.numel(), self.pool.shape[1])
        if add:
            self.pool.index_add_(0, selected, values, alpha=scale)
        else:
            self.pool.index_copy_(0, selected, values if scale == 1.0 else values * scale)


@dataclass(frozen=True, slots=True)
class PagedRows:
    pool: torch.Tensor
    slots: torch.Tensor
    order: torch.Tensor
    offsets: torch.Tensor
    states: torch.Tensor
    validate: Callable[[], None]
    active: torch.Tensor
    write_slots: torch.Tensor
    padded: bool = False
    validate_write: Callable[[], None] = _readonly

    @staticmethod
    def adamw(fields: tuple["PagedRows", ...], gradients: torch.Tensor, config: AdamWConfig) -> None:
        """Update five original private fields; row selection does not grant write authority."""
        if (len(fields) != 5 or fields[2].pool.shape[1] != 1
            or any(field.states is not fields[0].states for field in fields)):
            raise ValueError("paged AdamW requires five fields with their original shared state mapping")
        for field in fields:
            field.validate_write()
        paged_adamw(tuple(field.pool[:, 0] if index == 2 else field.pool
            for index, field in enumerate(fields)), torch.stack(tuple(field.write_slots for field in fields)), gradients, config)

    @classmethod
    def from_fields(cls, pools: tuple[torch.Tensor, ...], slots: torch.Tensor, states: torch.Tensor,
                    validate: Callable[[], None], validate_write: Callable[[], None]) -> tuple["PagedRows", ...]:
        """All fields share one logical grouping of the original captured roots."""
        safe_states = states.clamp_min(0)
        active = (states >= 0) & (slots[0].index_select(0, safe_states) >= 0)
        order = torch.argsort(torch.where(active, safe_states, slots.shape[1]), stable=True)
        counts = torch.zeros(slots.shape[1], device=slots.device, dtype=torch.int32)
        counts.scatter_add_(0, safe_states, active.to(torch.int32))
        offsets = counts.cumsum(0, dtype=torch.int32)
        return tuple(cls(pool, slots[index], order, offsets, safe_states, validate,
            active, slots[index], True, validate_write) for index, pool in enumerate(pools))

    @staticmethod
    def select_rows(fields: tuple["PagedRows", ...], mask: torch.Tensor) -> tuple["PagedRows", ...]:
        """Restrict computation, not write authority; all fields share one device grouping."""
        if not fields or mask.dtype != torch.bool or mask.shape != fields[0].states.shape or mask.device != fields[0].pool.device:
            raise ValueError("paged selection requires a boolean mask over the original rows")
        first = fields[0]
        for field in fields:
            field.validate()
            if field.states is not first.states or field.active is not first.active or field.offsets is not first.offsets:
                raise ValueError("paged fields must share their original row mapping")
        order, offsets, active = select_rows(first.order, first.offsets, first.active, mask)
        return tuple(replace(field, active=active, order=order, offsets=offsets, padded=True) for field in fields)

    def write(self, values: torch.Tensor, selected: torch.Tensor, *, add: bool = False, scale: float = 1.0) -> None:
        self.validate_write()
        if (self.pool.dtype != torch.float32 or values.ndim < 1 or selected.ndim != 1
            or values.numel() != selected.numel() * self.pool.shape[1] or values.shape[0] != selected.numel()
            or selected.dtype != torch.long or selected.device != self.pool.device
            or values.device != self.pool.device or values.dtype != self.pool.dtype or values.requires_grad):
            raise ValueError("paged writes differ from the original field and selected states")
        write_pages(self.pool, self.write_slots, selected, values, add=add, scale=scale)

    def _mask(self, value: torch.Tensor) -> torch.Tensor:
        return value.masked_fill(~self.active.reshape(-1, *(1 for _ in value.shape[1:])), 0) if self.padded else value

    def _matrix(self, start: int, shape: tuple[int, int]) -> torch.Tensor:
        self.validate()
        width = shape[0] * shape[1]
        if min(shape) < 1 or start < 0 or start + width > self.pool.shape[1]:
            raise ValueError("paged matrix slice exceeds the original field")
        matrix = self.pool[:, start:start + width].view(self.pool.shape[0], *shape)
        if matrix.dtype != torch.float32:
            raise ValueError("paged grouped linear requires FP32 matrices")
        return matrix

    def _operand(self, value: torch.Tensor, width: int) -> None:
        if (value.shape != (self.order.numel(), width) or width < 1
            or value.dtype != self.pool.dtype or value.device != self.pool.device):
            raise ValueError("paged linear operand differs from its row mapping or matrix layout")
        if value.requires_grad:
            raise ValueError("paged linear uses explicit derivatives, not implicit autograd")

    def _linear(self, inputs: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        self._operand(inputs, matrix.shape[1])
        if not len(inputs):
            return inputs.new_empty((0, matrix.shape[2]))
        if matrix.shape[1] >= 128 and matrix.shape[2] <= 16:
            return split_linear(inputs, matrix, self.slots, self.states, self.active)
        return self._mask(grouped_linear(inputs, matrix, self.offsets, self.slots, self.order))

    def linear(self, inputs: torch.Tensor, start: int, shape: tuple[int, int]) -> torch.Tensor:
        return self._linear(inputs, self._matrix(start, shape))

    def input_gradient(self, gradient: torch.Tensor, start: int, shape: tuple[int, int]) -> torch.Tensor:
        return self._linear(gradient, self._matrix(start, shape).transpose(1, 2))

    def parameter_gradient(self, inputs: torch.Tensor, gradient: torch.Tensor, selected: torch.Tensor | None = None) -> torch.Tensor:
        self.validate()
        if inputs.ndim != 2 or gradient.ndim != 2:
            raise ValueError("paged linear gradient requires matrices")
        self._operand(inputs, inputs.shape[1])
        self._operand(gradient, gradient.shape[1])
        if selected is not None and (selected.ndim != 1 or selected.dtype != torch.long or selected.device != self.pool.device):
            raise ValueError("gradient states require original device indices")
        if not len(inputs):
            return inputs.new_zeros((self.offsets.numel() if selected is None else selected.numel(), inputs.shape[1], gradient.shape[1]))
        return grouped_parameter_gradient(inputs, gradient, self.offsets, self.order, selected)

    def values(self, start: int, width: int) -> torch.Tensor:
        """Vector parameters (bias/norm), not expanded matrix parameters."""
        self.validate()
        if start < 0 or width < 1 or start + width > self.pool.shape[1]:
            raise ValueError("paged vector slice exceeds the original field")
        return self._mask(self.pool[:, start:start + width].index_select(0, self.slots.index_select(0, self.states).clamp_min(0)))

    def reduce(self, values: torch.Tensor) -> torch.Tensor:
        self.validate()
        if values.ndim < 1 or values.shape[0] != self.states.numel() or values.device != self.pool.device:
            raise ValueError("segmented values differ from the original row mapping")
        return values.new_zeros((self.offsets.numel(), *values.shape[1:])).index_add_(0, self.states, self._mask(values))

    def broadcast(self, values: torch.Tensor) -> torch.Tensor:
        self.validate()
        return values.index_select(0, self.states)
