"""State declared as an `nn.Module`: its parameters live flattened in one state field.

`pllm.InstanceState(module=...)` and `pllm.RequestState(module=...)` let an author keep an existing PyTorch
module. Authoring (`pllm.usage`) uses `parameter_count` for the field shape and `ModuleCall` to hand
operations a callable with the module's interface; the workload binding uses `initial_values` once at bind
time to seed every new state version.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from math import prod

import torch
from torch import nn
from torch.func import functional_call, vmap

from pllm.execution.declaration import Dimensions

ModuleFactory = Callable[[Dimensions], nn.Module]


def skeleton(factory: ModuleFactory, dimensions: Dimensions) -> nn.Module:
    """Build the module's structure without allocating or initializing its parameters."""
    with torch.device("meta"):
        module = factory(dimensions)
    if next(module.buffers(), None) is not None:
        raise ValueError("module state holds parameters only; register buffers as parameters or tensor fields")
    return module


def parameter_count(factory: ModuleFactory) -> Callable[[Dimensions], tuple[int]]:
    """The flat field's shape: every parameter of the module laid end to end."""
    return lambda dimensions: (sum(p.numel() for p in skeleton(factory, dimensions).parameters()),)


def initial_values(factory: ModuleFactory, dimensions: Dimensions, seed: int, dtype: torch.dtype) -> torch.Tensor:
    """The module's own initialization (or loaded weights), flattened; one build per bind."""
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
        module = factory(dimensions)
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()]).to(dtype)


def copy_initial(values: torch.Tensor, root: tuple[torch.Tensor, ...], generator: torch.Generator) -> None:
    """Seed a new state version: the flat parameters are the state's first field."""
    root[0].copy_(values)


def _views(module: nn.Module, flat: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    offset = 0
    for name, p in module.named_parameters():
        yield name, flat.narrow(-1, offset, p.numel()).view(*flat.shape[:-1], *p.shape)
        offset += prod(p.shape)


class _Method(nn.Module):
    """Expose one method of the wrapped module as `forward`, so `functional_call` can reach it."""

    def __init__(self, module: nn.Module, method: str) -> None:
        super().__init__()
        self.module = module
        self.method = method

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return getattr(self.module, self.method)(*args)


class ModuleCall:
    """The module as an operation sees it: calls use the state's parameters, one set per row for request state."""

    def __init__(self, module: nn.Module, flat: torch.Tensor, per_row: bool) -> None:
        self._module, self._flat, self._per_row = module, flat, per_row

    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        return self._run(self._module, "", args)

    def __getattr__(self, method: str) -> Callable[..., torch.Tensor]:
        if method.startswith("_"):
            raise AttributeError(method)
        wrapper = _Method(self._module, method)
        return lambda *args: self._run(wrapper, "module.", args)

    def _run(self, module: nn.Module, prefix: str, args: tuple[torch.Tensor, ...]) -> torch.Tensor:
        def call(flat: torch.Tensor, *call_args: torch.Tensor) -> torch.Tensor:
            parameters = {prefix + name: view for name, view in _views(self._module, flat)}
            return functional_call(module, parameters, call_args)

        def row(flat: torch.Tensor, *row_args: torch.Tensor) -> torch.Tensor:
            # The module sees a batch of one row, as it would in ordinary PyTorch code.
            return call(flat, *(arg.unsqueeze(0) for arg in row_args)).squeeze(0)

        return vmap(row)(self._flat, *args) if self._per_row else call(self._flat, *args)
