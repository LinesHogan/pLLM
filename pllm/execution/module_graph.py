"""Run module calls with parameter-backed state for tensor tasks."""

from collections.abc import Callable, Iterator

import torch
from torch.func import functional_call, vmap


def _views(module: torch.nn.Module, flat: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    offset = 0
    for name, p in module.named_parameters():
        yield name, flat.narrow(-1, offset, p.numel()).view(*flat.shape[:-1], *p.shape)
        offset += p.numel()


class _Method(torch.nn.Module):
    """Expose one method of the wrapped module as `forward`, so `functional_call` can reach it."""

    def __init__(self, module: torch.nn.Module, method: str) -> None:
        super().__init__()
        self.module = module
        self.method = method

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return getattr(self.module, self.method)(*args)


class ModuleCall:
    """The module as an operation sees it: calls use the state's parameters, one set per row for request state."""

    def __init__(self, module: torch.nn.Module, flat: torch.Tensor, per_row: bool) -> None:
        self._module, self._flat, self._per_row = module, flat, per_row

    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        return self._run(self._module, "", args)

    def __getattr__(self, method: str) -> Callable[..., torch.Tensor]:
        if method.startswith("_"):
            raise AttributeError(method)
        wrapper = _Method(self._module, method)
        return lambda *args: self._run(wrapper, "module.", args)

    def _run(self, module: torch.nn.Module, prefix: str, args: tuple[torch.Tensor, ...]) -> torch.Tensor:
        def call(flat: torch.Tensor, *call_args: torch.Tensor) -> torch.Tensor:
            parameters = {prefix + name: view for name, view in _views(self._module, flat)}
            return functional_call(module, parameters, call_args)

        def row(flat: torch.Tensor, *row_args: torch.Tensor) -> torch.Tensor:
            # The module sees a batch of one row, as it would in ordinary PyTorch code.
            return call(flat, *(arg.unsqueeze(0) for arg in row_args)).squeeze(0)

        return vmap(row)(self._flat, *args) if self._per_row else call(self._flat, *args)
