"""Adapt dynamic nn.Module calls for graph capture, model overlays and parameter-backed state."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import cast

import torch
from torch.func import functional_call, vmap
from torch.fx import _symbolic_trace

_ASSERT_IS_NONE: object = getattr(_symbolic_trace, "_assert_is_none")


def module_has_execution_hooks(module: torch.nn.Module) -> bool:
    for owner, names in (
        (module, ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")),
        (
            torch.nn.modules.module,
            (
                "_global_forward_hooks",
                "_global_forward_pre_hooks",
                "_global_backward_hooks",
                "_global_backward_pre_hooks",
            ),
        ),
    ):
        for name in names:
            registry: object = getattr(owner, name)
            if not isinstance(registry, dict) or bool(cast(dict[int, object], registry)):
                return True
    return False


@dataclass(frozen=True, slots=True)
class ModuleCallBinding:
    """Local method and parameter-field binding; values may update in place."""

    module: torch.nn.Module
    forward: object
    parameters: tuple[tuple[str, torch.Tensor], ...]

    @classmethod
    def bind(cls, module: torch.nn.Module) -> "ModuleCallBinding":
        if module_has_execution_hooks(module):
            raise ValueError("model source requires explicit hook effects")
        return cls(module, module.forward, tuple(module.named_parameters(recurse=False)))

    def validate(self, module: torch.nn.Module) -> None:
        if module is not self.module or module.forward != self.forward or module_has_execution_hooks(module):
            raise ValueError("model call changed after binding its source definition")
        parameters = tuple(module.named_parameters(recurse=False))
        if len(parameters) != len(self.parameters) or any(
            name != original_name or value is not original
            for (name, value), (original_name, original) in zip(parameters, self.parameters, strict=True)
        ):
            raise ValueError("model parameter fields changed after binding its source definition")


class _ModuleCalls(torch.fx.Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        # Keep child calls in their original representation, including custom
        # kernels and mutations. The backend, not the tracer, owns their effects.
        return bool(module_qualified_name)


def capture_module_calls(module: torch.nn.Module, *, none_inputs: tuple[str, ...] = ()) -> torch.fx.GraphModule:
    """Capture original child calls, retaining explicit None specializations."""
    for child in module.modules():
        ModuleCallBinding.bind(child)
    graph = _ModuleCalls().trace(module, concrete_args={name: None for name in none_inputs} or None)
    return torch.fx.GraphModule(module, graph)


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
