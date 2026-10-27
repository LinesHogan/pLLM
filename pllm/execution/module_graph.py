"""Inspect original module calls and hooks before installing model overlays."""

from dataclasses import dataclass
from typing import cast

import torch
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
