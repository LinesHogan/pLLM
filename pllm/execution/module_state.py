"""State declared as an `nn.Module`: its parameters live flattened in one state field.

`pllm.InstanceState(module=...)` and `pllm.RequestState(module=...)` let an author keep an existing PyTorch
module. Authoring (`pllm.usage`) uses `parameter_count` for the field shape and `module_graph.ModuleCall` to hand
operations a callable with the module's interface; the workload binding uses `initial_values` once at bind
time to seed every new state version.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from pllm.execution.checkpoint import PretrainedFile
from pllm.execution.declaration import Dimensions
from pllm.execution.torch_private import fork_rng, manual_seed

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


def initial_values(
    factory: ModuleFactory,
    dimensions: Dimensions,
    seed: int,
    dtype: torch.dtype,
    pretrained: PretrainedFile | None = None,
) -> torch.Tensor:
    """The module's own initialization (or loaded weights), flattened; one build per bind."""
    if pretrained is None:
        with fork_rng([]):
            manual_seed(seed)
            module = factory(dimensions)
    else:
        module = skeleton(factory, dimensions)
        try:
            module.load_state_dict(pretrained.module_values(), strict=True, assign=True)
        except RuntimeError as error:
            raise ValueError(
                f"{pretrained.path} cannot load module state {pretrained.state!r}: {error}; "
                "save the file with safetensors.torch.save_file(module.state_dict(), path) from the same module class"
            ) from error
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()]).to(dtype)


def copy_initial(values: torch.Tensor, root: tuple[torch.Tensor, ...], generator: torch.Generator) -> None:
    """Seed a new state version: the flat parameters are the state's first field."""
    root[0].copy_(values)
