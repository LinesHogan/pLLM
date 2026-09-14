"""Local tensor compiler interface; executable objects never cross the wire."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from pllm.contracts.region_imports import KernelAllocation


class UnsupportedTensorKernel(ValueError):
    """This implementation cannot provide the required execution/ownership ABI."""


@dataclass(frozen=True, slots=True)
class CompiledTensorCode:
    """Readonly inputs, issuing-stream completion, and explicit allocations.

    Allocations are the backend's capacity account: they must cover result
    backings and any internal workspace with a stated upper bound, and are
    derived without timing or profiling the compiled artifact. The caller
    copies results into independently owned SSA values and accounts for those
    copies. Backend/code identities must be reproducible when rebuilding a
    selection. Generated call counts are diagnostics, not execution or
    capacity evidence.
    """

    function: Callable[..., tuple[torch.Tensor, ...]]
    backend: str
    code_digest: str
    allocations: tuple[KernelAllocation, ...]


class TensorCompiler(Protocol):
    """A feasibility query is provisional; compilation must supply full facts."""

    def supports(self, source: torch.fx.GraphModule) -> bool: ...

    def compile(self, source: torch.fx.GraphModule) -> CompiledTensorCode: ...
