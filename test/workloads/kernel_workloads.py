"""Plain Torch custom operators used to verify workload signing, checks and graph capture."""

from dataclasses import dataclass
from math import prod

import torch
import triton
import triton.language as tl
from torch.utils.flop_counter import register_flop_formula

import pllm
from pllm.model import Layers


@triton.jit
def _add(x, y, output, count: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + index, index < count, 0).to(tl.float32)
    branch = tl.load(y + index, index < count, 0).to(tl.float32)
    tl.store(output + index, value + branch, index < count)


@torch.library.custom_op("kernel_test::add", mutates_args=(), device_types="cuda")
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A Triton residual add with the same arithmetic as its Torch CPU reference."""
    output = torch.empty_like(x)
    _add[(triton.cdiv(x.numel(), 256),)](x, y, output, x.numel(), BLOCK=256)
    return output


@add.register_fake
def _fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@add.register_kernel("cpu")
def _cpu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x.float() + y.float()).to(x.dtype)


def _backward(ctx, gradient):
    return gradient, gradient


add.register_autograd(_backward)


@add.register_vmap
def _vmap(info, in_dims, x, y):
    return add(x, y), in_dims[0]


@register_flop_formula(torch.ops.kernel_test.add)
def _flops(x_shape, *args, **kwargs):
    return prod(x_shape)


@torch.library.custom_op("kernel_test::scale", mutates_args=())
def scale(x: torch.Tensor) -> torch.Tensor:
    """An opaque Torch-composite op, deliberately without a FLOP formula."""
    return x * 1.0


@scale.register_fake
def _scale_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@scale.register_vmap
def _scale_vmap(info, in_dims, x):
    return scale(x), in_dims[0]


@dataclass(frozen=True)
class Residual(pllm.Workload):
    """Use both opaque operators inside one residual hook."""

    @pllm.custom_residual(site="attention", layers=(0,))
    def residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return scale(add(x, y))


@dataclass(frozen=True)
class Forward(pllm.Workload):
    """Use both operators in a per-token cell after the native decoder layers."""

    constant: bool = False

    @pllm.custom_forward()
    def forward(self, layers: Layers) -> torch.Tensor:
        hidden = layers(0, layers.layer_count)[-1]
        if self.constant:
            hidden = torch.ones((1, hidden.shape[-1]), dtype=hidden.dtype, device=hidden.device)
        return scale(add(hidden, hidden * 0.0))
