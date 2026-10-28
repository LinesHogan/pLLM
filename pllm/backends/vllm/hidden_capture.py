"""Hidden-state taps that live inside vLLM's compiled forward and CUDA graphs.

Taps are installed when the model loads, before vLLM compiles and captures it. On every
forward, each tapped module copies its output into a fixed fp32 buffer through the
`pllm::tap` custom op, so CUDA graph replays keep writing the same memory. A workload
session then reads the current step's rows from those buffers as views.

The views stay valid until the next forward overwrites the buffers; consumers on the
forward's stream are ordered before that write.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import ParamSpec

import torch
from torch import Tensor

from pllm.contracts.packed_hidden import PackedHiddenResourceBinding, PackedHiddenSnapshot, PackedHiddenSnapshotBatch
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.step_input import TensorStepInput


@torch.library.custom_op("pllm::tap", mutates_args=("buffer",))
def tap(hidden: Tensor, buffer: Tensor) -> None:
    """Copy `hidden` into the leading rows of `buffer`."""
    buffer[: hidden.shape[0]].copy_(hidden)


@tap.register_fake  # pyright: ignore[reportUnknownMemberType]
def _(hidden: Tensor, buffer: Tensor) -> None:
    return None


P = ParamSpec("P")


def _tapped(
    buffer: Tensor, forward: Callable[P, Tensor | tuple[Tensor, ...]]
) -> Callable[P, Tensor | tuple[Tensor, ...]]:
    def tapped(*args: P.args, **kwargs: P.kwargs) -> Tensor | tuple[Tensor, ...]:
        output = forward(*args, **kwargs)
        # vLLM decoder layers return (layer output, residual); the tap keeps the layer output.
        torch.ops.pllm.tap(output if isinstance(output, Tensor) else output[0], buffer)
        return output

    return tapped


class HiddenTaps:
    """Fixed buffers holding the latest forward's output of chosen modules, keyed by module path."""

    def __init__(
        self, model: torch.nn.Module, paths: Iterable[str], max_tokens: int, hidden_size: int, device: torch.device
    ) -> None:
        self.buffers: dict[str, Tensor] = {}
        for path in dict.fromkeys(paths):
            module = model.get_submodule(path)
            buffer = torch.zeros(max_tokens, hidden_size, dtype=torch.float32, device=device)
            module.register_buffer("pllm_tap", buffer, persistent=False)
            module.forward = _tapped(buffer, module.forward)
            self.buffers[path] = buffer

    def publish(self, value: TensorStepInput, resources: tuple[PackedHiddenResourceBinding, ...]) -> TensorStepInput:
        """Attach the current step's hidden rows for `resources` to `value`."""
        missing = [resource.module_path for resource in resources if resource.module_path not in self.buffers]
        if missing:
            raise ValueError(f"hidden outputs of {missing} were not tapped when the model was loaded")
        rows = value.rows.total_rows
        snapshots = tuple(
            PackedHiddenSnapshot(resource, buffer[:rows], buffer)
            for resource in resources
            for buffer in (self.buffers[resource.module_path],)
        )
        stream = torch.cuda.current_stream()
        return replace(
            value,
            hidden=PackedHiddenSnapshotBatch(value.rows, snapshots),
            hidden_readiness=CudaStreamReadiness(stream),
        )


__all__ = ["HiddenTaps", "tap"]
