"""Declaration-derived capture layouts and hidden-state taps used by vLLM runners.

Taps are installed when the model loads, before vLLM compiles and captures it. On every
forward, each tapped module copies its output into a fixed fp32 buffer through the
`pllm::tap` custom op, so CUDA graph replays keep writing the same memory. A workload
session then reads the current step's rows from those buffers as views.

The views stay valid until the next forward overwrites the buffers; consumers on the
forward's stream are ordered before that write.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import ParamSpec

import torch
from torch import Tensor

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_hidden import (
    PackedHiddenResourceBinding,
    PackedHiddenSnapshot,
    PackedHiddenSnapshotBatch,
    hidden_binding,
)
from pllm.contracts.semantic_graph import HiddenInputSource
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType, TensorFieldSpec, TensorLayoutSpec
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.ptts.page_bundle import SlotPools
from pllm.execution.step_input import TensorStepInput

CAPTURE_ENV = "PLLM_HIDDEN_CAPTURE"


@dataclass(frozen=True, slots=True)
class HiddenCaptureLayout:
    """One module's union of latest-decode and completed-prompt-ending rows.

    All consumers share each captured row. Two generations fund a producer and an
    outstanding reader; reuse must wait for every reader's completion, regardless of
    publish_lag. This describes the slot pool, not the full-token taps still in use.
    """

    module_path: str
    phases: tuple[HiddenPhase, ...]
    capture_rows: int
    hidden_size: int
    dtype: TensorDType

    @property
    def retained_rows(self) -> int:
        """Total slot capacity, including the generation being written."""
        return 2 * self.capture_rows

    def create_pool(self, device: torch.device) -> SlotPools:
        """Allocate stable storage before capture; steps lease free rows from field zero."""
        field = TensorFieldSpec(LayoutFieldId(0), (self.hidden_size,), self.dtype, self.dtype.itemsize)
        return SlotPools(TensorLayoutSpec("pllm.hidden@1", (field,)), self.retained_rows, device)


@dataclass(frozen=True, slots=True)
class HiddenCaptureAnnouncement:
    """Declaration bounds sent before engine construction, resolved when the model loads."""

    inputs: tuple[HiddenInputSource, ...] = ()
    max_num_seqs: int = 1
    dtype: TensorDType = TensorDType.FLOAT32

    def resolve(self, layer_count: int, hidden_size: int) -> tuple[HiddenCaptureLayout, ...]:
        """Merge aliases and consumers before capture; phases share one request bound.

        Prefill here selects prompt endings, never all prompt tokens. An all-token
        declaration needs its own token bound rather than this request capacity.
        """
        modules: dict[str, set[HiddenPhase]] = {}
        for source in self.inputs:
            path = hidden_binding(source.layer, source.site, layer_count).module_path
            modules.setdefault(path, set()).add(source.phase)
        return tuple(
            HiddenCaptureLayout(path, tuple(sorted(phases)), self.max_num_seqs, hidden_size, self.dtype)
            for path, phases in sorted(modules.items())
        )


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


__all__ = ["CAPTURE_ENV", "HiddenCaptureAnnouncement", "HiddenCaptureLayout", "HiddenTaps", "tap"]
