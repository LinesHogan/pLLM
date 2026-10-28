"""Declaration-derived capture layouts and hidden-state taps used by vLLM runners.

Taps are installed before vLLM compiles and captures the model. Persistent index tables
select packed rows to copy into PTTS slots. The full-token fp32 buffers remain in parallel
until consumers migrate to slots; sessions still publish views of those buffers.

The views stay valid until the next forward overwrites the buffers; consumers on the
forward's stream are ordered before that write.
"""

from __future__ import annotations

from collections.abc import Callable
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
from pllm.contracts.packed_rows import PackedPhaseRowBatch
from pllm.contracts.semantic_graph import HiddenInputSource
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType, TensorFieldSpec, TensorLayoutSpec
from pllm.execution.cuda_stream import CudaStreamReadiness, PinnedStaging
from pllm.execution.ptts.page_bundle import SlotPools, Slots
from pllm.execution.step_input import TensorStepInput
from pllm.kernels.page_copy import copy_pages

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
    buffer: Tensor, slots: Tensor, indices: Tensor, forward: Callable[P, Tensor | tuple[Tensor, ...]]
) -> Callable[P, Tensor | tuple[Tensor, ...]]:
    def tapped(*args: P.args, **kwargs: P.kwargs) -> Tensor | tuple[Tensor, ...]:
        output = forward(*args, **kwargs)
        # vLLM decoder layers return (layer output, residual); the tap keeps the layer output.
        hidden = output if isinstance(output, Tensor) else output[0]
        torch.ops.pllm.tap(hidden, buffer)
        copy_pages(hidden, slots, indices[0], indices[1])
        return output

    return tapped


class HiddenTaps:
    """Fixed buffers holding the latest forward's output of chosen modules, keyed by module path."""

    def __init__(
        self,
        model: torch.nn.Module,
        layouts: tuple[HiddenCaptureLayout, ...],
        pools: dict[str, SlotPools],
        max_tokens: int,
        device: torch.device,
    ) -> None:
        self.layouts, self.pools = layouts, pools
        self.buffers: dict[str, Tensor] = {}
        self.indices: dict[str, Tensor] = {}
        self._slots: dict[str, Slots] = {}
        self._staging = PinnedStaging()
        for layout in layouts:
            path = layout.module_path
            module = model.get_submodule(path)
            buffer = torch.zeros(max_tokens, layout.hidden_size, dtype=torch.float32, device=device)
            indices = torch.full((2, layout.capture_rows), -1, dtype=torch.long, device=device)
            slots = pools[path].tensors[0]
            module.register_buffer("pllm_tap", buffer, persistent=False)
            module.register_buffer("pllm_hidden_slots", slots, persistent=False)
            module.register_buffer("pllm_hidden_indices", indices, persistent=False)
            module.forward = _tapped(buffer, slots, indices, module.forward)
            self.buffers[path] = buffer
            self.indices[path] = indices

    def prepare(self, rows: PackedPhaseRowBatch) -> None:
        """Reserve the declared row union and upload its packed-row to slot mapping before forward."""
        for layout in self.layouts:
            path = layout.module_path
            selected = tuple(row for phase in layout.phases for row in rows.producer_rows(phase))
            indices = self.indices[path]
            slots, _ = self.pools[path].allocate(0, len(selected), torch.cuda.current_stream(indices.device))
            self._slots[path] = slots[:, None]
            host = self._staging.take(indices.numel()).view(indices.shape).numpy()
            host.fill(-1)
            host[0, : len(selected)] = [row.packed_row_index for row in selected]
            host[1, : len(selected)] = slots
            self._staging.copy_to(indices)

    @torch.inference_mode()
    def clear(self) -> None:
        """Disable capture and release producer holds; slot readers will join this lifecycle in A4."""
        completion = torch.cuda.Event() if self._slots else None
        if completion is not None:
            completion.record()
        for path, slots in self._slots.items():
            self.pools[path].release(slots, completion)
        self._slots.clear()
        for indices in self.indices.values():
            indices[1].fill_(-1)

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
