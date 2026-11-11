"""Declaration-derived tap layouts and hidden-state taps used by vLLM runners.

LayerOutputs are installed before vLLM compiles and taps the model. Persistent index tables
select packed rows to copy into PTTS slots. Consumers retain holds until their own
streams complete; the session releases producer holds after publication.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import prod
from typing import ParamSpec

import numpy as np
import torch
from torch import Tensor

from pllm.backends.vllm.architecture import hidden_binding
from pllm.config import ModelConfig
from pllm.contracts.hidden_locator import HiddenPhase, HiddenSite, logit_layers
from pllm.contracts.packed_hidden import (
    HiddenRowBatch,
    HiddenRows,
    PackedHiddenResourceBinding,
)
from pllm.contracts.packed_rows import PackedPhaseRowBatch, tap_selection
from pllm.contracts.semantic_graph import HiddenInputSource
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType, TensorFieldSpec, TensorLayoutSpec
from pllm.execution.compile_facts import BackendRequirements
from pllm.execution.cuda_stream import CudaStreamReadiness, PinnedStaging, upload
from pllm.execution.ptts.slot_pools import SlotPools, Slots
from pllm.execution.state_execution import DataT, StateTaskContext
from pllm.execution.step_input import StepInput

TAPS_ENV = "PLLM_HIDDEN_TAPS"


@dataclass(frozen=True, slots=True)
class HiddenTapLayout:
    """One module output's union of the rows its readers select (see tap_selection).

    All consumers share each tapped row. Two generations fund a producer and an
    outstanding reader; reuse must wait for every reader's completion, regardless of
    publish_lag. Layer logits use the same slots but are written by the session at the LM head, not by a hook.
    """

    tap: tuple[str, int]
    site: HiddenSite
    phases: tuple[HiddenPhase, ...]
    tap_rows: int
    shape: tuple[int, ...]
    dtype: TensorDType

    @property
    def retained_rows(self) -> int:
        """Total slot capacity, including the generation being written."""
        return 2 * self.tap_rows

    @property
    def resident_bytes(self) -> int:
        """Slots plus the persistent source/destination int64 table.

        Retention already includes the producer's slots; there is no separate
        capture or handoff allocation. Host ownership and pinned staging are not GPU bytes.
        """
        return self.retained_rows * prod(self.shape) * self.dtype.itemsize + 2 * self.tap_rows * 8

    def create_pool(self, device: torch.device) -> SlotPools:
        """Allocate stable storage before capture; steps lease free rows from field zero."""
        field = TensorFieldSpec(LayoutFieldId(0), self.shape, self.dtype, self.dtype.itemsize)
        return SlotPools(TensorLayoutSpec("pllm.hidden@1", (field,)), (self.retained_rows,), device)


@dataclass(frozen=True, slots=True)
class HiddenTapAnnouncement:
    """Declaration bounds sent before engine construction, resolved when the model loads."""

    inputs: tuple[HiddenInputSource, ...] = ()
    max_num_seqs: int = 1
    max_num_batched_tokens: int = 1
    dtype: TensorDType = TensorDType.FLOAT32
    # The layers pllm.layer_logits reads; the residual stream after each is tapped.
    layer_logits: tuple[int, ...] = ()

    @classmethod
    def declared(cls, requirements: BackendRequirements, config: ModelConfig) -> HiddenTapAnnouncement:
        """The hidden inputs `requirements` declares, bounded by `config`'s request and token capacities."""
        return cls(
            requirements.hidden_inputs,
            config.max_num_seqs,
            config.max_num_batched_tokens,
            TensorDType(config.dtype),
            requirements.layer_logits,
        )

    def resolve(self, layer_count: int, hidden_size: int, vocab_size: int) -> tuple[HiddenTapLayout, ...]:
        """Merge aliases and consumers before capture. Prompt endings and decode rows need one row per request;
        a tap any reader of every token shares needs one per scheduled token."""
        layers = logit_layers(self.layer_logits, layer_count)
        sources = (
            *(source for source in self.inputs if source.site is not HiddenSite.LAYER_LOGITS),
            *(
                HiddenInputSource(layer, HiddenSite.RESIDUAL, phase)
                for layer in layers
                for phase in (HiddenPhase.PREFILL, HiddenPhase.DECODE)
            ),
        )
        taps: dict[tuple[str, int], tuple[HiddenSite, set[HiddenPhase]]] = {}
        for source in sources:
            binding = hidden_binding(source.layer, source.site, layer_count)
            taps.setdefault(binding.tap, (binding.site, set()))[1].add(source.phase)
        layouts = tuple(
            HiddenTapLayout(
                tap,
                site,
                tuple(sorted(phases)),
                self.max_num_batched_tokens if HiddenPhase.ALL in phases else self.max_num_seqs,
                (hidden_size,),
                self.dtype,
            )
            for tap, (site, phases) in sorted(taps.items())
        )
        if not layers:
            return layouts
        logits = hidden_binding(-1, HiddenSite.LAYER_LOGITS, layer_count)
        return (
            *layouts,
            HiddenTapLayout(
                logits.tap,
                logits.site,
                (HiddenPhase.PREFILL, HiddenPhase.DECODE),
                self.max_num_seqs,
                (len(layers), vocab_size),
                TensorDType.FLOAT32,
            ),
        )


P = ParamSpec("P")


@dataclass(slots=True)
class Tap:
    """Copies one module output's selected rows into its slots; the slot table is uploaded before each forward."""

    pool: SlotPools
    indices: Tensor

    rows: Tensor | None = None
    active: Tensor | None = None

    def copy(self, output: Tensor) -> None:
        """Map packed selections to the current wave; padded bucket rows never publish."""
        rows, slots = self.indices
        if self.rows is not None:
            matches = rows[:, None] == self.rows[None, :]
            if self.active is not None:
                matches = matches & self.active.flatten()[None, :]
            rows = matches.long().argmax(1)
            slots = slots.masked_fill(~matches.any(1), -1)
        self.pool.copy_from(0, output, rows, slots)


def _tapped(
    tap: Tap, forward: Callable[P, Tensor | tuple[Tensor, ...]], index: int
) -> Callable[P, Tensor | tuple[Tensor, ...]]:
    def tapped(*args: P.args, **kwargs: P.kwargs) -> Tensor | tuple[Tensor, ...]:
        output = forward(*args, **kwargs)
        # Decoder layers return (layer output, residual) and norms (normalized, residual stream); the tap keeps
        # output `index`. The norm's sum is written in place, so it is copied before the next layer reuses it.
        tap.copy(output if isinstance(output, Tensor) else output[index])
        return output

    return tapped


class HiddenTaps:
    """Selective model-dtype copies of tapped rows into stable PTTS pools, keyed by module output."""

    def __init__(
        self,
        model: torch.nn.Module,
        layouts: tuple[HiddenTapLayout, ...],
        pools: dict[tuple[str, int], SlotPools],
        device: torch.device,
    ) -> None:
        self.layouts, self.pools = layouts, pools
        self.indices = torch.full(
            (sum(2 * layout.tap_rows for layout in layouts),), -1, dtype=torch.long, device=device
        )
        self._slots: dict[tuple[str, int], Slots] = {}
        self._rows: dict[tuple[str, int], Slots] = {}
        # The packed rows each tap selected this step.
        self._selected: dict[tuple[str, int], Slots] = {}
        self._staging = PinnedStaging()
        # The taps wrapping module outputs; a model-site hook that takes over a module copies through them itself.
        self.taps: dict[tuple[str, int], Tap] = {}
        offset = 0
        for layout in layouts:
            path, index = layout.tap
            size = 2 * layout.tap_rows
            indices = self.indices[offset : offset + size].view(2, layout.tap_rows)
            offset += size
            if layout.site is HiddenSite.LAYER_LOGITS:
                continue
            module = model.get_submodule(path)
            module.register_buffer(f"pllm_hidden_slots_{index}", pools[layout.tap].tensors[0], persistent=False)
            module.register_buffer(f"pllm_hidden_indices_{index}", indices, persistent=False)
            self.taps[layout.tap] = Tap(pools[layout.tap], indices)
            module.forward = _tapped(self.taps[layout.tap], module.forward, index)

    def prepare(self, rows: PackedPhaseRowBatch) -> None:
        """Reserve the declared row union and upload its packed-row to slot mapping before forward."""
        host = self._staging.take(self.indices.numel())
        host.fill(-1)
        stream = torch.cuda.current_stream(self.indices.device)
        offset = 0
        for layout in self.layouts:
            tap = layout.tap
            kind = tap_selection(layout.phases)
            selected = rows.tap_selected[kind]
            slots, _ = self.pools[tap].allocate(0, len(selected), stream)
            self._slots[tap] = slots[:, None]
            mapping = slots[rows.tap_ranks[kind]] if slots.size else np.zeros(rows.total_rows, dtype=np.int64)
            self._rows[tap] = mapping
            self._selected[tap] = selected
            size = 2 * layout.tap_rows
            indices = host[offset : offset + size].reshape(2, layout.tap_rows)
            indices[0, : len(selected)] = selected
            indices[1, : len(selected)] = slots
            offset += size
        self._staging.copy_to(self.indices)

    @torch.inference_mode()
    def clear(self) -> None:
        """Disable the taps and release producer holds independently of reader completion."""
        completion = torch.cuda.Event() if self._slots else None
        if completion is not None:
            completion.record()
        for tap, slots in self._slots.items():
            self.pools[tap].release(slots, completion)
        self._slots.clear()
        self._rows.clear()
        self._selected.clear()
        self.indices.fill_(-1)

    def selected(self, tap: tuple[str, int]) -> Slots:
        """The packed rows `tap` keeps this step."""
        return self._selected[tap]

    def read(self, tap: tuple[str, int], packed: Slots) -> Tensor:
        """This step's rows of `tap` at `packed` rows it keeps."""
        storage = self.pools[tap].tensors[0]
        return storage.index_select(0, upload(self._rows[tap][packed], storage.device))

    def write(self, tap: tuple[str, int], packed: Slots, values: Tensor) -> None:
        """Fill the rows `tap` keeps at `packed`; used for layer logits, which no module hook produces."""
        storage = self.pools[tap].tensors[0]
        storage.index_copy_(0, upload(self._rows[tap][packed], storage.device), values)

    def publish(self, value: StepInput, resources: tuple[PackedHiddenResourceBinding, ...]) -> StepInput:
        """Attach the current step's hidden rows for `resources` to `value`."""
        taps = tuple(
            HiddenRows(resource, self.pools[resource.tap].tensors[0], self._rows[resource.tap])
            for resource in resources
        )
        stream = torch.cuda.current_stream()
        return replace(
            value,
            hidden=HiddenRowBatch(value.rows, taps),
            hidden_readiness=CudaStreamReadiness(stream),
        )

    def hold(self, context: StateTaskContext[DataT]) -> None:
        """Acquire the reader's holds before the runner releases the producer."""
        context.hold_slots(tuple((self.pools[tap], slots) for tap, slots in self._slots.items()))


__all__ = ["TAPS_ENV", "HiddenTapAnnouncement", "HiddenTapLayout", "HiddenTaps"]
