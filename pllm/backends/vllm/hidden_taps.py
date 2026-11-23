"""Declaration-derived tap layouts and hidden-state taps used by vLLM runners.

LayerOutputs are installed before vLLM compiles and taps the model. Persistent index tables
select packed rows to copy into PTTS slots. Consumers retain holds until their own
streams complete; the session releases producer holds after publication.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from math import prod
from typing import ParamSpec

import numpy as np
import torch
from torch import Tensor

from pllm.backends.vllm.architecture import CausalLM, hidden_binding, lm_head_module, post_attention_norm_name
from pllm.config import ModelConfig
from pllm.contracts.hidden_locator import (
    HiddenPhase,
    HiddenSite,
    LayerRouteSelector,
    logit_layers,
    resolve_route_layers,
    route_locator_tag,
)
from pllm.contracts.packed_hidden import (
    HiddenRowBatch,
    HiddenRows,
    PackedHiddenResourceBinding,
)
from pllm.contracts.packed_rows import PackedPhaseRowBatch, tap_selection
from pllm.contracts.parameter_overlay import FieldView
from pllm.contracts.semantic_graph import HiddenInputSource, StateResource
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType, TensorFieldSpec, TensorLayoutSpec
from pllm.execution.compile_facts import BackendRequirements
from pllm.execution.cuda_stream import CudaStreamReadiness, PinnedStaging, upload
from pllm.execution.layer_route import RouteSlotGather, routed_tap_copy
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.ptts.slot_pools import SlotPools, Slots
from pllm.execution.state_execution import StateTaskContext
from pllm.execution.step_input import StepInput

TAPS_ENV = "PLLM_HIDDEN_TAPS"


@dataclass(frozen=True, slots=True)
class HiddenTapLayout:
    """One module output's or runtime LayerRoute's union of the rows its readers select (see tap_selection).

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
    route: LayerRouteSelector | None = None
    candidates: tuple[tuple[int, tuple[str, int]], ...] = ()

    @property
    def retained_rows(self) -> int:
        """Total slot capacity, including the generation being written."""
        return 2 * self.tap_rows

    @property
    def resident_bytes(self) -> int:
        """Slots plus persistent source/destination (and optional route) int64 tables.

        Retention already includes the producer's slots; there is no separate
        capture or handoff allocation. Host ownership and pinned staging are not GPU bytes.
        """
        route_bytes = (2 * self.tap_rows + self.retained_rows) * 8 if self.route is not None else 0
        return self.retained_rows * prod(self.shape) * self.dtype.itemsize + 2 * self.tap_rows * 8 + route_bytes

    def create_pool(self, device: torch.device) -> SlotPools:
        """Allocate stable storage before capture; steps lease free rows from field zero."""
        spec = TensorFieldSpec(LayoutFieldId(0), self.shape, self.dtype, self.dtype.itemsize)
        return SlotPools(TensorLayoutSpec("pllm.hidden@1", (spec,)), (self.retained_rows,), device)


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

    def resolve_model(self, model: CausalLM) -> tuple[HiddenTapLayout, ...]:
        """Resolve the worker's tap widths and residual-add paths from its loaded decoder."""
        return self.resolve(
            len(model.model.layers),
            lm_head_module(model).weight.shape[1],
            model.config.vocab_size,
            post_attention_norm_name(type(model).__name__),
        )

    def resolve(
        self, layer_count: int, hidden_size: int, vocab_size: int, post_attn_norm: str = "post_attention_layernorm"
    ) -> tuple[HiddenTapLayout, ...]:
        """Merge aliases and consumers before capture. Prompt endings and decode rows need one row per request;
        a tap any reader of every token shares needs one per scheduled token."""
        layers = logit_layers(self.layer_logits, layer_count)
        static_sources = (
            *(source for source in self.inputs if source.route is None and source.site is not HiddenSite.LAYER_LOGITS),
            *(
                HiddenInputSource(layer, HiddenSite.RESIDUAL, phase)
                for layer in layers
                for phase in (HiddenPhase.PREFILL, HiddenPhase.DECODE)
            ),
        )
        taps: dict[tuple[str, int], tuple[HiddenSite, set[HiddenPhase]]] = {}
        for source in static_sources:
            binding = hidden_binding(source.layer, source.site, layer_count, post_attn_norm)
            taps.setdefault(binding.tap, (binding.site, set()))[1].add(source.phase)
        layouts: list[HiddenTapLayout] = [
            HiddenTapLayout(
                tap,
                site,
                tuple(sorted(phases)),
                self.max_num_batched_tokens if HiddenPhase.ALL in phases else self.max_num_seqs,
                (hidden_size,),
                self.dtype,
            )
            for tap, (site, phases) in sorted(taps.items())
        ]
        if layers:
            logits = hidden_binding(-1, HiddenSite.LAYER_LOGITS, layer_count)
            layouts.append(
                HiddenTapLayout(
                    logits.tap,
                    logits.site,
                    (HiddenPhase.PREFILL, HiddenPhase.DECODE),
                    self.max_num_seqs,
                    (len(layers), vocab_size),
                    TensorDType.FLOAT32,
                )
            )
        layouts.extend(self._resolve_routed(layer_count, hidden_size, vocab_size, post_attn_norm))
        return tuple(layouts)

    def _resolve_routed(
        self, layer_count: int, hidden_size: int, vocab_size: int, post_attn_norm: str
    ) -> list[HiddenTapLayout]:
        routed_taps: dict[tuple[LayerRouteSelector, HiddenSite], set[HiddenPhase]] = {}
        routed_logits: list[LayerRouteSelector] = []
        for source in self.inputs:
            if source.route is None:
                continue
            resolved = replace(source.route, over=resolve_route_layers(source.route.over, layer_count))
            if source.site is HiddenSite.LAYER_LOGITS:
                if resolved not in routed_logits:
                    routed_logits.append(resolved)
                phases = routed_taps.setdefault((resolved, HiddenSite.RESIDUAL), set())
                phases.update((HiddenPhase.PREFILL, HiddenPhase.DECODE))
            else:
                routed_taps.setdefault((resolved, source.site), set()).add(source.phase)
        result: list[HiddenTapLayout] = []
        for (resolved, site), phases in sorted(
            routed_taps.items(), key=lambda item: (route_locator_tag(item[0][0]), item[0][1].value)
        ):
            tag = route_locator_tag(resolved)
            candidates = tuple(
                (layer_id, hidden_binding(layer_id, site, layer_count, post_attn_norm).tap)
                for layer_id in resolved.over
            )
            result.append(
                HiddenTapLayout(
                    (f"routed:{tag}:{site.value}", 0),
                    site,
                    tuple(sorted(phases)),
                    self.max_num_batched_tokens if HiddenPhase.ALL in phases else self.max_num_seqs,
                    (hidden_size,),
                    self.dtype,
                    route=resolved,
                    candidates=candidates,
                )
            )
        for resolved in sorted(routed_logits, key=route_locator_tag):
            tag = route_locator_tag(resolved)
            result.append(
                HiddenTapLayout(
                    (f"routed:{tag}:{HiddenSite.LAYER_LOGITS.value}", 0),
                    HiddenSite.LAYER_LOGITS,
                    (HiddenPhase.PREFILL, HiddenPhase.DECODE),
                    self.max_num_seqs,
                    (vocab_size,),
                    TensorDType.FLOAT32,
                    route=resolved,
                )
            )
        return result


P = ParamSpec("P")


@dataclass(slots=True)
class Tap:
    """Copies one module output's selected rows into static and/or dynamically routed PTTS slots."""

    static: tuple[SlotPools, Tensor] | None = None
    routed: list[tuple[int, SlotPools, Tensor, Tensor, Tensor]] = field(default_factory=lambda: [])
    rows: Tensor | None = None
    active: Tensor | None = None
    inverse: Tensor | None = None

    def _map_rows(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        rows, slots = indices
        if self.inverse is not None:
            mapped = self.inverse.index_select(0, rows.clamp_min(0))
            valid = (rows >= 0) & (mapped >= 0)
            return mapped.clamp_min(0), slots.masked_fill(~valid, -1)
        if self.rows is not None:
            matches = rows[:, None] == self.rows[None, :]
            if self.active is not None:
                matches = matches & self.active.flatten()[None, :]
            rows = matches.long().argmax(1)
            slots = slots.masked_fill(~matches.any(1), -1)
        return rows, slots

    def copy(self, output: Tensor) -> None:
        """Map packed selections to the current wave; padded bucket and unmatched routed rows never publish."""
        if self.static is not None:
            pool, indices = self.static
            rows, slots = self._map_rows(indices)
            pool.copy_from(0, output, rows, slots)
        for layer_id, pool, indices, row_layers, slot_layers in self.routed:
            rows, slots = self._map_rows(indices)
            routed_tap_copy(output, rows, slots, row_layers, pool.tensors[0], slot_layers, layer_id)


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
    """Selective model-dtype copies of tapped rows into stable PTTS pools, keyed by module output or LayerRoute."""

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
        self._indices_views: dict[tuple[str, int], Tensor] = {}
        self._route_layers: dict[tuple[str, int], Tensor] = {}
        self._slot_layers: dict[tuple[str, int], Tensor] = {}
        self._route_gathers: dict[tuple[str, int], RouteSlotGather] = {}
        self._slots: dict[tuple[str, int], Slots] = {}
        self._rows: dict[tuple[str, int], Slots] = {}
        # The packed rows each tap selected this step.
        self._selected: dict[tuple[str, int], Slots] = {}
        self._staging = PinnedStaging()
        # The taps wrapping module outputs; a model-site hook that takes over a module copies through them itself.
        self.taps: dict[tuple[str, int], Tap] = {}
        wrapped: set[tuple[str, int]] = set()
        offset = 0
        for channel, layout in enumerate(layouts):
            size = 2 * layout.tap_rows
            indices = self.indices[offset : offset + size].view(2, layout.tap_rows)
            self._indices_views[layout.tap] = indices
            offset += size
            if layout.route is not None:
                self._route_layers[layout.tap] = torch.full((layout.tap_rows,), -1, dtype=torch.long, device=device)
                self._slot_layers[layout.tap] = torch.full((layout.retained_rows,), -1, dtype=torch.long, device=device)
                self._route_gathers[layout.tap] = RouteSlotGather(layout.tap_rows, device)
            if layout.site is HiddenSite.LAYER_LOGITS:
                continue
            if layout.route is None:
                path, index = layout.tap
                module = model.get_submodule(path)
                module.register_buffer(f"pllm_hidden_slots_{index}", pools[layout.tap].tensors[0], persistent=False)
                module.register_buffer(f"pllm_hidden_indices_{index}", indices, persistent=False)
                tap_entry = self.taps.setdefault(layout.tap, Tap())
                tap_entry.static = (pools[layout.tap], indices)
                if layout.tap not in wrapped:
                    module.forward = _tapped(tap_entry, module.forward, index)
                    wrapped.add(layout.tap)
            else:
                row_layers = self._route_layers[layout.tap]
                slot_layers = self._slot_layers[layout.tap]
                for layer_id, cand_tap in layout.candidates:
                    cand_path, cand_index = cand_tap
                    module = model.get_submodule(cand_path)
                    module.register_buffer(
                        f"pllm_routed_slots_{cand_index}_{channel}", pools[layout.tap].tensors[0], persistent=False
                    )
                    module.register_buffer(f"pllm_routed_indices_{cand_index}_{channel}", indices, persistent=False)
                    module.register_buffer(f"pllm_routed_layers_{cand_index}_{channel}", row_layers, persistent=False)
                    module.register_buffer(
                        f"pllm_routed_slot_layers_{cand_index}_{channel}", slot_layers, persistent=False
                    )
                    tap_entry = self.taps.setdefault(cand_tap, Tap())
                    tap_entry.routed.append((layer_id, pools[layout.tap], indices, row_layers, slot_layers))
                    if cand_tap not in wrapped:
                        module.forward = _tapped(tap_entry, module.forward, cand_index)
                        wrapped.add(cand_tap)

    @property
    def read_buffers(self) -> list[Tensor]:
        """Stable index and route-selector buffers read by boundary tap hooks."""
        return [self.indices, *self._route_layers.values()]

    @property
    def write_buffers(self) -> list[Tensor]:
        """Stable PTTS hidden-slot pools mutated by boundary tap hooks."""
        return [pool.tensors[0] for pool in self.pools.values()]

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
        for layout in self.layouts:
            if layout.route is not None:
                self._route_layers[layout.tap].fill_(-1)
                count = len(self._selected[layout.tap])
                if count > 0:
                    dst_slots = self._indices_views[layout.tap][1, :count]
                    self._slot_layers[layout.tap].index_fill_(0, dst_slots, -1)
                    if layout.site is not HiddenSite.LAYER_LOGITS:
                        self.pools[layout.tap].tensors[0].index_fill_(0, dst_slots, 0)

    def bind_routes(
        self,
        rows: PackedPhaseRowBatch,
        context: StateTaskContext,
        directory: RequestStateDirectory,
        route_views: Mapping[tuple[str, str], FieldView],
    ) -> None:
        """Populate per-row int64 layer indices on device for every routed tap channel."""
        cached: dict[tuple[str, str, tuple[int, ...]], Tensor] = {}
        for layout in self.layouts:
            if layout.route is None:
                continue
            route = layout.route
            selected = self._selected[layout.tap]
            count = len(selected)
            key = (route.resource, route.field, tuple(selected.tolist()))
            target = self._route_layers[layout.tap]
            if key in cached:
                source = cached[key]
                target[:count].copy_(source[:count])
            else:
                view = route_views[route.resource, route.field]
                self._route_gathers[layout.tap].gather(
                    rows,
                    selected,
                    StateResource(route.resource),
                    view,
                    context,
                    directory,
                    latest=True,
                    out=target,
                )
                if count > 0:
                    active = target[:count]
                    matched = torch.zeros_like(active, dtype=torch.bool)
                    for layer_id in route.over:
                        matched = matched | (active == layer_id)
                    active.masked_fill_(~matched, -1)
                cached[key] = target

    def matched_layers(self, tap: tuple[str, int], packed: Slots) -> Tensor:
        """Forward-visited layer indices `[len(packed)]` (-1 when unvisited or disabled) for routed `tap`."""
        slot_layers = self._slot_layers[tap]
        return slot_layers.index_select(0, upload(self._rows[tap][packed], slot_layers.device))

    def write_matched_layers(self, tap: tuple[str, int], packed: Slots, layers: Tensor) -> None:
        """Record visited layer indices into `tap`'s slot-layer provenance buffer (used by LayerLogits)."""
        slot_layers = self._slot_layers[tap]
        slot_layers.index_copy_(0, upload(self._rows[tap][packed], slot_layers.device), layers)

    def route_layers(self, tap: tuple[str, int], count: int) -> Tensor:
        """Device int64 target layers `[count]` resolved for routed `tap` in this step."""
        return self._route_layers[tap][:count]

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
        for row_layers in self._route_layers.values():
            row_layers.fill_(-1)

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
            HiddenRows(
                resource,
                self.pools[resource.tap].tensors[0],
                self._rows[resource.tap],
                self._slot_layers.get(resource.tap),
            )
            for resource in resources
        )
        stream = torch.cuda.current_stream()
        return replace(
            value,
            hidden=HiddenRowBatch(value.rows, taps),
            hidden_readiness=CudaStreamReadiness(stream),
        )

    def holds(self) -> tuple[tuple[SlotPools, Slots], ...]:
        """This step's producer slots, which the step's readers hold before the runner releases them."""
        return tuple((self.pools[tap], slots) for tap, slots in self._slots.items())


__all__ = ["TAPS_ENV", "HiddenTapAnnouncement", "HiddenTapLayout", "HiddenTaps"]
