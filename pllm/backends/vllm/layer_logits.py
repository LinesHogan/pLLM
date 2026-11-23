"""pllm.layer_logits and LayerRoute.layer_logits on vLLM: final norm and LM head applied to tapped residual streams.

The residual taps of the requested layers run inside the model (under CUDA graphs like any hidden tap); this module
turns their rows into logits once per step, at the LM head and outside the captured forward, as a GEMM separate from
the model's own rows so the model's logits stay exactly vLLM's. The last layer's logits are the model's logits
tensor itself where vLLM computed them (sampled rows); a prompt end that vLLM does not sample, inside a recomputed
span, gets the same norm and head over its tapped stream. Used by VllmSession.
"""

from __future__ import annotations

import torch

from pllm.backends.vllm.architecture import Decoder, hidden_binding
from pllm.backends.vllm.hidden_taps import HiddenTaps
from pllm.contracts.hidden_locator import HiddenSite, LayerRouteSelector, route_locator_tag
from pllm.contracts.packed_rows import PackedSamplerRowBatch
from pllm.execution.cuda_stream import upload


class LayerLogits:
    """Write [rows, layers, vocab] and/or routed [rows, vocab] FP32 logits into layer-logits taps."""

    def __init__(
        self,
        decoder: Decoder,
        taps: HiddenTaps,
        layers: tuple[int, ...],
        routed: tuple[LayerRouteSelector, ...] = (),
    ) -> None:
        self.count = len(decoder.model.model.layers)
        self.decoder, self.taps, self.layers, self.routed = decoder, taps, layers, routed
        self.output = hidden_binding(-1, HiddenSite.LAYER_LOGITS, self.count).tap
        self.streams = tuple(hidden_binding(layer, HiddenSite.RESIDUAL, self.count).tap for layer in layers)
        self.final = layers.index(self.count - 1) if self.count - 1 in layers else None

    def compute(self, rows: PackedSamplerRowBatch, logits: torch.Tensor) -> None:
        """Fill this step's static and routed layer-logits rows from the residual taps."""
        sampler = {packed: index for index, packed in enumerate(rows.packed_indices.tolist())}
        if self.layers:
            self._compute_static(sampler, logits)
        for route in self.routed:
            self._compute_routed(route, sampler, logits)

    def _compute_static(self, sampler: dict[int, int], logits: torch.Tensor) -> None:
        packed = self.taps.selected(self.output)
        if not packed.size:
            return
        streams = torch.cat(tuple(self.taps.read(tap, packed) for tap in self.streams))
        computed = self.decoder.compute_logits(self.decoder.norms[-1](streams))
        assert computed is not None
        values = computed.view(len(self.streams), len(packed), -1).transpose(0, 1).float()
        sampled = tuple((index, sampler[row]) for index, row in enumerate(packed.tolist()) if row in sampler)
        if self.final is not None and sampled:
            kept, positions = (upload(column, logits.device) for column in zip(*sampled, strict=True))
            values[kept, self.final] = logits.index_select(0, positions).float()
        self.taps.write(self.output, packed, values)

    def _compute_routed(self, route: LayerRouteSelector, sampler: dict[int, int], logits: torch.Tensor) -> None:
        tag = route_locator_tag(route)
        res_tap = (f"routed:{tag}:{HiddenSite.RESIDUAL.value}", 0)
        out_tap = (f"routed:{tag}:{HiddenSite.LAYER_LOGITS.value}", 0)
        packed = self.taps.selected(out_tap)
        if not packed.size:
            return
        stream = self.taps.read(res_tap, packed)
        computed = self.decoder.compute_logits(self.decoder.norms[-1](stream))
        assert computed is not None
        values = computed.float()
        matched_layers = self.taps.matched_layers(res_tap, packed)
        sampled = tuple((index, sampler[row]) for index, row in enumerate(packed.tolist()) if row in sampler)
        if self.count - 1 in route.over and sampled:
            kept, positions = (upload(column, logits.device) for column in zip(*sampled, strict=True))
            is_final = matched_layers.index_select(0, kept) == (self.count - 1)
            values.index_copy_(
                0,
                kept,
                torch.where(is_final[:, None], logits.index_select(0, positions).float(), values.index_select(0, kept)),
            )
        values = values.masked_fill((matched_layers < 0)[:, None], 0.0)
        self.taps.write(out_tap, packed, values)
        self.taps.write_matched_layers(out_tap, packed, matched_layers)
