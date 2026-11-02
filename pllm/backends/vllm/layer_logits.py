"""pllm.layer_logits on vLLM: the model's own final norm and LM head applied to earlier residual streams.

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
from pllm.contracts.hidden_locator import HiddenSite
from pllm.contracts.packed_rows import PackedSamplerRowBatch
from pllm.execution.cuda_stream import upload


class LayerLogits:
    """Write [rows, layers, vocab] FP32 logits into the layer-logits tap for the rows the step keeps."""

    def __init__(self, decoder: Decoder, taps: HiddenTaps, layers: tuple[int, ...]) -> None:
        count = len(decoder.model.model.layers)
        self.decoder, self.taps, self.layers = decoder, taps, layers
        self.output = hidden_binding(-1, HiddenSite.LAYER_LOGITS, count).tap
        self.streams = tuple(hidden_binding(layer, HiddenSite.RESIDUAL, count).tap for layer in layers)
        self.final = layers.index(count - 1) if count - 1 in layers else None

    def compute(self, rows: PackedSamplerRowBatch, logits: torch.Tensor) -> None:
        """Fill this step's rows from the residual taps; `logits` are the model's rows in sampler order."""
        packed = self.taps.selected(self.output)
        if not packed.size:
            return
        streams = torch.cat(tuple(self.taps.read(tap, packed) for tap in self.streams))
        computed = self.decoder.compute_logits(self.decoder.norms[-1](streams))
        assert computed is not None
        values = computed.view(len(self.streams), len(packed), -1).transpose(0, 1).float()
        sampler = {link.packed_row_index: link.sampler_row_index for link in rows.links}
        sampled = tuple((index, sampler[row]) for index, row in enumerate(packed.tolist()) if row in sampler)
        if self.final is not None and sampled:
            kept, positions = (upload(column, logits.device) for column in zip(*sampled, strict=True))
            values[kept, self.final] = logits.index_select(0, positions).float()
        self.taps.write(self.output, packed, values)
