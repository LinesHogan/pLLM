"""Repeat layers 5..8 on decode; compare captured execution with eager whole-layer KV and output."""

from unittest.mock import patch

import torch
from vllm import SamplingParams
from vllm.forward_context import override_forward_context

import pllm
from pllm.config import ModelConfig


class LayerLoop(pllm.Workload):
    """A decode token traverses 0..8, 5..8, then the rest of the model."""

    @pllm.custom_forward(tokens="decode")
    def loop(self, layers: pllm.model.Layers) -> torch.Tensor:
        """Feed the first layer-8 output back into layer 5."""
        first = layers(stop=9)
        second = layers(start=5, stop=9, hidden=first[8])
        return layers(start=9, hidden=second[8])[-1]


def main():
    handle = pllm.make_llm(
        workload=LayerLoop(),
        config=ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", "bfloat16", 64, 1, 64, 0.25, seed=7),
    )
    primitive = handle.runner.require_session()._custom_forward
    native_replay, checked = torch.cuda.CUDAGraph.replay, []

    def replay(graph):
        capture = next(value for value in primitive.captures.values() if value.graph is graph)
        enabled = capture.signature[0]
        slots = tuple(
            (
                layer.self_attn.attn.kv_cache,
                int(capture.attention.context.slot_mapping[layer.self_attn.attn.layer_name].item()),
            )
            for layer in primitive.model.layers
        )
        saved = tuple(cache[slot // cache.shape[2], :, slot % cache.shape[2]].clone() for cache, slot in slots)
        native_replay(graph)
        live = tuple(cache[slot // cache.shape[2], :, slot % cache.shape[2]].clone() for cache, slot in slots)
        for (cache, slot), value in zip(slots, saved, strict=True):
            cache[slot // cache.shape[2], :, slot % cache.shape[2]].copy_(value)
        with override_forward_context(capture.attention.context):
            hidden, residual = capture.seed.clone(), None
            for segment in (range(9), range(5, 9), range(9, len(slots))) if enabled else (range(len(slots)),):
                hidden = hidden if residual is None else hidden + residual
                residual = None
                for ordinal in segment:
                    hidden, residual = primitive.model.layers[ordinal].forward(capture.positions, hidden, residual)
            stream = hidden + residual if enabled else hidden
            expected, _ = primitive.model.norm(stream, torch.zeros_like(stream) if enabled else residual)
        torch.testing.assert_close(capture.output, expected, rtol=0, atol=0)
        for (cache, slot), value in zip(slots, live, strict=True):
            torch.testing.assert_close(cache[slot // cache.shape[2], :, slot % cache.shape[2]], value, rtol=0, atol=0)
        checked.append(enabled)

    try:
        with patch.object(torch.cuda.CUDAGraph, "replay", replay):
            handle.generate(["The sky is blue"], SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True))
        assert checked.count(True) == 2 and checked.count(False) == 0
        print(f"layer_loop: {len(checked)} token outputs and all-layer KV matched eager; decode_rows=2")
    finally:
        handle.close()


if __name__ == "__main__":
    main()
