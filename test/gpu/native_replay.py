"""Same-shape native vLLM references for replay smokes, including exact KV comparisons."""

from dataclasses import replace

import torch
from vllm.forward_context import get_forward_context, override_forward_context


@torch.no_grad()
def native_pass(primitive, seed, positions, context):
    """Run intact layers and normalize the authored combined carry against the same paged history."""
    cells = tuple(
        (attention.kv_cache, context.slot_mapping[attention.layer_name]) for attention in primitive.attentions
    )
    saved = tuple(cache[slots // cache.shape[2], :, slots % cache.shape[2]].clone() for cache, slots in cells)
    with override_forward_context(context):
        hidden, residual = seed.clone(), None
        for layer in primitive.model.layers:
            hidden, residual = layer(positions, hidden, residual)
        carry = hidden + residual
        # A route returns a materialized tap; vLLM's standalone norm is its native consumer.
        normal = primitive.model.norm(carry)
    for (cache, slots), value in zip(cells, saved, strict=True):
        torch.testing.assert_close(cache[slots // cache.shape[2], :, slots % cache.shape[2]], value, rtol=0, atol=0)
    return carry, normal


def native_span(primitive, rows):
    """Select the span's actual rows; padding and GEMM width are part of the numerical reference."""
    context = get_forward_context()
    packed = torch.tensor([row.packed_row_index for row in rows], device=primitive.hidden.device)
    requests = tuple(dict.fromkeys(row.request_ordinal for row in rows))
    groups = tuple(tuple(row for row in rows if row.request_ordinal == request) for request in requests)
    indices = packed.new_tensor(requests)
    lengths = packed.new_tensor([0, *(len(group) for group in groups)], dtype=torch.int32).cumsum(0).int()
    metadata = {
        attention.layer_name: replace(
            context.attn_metadata[attention.layer_name],
            num_actual_tokens=len(rows),
            max_query_len=max(len(group) for group in groups),
            query_start_loc=lengths,
            seq_lens=packed.new_tensor([group[-1].token_offset + 1 for group in groups], dtype=torch.int32),
            block_table=context.attn_metadata[attention.layer_name].block_table.index_select(0, indices),
            slot_mapping=context.slot_mapping[attention.layer_name].index_select(0, packed),
            num_decode_reqs=0,
            num_decode_tokens=0,
            num_prefill_reqs=len(groups),
            num_prefill_tokens=len(rows),
            scheduler_metadata=None,
        )
        for attention in primitive.attentions
    }
    context = replace(
        context, attn_metadata=metadata, slot_mapping={name: m.slot_mapping for name, m in metadata.items()}
    )
    return native_pass(primitive, primitive.hidden[packed], primitive.positions[packed], context)


class NativeSteps:
    """vLLM worker extension for native numerical checks without serializing callbacks."""

    def begin_native_steps(self):
        """Install the same native StepLoopWorkload checks in either an in-process or EngineCore worker."""
        from contextlib import ExitStack
        from unittest.mock import patch

        from pllm.backends.vllm.custom_forward import ProgramWave, VllmCustomForward

        waves, seeds = {}, {}
        stage, capture, replay, save = (
            ProgramWave.stage,
            ProgramWave.capture,
            torch.cuda.CUDAGraph.replay,
            VllmCustomForward.save_span,
        )

        def staged(wave, rows):
            stage(wave, rows)
            seeds[wave] = torch.where(*wave.position.entry(()), wave.seed).clone()

        def captured(wave):
            capture(wave)
            waves[wave.graph] = wave

        def replayed(graph):
            replay(graph)
            if graph in waves:
                wave = waves[graph]
                carry, normal = native_pass(wave.custom_forward, seeds[wave], wave.positions, wave.attention.context)
                torch.testing.assert_close(wave.output, normal, rtol=0, atol=0)
                torch.testing.assert_close(wave.position.tables[0][wave.position.indices[0, 0]], carry, rtol=0, atol=0)

        def saved(primitive, rows, groups, requests, hidden, values):
            carry, normal = native_span(primitive, rows)
            torch.testing.assert_close(hidden, carry, rtol=0, atol=0)
            packed = [row.packed_row_index for row in rows]
            torch.testing.assert_close(primitive.output[packed], normal, rtol=0, atol=0)
            return save(primitive, rows, groups, requests, hidden, values)

        self.step_reference = ExitStack()
        for owner, name, function in (
            (ProgramWave, "stage", staged),
            (ProgramWave, "capture", captured),
            (torch.cuda.CUDAGraph, "replay", replayed),
            (VllmCustomForward, "save_span", saved),
        ):
            self.step_reference.enter_context(patch.object(owner, name, function))

    def end_native_steps(self):
        """Restore worker methods before testing workload replacement."""
        self.step_reference.close()
