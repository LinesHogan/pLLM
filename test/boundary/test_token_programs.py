"""Model regions: decoder bodies, replay control and vLLM attention metadata as region imports.

The backend exposes decoder-layer dataflow, KV and weights as region functions
and resources. These tests check that composition keeps causality and readonly
ownership, that replay control matches the original loop, and that attention
metadata binds to the original resources. Real vLLM kernels inside a model are
covered by test/gpu/vllm_recirculation_smoke.py.
"""

from dataclasses import replace

import pytest
import torch
from vllm.config import AttentionConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import ForwardContext
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionMetadata

from pllm.backends.vllm.custom_forward_attention import (
    attention_paged_access,
    validate_attention_resources,
)
from pllm.execution.token_program import TokenTensor
from test.state_fixtures import paged_rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="vLLM FlashAttention backend requires CUDA")
def test_original_attention_resource_binds_without_a_model():
    config = VllmConfig(attention_config=AttentionConfig(flash_attn_version=2))
    with set_current_vllm_config(config):
        attention = Attention(
            2, 64, 0.125, num_kv_heads=1, prefix="independent.attention", attn_backend=FlashAttentionBackend
        )
    # Resource-only check: real backend object, CPU storage, no attention forward.
    attention.kv_cache = torch.zeros(4, 1, 16, 128)
    validate_attention_resources((attention,), hidden_size=128)
    slots = torch.tensor([16, 17, 32, 33])
    metadata = FlashAttentionMetadata(
        4,
        2,
        torch.tensor([0, 2, 4], dtype=torch.int32),
        18,
        torch.tensor([18, 18], dtype=torch.int32),
        torch.tensor([[0, 1], [0, 2]]),
        slots,
        False,
        0,
        None,
        None,
        None,
    )
    name = attention.layer_name
    context = ForwardContext({name: attention}, {name: metadata}, {name: slots})
    batch = paged_rows((16, 17, 16, 17))
    access = attention_paged_access(context, (attention,), batch)
    block, offset = access.cell(name, batch[2])
    assert (block.tolist(), offset.tolist()) == ([2], [0])
    with pytest.raises(ValueError):  # Non-causal attention cannot expose ordinary paged KV.
        attention_paged_access(
            replace(context, attn_metadata={name: replace(metadata, causal=False)}), (attention,), batch
        )


@pytest.mark.parametrize("width", (1, 2, 8))
def test_direct_route_batches_positions_taps_and_state(width):
    from pllm.execution.token_program import TokenProgram
    from pllm.execution.token_program_executor import execute_token_program

    def route(layers):
        seed = torch.where(layers.generated, layers.load("history"), layers.embeddings)
        first = layers(stop=2, hidden=seed + layers.position + seed - seed.mean(dim=0, keepdim=True))
        last = layers(start=1, hidden=first[0].reshape(first[0].shape))[-1]
        layers.save("history", last)
        return first[1]

    program = TokenProgram.trace(route, TokenTensor((1, 4), torch.float32, "cpu"), 3, ("history",))
    seed = torch.arange(width * 4, dtype=torch.float32).reshape(width, 4)
    previous = seed + 10
    positions = torch.arange(width)
    generated = (positions % 2 == 0).reshape(-1, 1)

    def layer(ordinal, hidden, residual, kv):
        if residual is not None:
            residual.add_(hidden)
        return hidden * (ordinal + 1), hidden.clone() if residual is None else residual

    output, (saved,) = execute_token_program(program, seed, positions, generated, (previous,), layer, lambda: None)
    entry = torch.where(generated, previous, seed) + positions[:, None]
    torch.testing.assert_close(output, 4 * entry)
    torch.testing.assert_close(saved, 18 * entry)
    torch.testing.assert_close(seed, torch.arange(width * 4, dtype=torch.float32).reshape(width, 4))


def test_prompt_classification_is_structural_and_requires_the_complete_stack():
    from pllm.execution.token_program import TokenProgram
    from pllm.workloads.recirculation import RecirculationWorkload
    from pllm.workloads.step_loop import StepLoopWorkload

    tensor = TokenTensor((1, 4), torch.float32, "cpu")
    assert TokenProgram.trace(StepLoopWorkload().loop, tensor, 3, ("history",)).prompt_pass_is_identity
    assert not TokenProgram.trace(RecirculationWorkload().recirculate, tensor, 3).prompt_pass_is_identity
    assert not TokenProgram.trace(lambda layers: layers(stop=2)[-2], tensor, 3).prompt_pass_is_identity


@pytest.mark.parametrize("kind", ("shape", "dtype", "device"))
def test_segments_reject_incompatible_hidden_tensors(kind):
    from pllm.execution.token_program import TokenProgram

    def route(layers):
        hidden = {
            "shape": layers.embeddings[:, :1],
            "dtype": layers.embeddings.double(),
            "device": layers.embeddings.to("meta"),
        }[kind]
        return layers(hidden=hidden)[-1]

    with pytest.raises(ValueError, match="hidden tensor"):
        TokenProgram.trace(route, TokenTensor((1, 4), torch.float32, "cpu"), 3)


@pytest.mark.parametrize("width", (1, 8))
def test_constant_cells_expand_without_reading_other_requests(width):
    from pllm.execution.token_program import TokenProgram
    from pllm.execution.token_program_executor import execute_token_program

    program = TokenProgram.trace(
        lambda layers: layers(hidden=torch.ones((1, 4)))[-1], TokenTensor((1, 4), torch.float32, "cpu"), 1
    )
    hidden, _ = execute_token_program(
        program,
        torch.empty(width, 4),
        torch.zeros(width),
        torch.zeros(width, 1, dtype=torch.bool),
        (),
        lambda layer, hidden, residual, kv: (hidden + 1, torch.zeros_like(hidden)),
        lambda: None,
    )
    torch.testing.assert_close(hidden, torch.full((width, 4), 2.0))
