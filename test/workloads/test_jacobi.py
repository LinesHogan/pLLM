"""Capture boundaries, span tensor semantics and read-only diagonal attention evidence."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

import pllm
from pllm.backends.vllm import custom_forward_attention as attention
from pllm.backends.vllm.prefix_cache import CachePolicy
from pllm.execution.token_program import TokenProgram, TokenTensor
from pllm.execution.token_program_executor import execute_token_program
from pllm.workloads.recirculation import JacobiRecirculationWorkload
from test.workloads.test_recirculation import FACTS


@pytest.mark.parametrize("tokens", ("all", "decode", "prefill"))
def test_overlapping_routes_rejected(tokens):
    class Overlap(JacobiRecirculationWorkload):
        @pllm.custom_forward(tokens=tokens)
        def extra(self, layers):
            return layers()[-1]

    with pytest.raises(ValueError, match="tokens overlap:.*" + ("decode" if tokens == "all" else tokens)):
        Overlap().build(FACTS)


@pytest.mark.parametrize(
    "over,route,message",
    [
        ("span", lambda l: l.load("state"), "serial predecessor"),
        ("token", lambda l: l(kv="peek")[-1], "commit every layer"),
        ("span", lambda l: l(start=1, hidden=l(stop=1)[0], kv="peek")[-1], "commit every layer"),
    ],
)
def test_capture_rejects_invalid_dependencies(over, route, message):
    with pytest.raises(ValueError, match=message):
        TokenProgram.trace(route, TokenTensor((1, 4), torch.float32, "cpu"), 3, ("state",), over)


@pytest.mark.parametrize("width", (1, 7))
def test_span_cells_keep_packed_tensor_semantics(width):
    def route(layers):
        value = layers.embeddings - layers.embeddings.mean(0, keepdim=True) + layers.position + layers.generated
        shape = layers.embeddings.shape
        layers.save("state", value.reshape(shape))
        return layers(hidden=value.reshape(shape))[-1]

    program = TokenProgram.trace(route, TokenTensor((1, 4), torch.float32, "cpu"), 1, ("state",), "span")
    seed = torch.arange(width * 4, dtype=torch.float32).reshape(width, 4)
    output, (saved,) = execute_token_program(
        program,
        seed,
        torch.arange(width),
        torch.ones(width, 1, dtype=torch.bool),
        (seed,),
        lambda ordinal, hidden, residual, kv: (hidden, torch.zeros_like(hidden)),
        lambda: None,
    )
    torch.testing.assert_close(output, seed - seed.mean(0, keepdim=True) + torch.arange(width)[:, None] + 1)
    torch.testing.assert_close(saved, output)


@pytest.mark.gpu
@pytest.mark.parametrize("start,length", ((0, 1), (0, 7), (3, 5), (16, 4)))
def test_peek_dense_reference_and_unchanged_cache(start, length):
    torch.manual_seed(7)
    q, k, v = (torch.randn(length, heads, 8, device="cuda") for heads in (4, 2, 2))
    cache = torch.randn(start + length, 2, 8, 16, device="cuda")
    before = cache.clone()
    impl = object.__new__(FlashAttentionImpl)
    impl.scale = 8**-0.5
    endpoint = SimpleNamespace(impl=impl, num_heads=4, num_kv_heads=2, head_size=8)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, length]),
        max_query_len=length,
        seq_lens=torch.tensor([start + length]),
        max_seq_len=start + length,
        block_table=None,
    )
    positions = torch.arange(start, start + length, device="cuda")
    keys, values = (part[:, :, 0].repeat_interleave(2, dim=1) for part in cache.chunk(2, -1))
    scores = torch.einsum("mhd,nhd->hmn", q, keys) * impl.scale
    earlier = scores.masked_fill(
        torch.arange(start + length, device="cuda")[None, None, :] >= positions[None, :, None], -torch.inf
    )
    off = torch.einsum("hmn,nhd->mhd", earlier.softmax(-1).nan_to_num(), values).contiguous()
    scores[:, torch.arange(length, device="cuda"), positions] = (q * k.repeat_interleave(2, 1)).sum(-1).T * impl.scale
    probabilities = scores.masked_fill(
        torch.arange(start + length, device="cuda")[None, None, :] > positions[None, :, None], -torch.inf
    ).softmax(-1)
    diagonal = probabilities[:, torch.arange(length, device="cuda"), positions].T.clone()
    probabilities[:, torch.arange(length, device="cuda"), positions] = 0
    expected = torch.einsum("hmn,nhd->mhd", probabilities, values) + diagonal[..., None] * v.repeat_interleave(2, 1)
    with (
        patch.object(attention, "attention_metadata", return_value=(metadata, cache, None)),
        patch.object(attention, "get_forward_context"),
        patch.object(
            attention,
            "flash_attn_varlen_func",
            return_value=(off, earlier.logsumexp(-1).masked_fill(positions[None, :] == 0, float("inf"))),
        ) as fa,
    ):
        actual = attention.peek_attention(endpoint, positions, q, k, v)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(cache, before, rtol=0, atol=0)
    assert fa.call_args.kwargs["seqused_k"].tolist() == [start + length - 1]


def test_solver_modes_are_role_sensitive():
    for rounds in (1, 2, 3):
        attachment = JacobiRecirculationWorkload(rounds=rounds).compile_backend(FACTS).attachment
        assert CachePolicy.for_attachment(attachment, "isolate", 0).role_sensitive
