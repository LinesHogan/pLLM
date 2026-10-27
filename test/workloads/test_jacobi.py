"""Capture boundaries, span tensor semantics and read-only diagonal attention evidence."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

import pllm
from pllm.backends.vllm import region_attention as attention
from pllm.backends.vllm.prefix_cache import CachePolicy
from pllm.execution.route_executor import execute_route
from pllm.execution.token_program import TokenProgram, TokenTensor
from pllm.workloads.recirculation import JacobiRecirculation
from test.workloads.test_recirculation import FACTS


@pytest.mark.parametrize("tokens", ("all", "decode", "prefill"))
def test_overlapping_routes_rejected(tokens):
    class Overlap(JacobiRecirculation):
        @pllm.layer_route(tokens=tokens)
        def extra(self, layers):
            return layers()[-1]

    with pytest.raises(ValueError, match="tokens overlap:.*" + ("decode" if tokens == "all" else tokens)):
        Overlap().declare(FACTS)


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
        TokenProgram.capture(route, TokenTensor((1, 4), torch.float32, "cpu"), 3, ("state",), over)


@pytest.mark.parametrize("width", (1, 7))
def test_span_cells_keep_packed_tensor_semantics(width):
    def route(layers):
        value = layers.seed - layers.seed.mean(0, keepdim=True) + layers.position + layers.generated
        shape = layers.seed.shape
        layers.save("state", value.reshape(shape))
        return layers(hidden=value.reshape(shape))[-1]

    program = TokenProgram.capture(route, TokenTensor((1, 4), torch.float32, "cpu"), 1, ("state",), "span")
    seed = torch.arange(width * 4, dtype=torch.float32).reshape(width, 4)
    output, (saved,) = execute_route(
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


@pytest.mark.parametrize("start,length", ((0, 1), (0, 7), (3, 5), (16, 4)))
def test_peek_dense_reference_and_unchanged_cache(start, length):
    torch.manual_seed(7)
    q, k, v = torch.randn(length, 4, 8), torch.randn(length, 2, 8), torch.randn(length, 2, 8)
    cache = torch.randn(start + length, 2, 8, 16)
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
    positions = torch.arange(start, start + length)
    keys, values = (part[:, :, 0].repeat_interleave(2, dim=1) for part in cache.chunk(2, -1))
    scores = torch.einsum("mhd,nhd->hmn", q, keys) * impl.scale
    earlier = scores.masked_fill(torch.arange(start + length)[None, None, :] >= positions[None, :, None], -torch.inf)
    off = torch.einsum("hmn,nhd->mhd", earlier.softmax(-1).nan_to_num(), values)
    scores[:, torch.arange(length), positions] = (q * k.repeat_interleave(2, 1)).sum(-1).T * impl.scale
    probabilities = scores.masked_fill(
        torch.arange(start + length)[None, None, :] > positions[None, :, None], -torch.inf
    ).softmax(-1)
    diagonal = probabilities[:, torch.arange(length), positions].T.clone()
    probabilities[:, torch.arange(length), positions] = 0
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


def test_solver_modes_change_cache_identity():
    policies = [
        CachePolicy.for_attachment(JacobiRecirculation(rounds=r).compile_backend(FACTS).attachment, "isolate")
        for r in (1, 2, 3)
    ]
    assert len({policy.namespace for policy in policies}) == 3 and all(policy.role_sensitive for policy in policies)
