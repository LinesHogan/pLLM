"""Adaptive decode contracts: mask validation, skipped KV, neutral execution and schedule independence."""

from dataclasses import dataclass

import pytest
import torch

import pllm
from pllm.execution.token_program import Segment, TokenProgram, TokenTensor
from pllm.testing import check
from pllm.testing_depth import check_depth, check_never_exit


@dataclass(frozen=True)
class Exit(pllm.Workload):
    @pllm.custom_forward(tokens="decode")
    def forward(self, layers):
        middle = layers.layer_count // 2
        hidden = layers(0, middle)[middle - 1]
        active = (hidden[:, :1] > 0) & (layers.position % 2 == 0)
        hidden = layers(middle, hidden=hidden, active=active)[-1]
        # A later skipped commit must retain the first pass's KV, including propagated slots.
        return layers(middle, hidden=hidden, active=~active)[-1]


def test_adaptive_contracts_and_explanation():
    config = pllm.ModelConfig(model="Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=3)
    result = check(Exit(), config)
    assert "never exit bitwise" in result.passed
    assert "adaptive KV policy" in result.passed
    report = pllm.explain(Exit(), config)
    assert "CUDA conditional nodes" in str(report)
    assert "0 blocking host syncs" in str(report)
    assert "adaptive depth" in report.recompute_rejection


@pytest.mark.parametrize("rule", ("all", "none", "mixed", "cosine"))
def test_masked_oracle_and_compaction(rule):
    def forward(layers):
        hidden = layers(0, 1)[0]
        active = {
            "all": torch.ones_like(hidden[:, :1], dtype=torch.bool),
            "none": torch.zeros_like(hidden[:, :1], dtype=torch.bool),
            "mixed": hidden[:, :1] > 0,
            "cosine": torch.nn.functional.cosine_similarity(
                hidden.float(), layers.embeddings.float(), dim=-1
            ).unsqueeze(-1)
            < 0.99,
        }[rule]
        return layers(1, hidden=hidden, active=active)[-1]

    program = TokenProgram.trace(forward, TokenTensor((1, 4), torch.float32, "cpu"), 3)
    check_depth(program, 3, 4, torch.float32)
    check_never_exit(program, 3, 4, torch.float32)
    assert not program.prompt_pass_is_identity


@pytest.mark.parametrize("kind", ("dtype", "shape", "span"))
def test_active_validation(kind):
    def forward(layers):
        mask = torch.ones_like(layers.embeddings[:, :1], dtype=torch.bool)
        mask = mask.float() if kind == "dtype" else mask.flatten() if kind == "shape" else mask
        return layers(active=mask)[-1]

    with pytest.raises(ValueError, match="active"):
        TokenProgram.trace(
            forward, TokenTensor((1, 4), torch.float32, "cpu"), 3, over="span" if kind == "span" else "token"
        )


def test_propagation_is_per_layer_and_peek_keeps_kv():
    def forward(layers):
        hidden = layers(0, 2)[1]
        active = hidden[:, :1] > 0
        hidden = layers(1, hidden=hidden, active=active)[-1]
        return layers(hidden=hidden, active=active, kv="peek")[-1]

    program = TokenProgram.trace(forward, TokenTensor((1, 4), torch.float32, "cpu"), 3)
    assert [step.propagate for step in program.steps if isinstance(step, Segment)] == [(), (2,), ()]
    check_depth(program, 3, 4, torch.float32)
