"""Engram-lite's CPU contracts and explicit-placement admission share the ordinary planning path."""

import json
from dataclasses import dataclass, replace

import pytest
import torch
from safetensors.torch import save_file

import pllm
from examples.engram_lite import EngramLite
from pllm.testing import ContractError, check


@dataclass(frozen=True)
class WholeTable(EngramLite):
    """A host table must be gathered before arithmetic."""

    def lookup(self, ids=pllm.input_tokens(history=2), table=pllm.parameter("memory", "table")):
        return table.sum(0).expand(len(ids), -1)


def test_explicit_placement_and_engram_contracts(tmp_path):
    path = tmp_path / "memory.safetensors"
    save_file({"table": torch.randn(1024, 2), "wk": torch.randn(4, 4), "wv": torch.randn(4, 4)}, path)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "vocab_size": 32,
                "intermediate_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
            }
        )
    )
    config = pllm.ModelConfig(
        str(tmp_path),
        max_num_seqs=2,
        max_num_batched_tokens=16,
        execution=pllm.ExecutionOptions(state_page_budget_bytes=1024),
    )
    host = EngramLite(str(path), buckets=1024, width=2, placement="host", layers=(0,))
    report = pllm.explain(host, config)
    memory = next(state for state in report.states if state.name == "memory")
    assert memory.placement == "host" and memory.host_bytes >= 4096 and memory.reserved_bytes == 0
    assert memory.transfer_bytes_per_row == 8 and memory.transfer_bytes_per_step == 16
    assert {"mixed prefill/decode", "preempted", "neutral"} <= set(
        check(host, config, neutral=replace(host, gate=0)).passed
    )
    with pytest.raises(ContractError, match=r"\[neutral\] custom residual `add`"):
        check(host, config, neutral=replace(host, gate=1.0))
    rejected = pllm.explain(replace(host, placement="device"), config)
    assert rejected.budget_errors and 'placement="host"' in rejected.budget_errors[0]
    assert next(state for state in rejected.states if state.name == "memory").placement == "device"
    device = pllm.explain(replace(host, placement="device"), replace(config, execution=pllm.ExecutionOptions()))
    assert all(state.placement == "device" and state.transfer_bytes_per_step == 0 for state in device.states)
    with pytest.raises(ValueError, match="row gathers only"):
        pllm.explain(WholeTable(str(path), buckets=1024, width=2, placement="host", layers=(0,)), config)
