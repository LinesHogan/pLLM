"""One cold HF/PEFT and objective boundary for canonical TLM."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from tllm.canonical.kernels.tlm_objective import (
    tlm_weighted_selection_objective,
)
from tllm.canonical.workloads.tlm import HfPeftQvLoader, HfPeftQvLoaderConfig
from tllm.contracts.parameter_overlay import ParameterRole
from tllm.contracts.tensor_layout import TensorDType


def test_tlm_cold_qv_facts_and_weighted_selection_match_oracle() -> None:
    pytest.importorskip("peft")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    loader_config = HfPeftQvLoaderConfig(
        "unused-local-model",
        "main",
        True,
        False,
        2,
        4,
        0.0,
        TensorDType.FLOAT32,
    )
    specialization = HfPeftQvLoader.inspect(
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        ),
        loader_config,
    )
    assert tuple(target.target.role for target in specialization.targets) == (
        ParameterRole.QUERY_PROJECTION_WEIGHT,
        ParameterRole.VALUE_PROJECTION_WEIGHT,
    )
    assert tuple(
        (target.input_size, target.output_size)
        for target in specialization.targets
    ) == ((16, 16), (16, 16))
    assert specialization.rank == 2
    assert specialization.alpha == 4

    logits = torch.tensor(
        (
            ((3.0, 0.0), (0.0, 3.0), (3.0, 0.0)),
            ((0.0, 3.0), (3.0, 0.0), (0.0, 3.0)),
        )
    )
    labels = torch.tensor(((0, 1, 0), (1, 1, -100)))
    result = tlm_weighted_selection_objective(
        logits,
        labels,
        selection_logits=None,
        threshold=0.2,
        lamb=0.5,
        ignore_index=-100,
    )
    rows = torch.stack(
        (
            F.cross_entropy(logits[0, :-1], labels[0, 1:]),
            F.cross_entropy(logits[1, :1], labels[1, 1:2]),
        )
    )
    selected = rows > 0.2
    expected = (
        rows * (0.5 * torch.exp(rows.detach() - 0.2)) * selected
    ).sum() / selected.sum()
    torch.testing.assert_close(result.loss, expected)
    torch.testing.assert_close(result.selected, selected)
