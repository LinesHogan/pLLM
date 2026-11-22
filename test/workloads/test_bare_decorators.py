"""Bare decorator parity and route-keyed CPU contract checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

import pllm
from pllm import testing


@pytest.fixture
def cpu_model_config(tmp_path: Path) -> pllm.ModelConfig:
    """Minimal HF config directory so explain() and testing.check() run on CPU without weights."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "vocab_size": 32,
                "intermediate_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
            }
        )
    )
    return pllm.ModelConfig(str(tmp_path), max_num_seqs=3, max_num_batched_tokens=16)


@dataclass(frozen=True)
class BareAddLogitsExport(pllm.Workload):
    @pllm.add_logits
    def bias(self, hidden: torch.Tensor = pllm.final_hidden()) -> torch.Tensor:
        return hidden[:, :1].float().expand(-1, 32)

    @pllm.export
    def tap(self, hidden: torch.Tensor = pllm.final_hidden()) -> torch.Tensor:
        return hidden


@dataclass(frozen=True)
class CalledAddLogitsExport(pllm.Workload):
    @pllm.add_logits()
    def bias(self, hidden: torch.Tensor = pllm.final_hidden()) -> torch.Tensor:
        return hidden[:, :1].float().expand(-1, 32)

    @pllm.export()
    def tap(self, hidden: torch.Tensor = pllm.final_hidden()) -> torch.Tensor:
        return hidden


@dataclass(frozen=True)
class BareCustomForward(pllm.Workload):
    @pllm.custom_forward
    def loop(self, layers: pllm.model.Layers) -> torch.Tensor:
        return layers()[-1]


@dataclass(frozen=True)
class CalledCustomForward(pllm.Workload):
    @pllm.custom_forward()
    def loop(self, layers: pllm.model.Layers) -> torch.Tensor:
        return layers()[-1]


@dataclass(frozen=True)
class BareDifferentiateWorkload(pllm.Workload):
    weight = pllm.InstanceState({"w": (8, 8)}, optimizer=pllm.AdamW(1e-2))

    @pllm.export
    @pllm.differentiate
    def project(
        self,
        hidden: torch.Tensor = pllm.final_hidden(),
        weight: torch.Tensor = pllm.parameter("weight", "w"),
    ) -> torch.Tensor:
        return hidden @ weight

    @pllm.train(state="weight")
    def loss(
        self,
        hidden: torch.Tensor = pllm.final_hidden(),
        weight: torch.Tensor = pllm.parameter("weight", "w"),
    ) -> torch.Tensor:
        return (hidden @ weight).square().mean(dim=-1)


@dataclass(frozen=True)
class CalledDifferentiateWorkload(pllm.Workload):
    weight = pllm.InstanceState({"w": (8, 8)}, optimizer=pllm.AdamW(1e-2))

    @pllm.export()
    @pllm.differentiate()
    def project(
        self,
        hidden: torch.Tensor = pllm.final_hidden(),
        weight: torch.Tensor = pllm.parameter("weight", "w"),
    ) -> torch.Tensor:
        return hidden @ weight

    @pllm.train(state="weight")
    def loss(
        self,
        hidden: torch.Tensor = pllm.final_hidden(),
        weight: torch.Tensor = pllm.parameter("weight", "w"),
    ) -> torch.Tensor:
        return (hidden @ weight).square().mean(dim=-1)


@dataclass(frozen=True)
class RouteKeyedWorkload(pllm.Workload):
    session = pllm.RequestState({"bias": (8,)}, key_fields=("route",))

    @pllm.export
    def read_route(
        self,
        hidden: torch.Tensor = pllm.final_hidden(),
        bias: torch.Tensor = pllm.parameter("session", "bias"),
    ) -> torch.Tensor:
        return hidden.float() + bias.float()


@pytest.mark.parametrize(
    ("bare", "called"),
    (
        (BareAddLogitsExport(), CalledAddLogitsExport()),
        (BareCustomForward(), CalledCustomForward()),
        (BareDifferentiateWorkload(), CalledDifferentiateWorkload()),
    ),
)
def test_bare_decorators_match_called_forms(
    bare: pllm.Workload, called: pllm.Workload, cpu_model_config: pllm.ModelConfig
) -> None:
    assert bare.backend_requirements() == called.backend_requirements()
    assert pllm.explain(bare, cpu_model_config) == pllm.explain(called, cpu_model_config)
    assert testing.check(bare, cpu_model_config) == testing.check(called, cpu_model_config)


def test_bare_differentiate_marker_parity() -> None:
    def bare(self: object) -> None:
        return None

    def called(self: object) -> None:
        return None

    assert vars(pllm.differentiate(bare)) == vars(pllm.differentiate()(called))


def test_cpu_check_supports_route_key_fields(cpu_model_config: pllm.ModelConfig) -> None:
    result = testing.check(RouteKeyedWorkload(), cpu_model_config)
    assert "row permutation" in result.passed
    assert "request isolation" in result.passed


@dataclass(frozen=True)
class _Settings:
    limit: int = 3


@dataclass(frozen=True)
class _ConfiguredSamples(pllm.Workload):
    config: _Settings = _Settings()

    @pllm.export
    def sample(self, index=pllm.sample_index(limit=pllm.cfg("config.limit"))):
        return index


def test_cfg_resolves_on_the_bound_workload() -> None:
    for limit in (2, 5):
        workload = _ConfiguredSamples(_Settings(limit))
        assert workload.backend_requirements().sample_limit == limit
        assert pllm.cfg("config")(workload) is workload.config


@pytest.mark.parametrize("path", ("", ".config", "config.", "config[0]", "config..limit"))
def test_cfg_rejects_invalid_paths(path: str) -> None:
    with pytest.raises(ValueError, match="dotted Python attribute path"):
        pllm.cfg(path)
