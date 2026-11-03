"""One CPU schedule contract table for the maintained model workloads."""

import json
from dataclasses import asdict, dataclass, replace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

import pllm
from pllm import testing
from pllm.execution.post_sample import PostSampleOverlap
from pllm.workloads import (
    DecodeFactorsWorkload,
    DummyWorkload,
    ESampConfig,
    ESampWorkload,
    FTTTWorkload,
    InPlaceTTTConfig,
    InPlaceTTTWorkload,
    JacobiRecirculationWorkload,
    OutputLoRAWorkload,
    OverRIDEConfig,
    OverRIDEWorkload,
    PrefillAdaptersWorkload,
    PromptFactorsWorkload,
    PromptMixtureWorkload,
    RecirculationWorkload,
    ReFTConfig,
    ReFTWorkload,
    StepLoopWorkload,
    TLMConfig,
    TLMWorkload,
    TTRLWorkload,
)
from pllm.workloads.dola import DoLaWorkload
from pllm.workloads.gated_delta import GatedDeltaConfig, GatedDeltaWorkload
from pllm.workloads.state_producers import FactorAdapterWorkload
from pllm.workloads.video import VideoWorkload


class PrefillTrainer(pllm.Workload):
    """A prompt-only update must not run a second time when request state is retained."""

    @property
    def learned(self) -> pllm.RequestState:
        return pllm.RequestState({"w": (8,)}, optimizer=pllm.AdamW(lr=0.01))

    @pllm.train(state="learned")
    def learn(self, x=pllm.hidden(0, tokens="prefill"), w=pllm.parameter("learned", "w")):
        return (w - x).square().mean(-1)


class Steering(pllm.Workload):
    """User tensor math independent of the other sampler stages."""

    @pllm.add_logits()
    def bias(self, logits=pllm.layer_logits((-1,))):
        return logits[:, -1].tanh() * 0.1


@dataclass(frozen=True)
class Stack(pllm.Workload):
    esamp: ESampWorkload = ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5))
    dola: DoLaWorkload = DoLaWorkload(layers=(0, 2, 4))
    steer: Steering = Steering()


@dataclass(frozen=True)
class PositionStack(pllm.Workload):
    loop: StepLoopWorkload = StepLoopWorkload()


class BatchMeanTrainer(PrefillTrainer):
    """An explicit batch mean breaks independent request updates."""

    @pllm.train(state="learned", reduction="mean")
    def learn(self, x=pllm.hidden(0), w=pllm.parameter("learned", "w")):
        return (w - x).square().mean(-1)


@dataclass(frozen=True)
class BorrowedMutation(pllm.Workload):
    """A view write is invalid for either borrowed input kind."""

    parameter: bool
    head = pllm.InstanceState({"w": (8,)})

    @pllm.export(max_rows=3)
    def change(self, h=pllm.hidden(0), w=pllm.parameter("head", "w")):
        target = w if self.parameter else h
        target.view_as(target).add_(1)
        return h + w


class BadLogits(pllm.Workload):
    """Hidden width is not vocabulary width."""

    @pllm.add_logits()
    def logits(self, h=pllm.hidden(0)):
        return h


def sampler_feedback(source):
    """Feedback is unavailable even when a sampler reads it through a helper."""

    class Feedback(pllm.Workload):
        @pllm.on(pllm.Phase.AFTER_SAMPLE)
        def feedback(self, value=source):
            return value

        @pllm.add_logits()
        def logits(self, h=pllm.hidden(0), value=pllm.result("feedback")):
            return h

    return Feedback()


class SelectedTrainer(PrefillTrainer):
    """Standalone request training cannot select updates."""

    def selected(self, h=pllm.hidden(0)):
        return h[:, 0] > 0

    @pllm.train(state="learned", rows="selected")
    def learn(self, x=pllm.hidden(0), w=pllm.parameter("learned", "w")):
        return (w - x).square().mean(-1)


class PromptMask(pllm.Workload):
    """Adaptive masks must leave prompt layers unmasked."""

    @pllm.custom_forward(tokens="all")
    def forward(self, layers):
        return layers(active=layers.embeddings[:, :1] > 0)[-1]


@pytest.mark.parametrize(
    "workload, error",
    [
        (workload, "")
        for workload in (
            DummyWorkload(),
            Stack(),
            PositionStack(),
            PrefillTrainer(),
            DoLaWorkload(layers=(0, 2, 4, 6, 8)),
            ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5)),
            FactorAdapterWorkload(),
            PromptFactorsWorkload(),
            PromptMixtureWorkload(),
            DecodeFactorsWorkload(),
            OutputLoRAWorkload(),
            OverRIDEWorkload(OverRIDEConfig(rank=2, num_iterations=3)),
            PrefillAdaptersWorkload(rank=2, chunk_size=4),
            GatedDeltaWorkload(GatedDeltaConfig((pllm.model.Projection("down_proj", 0),), head_width=4, chunk_size=4)),
            InPlaceTTTWorkload(
                InPlaceTTTConfig((pllm.model.Projection("down_proj", 0),), 4, 0.01), dtype=torch.bfloat16
            ),
            InPlaceTTTWorkload(InPlaceTTTConfig((pllm.model.Projection("down_proj", 0),), 4, 0.3, track="T2")),
            ReFTWorkload(ReFTConfig((0,))),
            RecirculationWorkload(),
            JacobiRecirculationWorkload(),
            StepLoopWorkload(),
            FTTTWorkload(),
            TTRLWorkload(),
            TLMWorkload(TLMConfig()),
            VideoWorkload(),
        )
    ]
    + [
        (BatchMeanTrainer(), 'method `learn`.*couples RequestState.*reduction="state_mean"'),
        (BorrowedMutation(False), "method `change` mutates borrowed input `h`.*clone"),
        (BorrowedMutation(True), "method `change` mutates borrowed input `w`.*clone"),
        (BadLogits(), r"@pllm.add_logits.*\[rows, vocab_size\].*got \(3, 8\)"),
        (sampler_feedback(pllm.base_logits()), r"pllm.base_logits\(\).*Phase.AFTER_SAMPLE.*pllm.layer_logits"),
        (sampler_feedback(pllm.sampled_tokens()), r"pllm.sampled_tokens\(\).*Phase.AFTER_SAMPLE.*pllm.final_hidden"),
        (SelectedTrainer(), "rows=.*require a model effect without sampler guidance, or @pllm.add_logits"),
        (PromptMask(), "active=.*only in @pllm.custom_forward.*decode.*separate unmasked"),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, pllm.Workload) else "",
)
def test_schedule_contracts(workload, error, model_config):
    cuda_initialized = torch.cuda.is_initialized()
    if error:
        with pytest.raises(testing.ContractError, match=error) as caught:
            testing.check(workload, model_config)
        assert "framework bug" not in str(caught.value)
        assert torch.cuda.is_initialized() == cuda_initialized
        return
    result = testing.check(workload, model_config)
    assert torch.cuda.is_initialized() == cuda_initialized
    assert result.passed or result.skipped
    json.dumps(asdict(result))
    if isinstance(workload, Stack):
        assert "component independence" in result.passed
        report = pllm.explain(workload, model_config)
        assert {member.name for member in report.members} >= {
            "esamp.loss",
            "esamp.predict",
            "dola.contrast",
            "steer.bias",
        }
    if isinstance(workload, (ESampWorkload, PrefillTrainer, InPlaceTTTWorkload, PrefillAdaptersWorkload)):
        assert "preempted" in result.passed
    if isinstance(workload, (InPlaceTTTWorkload, PrefillAdaptersWorkload)):
        assert "visibility=exact (chunking / replay)" in result.passed


@pytest.fixture
def model_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 8,
                "num_hidden_layers": 10,
                "vocab_size": 32,
                "intermediate_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
            }
        )
    )
    save_file(
        {
            "model.layers.0.mlp.ttt_conv.weight": torch.ones(8, 1, 5) / 5,
            "model.layers.0.mlp.ttt_proj.weight": torch.eye(8),
        },
        tmp_path / "model.safetensors",
    )
    return pllm.ModelConfig(str(tmp_path))


@pytest.mark.parametrize("workload", (PrefillTrainer(), ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5))))
def test_check_detects_duplicate_updates_on_split_replay(workload, model_config):
    with patch("pllm.testing_contracts.retained_rows", side_effect=lambda rows, frontiers: rows):
        with pytest.raises(testing.ContractError, match=r"\[preempted\].*state"):
            testing.check(workload, model_config)


@pytest.mark.parametrize("lag", (0, 1, 2))
def test_check_versions_follow_lag_and_detect_wrong_history(lag, model_config):
    workload = ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5))
    config = replace(model_config, execution=pllm.ExecutionOptions(publish_lag=lag))
    assert "publish_lag" in testing.check(workload, config).passed
    if lag == 1:
        initialize = PostSampleOverlap.__init__
        for wrong_lag in (0, 2):

            def wrong_history(self, arena, directory, lag, stream, anchor):
                initialize(self, arena, directory, wrong_lag, stream, anchor)

            with patch.object(PostSampleOverlap, "__init__", wrong_history):
                with pytest.raises(testing.ContractError, match=r"\[publish_lag / visibility\].*read version"):
                    testing.check(workload, config)


def test_check_reports_unsupported_replay_with_actionable_visibility(model_config):
    workload = GatedDeltaWorkload(
        GatedDeltaConfig(
            (pllm.model.Projection("down_proj", 0),), head_width=4, chunk_size=4, visibility="forward_pinned"
        )
    )
    result = testing.check(workload, model_config)
    skipped = {item.property: item.reason for item in result.skipped}
    assert 'visibility="exact"' in skipped["preempted"]
    assert "GPU" in skipped["visibility=forward_pinned"]
    assert "preempted" not in result.passed


class LoopedDepth(pllm.Workload):
    """A learned scalar exit gate drives adaptive decoder slices."""

    steps = 2

    @property
    def gate(self) -> pllm.InstanceState:
        return pllm.InstanceState({"w": lambda m: (m.hidden_size,), "b": (1,)}, dtype=torch.float32)

    @pllm.custom_forward(tokens="decode")
    def forward(self, layers, w=pllm.parameter("gate", "w"), b=pllm.parameter("gate", "b"), steps=pllm.Attr("steps")):
        hidden = layers(stop=1)[0]
        hidden = layers(hidden=hidden, active=layers.generated & (layers.position % 2 == 0))[-1]
        for step in range(steps):
            active = torch.sigmoid(hidden.float() @ w + b)[:, None] <= 0.6
            hidden = layers(hidden=hidden, active=active, kv="peek" if step == steps - 1 else "commit")[-1]
        return hidden


class MoR(pllm.Workload):
    """A module router selects each token's recursion depth."""

    @property
    def router(self) -> pllm.InstanceState:
        return pllm.InstanceState(module=lambda m: torch.nn.Linear(m.hidden_size, 1), dtype=torch.float32)

    @pllm.custom_forward(tokens="prefill")
    def prompt(self, layers, router=pllm.module("router")):
        hidden = layers()[-1]
        return (hidden.float() * (1 + router(hidden.float()).sigmoid() * 0.01)).to(hidden.dtype)

    @pllm.custom_forward(tokens="decode")
    def forward(self, layers, router=pllm.module("router")):
        hidden = layers()[-1]
        depth = 1 + (torch.sigmoid(router(hidden.float())) * 2).round()
        for step in range(3):
            hidden = layers(hidden=hidden, active=depth > step)[-1]
        return hidden


@pytest.mark.parametrize("workload,state", ((LoopedDepth(), "gate"), (MoR(), "router")))
def test_custom_forward_state_check_and_explain(workload, state, model_config):
    result = testing.check(workload, model_config)
    assert {
        "custom-forward instance reads / CPU decoder",
        "adaptive row permutation / isolation",
        "adaptive KV policy",
        "preempted",
        "never exit bitwise",
    } <= set(result.passed)
    report = pllm.explain(workload, model_config)
    assert report.recompute_rejection is None
    assert report.preemption == "reset"
    assert "CUDA conditional nodes" in str(report) and "0 blocking host syncs" in str(report)
    member = next(member for member in report.members if member.name == "forward")
    assert state in member.tasks[0].reads
