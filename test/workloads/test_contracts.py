"""One CPU schedule contract table for the maintained model workloads."""

import json
from dataclasses import asdict, dataclass, replace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

import pllm
from pllm import testing
from pllm.backends.vllm import compat
from pllm.contracts.host_batch import validate_host_batch
from pllm.execution.post_sample import PostSampleOverlap
from pllm.execution.shared_tensor import SharedTensor
from pllm.execution.tensor_declaration import TensorDeclaration
from pllm.execution.trace import TraceEventKind, TraceSelection
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
    RewardHeadWorkload,
    StepLoopWorkload,
    TLMConfig,
    TLMWorkload,
)
from pllm.workloads.dola import DoLaWorkload
from pllm.workloads.gated_delta import GatedDeltaConfig, GatedDeltaWorkload
from pllm.workloads.state_producers import FactorAdapterWorkload
from pllm.workloads.video import VideoWorkload
from test.workloads.attention_workloads import (
    CacheLayout,
    CoupledAttention,
    InstanceAttention,
    Monitor,
    Quest,
    SharedAttention,
    SharedWindowAttention,
    SinkAttention,
    WindowAttention,
)
from test.workloads.kernel_workloads import Forward, Residual


class PrefillTrainer(pllm.Workload):
    """A prompt-only update must not run a second time when request state is retained."""

    @property
    def learned(self) -> pllm.RequestState:
        return pllm.RequestState({"w": lambda m: (getattr(m, "hidden_size"),)}, optimizer=pllm.AdamW(lr=0.01))

    @pllm.train(state="learned")
    def learn(self, x=pllm.hidden(0, tokens="prefill"), w=pllm.parameter("learned", "w")):
        dtype = getattr(x, "dtype", torch.float32) if hasattr(x, "dtype") else torch.float32
        return (w - x.to(dtype)).square().mean(-1)


class Steering(pllm.Workload):
    """User tensor math independent of the other sampler stages."""

    @pllm.add_logits()
    def bias(self, logits=pllm.layer_logits((-1,))):
        return logits[:, -1].tanh() * 0.1


@dataclass(frozen=True)
class Guided(pllm.Workload):
    """Guidance alone: at beta=0 it declares nothing, so it is exactly the base model."""

    strength: float = 0.5

    def beta(self) -> float:
        return self.strength

    @pllm.guide_sampling(beta=beta)
    def push(self, h=pllm.final_hidden()):
        return h


@dataclass(frozen=True)
class SelectedSteering(pllm.Workload):
    """Row masks may stay enabled when additive logits have zero strength."""

    strength: float = 0.1

    def selected(self, samples=pllm.sample_index()):
        """Select some samples independently of their packed row positions."""
        return samples % 2 == 0

    @pllm.add_logits(rows="selected")
    def bias(self, logits=pllm.layer_logits((-1,))):
        """Return zero deltas when the author disables steering."""
        return logits[:, -1].tanh() * self.strength


@dataclass(frozen=True)
class Stack(pllm.Workload):
    esamp: ESampWorkload = ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5))
    dola: DoLaWorkload = DoLaWorkload(layers=(0, 2, 4))
    steer: Steering = Steering()


@dataclass(frozen=True)
class PositionStack(pllm.Workload):
    loop: StepLoopWorkload = StepLoopWorkload()


class MeanTrainer(PrefillTrainer):
    """An explicit mean preserves independent request updates."""

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


class TokenHistory(pllm.Workload):
    """Consume every scheduled token before a neutral residual and export the sampled rows."""

    def lookup(self, ids=pllm.input_tokens(history=2)):
        return ids

    @pllm.custom_residual(site="attention", layers=(0,))
    def add(self, x, y, ids=pllm.result("lookup")):
        return x + y + (ids.sum(-1, keepdim=True) * 0).to(x.dtype)

    @pllm.export(max_rows=3, max_cols=3)
    def observe(self, ids=pllm.input_tokens(history=2)):
        return ids


class AllTokens(pllm.Workload):
    """Every prompt and decode token's residual, exported once per token across chunks and replay."""

    @pllm.export(max_rows=64, max_cols=8, overflow="block")
    def residual(self, h=pllm.residual(3, tokens="all"), w=pllm.parameter("scale", "w")):
        return h + w

    scale = pllm.InstanceState({"w": (8,)})


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


class ResultCycle(pllm.Workload):
    """Two methods read each other's results, so neither can run first."""

    def first(self, value=pllm.result("second")):
        return value

    @pllm.add_logits()
    def second(self, h=pllm.hidden(0), value=pllm.result("first")):
        return h + value


class SelectedTrainer(PrefillTrainer):
    """Standalone request training cannot select updates."""

    def selected(self, h=pllm.hidden(0)):
        return h[:, 0] > 0

    @pllm.train(state="learned", rows="selected")
    def learn(self, x=pllm.hidden(0), w=pllm.parameter("learned", "w")):
        return (w - x).square().mean(-1)


class ObservedSelectedTrainer(SelectedTrainer):
    """An unrelated export does not make standalone training selectable."""

    @pllm.export()
    def observe(self, h=pllm.hidden(0)):
        return h


class PromptMask(pllm.Workload):
    """Adaptive masks must leave prompt layers unmasked."""

    @pllm.custom_forward(tokens="all")
    def forward(self, layers):
        return layers(active=layers.embeddings[:, :1] > 0)[-1]


@pytest.mark.parametrize(
    "workload, neutral, error",
    [
        (workload, None, "")
        for workload in (
            DummyWorkload(),
            Residual(),
            Monitor(),
            Quest(),
            InstanceAttention(),
            CacheLayout(),
            WindowAttention(),
            SinkAttention(),
            SharedAttention(),
            SharedWindowAttention(),
            Forward(),
            TokenHistory(),
            AllTokens(),
            Stack(),
            PositionStack(),
            PrefillTrainer(),
            MeanTrainer(),
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
            RewardHeadWorkload(),
            TLMWorkload(TLMConfig()),
            VideoWorkload(),
        )
    ]
    + [
        (SelectedSteering(), SelectedSteering(0.0), ""),
        (Guided(), Guided(0.0), ""),
        (SelectedSteering(), SelectedSteering(), r"\[neutral\].*sampler bias.*exactly zero"),
    ]
    + [
        (workload, None, error)
        for workload, error in (
            (CoupledAttention(), r"\[isolation\]"),
            (BorrowedMutation(False), "method `change` mutates borrowed input `h`.*clone"),
            (BorrowedMutation(True), "method `change` mutates borrowed input `w`.*clone"),
            (BadLogits(), r"@pllm.add_logits.*\[rows, vocab_size\].*got \(3, 8\)"),
            (sampler_feedback(pllm.base_logits()), r"pllm.base_logits\(\).*Phase.AFTER_SAMPLE.*pllm.layer_logits"),
            (
                sampler_feedback(pllm.sampled_tokens()),
                r"pllm.sampled_tokens\(\).*Phase.AFTER_SAMPLE.*pllm.final_hidden",
            ),
            (ResultCycle(), r"(first|second) reads pllm.result\('(second|first)'\) and .*remove one of the reads"),
            (SelectedTrainer(), "rows=.*require a model effect without sampler guidance, or @pllm.add_logits"),
            (ObservedSelectedTrainer(), "rows=.*require a model effect without sampler guidance"),
            (
                PromptMask(),
                "Use @pllm.custom_forward.*decode.*active=.*approximate K/V.*every later token reads.*"
                'full model by default.*prefill.*only to change prompt computation.*over="span"',
            ),
        )
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, pllm.Workload) else "",
)
def test_schedule_contracts(workload, neutral, error, model_config):
    cuda_initialized = torch.cuda.is_initialized()
    if error:
        with pytest.raises(testing.ContractError, match=error) as caught:
            testing.check(workload, model_config, neutral=neutral)
        assert "framework bug" not in str(caught.value)
        assert torch.cuda.is_initialized() == cuda_initialized
        return
    result = testing.check(workload, model_config, neutral=neutral)
    assert torch.cuda.is_initialized() == cuda_initialized
    # Model-site and causal/host-feed workloads skip the CPU schedules with a reason; every other one runs them.
    unscheduled = (
        Residual,
        PositionStack,
        JacobiRecirculationWorkload,
        StepLoopWorkload,
        FTTTWorkload,
        RewardHeadWorkload,
        TLMWorkload,
        VideoWorkload,
    )
    assert ("row permutation" in result.passed) != isinstance(workload, unscheduled), result
    json.dumps(asdict(result))
    if neutral is not None:
        assert "neutral" in result.passed
    if isinstance(workload, Stack):
        assert "component independence" in result.passed
        report = pllm.explain(workload, model_config)
        assert {member.name for member in report.members} >= {
            "esamp.loss",
            "esamp.predict",
            "dola.contrast",
            "steer.bias",
        }
    if isinstance(workload, (ESampWorkload, PrefillTrainer, InPlaceTTTWorkload, PrefillAdaptersWorkload, ReFTWorkload)):
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
    with patch("pllm.testing_contracts.retained_rows", side_effect=lambda labeler, value, counts, script: value):
        with pytest.raises(testing.ContractError, match=r"\[preempted\].*state"):
            testing.check(workload, model_config)


@pytest.mark.parametrize("lag", (0, 1, 2))
def test_check_versions_follow_lag_and_detect_wrong_history(lag, model_config):
    workload = ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5))
    config = replace(model_config, execution=pllm.ExecutionOptions(publish_lag=lag))
    assert "publish_lag" in testing.check(workload, config).passed
    if lag == 1:
        # A deeper history than planned has no reserved pages; a shallower one reads the wrong version.
        initialize = PostSampleOverlap.__init__

        def wrong_history(self, arena, directory, lag, stream, anchor):
            initialize(self, arena, directory, 0, stream, anchor)

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
    def forward(self, layers, w=pllm.parameter("gate", "w"), b=pllm.parameter("gate", "b")):
        hidden = layers(stop=1)[0]
        hidden = layers(hidden=hidden, active=layers.generated & (layers.position % 2 == 0))[-1]
        for step in range(self.steps):
            active = torch.sigmoid(hidden.float() @ w + b)[:, None] <= 0.6
            hidden = layers(hidden=hidden, active=active, kv="peek" if step == self.steps - 1 else "commit")[-1]
        return hidden


class MoR(pllm.Workload):
    """A module router selects each token's recursion depth."""

    @property
    def router(self) -> pllm.InstanceState:
        return pllm.InstanceState(
            init=pllm.FromModule(lambda m: torch.nn.Linear(m.hidden_size, 1)), dtype=torch.float32
        )

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
@pytest.mark.parametrize("tp", (1, 2))
def test_custom_forward_state_check_and_explain(workload, state, model_config, tp):
    model_config = replace(model_config, tensor_parallel_size=tp)
    if tp > 1 and not compat.VLLM_TP_CAPABLE:
        with pytest.raises(testing.ContractError, match="vLLM 0.32.0"):
            testing.check(workload, model_config)
        return
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
    assert report.tensor_parallel_size == tp
    assert "CUDA conditional nodes" in str(report) and "0 blocking host syncs" in str(report)
    member = next(member for member in report.members if member.name == "forward")
    assert state in member.tasks[0].reads


@dataclass(frozen=True)
class ImportedRows(pllm.Workload):
    """Expose imported rows directly, so the checker can observe their routing."""

    request: bool = True

    @property
    def steer(self) -> pllm.InstanceState:
        fields = {"vector": pllm.Field(lambda width, _: (width,))}
        return (
            pllm.RequestState(fields, init=pllm.Imported())
            if self.request
            else pllm.InstanceState(fields, init=pllm.Imported())
        )

    @pllm.export(name="imported", every=1, max_rows=3, max_cols=8)
    def observe(self, hidden=pllm.hidden(0), vector=pllm.parameter("steer", "vector", per_row=True)):
        return vector + hidden * 0


class ImportedTrainer(ImportedRows):
    """Training one state must read another state's import on the same step."""

    @property
    def learned(self) -> pllm.RequestState:
        return pllm.RequestState({"w": (8,)}, optimizer=pllm.AdamW(lr=0.01))

    @pllm.train(state="learned")
    def learn(self, hidden=pllm.hidden(0), w=pllm.parameter("learned", "w"), vector=pllm.parameter("steer", "vector")):
        return (w - vector + hidden * 0).square().mean(-1)


@pytest.mark.parametrize("workload", (ImportedRows(), ImportedTrainer()))
def test_check_detects_imports_routed_by_sorted_request_id(workload, model_config):
    assert "imported rows / request isolation" in testing.check(workload, model_config).passed
    prepare = TensorDeclaration.prepare_import

    def sorted_requests(self, update, batch, data):
        # 499e3a26 class: request-id order is not packed-row order.
        rows = dict(zip(sorted(data.step.import_rows), data.step.import_rows.values(), strict=True))
        data.step = replace(data.step, import_rows=rows)
        prepare(self, update, batch, data)

    with patch.object(TensorDeclaration, "prepare_import", sorted_requests):
        with pytest.raises(testing.ContractError, match="misrouted rows"):
            testing.check(workload, model_config)


@pytest.mark.parametrize("per_request", (True, False))
def test_import_shape_boundary_and_first_export(per_request):
    workload = ImportedRows(per_request)
    tap = testing.hidden_binding(0, layer_count=1)
    facts = testing.backend_facts(8, (tap,), layer_count=1, memory_bytes=1 << 20)
    declaration = workload.build(facts)
    values = torch.arange(16 if per_request else 8, dtype=torch.float32).reshape((2, 8) if per_request else (8,))
    for tensor, error in (
        (values, None),
        (torch.zeros(3, 8), "one row per prompt" if per_request else "field tensors themselves"),
    ):

        def validate():
            validate_host_batch(
                pllm.io.StateImport((tensor,)),
                declaration.graph,
                declaration.layouts,
                facts.timeline,
                SharedTensor.describe,
                hidden_size=8,
                vocab_size=0,
                prompt_count=2,
                import_fields=declaration.effects.import_fields,
            )

        if error is None:
            validate()
        else:
            with pytest.raises(ValueError, match=error):
                validate()
    execution = testing.compile_execution(workload, facts)
    identities = (testing.RequestIdentity("second", 91, 0), testing.RequestIdentity("first", 17, 0))
    execution.admit(identities)
    execution.trace.configure(TraceSelection(tensors=True, max_tensor_bytes=128))
    try:
        testing.run_step(
            execution, testing.step_input(identities, {tap: torch.zeros(2, 8)}, batch=pllm.io.StateImport((values,)))
        )
        event = next(
            event
            for event in execution.trace.snapshot().events
            if event.kind is TraceEventKind.TENSOR and event.tensor_name == "imported"
        )
        torch.testing.assert_close(event.tensor(), values if per_request else values.expand(2, -1))
    finally:
        execution.retire(identities, wait=True)
        execution.retire((), wait=True)
        execution.close()


def test_check_detects_chunk_local_token_history(model_config):
    from pllm import testing_contracts

    step = testing_contracts.contract_step_input

    def shifted(*args, **kwargs):
        value = step(*args, **kwargs)
        if value.rows.total_rows > len(value.rows.request_rows) and any(
            row.token_offset > 0 for row in value.rows.rows
        ):
            value.input_tokens[:, 1:] = value.input_tokens[:, :-1].clone()
        return value

    with (
        patch.object(testing_contracts, "contract_step_input", shifted),
        pytest.raises(testing.ContractError, match="mixed"),
    ):
        testing.check(TokenHistory(), model_config)


@pytest.mark.parametrize(
    "field", ("targets", "input_ids", "base_logits"), ids=("weighted-row", "causal", "weighted-ce")
)
def test_host_batch_rejects_reduction_at_declaration(field) -> None:
    """Every host-batch input schema owns its reduction, before tracing or execution."""

    class HostTrainer(pllm.Workload):
        """A scalar host loss cannot delegate its reduction to the request-row framework."""

        learned = pllm.InstanceState(
            {"w": (4,)}, optimizer=pllm.SGD(0.01) if field == "base_logits" else pllm.AdamW(0.01)
        )

        @pllm.train(state="learned", reduction="sum")
        def learn(self, values=pllm.batch(field), w=pllm.parameter("learned", "w")):
            """Return a scalar loss owned by the batch method."""
            return values.sum() * w.sum()

    facts = testing.backend_facts(4)
    with pytest.raises(
        ValueError, match="host-batch training returns one scalar and owns its reduction; remove reduction="
    ):
        HostTrainer().build(facts)
