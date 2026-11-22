"""CPU checks of model structure, cache eligibility, causal arithmetic and rank-local overlay fields."""

import json
from dataclasses import dataclass, replace

import pytest
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

import pllm
from pllm import testing
from pllm.backends.vllm import compat
from pllm.backends.vllm.architecture import hidden_binding, post_attention_norm_name
from pllm.backends.vllm.model_shape import ModelShape
from pllm.backends.vllm.training import causal_attention
from pllm.contracts.hidden_locator import HiddenSite
from pllm.contracts.parameter_overlay import Projection, ProjectionShape
from pllm.execution.causal_batch import LMHead
from pllm.workloads.video import VideoWorkload
from test.tp_overlays import ProjectionAdapter, SharedProjectionAdapter
from test.workloads.attention_workloads import Monitor, WindowAttention
from test.workloads.kernel_workloads import Residual


def config(path, family="Qwen2", model_type="qwen2", **settings):
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": [family + "ForCausalLM"],
                "model_type": model_type,
                "hidden_size": 8,
                "head_dim": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_hidden_layers": 3,
                "intermediate_size": 16,
                "vocab_size": 32,
                **settings,
            }
        )
    )
    return pllm.ModelConfig(str(path), max_num_seqs=2, max_num_batched_tokens=16, max_model_len=16)


@dataclass(frozen=True)
class TappedAdapter(SharedProjectionAdapter):
    @pllm.export(max_rows=2, max_cols=8)
    def tap(self, hidden=pllm.residual(0, after="attention")):
        return hidden


@dataclass(frozen=True)
class ExternalAdapter(SharedProjectionAdapter):
    path: str = ""

    @property
    def weights(self) -> pllm.InstanceState:
        return replace(super().weights, init=pllm.Pretrained(self.path) if self.path else pllm.Imported())


@pytest.mark.parametrize(
    "family,model_type",
    [
        ("Mistral", "mistral"),
        ("Phi3", "phi3"),
        ("Glm4", "glm4"),
        ("Gemma2", "gemma2"),
        ("Gemma3", "gemma3_text"),
    ],
)
def test_dense_model_shapes_and_residual_boundary(tmp_path, family, model_type):
    model = config(tmp_path, family, model_type, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    report = pllm.explain(TappedAdapter(), model)
    assert report.model.architecture == family + "ForCausalLM"
    assert report.model.base_forward_flops_per_token_estimate > 0
    expected = "pre_feedforward_layernorm" if family.startswith("Gemma") else "post_attention_layernorm"
    assert report.taps[0].module == f"model.layers.0.{expected}"
    assert report.taps[0].output == 1 and report.taps[0].members == ("tap",)
    assert hidden_binding(
        0, HiddenSite.ATTENTION_RESIDUAL, 3, post_attention_norm_name(report.model.architecture)
    ).tap == (report.taps[0].module, 1)


@pytest.mark.parametrize(
    "family,model_type", [("Mixtral", "mixtral"), ("Qwen2Moe", "qwen2_moe"), ("Qwen3Moe", "qwen3_moe")]
)
def test_moe_attention_projections_and_routed_layers(tmp_path, family, model_type):
    model = config(tmp_path, family, model_type, num_experts=4, num_local_experts=4, mlp_only_layers=[1])
    targets = tuple(Projection(name, 0) for name in ("q_proj", "k_proj", "v_proj"))
    for target, width in zip(targets, (8, 4, 4), strict=True):
        report = pllm.explain(SharedProjectionAdapter(targets=(target,)), model)
        assert report.model.targets[0].output_size == width
    for name in ("gate_proj", "up_proj", "down_proj"):
        with pytest.raises(ValueError, match="MLP overlays need a dense MLP"):
            ModelShape.of_config(model, (Projection(name, 0),))
    if family != "Mixtral":
        assert ModelShape.of_config(model, (Projection("up_proj", 1),)).targets[0].output_size == 16


def test_prefix_cache_state_reads_and_graph_eligibility(tmp_path):
    model = replace(config(tmp_path), cuda_graphs=True, enable_prefix_caching=True)
    report = pllm.explain(TappedAdapter(), model)
    assert not report.eager_reason
    assert next(member for member in report.members if member.effect == "Adapter").eager_reason is None
    assert not pllm.explain(Residual(), model).eager_reason
    for workload in (ProjectionAdapter(), ExternalAdapter()):
        with pytest.raises(ValueError, match="disable prefix caching.*weights"):
            pllm.explain(workload, model)
    assert not pllm.explain(ExternalAdapter(), replace(model, enable_prefix_caching=False)).eager_reason


class LocalAttention(pllm.Workload):
    @pllm.custom_attention(layers=(0,))
    def apply(self, q, kv):
        return kv.attend(q)[0]


class ValueRead(pllm.Workload):
    @pllm.custom_residual(site="mlp", layers=(1,))
    def apply(self, x, y, value=pllm.value(0)):
        return x + y + value.sum() * 0


def test_tp_requirements_and_rank_local_heads(tmp_path, monkeypatch):
    monkeypatch.setattr(compat, "VLLM_TP_CAPABLE", True)
    model = replace(config(tmp_path), tensor_parallel_size=2)
    for workload in (Residual(), LocalAttention(), VideoWorkload()):
        report = pllm.explain(workload, model)
        assert (report.model.attention_heads, report.model.kv_heads) == (1, 1)
    for workload in (ValueRead(), Monitor(), WindowAttention()):
        with pytest.raises(ValueError, match="tensor_parallel_size=1"):
            pllm.explain(workload, model)


@pytest.mark.parametrize("name,axis", [("q_proj", 0), ("down_proj", 1)])
@pytest.mark.parametrize("pretrained", [True, False])
def test_tp_external_field_slices(tmp_path, monkeypatch, name, axis, pretrained):
    monkeypatch.setattr(compat, "VLLM_TP_CAPABLE", True)
    model = replace(config(tmp_path), tensor_parallel_size=2)
    target = Projection(name, 0)
    full = torch.arange((8 if axis == 0 else 16) * 8, dtype=torch.float32).view(8, -1)
    path = tmp_path / "weights.safetensors"
    save_file({"value_0": full}, path)
    workload = ExternalAdapter(targets=(target,), path=str(path) if pretrained else "")
    report = pllm.explain(workload, model)
    local = full.chunk(2, axis)[0]
    assert report.states[0].fields[0].shape == tuple(local.shape)
    projection = ProjectionShape(target, local.shape[1], local.shape[0], 2 if axis else 1, 1 if axis else 2)
    facts = testing.backend_facts(8, layer_count=3, memory_bytes=1 << 24)
    facts = replace(facts, probe=replace(facts.probe, targets=(projection,), tensor_parallel_size=2))
    for rank, expected in enumerate(full.chunk(2, axis)):
        funded, binding = workload.specialize(facts, pllm.ExecutionOptions())
        binding.tensor_parallel_rank = rank
        if pretrained:
            root = (torch.zeros_like(expected),)
            binding.initializer("weights", torch.device("cpu"), 7)((root,), (None,))
            torch.testing.assert_close(root[0], expected)
        else:
            attachment = workload.plan_backend(funded, binding)
            assert attachment.spec.import_fields[0][1].shape == tuple(full.shape)
            execution = workload.bind_backend(funded, attachment, binding).execution
            identity = testing.RequestIdentity("import", 0, 0)
            execution.admit_instances()
            execution.admit((identity,))
            result = testing.run_step(execution, testing.step_input((identity,), batch=pllm.io.StateImport((full,))))
            (actual,) = execution.snapshot(result.publication.after_refs[0])
            torch.testing.assert_close(actual.float(), expected)
            execution.retire((identity,), wait=True)
            execution.close()


def test_soft_capped_head_logits_logprobs_and_gradient():
    generator = torch.Generator().manual_seed(21)
    head = LMHead(torch.randn(9, 3, dtype=torch.float64, generator=generator), 7, 1.5, 0.75)
    hidden = torch.randn(259, 3, dtype=torch.float64, generator=generator, requires_grad=True)
    labels = torch.arange(len(hidden)) % 7
    labels[::3] = -100
    expected = ((hidden @ head.weight[:7].T) / 0.75).tanh() * 0.75 * 1.5
    reference = -F.cross_entropy(expected, labels, reduction="none")
    result = head.label_logprobs(hidden, labels, -100)
    torch.testing.assert_close(head(hidden), expected)
    torch.testing.assert_close(result, reference)
    torch.testing.assert_close(
        torch.autograd.grad(result.sum(), hidden)[0], torch.autograd.grad(reference.sum(), hidden)[0]
    )


def test_soft_capped_window_attention_ignores_padding():
    gen = torch.Generator().manual_seed(17)
    q = torch.randn(1, 2, 5, 4, generator=gen)
    k, v = torch.randn(2, 1, 1, 5, 4, generator=gen)
    mask = torch.tensor([False, True, True, True, True]).view(1, 1, 1, 5)
    actual = causal_attention(q, k, v, mask, 2, 0.75, 0.2)
    expected = torch.zeros_like(q)
    for row in range(1, 5):
        window = slice(max(1, row - 1), row + 1)
        scores = q[:, :, row : row + 1] @ k[:, :, window].transpose(-1, -2)
        probs = ((scores * 0.2 / 0.75).tanh() * 0.75).softmax(-1)
        expected[:, :, row : row + 1] = probs @ v[:, :, window]
    torch.testing.assert_close(actual, expected)


def bare_module(cls):
    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    return module


@pytest.mark.parametrize("family", ["gemma2", "gemma3", "glm4"])
def test_four_norm_training_matches_native_branch_order(family):
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
    from vllm.model_executor.models.gemma2 import Gemma2DecoderLayer, Gemma2ForCausalLM
    from vllm.model_executor.models.gemma3 import Gemma3DecoderLayer, Gemma3ForCausalLM
    from vllm.model_executor.models.glm4 import Glm4DecoderLayer, Glm4ForCausalLM

    from pllm.backends.vllm.architecture import lm_head_module, sublayer_norms
    from pllm.backends.vllm.training import feedforward

    models = {"gemma2": Gemma2ForCausalLM, "gemma3": Gemma3ForCausalLM, "glm4": Glm4ForCausalLM}
    layers = {"gemma2": Gemma2DecoderLayer, "gemma3": Gemma3DecoderLayer, "glm4": Glm4DecoderLayer}
    gemma = family != "glm4"
    model, layer = bare_module(models[family]), bare_module(layers[family])
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([layer])
    model.model.embed_tokens = torch.nn.Embedding(7, 4)
    if family != "gemma2":
        model.lm_head = torch.nn.Linear(4, 7)
    assert lm_head_module(model) is (model.model.embed_tokens if family == "gemma2" else model.lm_head)

    def rms(weight):
        module = bare_module(GemmaRMSNorm if gemma else RMSNorm)
        module.weight = torch.nn.Parameter(torch.tensor(weight))
        module.variance_epsilon = 1e-5
        if not gemma:
            module.variance_size_override = None
        return module

    branch_norm = rms([0.3, 0.5, -0.1, 1.2])
    residual_norm = rms([1.0, 2.0, 0.5, 1.5])
    mlp_norm = rms([0.2, 0.9, 1.5, -0.5])
    model.model.norm = layer.input_layernorm = rms([1.0, 1.0, 1.0, 1.0])
    if gemma:
        layer.post_attention_layernorm, layer.pre_feedforward_layernorm = branch_norm, residual_norm
        layer.post_feedforward_layernorm = mlp_norm
    else:
        layer.post_self_attn_layernorm, layer.post_attention_layernorm = branch_norm, residual_norm
        layer.post_mlp_layernorm = mlp_norm
    assert sublayer_norms(model) == (layer.input_layernorm, residual_norm, model.model.norm)

    layer.mlp = torch.nn.Module()
    layer.mlp.gate_up_proj = layer.mlp.down_proj = torch.nn.Identity()
    layer.mlp.act_fn = torch.nn.Module()
    layer.mlp.act_fn.forward_native = torch.tanh
    branch = torch.tensor([[0.1, 0.5, 0.3, 1.5]])
    residual = torch.tensor([[1.5, 0.2, 0.5, 1.0]])

    def reference(module, x):
        return F.rms_norm(x, (4,), module.weight + int(gemma), module.variance_epsilon)

    expected_residual = residual + reference(branch_norm, branch)
    expected = reference(mlp_norm, reference(residual_norm, expected_residual).tanh())
    output, stream = feedforward(layer, branch, residual, lambda projection, x: projection(x))
    torch.testing.assert_close(stream, expected_residual)
    torch.testing.assert_close(output, expected)


@dataclass(frozen=True)
class PrefixPatch(pllm.Workload):
    last: int = 0
    state = pllm.InstanceState({"value": (8,)})

    @property
    def patch(self) -> pllm.model.ActivationPatch:
        return pllm.model.ActivationPatch("state", (0,), first_tokens=1, last_tokens=self.last, operator="add")


class MutablePrefixPatch(PrefixPatch):
    @pllm.write(state="state", fields=("value",))
    def update(self, value=pllm.parameter("state", "value")):
        return (value + 1,)


def test_prefix_cache_patch_window_and_transitions(tmp_path):
    model = replace(config(tmp_path), enable_prefix_caching=True)
    assert not pllm.explain(PrefixPatch(), model).eager_reason
    with pytest.raises(ValueError, match="last_tokens"):
        pllm.explain(PrefixPatch(last=1), model)
    with pytest.raises(ValueError, match="state 'state'"):
        pllm.explain(MutablePrefixPatch(), model)


class Sublayers(pllm.Workload):
    @pllm.custom_residual(site="both", layers=(0,))
    def scale_branch(self, stream, branch):
        return stream + branch * 0.5


@pytest.mark.parametrize("family", ["DeepseekV2", "DeepseekV3", "GptOss", "Mixtral", "Qwen2Moe"])
def test_sublayer_hooks_do_not_require_value_projections(family):
    from pllm.backends.vllm.architecture import KV_MOE_ARCHITECTURES, READ_ONLY_ARCHITECTURES, bind_model
    from pllm.backends.vllm.model_sites import ModelSites
    from test.workloads.test_model_sites import LAYERS, WIDTH, Body

    cls = next(
        model for model in (*KV_MOE_ARCHITECTURES, *READ_ONLY_ARCHITECTURES) if model.__name__ == family + "ForCausalLM"
    )
    model = bare_module(cls)
    model.model = Body()
    decoder = bind_model(model)
    facts = testing.backend_facts(WIDTH, layer_count=LAYERS, dtype=torch.float64)
    with testing.compile_execution(Sublayers(), facts) as execution:
        sites = ModelSites(decoder, execution.math.model_sites, None, None, 16, 2, execution.arena)
        assert len(sites.hooks) == 2
        branch = torch.ones(1, WIDTH, dtype=torch.float64)
        for hook in sites.hooks:
            _, stream = hook(branch.clone(), torch.ones_like(branch))
            torch.testing.assert_close(stream, torch.full_like(branch, 1.5))
        sites.close()
