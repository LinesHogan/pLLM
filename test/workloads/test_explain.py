"""CPU explanation contracts: built-in coverage, cost attribution and budget fallback without CUDA or weights."""

import json
from dataclasses import asdict, dataclass, replace
from math import prod
from pathlib import Path
from typing import Literal

import pytest
import torch

import pllm
from pllm import testing
from pllm.backends.vllm import compat
from pllm.contracts.tensor_layout import TensorDType
from pllm.workloads import (
    DoLaWorkload,
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
    RecirculationWorkload,
    ReFTConfig,
    ReFTWorkload,
    RewardHeadWorkload,
    StepLoopWorkload,
    TLMConfig,
    TLMWorkload,
)
from pllm.workloads.gated_delta import GatedDeltaConfig, GatedDeltaWorkload
from pllm.workloads.video import VideoWorkload
from test.workloads.attention_workloads import SharedAttention, SinkAttention, WindowAttention
from test.workloads.kernel_workloads import Forward, Residual


@dataclass(frozen=True)
class _PrefillBias(pllm.Workload):
    """A prefill-only addition before a separate decoder contribution."""

    @pllm.add_logits()
    def bias(self, h: torch.Tensor = pllm.hidden(0, tokens="prefill")) -> torch.Tensor:
        return h[:, :1].float().expand(-1, 32)


@dataclass(frozen=True)
class _MixedAdditions(pllm.Workload):
    prefill: _PrefillBias = _PrefillBias()
    decode: DoLaWorkload = DoLaWorkload(layers=(0, 2))


@pytest.fixture
def model_config(tmp_path, monkeypatch):
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

    def forbidden(*args, **kwargs):
        pytest.fail("explain must not initialize CUDA, query GPU memory or open model weights")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "mem_get_info", forbidden)
    monkeypatch.setattr("safetensors.safe_open", forbidden)
    return pllm.ModelConfig(str(tmp_path), max_num_seqs=3, max_num_batched_tokens=16)


@pytest.mark.parametrize(
    "workload",
    (
        DummyWorkload(),
        WindowAttention(),
        SinkAttention(),
        SharedAttention(),
        Residual(),
        Forward(),
        Forward(constant=True),
        DoLaWorkload(layers=(0, 2, 4)),
        ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5)),
        PromptFactorsWorkload(),
        OutputLoRAWorkload(),
        OverRIDEWorkload(OverRIDEConfig(rank=2, num_iterations=3)),
        PrefillAdaptersWorkload(rank=2, chunk_size=4),
        GatedDeltaWorkload(GatedDeltaConfig((pllm.model.Projection("down_proj", 0),), head_width=4, chunk_size=4)),
        InPlaceTTTWorkload(InPlaceTTTConfig((pllm.model.Projection("down_proj", 0),), 4, 0.01)),
        ReFTWorkload(ReFTConfig((0,))),
        RecirculationWorkload(),
        JacobiRecirculationWorkload(),
        StepLoopWorkload(),
        FTTTWorkload(),
        RewardHeadWorkload(),
        TLMWorkload(TLMConfig()),
        VideoWorkload(),
    ),
    ids=lambda workload: type(workload).__name__,
)
def test_builtin_explanations_are_cpu_only_and_json_ready(workload, model_config):
    report = pllm.explain(workload, model_config)
    assert json.loads(json.dumps(asdict(report)))["model"]["hidden_size"] == 8
    assert report.peak_memory_bytes >= sum(state.reserved_bytes for state in report.states)
    assert report.members and not report.budget_errors
    assert "[exact; state_page_budget_bytes]" in str(report)
    if isinstance(workload, (WindowAttention, SinkAttention, SharedAttention)):
        assert len(report.kv_capacity) == 10 and "KV bytes/request" in str(report)
        full = report.kv_capacity[2].bytes
        if isinstance(workload, WindowAttention):
            assert report.kv_capacity[0].bytes < full
        elif isinstance(workload, SharedAttention):
            assert sum(row.bytes for row in report.kv_capacity) == 5 * full
        with pytest.raises(ValueError, match="disable prefix caching"):
            pllm.explain(workload, replace(model_config, enable_prefix_caching=True))
    if isinstance(workload, (Residual, Forward)):
        add, scale = report.custom_ops
        assert add.flops_per_row == 8 and scale.flops_per_row is None
        assert add.cpu_reference and "unknown [no flop formula, counted 0]" in str(report)
    if isinstance(workload, Forward):
        assert report.members[0].tasks[0].estimate is None
        assert report.members[0].kernels.total > 0
    if isinstance(workload, (StepLoopWorkload, PrefillAdaptersWorkload)):
        tokens = tuple(item for item in report.escape_hatches if item.setting == "tokens")
        assert tokens and all(item.is_default for item in tokens)
    if report.eager_reason and "exact train_in_forward" not in report.eager_reason:
        with pytest.raises(ValueError, match='cuda_graphs="auto"'):
            pllm.explain(workload, replace(model_config, cuda_graphs=True))


def test_esamp_attribution_and_budget_fallback(model_config):
    workload = ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5))
    report = pllm.explain(workload, model_config)
    members = {member.name: member for member in report.members}
    assert members["loss"].effect == "train"
    assert any(task.lane == "side" and task.captured for task in members["loss"].tasks)
    prediction = next(task for task in members["predict"].tasks if task.name == "predict")
    assert prediction.reads == ("distiller",)
    assert report.inline_peak_memory_bytes < report.overlap_threshold_bytes == report.peak_memory_bytes
    inline = pllm.explain(
        workload,
        replace(
            model_config,
            execution=pllm.ExecutionOptions(
                execution_peak_budget_bytes=report.inline_peak_memory_bytes,
            ),
        ),
    )
    assert not inline.budget_errors and all(task.lane == "main" for member in inline.members for task in member.tasks)
    failed = pllm.explain(
        workload, replace(model_config, execution=pllm.ExecutionOptions(execution_peak_budget_bytes=1))
    )
    assert failed.budget_errors and "max_num_batched_tokens" in str(failed)
    assert failed.peak_memory_bytes == report.peak_memory_bytes
    state_failed = pllm.explain(
        workload, replace(model_config, execution=pllm.ExecutionOptions(state_page_budget_bytes=1))
    )
    assert state_failed.budget_errors


def test_declared_flop_estimates(model_config):
    report = pllm.explain(
        ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5)),
        replace(model_config, execution=pllm.ExecutionOptions(guidance_capacity=None)),
    )
    estimates = {task.name: task.estimate for member in report.members for task in member.tasks}
    rows, hidden, rank, vocab, padded_vocab = 3, 8, 2, 32, 64
    assert estimates["predict"].matmul_flops_estimate == rows * 6 * hidden * rank
    assert estimates["predict"].projection_flops_estimate == rows * 2 * hidden * padded_vocab
    # Input hidden rows are constants: backward needs four GEMMs, forward needs three.
    assert estimates["loss"].matmul_flops_estimate == rows * 14 * hidden * rank
    assert estimates["loss"].optimizer_bytes_estimate > 0
    assert estimates["predict"].sampler_bytes_estimate == rows * vocab * 10
    bounded = pllm.explain(
        ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5)),
        model_config,
    )
    prediction = next(task for member in bounded.members for task in member.tasks if task.name == "predict")
    assert prediction.estimate.projection_flops_estimate == rows * 2 * hidden * vocab
    assert prediction.estimate.sampler_bytes_estimate == rows * (
        16 * vocab + vocab * (hidden * 2 + 20) + hidden * 2 + 16
    )
    assert "highest 4096 survivors/row" in str(bounded) and "tail unguided" in str(bounded)
    assert "smaller K is slower because overflow rows need top-K selection" in str(bounded)
    for config, reason in (
        (replace(model_config, runner="mrv1"), "MRv1"),
        (replace(model_config, tensor_parallel_size=2), "TP>1"),
    ):
        if config.tensor_parallel_size > 1 and compat.VLLM_027:
            continue
        fallback = pllm.explain(ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5)), config)
        prediction = next(task for member in fallback.members for task in member.tasks if task.name == "predict")
        assert prediction.estimate.projection_flops_estimate == rows * 2 * hidden * padded_vocab
        assert "exact dense post-filter guidance" in str(fallback) and reason in str(fallback)
    dola = pllm.explain(DoLaWorkload(layers=(0, 2)), model_config)
    contrast = next(task for member in dola.members for task in member.tasks if task.name == "contrast")
    assert contrast.estimate.projection_flops_estimate == rows * 3 * 2 * hidden * padded_vocab
    assert contrast.estimate.sampler_bytes_estimate == rows * vocab * 10
    mixed = pllm.explain(_MixedAdditions(), model_config)
    additions = {task.name: task.estimate for member in mixed.members for task in member.tasks}
    assert (
        additions["prefill.bias"].sampler_bytes_estimate == additions["prefill.bias"].eager_kernel_count_estimate == 0
    )
    assert additions["decode.contrast"].sampler_bytes_estimate == contrast.estimate.sampler_bytes_estimate
    assert additions["decode.contrast"].eager_kernel_count_estimate == contrast.estimate.eager_kernel_count_estimate
    lora = pllm.explain(OutputLoRAWorkload(), model_config)
    assert any(
        task.estimate is not None and task.estimate.optimizer_bytes_estimate > 0
        for member in lora.members
        for task in member.tasks
    )
    assert "estimate / step" in str(report)
    assert report.kernels.captured > 0 and report.kernels.eager > 0
    assert report.kernels.total == sum(member.kernels.total for member in report.members) + sum(
        task.estimate.kernel_count_estimate + task.estimate.eager_kernel_count_estimate
        for task in report.framework_tasks
        if task.estimate is not None
    )
    assert str(report).count("kernels  ") == len(report.members) + 1


def test_kernel_estimate_excludes_metadata_and_counts_mutations():
    from pllm.execution.torch_private import KernelCountMode
    from pllm.kernels.page_copy import copy_pages

    x = torch.empty((3, 8), device="meta")
    indices = torch.empty(3, device="meta", dtype=torch.long)
    with KernelCountMode() as counter:
        view = x.view(3, 2, 4).transpose(1, 2)
        torch.empty_like(view)
        x.new_empty(x.shape)
        x.copy_(x.view_as(x))
        x.copy_(torch.empty(x.shape))
    assert counter.count == 0
    with KernelCountMode() as counter:
        result = x + 1
        result.zero_()
        copy_pages(result, x, indices, indices)
    assert counter.count == 3


@pytest.mark.parametrize("workload", (StepLoopWorkload(), RecirculationWorkload()))
def test_tp_route_reports_and_cpu_check(workload, model_config):
    from pllm import testing
    from pllm.backends.vllm import compat

    config = replace(model_config, tensor_parallel_size=2)
    if compat.VLLM_027:
        with pytest.raises(ValueError, match="vLLM 0.32.0"):
            pllm.explain(workload, config)
        with pytest.raises(testing.ContractError, match="vLLM 0.32.0"):
            testing.check(workload, config)
        return
    report = pllm.explain(workload, config)
    assert asdict(report)["tensor_parallel_size"] == 2
    assert "per-rank" in report.memory_label and "replicated across TP ranks" in str(report)
    checked = testing.check(workload, config)
    assert "tensor_parallel_size=2: admission" in checked.passed
    assert any(item.property == "TP rank equality" for item in checked.skipped)


@dataclass(frozen=True)
class PrecisionRoles(pllm.Workload):
    """Exercise role inference within a state as well as optimizer and chunk state."""

    dtype: torch.dtype | Literal["auto"] = "auto"

    @property
    def values(self) -> pllm.InstanceState:
        return pllm.InstanceState({name: (8,) for name in ("read", "update", "write")}, dtype=self.dtype)

    @property
    def imported(self) -> pllm.InstanceState:
        return pllm.InstanceState({"w": (8,)}, imported=True, dtype=self.dtype)

    @property
    def trained(self) -> pllm.RequestState:
        return pllm.RequestState({"w": (8,)}, optimizer=pllm.AdamW(lr=0.01), dtype=self.dtype)

    @property
    def memory(self) -> pllm.RequestState:
        return pllm.RequestState({"carry": (8,)}, dtype=self.dtype)

    @pllm.write(state="values", fields=("update", "write"))
    def publish(self, read=pllm.parameter("values", "read"), update=pllm.parameter("values", "update")):
        return update + read, read * 2

    @pllm.train(state="trained")
    def loss(self, hidden=pllm.hidden(0), w=pllm.parameter("trained", "w"), target=pllm.parameter("imported", "w")):
        return (hidden - w - target).square().mean(-1)

    @pllm.custom_residual(site="mlp", layers=(0,), state="memory", chunk=4)
    def remember(self, x, y, carry=pllm.parameter("memory", "carry")):
        return x + y + carry.to(x), carry + x.float().sum(0)


@pytest.mark.parametrize("dtype", ("auto", torch.float16, torch.float32))
@pytest.mark.parametrize("model_dtype", ("bfloat16", "float16"))
def test_storage_precision_follows_declared_role(dtype, model_dtype, model_config):
    declaration = PrecisionRoles(dtype).build(
        testing.backend_facts(8, (testing.hidden_binding(0, 10),), dtype=TensorDType(model_dtype), layer_count=10)
    )
    expected = {
        ("values", "read"): model_dtype,
        ("values", "update"): "float32",
        ("values", "write"): model_dtype,
        ("imported", "w"): model_dtype,
        ("trained", "w"): "float32",
        ("memory", "carry"): "float32",
    }
    layouts = {entry.resource: entry.layout for entry in declaration.layouts.layouts}
    assert {key: layouts[key[0]].fields[declaration.field_views[key].field].dtype.value for key in expected} == {
        key: value if dtype == "auto" else str(dtype).removeprefix("torch.") for key, value in expected.items()
    }


@pytest.mark.parametrize(
    "metadata,reason",
    (
        (
            {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "logit_scale": 2.0},
            "scaled or transformed LM head",
        ),
        (
            {
                "quantization_config": {
                    "quant_method": "auto_gptq",
                    "bits": 4,
                    "group_size": 128,
                    "sym": True,
                    "desc_act": False,
                    "lm_head": True,
                    "modules_in_block_to_quantize": ["lm_head"],
                }
            },
            "LM-head method AutoGPTQLinearMethod",
        ),
        ({"quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic"}}, "bounded fused guidance"),
        ({"head_dtype": "float32"}, "scaled or transformed LM head"),
        (
            {
                "architectures": ["GptOssForCausalLM"],
                "model_type": "gpt_oss",
                "quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"},
            },
            "bounded fused guidance",
        ),
    ),
)
def test_guidance_head_selection(model_config, metadata, reason):
    path = Path(model_config.model) / "config.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | metadata))
    report = pllm.explain(ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5)), model_config)
    prediction = next(task for member in report.members for task in member.tasks if task.name == "predict")
    assert prediction.estimate.projection_flops_estimate == 3 * 2 * 8 * (
        32 if reason == "bounded fused guidance" else 64
    )
    assert reason in str(report)


@pytest.mark.parametrize(
    "kind", ["dense", "low_rank", "dense_train", "low_rank_train", "dense_shared", "low_rank_shared"]
)
@pytest.mark.parametrize("target", ["down_proj", "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"])
def test_tp_projection_shard_costs(model_config, kind, target):
    from pllm.backends.vllm import compat
    from test.tp_overlays import DenseTraining, LowRankTraining, ProjectionAdapter, SharedProjectionAdapter

    if compat.VLLM_027:
        pytest.skip("TP needs vLLM 0.32")
    workload = {
        "dense": ProjectionAdapter(),
        "low_rank": ProjectionAdapter("low_rank"),
        "dense_train": DenseTraining(),
        "low_rank_train": LowRankTraining(),
        "dense_shared": SharedProjectionAdapter(),
        "low_rank_shared": SharedProjectionAdapter("low_rank"),
    }[kind]
    workload = replace(workload, targets=(pllm.model.Projection(target, 0),))
    whole = pllm.explain(workload, model_config).states[0]
    sharded = pllm.explain(workload, replace(model_config, tensor_parallel_size=2)).states[0]
    row = target == "down_proj"
    sharded_index = int(kind.startswith("low_rank") and not row)
    axis = -1 if row else 0
    # The config has one KV head: both ranks own the full key/value delta.
    partitions = 1 if target in ("k_proj", "v_proj") else 2
    for index, (part, full) in enumerate(zip(sharded.fields, whole.fields)):
        shape = list(full.shape)
        if index == sharded_index:
            shape[axis] //= partitions
        assert part.shape == tuple(shape)
    assert sharded.bytes_per_root == sum(field.dtype.itemsize * prod(field.shape) for field in sharded.fields)
    from pllm.contracts.parameter_overlay import ProjectionShape

    whole = ()
    for tp in (1, 2, 4):
        # Two global KV heads at TP4 exercise adjacent-rank replica grouping.
        partitions = min(tp, 2) if target in ("k_proj", "v_proj") else tp
        projection = ProjectionShape(
            workload.targets[0],
            8 // partitions if row else 8,
            8 if row else 8 // partitions,
            partitions if row else 1,
            1 if row else partitions,
        )
        facts = testing.backend_facts(8, layer_count=10, memory_bytes=1 << 20, targets=(projection,))
        binding = workload.build(replace(facts, probe=replace(facts.probe, tensor_parallel_size=tp)))
        for rank in range(tp):
            binding.tensor_parallel_rank = rank
            root = tuple(
                torch.zeros(spec.shape, dtype=spec.dtype.torch) for spec in binding.layouts.layouts[0].layout.fields
            )
            binding.initializer("weights", torch.device("cpu"), 7)((root,), (None,))
            if tp == 1:
                whole = root
            for index, (part, full) in enumerate(zip(root, whole)):
                reference = full.chunk(partitions, axis)[rank // (tp // partitions)] if index == sharded_index else full
                assert torch.equal(part, reference)


def test_tp_projection_admission(model_config):
    from pllm.backends.vllm import compat

    if compat.VLLM_027:
        pytest.skip("TP needs vLLM 0.32")
    from test.tp_overlays import DenseChunkTraining, DenseGradientTraining, TensorConstantUpdate, TrainColumnA

    chunked = replace(DenseChunkTraining(), targets=(pllm.model.Projection("q_proj", 0),))
    state = pllm.explain(chunked, replace(model_config, tensor_parallel_size=2)).states[0]
    assert state.fields[2].shape == (2, 8)  # Pending targets retain the replicated hidden width.

    pllm.explain(
        replace(DenseGradientTraining(), targets=chunked.targets), replace(model_config, tensor_parallel_size=2)
    )
    with pytest.raises(ValueError, match="tensor constants"):
        pllm.explain(TensorConstantUpdate(), replace(model_config, tensor_parallel_size=2))
    with pytest.raises(ValueError, match="not shard-local.*gradient needs other ranks"):
        pllm.explain(TrainColumnA(), replace(model_config, tensor_parallel_size=2))
    with pytest.raises(ValueError, match="not shard-local.*gradient needs other ranks"):
        pllm.explain(PrefillAdaptersWorkload(rank=2, chunk_size=4), replace(model_config, tensor_parallel_size=2))

    from test.tp_overlays import ExportB, ExportDown, ReadDown

    for workload in (ExportDown(), ReadDown()):
        with pytest.raises(ValueError, match="sharded fields are projection-local.*tensor_parallel_size=1"):
            pllm.explain(workload, replace(model_config, tensor_parallel_size=2))
    pllm.explain(ExportB(), replace(model_config, tensor_parallel_size=2))
