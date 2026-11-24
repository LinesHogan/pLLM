"""CPU unit, contract, budget and boundary tests for Concertina, SharedLayers and custom projection operators."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from vllm.model_executor.layers.linear import LinearMethodBase, RowParallelLinear, UnquantizedLinearMethod

import pllm
from pllm import testing
from pllm.backends.vllm.checkpoint import _is_follower_linear, _tie_linear, load_options, tie_shared_layers
from pllm.backends.vllm.model_shape import ModelShape, _head_partitions
from pllm.contracts.parameter_overlay import ParameterOverlayOperator
from pllm.execution.parameter_projection import parameter_correction, prepare_static_projection_read, projection_read
from pllm.host import HostConfig, make_host
from pllm.model import Projection, SharedLayers
from pllm.workloads.concertina import (
    ConcertinaConfig,
    ConcertinaWorkload,
    concertina_gated_low_rank,
    prepare_concertina_checkpoint,
)


def _write_synthetic_qwen3_config(
    path: Path,
    *,
    num_layers: int = 4,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    quantization: dict[str, object] | None = None,
    architecture: str = "Qwen3ForCausalLM",
) -> None:
    cfg: dict[str, object] = {
        "architectures": [architecture],
        "model_type": "qwen3" if architecture == "Qwen3ForCausalLM" else "qwen3_moe",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": 8,
        "vocab_size": 128,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
    }
    if quantization is not None:
        cfg["quantization_config"] = quantization
    (path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")


def _reference_concertina_linear(
    x: torch.Tensor,
    w0: torch.Tensor,
    a_list: list[torch.Tensor],
    b_list: list[torch.Tensor],
    wg: torch.Tensor,
    gb: torch.Tensor,
    *,
    has_zero_expert: bool,
) -> torch.Tensor:
    """Exact reference of concertina.py's GatedLoRALinear (per-token softmax gate)."""
    base = torch.nn.functional.linear(x, w0)
    logits = torch.matmul(x, wg) + gb
    gates = torch.softmax(logits.float(), dim=-1).to(x.dtype)
    offset = 1 if has_zero_expert else 0
    delta = torch.zeros_like(base)
    for k, (a_k, b_k) in enumerate(zip(a_list, b_list, strict=True)):
        mid = torch.nn.functional.linear(x, a_k)
        delta = delta + gates[..., k + offset : k + offset + 1] * torch.nn.functional.linear(mid, b_k)
    return base + delta


@pytest.mark.parametrize("experts,gates", [(1, 2), (2, 3), (2, 2), (3, 4)])
def test_concertina_custom_op_matches_reference_and_vmap(experts: int, gates: int) -> None:
    """concertina_gated_low_rank matches concertina.py to 1e-12 across steps, unbatched and under torch.vmap."""
    gen = torch.Generator().manual_seed(42 + experts * 10 + gates)
    rows, in_f, out_f, rank = 7, 16, 20, 4
    fold_group = gates
    x = torch.randn(rows, in_f, generator=gen, dtype=torch.float64)
    w0 = torch.randn(out_f, in_f, generator=gen, dtype=torch.float64) * 0.1
    a_list = [torch.randn(rank, in_f, generator=gen, dtype=torch.float64) * 0.1 for _ in range(experts)]
    b_list = [torch.randn(out_f, rank, generator=gen, dtype=torch.float64) * 0.1 for _ in range(experts)]
    wg_hf_list = [torch.randn(in_f, gates, generator=gen, dtype=torch.float64) * 0.1 for _ in range(fold_group)]
    gb_list = [torch.randn(gates, generator=gen, dtype=torch.float64) * 0.1 for _ in range(fold_group)]

    a_gates = torch.cat([*a_list, *(wg.transpose(0, 1) for wg in wg_hf_list)], dim=0).contiguous()
    b = torch.stack(b_list, dim=1).contiguous()

    for step in range(fold_group):
        expected_full = _reference_concertina_linear(
            x, w0, a_list, b_list, wg_hf_list[step], gb_list[step], has_zero_expert=(gates == experts + 1)
        )
        step_idx = torch.tensor(step, dtype=torch.int64)
        actual_delta = parameter_correction(
            x,
            (a_gates, b, gb_list[step], step_idx),
            ParameterOverlayOperator.CUSTOM,
            batched=False,
            custom_operator=concertina_gated_low_rank,
        )
        torch.testing.assert_close(
            torch.nn.functional.linear(x, w0) + actual_delta, expected_full, rtol=1e-12, atol=1e-12
        )

    # Also check RequestState batched path via projection_read (torch.vmap) with step=1.
    step_idx = torch.tensor(1, dtype=torch.int64)
    gb = gb_list[1]
    expected_step1 = parameter_correction(
        x,
        (a_gates, b, gb, step_idx),
        ParameterOverlayOperator.CUSTOM,
        batched=False,
        custom_operator=concertina_gated_low_rank,
    )
    pool_ag = torch.stack([torch.zeros_like(a_gates), a_gates], dim=0)
    pool_b = torch.stack([torch.zeros_like(b), b], dim=0)
    pool_gb = torch.stack([torch.zeros_like(gb), gb], dim=0)
    pool_step = torch.stack([torch.zeros_like(step_idx), step_idx], dim=0)
    slots = torch.tensor([[1, 1, 1, 1]])
    row_idx = torch.arange(rows).unsqueeze(0)
    owners = torch.arange(rows)
    (batched_delta,) = projection_read(
        x,
        slots,
        row_idx,
        owners,
        pool_ag,
        pool_b,
        pool_gb,
        pool_step,
        operator=ParameterOverlayOperator.CUSTOM,
        custom_operator=concertina_gated_low_rank,
    )
    torch.testing.assert_close(batched_delta, expected_step1, rtol=1e-12, atol=1e-12)


def test_shared_layers_validation_and_make_host_rejection(tmp_path: Path) -> None:
    """SharedLayers rejects non-tuples, overlaps, out-of-range indices, non-dense/quantized models and make_host."""
    with pytest.raises(TypeError, match="tuple of layer-index tuples"):
        SharedLayers([(0, 1)])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="each group to be a tuple"):
        SharedLayers(([0, 1],))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least two"):
        SharedLayers(((0,),))
    with pytest.raises(ValueError, match="more than one group"):
        SharedLayers(((0, 1), (1, 2)))
    with pytest.raises(ValueError, match="maps multiple keys"):
        SharedLayers(((0, 1),), checkpoint_rename={"a": "x", "b": "x"})
    with pytest.raises(TypeError, match="checkpoint_exclude"):
        SharedLayers(((0, 1),), checkpoint_exclude=["a"])  # type: ignore[arg-type]

    @dataclass(frozen=True)
    class PureShared(pllm.Workload):
        groups: tuple[tuple[int, ...], ...] = ((0, 1),)

        @property
        def shared(self) -> SharedLayers:
            return SharedLayers(self.groups)

    with pytest.raises(ValueError, match="make_llm, not make_host"):
        make_host(HostConfig(hidden_size=32, device=torch.device("cpu")), workload=PureShared())

    # Pure SharedLayers passes explain (affects_model == True) on a dense unquantized model.
    dense_dir = tmp_path / "dense"
    dense_dir.mkdir()
    _write_synthetic_qwen3_config(dense_dir, num_layers=4)
    report = pllm.explain(
        PureShared(((0, 1), (2, 3))), pllm.ModelConfig(str(dense_dir), max_model_len=128, dtype="bfloat16")
    )
    assert report.shared_layer_groups == ((0, 1), (2, 3))
    assert "shared   2 groups" in str(report)

    # Out-of-range SharedLayers fails in explain.
    with pytest.raises(ValueError, match="outside the model's 4 layers"):
        pllm.explain(PureShared(((0, 4),)), pllm.ModelConfig(str(dense_dir), max_model_len=128, dtype="bfloat16"))

    # Quantized checkpoint fails closed in explain.
    quant_dir = tmp_path / "quant"
    quant_dir.mkdir()
    _write_synthetic_qwen3_config(quant_dir, num_layers=4, quantization={"quant_method": "fp8"})
    with pytest.raises(ValueError, match="unquantized linear layers"):
        pllm.explain(PureShared(((0, 1),)), pllm.ModelConfig(str(quant_dir), max_model_len=128, dtype="bfloat16"))

    # Non-dense architecture fails closed in explain.
    moe_dir = tmp_path / "moe"
    moe_dir.mkdir()
    _write_synthetic_qwen3_config(moe_dir, num_layers=4, architecture="Qwen3MoeForCausalLM")
    with pytest.raises(ValueError, match="dense decoder architecture"):
        pllm.explain(PureShared(((0, 1),)), pllm.ModelConfig(str(moe_dir), max_model_len=128, dtype="bfloat16"))


def test_tie_linear_and_quantized_rejection() -> None:
    """_tie_linear ties weight/bias storage on unquantized layers and rejects quantized linear methods."""

    class FakeQuantMethod(LinearMethodBase):
        def create_weights(self, *args, **kwargs):
            pass

        def apply(self, layer, x, bias=None):
            return x

    def _make_linear(quant: LinearMethodBase, *, with_bias: bool = True) -> RowParallelLinear:
        mod = RowParallelLinear.__new__(RowParallelLinear)
        torch.nn.Module.__init__(mod)
        mod.weight = torch.nn.Parameter(torch.randn(4, 4))
        mod.bias = torch.nn.Parameter(torch.randn(4)) if with_bias else None
        mod.quant_method = quant
        return mod

    leader = _make_linear(UnquantizedLinearMethod(), with_bias=True)
    follower = _make_linear(UnquantizedLinearMethod(), with_bias=True)
    assert follower.weight is not leader.weight
    assert follower.bias is not leader.bias
    _tie_linear(leader, follower)
    assert follower.weight is leader.weight
    assert follower.bias is leader.bias

    q_leader = _make_linear(FakeQuantMethod(), with_bias=False)
    q_follower = _make_linear(FakeQuantMethod(), with_bias=False)
    with pytest.raises(ValueError, match="unquantized linear layers"):
        _tie_linear(q_leader, q_follower)

    assert _is_follower_linear("model.layers.1.self_attn.q_proj.weight", frozenset({1}))
    assert _is_follower_linear("model.layers.1.self_attn.q_proj.bias", frozenset({1}))
    assert not _is_follower_linear("model.layers.1.input_layernorm.weight", frozenset({1}))
    assert not _is_follower_linear("model.layers.0.self_attn.q_proj.weight", frozenset({1}))
    assert tie_shared_layers(torch.nn.Linear(2, 2), ()) is None

    opts = load_options(
        (),
        shared_layer_groups=((0, 1),),
        checkpoint_rename=(("a", "b"),),
        checkpoint_exclude=("c",),
    )
    assert opts["load_format"] == "pllm"
    extra = opts["model_loader_extra_config"]
    assert isinstance(extra, dict)
    assert extra["checkpoint_rename"] == (("a", "b"),)
    assert extra["checkpoint_exclude"] == ("c",)


def test_o_proj_tp_head_boundary(tmp_path: Path) -> None:
    """o_proj rejects tensor_parallel_size > num_attention_heads just like q_proj."""
    assert _head_partitions("o_proj", 8, 4) == 4
    with pytest.raises(ValueError, match="o_proj needs at least tensor_parallel_size query heads"):
        _head_partitions("o_proj", 2, 4)

    model_dir = tmp_path / "qwen3_2heads"
    model_dir.mkdir()
    _write_synthetic_qwen3_config(model_dir, num_layers=2, num_heads=2, num_kv_heads=2)
    with pytest.raises(ValueError, match="o_proj needs at least tensor_parallel_size query heads"):
        ModelShape.of_config(
            pllm.ModelConfig(str(model_dir), max_model_len=128, tensor_parallel_size=4, dtype="bfloat16"),
            targets=(Projection("o_proj", 0),),
        )


def test_prepare_concertina_checkpoint_group_sharing_and_fail_closed(tmp_path: Path) -> None:
    """prepare_concertina_checkpoint stores A/B once per group and from_checkpoint fails closed if adapters missing."""
    src = tmp_path / "hf_concertina"
    dst = tmp_path / "pllm_concertina"
    src.mkdir()

    raw_cfg = {
        "architectures": ["ConcertinaForCausalLM"],
        "model_type": "concertina",
        "auto_map": {"AutoModelForCausalLM": "modeling_concertina.ConcertinaForCausalLM"},
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 128,
        "fold_group": 2,
        "fold_rank": 4,
        "fold_gate": True,
        "tie_word_embeddings": True,
    }
    (src / "config.json").write_text(json.dumps(raw_cfg), encoding="utf-8")

    # Missing adapter file fails closed.
    with pytest.raises(FileNotFoundError, match="Concertina adapter file not found"):
        ConcertinaConfig.from_checkpoint(str(src))

    dims = {
        "q_proj": (32, 32),
        "k_proj": (32, 16),
        "v_proj": (32, 16),
        "o_proj": (32, 32),
        "gate_proj": (32, 64),
        "up_proj": (32, 64),
        "down_proj": (64, 32),
    }
    proj_paths = {
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
        "gate_proj": "mlp.gate_proj",
        "up_proj": "mlp.up_proj",
        "down_proj": "mlp.down_proj",
    }
    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.zeros(128, 32, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(32, dtype=torch.bfloat16),
    }
    for start in (0, 2):
        for step in (0, 1):
            tensors[f"model.layers.{start}.shared.input_layernorm.weight.{step}"] = torch.ones(32, dtype=torch.bfloat16)
            tensors[f"model.layers.{start}.shared.post_attention_layernorm.weight.{step}"] = torch.ones(
                32, dtype=torch.bfloat16
            )
            tensors[f"model.layers.{start}.shared.self_attn.q_norm.weight.{step}"] = torch.ones(8, dtype=torch.bfloat16)
            tensors[f"model.layers.{start}.shared.self_attn.k_norm.weight.{step}"] = torch.ones(8, dtype=torch.bfloat16)
        for name, (in_f, out_f) in dims.items():
            base = f"model.layers.{start}.shared.{proj_paths[name]}"
            tensors[f"{base}.base.weight"] = torch.zeros(out_f, in_f, dtype=torch.bfloat16)
            tensors[f"{base}.A.1"] = torch.full((4, in_f), float(start + 1), dtype=torch.bfloat16)
            tensors[f"{base}.B.1"] = torch.full((out_f, 4), float(start + 2), dtype=torch.bfloat16)
            for step in (0, 1):
                tensors[f"{base}.Wg.{step}"] = torch.full((in_f, 2), float(step + 1), dtype=torch.bfloat16)
                tensors[f"{base}.gb.{step}"] = torch.tensor([0.1, -0.1], dtype=torch.bfloat16)
    save_file(tensors, str(src / "model.safetensors"))

    prepared_dir, cfg = prepare_concertina_checkpoint(str(src), str(dst))
    assert (Path(prepared_dir) / "model.safetensors").is_symlink()
    loaded_cfg = ConcertinaConfig.from_checkpoint(prepared_dir)
    assert loaded_cfg == cfg
    assert loaded_cfg.from_folded_checkpoint is True
    assert len(loaded_cfg.checkpoint_rename) == 2 * 7 + 4 * 4  # 14 linear bases + 16 norms
    assert len(loaded_cfg.checkpoint_exclude) == 2 * 7 * (2 + 4)  # A.1, B.1, Wg.{0,1}, gb.{0,1}

    # Verify explain, PTTS FieldView sharing, custom_ops reporting, prefix caching and CPU contracts!
    workload = ConcertinaWorkload(loaded_cfg)
    model_cfg = pllm.ModelConfig(
        prepared_dir,
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    report = pllm.explain(workload, model_cfg)
    assert report.eager_reason == ""
    assert report.budget_errors == ()
    assert report.shared_layer_groups == ((0, 1), (2, 3))
    assert any(op.name == "concertina::gated_low_rank" and op.flops_per_row is not None for op in report.custom_ops)
    # 2 groups * 7 projs = 14 shared a + 14 shared b + 28 gate_weight + 28 gate_bias = 84 fields (not 112!).
    (adapter_state,) = [s for s in report.states if s.name == "adapters"]
    assert len(adapter_state.fields) == 84

    # CPU contract check runs concertina_gated_low_rank row-permutation check too.
    check_res = testing.check(workload, model_cfg)
    assert "custom projection operator math / CPU schedules" in check_res.passed

    # Custom projection operator fails closed under TP > 1.
    with pytest.raises(ValueError, match="custom projection operators need tensor_parallel_size=1"):
        pllm.explain(
            workload, pllm.ModelConfig(prepared_dir, max_model_len=128, tensor_parallel_size=2, dtype="bfloat16")
        )


def test_static_instance_budget_enforcement() -> None:
    """prepare_static_projection_read enforces FakeTensorMode workspace budgets on static adapters."""
    hidden = torch.randn(32, 32, dtype=torch.bfloat16)
    a_gates = torch.randn(8, 32, dtype=torch.bfloat16)
    b = torch.randn(32, 1, 4, dtype=torch.bfloat16)
    gb = torch.randn(2, dtype=torch.bfloat16)
    step = torch.tensor(0, dtype=torch.int64)
    workspace = prepare_static_projection_read(
        hidden,
        (a_gates, b, gb, step),
        ParameterOverlayOperator.CUSTOM,
        budget=1 << 20,
        blas_workspace_bytes=0,
        custom_operator=concertina_gated_low_rank,
    )
    assert workspace > 0
    with pytest.raises(ValueError, match="projection execution budget"):
        prepare_static_projection_read(
            hidden,
            (a_gates, b, gb, step),
            ParameterOverlayOperator.CUSTOM,
            budget=workspace - 1,
            blas_workspace_bytes=0,
            custom_operator=concertina_gated_low_rank,
        )
