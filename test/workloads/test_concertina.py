"""CPU verification for o_proj overlays, gated_low_rank math, SharedLayers and ConcertinaWorkload."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import pllm
from pllm import testing
from pllm.backends.vllm.checkpoint import _is_follower_linear, _remap_folded_weight
from pllm.contracts.parameter_overlay import ParameterOverlayOperator, Projection
from pllm.execution.parameter_projection import parameter_correction
from pllm.workloads.concertina import (
    CONCERTINA_PROJECTIONS,
    ConcertinaConfig,
    ConcertinaWorkload,
    prepare_concertina_checkpoint,
)

_PROJ_PATHS = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}


def _concertina_reference(
    hidden: torch.Tensor,
    a_list: list[torch.Tensor],
    b_list: list[torch.Tensor],
    wg: torch.Tensor,
    gb: torch.Tensor,
    *,
    asym: bool,
) -> torch.Tensor:
    """Direct equation from concertina.py FoldedLinear.forward."""
    logits = hidden @ wg.to(hidden.dtype) + gb.to(hidden.dtype)
    gates = torch.softmax(logits.float(), dim=-1).to(hidden.dtype)
    offset = 1 if asym else 0
    out = torch.zeros((*hidden.shape[:-1], b_list[0].shape[0]), dtype=hidden.dtype, device=hidden.device)
    for k, (a, b) in enumerate(zip(a_list, b_list, strict=True)):
        out = out + gates[..., offset + k : offset + k + 1] * ((hidden @ a.to(hidden.dtype).T) @ b.to(hidden.dtype).T)
    return out


@pytest.mark.parametrize(("experts", "asym"), ((1, True), (2, True), (2, False)))
@pytest.mark.parametrize("batched", (False, True))
def test_gated_low_rank_matches_concertina_reference(experts: int, asym: bool, batched: bool) -> None:
    torch.manual_seed(0)
    rows, in_features, out_features, rank = 7, 16, 24, 4
    n_gates = experts + 1 if asym else experts
    a_list = [torch.randn(rank, in_features, dtype=torch.float64) for _ in range(experts)]
    b_list = [torch.randn(out_features, rank, dtype=torch.float64) for _ in range(experts)]
    wg = torch.randn(in_features, n_gates, dtype=torch.float64)
    gb = torch.randn(n_gates, dtype=torch.float64)

    a = torch.stack(a_list, dim=0)
    b = torch.stack(b_list, dim=0)
    wg_t = wg.T.contiguous()

    if batched:
        blocks = 3
        hidden = torch.randn(blocks, rows, in_features, dtype=torch.float64)
        expected = _concertina_reference(hidden, a_list, b_list, wg, gb, asym=asym)
        actual = parameter_correction(
            hidden,
            (
                a.unsqueeze(0).expand(blocks, -1, -1, -1),
                b.unsqueeze(0).expand(blocks, -1, -1, -1),
                wg_t.unsqueeze(0).expand(blocks, -1, -1),
                gb.unsqueeze(0).expand(blocks, -1),
            ),
            ParameterOverlayOperator.GATED_LOW_RANK_ADD,
            batched=True,
        )
    else:
        hidden = torch.randn(rows, in_features, dtype=torch.float64)
        expected = _concertina_reference(hidden, a_list, b_list, wg, gb, asym=asym)
        actual = parameter_correction(
            hidden,
            (a, b, wg_t, gb),
            ParameterOverlayOperator.GATED_LOW_RANK_ADD,
            batched=False,
        )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_shared_layers_validation_and_key_remapping() -> None:
    with pytest.raises(ValueError, match="at least two"):
        pllm.model.SharedLayers(((0,),))
    with pytest.raises(ValueError, match="more than one group"):
        pllm.model.SharedLayers(((0, 1), (1, 2)))

    group_starts = {0: (0, 1), 2: (2, 3)}
    assert _remap_folded_weight("model.layers.0.shared.self_attn.o_proj.base.weight", group_starts) == (
        "model.layers.0.self_attn.o_proj.weight",
        True,
    )
    assert _remap_folded_weight("model.layers.0.shared.self_attn.q_norm.weight.1", group_starts) == (
        "model.layers.1.self_attn.q_norm.weight",
        True,
    )
    assert _remap_folded_weight("model.layers.0.shared.mlp.down_proj.A.1", group_starts) == (None, True)
    assert _remap_folded_weight("model.layers.0.shared.mlp.down_proj.Wg.0", group_starts) == (None, True)
    assert _remap_folded_weight("model.embed_tokens.weight", group_starts) == ("model.embed_tokens.weight", False)

    followers = frozenset({1, 3})
    assert _is_follower_linear("model.layers.1.self_attn.o_proj.weight", followers)
    assert _is_follower_linear("model.layers.3.mlp.down_proj.weight", followers)
    assert not _is_follower_linear("model.layers.0.self_attn.o_proj.weight", followers)
    assert not _is_follower_linear("model.layers.1.input_layernorm.weight", followers)


def _write_synthetic_concertina_checkpoint(
    root: Path, *, num_layers: int = 4, fold_group: int = 2, fold_rank: int = 4, fold_gate: bool = True
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    hidden, intermediate, heads, kv_heads, head_dim, vocab = 64, 128, 4, 2, 16, 128
    config = {
        "architectures": ["ConcertinaForCausalLM"],
        "model_type": "concertina",
        "auto_map": {
            "AutoConfig": "configuration_concertina.ConcertinaConfig",
            "AutoModelForCausalLM": "modeling_concertina.ConcertinaForCausalLM",
        },
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": num_layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "vocab_size": vocab,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "fold_group": fold_group,
        "fold_rank": fold_rank,
        "fold_gate": fold_gate,
    }
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    proj_dims = {
        "q_proj": (hidden, heads * head_dim),
        "k_proj": (hidden, kv_heads * head_dim),
        "v_proj": (hidden, kv_heads * head_dim),
        "o_proj": (heads * head_dim, hidden),
        "gate_proj": (hidden, intermediate),
        "up_proj": (hidden, intermediate),
        "down_proj": (intermediate, hidden),
    }
    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.zeros(vocab, hidden, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(hidden, dtype=torch.bfloat16),
    }
    folded = num_layers - (num_layers % fold_group)
    for start in range(0, folded, fold_group):
        for name in CONCERTINA_PROJECTIONS:
            in_f, out_f = proj_dims[name]
            base = f"model.layers.{start}.shared.{_PROJ_PATHS[name]}"
            tensors[f"{base}.base.weight"] = torch.zeros(out_f, in_f, dtype=torch.bfloat16)
            for k in range(1, fold_group):
                tensors[f"{base}.A.{k}"] = torch.randn(fold_rank, in_f, dtype=torch.bfloat16) * 0.01
                tensors[f"{base}.B.{k}"] = torch.randn(out_f, fold_rank, dtype=torch.bfloat16) * 0.01
            if fold_gate:
                for step in range(fold_group):
                    tensors[f"{base}.Wg.{step}"] = torch.zeros(in_f, fold_group, dtype=torch.bfloat16)
                    gb = torch.zeros(fold_group, dtype=torch.bfloat16)
                    gb[step] = 8.0
                    tensors[f"{base}.gb.{step}"] = gb
        for norm, width in (
            ("input_layernorm", hidden),
            ("post_attention_layernorm", hidden),
            ("self_attn.q_norm", head_dim),
            ("self_attn.k_norm", head_dim),
        ):
            for step in range(fold_group):
                tensors[f"model.layers.{start}.shared.{norm}.weight.{step}"] = torch.ones(width, dtype=torch.bfloat16)
    save_file(tensors, root / "model.safetensors")


@pytest.mark.parametrize("fold_gate", (True, False))
def test_prepare_and_cpu_contracts_for_concertina_workload(tmp_path: Path, fold_gate: bool) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "prepared"
    _write_synthetic_concertina_checkpoint(src, num_layers=4, fold_group=2, fold_rank=4, fold_gate=fold_gate)

    prepared_dir, cfg = prepare_concertina_checkpoint(str(src), str(dst))
    assert Path(prepared_dir, "model.safetensors").is_symlink()
    loaded_cfg = ConcertinaConfig.from_checkpoint(prepared_dir)
    assert loaded_cfg == cfg
    assert cfg.groups == ((0, 1), (2, 3))
    assert len(cfg.targets) == (4 * 7 if fold_gate else 2 * 7)
    assert Projection("o_proj", 1) in cfg.targets

    workload = ConcertinaWorkload(cfg)
    model_cfg = pllm.ModelConfig(prepared_dir, max_num_seqs=4, max_model_len=64)
    report = pllm.explain(workload, model_cfg)
    assert report.eager_reason == ""
    assert report.budget_errors == ()
    result = testing.check(workload, model_cfg)
    assert "compile" in result.passed


def test_o_proj_and_gated_low_rank_tp_rules(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "prepared"
    _write_synthetic_concertina_checkpoint(src, num_layers=4, fold_group=2, fold_rank=4, fold_gate=True)
    prepared_dir, cfg = prepare_concertina_checkpoint(str(src), str(dst))

    with pytest.raises(ValueError, match="gated_low_rank adapters need tensor_parallel_size=1"):
        pllm.explain(
            ConcertinaWorkload(cfg),
            pllm.ModelConfig(prepared_dir, max_num_seqs=4, max_model_len=64, tensor_parallel_size=2),
        )

    # Plain low_rank on o_proj works under TP=2 (row-parallel sharding like down_proj).
    class OProjLoRA(pllm.Workload):
        lora = pllm.InstanceState(
            {
                "a": pllm.Field(lambda inputs, _outputs: (4, inputs)),
                "b": pllm.Field(lambda _inputs, outputs: (outputs, 4)),
            }
        )
        apply = pllm.model.Adapter("lora", (Projection("o_proj", 0),), operator="low_rank")

    report = pllm.explain(
        OProjLoRA(),
        pllm.ModelConfig(prepared_dir, max_num_seqs=4, max_model_len=64, tensor_parallel_size=2),
    )
    assert report.eager_reason == ""
    assert report.budget_errors == ()
