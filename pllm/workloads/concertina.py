"""Concertina folded-layer workload: shared linear blocks with per-step norms and gated low-rank corrections.

Concertina (arXiv:2611.18113) folds each group of N adjacent decoder layers into one shared linear backbone W0
executed N times with per-step RMSNorm weights and per-token gated rank-r step corrections:

    W(x, t) = W0 + sum_{k=1}^{N-1} g_k(x, t) B_k A_k,    g(x, t) = softmax(x Wg[t] + b[t]).

In pLLM this maps cleanly onto a native N-folded decoder (e.g., 36-layer Qwen3ForCausalLM for Concertina-2.27B):
- Each logical step keeps its native decoder layer's RMSNorm weights (including q_norm/k_norm) and KV cache slot.
- `pllm.model.SharedLayers` ties linear weights (qkv_proj, o_proj, gate_up_proj, down_proj) across each group at
  model load so GPU weight memory stays at the folded 2.27B footprint rather than expanding to 4.02B.
- `pllm.model.Adapter(..., operator="gated_low_rank")` applies the token-gated step corrections across all seven
  projections inside vLLM's native CUDA graphs.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import torch
from safetensors import safe_open
from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]

import pllm
from pllm.contracts.parameter_overlay import Projection, ProjectionName

CONCERTINA_PROJECTIONS: tuple[ProjectionName, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_PROJ_PATHS: dict[ProjectionName, str] = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}
_ADAPTER_RELPATH = Path("pllm_adapters") / "concertina_adapters.safetensors"
_SMALL_CONFIG_FILES = (
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "chat_template.jinja",
)


@dataclass(frozen=True, slots=True)
class ConcertinaConfig:
    """Fold geometry and adapter artifact location for a Concertina checkpoint."""

    num_layers: int
    fold_group: int = 2
    fold_rank: int = 64
    fold_gate: bool = True
    experts_per_group: int | None = None
    adapter_path: str = ""

    def __post_init__(self) -> None:
        if self.num_layers < 2:
            raise ValueError(f"ConcertinaConfig needs num_layers >= 2; got {self.num_layers}")
        if self.fold_group < 2 or self.fold_group > self.num_layers:
            raise ValueError(f"ConcertinaConfig needs 2 <= fold_group <= num_layers; got {self.fold_group}")
        if self.fold_rank < 1:
            raise ValueError(f"ConcertinaConfig needs fold_rank >= 1; got {self.fold_rank}")
        if self.experts_per_group is not None and self.experts_per_group not in (self.fold_group - 1, self.fold_group):
            raise ValueError(
                f"ConcertinaConfig needs experts_per_group in ({self.fold_group - 1}, {self.fold_group}); "
                f"got {self.experts_per_group}"
            )

    @property
    def num_experts(self) -> int:
        """Number of low-rank correction experts per folded group (N - 1 for asymmetric Concertina)."""
        return self.fold_group - 1 if self.experts_per_group is None else self.experts_per_group

    @property
    def groups(self) -> tuple[tuple[int, ...], ...]:
        """Layer index groups that share linear projection weights W0."""
        folded = self.num_layers - (self.num_layers % self.fold_group)
        return tuple(tuple(range(start, start + self.fold_group)) for start in range(0, folded, self.fold_group))

    @property
    def adapted_layers(self) -> tuple[int, ...]:
        """Logical layers that carry low-rank adapters (all folded layers when gated; steps >= 1 when static)."""
        if self.fold_gate or self.num_experts == self.fold_group:
            return tuple(layer for group in self.groups for layer in group)
        return tuple(layer for group in self.groups for layer in group[1:])

    @property
    def targets(self) -> tuple[Projection, ...]:
        """Sorted unique projection targets across all adapted layers."""
        return tuple(
            sorted(Projection(name, layer) for layer in self.adapted_layers for name in CONCERTINA_PROJECTIONS)
        )

    @classmethod
    def from_checkpoint(cls, model_dir: str, *, adapter_path: str = "") -> Self:
        """Read fold geometry from a prepared or native Concertina checkpoint directory."""
        root = Path(model_dir)
        raw = json.loads((root / "config.json").read_text(encoding="utf-8"))
        resolved = adapter_path
        if not resolved and (candidate := root / _ADAPTER_RELPATH).is_file():
            resolved = str(candidate)
        raw_experts = raw.get("fold_experts_per_group")
        return cls(
            num_layers=int(raw["num_hidden_layers"]),
            fold_group=int(raw.get("fold_group", 2)),
            fold_rank=int(raw.get("fold_rank", 64)),
            fold_gate=bool(raw.get("fold_gate", True)),
            experts_per_group=None if raw_experts is None else int(raw_experts),
            adapter_path=resolved,
        )


def _resolve_source_dir(source: str, revision: str | None = None) -> Path:
    path = Path(source)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download  # pyright: ignore[reportUnknownVariableType]

    return Path(snapshot_download(source, revision=revision))


def _load_safetensors_index(source_dir: Path) -> tuple[dict[str, Path], tuple[Path, ...]]:
    index_path = source_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map: dict[str, str] = index_data["weight_map"]
        shards = tuple(sorted({source_dir / shard for shard in weight_map.values()}))
        return {name: source_dir / shard for name, shard in weight_map.items()}, shards
    single = source_dir / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"No model.safetensors or model.safetensors.index.json found in {source_dir}")
    with safe_open(single, framework="pt", device="cpu") as handle:
        return {name: single for name in handle.keys()}, (single,)


def prepare_concertina_checkpoint(
    source: str,
    output_dir: str,
    *,
    revision: str | None = None,
    copy_weights: bool = False,
) -> tuple[str, ConcertinaConfig]:
    """Prepare a HF Concertina checkpoint for native vLLM loading with zero-copy base weight symlinks.

    Writes a standard `Qwen3ForCausalLM` config.json (removing `auto_map`), links the original safetensors
    without duplicating the base weights, and extracts the compact low-rank / gate tensors into
    `concertina_adapters.safetensors` for `ConcertinaWorkload`.
    """
    src = _resolve_source_dir(source, revision=revision)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    raw_config: dict[str, Any] = json.loads((src / "config.json").read_text(encoding="utf-8"))
    num_layers = int(str(raw_config["num_hidden_layers"]))
    fold_group = int(str(raw_config.get("fold_group", 2)))
    fold_rank = int(str(raw_config.get("fold_rank", 64)))
    fold_gate = bool(raw_config.get("fold_gate", True))

    vllm_config = dict(raw_config)
    vllm_config["architectures"] = ["Qwen3ForCausalLM"]
    vllm_config["model_type"] = "qwen3"
    vllm_config.pop("auto_map", None)
    (dst / "config.json").write_text(json.dumps(vllm_config, indent=2) + "\n", encoding="utf-8")

    for filename in _SMALL_CONFIG_FILES:
        src_file = src / filename
        if src_file.is_file():
            shutil.copy2(src_file, dst / filename)

    tensor_files, shards = _load_safetensors_index(src)
    for shard in shards:
        target = dst / shard.name
        if target.exists() or target.is_symlink():
            target.unlink()
        if copy_weights:
            shutil.copy2(shard, target)
        else:
            os.symlink(shard.resolve(), target)
    index_src = src / "model.safetensors.index.json"
    if index_src.is_file():
        shutil.copy2(index_src, dst / index_src.name)

    # Detect whether step 0 also stores an expert (avg-init) or only steps 1..N-1 (asym Concertina).
    sample_prefix = f"model.layers.0.shared.{_PROJ_PATHS[CONCERTINA_PROJECTIONS[0]]}"
    has_step0_expert = f"{sample_prefix}.A.0" in tensor_files
    expert_indices = tuple(range(0 if has_step0_expert else 1, fold_group))
    experts_per_group = len(expert_indices) if has_step0_expert else None
    if experts_per_group is not None:
        vllm_config["fold_experts_per_group"] = experts_per_group
        (dst / "config.json").write_text(json.dumps(vllm_config, indent=2) + "\n", encoding="utf-8")
    adapter_file = dst / _ADAPTER_RELPATH
    adapter_file.parent.mkdir(parents=True, exist_ok=True)
    config = ConcertinaConfig(
        num_layers=num_layers,
        fold_group=fold_group,
        fold_rank=fold_rank,
        fold_gate=fold_gate,
        experts_per_group=experts_per_group,
        adapter_path=str(adapter_file),
    )

    handles = {shard: safe_open(shard, framework="pt", device="cpu") for shard in shards}
    try:
        adapter_tensors: dict[str, torch.Tensor] = {}
        layer_to_group = {layer: (group[0], step) for group in config.groups for step, layer in enumerate(group)}
        for target_index, target in enumerate(config.targets):
            assert target.layer is not None
            start, step = layer_to_group[target.layer]
            base_key = f"model.layers.{start}.shared.{_PROJ_PATHS[target.name]}"
            if config.fold_gate:
                a_list = [
                    handles[tensor_files[f"{base_key}.A.{k}"]].get_tensor(f"{base_key}.A.{k}") for k in expert_indices
                ]
                b_list = [
                    handles[tensor_files[f"{base_key}.B.{k}"]].get_tensor(f"{base_key}.B.{k}") for k in expert_indices
                ]
                wg = handles[tensor_files[f"{base_key}.Wg.{step}"]].get_tensor(f"{base_key}.Wg.{step}")
                gb = handles[tensor_files[f"{base_key}.gb.{step}"]].get_tensor(f"{base_key}.gb.{step}")
                adapter_tensors[f"a_{target_index}"] = torch.stack(a_list, dim=0).contiguous()
                adapter_tensors[f"b_{target_index}"] = torch.stack(b_list, dim=0).contiguous()
                # Concertina stores Wg as (in_features, n_gates); transpose to (n_gates, in_features).
                adapter_tensors[f"gate_weight_{target_index}"] = wg.transpose(0, 1).contiguous()
                adapter_tensors[f"gate_bias_{target_index}"] = gb.contiguous()
            else:
                a_key = f"{base_key}.A.{step}"
                b_key = f"{base_key}.B.{step}"
                adapter_tensors[f"a_{target_index}"] = handles[tensor_files[a_key]].get_tensor(a_key).contiguous()
                adapter_tensors[f"b_{target_index}"] = handles[tensor_files[b_key]].get_tensor(b_key).contiguous()
        save_file(adapter_tensors, config.adapter_path)
    finally:
        del handles

    return str(dst), config


@dataclass(frozen=True)
class ConcertinaWorkload(pllm.Workload):
    """Serve a Concertina folded decoder on unmodified vLLM with shared linear weights and gated step LoRAs."""

    config: ConcertinaConfig
    scale: float = 1.0
    share_weights: bool = True

    @property
    def shared(self) -> pllm.model.SharedLayers:
        """Tie each folded group's linear projection weights to its first layer at model load."""
        return pllm.model.SharedLayers(self.config.groups if self.share_weights else ())

    @property
    def adapters(self) -> pllm.InstanceState:
        """Frozen low-rank step corrections (and per-token gates when fold_gate=True)."""
        init = pllm.Pretrained(self.config.adapter_path) if self.config.adapter_path else pllm.Zeros()
        rank = self.config.fold_rank
        if self.config.fold_gate:
            experts = self.config.num_experts
            gates = self.config.fold_group
            return pllm.InstanceState(
                {
                    "a": pllm.Field(lambda inputs, _outputs: (experts, rank, inputs)),
                    "b": pllm.Field(lambda _inputs, outputs: (experts, outputs, rank)),
                    "gate_weight": pllm.Field(lambda inputs, _outputs: (gates, inputs)),
                    "gate_bias": pllm.Field(lambda _inputs, _outputs: (gates,)),
                },
                init=init,
            )
        return pllm.InstanceState(
            {
                "a": pllm.Field(lambda inputs, _outputs: (rank, inputs)),
                "b": pllm.Field(lambda _inputs, outputs: (outputs, rank)),
            },
            init=init,
        )

    @property
    def overlay(self) -> pllm.model.Adapter:
        """Apply gated or static low-rank step corrections across all seven projections per folded layer."""
        if self.config.fold_gate:
            return pllm.model.Adapter(
                "adapters",
                self.config.targets,
                operator="gated_low_rank",
                factors=("a", "b", "gate_weight", "gate_bias"),
                scale=self.scale,
            )
        return pllm.model.Adapter(
            "adapters",
            self.config.targets,
            operator="low_rank",
            factors=("a", "b"),
            scale=self.scale,
        )


__all__ = [
    "CONCERTINA_PROJECTIONS",
    "ConcertinaConfig",
    "ConcertinaWorkload",
    "prepare_concertina_checkpoint",
]
