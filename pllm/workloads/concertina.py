"""Concertina folded-layer workload: shared linear blocks with per-step norms and gated low-rank corrections.

Concertina (arXiv:2611.18113) folds each group of N adjacent decoder layers into one shared linear backbone W0
executed N times with per-step RMSNorm weights and per-token gated rank-r step corrections:

    W(x, t) = W0 + sum_{k=1}^{N-1} g_k(x, t) B_k A_k,    g(x, t) = softmax(x Wg[t] + b[t]).

In pLLM this maps cleanly onto an unmodified native N-folded decoder (e.g., 36-layer Qwen3ForCausalLM for
Concertina-2.27B):
- Each logical step keeps its native decoder layer's RMSNorm weights (including q_norm/k_norm) and KV cache slot.
- `pllm.model.SharedLayers` ties linear weights (qkv_proj, o_proj, gate_up_proj, down_proj) across each group at
  model load so GPU weight memory stays at the folded 2.27B footprint rather than expanding to 4.02B.
- `pllm.model.Adapter(..., operator=concertina_gated_low_rank, share={"a": groups, "b": groups})` shares A/B
  storage across each group via PTTS FieldViews and runs a fused Triton gate+softmax+scale kernel inside vLLM's
  native CUDA graphs.
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Self

import torch
import triton
import triton.language as tl
from cattrs import structure
from safetensors import safe_open
from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]
from torch.utils.flop_counter import register_flop_formula  # pyright: ignore[reportUnknownVariableType]

import pllm
from pllm.model import Projection

ConcertinaProjectionName = Literal["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

CONCERTINA_PROJECTIONS: tuple[ConcertinaProjectionName, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_PROJ_PATHS: dict[ConcertinaProjectionName, str] = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}
_DEFAULT_NORMS: tuple[str, ...] = (
    "input_layernorm",
    "post_attention_layernorm",
    "self_attn.q_norm",
    "self_attn.k_norm",
)
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


@triton.jit
def _concertina_softmax_scale_kernel(
    both_ptr: torch.Tensor,
    bg_ptr: torch.Tensor,
    step_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    stride_bm: int,
    stride_om: int,
    num_gates: tl.constexpr,
    num_experts: tl.constexpr,
    rank: tl.constexpr,
    drop_zero: tl.constexpr,
    g_pad: tl.constexpr,
    block_kr: tl.constexpr,
) -> None:
    """Fused bias + fp32 softmax + optional zero-expert slice + mid scaling on packed [A; Wg_0; ...; Wg_{N-1}] output."""
    row = tl.program_id(0)
    total_kr = num_experts * rank
    step = tl.load(step_ptr)
    g_offs = tl.arange(0, g_pad)
    g_mask = g_offs < num_gates
    gate_col = total_kr + step * num_gates + g_offs
    raw = tl.load(both_ptr + row * stride_bm + gate_col, mask=g_mask, other=0.0).to(tl.float32)
    bg = tl.load(bg_ptr + g_offs, mask=g_mask, other=0.0).to(tl.float32)
    logits = tl.where(g_mask, raw + bg, -float("inf"))
    shifted = logits - tl.max(logits, axis=0)
    probs = tl.exp(shifted)
    probs = tl.where(g_mask, probs, 0.0)
    probs = probs / tl.sum(probs, axis=0)
    for kr_start in range(0, total_kr, block_kr):
        kr_offs = kr_start + tl.arange(0, block_kr)
        kr_mask = kr_offs < total_kr
        mid_val = tl.load(both_ptr + row * stride_bm + kr_offs, mask=kr_mask, other=0.0)
        gates = probs.to(mid_val.dtype)
        k_idx = kr_offs // rank + drop_zero
        selected = tl.sum(
            tl.where(g_offs[:, None] == k_idx[None, :], gates[:, None], 0.0),
            axis=0,
        ).to(mid_val.dtype)
        tl.store(out_ptr + row * stride_om + kr_offs, mid_val * selected, mask=kr_mask)


def _concertina_gated_low_rank_ref(
    hidden: torch.Tensor,
    a_gates: torch.Tensor,
    b: torch.Tensor,
    gate_bias: torch.Tensor,
    step_index: torch.Tensor,
) -> torch.Tensor:
    """Reference PyTorch implementation of Concertina's per-token gated low-rank projection delta.

    Shapes:
      hidden:     (..., in_features)
      a_gates:    (experts * rank + fold_group * gates, in_features)  # group-shared [A; Wg_0; ...; Wg_{N-1}]
      b:          (out_features, experts, rank)                        # group-shared zero-copy layout
      gate_bias:  (gates,)                                             # per-layer bias, gates in (experts, experts + 1)
      step_index: () int64                                             # per-layer step within group
    """
    dtype = hidden.dtype
    out_features, experts, rank = b.shape
    num_gates = gate_bias.shape[0]
    total_kr = experts * rank
    both = torch.matmul(hidden, a_gates.to(dtype).transpose(-1, -2))
    mid = both[..., :total_kr]
    all_logits = both[..., total_kr:].reshape(*hidden.shape[:-1], -1, num_gates)
    logits = all_logits.index_select(-2, step_index.reshape(1)).squeeze(-2) + gate_bias.to(dtype)
    gates = torch.softmax(logits.float(), dim=-1).to(dtype)
    if num_gates == experts + 1:
        gates = gates[..., 1:]
    b_flat = b.reshape(out_features, total_kr).to(dtype)
    mid = (mid.reshape(*hidden.shape[:-1], experts, rank) * gates.unsqueeze(-1)).reshape(*hidden.shape[:-1], total_kr)
    return torch.matmul(mid, b_flat.transpose(-1, -2))


@torch.library.custom_op("concertina::gated_low_rank", mutates_args=())
def concertina_gated_low_rank(
    hidden: torch.Tensor,
    a_gates: torch.Tensor,
    b: torch.Tensor,
    gate_bias: torch.Tensor,
    step_index: torch.Tensor,
) -> torch.Tensor:
    """Concertina gated low-rank delta (CPU reference and default fallback)."""
    return _concertina_gated_low_rank_ref(hidden, a_gates, b, gate_bias, step_index)


def _concertina_gated_low_rank_cuda(
    hidden: torch.Tensor,
    a_gates: torch.Tensor,
    b: torch.Tensor,
    gate_bias: torch.Tensor,
    step_index: torch.Tensor,
) -> torch.Tensor:
    """CUDA implementation: 1 fused down+gate GEMM + 1 lightweight Triton softmax/scale kernel + 1 up GEMM."""
    if hidden.ndim != 2 or hidden.dtype == torch.float64:
        return _concertina_gated_low_rank_ref(hidden, a_gates, b, gate_bias, step_index)
    rows = hidden.shape[0]
    out_features, experts, rank = b.shape
    num_gates = gate_bias.shape[0]
    if rows == 0:
        return hidden.new_empty((0, out_features))
    dtype = hidden.dtype
    hidden_c = hidden if hidden.stride(-1) == 1 else hidden.contiguous()
    ag_c = a_gates.to(dtype)
    b_flat = b.reshape(out_features, experts * rank).to(dtype)
    bg_c = gate_bias.to(dtype).contiguous()
    both = torch.matmul(hidden_c, ag_c.transpose(0, 1))
    total_kr = experts * rank
    mid_gated = hidden_c.new_empty((rows, total_kr))
    g_pad = 1 << (num_gates - 1).bit_length()
    if g_pad < 2:
        g_pad = 2
    drop_zero = num_gates - experts
    block_kr = min(256, max(16, 1 << (total_kr - 1).bit_length()))
    _concertina_softmax_scale_kernel[(rows,)](
        both,
        bg_c,
        step_index,
        mid_gated,
        both.stride(0),
        mid_gated.stride(0),
        num_gates=num_gates,
        num_experts=experts,
        rank=rank,
        drop_zero=drop_zero,
        g_pad=g_pad,
        block_kr=block_kr,
    )
    return torch.matmul(mid_gated, b_flat.transpose(0, 1))


concertina_gated_low_rank.register_kernel("cuda", _concertina_gated_low_rank_cuda)  # pyright: ignore[reportUnknownMemberType]


def _concertina_gated_low_rank_fake(
    hidden: torch.Tensor,
    a_gates: torch.Tensor,
    b: torch.Tensor,
    gate_bias: torch.Tensor,
    step_index: torch.Tensor,
) -> torch.Tensor:
    if a_gates.ndim != 2 or b.ndim != 3 or gate_bias.ndim != 1 or step_index.ndim != 0:
        raise ValueError(
            "concertina_gated_low_rank expects 2-D a_gates, 3-D b, 1-D gate_bias and 0-D step_index; "
            f"got {tuple(a_gates.shape)}, {tuple(b.shape)}, {tuple(gate_bias.shape)}, {tuple(step_index.shape)}"
        )
    out_features, experts, rank = b.shape
    in_features = a_gates.shape[1]
    gates = gate_bias.shape[0]
    gate_rows = a_gates.shape[0] - experts * rank
    if (
        hidden.shape[-1] != in_features
        or gates not in (experts, experts + 1)
        or gate_rows < gates
        or gate_rows % gates != 0
        or step_index.dtype != torch.int64
    ):
        raise ValueError(
            "concertina_gated_low_rank expects hidden (..., in), a_gates (K*r + N*G, in), b (out, K, r), "
            f"gate_bias (G,) with G in (K, K+1), step_index () int64; got {tuple(hidden.shape)}, "
            f"{tuple(a_gates.shape)}, {tuple(b.shape)}, {tuple(gate_bias.shape)}, {tuple(step_index.shape)}"
        )
    return hidden.new_empty((*hidden.shape[:-1], out_features))


concertina_gated_low_rank.register_fake(_concertina_gated_low_rank_fake)  # pyright: ignore[reportUnknownMemberType]


class _VmapInfo(Protocol):
    """Opaque batching metadata passed by torch.library.register_vmap."""

    batch_size: int


def _concertina_gated_low_rank_vmap(
    info: _VmapInfo,
    in_dims: tuple[int | None, ...],
    hidden: torch.Tensor,
    a_gates: torch.Tensor,
    b: torch.Tensor,
    gate_bias: torch.Tensor,
    step_index: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Batch rule for torch.vmap over RequestState block gathers."""
    batch = info.batch_size
    h_b = hidden.movedim(in_dims[0], 0) if in_dims[0] is not None else hidden.unsqueeze(0).expand(batch, *hidden.shape)
    ag_b = (
        a_gates.movedim(in_dims[1], 0) if in_dims[1] is not None else a_gates.unsqueeze(0).expand(batch, *a_gates.shape)
    )
    b_b = b.movedim(in_dims[2], 0) if in_dims[2] is not None else b.unsqueeze(0).expand(batch, *b.shape)
    gb_b = (
        gate_bias.movedim(in_dims[3], 0)
        if in_dims[3] is not None
        else gate_bias.unsqueeze(0).expand(batch, *gate_bias.shape)
    )
    st_b = (
        step_index.movedim(in_dims[4], 0)
        if in_dims[4] is not None
        else step_index.unsqueeze(0).expand(batch, *step_index.shape)
    )
    dtype = h_b.dtype
    out_features, experts, rank = b_b.shape[1:]
    num_gates = gb_b.shape[-1]
    total_kr = experts * rank
    both = torch.bmm(h_b, ag_b.to(dtype).transpose(-1, -2))
    mid = both[:, :, :total_kr]
    rows = h_b.shape[1]
    all_logits = both[:, :, total_kr:].reshape(batch, rows, -1, num_gates)
    gather_idx = st_b.reshape(batch, 1, 1, 1).expand(batch, rows, 1, num_gates)
    logits = all_logits.gather(2, gather_idx).squeeze(2) + gb_b.to(dtype).unsqueeze(1)
    gates = torch.softmax(logits.float(), dim=-1).to(dtype)
    if num_gates == experts + 1:
        gates = gates[..., 1:]
    b_flat = b_b.reshape(batch, out_features, total_kr).to(dtype)
    mid = (mid.reshape(batch, rows, experts, rank) * gates.unsqueeze(-1)).reshape(batch, rows, total_kr)
    return torch.bmm(mid, b_flat.transpose(-1, -2)), 0


concertina_gated_low_rank.register_vmap(_concertina_gated_low_rank_vmap)  # pyright: ignore[reportUnknownMemberType]


def _concertina_gated_low_rank_flops(
    hidden_shape: tuple[int, ...],
    a_gates_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    gate_bias_shape: tuple[int, ...],
    step_index_shape: tuple[int, ...],
    *,
    out_shape: tuple[int, ...] | None = None,
) -> int:
    """Exact matmul FLOPs for packed [A; Wg_0..Wg_{N-1}] down-projection + up-projection B."""
    del gate_bias_shape, step_index_shape, out_shape
    rows = 1
    for dim in hidden_shape[:-1]:
        rows *= dim
    ag_rows, in_features = a_gates_shape
    out_features, experts, rank = b_shape
    return 2 * rows * (in_features * ag_rows + experts * rank * out_features)


register_flop_formula(torch.ops.concertina.gated_low_rank)(_concertina_gated_low_rank_flops)


@dataclass(frozen=True)
class _ConcertinaJsonConfig:
    """Typed subset of Concertina's config.json parsed via cattrs."""

    num_hidden_layers: int
    fold_group: int = 2
    fold_rank: int = 64
    fold_gate: bool = True
    fold_experts_per_group: int | None = None
    fold_norms: tuple[str, ...] = _DEFAULT_NORMS
    fold_bias_projections: tuple[ConcertinaProjectionName, ...] = ()
    fold_source_folded: bool = True
    architectures: tuple[str, ...] = ()
    model_type: str = ""
    base_architecture: str = ""
    base_model_type: str = ""


@dataclass(frozen=True)
class _SafetensorsIndex:
    """Typed view of HuggingFace model.safetensors.index.json."""

    weight_map: dict[str, str]


@dataclass(frozen=True, slots=True)
class ConcertinaConfig:
    """Fold geometry, checkpoint key layout and adapter artifact location for a Concertina checkpoint."""

    num_layers: int
    fold_group: int = 2
    fold_rank: int = 64
    fold_gate: bool = True
    experts_per_group: int | None = None
    adapter_path: str = ""
    norms: tuple[str, ...] = _DEFAULT_NORMS
    bias_projections: tuple[ConcertinaProjectionName, ...] = ()
    from_folded_checkpoint: bool = False

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
    def expert_indices(self) -> tuple[int, ...]:
        """Expert step indices stored inside each folded group's checkpoint."""
        start = 0 if self.num_experts == self.fold_group else 1
        return tuple(range(start, self.fold_group))

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

    @property
    def checkpoint_rename(self) -> dict[str, str]:
        """Explicit 1-to-1 rename map from folded Concertina safetensors keys to native decoder names."""
        if not self.from_folded_checkpoint:
            return {}
        rename: dict[str, str] = {}
        bias_set = set(self.bias_projections)
        for group in self.groups:
            start = group[0]
            for proj in CONCERTINA_PROJECTIONS:
                path = _PROJ_PATHS[proj]
                rename[f"model.layers.{start}.shared.{path}.base.weight"] = f"model.layers.{start}.{path}.weight"
                if proj in bias_set:
                    rename[f"model.layers.{start}.shared.{path}.base.bias"] = f"model.layers.{start}.{path}.bias"
            for step, layer in enumerate(group):
                for norm in self.norms:
                    rename[f"model.layers.{start}.shared.{norm}.weight.{step}"] = f"model.layers.{layer}.{norm}.weight"
        return rename

    @property
    def checkpoint_exclude(self) -> tuple[str, ...]:
        """Explicit list of folded adapter keys loaded separately via pllm.Pretrained."""
        if not self.from_folded_checkpoint:
            return ()
        excluded: list[str] = []
        experts = self.expert_indices
        for group in self.groups:
            start = group[0]
            for proj in CONCERTINA_PROJECTIONS:
                base = f"model.layers.{start}.shared.{_PROJ_PATHS[proj]}"
                for k in experts:
                    excluded.append(f"{base}.A.{k}")
                    excluded.append(f"{base}.B.{k}")
                if self.fold_gate:
                    for step in range(len(group)):
                        excluded.append(f"{base}.Wg.{step}")
                        excluded.append(f"{base}.gb.{step}")
        return tuple(excluded)

    @classmethod
    def from_checkpoint(cls, model_dir: str, *, adapter_path: str = "") -> Self:
        """Read fold geometry from a prepared Concertina checkpoint directory; fail closed if adapters are missing."""
        root = Path(model_dir)
        parsed = structure(json.loads((root / "config.json").read_text(encoding="utf-8")), _ConcertinaJsonConfig)
        resolved = adapter_path or str(root / _ADAPTER_RELPATH)
        if not Path(resolved).is_file():
            raise FileNotFoundError(
                f"Concertina adapter file not found at {resolved!r}; run prepare_concertina_checkpoint(...) first "
                "or pass an explicit adapter_path="
            )
        return cls(
            num_layers=parsed.num_hidden_layers,
            fold_group=parsed.fold_group,
            fold_rank=parsed.fold_rank,
            fold_gate=parsed.fold_gate,
            experts_per_group=parsed.fold_experts_per_group,
            adapter_path=resolved,
            norms=parsed.fold_norms,
            bias_projections=parsed.fold_bias_projections,
            from_folded_checkpoint=parsed.fold_source_folded,
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
        index_data = structure(json.loads(index_path.read_text(encoding="utf-8")), _SafetensorsIndex)
        shards = tuple(sorted({source_dir / shard for shard in index_data.weight_map.values()}))
        return {name: source_dir / shard for name, shard in index_data.weight_map.items()}, shards
    single = source_dir / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"No model.safetensors or model.safetensors.index.json found in {source_dir}")
    with safe_open(single, framework="pt", device="cpu") as handle:
        return {name: single for name in handle.keys()}, (single,)


def _detect_folded_norms(tensor_files: dict[str, Path]) -> tuple[str, ...]:
    """Detect per-step norm parameter paths present under model.layers.0.shared.*."""
    prefix = "model.layers.0.shared."
    norms = sorted(
        {
            name[len(prefix) :].rpartition(".weight.")[0]
            for name in tensor_files
            if name.startswith(prefix) and name.endswith(".weight.0")
        }
    )
    return tuple(norms) if norms else _DEFAULT_NORMS


def _detect_bias_projections(tensor_files: dict[str, Path]) -> tuple[ConcertinaProjectionName, ...]:
    """Detect folded linear projections that also carry a base bias tensor."""
    return tuple(
        proj
        for proj in CONCERTINA_PROJECTIONS
        if f"model.layers.0.shared.{_PROJ_PATHS[proj]}.base.bias" in tensor_files
    )


def _resolve_base_architecture(
    parsed: _ConcertinaJsonConfig, architecture: str | None, model_type: str | None
) -> tuple[str, str]:
    """Infer the native vLLM decoder architecture and model_type without hardcoding a single family."""
    resolved_arch = architecture or parsed.base_architecture
    if not resolved_arch:
        non_folded = next((arch for arch in parsed.architectures if "Concertina" not in arch), "")
        resolved_arch = non_folded or "Qwen3ForCausalLM"
    resolved_type = model_type or parsed.base_model_type
    if not resolved_type:
        resolved_type = parsed.model_type if parsed.model_type and parsed.model_type != "concertina" else "qwen3"
    return resolved_arch, resolved_type


def prepare_concertina_checkpoint(
    source: str,
    output_dir: str,
    *,
    revision: str | None = None,
    copy_weights: bool = False,
    architecture: str | None = None,
    model_type: str | None = None,
) -> tuple[str, ConcertinaConfig]:
    """Prepare a HF Concertina checkpoint for native vLLM loading with zero-copy base weight symlinks.

    Writes a native decoder config.json (removing `auto_map`), symlinks the original safetensors without
    duplicating base weights, and extracts compact adapters into `pllm_adapters/concertina_adapters.safetensors`
    with group-shared A/B factors (stored once per folded group) and zero-copy B layout `(out, experts, rank)`.
    """
    src = _resolve_source_dir(source, revision=revision)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    raw_text = (src / "config.json").read_text(encoding="utf-8")
    parsed = structure(json.loads(raw_text), _ConcertinaJsonConfig)
    tensor_files, shards = _load_safetensors_index(src)

    sample_prefix = f"model.layers.0.shared.{_PROJ_PATHS[CONCERTINA_PROJECTIONS[0]]}"
    has_step0_expert = f"{sample_prefix}.A.0" in tensor_files
    expert_indices = tuple(range(0 if has_step0_expert else 1, parsed.fold_group))
    experts_per_group = len(expert_indices) if has_step0_expert else None
    norms = _detect_folded_norms(tensor_files)
    bias_projections = _detect_bias_projections(tensor_files)
    resolved_arch, resolved_type = _resolve_base_architecture(parsed, architecture, model_type)

    vllm_config: dict[str, Any] = json.loads(raw_text)
    vllm_config["architectures"] = [resolved_arch]
    vllm_config["model_type"] = resolved_type
    vllm_config.pop("auto_map", None)
    vllm_config["fold_norms"] = list(norms)
    vllm_config["fold_bias_projections"] = list(bias_projections)
    vllm_config["fold_source_folded"] = True
    if experts_per_group is not None:
        vllm_config["fold_experts_per_group"] = experts_per_group
    (dst / "config.json").write_text(json.dumps(vllm_config, indent=2) + "\n", encoding="utf-8")

    for filename in _SMALL_CONFIG_FILES:
        src_file = src / filename
        if src_file.is_file():
            shutil.copy2(src_file, dst / filename)

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

    adapter_file = dst / _ADAPTER_RELPATH
    adapter_file.parent.mkdir(parents=True, exist_ok=True)
    config = ConcertinaConfig(
        num_layers=parsed.num_hidden_layers,
        fold_group=parsed.fold_group,
        fold_rank=parsed.fold_rank,
        fold_gate=parsed.fold_gate,
        experts_per_group=experts_per_group,
        adapter_path=str(adapter_file),
        norms=norms,
        bias_projections=bias_projections,
        from_folded_checkpoint=True,
    )

    with ExitStack() as stack:
        handles = {shard: stack.enter_context(safe_open(shard, framework="pt", device="cpu")) for shard in shards}
        adapter_tensors: dict[str, torch.Tensor] = {}
        layer_to_group = {layer: (group[0], step) for group in config.groups for step, layer in enumerate(group)}
        for target_index, target in enumerate(config.targets):
            assert target.layer is not None
            start, step = layer_to_group[target.layer]
            base_key = f"model.layers.{start}.shared.{_PROJ_PATHS[target.name]}"
            if config.fold_gate:
                # Pack [A; Wg_0; ...; Wg_{N-1}] and B once per group via Adapter(share={"a_gates": groups, "b": groups}).
                if step == 0:
                    a_list = [
                        handles[tensor_files[f"{base_key}.A.{k}"]].get_tensor(f"{base_key}.A.{k}")
                        for k in expert_indices
                    ]
                    wg_list = [
                        handles[tensor_files[f"{base_key}.Wg.{s}"]].get_tensor(f"{base_key}.Wg.{s}").transpose(0, 1)
                        for s in range(config.fold_group)
                    ]
                    b_list = [
                        handles[tensor_files[f"{base_key}.B.{k}"]].get_tensor(f"{base_key}.B.{k}")
                        for k in expert_indices
                    ]
                    adapter_tensors[f"a_gates_{target_index}"] = torch.cat([*a_list, *wg_list], dim=0).contiguous()
                    # Zero-copy layout (out_features, experts, rank) avoids per-step permute copies.
                    adapter_tensors[f"b_{target_index}"] = torch.stack(b_list, dim=1).contiguous()
                gb = handles[tensor_files[f"{base_key}.gb.{step}"]].get_tensor(f"{base_key}.gb.{step}")
                adapter_tensors[f"gate_bias_{target_index}"] = gb.contiguous()
                adapter_tensors[f"step_index_{target_index}"] = torch.tensor(step, dtype=torch.int64)
            else:
                a_key = f"{base_key}.A.{step}"
                b_key = f"{base_key}.B.{step}"
                adapter_tensors[f"a_{target_index}"] = handles[tensor_files[a_key]].get_tensor(a_key).contiguous()
                adapter_tensors[f"b_{target_index}"] = handles[tensor_files[b_key]].get_tensor(b_key).contiguous()
        save_file(adapter_tensors, config.adapter_path)

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
        return pllm.model.SharedLayers(
            self.config.groups if self.share_weights else (),
            checkpoint_rename=self.config.checkpoint_rename,
            checkpoint_exclude=self.config.checkpoint_exclude,
        )

    @property
    def adapters(self) -> pllm.InstanceState:
        """Frozen low-rank step corrections (and per-token gates when fold_gate=True)."""
        init = pllm.Pretrained(self.config.adapter_path) if self.config.adapter_path else pllm.Zeros()
        rank = self.config.fold_rank
        if self.config.fold_gate:
            experts = self.config.num_experts
            gates = self.config.fold_group
            ag_rows = experts * rank + self.config.fold_group * gates
            return pllm.InstanceState(
                {
                    "a_gates": pllm.Field(lambda inputs, _outputs: (ag_rows, inputs)),
                    "b": pllm.Field(lambda _inputs, outputs: (outputs, experts, rank)),
                    "gate_bias": pllm.Field(lambda _inputs, _outputs: (gates,)),
                    "step_index": pllm.Field((), torch.int64),
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
                operator=concertina_gated_low_rank,
                factors=("a_gates", "b", "gate_bias", "step_index"),
                scale=self.scale,
                share={"a_gates": self.config.groups, "b": self.config.groups},
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
    "concertina_gated_low_rank",
    "prepare_concertina_checkpoint",
]
