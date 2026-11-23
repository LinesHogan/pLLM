"""Register vLLM's ordinary loader with exact exclusions, layer weight tying and folded key remapping."""

from collections.abc import Generator
from dataclasses import replace
from typing import Any

import torch
from cattrs import structure
from vllm.config import LoadConfig, ModelConfig
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from pllm.backends.vllm.architecture import DenseLayer, bind_model
from pllm.execution.checkpoint import checkpoint_directory, checkpoint_files

_LINEAR_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.qkv_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "mlp.gate_up_proj",
)
_NORM_SUFFIXES = (
    "input_layernorm",
    "post_attention_layernorm",
    "self_attn.q_norm",
    "self_attn.k_norm",
)


def load_options(
    names: tuple[str, ...], *, shared_layer_groups: tuple[tuple[int, ...], ...] = ()
) -> dict[str, str | dict[str, tuple[str, ...] | tuple[tuple[int, ...], ...]]]:
    """Carry workload exclusions and shared-layer groups through vLLM's serialized loader configuration."""
    if not names and not shared_layer_groups:
        return {}
    extra: dict[str, tuple[str, ...] | tuple[tuple[int, ...], ...]] = {"checkpoint_names": names}
    if shared_layer_groups:
        extra["shared_layer_groups"] = shared_layer_groups
    return {"load_format": "pllm", "model_loader_extra_config": extra}


def _tie_linear(leader: LinearBase, follower: LinearBase) -> None:
    """Point a follower linear module's weight (and optional bias) at the leader's GPU storage."""
    follower.register_parameter("weight", leader.get_parameter("weight"))
    if leader.bias is not None:
        follower.register_parameter("bias", leader.bias)


def tie_shared_layers(model: torch.nn.Module, groups: tuple[tuple[int, ...], ...]) -> None:
    """Share dense linear weights across each declared group while keeping norms and KV caches per layer."""
    if not groups:
        return
    decoder = bind_model(model).writable("pllm.model.SharedLayers")
    layers = decoder.model.layers
    for group in groups:
        leader = layers[group[0]]
        if not isinstance(leader, DenseLayer):
            raise ValueError(f"pllm.model.SharedLayers needs dense decoder layers; got {type(leader).__name__}")
        for follower_index in group[1:]:
            follower = layers[follower_index]
            if not isinstance(follower, DenseLayer):
                raise ValueError(f"pllm.model.SharedLayers needs dense decoder layers; got {type(follower).__name__}")
            _tie_linear(leader.self_attn.qkv_proj, follower.self_attn.qkv_proj)
            _tie_linear(leader.self_attn.o_proj, follower.self_attn.o_proj)
            _tie_linear(leader.mlp.gate_up_proj, follower.mlp.gate_up_proj)
            _tie_linear(leader.mlp.down_proj, follower.mlp.down_proj)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _remap_folded_weight(name: str, group_starts: dict[int, tuple[int, ...]]) -> tuple[str | None, bool]:
    """Remap Concertina `model.layers.{start}.shared.*` keys to native decoder names; drop adapter tensors."""
    prefix = "model.layers."
    if not name.startswith(prefix):
        return name, False
    rest = name[len(prefix) :]
    layer_str, sep, tail = rest.partition(".")
    if not sep or not layer_str.isdigit() or not tail.startswith("shared."):
        return name, False
    start = int(layer_str)
    group = group_starts.get(start)
    if group is None:
        return None, True
    shared_tail = tail[len("shared.") :]
    if shared_tail.endswith(".base.weight"):
        proj = shared_tail[: -len(".base.weight")]
        if proj in _LINEAR_SUFFIXES:
            return f"model.layers.{start}.{proj}.weight", True
    if ".weight." in shared_tail:
        norm, _, step_str = shared_tail.rpartition(".weight.")
        if norm in _NORM_SUFFIXES and step_str.isdigit():
            step = int(step_str)
            if 0 <= step < len(group):
                return f"model.layers.{group[step]}.{norm}.weight", True
    return None, True


def _is_follower_linear(name: str, followers: frozenset[int]) -> bool:
    """Whether a standard checkpoint weight belongs to a tied follower layer's linear projection."""
    prefix = "model.layers."
    if not followers or not name.startswith(prefix):
        return False
    rest = name[len(prefix) :]
    layer_str, sep, tail = rest.partition(".")
    if not sep or not layer_str.isdigit() or int(layer_str) not in followers:
        return False
    return any(tail in (f"{proj}.weight", f"{proj}.bias") for proj in _LINEAR_SUFFIXES)


@register_model_loader("pllm")
class CheckpointLoader(DefaultModelLoader):
    """Load every base tensor normally, omitting workload-owned tensors and tying shared layer groups."""

    def __init__(self, load_config: LoadConfig) -> None:
        options: dict[str, Any] = dict(load_config.model_loader_extra_config)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        self.excluded = frozenset(structure(options.pop("checkpoint_names", ()), tuple[str, ...]))
        self.shared_layer_groups = structure(options.pop("shared_layer_groups", ()), tuple[tuple[int, ...], ...])
        super().__init__(replace(load_config, load_format="auto", model_loader_extra_config=options))

    def load_weights(self, model: torch.nn.Module, model_config: ModelConfig) -> None:
        """Tie shared linear parameters before loading so follower storage is freed immediately."""
        tie_shared_layers(model, self.shared_layer_groups)
        super().load_weights(model, model_config)  # pyright: ignore[reportUnknownMemberType]

    def get_all_weights(
        self, model_config: ModelConfig, model: torch.nn.Module
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Retain vLLM's sharding, quantization and name mapping while handling exclusions and shared layers."""
        super().download_model(model_config)
        if self.excluded:
            checkpoint = checkpoint_directory(model_config.model, model_config.revision)
            checkpoint_files(checkpoint, tuple(sorted(self.excluded)))
        group_starts = {group[0]: group for group in self.shared_layer_groups}
        followers = frozenset(layer for group in self.shared_layer_groups for layer in group[1:])
        for name, tensor in super().get_all_weights(model_config, model):
            if name in self.excluded:
                continue
            if group_starts:
                mapped, handled = _remap_folded_weight(name, group_starts)
                if handled:
                    if mapped is not None:
                        yield mapped, tensor
                    continue
            if _is_follower_linear(name, followers):
                continue
            yield name, tensor
