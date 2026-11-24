"""Register vLLM's ordinary loader with exact exclusions, explicit 1-to-1 renames and layer weight tying."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import replace
from typing import Any

import torch
from cattrs import structure
from vllm.config import LoadConfig, ModelConfig
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
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

_LoaderExtraValue = tuple[str, ...] | tuple[tuple[int, ...], ...] | tuple[tuple[str, str], ...]


def load_options(
    names: tuple[str, ...],
    *,
    shared_layer_groups: tuple[tuple[int, ...], ...] = (),
    checkpoint_rename: tuple[tuple[str, str], ...] = (),
    checkpoint_exclude: tuple[str, ...] = (),
) -> dict[str, str | dict[str, _LoaderExtraValue]]:
    """Carry workload exclusions, explicit renames and shared-layer groups through vLLM's loader configuration."""
    if not names and not shared_layer_groups and not checkpoint_rename and not checkpoint_exclude:
        return {}
    extra: dict[str, _LoaderExtraValue] = {"checkpoint_names": names}
    if shared_layer_groups:
        extra["shared_layer_groups"] = shared_layer_groups
    if checkpoint_rename:
        extra["checkpoint_rename"] = checkpoint_rename
    if checkpoint_exclude:
        extra["checkpoint_exclude"] = checkpoint_exclude
    return {"load_format": "pllm", "model_loader_extra_config": extra}


def _tie_linear(leader: LinearBase, follower: LinearBase) -> None:
    """Point every parameter of an unquantized follower linear module at its leader's GPU storage."""
    if not isinstance(leader.quant_method, UnquantizedLinearMethod) or not isinstance(
        follower.quant_method, UnquantizedLinearMethod
    ):
        raise ValueError(
            "pllm.model.SharedLayers requires unquantized linear layers; "
            f"got {type(leader.quant_method).__name__} and {type(follower.quant_method).__name__}"
        )
    leader_params = dict(leader.named_parameters(recurse=False))
    follower_params = dict(follower.named_parameters(recurse=False))
    if set(leader_params) != set(follower_params) or not leader_params:
        raise ValueError(
            "pllm.model.SharedLayers requires identical parameter sets on tied linear layers; "
            f"got {sorted(leader_params)} and {sorted(follower_params)}"
        )
    for name, parameter in leader_params.items():
        follower.register_parameter(name, parameter)


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
    """Load base tensors normally, honoring workload exclusions, explicit 1-to-1 renames and shared layers."""

    def __init__(self, load_config: LoadConfig) -> None:
        options: dict[str, Any] = dict(load_config.model_loader_extra_config)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        self.excluded = frozenset(structure(options.pop("checkpoint_names", ()), tuple[str, ...]))
        self.shared_layer_groups = structure(options.pop("shared_layer_groups", ()), tuple[tuple[int, ...], ...])
        self.renamed = dict(structure(options.pop("checkpoint_rename", ()), tuple[tuple[str, str], ...]))
        self.skipped = frozenset(structure(options.pop("checkpoint_exclude", ()), tuple[str, ...]))
        super().__init__(replace(load_config, load_format="auto", model_loader_extra_config=options))

    def load_weights(self, model: torch.nn.Module, model_config: ModelConfig) -> None:
        """Tie shared linear parameters before loading so follower storage is freed immediately."""
        tie_shared_layers(model, self.shared_layer_groups)
        super().load_weights(model, model_config)  # pyright: ignore[reportUnknownMemberType]

    def get_all_weights(
        self, model_config: ModelConfig, model: torch.nn.Module
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Retain vLLM's sharding, quantization and name mapping while applying explicit exclusions and renames."""
        super().download_model(model_config)
        if self.excluded:
            checkpoint = checkpoint_directory(model_config.model, model_config.revision)
            checkpoint_files(checkpoint, tuple(sorted(self.excluded)))
        followers = frozenset(layer for group in self.shared_layer_groups for layer in group[1:])
        for name, tensor in super().get_all_weights(model_config, model):
            if name in self.excluded or name in self.skipped:
                continue
            mapped = self.renamed.get(name, name)
            if _is_follower_linear(mapped, followers):
                continue
            yield mapped, tensor
