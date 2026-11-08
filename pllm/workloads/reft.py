"""LoReFT and constant activation edits via checkpoint-backed state."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

import pllm


def load_loreft_interventions(
    directory: str | Path, layers: tuple[int, ...], device: torch.device, dtype: torch.dtype
) -> pllm.io.StateImport[torch.Tensor]:
    """Read pyreft checkpoints once into the original shared publication input."""
    loaded: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for path in sorted(Path(directory).glob("intkey_*.bin")):
        match = re.fullmatch(r"intkey_layer\.(\d+)\.comp\.block_output\..*\.bin", path.name)
        if match is None or int(match[1]) not in layers:
            continue
        layer = int(match[1])
        if layer in loaded:
            raise ValueError(f"multiple intervention checkpoints for layer {layer}")
        weight, bias, rotation = pllm.io.load_tensor_fields(path, ("weight", "bias", "rotate_layer"))
        loaded[layer] = (
            weight.to(device=device, dtype=dtype),
            bias.to(device=device, dtype=dtype),
            rotation.to(device=device, dtype=dtype),
        )
    if set(loaded) != set(layers):
        raise FileNotFoundError("missing requested LoReFT layer checkpoints")
    return pllm.io.StateImport(tuple(value for layer in layers for value in loaded[layer]))


@dataclass(frozen=True, slots=True)
class ReFTConfig:
    layers: tuple[int, ...]
    rank: int = 4
    first_tokens: int = 0
    last_tokens: int = 2
    operator: Literal["low_rank_affine", "add"] = "low_rank_affine"
    scope: Literal["instance", "request"] = "instance"

    def __post_init__(self) -> None:
        if (
            not self.layers
            or tuple(sorted(set(self.layers))) != self.layers
            or min(self.first_tokens, self.last_tokens) < 0
        ):
            raise ValueError("activation edits require sorted unique layers and nonnegative token bounds")


@dataclass(frozen=True)
class ReFTWorkload(pllm.Workload):
    config: ReFTConfig

    @property
    def intervention(self) -> pllm.InstanceState:
        rank = self.config.rank
        fields = (
            pllm.Field(lambda in_features, out_features: (rank, in_features)),
            pllm.Field((rank,)),
            pllm.Field(lambda in_features, out_features: (out_features, rank)),
        )
        if self.config.operator == "add":
            fields = (pllm.Field(lambda in_features, out_features: (in_features,)),)
        state = pllm.RequestState if self.config.scope == "request" else pllm.InstanceState
        return state(
            {f"layer_{layer}_{offset}": field for layer in self.config.layers for offset, field in enumerate(fields)},
            init=pllm.Imported(),
        )

    @property
    def patch(self) -> pllm.model.ActivationPatch:
        return pllm.model.ActivationPatch(
            "intervention", self.config.layers, self.config.first_tokens, self.config.last_tokens, self.config.operator
        )
