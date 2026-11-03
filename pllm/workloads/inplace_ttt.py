"""Official In-Place TTT chunk arithmetic, used with checkpoints containing learned targets."""

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Self

import torch
from cattrs import structure

import pllm
from pllm.model import Projection


@dataclass(frozen=True)
class _CheckpointConfig:
    ttt_mode: bool
    ttt_layers: tuple[int, ...]
    ttt_chunk: int
    ttt_lr: float
    ttt_target: str
    ttt_proj: bool = True


@dataclass(frozen=True, slots=True)
class InPlaceTTTConfig:
    """Checkpoint hyperparameters for hidden-state targets, without paper-only clipping."""

    targets: tuple[Projection, ...]
    chunk_size: int
    learning_rate: float

    def __post_init__(self) -> None:
        if (
            not self.targets
            or tuple(sorted(set(self.targets))) != self.targets
            or any(target.name != "down_proj" or target.layer is None for target in self.targets)
            or self.chunk_size < 1
            or not isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError(
                "InPlaceTTT needs sorted down_proj layers, chunk_size >= 1 and a positive finite learning_rate"
            )

    @classmethod
    def from_checkpoint(cls, model: str, *, revision: str | None = None) -> Self:
        """Read the served checkpoint's layer list, chunk size and learning rate."""
        from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType]

        path = (
            Path(model) / "config.json"
            if Path(model).is_dir()
            else Path(hf_hub_download(model, "config.json", revision=revision))
        )
        config = structure(json.loads(path.read_text()), _CheckpointConfig)
        if not config.ttt_mode or config.ttt_target != "hidden_states" or not config.ttt_proj:
            raise ValueError("InPlaceTTT needs a TTT checkpoint with hidden_states targets and ttt_proj enabled")
        return cls(
            tuple(Projection("down_proj", layer) for layer in config.ttt_layers), config.ttt_chunk, config.ttt_lr
        )


@dataclass(frozen=True)
class InPlaceTTTWorkload(pllm.Workload):
    """Train on every token, carrying partial chunks and exposing only earlier chunks' updates.

    FP32 keeps a separate delta. BF16 follows official fast-weight rounding; its exact difference from the
    frozen base is retained in FP32 so the residual interface loses no additional bits.
    """

    config: InPlaceTTTConfig
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("InPlaceTTT dtype must be torch.float32 or torch.bfloat16 (official arithmetic)")

    @property
    def target_parameters(self) -> pllm.InstanceState:
        """Learned target tensors, shared by requests and selected by projection layer."""
        return pllm.InstanceState(
            {
                "conv": pllm.Field(lambda d, _: (d, 1, 5)),
                "proj": pllm.Field(lambda d, _: (d, d)),
            },
            layers=tuple(target.layer for target in self.config.targets if target.layer is not None),
            checkpoint={
                "conv": "model.layers.{layer}.mlp.ttt_conv.weight",
                "proj": "model.layers.{layer}.mlp.ttt_proj.weight",
            },
        )

    @property
    def dense_fast_weights(self) -> pllm.RequestState:
        """One zero-initialized delta per request and selected down projection."""
        return pllm.RequestState(
            {"delta": pllm.Field(lambda inputs, outputs: (outputs, inputs))},
            key_fields=("request_id", "prompt_index", "sample_index"),
        )

    @pllm.train_in_forward(
        visibility="exact",
        state="dense_fast_weights",
        targets=pllm.Attr("config.targets"),
        weight="delta",
        chunk=pllm.Attr("config.chunk_size"),
        tokens="all",
    )
    def step(
        self,
        delta: torch.Tensor = pllm.parameter("dense_fast_weights", "delta"),
        inputs: torch.Tensor = pllm.operand("input"),
        targets: torch.Tensor = pllm.operand("target"),
        base: torch.Tensor = pllm.operand("weight"),
        conv: torch.Tensor = pllm.parameter("target_parameters", "conv"),
        proj: torch.Tensor = pllm.parameter("target_parameters", "proj"),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convolve inside this chunk, read the incoming weight, then sum the positive outer products."""
        values = (
            torch.nn.functional.conv1d(targets.T.unsqueeze(0), conv, padding=2, groups=targets.shape[1]).squeeze(0).T
            @ proj
        )
        if self.dtype is torch.bfloat16:
            update = (values.T @ inputs) * self.config.learning_rate
            weight = (base.float() + delta).to(torch.bfloat16)
            output = (inputs @ weight.T).float() - (inputs @ base.T).float()
            return (weight + update).float() - base.float(), output
        update = (values.float().T @ inputs.float()) * self.config.learning_rate
        return delta + update, inputs @ delta.to(inputs.dtype).T


__all__ = ["InPlaceTTTConfig", "InPlaceTTTWorkload"]
