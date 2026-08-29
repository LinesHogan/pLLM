"""LoReFT activation intervention as a published module and generic mount."""

from __future__ import annotations

import torch

from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.kit.workload import Workload
from tllm.workload import (
    NumericConstraint,
    PublishedModule,
    State,
    WorkloadConfig,
    config_field,
    workload,
)


class ReftConfig(WorkloadConfig):
    enabled: bool = config_field(default=True)
    layers: tuple[int, ...] = config_field(default=(15,))
    rank: int = config_field(default=4, constraint=NumericConstraint.POSITIVE)
    first_tokens: int = config_field(
        default=0, constraint=NumericConstraint.NONNEGATIVE
    )
    last_tokens: int = config_field(
        default=2, constraint=NumericConstraint.NONNEGATIVE
    )
    learning_rate: float = config_field(
        default=4e-3, constraint=NumericConstraint.POSITIVE
    )


class ReftParameters(torch.nn.Module):
    def __init__(
        self,
        layer_count: int,
        hidden_size: int,
        rank: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        source = torch.nn.Linear(hidden_size, rank)
        self.source_w = torch.nn.Parameter(
            source.weight.detach().to(device=device, dtype=dtype).repeat(layer_count, 1, 1)
        )
        self.source_b = torch.nn.Parameter(
            source.bias.detach().to(device=device, dtype=dtype).repeat(layer_count, 1)
        )
        rotate = torch.empty((layer_count, hidden_size, rank), dtype=torch.float32)
        for slot in range(layer_count):
            torch.nn.init.orthogonal_(rotate[slot])
        self.rotate = torch.nn.Parameter(rotate.to(device=device, dtype=dtype))


@workload(id="reft", config=ReftConfig, enabled=ReftConfig.enabled)
class ReftWorkload(Workload[ReftConfig]):
    intervention: PublishedModule[ReftParameters] = State.published_module()

    @intervention.initialize
    def init_intervention(self, spec: RuntimeSpecializationSpec) -> ReftParameters:
        return ReftParameters(
            layer_count=len(self.config.layers),
            hidden_size=spec.hidden_size,
            rank=self.config.rank,
            device=spec.device,
            dtype=spec.dtype,
        )

    def loreft_delta(self, layer_slot: int, rows: torch.Tensor) -> torch.Tensor:
        weight = self.intervention.source_w[layer_slot]
        rotate = self.intervention.rotate[layer_slot]
        projected = torch.nn.functional.linear(
            rows, weight, self.intervention.source_b[layer_slot]
        )
        return (projected - rows @ rotate) @ rotate.transpose(0, 1)


__all__ = ["ReftConfig", "ReftParameters", "ReftWorkload"]
