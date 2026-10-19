"""Request distillation expressed as prediction, loss and an AdamW update."""

from dataclasses import dataclass
from math import isfinite
from typing import Annotated

import torch

import pllm
from pllm import authoring as a
from pllm.contracts.backend_attachment import BackendAttachmentSpec
from pllm.contracts.request_identity import RequestKeyField, canonical_request_key_fields
from pllm.contracts.semantic_graph import (
    SemanticGraph,
)
from pllm.contracts.tensor_layout import TensorLayoutRequest
from pllm.execution.tensor_frame import Tensors


@dataclass(frozen=True, slots=True)
class ESampRequestConfig:
    rank: int = 128
    source_layer: int = 0
    target_layer: int = -1
    output_layernorm: bool = True
    intervention: bool = False
    intervention_beta: float = 0.0
    learning_rate: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    key_fields: tuple[RequestKeyField, ...] = (RequestKeyField.REQUEST_ID,)
    paged: bool = False

    def __post_init__(self) -> None:
        a.AdamW(self.learning_rate, self.betas, self.eps, self.weight_decay)
        canonical_request_key_fields(self.key_fields)
        if self.rank < 1 or not isfinite(self.intervention_beta) or self.intervention_beta < 0.0:
            raise ValueError("ESamp request configuration is invalid")


def initialize(parameters: Tensors, generator: torch.Generator) -> None:
    for name in ("up", "gate", "down"):
        parameters[name].normal_(0.0, 0.02, generator=generator)
    parameters["bias"].fill_(1.0)
    parameters["scale"].fill_(1.0)
    parameters["shift"].zero_()


@dataclass(frozen=True)
class ESampRequestWorkload(pllm.Workload):
    name = "esamp.request"
    state_name = "request_distiller"
    config: ESampRequestConfig = ESampRequestConfig()

    @property
    def request_distiller(self) -> pllm.RequestState:
        config = self.config
        rank, norm = config.rank, config.output_layernorm
        return pllm.RequestState(
            {
                "up": lambda m: (m.hidden_size, rank),
                "gate": lambda m: (m.hidden_size, rank),
                "down": lambda m: (rank, m.hidden_size),
                "bias": (rank,),
                "scale": lambda m: (m.hidden_size if norm else 0,),
                "shift": lambda m: (m.hidden_size if norm else 0,),
            },
            initialize=initialize,
            optimizer=self.optimizer,
            alignment=4,
            namespace="canonical.low_rank_gated_adamw@1",
            keys=config.key_fields,
        )

    def ready(self, versions: Annotated[torch.Tensor, pllm.StateVersion("request_distiller")]) -> torch.Tensor:
        return (versions > 0) & self.config.intervention

    def forward(
        self,
        source: torch.Tensor,
        up: torch.Tensor,
        gate: torch.Tensor,
        down: torch.Tensor,
        bias: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        x = source.unsqueeze(1)
        product = torch.bmm(x, up).squeeze(1) * torch.nn.functional.silu(torch.bmm(x, gate).squeeze(1) + bias)
        output = source + torch.bmm(product.unsqueeze(1), down).squeeze(1)
        return (
            torch.nn.functional.layer_norm(output, (source.shape[1],)) * scale + shift
            if self.config.output_layernorm
            else output
        )

    @pllm.on(pllm.Phase.BEFORE_SAMPLE, rows="ready")
    @pllm.guide_sampling(beta=pllm.Config("config.intervention_beta"))
    def predict(
        self,
        source: Annotated[torch.Tensor, pllm.Hidden(pllm.Config("config.source_layer"))],
        up: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "up")],
        gate: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "gate")],
        down: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "down")],
        bias: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "bias")],
        scale: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "scale")],
        shift: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "shift")],
        versions: Annotated[torch.Tensor, pllm.StateVersion("request_distiller")],
    ) -> torch.Tensor:
        return self.forward(source, up, gate, down, bias, scale, shift)

    @pllm.on(pllm.Phase.AFTER_SAMPLE)
    @pllm.train(state="request_distiller", reduction="state_mean")
    def loss(
        self,
        source: Annotated[torch.Tensor, pllm.Hidden(pllm.Config("config.source_layer"))],
        target: Annotated[torch.Tensor, pllm.Hidden(pllm.Config("config.target_layer"))],
        up: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "up")],
        gate: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "gate")],
        down: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "down")],
        bias: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "bias")],
        scale: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "scale")],
        shift: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "shift")],
    ) -> torch.Tensor:
        return (self.forward(source, up, gate, down, bias, scale, shift) - target).square().mean(1)

    @property
    def optimizer(self) -> a.AdamW:
        config = self.config
        return a.AdamW(config.learning_rate, config.betas, config.eps, config.weight_decay)

    def declare(self, facts: a.BackendCompileFacts) -> tuple[SemanticGraph, TensorLayoutRequest, BackendAttachmentSpec]:
        if self.config.key_fields != (RequestKeyField.REQUEST_ID,):
            raise ValueError("attached ESamp requires request-id state keys")
        return super().declare(facts)


__all__ = ["ESampRequestConfig", "ESampRequestWorkload"]
