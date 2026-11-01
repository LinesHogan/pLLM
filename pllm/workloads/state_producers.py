"""Request projection state produced by prompt networks or decode AdamW; examples for workload authors.

Prompt producers publish after completed prefill for the next decode forward. Old KV keeps the
factors used when it was computed. These examples target eager, unsharded q/v projections.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

import pllm

TARGETS = tuple(
    pllm.model.ParameterTarget(layer, pllm.model.ParameterModuleKind.ATTENTION, role)
    for layer, role in (
        (0, pllm.model.ParameterRole.QUERY_PROJECTION_WEIGHT),
        (1, pllm.model.ParameterRole.VALUE_PROJECTION_WEIGHT),
    )
)


def seed_factors(fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
    """Initialize A randomly and B to zero, so the first optimizer step can learn B."""
    for name, value in fields.items():
        value.normal_(std=0.02, generator=generator) if name.startswith("a_") else value.zero_()


@dataclass(frozen=True)
class FactorAdapterWorkload(pllm.Workload):
    """The same state and consumer declaration for every producer below."""

    rank: int = 2
    optimizer: ClassVar[pllm.AdamW | None] = None

    @property
    def factor_rank(self) -> int:
        """Rank read by each target projection."""
        return self.rank

    @property
    def factors(self) -> pllm.RequestState:
        """Resolve factor widths from the inspected model projections."""
        return pllm.RequestState(
            {
                "a": pllm.Buffer(lambda inputs, outputs: (self.factor_rank, inputs)),
                "b": pllm.Buffer(lambda inputs, outputs: (outputs, self.factor_rank)),
            },
            optimizer=self.optimizer,
            initialize=seed_factors,
        )

    @property
    def consume(self) -> pllm.model.Overlay:
        """Read each request's pinned factors inside the q/v projections."""
        return pllm.model.Overlay("factors", TARGETS, operator=pllm.model.ParameterOverlayOperator.LOW_RANK_ADD)


class PromptFactorsWorkload(FactorAdapterWorkload):
    """A shared illustrative hypernetwork produces each request's factors from its prompt."""

    @property
    def network(self) -> pllm.InstanceState:
        """Keep shared network weights separate from request factors."""
        return pllm.InstanceState(
            module=lambda model: nn.Sequential(
                nn.Linear(model.hidden_size, model.hidden_size),
                nn.Tanh(),
                nn.Linear(model.hidden_size, self.factor_rank * sum(i + o for i, o in model.target_shapes)),
            )
        )

    @pllm.write(state="factors", fields=("a_0", "b_0", "a_1", "b_1"))
    def produce(
        self,
        h: torch.Tensor = pllm.hidden(8, tokens="prefill"),
        net: Callable[[torch.Tensor], torch.Tensor] = pllm.module("network"),
        aq: torch.Tensor = pllm.parameter("factors", "a_0"),
        bq: torch.Tensor = pllm.parameter("factors", "b_0"),
        av: torch.Tensor = pllm.parameter("factors", "a_1"),
        bv: torch.Tensor = pllm.parameter("factors", "b_1"),
    ) -> tuple[torch.Tensor, ...]:
        """Publish the complete tuple atomically, preserving the request axis."""
        templates = (aq, bq, av, bv)
        return tuple(
            value.reshape_as(template).to(template)
            for value, template in zip(
                torch.split(net(h), [value.flatten(1).shape[1] for value in templates], dim=-1),
                templates,
                strict=True,
            )
        )


class Mixture(nn.Module):
    """Concatenated factors implement the exact weighted sum of K low-rank matrices."""

    def __init__(self, hidden: int, sizes: tuple[tuple[int, int], ...], rank: int, components: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden, components)
        self.a = nn.ParameterList(nn.Parameter(torch.randn(components, rank, inputs) * 0.02) for inputs, _ in sizes)
        self.b = nn.ParameterList(nn.Parameter(torch.randn(components, outputs, rank) * 0.02) for _, outputs in sizes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Weight B and concatenate A; averaging both factors would introduce cross terms."""
        probabilities = self.gate(h).softmax(-1)
        return torch.cat(
            tuple(
                value.flatten(1)
                for a, b in zip(self.a, self.b, strict=True)
                for value in (
                    a.flatten(0, 1).expand(len(h), -1, -1),
                    (probabilities[:, :, None, None] * b).permute(0, 2, 1, 3).flatten(2),
                )
            ),
            dim=-1,
        )


@dataclass(frozen=True)
class PromptMixtureWorkload(PromptFactorsWorkload):
    """A prompt discriminator selects a mixture of shared LoRAs for each request."""

    components: int = 3

    @property
    def factor_rank(self) -> int:
        """Exact mixtures consume K times the base rank."""
        return self.components * self.rank

    @property
    def network(self) -> pllm.InstanceState:
        """Allocate the gate and factor bank as shared instance state."""
        return pllm.InstanceState(
            module=lambda model: Mixture(model.hidden_size, model.target_shapes, self.rank, self.components)
        )


class DecodeFactorsWorkload(FactorAdapterWorkload):
    """Train side predictions with AdamW; internal projections read the update next forward."""

    optimizer = pllm.AdamW(lr=1e-3)

    @pllm.train(state="factors", reduction="state_mean")
    def learn(
        self,
        x: torch.Tensor = pllm.hidden(0),
        target: torch.Tensor = pllm.hidden(9),
        aq: torch.Tensor = pllm.parameter("factors", "a_0"),
        bq: torch.Tensor = pllm.parameter("factors", "b_0"),
        av: torch.Tensor = pllm.parameter("factors", "a_1"),
        bv: torch.Tensor = pllm.parameter("factors", "b_1"),
    ) -> torch.Tensor:
        """Predict the detached hidden row's mean at every q/v output, without gradients through vLLM or KV."""
        row = x.to(aq).unsqueeze(1)
        q = ((row @ aq.transpose(-1, -2)) @ bq.transpose(-1, -2)).squeeze(1)
        v = ((row @ av.transpose(-1, -2)) @ bv.transpose(-1, -2)).squeeze(1)
        target = target.mean(-1, keepdim=True)
        return (q.float() - target).square().mean(-1) + (v.float() - target).square().mean(-1)
