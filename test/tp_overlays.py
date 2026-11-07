"""Down deltas with shard-local updates, shared by CPU admission and multi-GPU parity checks."""

from dataclasses import dataclass
from typing import Literal

import torch

import pllm

TARGETS = (pllm.model.Projection("down_proj", 0),)


@dataclass(frozen=True)
class DownState(pllm.Workload):
    """Projection-sized random fields make accidental rank-local initialization observable."""

    operator: Literal["dense", "low_rank"] = "dense"

    @property
    def weights(self) -> pllm.RequestState:
        """Dense input columns or low-rank A columns, with replicated B."""
        fields = (
            {"value": pllm.Field(lambda inputs, outputs: (outputs, inputs))}
            if self.operator == "dense"
            else {
                "a": pllm.Field(lambda inputs, outputs: (4, inputs)),
                "b": pllm.Field(lambda inputs, outputs: (outputs, 4)),
            }
        )
        return pllm.RequestState(fields, dtype=torch.float32, initialize=self.initialize)

    def initialize(self, fields, generator):
        """Consume the same global random stream in TP1 and TP2."""
        for value in fields.values():
            value.normal_(std=0.02 if self.operator == "low_rank" else 0.002, generator=generator)


@dataclass(frozen=True)
class DownAdapter(DownState):
    """Read a randomly initialized down delta without updates."""

    @property
    def adapter(self) -> pllm.model.Adapter:
        """Use the framework's dense or low-rank reader."""
        return pllm.model.Adapter("weights", TARGETS, operator=self.operator)


@dataclass(frozen=True)
class DenseDownTraining(DownState):
    """Train each dense input column from the replicated target and local input."""

    @pllm.train_in_forward(state="weights", targets=TARGETS, visibility="exact", tokens="all")
    def update(
        self,
        value=pllm.parameter("weights", "value"),
        inputs=pllm.operand("input"),
        target=pllm.operand("target"),
    ):
        output = inputs.float() @ value.T
        value.add_(1e-6 * (target.float().T @ inputs.float()))
        return pllm.UpdateOutput(
            output,
            output.new_ones((), dtype=torch.bool),
            output.new_ones((), dtype=torch.int64),
            output.new_full((), len(inputs), dtype=torch.int64),
        )


@dataclass(frozen=True)
class LowRankDownTraining(DownState):
    """Update A input columns while B stays replicated and frozen."""

    operator: Literal["dense", "low_rank"] = "low_rank"

    @pllm.train_in_forward(state="weights", targets=TARGETS, operator="low_rank", visibility="exact", tokens="all")
    def update(
        self,
        a=pllm.parameter("weights", "a"),
        b=pllm.parameter("weights", "b"),
        inputs=pllm.operand("input"),
        target=pllm.operand("target"),
    ):
        output = (inputs.float() @ a.T) @ b.T
        a.add_(1e-6 * ((target.float() @ b).T @ inputs.float()))
        return pllm.UpdateOutput(
            output,
            output.new_ones((), dtype=torch.bool),
            output.new_ones((), dtype=torch.int64),
            output.new_full((), len(inputs), dtype=torch.int64),
        )


@dataclass(frozen=True)
class SharedDownAdapter(DownAdapter):
    """The immutable InstanceState path initializes directly into its sole root."""

    @property
    def weights(self) -> pllm.InstanceState:
        """Share the same declared delta across requests."""
        state = super().weights
        return pllm.InstanceState(state.fields, dtype=state.dtype, initialize=state.initialize)


@dataclass(frozen=True)
class ExportDown(DownAdapter):
    """A global-looking export must not silently publish rank zero's input slice."""

    @pllm.export(max_rows=3, max_cols=1)
    def observe(self, value=pllm.parameter("weights", "value_0", per_row=True)):
        return value.mean(dim=(-2, -1)).reshape(-1, 1)


@dataclass(frozen=True)
class ReadDown(SharedDownAdapter):
    """A forward cannot derive replicated control or outputs from local input shards."""

    @pllm.custom_forward(tokens="all")
    def run(self, layers, value=pllm.parameter("weights", "value_0")):
        return layers()[-1] + value.sum()


@dataclass(frozen=True)
class ExportB(DownAdapter):
    """Replicated B remains readable through the ordinary state ports."""

    operator: Literal["dense", "low_rank"] = "low_rank"

    @pllm.export(max_rows=3, max_cols=1)
    def observe(self, value=pllm.parameter("weights", "b_0", per_row=True)):
        return value.mean(dim=(-2, -1)).reshape(-1, 1)
