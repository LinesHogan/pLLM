"""Projection deltas with shard-local updates, shared by CPU admission and multi-GPU parity checks."""

from dataclasses import dataclass, replace
from typing import Literal

import torch

import pllm

TARGETS = (pllm.model.Projection("down_proj", 0),)


@dataclass(frozen=True)
class ProjectionState(pllm.Workload):
    """Projection-sized random fields make accidental rank-local initialization observable."""

    operator: Literal["dense", "low_rank"] = "dense"
    targets: tuple[pllm.model.Projection, ...] = TARGETS
    optimizer: pllm.SGD | None = None

    @property
    def weights(self) -> pllm.RequestState:
        """Projection-sized dense weights or low-rank factors."""
        fields = (
            {"value": pllm.Field(lambda inputs, outputs: (outputs, inputs))}
            if self.operator == "dense"
            else {
                "a": pllm.Field(lambda inputs, outputs: (4, inputs)),
                "b": pllm.Field(lambda inputs, outputs: (outputs, 4)),
            }
        )
        return pllm.RequestState(fields, init=pllm.Computed(self.initialize), optimizer=self.optimizer)

    def initialize(self, fields, generator):
        """Consume the same global random stream in TP1 and TP2."""
        for value in fields.values():
            value.normal_(std=0.02 if self.operator == "low_rank" else 0.002, generator=generator)


@dataclass(frozen=True)
class ProjectionAdapter(ProjectionState):
    """Read a randomly initialized projection delta without updates."""

    @property
    def adapter(self) -> pllm.model.Adapter:
        """Use the framework's dense or low-rank reader."""
        return pllm.model.Adapter("weights", self.targets, operator=self.operator)


@dataclass(frozen=True)
class DenseTraining(ProjectionState):
    """Train dense shards from operands available on this rank."""

    @pllm.train_in_forward(state="weights", targets=lambda self: self.targets, visibility="exact", tokens="all")
    def update(
        self,
        value=pllm.parameter("weights", "value"),
        inputs=pllm.operand("input"),
        target=pllm.operand("target"),
    ):
        output = inputs.float() @ value.T
        signal = target.float() if self.targets[0].name == "down_proj" else output
        value.add_(1e-6 * (signal.T @ inputs.float()))
        return pllm.UpdateOutput(
            output,
            output.new_ones((), dtype=torch.bool),
            output.new_ones((), dtype=torch.int64),
            output.new_full((), len(inputs), dtype=torch.int64),
        )


@dataclass(frozen=True)
class DenseChunkTraining(ProjectionState):
    """Retain replicated target rows across chunk boundaries while updating dense shards."""

    @pllm.train_in_forward(
        state="weights", targets=lambda self: self.targets, visibility="exact", tokens="all", chunk=2
    )
    def update(
        self, value=pllm.parameter("weights", "value"), inputs=pllm.operand("input"), target=pllm.operand("target")
    ):
        output = inputs.float() @ value.T
        signal = target.float() if self.targets[0].name == "down_proj" else target.float() @ value.T
        return value + 1e-6 * (signal.T @ inputs.float()), output


@dataclass(frozen=True)
class DenseGradientTraining(ProjectionState):
    """Differentiate a sum over independent output rows and let the framework apply SGD."""

    optimizer: pllm.SGD | None = pllm.SGD(1e-6)
    scale: float = 1.0

    @property
    def weights(self) -> pllm.RequestState:
        """Retain an initialization checksum to detect a silent zero-gradient update."""
        state = super().weights
        return replace(state, fields={**state.fields, "initial_norm": pllm.Field((2,), dtype=torch.float64)})

    def initialize(self, fields, generator):
        """Measure the full initialized matrix before the framework slices its output rows."""
        fields["value"].normal_(std=0.002, generator=generator)
        value = fields["value"]
        rows = torch.stack((value[0], value[len(value) // 2])).double()
        fields["initial_norm"].copy_(rows.square().sum(-1))

    @pllm.train_in_forward(
        state="weights",
        targets=lambda self: self.targets,
        scale=lambda self: self.scale,
        visibility="exact",
        tokens="decode",
        chunk=2,
    )
    def loss(self, value=pllm.parameter("weights", "value"), inputs=pllm.operand("input")):
        return (inputs.float() @ value.T).square().sum(-1)


@dataclass(frozen=True)
class LowRankTraining(ProjectionState):
    """Update the sharded factor while the replicated factor stays frozen."""

    operator: Literal["dense", "low_rank"] = "low_rank"

    @pllm.train_in_forward(
        state="weights", targets=lambda self: self.targets, operator="low_rank", visibility="exact", tokens="all"
    )
    def update(
        self,
        a=pllm.parameter("weights", "a"),
        b=pllm.parameter("weights", "b"),
        inputs=pllm.operand("input"),
        target=pllm.operand("target"),
    ):
        output = (inputs.float() @ a.T) @ b.T
        if self.targets[0].name == "down_proj":
            a.add_(1e-6 * ((target.float() @ b).T @ inputs.float()))
        else:
            b.add_(1e-6 * (output.T @ (inputs.float() @ a.T)))
        return pllm.UpdateOutput(
            output,
            output.new_ones((), dtype=torch.bool),
            output.new_ones((), dtype=torch.int64),
            output.new_full((), len(inputs), dtype=torch.int64),
        )


@dataclass(frozen=True)
class SharedProjectionAdapter(ProjectionAdapter):
    """The immutable InstanceState path initializes directly into its sole root."""

    @property
    def weights(self) -> pllm.InstanceState:
        """Share the same declared delta across requests."""
        state = super().weights
        return pllm.InstanceState(state.fields, dtype=state.dtype, init=state.init)


@dataclass(frozen=True)
class ExportDown(ProjectionAdapter):
    """A global-looking export must not silently publish rank zero's input slice."""

    @pllm.export(max_rows=3, max_cols=1)
    def observe(self, value=pllm.parameter("weights", "value_0", per_row=True)):
        return value.mean(dim=(-2, -1)).reshape(-1, 1)


@dataclass(frozen=True)
class ReadDown(SharedProjectionAdapter):
    """A forward cannot derive replicated control or outputs from local input shards."""

    @pllm.custom_forward(tokens="all")
    def run(self, layers, value=pllm.parameter("weights", "value_0")):
        return layers()[-1] + value.sum()


@dataclass(frozen=True)
class ExportB(ProjectionAdapter):
    """Replicated B remains readable through the ordinary state ports."""

    operator: Literal["dense", "low_rank"] = "low_rank"

    @pllm.export(max_rows=3, max_cols=1)
    def observe(self, value=pllm.parameter("weights", "b_0", per_row=True)):
        return value.mean(dim=(-2, -1)).reshape(-1, 1)


@dataclass(frozen=True)
class TrainColumnA(ProjectionState):
    """A column reconstruction gradient for replicated A needs a sum over output shards."""

    operator: Literal["dense", "low_rank"] = "low_rank"
    targets: tuple[pllm.model.Projection, ...] = (pllm.model.Projection("q_proj", 0),)
    optimizer: pllm.SGD | None = pllm.SGD(1e-6)

    @pllm.train_in_forward(
        state="weights",
        targets=lambda self: self.targets,
        operator="low_rank",
        visibility="exact",
        tokens="all",
        chunk=2,
    )
    def loss(self, a=pllm.parameter("weights", "a"), b=pllm.parameter("weights", "b"), inputs=pllm.operand("input")):
        return ((inputs.float() @ a.T) @ b.T).square().sum(-1)


@dataclass(frozen=True)
class TensorConstantUpdate(ProjectionState):
    """Tensor literals can hide shape-dependent values behind identical FX attribute names."""

    targets: tuple[pllm.model.Projection, ...] = (pllm.model.Projection("q_proj", 0),)

    @pllm.train_in_forward(state="weights", targets=lambda self: self.targets, visibility="exact", tokens="all")
    def update(self, value=pllm.parameter("weights", "value"), inputs=pllm.operand("input")):
        value.add_(torch.tensor(1.0 if value.shape[0] > value.shape[1] // 2 else 2.0))
        output = inputs.float() @ value.T
        return pllm.UpdateOutput(
            output,
            output.new_ones((), dtype=torch.bool),
            output.new_ones((), dtype=torch.int64),
            output.new_full((), len(inputs), dtype=torch.int64),
        )
