"""LoReFT and constant activation edits via checkpoint-backed state."""

from dataclasses import dataclass, replace
from math import prod
from pathlib import Path
import re

import torch
import pllm
from pllm import authoring as a
from pllm.contracts.semantic_graph import SchedulerTimeline, TaskResourceFact, TaskSymbol, ValueResourceFact


def load_loreft_interventions(directory: str | Path, layers: tuple[int, ...],
                              device: torch.device, dtype: torch.dtype) -> pllm.StateTensorUpdate[torch.Tensor]:
    """Read pyreft checkpoints once into the original shared publication input."""
    loaded: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for path in sorted(Path(directory).glob("intkey_*.bin")):
        match = re.fullmatch(r"intkey_layer\.(\d+)\.comp\.block_output\..*\.bin", path.name)
        if match is None or int(match[1]) not in layers:
            continue
        layer = int(match[1])
        if layer in loaded:
            raise ValueError(f"multiple intervention checkpoints for layer {layer}")
        weight, bias, rotation = pllm.load_tensor_fields(path, ("weight", "bias", "rotate_layer"))
        loaded[layer] = (weight.to(device=device, dtype=dtype), bias.to(device=device, dtype=dtype),
            rotation.to(device=device, dtype=dtype))
    if set(loaded) != set(layers):
        raise FileNotFoundError("missing requested LoReFT layer checkpoints")
    return pllm.StateTensorUpdate(tuple(value for layer in layers for value in loaded[layer]))


@dataclass(frozen=True, slots=True)
class ActivationConfig:
    layers: tuple[int, ...]
    rank: int = 4
    first_tokens: int = 0
    last_tokens: int = 2
    operator: pllm.ActivationPatchOperator = pllm.ActivationPatchOperator.LOW_RANK_AFFINE
    scope: pllm.StateScope = pllm.StateScope.INSTANCE

    def __post_init__(self) -> None:
        if (not self.layers or tuple(sorted(set(self.layers))) != self.layers or self.layers[0] < 0
            or self.rank < 1 or min(self.first_tokens, self.last_tokens) < 0):
            raise ValueError("activation edits require sorted layers, positive rank and nonnegative token bounds")


@dataclass(frozen=True)
class ActivationWorkload(pllm.Workload):
    name = "activation"
    config: ActivationConfig

    def resource_facts(self, facts: a.BackendCompileFacts, options: pllm.ExecutionOptions) -> SchedulerTimeline:
        timeline = super().resource_facts(facts, options)
        if not timeline.task_resources and timeline.memory_budget_bytes is None and options.memory_budget_bytes is None:
            return timeline
        layout, = self.declaration(facts).layouts.layouts
        # isfinite/reduction visits one original field at a time; copying needs no owned tensor.
        workspace = 16 * max(prod(field.shape) for field in layout.layout.fields)
        return replace(timeline,
            task_resources=(TaskResourceFact(TaskSymbol("publish"), 0), TaskResourceFact(TaskSymbol("update"), workspace)),
            value_resources=(ValueResourceFact(TaskSymbol("publish"), "publication", 0),
                ValueResourceFact(TaskSymbol("update"), "updated", 0)))

    @property
    def intervention(self) -> pllm.InstanceState:
        rank = self.config.rank
        fields = (pllm.Buffer(lambda inputs, outputs: (rank, inputs)),
            pllm.Buffer((rank,)), pllm.Buffer(lambda inputs, outputs: (outputs, rank)))
        if self.config.operator is pllm.ActivationPatchOperator.ADD:
            fields = (pllm.Buffer(lambda inputs, outputs: (inputs,), dtype=torch.float32),)
        state = pllm.RequestState if self.config.scope is pllm.StateScope.REQUEST else pllm.InstanceState
        return state({f"layer_{layer}_{offset}": buffer for layer in self.config.layers for offset, buffer in enumerate(fields)},
            namespace="canonical.activation@1")

    @property
    def patch(self) -> pllm.Activation:
        return pllm.Activation("intervention", self.config.layers, self.config.first_tokens,
            self.config.last_tokens, self.config.operator)
