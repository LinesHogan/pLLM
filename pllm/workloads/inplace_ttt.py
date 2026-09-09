"""Causal chunk mathematics over request-owned, per-target buffers."""

from dataclasses import dataclass, replace
from typing import Annotated

import torch
import pllm
from pllm import TensorUpdate, ParameterTarget, ConditionalLayerReplay
from pllm import authoring as a
from pllm.contracts.semantic_graph import SchedulerTimeline, TaskResourceFact, TaskSymbol, ValueResourceFact
from pllm.contracts.tensor_layout import TensorDType

Input = Annotated[torch.Tensor, pllm.Operand.INPUT]
Target = Annotated[torch.Tensor, pllm.Operand.TARGET]
Weight = Annotated[torch.Tensor, pllm.Operand.WEIGHT]
Delta = Annotated[torch.Tensor, pllm.Parameter("dense_fast_weights", "delta")]
TailInput = Annotated[torch.Tensor, pllm.Parameter("dense_fast_weights", "tail_input")]
TailTarget = Annotated[torch.Tensor, pllm.Parameter("dense_fast_weights", "tail_target")]
TailLength = Annotated[torch.Tensor, pllm.Parameter("dense_fast_weights", "tail_length")]


@dataclass(frozen=True, slots=True)
class InPlaceTTTConfig:
    targets: tuple[ParameterTarget, ...]
    chunk_size: int = 4096
    learning_rate: float = 1e-3
    update_prefill: bool = True
    update_decode: bool = False
    replay: ConditionalLayerReplay | None = None

    def __post_init__(self) -> None:
        if (not self.targets or tuple(sorted(set(self.targets))) != self.targets
            or self.chunk_size < 1 or self.learning_rate <= 0.0):
            raise ValueError("In-Place-TTT mathematical configuration is invalid")


@dataclass(frozen=True)
class InPlaceTTTWorkload(pllm.Workload):
    name = "inplace_ttt.dense_fast_weight"
    config: InPlaceTTTConfig

    def resource_facts(self, facts: a.BackendCompileFacts, options: pllm.ExecutionOptions) -> SchedulerTimeline:
        timeline = super().resource_facts(facts, options)
        if not timeline.task_resources and timeline.memory_budget_bytes is None and options.memory_budget_bytes is None:
            return timeline
        self.require_native_methods(_NATIVE, resource_declaration=_RESOURCES)
        if self.config.replay is not None:
            raise ValueError("finite in-place costs require a separate declaration for model replay scratch")
        rows = facts.probe.max_packed_rows + facts.max_admitted_lineages * (self.config.chunk_size - 1)
        width = 8 if facts.probe.tensor_dtype is TensorDType.FLOAT64 else 4
        # Concatenated tails/inputs/targets, per-chunk outputs plus final cat, dense update temporaries.
        workspace = max(width * (rows * (target.input_size + 3 * target.output_size)
            + 3 * target.input_size * target.output_size) for target in facts.targets)
        output = width * rows * sum(target.output_size for target in facts.targets)
        return replace(timeline,
            task_resources=(TaskResourceFact(TaskSymbol("publish"), 0), TaskResourceFact(TaskSymbol("transition"), workspace)),
            value_resources=(ValueResourceFact(TaskSymbol("publish"), "publication", 0),
                ValueResourceFact(TaskSymbol("transition"), "transition", output)))

    @property
    def dense_fast_weights(self) -> pllm.RequestState:
        tail = max(1, self.config.chunk_size - 1)
        return pllm.RequestState({
            "delta": pllm.Buffer(lambda inputs, outputs: (outputs, inputs)),
            "tail_input": pllm.Buffer(lambda inputs, outputs: (tail, inputs)),
            "tail_target": pllm.Buffer(lambda inputs, outputs: (tail, outputs)),
            "tail_length": pllm.Buffer((), dtype=torch.int64),
        }, namespace="canonical.dense_fast_weight.state@1", keys=(pllm.RequestKeyField.REQUEST_ID,
            pllm.RequestKeyField.PROMPT_IDX, pllm.RequestKeyField.SAMPLE_IDX))

    @pllm.on(pllm.Phase.MODEL_UPDATE)
    @pllm.overlay(state="dense_fast_weights", targets=pllm.Config("config.targets"), weight="delta",
        prefill=pllm.Config("config.update_prefill"), decode=pllm.Config("config.update_decode"),
        replay=pllm.Config("config.replay"))
    def transition(self, hidden: Input, target: Target, base: Weight, delta: Delta,
                   tail_input: TailInput, tail_target: TailTarget, tail_length: TailLength) -> pllm.TensorUpdate:
        prefix = int(tail_length.item())
        inputs = torch.cat((tail_input[:prefix], hidden), dim=0)
        targets = torch.cat((tail_target[:prefix], target), dim=0)
        chunk = self.config.chunk_size
        trained = (inputs.shape[0] // chunk) * chunk
        outputs: list[torch.Tensor] = []
        # Each chunk emits against its input version before updating the next chunk's weights.
        for start in range(0, trained, chunk):
            x, y = inputs[start:start + chunk], targets[start:start + chunk]
            outputs.append(x @ (base + delta).T)
            delta.add_((self.config.learning_rate * (y.T @ x)).to(delta))
        remaining = inputs.shape[0] - trained
        tail_input.zero_()
        tail_target.zero_()
        tail_input[:remaining].copy_(inputs[trained:])
        tail_target[:remaining].copy_(targets[trained:])
        tail_length.fill_(remaining)
        if remaining:
            outputs.append(inputs[trained:] @ (base + delta).T)
        output = torch.cat(outputs, dim=0)[-hidden.shape[0]:]
        return TensorUpdate(output, changed=trained > 0, update_count=trained // chunk,
            trained_rows=trained, metrics={"tail_rows": remaining})


__all__ = ["InPlaceTTTConfig", "InPlaceTTTWorkload"]

_NATIVE = pllm.Workload.native_methods({"transition": InPlaceTTTWorkload.transition})
_RESOURCES = InPlaceTTTWorkload.resource_facts
