"""Iterative logit heads written as tensor methods, state and sampler effects."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from math import isfinite
from typing import Annotated

import torch
from torch.nn import functional as F

import pllm
from pllm import authoring as a
from pllm.contracts.semantic_graph import SchedulerTimeline, TaskResourceFact, TaskSymbol, ValueResourceFact
from pllm.execution.tensor_binding import TensorBinding, TensorFrame, TensorOutput


@dataclass(frozen=True, slots=True)
class OverRIDEConfig:
    rank: int = 16
    num_iterations: int = 10
    lambd: float = 0.8
    learning_rate: float = 1e-3
    enabled: bool = True

    def __post_init__(self) -> None:
        if (
            min(self.rank, self.num_iterations) < 1
            or not isfinite(self.lambd)
            or not isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("OverRIDE requires positive dimensions/rate and finite reweighting")


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float(), labels.long())


def head_logits(hidden: torch.Tensor, down: torch.Tensor, up: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    down, up = down.index_select(0, indices), up.index_select(0, indices)
    return torch.bmm(torch.bmm(hidden.unsqueeze(1), down.transpose(1, 2)), up.transpose(1, 2)).squeeze(1)


Hidden = Annotated[torch.Tensor, pllm.Hidden(-1, site=pllm.HiddenSite.MODEL_OUTPUT)]
Samples = Annotated[torch.Tensor, pllm.Samples(pllm.Config("config.num_iterations"))]
Down = Annotated[torch.Tensor, pllm.Parameter("iteration_heads", "down")]
Up = Annotated[torch.Tensor, pllm.Parameter("iteration_heads", "up")]


@dataclass(frozen=True, slots=True)
class OverRIDEWorkload(pllm.Workload):
    name = "override"
    config: OverRIDEConfig = OverRIDEConfig()
    objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cross_entropy

    def resource_facts(self, facts: a.BackendCompileFacts, options: pllm.ExecutionOptions) -> SchedulerTimeline:
        timeline = pllm.Workload.resource_facts(self, facts, options)
        if not timeline.task_resources and timeline.memory_budget_bytes is None and options.memory_budget_bytes is None:
            return timeline
        self.require_native_methods(_NATIVE, resource_declaration=_RESOURCES)
        rows, hidden, vocab, rank = (
            facts.max_admitted_lineages,
            facts.probe.hidden_size,
            facts.vocab_size,
            self.config.rank,
        )
        state = self.iteration_heads
        dimensions = a.Dimensions(hidden, vocab)
        expected = {
            "down": (self.config.num_iterations - 1, rank, hidden),
            "up": (self.config.num_iterations - 1, vocab, rank),
        }
        if (
            state.dtype != torch.float32
            or set(state.fields) != expected.keys()
            or any(
                (shape(dimensions) if callable(shape) else shape) != expected[name]
                for name, shape in state.fields.items()
            )
            or self.declaration(facts).graph.states[0].scope is not pllm.StateScope.INSTANCE
        ):
            raise ValueError("customized iteration-head state requires its own resource_facts declaration")
        head = rank * (hidden + vocab)
        parameters = (self.config.num_iterations - 1) * head
        # FP32 head selection, saved CE tensors, scatter gradients and foreach Adam.
        prediction = 4 * rows * (2 * head + hidden + rank + 3 * vocab) + 32 * rows
        training = 4 * (8 * parameters + rows * (4 * head + 2 * hidden + 3 * rank + 8 * vocab)) + 64 * rows
        return replace(
            timeline,
            task_resources=(
                TaskResourceFact(TaskSymbol("__train_eligible"), rows * 32),
                TaskResourceFact(TaskSymbol("reweight"), prediction, input_shapes=((rows, hidden),)),
                TaskResourceFact(TaskSymbol("train"), training, input_shapes=((rows, hidden),)),
            ),
            value_resources=(
                ValueResourceFact(TaskSymbol("__train_eligible"), "value", rows),
                ValueResourceFact(TaskSymbol("reweight"), "value", rows * vocab * 4),
                ValueResourceFact(TaskSymbol("train"), "value", parameters * 4 + 4),
            ),
        )

    def operations(
        self, facts: a.BackendCompileFacts, plan: a.PlanBundle, b: TensorBinding
    ) -> a.MathBinding[a.TensorStepInput, TensorFrame, TensorOutput]:
        if not plan.timeline.task_resources:
            return pllm.Workload.operations(self, facts, plan, b)
        self.require_resources(plan, self.minimum_resources(facts, plan))
        prepare_frame = b.bind_prepare()

        def prepare(value: a.TensorStepInput, batch: a.TensorBatch) -> TensorFrame:
            data = prepare_frame(value, batch)
            if len(data.rows) > facts.max_admitted_lineages:
                raise ValueError("iteration-head rows exceed the signed workspace envelope")
            return data

        return a.MathBinding(
            replace(self._functions(b, facts.device, facts.initialization_seed), prepare_data=prepare), b.bind_all(plan)
        )

    @property
    def iteration_heads(self) -> pllm.InstanceState:
        config = self.config

        def initialize(parameters: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
            parameters["down"].normal_(std=0.02, generator=generator)

        return pllm.InstanceState(
            {
                "down": lambda m: (config.num_iterations - 1, config.rank, m.hidden_size),
                "up": lambda m: (config.num_iterations - 1, m.vocab_size, config.rank),
            },
            optimizer=pllm.AdamW(lr=config.learning_rate),
            initialize=initialize,
            namespace="canonical.iteration_heads@1",
        )

    @pllm.on(pllm.Phase.BEFORE_SAMPLE)
    @pllm.add_logits
    def reweight(self, hidden: Hidden, samples: Samples, down: Down, up: Up) -> torch.Tensor:
        if not self.config.enabled or self.config.num_iterations == 1:
            return hidden.new_zeros((len(hidden), up.shape[1]))
        return head_logits(hidden, down, up, (samples - 1).clamp_min(0)) * (
            (samples > 0).unsqueeze(1) * self.config.lambd
        )

    def eligible(self, samples: Samples) -> torch.Tensor:
        return (samples < self.config.num_iterations - 1) & self.config.enabled

    @pllm.on(pllm.Phase.AFTER_SAMPLE, rows="eligible")
    @pllm.train(state="iteration_heads")
    def train(
        self,
        hidden: Hidden,
        samples: Samples,
        down: Down,
        up: Up,
        base: Annotated[torch.Tensor, pllm.BaseLogits()],
        tokens: Annotated[torch.Tensor, pllm.SampledTokens()],
    ) -> torch.Tensor:
        return self.objective(base + head_logits(hidden, down, up, samples), tokens)


_NATIVE = pllm.Workload.native_methods(
    {
        "reweight": OverRIDEWorkload.reweight,
        "eligible": OverRIDEWorkload.eligible,
        "train": OverRIDEWorkload.train,
        "objective": cross_entropy,
    }
)
_RESOURCES = OverRIDEWorkload.resource_facts
