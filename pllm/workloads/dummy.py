"""Bounded hidden export expressed as an ordinary replaceable tensor function."""

from dataclasses import dataclass, replace
from collections.abc import Callable
import torch
import pllm
from pllm import authoring as a
from pllm.contracts.semantic_graph import SchedulerTimeline, TaskResourceFact, TaskSymbol, ValueResourceFact
from pllm.contracts.tensor_layout import TensorDType


@dataclass(frozen=True, slots=True)
class DummyConfig:
    enabled: bool = True
    export_every_n_steps: int = 256
    export_max_rows: int = 1
    export_max_cols: int = 16
    layer: int = 0

    def __post_init__(self) -> None:
        if min(self.export_every_n_steps, self.export_max_rows, self.export_max_cols) < 1:
            raise ValueError("dummy export requires positive stride and dimensions")


def hidden_rows(hidden: torch.Tensor) -> torch.Tensor:
    return hidden


@dataclass(frozen=True, slots=True)
class DummyWorkload(pllm.Workload):
    name = "dummy"
    config: DummyConfig = DummyConfig()
    transform: Callable[[torch.Tensor], torch.Tensor] = hidden_rows
    export_name: str = "hidden_rows"

    def resource_facts(self, facts: a.BackendCompileFacts, options: pllm.ExecutionOptions) -> SchedulerTimeline:
        timeline = pllm.Workload.resource_facts(self, facts, options)
        if not timeline.task_resources and timeline.memory_budget_bytes is None and options.memory_budget_bytes is None:
            return timeline
        self.require_native_methods(_NATIVE, resource_declaration=_RESOURCES)
        rows, hidden = min(self.config.export_max_rows, facts.probe.max_packed_rows), facts.probe.hidden_size
        width = 8 if facts.probe.tensor_dtype is TensorDType.FLOAT64 else 4
        workspace = rows * (width * hidden + 8) if self.config.enabled else 0
        output = rows * min(hidden, self.config.export_max_cols) * width if self.config.enabled else 0
        return replace(timeline, task_resources=(TaskResourceFact(TaskSymbol("export_hidden"), workspace),),
            value_resources=(ValueResourceFact(TaskSymbol("export_hidden"), "value", output),))

    @pllm.on(pllm.Phase.OBSERVE)
    @pllm.hook(pllm.Config("config.layer"))
    @pllm.export(name=pllm.Config("export_name"), every=pllm.Config("config.export_every_n_steps"),
        max_rows=pllm.Config("config.export_max_rows"), max_cols=pllm.Config("config.export_max_cols"),
        enabled=pllm.Config("config.enabled"))
    def export_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.transform(hidden)


_NATIVE = pllm.Workload.native_methods({"export_hidden": DummyWorkload.export_hidden, "transform": hidden_rows})
_RESOURCES = DummyWorkload.resource_facts
