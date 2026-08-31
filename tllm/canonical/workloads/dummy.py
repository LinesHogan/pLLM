"""Deterministic hidden-row export on the clean-room single-task core."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.canonical.workloads.base import CanonicalWorkload
from tllm.canonical.execution import SingleTaskExecution
from tllm.canonical.scheduler import (
    compile_schedule,
    lower_physical_plan,
    validate_physical_plan,
    validate_schedule,
)
from tllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from tllm.contracts.semantic_graph import (
    AnchorSymbol,
    ExecutionKind,
    HiddenInputSource,
    ImplementationId,
    SchedulerTimeline,
    SemanticGraph,
    SemanticInput,
    SemanticOutput,
    SemanticTask,
    StepIdInputSource,
    TaskImplementationSpec,
    TaskSymbol,
    TimingConstraint,
    WorkloadId,
)


_DUMMY_IMPLEMENTATION = ImplementationId("canonical.dummy:cpu_export@1")


@dataclass(frozen=True, slots=True)
class DummyConfig:
    enabled: bool = True
    export_every_n_steps: int = 256
    export_max_rows: int = 1
    export_max_cols: int = 16
    layer: int = 0

    def __post_init__(self) -> None:
        if (
            self.export_every_n_steps < 1
            or self.export_max_rows < 1
            or self.export_max_cols < 1
        ):
            raise ValueError("dummy export bounds and stride must be positive")


@dataclass(frozen=True, slots=True)
class DummyStep:
    hidden_rows: torch.Tensor
    step_id: int

    def __post_init__(self) -> None:
        if self.hidden_rows.ndim != 2 or self.step_id < 0:
            raise ValueError("dummy step requires rank-2 hidden rows and a step ID")


@dataclass(frozen=True, slots=True)
class DummySkipped:
    pass


@dataclass(frozen=True, slots=True)
class DummyExported:
    step_id: int
    rows: torch.Tensor


DummyResult = DummySkipped | DummyExported


@dataclass(frozen=True, slots=True)
class DummyDiagnostics:
    consumed_batches: int
    consumed_rows: int


class _DummyCpuExport:
    __slots__ = ("_config", "_consumed_batches", "_consumed_rows")

    def __init__(self, config: DummyConfig) -> None:
        self._config = config
        self._consumed_batches = 0
        self._consumed_rows = 0

    @property
    def implementation_id(self) -> ImplementationId:
        return _DUMMY_IMPLEMENTATION

    def execute(self, value: DummyStep) -> DummyResult:
        if (
            not self._config.enabled
            or value.step_id % self._config.export_every_n_steps != 0
        ):
            return DummySkipped()
        rows = value.hidden_rows[
            : self._config.export_max_rows,
            : self._config.export_max_cols,
        ].detach().to(device="cpu").clone()
        self._consumed_batches += 1
        self._consumed_rows += int(rows.shape[0])
        return DummyExported(value.step_id, rows)

    def diagnostics(self) -> DummyDiagnostics:
        return DummyDiagnostics(
            consumed_batches=self._consumed_batches,
            consumed_rows=self._consumed_rows,
        )


class DummyWorkload(CanonicalWorkload):
    """Public Dummy workload backed only by the canonical core."""

    __slots__ = ("config", "_implementation")

    def __init__(self, config: DummyConfig = DummyConfig()) -> None:
        self.config = config
        self._implementation = _DummyCpuExport(config)

    def semantic_graph(self) -> SemanticGraph:
        return SemanticGraph(
            workload_id=WorkloadId("dummy"),
            tasks=(
                SemanticTask(
                    symbol=TaskSymbol("export_hidden"),
                    timing=TimingConstraint(
                        after=AnchorSymbol("model.execute"),
                        before=AnchorSymbol("step.end"),
                    ),
                    inputs=(
                        SemanticInput(
                            "hidden_rows",
                            HiddenInputSource(
                                layer=self.config.layer,
                                site=HiddenSite.BLOCK_OUTPUT,
                                phase=HiddenPhase.DECODE,
                            ),
                        ),
                        SemanticInput("step_id", StepIdInputSource()),
                    ),
                    outputs=(SemanticOutput("export"),),
                    implementations=(
                        TaskImplementationSpec(
                            implementation_id=_DUMMY_IMPLEMENTATION,
                            execution_kind=ExecutionKind.HOST_SERVICE,
                            required_capabilities=("host/cpu_export@1",),
                        ),
                    ),
                ),
            ),
        )

    def compile(
        self,
        timeline: SchedulerTimeline,
        *,
        plan_epoch: int = 1,
    ) -> SingleTaskExecution[DummyStep, DummyResult]:
        graph = self.semantic_graph()
        decision = compile_schedule(graph, timeline)
        validate_schedule(graph, timeline, decision)
        physical = lower_physical_plan(
            graph,
            timeline,
            decision,
            plan_epoch=plan_epoch,
        )
        validate_physical_plan(graph, timeline, decision, physical)
        return SingleTaskExecution(physical, self._implementation)

    def diagnostics(self) -> DummyDiagnostics:
        return self._implementation.diagnostics()


__all__ = [
    "DummyConfig",
    "DummyDiagnostics",
    "DummyExported",
    "DummyResult",
    "DummySkipped",
    "DummyStep",
    "DummyWorkload",
]
