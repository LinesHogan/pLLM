"""Typed authoring and invocation boundary shared with model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from tllm.canonical.cuda_packed_hidden import CudaPackedHiddenSnapshotBatch
from tllm.canonical.ptts.page_bundle_planning import PageBundleProviderFacts
from tllm.canonical.state_execution import StateExecution
from tllm.contracts.backend_attachment import BackendPlacementProbe, SignedBackendAttachment
from tllm.contracts.packed_hidden import PackedHiddenSnapshotBatch
from tllm.contracts.packed_rows import PackedPhaseRowBatch, PackedSamplerRowBatch
from tllm.contracts.parameter_overlay import ParameterTarget, CapturedParameterOperand
from tllm.contracts.semantic_graph import HiddenInputSource, SchedulerTimeline


DataT = TypeVar("DataT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class TensorStepInput:
    rows: PackedPhaseRowBatch
    sampler_rows: PackedSamplerRowBatch
    hidden: PackedHiddenSnapshotBatch | CudaPackedHiddenSnapshotBatch | None = None
    operands: tuple[CapturedParameterOperand, ...] = ()

    def __post_init__(self) -> None:
        if self.sampler_rows.row_batch is not self.rows:
            raise ValueError("tensor input requires the original packed row provenance")
        match self.hidden:
            case None:
                pass
            case PackedHiddenSnapshotBatch(row_batch=rows):
                if rows is not self.rows:
                    raise ValueError("hidden input belongs to another packed step")
            case CudaPackedHiddenSnapshotBatch(batch=batch):
                if batch.row_batch is not self.rows:
                    raise ValueError("hidden input belongs to another packed step")


@dataclass(frozen=True, slots=True)
class BackendRequirements:
    hidden_inputs: tuple[HiddenInputSource, ...] = ()
    parameter_targets: tuple[ParameterTarget, ...] = ()


@dataclass(frozen=True, slots=True)
class ParameterTargetShape:
    target: ParameterTarget
    input_size: int
    output_size: int

    def __post_init__(self) -> None:
        if self.input_size < 1 or self.output_size < 1:
            raise ValueError("parameter target dimensions must be positive")


@dataclass(frozen=True, slots=True)
class BackendCompileFacts:
    probe: BackendPlacementProbe
    timeline: SchedulerTimeline
    providers: tuple[PageBundleProviderFacts, ...]
    max_admitted_lineages: int
    state_memory_budget_bytes: int
    initialization_seed: int
    targets: tuple[ParameterTargetShape, ...] = ()


@dataclass(frozen=True, slots=True)
class BackendDeployment(Generic[DataT, ResultT]):
    execution: StateExecution[TensorStepInput, DataT, ResultT]
    attachment: SignedBackendAttachment

    def __post_init__(self) -> None:
        if self.attachment.inputs.plan is not self.execution.arena.plan:
            raise ValueError("backend deployment requires the original execution plan")


class BackendWorkload(ABC, Generic[DataT, ResultT]):
    @abstractmethod
    def backend_requirements(self) -> BackendRequirements: ...

    @abstractmethod
    def compile_backend(self, facts: BackendCompileFacts) -> BackendDeployment[DataT, ResultT]: ...
