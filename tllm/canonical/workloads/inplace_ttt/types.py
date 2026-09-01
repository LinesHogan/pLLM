"""Strict values for canonical request-scoped dense fast weights."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256

import torch

from tllm.canonical.ptts.dense_page_planning import DensePageCpuProviderFacts
from tllm.contracts.parameter_overlay import (
    ParameterOverlayPublication,
    ParameterTarget,
)
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.semantic_graph import SchedulerTimeline
from tllm.contracts.state_ref import StateRef
from tllm.contracts.state_representation import StatePhysicalPlan
from tllm.contracts.tensor_layout import TensorDType


class InPlaceTTTPhase(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True, slots=True)
class InPlaceTTTConfig:
    targets: tuple[ParameterTarget, ...]
    chunk_size: int = 4096
    learning_rate: float = 1e-3
    update_prefill: bool = True
    update_decode: bool = False

    def __post_init__(self) -> None:
        if (
            not self.targets
            or tuple(sorted(self.targets)) != self.targets
            or len(self.targets) != len(set(self.targets))
            or self.chunk_size < 1
            or self.learning_rate <= 0.0
        ):
            raise ValueError("In-Place-TTT mathematical configuration is invalid")

    def updates(self, phase: InPlaceTTTPhase) -> bool:
        match phase:
            case InPlaceTTTPhase.PREFILL:
                return self.update_prefill
            case InPlaceTTTPhase.DECODE:
                return self.update_decode

    def execution_parameters_digest(self) -> str:
        fields = (
            "canonical.inplace_ttt.parameters@1",
            f"chunk_size={self.chunk_size}",
            f"learning_rate={self.learning_rate.hex()}",
            f"update_prefill={int(self.update_prefill)}",
            f"update_decode={int(self.update_decode)}",
            *(
                "target="
                + str(target.layer)
                + ":"
                + target.module.value
                + ":"
                + target.role.value
                for target in self.targets
            ),
        )
        return sha256("\x00".join(fields).encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class InPlaceTTTTargetSpecialization:
    target: ParameterTarget
    input_size: int
    output_size: int

    def __post_init__(self) -> None:
        if self.input_size < 1 or self.output_size < 1:
            raise ValueError("dense fast-weight target dimensions must be positive")


@dataclass(frozen=True, slots=True)
class InPlaceTTTSpecialization:
    targets: tuple[InPlaceTTTTargetSpecialization, ...]
    tensor_dtype: TensorDType
    device: torch.device

    def __post_init__(self) -> None:
        target_ids = tuple(target.target for target in self.targets)
        if (
            not self.targets
            or tuple(sorted(target_ids)) != target_ids
            or len(target_ids) != len(set(target_ids))
            or self.tensor_dtype not in (TensorDType.FLOAT32, TensorDType.FLOAT64)
            or self.device.type != "cpu"
        ):
            raise ValueError("canonical In-Place-TTT currently requires CPU floats")

    def execution_parameters_digest(self, config: InPlaceTTTConfig) -> str:
        if tuple(target.target for target in self.targets) != config.targets:
            raise ValueError("In-Place-TTT targets differ from specialization")
        fields = (
            "canonical.inplace_ttt.execution@1",
            config.execution_parameters_digest(),
            f"tensor_dtype={self.tensor_dtype.value}",
            f"device={self.device}",
            *(
                "shape="
                + str(target.target.layer)
                + ":"
                + target.target.module.value
                + ":"
                + target.target.role.value
                + f":{target.input_size}:{target.output_size}"
                for target in self.targets
            ),
        )
        return sha256("\x00".join(fields).encode("ascii")).hexdigest()

    @property
    def torch_dtype(self) -> torch.dtype:
        match self.tensor_dtype:
            case TensorDType.FLOAT32:
                return torch.float32
            case TensorDType.FLOAT64:
                return torch.float64
            case _:
                raise AssertionError("In-Place-TTT specialization lost its dtype")


@dataclass(frozen=True, slots=True)
class InPlaceTTTCompileOptions:
    timeline: SchedulerTimeline
    providers: tuple[DensePageCpuProviderFacts, ...]
    max_admitted_lineages: int
    state_memory_budget_bytes: int
    plan_epoch: int = 1

    def __post_init__(self) -> None:
        provider_ids = tuple(provider.provider_id for provider in self.providers)
        if (
            not self.providers
            or tuple(sorted(provider_ids)) != provider_ids
            or len(provider_ids) != len(set(provider_ids))
            or self.max_admitted_lineages < 1
            or self.state_memory_budget_bytes < 0
            or self.plan_epoch < 1
        ):
            raise ValueError("In-Place-TTT physical compile facts are invalid")


@dataclass(frozen=True, slots=True)
class InPlaceTTTOperandPair:
    target: ParameterTarget
    phase: InPlaceTTTPhase
    hidden: torch.Tensor
    desired: torch.Tensor
    base_weight: torch.Tensor


@dataclass(frozen=True, slots=True)
class InPlaceTTTStepInput:
    identity: RequestIdentity
    operands: tuple[InPlaceTTTOperandPair, ...]

    def __post_init__(self) -> None:
        targets = tuple(operand.target for operand in self.operands)
        if (
            not targets
            or tuple(sorted(targets)) != targets
            or len(targets) != len(set(targets))
        ):
            raise ValueError("In-Place-TTT operands must be sorted and unique")


@dataclass(frozen=True, slots=True)
class InPlaceTTTTargetResult:
    target: ParameterTarget
    output: torch.Tensor
    update_count: int
    trained_rows: int
    tail_rows: int


@dataclass(frozen=True, slots=True)
class InPlaceTTTStepResult:
    identity: RequestIdentity
    before_ref: StateRef
    after_ref: StateRef
    targets: tuple[InPlaceTTTTargetResult, ...]
    publication: ParameterOverlayPublication | None


@dataclass(frozen=True, slots=True)
class InPlaceTTTCleanupReceipt:
    retired: tuple[StateRef, ...]


@dataclass(frozen=True, slots=True)
class InPlaceTTTDiagnostics:
    active_lineages: int
    update_count: int
    trained_rows: int
    state_plan: StatePhysicalPlan
    fallback_count: int


__all__ = [
    "InPlaceTTTCleanupReceipt",
    "InPlaceTTTCompileOptions",
    "InPlaceTTTConfig",
    "InPlaceTTTDiagnostics",
    "InPlaceTTTOperandPair",
    "InPlaceTTTPhase",
    "InPlaceTTTSpecialization",
    "InPlaceTTTStepInput",
    "InPlaceTTTStepResult",
    "InPlaceTTTTargetResult",
    "InPlaceTTTTargetSpecialization",
]
