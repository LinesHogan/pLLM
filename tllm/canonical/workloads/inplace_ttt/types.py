"""Strict values for canonical request-scoped dense fast weights."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256

import torch

from tllm.canonical.cuda_stream import CudaStreamReadiness
from tllm.canonical.ptts.dense_page_planning import DensePageProviderFacts
from tllm.contracts.parameter_overlay import (
    ParameterOverlayPinnedTensor,
    ParameterOverlayPublication,
    ParameterTarget,
)
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.semantic_graph import SchedulerTimeline
from tllm.contracts.state_ref import StateRef
from tllm.contracts.state_representation import StateDeviceKind, StatePhysicalPlan
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
            or self.tensor_dtype
            not in (
                TensorDType.FLOAT16,
                TensorDType.BFLOAT16,
                TensorDType.FLOAT32,
                TensorDType.FLOAT64,
            )
            or self.device.type not in ("cpu", "cuda")
            or (self.device.type == "cpu" and self.device != torch.device("cpu"))
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
            case TensorDType.FLOAT16:
                return torch.float16
            case TensorDType.BFLOAT16:
                return torch.bfloat16
            case TensorDType.FLOAT32:
                return torch.float32
            case TensorDType.FLOAT64:
                return torch.float64
            case _:
                raise AssertionError("In-Place-TTT specialization lost its dtype")

    @property
    def device_kind(self) -> StateDeviceKind:
        return (
            StateDeviceKind.CPU
            if self.device.type == "cpu"
            else StateDeviceKind.CUDA
        )

    @property
    def device_ordinal(self) -> int:
        return 0 if self.device.type == "cpu" else int(self.device.index)


@dataclass(frozen=True, slots=True)
class InPlaceTTTCompileOptions:
    timeline: SchedulerTimeline
    providers: tuple[DensePageProviderFacts, ...]
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
class InPlaceTTTStepBatch:
    entries: tuple[InPlaceTTTStepInput, ...]

    def __post_init__(self) -> None:
        identities = tuple(entry.identity for entry in self.entries)
        if not identities or len(identities) != len(set(identities)):
            raise ValueError("In-Place-TTT step batch requires unique requests")


class CudaInPlaceTTTStepBatchInput:
    """One typed operand batch plus its CUDA producer readiness."""

    __slots__ = ("_readiness", "batch")

    def __init__(
        self,
        batch: InPlaceTTTStepBatch,
        readiness: CudaStreamReadiness,
    ) -> None:
        self.batch = batch
        self._readiness = readiness

    @classmethod
    def assert_produced_on(
        cls,
        batch: InPlaceTTTStepBatch,
        producer_stream: torch.cuda.Stream,
    ) -> "CudaInPlaceTTTStepBatchInput":
        return cls(batch, CudaStreamReadiness(producer_stream))

    def order_on(self, consumer_stream: torch.cuda.Stream) -> None:
        self._readiness.order_on(consumer_stream)


@dataclass(frozen=True, slots=True)
class InPlaceTTTCudaStepHandle:
    handle_id: int
    identities: tuple[RequestIdentity, ...]
    refs: tuple[StateRef, ...]
    deltas: tuple[ParameterOverlayPinnedTensor, ...]

    def __post_init__(self) -> None:
        if self.handle_id < 1:
            raise ValueError("CUDA In-Place-TTT handle requires an ID")


def validate_inplace_ttt_step(
    step: InPlaceTTTStepInput,
    specialization: InPlaceTTTSpecialization,
) -> None:
    specialization_by_target = {
        target.target: target for target in specialization.targets
    }
    for operand in step.operands:
        try:
            target = specialization_by_target[operand.target]
        except KeyError as exc:
            raise ValueError("operand target is outside specialization") from exc
        if (
            operand.hidden.ndim != 2
            or operand.desired.ndim != 2
            or operand.base_weight.ndim != 2
            or tuple(operand.hidden.shape[1:]) != (target.input_size,)
            or tuple(operand.desired.shape)
            != (int(operand.hidden.shape[0]), target.output_size)
            or tuple(operand.base_weight.shape)
            != (target.output_size, target.input_size)
            or operand.hidden.dtype is not operand.desired.dtype
            or operand.hidden.dtype is not operand.base_weight.dtype
            or operand.hidden.dtype is not specialization.torch_dtype
            or int(operand.hidden.shape[0]) < 1
            or operand.hidden.device != specialization.device
            or operand.desired.device != specialization.device
            or operand.base_weight.device != specialization.device
        ):
            raise ValueError("operand tensors differ from specialization")


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
class InPlaceTTTStepBatchResult:
    entries: tuple[InPlaceTTTStepResult, ...]


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
    pending_state_releases: int


__all__ = [
    "CudaInPlaceTTTStepBatchInput",
    "InPlaceTTTCleanupReceipt",
    "InPlaceTTTCompileOptions",
    "InPlaceTTTConfig",
    "InPlaceTTTDiagnostics",
    "InPlaceTTTOperandPair",
    "InPlaceTTTPhase",
    "InPlaceTTTSpecialization",
    "InPlaceTTTStepInput",
    "InPlaceTTTStepBatch",
    "InPlaceTTTStepBatchResult",
    "InPlaceTTTCudaStepHandle",
    "InPlaceTTTStepResult",
    "InPlaceTTTTargetResult",
    "InPlaceTTTTargetSpecialization",
    "validate_inplace_ttt_step",
]
