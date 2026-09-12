"""User limits and model configuration, before backend specialization."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class VllmDType(StrEnum):
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class RunnerKind(StrEnum):
    MRV1 = "mrv1"
    MRV2 = "mrv2"


@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    capture: bool = False
    training_rows: int = 1
    capture_memory_bytes: int = 64 * 1024**2
    explicit_math: bool = False
    state_memory_bytes: int = 0  # Zero uses the backend's observed available capacity.
    memory_budget_bytes: int | None = None  # Original AER declared-memory limit, not total CUDA residency.
    allow_numerical_transforms: bool = True
    schedule_policy: Literal["auto", "serial"] = "auto"

    def __post_init__(self) -> None:
        if self.schedule_policy not in ("auto", "serial"):
            raise ValueError("schedule_policy must be auto or serial")
        if (self.training_rows < 1 or self.capture_memory_bytes < 1 or self.state_memory_bytes < 0
            or self.memory_budget_bytes is not None and self.memory_budget_bytes < 0):
            raise ValueError("training rows/capture capacity must be positive and memory limits nonnegative")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model: str
    dtype: VllmDType = VllmDType.BFLOAT16
    max_model_len: int = 512
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 512
    gpu_memory_utilization: float = 0.5
    seed: int = 0
    trust_remote_code: bool = False
    enable_prefix_caching: bool = False
    revision: str | None = None
    multiprocessing: bool = False
    runner: RunnerKind = RunnerKind.MRV2
    execution: ExecutionOptions = ExecutionOptions()

    def __post_init__(self) -> None:
        if (not self.model or self.max_model_len < 2 or self.max_num_seqs < 1
            or self.max_num_batched_tokens < self.max_num_seqs
            or not 0 < self.gpu_memory_utilization < 1 or self.seed < 0):
            raise ValueError("invalid model configuration")
