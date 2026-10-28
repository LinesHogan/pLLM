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
class KVOffload:
    """Batched same-engine CPU offload of complete allocation pages."""

    connector: Literal["OffloadingConnector", "SimpleCPUOffloadConnector"] = "OffloadingConnector"
    cpu_bytes: int = 512 * 1024**2
    prompt_only: bool = False

    def __post_init__(self) -> None:
        if self.connector not in ("OffloadingConnector", "SimpleCPUOffloadConnector") or self.cpu_bytes <= 0:
            raise ValueError(
                "use OffloadingConnector or SimpleCPUOffloadConnector with positive cpu_bytes; LMCache does not support PositionState"
            )
        if self.prompt_only and self.connector == "SimpleCPUOffloadConnector":
            raise ValueError(
                "prompt_only needs OffloadingConnector; SimpleCPUOffloadConnector offloads completed request pages"
            )

    def options(self) -> dict[str, object]:
        """Configure stock batched transfers with one complete allocation page per chunk."""
        return {
            "kv_connector": self.connector,
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "cpu_bytes_to_use": self.cpu_bytes,
                "blocks_per_chunk": 1,
                "offload_prompt_only": self.prompt_only,
            },
        }


@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    """Framework execution policy; lag counts post-sampler steps hidden from readers."""

    training_rows: int = 1
    capture_memory_bytes: int = 64 * 1024**2
    explicit_math: bool = False
    state_memory_bytes: int = 0  # Zero uses the backend's observed available capacity.
    memory_budget_bytes: int | None = None  # Original AER declared-memory limit, not total CUDA residency.
    allow_numerical_transforms: bool = True
    schedule_policy: Literal["auto", "serial"] = "auto"
    # Deterministic preparation and funded stream overlap by default.
    # Only "measured" searches and times additional implementations.
    tuning: Literal["heuristic", "measured"] = "heuristic"
    publish_lag: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.publish_lag) is not int
            or self.publish_lag < 0
            or self.schedule_policy not in ("auto", "serial")
            or self.tuning not in ("heuristic", "measured")
            or self.training_rows < 1
            or self.capture_memory_bytes < 1
            or self.state_memory_bytes < 0
            or self.memory_budget_bytes is not None
            and self.memory_budget_bytes < 0
        ):
            raise ValueError(
                "ExecutionOptions needs nonnegative int publish_lag, schedule_policy in ('auto', 'serial'), "
                "tuning in ('heuristic', 'measured'), positive training_rows/capture_memory_bytes, and nonnegative memory limits"
            )


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
    enable_chunked_prefill: bool = False
    generated_kv: Literal["reuse", "isolate"] = "isolate"
    revision: str | None = None
    multiprocessing: bool = False
    runner: RunnerKind = RunnerKind.MRV2
    execution: ExecutionOptions = ExecutionOptions()
    # Use vLLM model graphs; in-model effects need a supported graph binding.
    cuda_graphs: bool = False
    kv_offload: KVOffload | None = None

    def __post_init__(self) -> None:
        if (
            self.generated_kv not in ("reuse", "isolate")
            or not self.model
            or self.max_model_len < 2
            or self.max_num_seqs < 1
            or self.max_num_batched_tokens < self.max_num_seqs
            or not 0 < self.gpu_memory_utilization < 1
            or self.seed < 0
        ):
            raise ValueError(
                "ModelConfig needs a non-empty model, generated_kv in ('reuse', 'isolate'), max_model_len >= 2, "
                "1 <= max_num_seqs <= max_num_batched_tokens, 0 < gpu_memory_utilization < 1, and seed >= 0"
            )
