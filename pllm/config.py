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
        if self.connector not in ("OffloadingConnector", "SimpleCPUOffloadConnector"):
            raise ValueError(
                "use OffloadingConnector or SimpleCPUOffloadConnector; LMCache does not support PositionState"
            )
        if self.cpu_bytes <= 0:
            raise ValueError("cpu_bytes must be positive, e.g. 512 * 1024**2")
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
        if self.publish_lag < 0:
            raise ValueError("publish_lag must be >= 0; use 0 for strict ordering")
        if self.tuning not in ("heuristic", "measured"):
            raise ValueError("tuning must be heuristic or measured")
        if self.training_rows < 1:
            raise ValueError("training_rows must be >= 1")
        if self.capture_memory_bytes < 1:
            raise ValueError("capture_memory_bytes must be positive")


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
        if self.generated_kv not in ("reuse", "isolate"):
            raise ValueError('generated_kv must be "reuse" or "isolate"')
