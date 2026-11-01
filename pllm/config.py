"""User limits and model configuration, before backend specialization."""

from dataclasses import dataclass
from typing import Literal, get_args

# vLLM's names for the weight dtype; its aliases "half" and "float" are not accepted.
ModelDType = Literal["bfloat16", "float16", "float32"]
# vLLM model runner: "mrv2" is the V2 runner with async scheduling, "mrv1" the GPUModelRunner.
RunnerName = Literal["mrv1", "mrv2"]


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

    def options(self) -> dict[str, str | dict[str, int | bool]]:
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

    max_training_rows: int = 1
    capture_memory_bytes: int = 64 * 1024**2
    state_memory_bytes: int = 0  # Zero uses the backend's observed available capacity.
    memory_budget_bytes: int | None = None  # Original AER declared-memory limit, not total CUDA residency.
    allow_numerical_transforms: bool = True
    # Deterministic preparation and funded stream overlap by default.
    # Only "measured" searches and times additional implementations.
    tuning: Literal["heuristic", "measured"] = "heuristic"
    publish_lag: int = 1

    def __post_init__(self) -> None:
        if self.publish_lag < 0:
            raise ValueError("publish_lag must be >= 0; use 0 for strict ordering")
        if self.tuning not in ("heuristic", "measured"):
            raise ValueError("tuning must be heuristic or measured")
        if self.max_training_rows < 1:
            raise ValueError("max_training_rows must be >= 1")
        if self.capture_memory_bytes < 1:
            raise ValueError("capture_memory_bytes must be positive")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model: str
    dtype: ModelDType = "bfloat16"
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
    runner: RunnerName = "mrv2"
    execution: ExecutionOptions = ExecutionOptions()
    # "auto" runs the model through vLLM's CUDA graphs when the workload supports them, else eagerly, with the reason
    # in diagnostics().eager_reason; True requires graphs (an unsupported workload raises), False runs eagerly.
    cuda_graphs: Literal["auto"] | bool = "auto"
    kv_offload: KVOffload | None = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1

    def __post_init__(self) -> None:
        if type(self.tensor_parallel_size) is not int or self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be an integer >= 1")
        if self.dtype not in get_args(ModelDType):
            raise ValueError(f'dtype must be "bfloat16", "float16" or "float32"; got {self.dtype!r}')
        if self.runner not in get_args(RunnerName):
            raise ValueError(f'runner must be "mrv2" or "mrv1"; got {self.runner!r}')
        if self.generated_kv not in ("reuse", "isolate"):
            raise ValueError('generated_kv must be "reuse" or "isolate"')
