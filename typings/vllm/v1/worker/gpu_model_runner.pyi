"""MRv1 methods pllm overrides, in the form shared by vLLM 0.27.1 and 0.32.0."""

import numpy as np
import numpy.typing as npt
import torch
from vllm.config import CacheConfig, LoRAConfig, ModelConfig, ParallelConfig, SpeculativeConfig, VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput, SamplerOutput
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import InputBatch


class ExecuteModelState:
    sample_hidden_states: torch.Tensor


class GPUModelRunner:
    model: torch.nn.Module
    sampler: Sampler
    model_config: ModelConfig
    cache_config: CacheConfig
    parallel_config: ParallelConfig
    speculative_config: SpeculativeConfig | None
    lora_config: LoRAConfig | None
    device: torch.device
    dtype: torch.dtype
    max_num_tokens: int
    max_num_reqs: int
    use_async_scheduling: bool
    input_batch: InputBatch
    execute_model_state: ExecuteModelState | None

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None: ...
    def load_model(self, load_dummy_weights: bool = False) -> None: ...
    # Logits indices and speculative metadata; 0.32.0 appends the maximum sampled-token count.
    def _prepare_inputs(self, scheduler_output: SchedulerOutput, num_scheduled_tokens: npt.NDArray[np.int32]
                        ) -> tuple[torch.Tensor, SpecDecodeMetadata | None, *tuple[int, ...]]: ...
    def execute_model(self, scheduler_output: SchedulerOutput, intermediate_tensors: IntermediateTensors | None = None
                      ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None: ...
    def _sample(self, logits: torch.Tensor | None, spec_decode_metadata: SpecDecodeMetadata | None) -> SamplerOutput: ...
    def sample_tokens(self, grammar_output: GrammarOutput | None
                      ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors: ...
