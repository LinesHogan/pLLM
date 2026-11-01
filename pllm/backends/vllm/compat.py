"""Every difference between the supported vLLM releases, 0.27.1 and 0.32.0, behind 0.32.0's interfaces.

The worker, runner, graph and sampler modules subclass the four classes below and are written against 0.32.0. On
0.32.0 the names are vLLM's own classes; on 0.27.1 they are thin subclasses that accept 0.32.0's arguments and
provide its seams. Dropping 0.27.1 deletes the `if VLLM_027` branch. Pyright checks each branch against its own
release: pyrightconfig.json defines VLLM_027 true for .venv, pyrightconfig.vllm032.json false for .venv-vllm032.
"""

from vllm import __version__
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler as VllmSampler
from vllm.v1.worker.gpu_worker import Worker as VllmWorker

SUPPORTED_VLLM = ("0.27.1", "0.32.0")
if __version__ not in SUPPORTED_VLLM:
    raise ImportError(f"pllm runs on vLLM 0.27.1 or 0.32.0; install one of them instead of vLLM {__version__}")
VLLM_027 = __version__ == "0.27.1"

if VLLM_027:
    from contextlib import AbstractContextManager
    from typing import Any

    import numpy as np
    import torch
    from vllm.config.model import PROCESSED_LOGPROBS_MODES
    from vllm.sequence import IntermediateTensors
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, flashinfer_sample
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as Mrv1GPUModelRunner

    class Worker(VllmWorker):
        """0.27.1 builds its runner inside init_device; 0.32.0 asks this factory, which pllm's worker overrides."""

        def init_device(self) -> None:
            super().init_device()
            self.model_runner = self._make_model_runner()

        def _make_model_runner(self) -> GPUModelRunner | Mrv1GPUModelRunner:
            return self.model_runner

    class ModelRunner(GPUModelRunner):
        """The MRv2 runner taking 0.32.0's extra arguments; 0.27.1 never passes them, so they keep their defaults."""

        def initialize_kv_cache(
            self,
            kv_cache_config: KVCacheConfig,
            is_profiling: bool = False,
            kv_cache_allocation_context: AbstractContextManager[Any, bool | None] | None = None,
        ) -> None:
            super().initialize_kv_cache(kv_cache_config)

        def capture_model(self, *, profile_only: bool = False) -> int:
            return super().capture_model()

        def execute_model(
            self,
            scheduler_output: SchedulerOutput,
            intermediate_tensors: IntermediateTensors | None = None,
            dummy_run: bool = False,
            skip_attn_for_dummy_run: bool = False,
            is_profile: bool = False,
            context_len: int = 0,
            valid_dummy_state_slots: bool = False,
            randomize_inputs: bool = False,
        ) -> ModelRunnerOutput | IntermediateTensors | None:
            return super().execute_model(
                scheduler_output, intermediate_tensors, dummy_run, skip_attn_for_dummy_run, is_profile
            )

    class CudaGraphManager(ModelCudaGraphManager):
        """The graph manager whose dispatch takes 0.32.0's query-length and micro-batch keywords."""

        def dispatch(
            self,
            num_reqs: int,
            num_tokens: int,
            uniform_token_count: int | None,
            num_active_loras: int,
            max_query_len: int | None = None,
            num_ubatches: int = 1,
        ) -> BatchExecutionDescriptor:
            return super().dispatch(num_reqs, num_tokens, uniform_token_count, num_active_loras)

    class Sampler(VllmSampler):
        """0.27.1's MRv2 sampler split as in 0.32.0: sample() applies the sampling parameters, _sample_random() draws."""

        def sample(
            self,
            logits: torch.Tensor,
            expanded_idx_mapping: torch.Tensor,
            idx_mapping_np: np.ndarray,
            pos: torch.Tensor,
            input_ids: torch.Tensor,
            expanded_local_pos: torch.Tensor,
            return_logprobs: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            processed_logits = self.apply_sampling_params(
                logits, expanded_idx_mapping, idx_mapping_np, pos, input_ids, expanded_local_pos, skip_top_k_top_p=True
            )
            top_k, top_p = self.sampling_states.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
            use_flashinfer = self.use_flashinfer and not (
                (top_k is None and top_p is None)
                or (return_logprobs and self.logprobs_mode in PROCESSED_LOGPROBS_MODES)
                or self.sampling_states.any_greedy(idx_mapping_np)
                or self.sampling_states.any_explicit_seed(idx_mapping_np)
            )
            return self._sample_random(
                processed_logits, expanded_idx_mapping, idx_mapping_np, pos, top_k, top_p, use_flashinfer
            )

        def _sample_random(
            self,
            processed_logits: torch.Tensor,
            expanded_idx_mapping: torch.Tensor,
            idx_mapping_np: np.ndarray,
            pos: torch.Tensor,
            top_k: torch.Tensor | None,
            top_p: torch.Tensor | None,
            use_fused_sampler: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if use_fused_sampler:
                return flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64), processed_logits
            processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
            sampled = gumbel_sample(
                processed_logits,
                expanded_idx_mapping,
                self.sampling_states.temperature.gpu,
                self.sampling_states.seeds.gpu,
                pos,
                apply_temperature=False,
                use_fp64=self.use_fp64_gumbel,
            )
            return sampled, processed_logits

else:
    Worker = VllmWorker
    ModelRunner = GPUModelRunner
    CudaGraphManager = ModelCudaGraphManager
    Sampler = VllmSampler
