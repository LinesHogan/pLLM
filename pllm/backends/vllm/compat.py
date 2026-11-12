"""Every difference between the supported vLLM releases, 0.27.1 and 0.32.0, behind 0.32.0's interfaces.

The worker, runner, graph and sampler modules subclass the four classes below and are written against 0.32.0. On
0.32.0 the runner names are vLLM's own classes; on 0.27.1 thin subclasses accept 0.32.0's arguments and
provide its seams. The sampler also exposes the FP32 copy seam on both releases. Dropping 0.27.1 deletes the `if VLLM_027` branch. Pyright checks each branch against its own
release: pyrightconfig.json defines VLLM_027 true for .venv, pyrightconfig.vllm032.json false for .venv-vllm032.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np
import torch
from vllm import __version__
from vllm.envs import disable_envs_cache
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler as VllmSampler
from vllm.v1.worker.gpu_worker import Worker as VllmWorker

from pllm.config import ModelConfig
from pllm.execution.compile_facts import BackendRequirements

SUPPORTED_VLLM = ("0.27.1", "0.32.0")
if __version__ not in SUPPORTED_VLLM:
    raise ImportError(f"pLLM runs on vLLM 0.27.1 or 0.32.0; install one of them instead of vLLM {__version__}")
VLLM_027 = __version__ == "0.27.1"


def admit_parallel(config: ModelConfig, requirements: BackendRequirements = BackendRequirements()) -> None:
    """Reject unsupported parallel configurations before creating any worker processes."""
    for size, name in (
        (config.pipeline_parallel_size, "pipeline_parallel_size"),
        (config.data_parallel_size, "data_parallel_size"),
    ):
        if type(size) is not int or size != 1:
            raise ValueError(
                f"pLLM needs {name}=1: it spreads a model over GPUs only by tensor parallelism; "
                "set tensor_parallel_size instead"
            )
    if requirements.custom_attention and config.tensor_parallel_size != 1:
        raise ValueError("@pllm.custom_attention needs tensor_parallel_size=1: each rank sees only its heads")
    if config.tensor_parallel_size == 1:
        return
    if VLLM_027:
        raise ValueError("multi-GPU needs vLLM 0.32.0; otherwise use tensor_parallel_size=1")
    for used, feature in (
        (requirements.causal_training, "causal_model training"),
        (requirements.feeds, "state feeds"),
        (requirements.custom_residual, "@pllm.custom_residual"),
    ):
        if used:
            raise ValueError(f"{feature} runs on one GPU only; set tensor_parallel_size=1")


if VLLM_027:
    from contextlib import AbstractContextManager
    from typing import Any

    from vllm.config.model import PROCESSED_LOGPROBS_MODES
    from vllm.model_executor.layers import batch_invariant
    from vllm.sequence import IntermediateTensors
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, flashinfer_sample
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    class Worker(VllmWorker):
        """0.27.1 builds its runner inside init_device; 0.32.0 asks this factory, which pLLM's worker overrides."""

        def init_device(self) -> None:
            super().init_device()
            self.model_runner = self._make_model_runner()

        def _make_model_runner(self) -> GPUModelRunner:
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

else:
    from vllm.model_executor.determinism import batch_invariant
    from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsContext

    Worker = VllmWorker
    ModelRunner = GPUModelRunner
    CudaGraphManager = ModelCudaGraphManager


class Sampler(VllmSampler):
    """Upstream parameter pipelines with a sampler-input copy seam for workload additions.

    Neither release exposes that seam; the bodies below retain its processor order
    and delegate all processing/filtering kernels to vLLM. On 0.27 sample() also
    exposes 0.32's draw seam for post-filter guidance.
    """

    def _requires_logits_processing(self, idx_mapping_np: np.ndarray) -> bool:
        if VLLM_027:
            return super()._requires_logits_processing(idx_mapping_np)
        return bool(np.any(self.needs_logits_processing[idx_mapping_np]))

    def _copy_logits(self, logits: torch.Tensor, idx_mapping_np: np.ndarray) -> torch.Tensor:
        return torch.empty_like(logits, dtype=torch.float32).copy_(logits)

    def _apply_sampling_filters(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        skip_top_k_top_p: bool,
    ) -> torch.Tensor:
        self.sampling_states.apply_temperature(logits, expanded_idx_mapping, idx_mapping_np)
        self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)
        return (
            logits
            if skip_top_k_top_p
            else self.sampling_states.apply_top_k_top_p(logits, expanded_idx_mapping, idx_mapping_np)
        )

    if VLLM_027:

        def apply_sampling_params(
            self,
            logits: torch.Tensor,
            expanded_idx_mapping: torch.Tensor,
            idx_mapping_np: np.ndarray,
            pos: torch.Tensor,
            input_ids: torch.Tensor,
            expanded_local_pos: torch.Tensor,
            skip_top_k_top_p: bool = False,
        ) -> torch.Tensor:
            if not self._requires_logits_processing(idx_mapping_np):
                return logits

            # Guidance can fold additions into this copy before any processors run.
            logits = self._copy_logits(logits, idx_mapping_np)

            self.logit_bias_state.apply_logit_bias(logits, expanded_idx_mapping, idx_mapping_np, pos)

            self.penalties_state.apply_penalties(
                logits,
                expanded_idx_mapping,
                idx_mapping_np,
                input_ids,
                expanded_local_pos,
            )

            self.bad_words_state.apply_bad_words(
                logits,
                expanded_idx_mapping,
                idx_mapping_np,
                input_ids,
                expanded_local_pos,
            )

            return self._apply_sampling_filters(logits, expanded_idx_mapping, idx_mapping_np, skip_top_k_top_p)

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

        def apply_sampling_params(
            self,
            logits: torch.Tensor,
            expanded_idx_mapping: torch.Tensor,
            idx_mapping: torch.Tensor,
            idx_mapping_np: np.ndarray,
            pos: torch.Tensor,
            input_ids: torch.Tensor,
            expanded_local_pos: torch.Tensor,
            seq_lens_upper_bound_np: np.ndarray,
            skip_top_k_top_p: bool = False,
        ) -> torch.Tensor:
            if not self._requires_logits_processing(idx_mapping_np):
                return logits

            # Guidance can fold additions into this copy before any processors run.
            logits = self._copy_logits(logits, idx_mapping_np)

            ctx = LogitsContext(
                expanded_idx_mapping=expanded_idx_mapping,
                idx_mapping=idx_mapping,
                idx_mapping_np=idx_mapping_np,
                expanded_local_pos=expanded_local_pos,
                input_ids=input_ids,
                pos=pos,
                seq_lens_upper_bound_np=seq_lens_upper_bound_np,
            )

            for processor in self.logits_processors:
                logits = processor.apply(logits, ctx)

            # Forcing runs last so no stage can overwrite the forced end marker
            # or weaken it by scaling.
            self.thinking_budget_state.apply(logits, ctx)

            return self._apply_sampling_filters(logits, expanded_idx_mapping, idx_mapping_np, skip_top_k_top_p)


@contextmanager
def batch_invariance() -> Generator[None]:
    """vLLM's batch-invariant kernels (VLLM_BATCH_INVARIANT=1) for the engines started inside, removed on exit.

    vLLM installs them process-wide when an engine starts and never removes them, so this installs them first and
    afterwards restores the environment, torch's matmul flags and the aten kernels (unless they were installed
    before). Inside, matmul and linear stay differentiable: on SM80/SM89 vLLM registers CUDA kernels for them, which
    hides their autograd decomposition ("matmul_backward" fails), so autograd gets stock torch's decomposition into
    mm/addmm back. 0.27.1 also rebinds torch.bmm, which bypasses dispatch modes (pLLM's memory tracing would launch
    it on fake tensors); its aten::bmm kernel already covers bmm, so torch's binding stays. vLLM caches its
    environment once an engine starts, so that cache is dropped on both sides.
    """
    installed = batch_invariant._batch_invariant_MODE  # pyright: ignore[reportPrivateUsage]
    environment = dict(os.environ)
    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    precision = matmul.fp32_precision, cudnn.conv.fp32_precision, cudnn.rnn.fp32_precision
    reduction = matmul.allow_fp16_reduced_precision_reduction, matmul.allow_bf16_reduced_precision_reduction
    blas, bmm = torch.backends.cuda.preferred_blas_library(), torch.bmm
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    disable_envs_cache()
    batch_invariant.init_batch_invariance()
    torch.bmm = bmm
    autograd = torch.library.Library("aten", "IMPL")
    for operator in (torch.ops.aten.matmul.default, torch.ops.aten.linear.default):
        autograd.impl(operator, operator.decompose, "AutogradCUDA")  # pyright: ignore[reportUnknownMemberType]
    try:
        yield
    finally:
        autograd._destroy()  # pyright: ignore[reportPrivateUsage]
        if not installed:
            library = batch_invariant._batch_invariant_LIB  # pyright: ignore[reportPrivateUsage]
            assert library is not None
            library._destroy()  # pyright: ignore[reportPrivateUsage]
            batch_invariant._batch_invariant_LIB = None  # pyright: ignore[reportPrivateUsage]
            batch_invariant._batch_invariant_MODE = False  # pyright: ignore[reportPrivateUsage]
            os.environ.clear()
            os.environ.update(environment)
            disable_envs_cache()
            matmul.fp32_precision, cudnn.conv.fp32_precision, cudnn.rnn.fp32_precision = precision
            matmul.allow_fp16_reduced_precision_reduction, matmul.allow_bf16_reduced_precision_reduction = reduction
            torch.backends.cuda.preferred_blas_library(blas)
