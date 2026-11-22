"""Every difference between supported vLLM releases (0.27 through 0.34) behind a common runner/sampler seam.

The worker, runner, graph and sampler modules subclass the four classes below and program against one uniform
interface (`ModelRunner`, `CudaGraphManager`, `Worker`, `Sampler`). When a release shifts upstream signatures,
adapter subclasses translate calls here. Pyright checks branches via `defineConstant` in `pyrightconfig*.json`.
`release_offload_pins` works around a vLLM connector bug shared across releases.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import torch
import vllm.envs as vllm_envs
from vllm import __version__
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import SimpleCPUOffloadConnector
from vllm.model_executor.kernels.linear.mixed_precision.marlin import MarlinLinearKernel
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler as VllmSampler
from vllm.v1.worker.gpu_worker import Worker as VllmWorker

from pllm.config import ModelConfig
from pllm.execution.compile_facts import BackendRequirements

if TYPE_CHECKING:
    from argparse import Namespace

    from fastapi import FastAPI
    from starlette.datastructures import State
    from vllm.config import ModelConfig as VllmModelConfig
    from vllm.engine.protocol import EngineClient
    from vllm.tasks import SupportedTask


def _vllm_minor(version: str) -> tuple[int, int]:
    parts = version.split("+", 1)[0].split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return (0, 0)
    return (int(parts[0]), int(parts[1]))


VLLM_MINOR = _vllm_minor(__version__)
if not ((0, 27) <= VLLM_MINOR <= (0, 34)):
    raise ImportError(
        f"pLLM runs on vLLM 0.27 through 0.34; install a supported 0.27..0.34 release instead of vLLM {__version__}"
    )

VLLM_027 = VLLM_MINOR == (0, 27)
VLLM_028_030 = (0, 28) <= VLLM_MINOR <= (0, 30)
VLLM_031 = VLLM_MINOR == (0, 31)
VLLM_033_PLUS = VLLM_MINOR >= (0, 33)
VLLM_034 = VLLM_MINOR == (0, 34)
VLLM_TP_CAPABLE = VLLM_MINOR >= (0, 32)
GUMBEL_MODE = 0 if VLLM_MINOR <= (0, 28) else 1 if VLLM_MINOR <= (0, 31) else 2 if VLLM_MINOR <= (0, 33) else 3


def marlin_has_g_idx(kernel: MarlinLinearKernel) -> bool:
    """Only vLLM <= 0.29 retains activation-order g_idx metadata on MarlinLinearKernel."""
    if VLLM_027:
        return bool(kernel.config.has_g_idx)
    if VLLM_028_030:
        return bool(VLLM_MINOR <= (0, 29) and kernel.config.has_g_idx)
    return False


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
    if config.tensor_parallel_size == 1:
        return
    if not VLLM_TP_CAPABLE:
        raise ValueError("multi-GPU needs vLLM 0.32.0 or newer; otherwise use tensor_parallel_size=1")
    for used, feature in (
        (requirements.causal_training, "causal_model training"),
        (requirements.custom_value_site, "@pllm.custom_residual using a value site"),
        (
            requirements.custom_attention_tp_unsafe,
            "@pllm.custom_attention with cross-head facts or a custom KV layout",
        ),
    ):
        if used:
            raise ValueError(f"{feature} runs on one GPU only; set tensor_parallel_size=1")


if VLLM_034:
    from vllm.config.utils import SupportsCompileFactors as AdditionalConfigFactor
else:
    from vllm.config.utils import SupportsHash as AdditionalConfigFactor

__all__ = [
    "VLLM_MINOR",
    "VLLM_027",
    "VLLM_028_030",
    "VLLM_031",
    "VLLM_033_PLUS",
    "VLLM_034",
    "VLLM_TP_CAPABLE",
    "GUMBEL_MODE",
    "AdditionalConfigFactor",
    "Worker",
    "ModelRunner",
    "CudaGraphManager",
    "Sampler",
    "admit_parallel",
    "batch_invariant_enabled",
    "build_openai_server_app",
    "clone_graph_manager",
    "enable_batch_invariance",
    "init_openai_server_state",
    "marlin_has_g_idx",
    "release_offload_pins",
]


if VLLM_027:
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
        """<=0.31 builds its runner inside init_device; >=0.32 asks this factory, which pLLM's worker overrides."""

        def init_device(self) -> None:
            super().init_device()
            self.model_runner = self._make_model_runner()

        def _make_model_runner(self) -> GPUModelRunner:
            return self.model_runner

    class ModelRunner(GPUModelRunner):
        """MRv2 runner normalized to 0.34's initialize_kv_cache / capture_model / execute_model signatures."""

        def initialize_kv_cache(
            self,
            kv_cache_config: KVCacheConfig,
            is_profiling: bool = False,
            kv_cache_allocation_context: AbstractContextManager[Any, bool | None] | None = None,
            extensible: bool = False,
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
        """Graph manager normalizing 0.27's 4-arg dispatch to 6-arg dispatch."""

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

elif VLLM_028_030:
    from vllm.config.model import PROCESSED_LOGPROBS_MODES
    from vllm.sequence import IntermediateTensors
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, flashinfer_sample
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    if VLLM_MINOR < (0, 29):
        from vllm.model_executor.layers import batch_invariant
    else:
        from vllm.model_executor.determinism import batch_invariant

    class Worker(VllmWorker):
        """<=0.31 builds its runner inside init_device; >=0.32 asks this factory."""

        def init_device(self) -> None:
            super().init_device()
            self.model_runner = self._make_model_runner()

        def _make_model_runner(self) -> GPUModelRunner:
            return self.model_runner

    class ModelRunner(GPUModelRunner):
        """MRv2 runner normalizing 0.28..0.30 signatures to the 0.34 interface."""

        def initialize_kv_cache(
            self,
            kv_cache_config: KVCacheConfig,
            is_profiling: bool = False,
            kv_cache_allocation_context: AbstractContextManager[Any, bool | None] | None = None,
            extensible: bool = False,
        ) -> None:
            if VLLM_MINOR == (0, 28):
                super().initialize_kv_cache(kv_cache_config)
            else:
                super().initialize_kv_cache(kv_cache_config, is_profiling, kv_cache_allocation_context)

        def capture_model(self, *, profile_only: bool = False) -> int:
            if VLLM_MINOR < (0, 30):
                return super().capture_model()
            return super().capture_model(profile_only=profile_only)

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
            if VLLM_MINOR < (0, 30):
                return super().execute_model(
                    scheduler_output,
                    intermediate_tensors,
                    dummy_run,
                    skip_attn_for_dummy_run,
                    is_profile,
                    context_len,
                )
            return super().execute_model(
                scheduler_output,
                intermediate_tensors,
                dummy_run,
                skip_attn_for_dummy_run,
                is_profile,
                context_len,
                valid_dummy_state_slots,
            )

    class CudaGraphManager(ModelCudaGraphManager):
        """Graph manager normalizing 0.28..0.29 5-arg dispatch to 6-arg dispatch."""

        def dispatch(
            self,
            num_reqs: int,
            num_tokens: int,
            uniform_token_count: int | None,
            num_active_loras: int,
            max_query_len: int | None = None,
            num_ubatches: int = 1,
        ) -> BatchExecutionDescriptor:
            if VLLM_MINOR < (0, 30):
                return super().dispatch(num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len)
            return super().dispatch(
                num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len, num_ubatches
            )

else:
    from vllm.model_executor.determinism import batch_invariant
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsContext

    if VLLM_031:

        class Worker(VllmWorker):
            def init_device(self) -> None:
                super().init_device()
                self.model_runner = self._make_model_runner()

            def _make_model_runner(self) -> GPUModelRunner:
                return self.model_runner
    else:
        Worker = VllmWorker

    if VLLM_034:
        ModelRunner = GPUModelRunner
    else:

        class ModelRunner(GPUModelRunner):
            """Normalize 0.31..0.33 initialize_kv_cache by absorbing 0.34's `extensible` keyword."""

            def initialize_kv_cache(
                self,
                kv_cache_config: KVCacheConfig,
                is_profiling: bool = False,
                kv_cache_allocation_context: AbstractContextManager[Any, bool | None] | None = None,
                extensible: bool = False,
            ) -> None:
                super().initialize_kv_cache(kv_cache_config, is_profiling, kv_cache_allocation_context)  # pyright: ignore[reportUnknownMemberType]

    CudaGraphManager = ModelCudaGraphManager


GraphManagerT = TypeVar("GraphManagerT", bound=ModelCudaGraphManager)


def clone_graph_manager(
    cls: type[GraphManagerT],
    config: VllmConfig,
    device: torch.device,
    manager: ModelCudaGraphManager,
) -> GraphManagerT:
    """Construct a CudaGraphManager subclass copying its decode-case configuration from `manager`."""
    if VLLM_034:
        return cls(config, device, manager.cudagraph_mode, manager.decode_max_reqs)
    return cls(config, device, manager.cudagraph_mode, manager.decode_query_len)


class Sampler(VllmSampler):
    """Upstream parameter pipelines with sampler-input copy and `_draw_random` seams across 0.27..0.34."""

    def _requires_logits_processing(self, idx_mapping_np: np.ndarray) -> bool:
        if VLLM_027 or VLLM_028_030:
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

    def _draw_random(
        self,
        processed_logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        use_fused_sampler: bool,
        has_topp_only_rows: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

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
            logits = self._copy_logits(logits, idx_mapping_np)
            self.logit_bias_state.apply_logit_bias(logits, expanded_idx_mapping, idx_mapping_np, pos)
            self.penalties_state.apply_penalties(
                logits, expanded_idx_mapping, idx_mapping_np, input_ids, expanded_local_pos
            )
            self.bad_words_state.apply_bad_words(
                logits, expanded_idx_mapping, idx_mapping_np, input_ids, expanded_local_pos
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
            return self._draw_random(
                processed_logits, expanded_idx_mapping, idx_mapping_np, pos, top_k, top_p, use_flashinfer
            )

        def _draw_random(
            self,
            processed_logits: torch.Tensor,
            expanded_idx_mapping: torch.Tensor,
            idx_mapping_np: np.ndarray,
            pos: torch.Tensor,
            top_k: torch.Tensor | None,
            top_p: torch.Tensor | None,
            use_fused_sampler: bool,
            has_topp_only_rows: bool = False,
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

    elif VLLM_028_030:

        def apply_sampling_params(
            self,
            logits: torch.Tensor,
            expanded_idx_mapping: torch.Tensor,
            idx_mapping: torch.Tensor,
            idx_mapping_np: np.ndarray,
            pos: torch.Tensor,
            input_ids: torch.Tensor,
            expanded_local_pos: torch.Tensor,
            skip_top_k_top_p: bool = False,
        ) -> torch.Tensor:
            if not self._requires_logits_processing(idx_mapping_np):
                return logits
            logits = self._copy_logits(logits, idx_mapping_np)
            self.logit_bias_state.apply_logit_bias(logits, expanded_idx_mapping, idx_mapping_np, pos)
            self.penalties_state.apply_penalties(
                logits, expanded_idx_mapping, idx_mapping_np, input_ids, expanded_local_pos
            )
            self.bad_words_state.apply_bad_words(
                logits, expanded_idx_mapping, idx_mapping_np, input_ids, expanded_local_pos
            )
            self.thinking_budget_state.apply(
                logits, expanded_idx_mapping, idx_mapping, idx_mapping_np, input_ids, expanded_local_pos
            )
            return self._apply_sampling_filters(logits, expanded_idx_mapping, idx_mapping_np, skip_top_k_top_p)

        if VLLM_MINOR < (0, 30):

            def sample(
                self,
                logits: torch.Tensor,
                expanded_idx_mapping: torch.Tensor,
                idx_mapping: torch.Tensor,
                idx_mapping_np: np.ndarray,
                pos: torch.Tensor,
                input_ids: torch.Tensor,
                expanded_local_pos: torch.Tensor,
                return_logprobs: bool = False,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                processed_logits = self.apply_sampling_params(
                    logits,
                    expanded_idx_mapping,
                    idx_mapping,
                    idx_mapping_np,
                    pos,
                    input_ids,
                    expanded_local_pos,
                    skip_top_k_top_p=True,
                )
                top_k, top_p = self.sampling_states.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
                use_flashinfer = self.use_flashinfer and not (
                    (top_k is None and top_p is None)
                    or (return_logprobs and self.logprobs_mode in PROCESSED_LOGPROBS_MODES)
                    or self.sampling_states.any_greedy(idx_mapping_np)
                    or self.sampling_states.any_explicit_seed(idx_mapping_np)
                )
                return self._draw_random(
                    processed_logits, expanded_idx_mapping, idx_mapping_np, pos, top_k, top_p, use_flashinfer
                )

            def _draw_random(
                self,
                processed_logits: torch.Tensor,
                expanded_idx_mapping: torch.Tensor,
                idx_mapping_np: np.ndarray,
                pos: torch.Tensor,
                top_k: torch.Tensor | None,
                top_p: torch.Tensor | None,
                use_fused_sampler: bool,
                has_topp_only_rows: bool = False,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if use_fused_sampler:
                    return flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64), processed_logits
                processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
                kwargs = {} if VLLM_MINOR < (0, 29) else {"is_drafting": False}
                sampled = gumbel_sample(
                    processed_logits,
                    expanded_idx_mapping,
                    self.sampling_states.temperature.gpu,
                    self.sampling_states.seeds.gpu,
                    pos,
                    apply_temperature=False,
                    use_fp64=self.use_fp64_gumbel,
                    **kwargs,
                )
                return sampled, processed_logits

        else:

            def _sample_random(
                self,
                processed_logits: torch.Tensor,
                expanded_idx_mapping: torch.Tensor,
                idx_mapping_np: np.ndarray,
                pos: torch.Tensor,
                top_k: torch.Tensor | None,
                top_p: torch.Tensor | None,
                use_flashinfer: bool,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return self._draw_random(
                    processed_logits, expanded_idx_mapping, idx_mapping_np, pos, top_k, top_p, use_flashinfer
                )

            def _draw_random(
                self,
                processed_logits: torch.Tensor,
                expanded_idx_mapping: torch.Tensor,
                idx_mapping_np: np.ndarray,
                pos: torch.Tensor,
                top_k: torch.Tensor | None,
                top_p: torch.Tensor | None,
                use_fused_sampler: bool,
                has_topp_only_rows: bool = False,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return super()._sample_random(
                    processed_logits, expanded_idx_mapping, idx_mapping_np, pos, top_k, top_p, use_fused_sampler
                )

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

            self.thinking_budget_state.apply(logits, ctx)

            return self._apply_sampling_filters(logits, expanded_idx_mapping, idx_mapping_np, skip_top_k_top_p)

        if VLLM_033_PLUS:

            def _sample_random(
                self,
                processed_logits: torch.Tensor,
                expanded_idx_mapping: torch.Tensor,
                idx_mapping_np: np.ndarray,
                pos: torch.Tensor,
                top_k: torch.Tensor | None,
                top_p: torch.Tensor | None,
                has_topp_only_rows: bool,
                use_fused_sampler: bool,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return self._draw_random(
                    processed_logits,
                    expanded_idx_mapping,
                    idx_mapping_np,
                    pos,
                    top_k,
                    top_p,
                    use_fused_sampler,
                    has_topp_only_rows,
                )

            def _draw_random(
                self,
                processed_logits: torch.Tensor,
                expanded_idx_mapping: torch.Tensor,
                idx_mapping_np: np.ndarray,
                pos: torch.Tensor,
                top_k: torch.Tensor | None,
                top_p: torch.Tensor | None,
                use_fused_sampler: bool,
                has_topp_only_rows: bool = False,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return super()._sample_random(
                    processed_logits,
                    expanded_idx_mapping,
                    idx_mapping_np,
                    pos,
                    top_k,
                    top_p,
                    has_topp_only_rows,
                    use_fused_sampler,
                )

        else:

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
                return self._draw_random(
                    processed_logits,
                    expanded_idx_mapping,
                    idx_mapping_np,
                    pos,
                    top_k,
                    top_p,
                    use_fused_sampler,
                )

            def _draw_random(
                self,
                processed_logits: torch.Tensor,
                expanded_idx_mapping: torch.Tensor,
                idx_mapping_np: np.ndarray,
                pos: torch.Tensor,
                top_k: torch.Tensor | None,
                top_p: torch.Tensor | None,
                use_fused_sampler: bool,
                has_topp_only_rows: bool = False,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return super()._sample_random(
                    processed_logits,
                    expanded_idx_mapping,
                    idx_mapping_np,
                    pos,
                    top_k,
                    top_p,
                    use_fused_sampler,
                )


_batch_invariant_enabled = False


def batch_invariant_enabled() -> bool:
    """Whether vLLM installed its batch-invariant ATen kernels in this process."""
    return bool(vllm_envs.VLLM_BATCH_INVARIANT or _batch_invariant_enabled)


def enable_batch_invariance() -> torch.library.Library:
    """Install vLLM's batch-invariant kernels in a process started with VLLM_BATCH_INVARIANT=1, before its engines.

    Two repairs keep pLLM working under them. On SM80/SM89 vLLM registers CUDA kernels for aten::matmul and
    aten::linear, which hides their autograd decomposition ("matmul_backward" fails); autograd gets stock torch's
    decomposition into mm/addmm (themselves batch-invariant) back. <=0.28 also rebinds torch.bmm, which bypasses
    dispatch modes (pLLM's memory tracing would launch it on fake tensors); its aten::bmm kernel already covers bmm,
    so torch's binding stays. Keep the returned library alive as long as the process trains.
    """
    global _batch_invariant_enabled
    bmm = torch.bmm
    batch_invariant.init_batch_invariance()
    _batch_invariant_enabled = True
    torch.bmm = bmm
    autograd = torch.library.Library("aten", "IMPL")
    for operator in (torch.ops.aten.matmul.default, torch.ops.aten.linear.default):
        autograd.impl(operator, operator.decompose, "AutogradCUDA")  # pyright: ignore[reportUnknownMemberType]
    return autograd


def release_offload_pins(config: VllmConfig) -> None:
    """Stop SimpleCPUOffloadConnector's copies and unpin its CPU KV cache before the worker is dropped.

    vLLM bug: the connector pins its CPU KV cache tensors with cudaHostRegister but never unregisters them, and its
    shutdown leaves the copy thread running. When the tensors are freed, CUDA still treats those host addresses as
    pinned, so a later non-blocking upload reads reused memory. Delete once upstream unpins cleanly.
    """
    transfer = config.kv_transfer_config
    if transfer is None or transfer.kv_connector != "SimpleCPUOffloadConnector" or not has_kv_transfer_group():
        return
    connector = get_kv_transfer_group()
    assert isinstance(connector, SimpleCPUOffloadConnector)
    worker = connector.worker_handler
    if worker is None or worker.cpu_kv_caches is None:
        return
    assert worker._backend is not None and worker.load_stream is not None and worker.store_stream is not None  # pyright: ignore[reportPrivateUsage]
    worker._backend.shutdown()  # pyright: ignore[reportPrivateUsage]
    worker.load_stream.synchronize()
    worker.store_stream.synchronize()
    for tensor in worker.cpu_kv_caches.values():
        torch.cuda.cudart().cudaHostUnregister(tensor.data_ptr())  # pyright: ignore[reportOptionalMemberAccess, reportUnknownMemberType]
    worker.cpu_kv_caches.clear()


def build_openai_server_app(
    args: Namespace,
    supported_tasks: tuple[SupportedTask, ...],
    model_config: VllmModelConfig,
) -> FastAPI:
    """Construct vLLM's OpenAI FastAPI app across 0.27..0.34 without triggering deprecation warnings."""
    if VLLM_027 or VLLM_028_030 or VLLM_031:
        from vllm.entrypoints.openai.api_server import build_app
    else:
        from vllm.entrypoints.launchers.app import build_app
    return build_app(args, supported_tasks, model_config)


async def init_openai_server_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    supported_tasks: tuple[SupportedTask, ...],
) -> None:
    """Initialize vLLM's OpenAI app state across 0.27..0.34."""
    if VLLM_027 or VLLM_028_030 or VLLM_031:
        from vllm.entrypoints.openai.api_server import init_app_state
    else:
        from vllm.entrypoints.launchers.api_server.app_state import init_app_state
    await init_app_state(engine_client, state, args, supported_tasks)
