"""vLLM graph startup for model hooks: deferred capture and hybrid dispatch.

This file holds only what a vLLM upgrade rewrites; hook math lives in parameter_overlay.py and model_sites.py.

Depends on GPUModelRunner.load_model/profile_run/capture_model, the capture call
from Worker.compile_or_warm_up_model, the decoder body’s do_not_compile flag,
ModelCudaGraphManager construction, VllmConfig.compute_hash and
aot_compile_hash_factors at the first deferred decoder body call, and
model compilation inside the upstream CUDA graph wrappers.
Hybrid dispatch uses GPUModelRunner.update_requests and ModelCudaGraphManager.dispatch
before prepare_inputs, and the decoder body’s do_not_compile flag to bypass compiled training bodies.
The upstream manager owns all capture/replay loops; PTTS owns every request version.
"""

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any, Protocol, TypedDict

import numpy as np
import torch
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode, VllmConfig
from vllm.config.utils import SupportsHash
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from pllm.backends.vllm import compat
from pllm.backends.vllm.admission import model_hooks
from pllm.backends.vllm.architecture import DecoderBody, bind_model
from pllm.backends.vllm.parameter_overlay import capacities, uniform_decode
from pllm.config import ModelConfig
from pllm.contracts.backend_attachment import AttachmentGraphMode, BackendAttachment
from pllm.contracts.codec import boundary_digest, decode_boundary, encode_boundary
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.execution.compile_facts import BackendRequirements
from pllm.execution.cuda_graph import conditional_graph_pool
from pllm.execution.torch_private import compilation_cleanups, cuda_graph_pool_bytes, release_compilation

_BUDGET = "pllm_model_graph_pool_cap_bytes"
_OVERLAY_GRAPHS = CUDAGraphMode.FULL_AND_PIECEWISE


class _GraphConfiguration(Protocol):
    @property
    def additional_config(self) -> Mapping[str, int | str] | SupportsHash: ...


def _additional_config(config: _GraphConfiguration) -> dict[str, int]:
    """pLLM's entries in vLLM's compile-cache key; runtime.py adds its forward digest as an integer, too."""
    return decode_boundary(encode_boundary(config.additional_config), dict[str, int])


def _graph_budget(config: _GraphConfiguration) -> int:
    value = _additional_config(config).get(_BUDGET, 0)
    if value < 0:
        raise ValueError("Overlay graph memory must be a nonnegative byte budget")
    return value


class GraphOptions(TypedDict, total=False):
    """The `LLM(...)` arguments for graph sizing and deferred overlay compilation."""

    additional_config: dict[str, int]
    compilation_config: CompilationConfig


def _compiler_backend(config: ModelConfig, requirements: BackendRequirements) -> str:
    """Select vLLM's compiler from measured hook costs and capture constraints."""
    if not config.execution.allow_numerical_transforms:
        # The numerical opt-out retains the model's source operators.
        return "eager"
    if requirements.custom_residual:
        # All-site LAuReL is 4.8% slower with Inductor on 7B/4090; stateful residuals fail nested FX tracing.
        return "eager"
    if requirements.parameter_targets:
        # The projection op mutates pools: AOT rejects packed optimizer views with different autograd bases.
        # Exact updates also recompile inside conditional graphs ("Generator not registered").
        return "eager"
    if any(layout.keep == "window" for layout in requirements.kv_layouts):
        # LocalGlobal is 4.6% slower with Inductor on 7B/4090; a repeat confirmed the loss with zero preemptions.
        return "eager"
    return "inductor"


def graph_options(config: ModelConfig, requirements: BackendRequirements) -> GraphOptions:
    """Announce model hooks before profiling; ordinary workloads keep vLLM's capture sizes."""
    if not model_hooks(requirements):
        return {}
    conditional_updates = requirements.stateful_residual or bool(requirements.exact_phases)
    # Request memories and exact decode updates both capture only decode capacity.
    capacity = (
        min(config.max_num_batched_tokens, config.max_num_seqs)
        if conditional_updates
        else config.max_num_batched_tokens
    )
    # Conditional subgraphs own library scratch; all decode counts share one capacity.
    sizes = (capacity,) if HiddenPhase.DECODE in requirements.exact_phases else capacities(capacity)
    return {
        "additional_config": {_BUDGET: config.execution.cuda_graph_pool_budget_bytes or 0},
        "compilation_config": CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            backend=_compiler_backend(config, requirements),
            cudagraph_mode=(CUDAGraphMode.FULL_DECODE_ONLY if conditional_updates else _OVERLAY_GRAPHS),
            cudagraph_num_of_warmups=1 if requirements.custom_residual else 0,
            cudagraph_capture_sizes=list(sizes),
            max_cudagraph_capture_size=capacity,
        ),
    }


class _HybridGraphManager(compat.CudaGraphManager):
    """Keep upstream decode replay ownership; prefill uses the unpadded source model."""

    def __init__(
        self,
        config: VllmConfig,
        device: torch.device,
        mode: CUDAGraphMode,
        decode_query_len: int,
    ) -> None:
        super().__init__(config, device, mode, decode_query_len)
        self.eager = False

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
        max_query_len: int | None = None,
        num_ubatches: int = 1,
    ) -> BatchExecutionDescriptor:
        """Use an unpadded source batch during prefill."""
        if self.eager:
            return BatchExecutionDescriptor(CUDAGraphMode.NONE, num_tokens, num_reqs)
        return super().dispatch(
            num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len, num_ubatches
        )


class OverlayGraphRunner(compat.ModelRunner):
    """Defer first compilation/capture until the ordinary workload installation completes."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, device)
        self.projection_model: DecoderBody | None = None
        self.overlay_graph_budget_bytes = _graph_budget(vllm_config)
        self._compile_cleanups: tuple[int, ...] = ()

    def load_model(self, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights)  # pyright: ignore[reportUnknownMemberType]
        self.decoder = bind_model(self.model)
        if (
            _BUDGET in _additional_config(self.vllm_config)
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            self.overlay_graph_mode = AttachmentGraphMode.CUDA_GRAPH
            self.projection_model = self.decoder.model.model
            self.projection_model.do_not_compile = True

    def shutdown(self) -> None:
        """Release the captured bodies' compiled-code globals before dropping the bound decoder."""
        super().shutdown()
        release_compilation(self._compile_cleanups)
        self._compile_cleanups = ()
        self.projection_model = None

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
        kv_cache_allocation_context: AbstractContextManager[Any, bool | None] | None = None,
    ) -> None:
        """vLLM picks the graph mode here; it must see the compiled body that finish_overlay_graphs captures.

        vLLM 0.32 silently resolves FULL_AND_PIECEWISE to decode-only graphs for a body that is not compiled,
        so the deferred compilation is lifted while it resolves.
        """
        deferred = self.projection_model
        if deferred is not None:
            deferred.do_not_compile = False
        super().initialize_kv_cache(kv_cache_config, is_profiling, kv_cache_allocation_context)  # pyright: ignore[reportUnknownMemberType]
        if deferred is not None:
            deferred.do_not_compile = True

    def capture_model(self, *, profile_only: bool = False) -> int:
        if self.projection_model is not None and self.projection_model.do_not_compile:
            return 0
        with conditional_graph_pool(current_platform.get_global_graph_pool()):
            return super().capture_model(profile_only=profile_only)

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        """Choose placement from CPU counters after upstream request updates."""
        super().update_requests(scheduler_output)
        manager = self.cudagraph_manager
        if isinstance(manager, _HybridGraphManager) and scheduler_output.num_scheduled_tokens:
            indices = np.fromiter(
                (self.req_states.req_id_to_index[key] for key in scheduler_output.num_scheduled_tokens), np.int64
            )
            start = self.req_states.num_computed_tokens_np[indices]
            prompt = self.req_states.prompt_len.np[indices]
            manager.eager = not uniform_decode(
                bool((start < prompt).any()), tuple(scheduler_output.num_scheduled_tokens.values())
            )
            assert self.projection_model is not None
            self.projection_model.do_not_compile = manager.eager

    def finish_overlay_graphs(self, attachment: BackendAttachment) -> None:
        """Capture only the bound model; dummy tables have no request ownership or live destinations."""
        if self.projection_model is None:
            return
        # AOT loads before inspecting wrappers: the complete bound plan must enter its key.
        self.vllm_config.additional_config = {
            **_additional_config(self.vllm_config),
            "pllm_pinned_projection_plan": boundary_digest(attachment),
        }
        manager = self.cudagraph_manager
        assert manager is not None
        mode = self.vllm_config.compilation_config.cudagraph_mode
        if manager.cudagraph_mode != mode:
            raise ValueError(
                f"vLLM resolved overlay CUDA graphs to {manager.cudagraph_mode.name}, not {mode.name}; "
                "use FlashAttention2 or set ModelConfig(cuda_graphs=False)"
            )
        if attachment.plan.semantic_graph.updates_in_forward:
            self.cudagraph_manager = _HybridGraphManager(
                self.vllm_config, self.device, manager.cudagraph_mode, manager.decode_query_len
            )
        self.projection_model.do_not_compile = False
        pool = current_platform.get_global_graph_pool()
        # vLLM shares this pool across engines; charge this engine's added residency.
        before = cuda_graph_pool_bytes(pool, self.device.index or 0)
        cleanups = compilation_cleanups()
        try:
            self.capture_model()
        finally:
            self._compile_cleanups += tuple(compilation_cleanups() - cleanups)
        used = cuda_graph_pool_bytes(pool, self.device.index or 0) - before
        if self.overlay_graph_budget_bytes and used > self.overlay_graph_budget_bytes:
            raise ValueError(
                f"CUDA graphs of the adapted model need {used} bytes; raise ExecutionOptions.cuda_graph_pool_budget_bytes"
            )
