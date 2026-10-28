"""vLLM model runners that host one pllm workload session.

Runner installs hidden taps and attaches a VllmSession; WorkloadModelRunner is the MRv2 runner class the
worker instantiates. Used by runtime.py (construction) and pllm.worker.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from vllm.config import CacheConfig, ModelConfig as VllmModelConfig, VllmConfig
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from pllm.authoring import (
    BackendDeployment,
    DataT,
    ResultT,
)
from pllm.backends.vllm.hidden_capture import HiddenTaps
from pllm.backends.vllm.overlay_graph import (
    OverlayGraphRunner,
)
from pllm.backends.vllm.packed_rows import encode_mrv2_rows
from pllm.backends.vllm.position_state import POSITION_ENV, PositionCacheLayer
from pllm.backends.vllm.prefix_cache import CachePolicy
from pllm.backends.vllm.sampler import (
    SamplerGuidance,
    Vllm027Sampler,
)
from pllm.backends.vllm.session import VllmSession
from pllm.config import RunnerKind
from pllm.contracts.backend_attachment import (
    AttachmentGraphMode,
    AttachmentTopology,
)
from pllm.contracts.hidden_locator import HiddenSite
from pllm.contracts.packed_hidden import hidden_binding
from pllm.execution.codec import decode_boundary
from pllm.execution.position_state import PositionAnnouncement
from pllm.execution.state_execution import StateDiagnostics

# Hidden sources the driver announces before construction, so the worker taps them at load time.
TAPS_ENV = "PLLM_HIDDEN_TAPS"


class Runner:
    """The workload attachment slot and persistent sampling hook shared by both runner ABIs."""

    # vLLM's two sampler slot types differ; sessions only need their shared guidance lifecycle.
    _sampler_guidance: SamplerGuidance
    model: torch.nn.Module
    cache_config: CacheConfig
    model_config: VllmModelConfig
    device: torch.device
    max_num_tokens: int
    runner_kind: RunnerKind
    max_num_reqs: int

    hidden_taps: HiddenTaps | None = None
    position_states: tuple[tuple[str, float], ...] = ()
    position_layers: tuple[PositionCacheLayer, ...] = ()

    def qwen2_model(self) -> Qwen2ForCausalLM:
        """Return the model supported by workload taps, probing and sampling."""
        if not isinstance(self.model, Qwen2ForCausalLM):
            raise ValueError("choose a Qwen2 model for vLLM workloads")
        return self.model

    def _install_model_hooks(self) -> None:
        """Install the logits hook and announced hidden taps before vLLM compiles the model."""
        model = self.qwen2_model()
        setattr(model, "compute_logits", partial(self._compute_logits, model))
        requested: list[tuple[int, str]] = json.loads(os.environ.get(TAPS_ENV, "[]"))
        if not requested:
            return
        layers = len(model.model.layers)
        paths = (hidden_binding(layer, HiddenSite(site), layers).module_path for layer, site in requested)
        hidden_size = model.lm_head.weight.shape[1]
        self.hidden_taps = HiddenTaps(model, paths, self.max_num_tokens, hidden_size, self.device)

    overlay_graph_bytes: int = 0

    def _initialize_workload(self, topology: AttachmentTopology) -> None:
        self.workload_topology = topology
        self._workload_session: VllmSession[Any, Any] | None = None

    def attach_workload(
        self,
        deployment: BackendDeployment[DataT, ResultT],
        cache_policy: CachePolicy,
    ) -> None:
        if self.cache_config.enable_prefix_caching and (
            deployment.attachment.spec.activation_bindings or deployment.attachment.spec.overlay_bindings
        ):
            raise ValueError("disable prefix caching when using parameter overlays or activation patches")
        if deployment.attachment.probe.position_states != self.position_states:
            raise ValueError("PositionState schema changed; create a new engine with this workload")
        spec = deployment.attachment.spec
        in_model = (
            spec.overlay_bindings
            or spec.activation_bindings
            or any(task.replay is not None for task in deployment.attachment.plan.semantic_graph.tasks)
        )
        if (
            in_model
            and not self.model_config.enforce_eager
            and deployment.attachment.probe.graph_mode is not AttachmentGraphMode.CUDA_GRAPH
        ):
            raise ValueError("parameter overlays, activation patches and layer replay do not run under CUDA graphs yet")
        model = self.qwen2_model()
        adapter = self._sampler_guidance if spec.guides_sampler else None
        session = VllmSession(
            deployment,
            model,
            adapter,
            self.max_num_reqs,
            self.hidden_taps,
            self.position_layers,
            self.cache_config.block_size,
            cache_policy=cache_policy,
        )
        self._workload_session = session
        try:
            self._finish_model_binding()
        except Exception:
            session.close()
            self._workload_session = None
            raise

    def _finish_model_binding(self) -> None:
        """Finish backend model preparation after the session has installed its effects."""

    def detach_workload(self) -> None:
        session = self.require_session()
        session.require_idle()
        session.close()
        self._workload_session = None

    def require_session(self) -> VllmSession[Any, Any]:
        session = self._workload_session
        assert session is not None
        return session

    def _compute_logits(self, model: Qwen2ForCausalLM, hidden: torch.Tensor) -> torch.Tensor | None:
        session = self._workload_session
        if session is None:
            return Qwen2ForCausalLM.compute_logits(model, hidden)
        return session.compute_logits(model, hidden)


class WorkloadModelRunner(Runner, OverlayGraphRunner):
    runner_kind = RunnerKind.MRV2

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        topology: AttachmentTopology = AttachmentTopology.IN_PROCESS,
    ) -> None:
        super().__init__(vllm_config, device)
        self._initialize_workload(topology)

    def load_model(self, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights)  # pyright: ignore[reportUnknownMemberType]
        assert self.sampler is not None
        self.sampler = self._sampler_guidance = Vllm027Sampler(self.sampler)
        self._install_model_hooks()
        self.position_states = decode_boundary(
            os.environ.get(POSITION_ENV, '{"states": []}'), PositionAnnouncement
        ).states
        if self.position_states and self.cache_config.cache_dtype != "auto":
            raise ValueError("PositionState requires cache_dtype=auto in the model dtype")
        self.position_layers = tuple(
            PositionCacheLayer(1, self.model_config.get_hidden_size(), prefix=f"pllm.position.{index}.state")
            for index, _ in enumerate(
                self.position_states, self.model_config.get_num_layers(self.vllm_config.parallel_config)
            )
        )

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Validate the admitted allocation layout once before binding route state."""
        if self.position_states:
            transfer = self.vllm_config.kv_transfer_config
            if transfer is not None and transfer.kv_connector not in (
                "OffloadingConnector",
                "SimpleCPUOffloadConnector",
            ):
                raise ValueError(
                    "PositionState cannot use LMCache or remote transfer; use OffloadingConnector or SimpleCPUOffloadConnector in batched mode"
                )
            if len(kv_cache_config.kv_cache_groups) != 1 or self.cache_config.prefix_match_unit not in (
                None,
                self.cache_config.block_size,
            ):
                raise ValueError("PositionState requires one KV group with whole allocation-block prefix hits")
        super().initialize_kv_cache(kv_cache_config)

    def _finish_model_binding(self) -> None:
        self.finish_overlay_graphs(self.require_session().attachment, self.require_session().parameter_phases)

    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        batch_desc: BatchExecutionDescriptor,
    ) -> InputBatch:
        input_batch = super().prepare_inputs(scheduler_output, batch_desc)
        session = self._workload_session
        if session is not None:
            session.prepare_input(
                encode_mrv2_rows(
                    input_batch,
                    self.req_states.prompt_len.np[input_batch.idx_mapping_np],
                    tuple(session.identity(request_id) for request_id in input_batch.req_ids),
                    step_id=session.next_step_id,
                )
            )
        return input_batch

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        session = self._workload_session
        if session is None:
            return super().execute_model(
                scheduler_output,
                intermediate_tensors,
                dummy_run,
                skip_attn_for_dummy_run,
                is_profile,
            )
        assert not dummy_run and not is_profile
        try:
            session.before_execute(scheduler_output)
            return super().execute_model(
                scheduler_output,
                intermediate_tensors,
                dummy_run,
                skip_attn_for_dummy_run,
                is_profile,
            )
        except Exception:
            session.abort_active()
            raise

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        session = self._workload_session
        if session is None:
            return super().sample(hidden_states, input_batch, grammar_output)
        try:
            sampled = super().sample(hidden_states, input_batch, grammar_output)
            session.complete_sampler()
        except Exception:
            session.abort_active()
            raise
        return sampled

    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None,
    ) -> AsyncOutput | ModelRunnerOutput | None:
        session = self._workload_session
        if session is None:
            return super().sample_tokens(grammar_output)
        try:
            output = super().sample_tokens(grammar_output)
            session.commit_token()
        except Exception:
            session.abort_active()
            raise
        return output

    def shutdown(self) -> None:
        if self._workload_session is not None:
            self.detach_workload()
        super().shutdown()


@dataclass(frozen=True, slots=True)
class BackendWorkerState:
    process_id: int
    diagnostics: StateDiagnostics
