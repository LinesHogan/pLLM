"""vLLM model runners that host one pllm workload session.

Runner installs hidden taps and attaches a VllmSession; WorkloadModelRunner is the MRv2 runner class the
worker instantiates. Used by runtime.py (construction) and pllm.worker.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import CacheConfig, ModelConfig as VllmModelConfig
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from pllm.authoring import (
    BackendDeployment,
    DataT,
    ResultT,
)
from pllm.backends.vllm.architecture import Decoder
from pllm.backends.vllm.checkpoint import CheckpointLoader as CheckpointLoader
from pllm.backends.vllm.hidden_taps import (
    TAPS_ENV,
    HiddenTapAnnouncement,
    HiddenTapLayout,
    HiddenTaps,
)
from pllm.backends.vllm.overlay_graph import (
    OverlayGraphRunner,
)
from pllm.backends.vllm.packed_rows import encode_mrv2_rows
from pllm.backends.vllm.position_state import POSITION_ENV, PositionCacheLayer
from pllm.backends.vllm.prefix_cache import CachePolicy
from pllm.backends.vllm.sampler import GuidedSampler, SamplerGuidance
from pllm.backends.vllm.session import VllmSession
from pllm.config import RunnerName
from pllm.contracts.codec import decode_boundary
from pllm.execution.position_state import PositionAnnouncement
from pllm.execution.ptts.slot_pools import SlotPools
from pllm.execution.state_execution import StateDiagnostics


class Runner:
    """The workload attachment slot and persistent sampling hook shared by both runner ABIs."""

    # vLLM's two sampler slot types differ; sessions only need their shared guidance lifecycle.
    _sampler_guidance: SamplerGuidance
    model: torch.nn.Module
    cache_config: CacheConfig
    model_config: VllmModelConfig
    device: torch.device
    max_num_tokens: int
    runner_kind: RunnerName
    max_num_reqs: int

    hidden_taps: HiddenTaps | None = None
    hidden_layouts: tuple[HiddenTapLayout, ...] = ()
    hidden_pools: dict[tuple[str, int], SlotPools]
    position_states: tuple[tuple[str, float], ...] = ()
    position_layers: tuple[PositionCacheLayer, ...] = ()

    decoder: Decoder

    def _install_model_hooks(self) -> None:
        """Install the logits hook and announced hidden taps before vLLM compiles the model."""
        model = self.decoder.model
        model.compute_logits = self._compute_logits
        announcement = decode_boundary(os.environ.get(TAPS_ENV, "{}"), HiddenTapAnnouncement)
        hidden_size = model.lm_head.weight.shape[1]
        self.hidden_layouts = announcement.resolve(len(model.model.layers), hidden_size, model.config.vocab_size)
        self.hidden_pools = {layout.tap: layout.create_pool(self.device) for layout in self.hidden_layouts}
        if not self.hidden_layouts:
            return
        self.hidden_taps = HiddenTaps(model, self.hidden_layouts, self.hidden_pools, self.device)

    overlay_graph_bytes: int = 0
    _workload_session: VllmSession[Any, Any] | None = None

    def attach_workload(
        self,
        deployment: BackendDeployment[DataT, ResultT],
        cache_policy: CachePolicy,
    ) -> None:
        spec = deployment.attachment.spec
        if self.cache_config.enable_prefix_caching and (spec.activation_bindings or spec.overlay_bindings):
            raise ValueError("disable prefix caching when using parameter overlays or activation patches")
        if deployment.attachment.probe.position_states != self.position_states:
            raise ValueError("PositionState schema changed; create a new engine with this workload")
        adapter = self._sampler_guidance if spec.sampler else None
        session = VllmSession(
            deployment,
            self.decoder,
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

    def _compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        session = self._workload_session
        if session is None:
            return self.decoder.compute_logits(hidden_states)
        return session.compute_logits(hidden_states)


class WorkloadModelRunner(Runner, OverlayGraphRunner):
    runner_kind: RunnerName = "mrv2"

    def load_model(self, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights)
        assert self.sampler is not None
        self.sampler = self._sampler_guidance = GuidedSampler(self.sampler)
        self._install_model_hooks()
        self.position_states = decode_boundary(
            os.environ.get(POSITION_ENV, '{"states": []}'), PositionAnnouncement
        ).states
        if self.position_states:
            self.decoder.kv_writable("a PositionState")
            if self.cache_config.cache_dtype != "auto":
                raise ValueError("PositionState requires cache_dtype=auto in the model dtype")
        self.position_layers = tuple(
            PositionCacheLayer(1, self.model_config.get_hidden_size(), prefix=f"pllm.position.{index}.state")
            for index, _ in enumerate(
                self.position_states, self.model_config.get_num_layers(self.vllm_config.parallel_config)
            )
        )

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
        kv_cache_allocation_context: AbstractContextManager[Any, bool | None] | None = None,
    ) -> None:
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
        super().initialize_kv_cache(kv_cache_config, is_profiling, kv_cache_allocation_context)

    def _finish_model_binding(self) -> None:
        self.finish_overlay_graphs(self.require_session().attachment)

    def capture_model(self, *, profile_only: bool = False) -> int:
        """Capture and warm up with no hidden-slot destinations, including deferred capture."""
        if self.hidden_taps is not None:
            self.hidden_taps.clear()
        return super().capture_model(profile_only=profile_only)

    def prepare_attn(self, input_batch: InputBatch) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Publish the step's rows to the session.

        vLLM calls this once per real step, right after prepare_inputs built the InputBatch; unlike
        prepare_inputs its signature is the same on every supported vLLM release.
        """
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
        return super().prepare_attn(input_batch)

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
        arguments = (
            scheduler_output,
            intermediate_tensors,
            dummy_run,
            skip_attn_for_dummy_run,
            is_profile,
            context_len,
            valid_dummy_state_slots,
            randomize_inputs,
        )
        session = self._workload_session
        if session is None or dummy_run or is_profile:
            if self.hidden_taps is not None:
                self.hidden_taps.clear()
            return super().execute_model(*arguments)
        try:
            session.before_execute(scheduler_output)
            return super().execute_model(*arguments)
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
