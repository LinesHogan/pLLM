"""Concrete MRv2 runner for one signed canonical parameter-overlay session."""

from __future__ import annotations

import torch

from tllm.canonical.workloads.inplace_ttt import InPlaceTTTWorkload
from tllm.canonical_runtime.parameter_overlay_deployment import (
    CanonicalParameterOverlayDeployment,
    compile_parameter_overlay_vllm_deployment,
)
from tllm.canonical_runtime.parameter_overlay_session import (
    CanonicalParameterOverlaySession,
)

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.output import SamplerOutput


class CanonicalParameterOverlayGPUModelRunner(GPUModelRunner):
    """Runner-owned generic overlay session with no workload scheduling policy."""

    __slots__ = ("_overlay_deployment", "_overlay_session")

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, device)
        self._overlay_deployment: CanonicalParameterOverlayDeployment | None = None
        self._overlay_session: CanonicalParameterOverlaySession | None = None

    @property
    def overlay_deployment(self) -> CanonicalParameterOverlayDeployment | None:
        return self._overlay_deployment

    @property
    def overlay_apply_count(self) -> int:
        session = self._overlay_session
        return 0 if session is None else session.apply_count

    @property
    def overlay_updated_apply_count(self) -> int:
        session = self._overlay_session
        return 0 if session is None else session.updated_apply_count

    def attach_parameter_overlay(
        self,
        workload: InPlaceTTTWorkload,
    ) -> CanonicalParameterOverlayDeployment:
        if self._overlay_session is not None:
            raise RuntimeError("parameter-overlay runner already has a session")
        match self.model:
            case Qwen2ForCausalLM() as model:
                pass
            case _:
                raise RuntimeError("parameter-overlay runner requires Qwen2")
        deployment = compile_parameter_overlay_vllm_deployment(workload, self)
        session = CanonicalParameterOverlaySession(deployment, model)
        self._overlay_deployment = deployment
        self._overlay_session = session
        return deployment

    def detach_parameter_overlay(self) -> None:
        session = self._overlay_session
        if session is None:
            raise RuntimeError("parameter-overlay runner has no session")
        session.close()
        self._overlay_session = None
        self._overlay_deployment = None

    def retire_completed_parameter_overlay_requests(self) -> None:
        session = self._require_session()
        with torch.inference_mode(False):
            session.retire_all()

    def abort_parameter_overlay_requests(self) -> None:
        session = self._require_session()
        with torch.inference_mode(False):
            session.abort_active()
            session.retire_all()

    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        batch_desc: BatchExecutionDescriptor,
    ) -> InputBatch:
        input_batch = super().prepare_inputs(scheduler_output, batch_desc)
        session = self._overlay_session
        if session is not None:
            session.prepare_input(input_batch)
        return input_batch

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        session = self._overlay_session
        if session is None:
            return super().execute_model(
                scheduler_output,
                intermediate_tensors,
                dummy_run,
                skip_attn_for_dummy_run,
                is_profile,
            )
        if dummy_run or is_profile:
            raise RuntimeError("attached parameter overlay rejects runner warmup")
        with torch.inference_mode(False):
            session.before_execute(scheduler_output)
        try:
            result = super().execute_model(
                scheduler_output,
                intermediate_tensors,
                dummy_run,
                skip_attn_for_dummy_run,
                is_profile,
            )
            with torch.inference_mode(False), torch.enable_grad():
                session.complete_forward()
        except Exception:
            session.abort_active()
            raise
        return result

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        session = self._overlay_session
        if session is None:
            return super().sample(hidden_states, input_batch, grammar_output)
        try:
            sampled = super().sample(hidden_states, input_batch, grammar_output)
            with torch.inference_mode(False):
                session.complete_sampler()
        except Exception:
            session.abort_active()
            raise
        return sampled

    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None,
    ) -> AsyncOutput | ModelRunnerOutput | None:
        session = self._overlay_session
        if session is None:
            return super().sample_tokens(grammar_output)
        try:
            output = super().sample_tokens(grammar_output)
            with torch.inference_mode(False):
                session.commit_token()
        except Exception:
            session.abort_active()
            raise
        return output

    def shutdown(self) -> None:
        if self._overlay_session is not None:
            self.detach_parameter_overlay()
        super().shutdown()

    def _require_session(self) -> CanonicalParameterOverlaySession:
        session = self._overlay_session
        if session is None:
            raise RuntimeError("parameter-overlay runner has no session")
        return session


__all__ = ["CanonicalParameterOverlayGPUModelRunner"]
