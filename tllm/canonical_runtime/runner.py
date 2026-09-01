"""Concrete vLLM 0.27.1 MRv2 runner with one canonical ESamp session."""

from __future__ import annotations

import torch

from tllm.canonical.workloads.esamp_request import ESampRequestWorkload
from tllm.canonical_runtime.deployment import (
    CanonicalEsampVllmDeployment,
    compile_esamp_vllm_deployment,
)
from tllm.canonical_runtime.session import CanonicalEsampVllmSession

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
from vllm.v1.worker.gpu.sample.sampler import Sampler


class CanonicalGPUModelRunner(GPUModelRunner):
    """MRv2 runner subclass; all canonical state remains instance-owned."""

    __slots__ = (
        "_canonical_deployment",
        "_canonical_session",
        "_canonical_upstream_sampler",
    )

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, device)
        self._canonical_deployment: CanonicalEsampVllmDeployment | None = None
        self._canonical_session: CanonicalEsampVllmSession | None = None
        self._canonical_upstream_sampler: Sampler | None = None

    @property
    def canonical_deployment(self) -> CanonicalEsampVllmDeployment | None:
        return self._canonical_deployment

    def attach_esamp(
        self,
        workload: ESampRequestWorkload,
    ) -> CanonicalEsampVllmDeployment:
        """Compile and attach only after vLLM has loaded the concrete model."""

        if self._canonical_session is not None:
            raise RuntimeError("canonical runner already owns a workload session")
        match self.model:
            case Qwen2ForCausalLM() as model:
                pass
            case _:
                raise RuntimeError("canonical ESamp runner requires loaded Qwen2")
        sampler = self.sampler
        if type(sampler) is not Sampler:
            raise RuntimeError("canonical ESamp runner requires the stock MRv2 sampler")
        deployment = compile_esamp_vllm_deployment(workload, self)
        session = CanonicalEsampVllmSession(deployment, model, sampler)
        self._canonical_upstream_sampler = sampler
        self._canonical_deployment = deployment
        self._canonical_session = session
        self.sampler = session.sampler
        return deployment

    def detach_esamp(self) -> None:
        """Retire canonical state and restore the exact upstream sampler instance."""

        session = self._canonical_session
        upstream = self._canonical_upstream_sampler
        if session is None or upstream is None:
            raise RuntimeError("canonical runner has no attached ESamp session")
        session.close()
        self.sampler = upstream
        self._canonical_session = None
        self._canonical_upstream_sampler = None
        self._canonical_deployment = None

    def retire_completed_esamp_requests(self) -> None:
        session = self._canonical_session
        if session is None:
            raise RuntimeError("canonical runner has no attached ESamp session")
        with torch.inference_mode(False):
            session.retire_all()

    def abort_esamp_requests(self) -> None:
        session = self._canonical_session
        if session is None:
            raise RuntimeError("canonical runner has no attached ESamp session")
        with torch.inference_mode(False):
            session.abort_active()
            session.retire_all()

    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        batch_desc: BatchExecutionDescriptor,
    ) -> InputBatch:
        input_batch = super().prepare_inputs(scheduler_output, batch_desc)
        session = self._canonical_session
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
        session = self._canonical_session
        if session is None:
            return super().execute_model(
                scheduler_output,
                intermediate_tensors,
                dummy_run,
                skip_attn_for_dummy_run,
                is_profile,
            )
        if dummy_run or is_profile:
            raise RuntimeError("attached canonical execution rejects runner warmup")
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
            with torch.inference_mode(False):
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
        session = self._canonical_session
        if session is None:
            return super().sample(hidden_states, input_batch, grammar_output)
        match self.model:
            case Qwen2ForCausalLM() as model:
                pass
            case _:
                raise RuntimeError("attached canonical sampler lost its Qwen2 model")
        try:
            active = session.begin_sampler(model)
            sampled = super().sample(hidden_states, input_batch, grammar_output)
            session.complete_sampler(active)
        except Exception:
            session.abort_active()
            raise
        return sampled

    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None,
    ) -> AsyncOutput | ModelRunnerOutput | None:
        session = self._canonical_session
        if session is None:
            return super().sample_tokens(grammar_output)
        try:
            output = super().sample_tokens(grammar_output)
            with torch.inference_mode(False), torch.enable_grad():
                session.commit_token()
        except Exception:
            session.abort_active()
            raise
        return output

    def shutdown(self) -> None:
        if self._canonical_session is not None:
            self.detach_esamp()
        super().shutdown()


__all__ = ["CanonicalGPUModelRunner"]
