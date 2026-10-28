"""MRv1 event ordering over the shared canonical attachment and original row plan."""

import numpy as np
import numpy.typing as npt
import torch
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput, SamplerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from pllm.backends.vllm.packed_rows import encode_mrv1_rows
from pllm.backends.vllm.runner import Runner
from pllm.backends.vllm.sampler import Mrv1Sampler
from pllm.config import RunnerKind
from pllm.contracts.backend_attachment import AttachmentTopology


class Mrv1Runner(Runner, GPUModelRunner):
    runner_kind = RunnerKind.MRV1

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        topology: AttachmentTopology = AttachmentTopology.IN_PROCESS,
    ) -> None:
        super().__init__(vllm_config, device)
        if self.use_async_scheduling:
            raise ValueError("MRv1 workload requires synchronous persistent batch metadata")
        self.sampler = self._sampler_guidance = Mrv1Sampler(self.sampler)
        self._initialize_workload(topology)

    def load_model(self, load_dummy_weights: bool = False) -> None:
        if self.speculative_config is not None:
            raise ValueError("disable speculative decoding for MRv1 workloads")
        super().load_model(load_dummy_weights)
        self._install_model_hooks()

    def _prepare_inputs(
        self, scheduler_output: SchedulerOutput, num_scheduled_tokens: npt.NDArray[np.int32]
    ) -> tuple[torch.Tensor, SpecDecodeMetadata | None]:
        result = super()._prepare_inputs(scheduler_output, num_scheduled_tokens)
        session = self._workload_session
        if session is not None:
            if result[1] is not None:
                raise RuntimeError("MRv1 workload rejects speculative row expansion")
            session.prepare_input(
                encode_mrv1_rows(
                    self.input_batch,
                    scheduler_output,
                    result[0],
                    tuple(session.identity(request_id) for request_id in self.input_batch.req_ids),
                    plan_epoch=session.attachment.physical.plan_epoch,
                    step_id=session.next_step_id,
                )
            )
        return result

    def execute_model(
        self, scheduler_output: SchedulerOutput, intermediate_tensors: IntermediateTensors | None = None
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        session = self._workload_session
        if session is None:
            return super().execute_model(scheduler_output, intermediate_tensors)
        try:
            session.before_execute(scheduler_output)
            return super().execute_model(scheduler_output, intermediate_tensors)
        except Exception:
            session.abort_active()
            raise

    def _sample(self, logits: torch.Tensor | None, spec_decode_metadata: SpecDecodeMetadata | None) -> SamplerOutput:
        session = self._workload_session
        if session is None:
            return super()._sample(logits, spec_decode_metadata)
        try:
            sampled = super()._sample(logits, spec_decode_metadata)
            session.complete_sampler()
        except Exception:
            session.abort_active()
            raise
        return sampled

    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        session, state = self._workload_session, self.execute_model_state
        if session is None:
            return super().sample_tokens(grammar_output)
        try:
            output = super().sample_tokens(grammar_output)
            if state is not None and self.completion_capture is not None:
                self.completion_capture.write(
                    tuple(session.identity(request_id) for request_id in self.input_batch.req_ids),
                    state.sample_hidden_states,
                )
            session.commit_token()
            return output
        except Exception:
            session.abort_active()
            raise
