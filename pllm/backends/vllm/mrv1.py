"""MRv1 event ordering over the shared canonical attachment and original row plan."""

import numpy as np
import numpy.typing as npt
import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput, SamplerOutput
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from pllm.backends.vllm.packed_rows import encode_mrv1_rows
from pllm.backends.vllm.runtime import Runner
from pllm.backends.vllm.sampler import Mrv1Sampler
from pllm.config import RunnerKind
from pllm.contracts.backend_attachment import AttachmentTopology


class Mrv1Runner(Runner[Sampler], GPUModelRunner):
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
        self._initialize_workload(topology)

    def _make_sampler(self, upstream: Sampler) -> Sampler:
        return Mrv1Sampler(upstream)

    def _prepare_inputs(
        self, scheduler_output: SchedulerOutput, num_scheduled_tokens: npt.NDArray[np.int32]
    ) -> tuple[torch.Tensor, SpecDecodeMetadata | None]:
        result = super()._prepare_inputs(scheduler_output, num_scheduled_tokens)
        session = self._workload_session
        if session is not None:
            if result[1] is not None:
                raise RuntimeError("MRv1 workload rejects speculative row expansion")
            with torch.inference_mode(False):
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
            with torch.inference_mode(False):
                session.before_execute(scheduler_output)
            result = super().execute_model(scheduler_output, intermediate_tensors)
            with torch.inference_mode(False), torch.enable_grad():
                session.complete_forward()
            return result
        except Exception:
            session.abort_active()
            raise

    def _sample(self, logits: torch.Tensor | None, spec_decode_metadata: SpecDecodeMetadata | None) -> SamplerOutput:
        session = self._workload_session
        if session is None:
            return super()._sample(logits, spec_decode_metadata)
        if not isinstance(self.model, Qwen2ForCausalLM):
            raise RuntimeError("attached MRv1 sampler lost its Qwen2 model")
        with torch.inference_mode(False), torch.enable_grad():
            active = session.begin_sampler(self.model)
        sampled = super()._sample(logits, spec_decode_metadata)
        with torch.inference_mode(False), torch.enable_grad():
            session.complete_sampler(active)
        return sampled

    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        session, state = self._workload_session, self.execute_model_state
        if session is None:
            return super().sample_tokens(grammar_output)
        try:
            output = super().sample_tokens(grammar_output)
            with torch.inference_mode(False), torch.enable_grad():
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
