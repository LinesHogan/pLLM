"""vLLM model runners that host one pllm workload session.

Runner installs hidden taps and attaches a VllmSession; WorkloadModelRunner is the MRv2 runner class the
worker instantiates. Used by runtime.py (construction) and pllm.worker.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Generic, Literal, Protocol, TypeVar

import torch
from vllm.config import CacheConfig, ModelConfig as VllmModelConfig, VllmConfig
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.sampler import Sampler as Mrv1Sampler
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler

from pllm.authoring import (
    AuxiliaryInput,
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
from pllm.backends.vllm.prefix_cache import POLICY_ENV, CachePolicy
from pllm.backends.vllm.sampler import (
    Mrv1Sampler as Mrv1Guidance,
    Vllm027Sampler,
)
from pllm.config import RunnerKind
from pllm.contracts.backend_attachment import (
    AttachmentGraphMode,
    AttachmentTopology,
    SignedBackendAttachment,
)
from pllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from pllm.contracts.packed_hidden import hidden_binding
from pllm.contracts.packed_rows import PackedSamplerRowBatch
from pllm.contracts.request_identity import RequestIdentity
from pllm.execution.codec import decode_boundary
from pllm.execution.host_runtime import CompletionCapture
from pllm.execution.position_state import PositionAnnouncement
from pllm.execution.state_execution import StateDiagnostics
from pllm.execution.trace import ExecutionTrace

if TYPE_CHECKING:
    pass


from pllm.backends.vllm.session import ReadonlyStepReceiver, VllmSession
from pllm.execution.state_feed import Feed

# Hidden sources the driver announces before construction, so the worker taps them at load time.
TAPS_ENV = "PLLM_HIDDEN_TAPS"


class _RunnerSession(Protocol):
    @property
    def trace(self) -> ExecutionTrace: ...
    @property
    def next_step_id(self) -> int: ...
    @property
    def apply_count(self) -> int: ...
    @property
    def updated_apply_count(self) -> int: ...
    def stage_input(
        self, batch: AuxiliaryInput[torch.Tensor] | None, identities: tuple[RequestIdentity, ...]
    ) -> None: ...
    def bind_readonly_input(self, receiver: ReadonlyStepReceiver) -> None: ...

    def feed(self, name: str) -> Feed: ...
    def identity(self, request_id: str) -> RequestIdentity: ...
    def before_execute(self, scheduler_output: SchedulerOutput) -> None: ...
    def prepare_input(self, rows: PackedSamplerRowBatch) -> None: ...
    def complete_forward(self) -> None: ...
    def begin_sampler(self, model: Qwen2ForCausalLM) -> bool: ...
    def complete_sampler(self, active: bool) -> None: ...
    def commit_token(self) -> None: ...
    def abort_active(self) -> None: ...
    def retire_all(self) -> None: ...
    def close(self) -> None: ...
    def require_idle(self) -> None: ...
    def diagnostics(self) -> StateDiagnostics: ...
    @property
    def attachment(self) -> SignedBackendAttachment: ...
    @property
    def parameter_phases(self) -> tuple[HiddenPhase, ...]: ...


SamplerT = TypeVar("SamplerT", Sampler, Mrv1Sampler)
AdapterT = TypeVar("AdapterT", Vllm027Sampler, Mrv1Guidance)


class Runner(Generic[SamplerT, AdapterT]):
    """The original attachment slot and cleanup, shared by concrete runner adapters."""

    sampler: SamplerT | AdapterT
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
    cache_policy: CachePolicy

    def qwen2_model(self) -> Qwen2ForCausalLM:
        """Return the model supported by workload taps, probing and sampling."""
        if not isinstance(self.model, Qwen2ForCausalLM):
            raise ValueError("choose a Qwen2 model for vLLM workloads")
        return self.model

    def _install_hidden_taps(self) -> None:
        """Tap the announced hidden outputs before vLLM compiles and captures the model."""
        requested: list[tuple[int, str]] = json.loads(os.environ.get(TAPS_ENV, "[]"))
        if not requested:
            return
        model = self.qwen2_model()
        layers = len(model.model.layers)
        paths = (hidden_binding(layer, HiddenSite(site), layers).module_path for layer, site in requested)
        hidden_size = model.lm_head.weight.shape[1]
        self.hidden_taps = HiddenTaps(model, paths, self.max_num_tokens, hidden_size, self.device)

    overlay_graph_bytes: int = 0

    def _initialize_workload(self, topology: AttachmentTopology) -> None:
        policy = os.environ.get(POLICY_ENV, "isolate")
        self._generated_kv: Literal["reuse", "isolate"] = "reuse" if policy == "reuse" else "isolate"
        self.workload_topology = topology
        self._workload_session: _RunnerSession | None = None
        self._workload_upstream_sampler: SamplerT | AdapterT | None = None
        self.completion_capture: CompletionCapture | None = None

    @property
    def overlay_apply_count(self) -> int:
        session = self._workload_session
        return 0 if session is None else session.apply_count

    @property
    def overlay_updated_apply_count(self) -> int:
        session = self._workload_session
        return 0 if session is None else session.updated_apply_count

    def attach_workload(
        self,
        deployment: BackendDeployment[DataT, ResultT],
    ) -> None:
        if self.cache_config.enable_prefix_caching and (
            deployment.attachment.inputs.request.activation_bindings
            or deployment.attachment.inputs.request.overlay_bindings
        ):
            raise ValueError("disable prefix caching when using parameter overlays or activation patches")
        if deployment.attachment.inputs.probe.position_states != self.position_states:
            raise ValueError("PositionState schema changed; create a new engine with this workload")
        request = deployment.attachment.inputs.request
        in_model = (
            request.overlay_bindings
            or request.activation_bindings
            or any(task.replay is not None for task in deployment.attachment.inputs.plan.semantic_graph.tasks)
        )
        if (
            in_model
            and not self.model_config.enforce_eager
            and deployment.attachment.inputs.probe.graph_mode is not AttachmentGraphMode.CUDA_GRAPH
        ):
            raise ValueError("parameter overlays, activation patches and layer replay do not run under CUDA graphs yet")
        model = self.qwen2_model()
        upstream = self.sampler
        adapter = self._make_sampler() if deployment.attachment.inputs.request.guides_sampler else None
        policy = CachePolicy.for_attachment(deployment.attachment, self._generated_kv)
        session = VllmSession(
            deployment,
            model,
            adapter,
            self.max_num_reqs,
            self.hidden_taps,
            self.position_layers,
            self.cache_config.block_size,
            cache_policy=policy,
        )
        self._workload_upstream_sampler = upstream
        self._workload_session = session
        try:
            self._finish_model_binding()
        except Exception:
            session.close()
            self._workload_session = None
            raise
        if adapter is not None:
            self.sampler = adapter
            # Both runner ABIs call model.compute_logits; guidance shares that LM-head read.
            setattr(model, "compute_logits", partial(session.compute_logits, model))
        self.cache_policy = policy

    def _finish_model_binding(self) -> None:
        """Finish backend model preparation after the session has installed its effects."""

    def detach_workload(self) -> None:
        session = self._workload_session
        upstream = self._workload_upstream_sampler
        assert session is not None and upstream is not None
        session.require_idle()
        session.close()
        if self.completion_capture is not None:
            self.completion_capture.synchronize()
        if self.sampler is not upstream:
            self.sampler = upstream
            self.model.__dict__.pop("compute_logits", None)
        self._workload_session = None
        self._workload_upstream_sampler = None
        self.completion_capture = None

    def stage_workload_input(
        self,
        batch: AuxiliaryInput[torch.Tensor] | None,
        identities: tuple[RequestIdentity, ...],
        completion: torch.Tensor | None = None,
    ) -> None:
        session = self.require_session()
        capture = None if completion is None else CompletionCapture(identities, completion)
        session.stage_input(batch, identities)
        self.completion_capture = capture

    def retire_completed_requests(self) -> None:
        session = self.require_session()
        if self.completion_capture is not None:
            self.completion_capture.require_complete()
            self.completion_capture.synchronize()
        with torch.inference_mode(False):
            session.retire_all()
        self.completion_capture = None

    def worker_state(self) -> BackendWorkerState:
        session = self.require_session()
        return BackendWorkerState(os.getpid(), session.diagnostics(), self.cache_policy)

    @property
    def trace(self) -> ExecutionTrace:
        return self.require_session().trace

    def abort_workload_requests(self) -> None:
        session = self.require_session()
        with torch.inference_mode(False):
            session.abort_active()
            session.retire_all()
        if self.completion_capture is not None:
            self.completion_capture.synchronize()
        self.completion_capture = None

    def require_session(self) -> _RunnerSession:
        session = self._workload_session
        assert session is not None
        return session

    def _make_sampler(self) -> AdapterT:
        raise NotImplementedError


class WorkloadModelRunner(Runner[Sampler, Vllm027Sampler], OverlayGraphRunner):
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
        self._install_hidden_taps()
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

    def _make_sampler(self) -> Vllm027Sampler:
        return Vllm027Sampler(self.sampler)

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
                    plan_epoch=session.attachment.physical.plan_epoch,
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
        session = self._workload_session
        if session is None:
            return super().sample(hidden_states, input_batch, grammar_output)
        model = self.qwen2_model()
        try:
            active = session.begin_sampler(model)
            sampled = super().sample(hidden_states, input_batch, grammar_output)
            session.complete_sampler(active)
            if self.completion_capture is not None:
                self.completion_capture.write(
                    tuple(session.identity(request_id) for request_id in input_batch.req_ids),
                    hidden_states[input_batch.logits_indices],
                )
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
            with torch.inference_mode(False), torch.enable_grad():
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
    cache_policy: CachePolicy
