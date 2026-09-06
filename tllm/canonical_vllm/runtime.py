"""Workload-neutral public handle for canonical vLLM MRv2 attachments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Generic, Literal, Protocol

import torch

from tllm.canonical.state_execution import StateInvocation
from tllm.canonical.backend_program import BackendCompileFacts, BackendDeployment, BackendWorkload, DataT, ResultT, TensorStepInput
from tllm.canonical.cuda_stream import CudaStreamReadiness
from tllm.contracts.request_state import IndexedRequestIdentityBatch, IndexedRequestIdentityRow

from tllm.canonical.backend_attachment import (
    validate_backend_attachment_physical,
)
from tllm.canonical.ptts.page_bundle_planning import PageBundleProviderFacts
from tllm.canonical.runtime_facts import CanonicalRuntimeFacts
from tllm.canonical_vllm.dense_overlay import (
    VllmDenseAddAdapter,
    inspect_dense_add_targets,
)
from tllm.canonical_vllm.hidden_capture import Vllm027EagerPackedHiddenCapture
from tllm.canonical_vllm.packed_rows import (
    Mrv2PackedStepRows,
    Mrv2RequestIdBatch,
    Mrv2RequestIdBinding,
    Vllm027Mrv2PackedRowCodec,
)
from tllm.canonical_vllm.sampler import (
    CanonicalVllm027PostFilterSampler,
    SignedPostFilterGuidance,
)
from tllm.contracts.backend_attachment import (
    AttachmentGraphMode,
    AttachmentHiddenResource,
    AttachmentKind,
    AttachmentTopology,
    BackendPlacementProbe,
)
from tllm.contracts.hidden_locator import HiddenLayerKind, HiddenSite
from tllm.contracts.packed_hidden import PackedHiddenResourceBinding
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.semantic_graph import (
    AnchorId,
)
from tllm.contracts.state_representation import StateDeviceKind
from tllm.contracts.tensor_layout import TensorDType

from vllm import RequestOutput, SamplingParams, __version__ as _VLLM_VERSION
from vllm.config import VllmConfig
from vllm.entrypoints.llm import LLM
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine.core_client import InprocClient
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu_worker import Worker


_MP_ENV = "VLLM_ENABLE_V1_MULTIPROCESSING"
_V2_RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"
_WORKER_CLASS = "tllm.canonical_vllm.runtime.CanonicalGPUWorker"
_ADAPTER_ID = "canonical.vllm_mrv2@1"
_SUPPORTED_VLLM_VERSION = "0.27.1"
_BACKEND_ABI_FIELDS = (
    "vllm=0.27.1",
    "runner=vllm.v1.worker.gpu.model_runner.GPUModelRunner",
    "input=vllm.v1.worker.gpu.input_batch.InputBatch",
    "sampler=vllm.v1.worker.gpu.sample.sampler.Sampler",
    "mode=in_process/eager/tp1/pp1/dp1/non_spec",
    "rows=one_logit_per_request",
    "attachment=shared_plan_v3",
    "attachments=request_sampler_training|dense_parameter_overlay",
)
_REQUEST_CAPABILITIES = (
    "backend/vllm_mrv2@1",
    "producer/packed_hidden@1",
    "sampler/post_filter_exact@1",
)
_OVERLAY_CAPABILITIES = (
    "backend/vllm_mrv2@1",
    "operand_capture/mlp_down_projection_pair@1",
    "overlay/dense_add@1",
)


class CanonicalVllmDType(StrEnum):
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


@dataclass(frozen=True, slots=True)
class CanonicalVllmConfig:
    model: str
    dtype: CanonicalVllmDType
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    seed: int = 0
    trust_remote_code: bool = False
    enable_prefix_caching: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.model) is not str
            or not self.model
            or type(self.dtype) is not CanonicalVllmDType
            or type(self.max_model_len) is not int
            or self.max_model_len < 2
            or type(self.max_num_seqs) is not int
            or self.max_num_seqs < 1
            or type(self.max_num_batched_tokens) is not int
            or self.max_num_batched_tokens < self.max_num_seqs
            or type(self.gpu_memory_utilization) is not float
            or not 0.0 < self.gpu_memory_utilization < 1.0
            or type(self.seed) is not int
            or self.seed < 0
            or type(self.trust_remote_code) is not bool
            or type(self.enable_prefix_caching) is not bool
        ):
            raise ValueError("canonical vLLM configuration is not closed")


def _digest(fields: tuple[str, ...]) -> str:
    return sha256("\x00".join(fields).encode("utf-8")).hexdigest()


def _tensor_dtype(dtype: torch.dtype) -> TensorDType:
    match dtype:
        case torch.float16:
            return TensorDType.FLOAT16
        case torch.bfloat16:
            return TensorDType.BFLOAT16
        case torch.float32:
            return TensorDType.FLOAT32
        case _:
            raise RuntimeError("canonical vLLM runner dtype is unsupported")


def _model_dtype(dtype: CanonicalVllmDType) -> Literal["bfloat16", "float16", "float32"]:
    match dtype:
        case CanonicalVllmDType.BFLOAT16:
            return "bfloat16"
        case CanonicalVllmDType.FLOAT16:
            return "float16"
        case CanonicalVllmDType.FLOAT32:
            return "float32"


class _RunnerSession(Protocol):
    @property
    def sampler(self) -> CanonicalVllm027PostFilterSampler: ...
    @property
    def apply_count(self) -> int: ...
    @property
    def updated_apply_count(self) -> int: ...
    def before_execute(self, scheduler_output: SchedulerOutput) -> None: ...
    def prepare_input(self, input_batch: InputBatch) -> None: ...
    def complete_forward(self) -> None: ...
    def begin_sampler(self, model: Qwen2ForCausalLM) -> bool: ...
    def complete_sampler(self, active: bool) -> None: ...
    def commit_token(self) -> None: ...
    def abort_active(self) -> None: ...
    def retire_all(self) -> None: ...
    def close(self) -> None: ...


@dataclass(slots=True)
class _PackedStep(Generic[DataT]):
    rows: Mrv2PackedStepRows
    invocation: StateInvocation[DataT] | None = None


class CanonicalVllmSession(Generic[DataT, ResultT]):
    """Model primitives over one original signed tensor-state invocation."""

    def __init__(self, deployment: BackendDeployment[DataT, ResultT],
                 model: Qwen2ForCausalLM, upstream_sampler: Sampler) -> None:
        attachment = deployment.attachment
        validate_backend_attachment_physical(attachment.inputs, attachment.decision, attachment.physical)
        self._deployment = deployment
        self._row_codec = Vllm027Mrv2PackedRowCodec(plan_epoch=attachment.physical.plan_epoch)
        probe = attachment.inputs.probe
        resources = tuple(resource.binding for resource in probe.hidden_resources)
        self._capture = Vllm027EagerPackedHiddenCapture(model=model, resources=resources,
            plan_epoch=attachment.physical.plan_epoch, max_packed_rows=probe.max_packed_rows,
            hidden_size=probe.hidden_size) if resources else None
        self._dense_adapter = VllmDenseAddAdapter(model,
            attachment.inputs.request.overlay_bindings) if attachment.inputs.request.overlay_bindings else None
        self._sampler = CanonicalVllm027PostFilterSampler(upstream_sampler, eager=True) if (
            attachment.inputs.request.attachment_kind is AttachmentKind.REQUEST_SAMPLER_TRAINING) else None
        self._identities: dict[str, RequestIdentity] = {}
        self._next_prompt_index = 0
        self._next_step_id = 0
        self._step: _PackedStep[DataT] | None = None

    @property
    def sampler(self) -> CanonicalVllm027PostFilterSampler:
        if self._sampler is None:
            raise RuntimeError("canonical session has no sampler intervention")
        return self._sampler

    @property
    def apply_count(self) -> int:
        return 0 if self._dense_adapter is None else self._dense_adapter.apply_count

    @property
    def updated_apply_count(self) -> int:
        return 0 if self._dense_adapter is None else self._dense_adapter.updated_apply_count

    def before_execute(self, scheduler_output: SchedulerOutput) -> None:
        if self._step is not None:
            raise RuntimeError("prior canonical vLLM step is still active")
        execution = self._deployment.execution
        execution.arena.collect_completed()
        finished = tuple(self._identities[request_id] for request_id in sorted(scheduler_output.finished_req_ids))
        if finished:
            execution.retire(finished)
            for identity in finished:
                self._identities.pop(identity.request_id)
        pending: list[RequestIdentity] = []
        for request in scheduler_output.scheduled_new_reqs:
            if request.req_id in self._identities:
                continue
            if type(request.req_id) is not str or not request.req_id:
                raise RuntimeError("vLLM scheduled an invalid request id")
            if type(request.sampling_params) is not SamplingParams or request.sampling_params.n != 1:
                raise RuntimeError("canonical vLLM requires sampling n=1")
            pending.append(RequestIdentity(request.req_id, self._next_prompt_index + len(pending), 0))
        if pending:
            execution.admit(tuple(pending), torch.cuda.current_stream(execution.device))
            for identity in pending:
                self._identities[identity.request_id] = identity
            self._next_prompt_index += len(pending)

    def prepare_input(self, input_batch: InputBatch) -> None:
        if self._step is not None:
            raise RuntimeError("canonical input preparation overlaps a step")
        requests = Mrv2RequestIdBatch(tuple(Mrv2RequestIdBinding(request_id, self._identities[request_id])
            for request_id in input_batch.req_ids))
        rows = self._row_codec.encode(input_batch, requests, step_id=self._next_step_id)
        self._next_step_id += 1
        step = _PackedStep[DataT](rows)
        self._step = step
        if self._capture is not None:
            self._capture.begin(rows.phase_rows)
        adapter = self._dense_adapter
        if adapter is None:
            return
        execution = self._deployment.execution
        identities = tuple(sorted((binding.identity for binding in requests.bindings),
            key=lambda identity: (identity.request_id, identity.prompt_idx, identity.sample_idx)))
        invocation = execution.begin(IndexedRequestIdentityBatch(tuple(
            IndexedRequestIdentityRow(index, identity) for index, identity in enumerate(identities))),
            torch.cuda.current_stream(execution.device))
        step.invocation = invocation
        adapter.prepare(rows.phase_rows, invocation.cursor.context)

    def complete_forward(self) -> None:
        step = self._step
        if step is None:
            return
        execution = self._deployment.execution
        stream = torch.cuda.current_stream(execution.device)
        hidden = None if self._capture is None else self._capture.publish()
        operands = () if self._dense_adapter is None else self._dense_adapter.take_captures()
        phases = {resource.phase for resource in self._deployment.attachment.inputs.probe.hidden_resources}
        if not operands and not any(row.phase in phases for row in step.rows.phase_rows.rows):
            if self._capture is not None:
                self._capture.finish()
            self._step = None
            return
        value = TensorStepInput(step.rows.phase_rows, step.rows.sampler_rows, hidden, operands)
        if step.invocation is None:
            step.invocation = execution.start_step(value, stream)
        else:
            execution.submit_input(step.invocation, value, CudaStreamReadiness(stream))
        for symbol in ("model.execute", "state.publish"):
            execution.execute_anchor(step.invocation, self._anchor(symbol))
        if self._capture is not None:
            self._capture.finish()
        if self._dense_adapter is not None:
            self._dense_adapter.clear()

    def begin_sampler(self, model: Qwen2ForCausalLM) -> bool:
        if self._step is None or self._sampler is None:
            return False
        execution = self._deployment.execution
        guidance = execution.sampler_guidance(self._invocation(), torch.cuda.current_stream(execution.device))
        logits = model.compute_logits(guidance.pred_hidden.to(dtype=model.lm_head.weight.dtype))
        if type(logits) is not torch.Tensor:
            raise RuntimeError("Qwen2 LM head did not produce dense logits")
        self._sampler.begin_guidance(SignedPostFilterGuidance(guidance, logits))
        return True

    def complete_sampler(self, active: bool) -> None:
        if self._step is None:
            if active:
                raise RuntimeError("sampler completion has no active packed step")
            return
        if self._sampler is not None:
            if not active:
                raise RuntimeError("canonical sampler skipped an active intervention")
            self._sampler.finish_guidance()
        execution = self._deployment.execution
        invocation = self._invocation()
        execution.execute_anchor(invocation, self._deployment.attachment.inputs.request.sampler_anchor_id)
        execution.complete_sampler(invocation, torch.cuda.current_stream(execution.device))

    def commit_token(self) -> None:
        if self._step is None:
            return
        execution = self._deployment.execution
        invocation = self._invocation()
        execution.execute_anchor(invocation, self._anchor("token.commit"))
        execution.finish_step(invocation)
        self._step = None
        execution.arena.collect_completed()

    def abort_active(self) -> None:
        if self._sampler is not None:
            self._sampler.abort_guidance()
        if self._capture is not None:
            self._capture.abort_if_active()
        if self._dense_adapter is not None:
            self._dense_adapter.clear()
        self._deployment.execution.cancel_active()
        self._step = None

    def retire_all(self) -> None:
        if self._step is not None:
            raise RuntimeError("canonical requests cannot retire during a step")
        identities = tuple(self._identities[request_id] for request_id in sorted(self._identities))
        if identities:
            self._deployment.execution.retire(identities)
        self._identities.clear()
        self._deployment.execution.arena.collect_completed()

    def close(self) -> None:
        self.abort_active()
        self.retire_all()
        if self._capture is not None:
            self._capture.close()
        if self._dense_adapter is not None:
            self._dense_adapter.close()

    def _invocation(self) -> StateInvocation[DataT]:
        if self._step is None or self._step.invocation is None:
            raise RuntimeError("model inputs have not started the signed invocation")
        return self._step.invocation

    def _anchor(self, symbol: str) -> AnchorId:
        return next(AnchorId(anchor.ordinal) for anchor in self._deployment.execution.arena.plan.timeline.anchors
            if anchor.symbol == symbol)

class CanonicalGPUModelRunner(GPUModelRunner):
    """MRv2 runner subclass with one generic canonical attachment slot."""

    __slots__ = (
        "_canonical_session",
        "_canonical_upstream_sampler",
    )

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, device)
        self._canonical_session: _RunnerSession | None = None
        self._canonical_upstream_sampler: Sampler | None = None

    @property
    def overlay_apply_count(self) -> int:
        session = self._canonical_session
        return 0 if session is None else session.apply_count

    @property
    def overlay_updated_apply_count(self) -> int:
        session = self._canonical_session
        return 0 if session is None else session.updated_apply_count

    def attach_canonical(
        self,
        workload: BackendWorkload[DataT, ResultT],
    ) -> BackendDeployment[DataT, ResultT]:
        if self._canonical_session is not None:
            raise RuntimeError("canonical runner already owns a workload session")
        match self.model:
            case Qwen2ForCausalLM() as model:
                pass
            case _:
                raise RuntimeError("canonical vLLM currently requires Qwen2")
        upstream = self.sampler
        if type(upstream) is not Sampler:
            raise RuntimeError("canonical vLLM requires the stock MRv2 sampler")
        deployment = _compile_vllm_deployment(workload, self)
        session = CanonicalVllmSession(deployment, model, upstream)
        self._canonical_upstream_sampler = upstream
        self._canonical_session = session
        if deployment.attachment.inputs.request.attachment_kind is AttachmentKind.REQUEST_SAMPLER_TRAINING:
            self.sampler = session.sampler
        return deployment

    def detach_canonical(self) -> None:
        session = self._canonical_session
        upstream = self._canonical_upstream_sampler
        if session is None or upstream is None:
            raise RuntimeError("canonical runner has no attached session")
        session.close()
        if self.sampler is not upstream:
            self.sampler = upstream
        self._canonical_session = None
        self._canonical_upstream_sampler = None

    def retire_completed_canonical_requests(self) -> None:
        session = self._require_session()
        with torch.inference_mode(False):
            session.retire_all()

    def abort_canonical_requests(self) -> None:
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
            self.detach_canonical()
        super().shutdown()

    def _require_session(self) -> _RunnerSession:
        session = self._canonical_session
        if session is None:
            raise RuntimeError("canonical runner has no attached session")
        return session


class CanonicalGPUWorker(Worker):
    """Single-process GPU worker with a canonical runner before model load."""

    def init_device(self) -> None:
        super().init_device()
        if not self.use_v2_model_runner:
            raise RuntimeError("canonical vLLM requires the MRv2 model runner")
        self.model_runner = CanonicalGPUModelRunner(self.vllm_config, self.device)


@dataclass(slots=True)
class CanonicalVllmLlm(Generic[DataT, ResultT]):
    """Single public handle for signed canonical vLLM workloads."""

    _llm: LLM
    runner: CanonicalGPUModelRunner
    deployment: BackendDeployment[DataT, ResultT]
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            raise RuntimeError("canonical vLLM handle is already closed")
        self.runner.detach_canonical()
        self._llm.llm_engine.engine_core.shutdown()
        self._closed = True

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams,
        *,
        use_tqdm: bool = False,
    ) -> list[RequestOutput]:
        if self._closed:
            raise RuntimeError("canonical vLLM handle is closed")
        if (
            type(prompts) not in (str, list)
            or (type(prompts) is str and not prompts)
            or (
                type(prompts) is list
                and (
                    not prompts
                    or any(type(prompt) is not str or not prompt for prompt in prompts)
                )
            )
            or type(sampling_params) is not SamplingParams
            or sampling_params.n != 1
            or type(use_tqdm) is not bool
        ):
            raise ValueError("canonical vLLM generate requires non-empty text and n=1")
        try:
            outputs = self._llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)
        except Exception:
            self.runner.abort_canonical_requests()
            raise
        self.runner.retire_completed_canonical_requests()
        return outputs


def create_canonical_vllm_llm(
    workload: BackendWorkload[DataT, ResultT],
    config: CanonicalVllmConfig,
) -> CanonicalVllmLlm[DataT, ResultT]:
    if _VLLM_VERSION != _SUPPORTED_VLLM_VERSION:
        raise RuntimeError("canonical vLLM requires vLLM 0.27.1 exactly")
    mp_env = os.environ.get(_MP_ENV)
    v2_env = os.environ.get(_V2_RUNNER_ENV)
    if mp_env not in (None, "0") or v2_env not in (None, "1"):
        raise RuntimeError(
            "canonical vLLM conflicts with explicit MP or legacy-runner environment"
        )
    os.environ[_MP_ENV] = "0"
    os.environ[_V2_RUNNER_ENV] = "1"
    try:
        llm = LLM(
            model=config.model,
            dtype=_model_dtype(config.dtype),
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            distributed_executor_backend="uni",
            worker_cls=_WORKER_CLASS,
            enforce_eager=True,
            max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            gpu_memory_utilization=config.gpu_memory_utilization,
            seed=config.seed,
            trust_remote_code=config.trust_remote_code,
            enable_prefix_caching=config.enable_prefix_caching,
            enable_chunked_prefill=False,
        )
    finally:
        if mp_env is None:
            os.environ.pop(_MP_ENV)
        else:
            os.environ[_MP_ENV] = mp_env
        if v2_env is None:
            os.environ.pop(_V2_RUNNER_ENV)
        else:
            os.environ[_V2_RUNNER_ENV] = v2_env

    match llm.llm_engine.engine_core:
        case InprocClient() as client:
            pass
        case _:
            raise RuntimeError("canonical vLLM did not construct an in-process client")
    match client.engine_core.model_executor:
        case UniProcExecutor() as executor:
            pass
        case _:
            client.shutdown()
            raise RuntimeError("canonical vLLM did not construct a uni-process executor")
    match executor.driver_worker.worker:
        case CanonicalGPUWorker() as worker:
            pass
        case _:
            client.shutdown()
            raise RuntimeError("canonical vLLM did not construct its worker")
    match worker.model_runner:
        case CanonicalGPUModelRunner() as runner:
            pass
        case _:
            client.shutdown()
            raise RuntimeError("canonical vLLM did not construct its MRv2 runner")
    try:
        deployment = runner.attach_canonical(workload)
    except Exception:
        client.shutdown()
        raise
    return CanonicalVllmLlm(llm, runner, deployment)


def _compile_vllm_deployment(
    workload: BackendWorkload[DataT, ResultT],
    runner: GPUModelRunner,
) -> BackendDeployment[DataT, ResultT]:
    _require_runner(runner)
    match runner.model:
        case Qwen2ForCausalLM() as model:
            pass
        case _:
            raise RuntimeError("canonical vLLM currently requires Qwen2")
    if model.quant_config is not None:
        raise RuntimeError("canonical vLLM requires an unquantized model")
    requirements = workload.backend_requirements()
    layer_count = len(model.model.layers)
    if any(source.site is not HiddenSite.BLOCK_OUTPUT for source in requirements.hidden_inputs):
        raise ValueError("Qwen2 hidden capture requires block outputs")
    hidden_resources = tuple(sorted((AttachmentHiddenResource(_hidden_binding(source.layer, layer_count), source.phase)
        for source in requirements.hidden_inputs), key=lambda resource: resource.resource_key))
    targets = inspect_dense_add_targets(model, requirements.parameter_targets, runner.dtype)
    probe = _base_probe(runner,
        model_fingerprint=_digest((
            f"model={runner.model_config.model}", f"revision={runner.model_config.revision or ''}",
            "class=vllm.model_executor.models.qwen2.Qwen2ForCausalLM",
            f"hidden_size={model.config.hidden_size}", f"layers={layer_count}",
            *(f"hidden={resource.binding.module_path}" for resource in hidden_resources),
            *(f"target={target.target}:{target.input_size}:{target.output_size}" for target in targets),
        )),
        hidden_size=model.config.hidden_size, hidden_resources=hidden_resources,
        capabilities=tuple(sorted(set((*_REQUEST_CAPABILITIES, *_OVERLAY_CAPABILITIES)))))
    timeline = CanonicalRuntimeFacts(runner.device).timeline((
        "state/dense_page_cuda@1", "state/fenced_reclaim@1", "transition/full_root@1"))
    free_bytes, _total_bytes = torch.cuda.mem_get_info(runner.device)
    facts = BackendCompileFacts(probe, timeline, (PageBundleProviderFacts("canonical.page_bundle_cuda@1", 3,
        probe.device_kind, probe.device_ordinal),), runner.max_num_reqs, free_bytes,
        runner.model_config.seed, targets)
    deployment = workload.compile_backend(facts)
    if deployment.execution.device != runner.device:
        raise RuntimeError("signed state placement differs from the model device")
    return deployment


def _require_runner(runner: GPUModelRunner) -> None:
    if (
        _VLLM_VERSION != _SUPPORTED_VLLM_VERSION
        or not runner.model_config.enforce_eager
        or runner.parallel_config.tensor_parallel_size != 1
        or runner.parallel_config.pipeline_parallel_size != 1
        or runner.dp_size != 1
        or runner.dcp_size != 1
        or runner.num_speculative_steps != 0
        or not runner.is_first_pp_rank
        or not runner.is_last_pp_rank
        or runner.lora_config is not None
        or runner.device.type != "cuda"
        or runner.max_num_reqs < 1
        or type(runner.sampler) is not Sampler
    ):
        raise RuntimeError(
            "canonical vLLM requires eager in-process TP1/PP1/DP1 non-spec MRv2"
        )


def _base_probe(
    runner: GPUModelRunner,
    *,
    model_fingerprint: str,
    hidden_size: int,
    hidden_resources: tuple[AttachmentHiddenResource, ...],
    capabilities: tuple[str, ...],
) -> BackendPlacementProbe:
    device_ordinal = runner.device.index
    if type(device_ordinal) is not int or device_ordinal < 0:
        raise RuntimeError("canonical vLLM requires an ordinal CUDA device")
    return BackendPlacementProbe(
        adapter_id=_ADAPTER_ID,
        backend_abi_digest=_digest(_BACKEND_ABI_FIELDS),
        runner_fingerprint=_digest(
            (
                _ADAPTER_ID,
                f"device=cuda:{device_ordinal}",
                f"max_num_tokens={runner.max_num_tokens}",
                f"max_num_reqs={runner.max_num_reqs}",
                f"dtype={runner.dtype}",
                f"seed={runner.model_config.seed}",
            )
        ),
        model_fingerprint=model_fingerprint,
        topology=AttachmentTopology.IN_PROCESS,
        graph_mode=AttachmentGraphMode.EAGER,
        device_kind=StateDeviceKind.CUDA,
        device_ordinal=device_ordinal,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_dtype=_tensor_dtype(runner.dtype),
        hidden_size=hidden_size,
        max_packed_rows=runner.max_num_tokens,
        hidden_resources=hidden_resources,
        capabilities=capabilities,
    )




def _hidden_binding(layer: int, layer_count: int) -> PackedHiddenResourceBinding:
    module_index = layer if layer >= 0 else layer_count + layer
    if module_index < 0 or module_index >= layer_count:
        raise ValueError("hidden layer does not identify a Qwen2 block")
    return PackedHiddenResourceBinding(
        str(layer),
        HiddenLayerKind.INDEX,
        HiddenSite.BLOCK_OUTPUT,
        f"model.layers.{module_index}",
    )




__all__ = [
    "CanonicalGPUModelRunner",
    "CanonicalGPUWorker",
    "CanonicalVllmConfig",
    "CanonicalVllmDType",
    "CanonicalVllmLlm",
    "CanonicalVllmSession",
    "create_canonical_vllm_llm",
]
