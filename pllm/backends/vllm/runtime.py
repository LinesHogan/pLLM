"""Workload-neutral public handle and vLLM engine construction: probing, installation and the VllmLlm API.

Sessions live in session.py and runners in runner.py; this module builds the engine and exposes the handle.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256
from types import TracebackType
from typing import TYPE_CHECKING, Generic, Literal, TypeVar, overload

import torch
from vllm import RequestOutput, SamplingParams, __version__ as _VLLM_VERSION
from vllm.config import VllmConfig
from vllm.entrypoints.llm import LLM
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.v1.engine.core_client import InprocClient
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.sample.sampler import Sampler as Mrv1Sampler
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu_worker import Worker

from pllm.authoring import (
    AuxiliaryInput,
    BackendDeployment,
    BackendRequirements,
    BackendWorkload,
    DataT,
    ResultT,
    StateTensorUpdate,
    map_auxiliary,
)
from pllm.backends.vllm.overlay_graph import (
    graph_mode,
    graph_options,
)
from pllm.backends.vllm.parameter_overlay import (
    inspect_parameter_targets,
)
from pllm.config import ModelConfig, RunnerKind, VllmDType
from pllm.contracts.auxiliary import validate_auxiliary
from pllm.contracts.backend_attachment import (
    TOKEN_CAPTURE_CAPABILITY,
    AttachmentGraphMode,
    AttachmentHiddenResource,
    BackendPlacementProbe,
    CausalModelDomain,
    SignedBackendAttachment,
)
from pllm.contracts.feedback import (
    BarrierState,
    Completion,
    CompletionBatch,
    CompletionFeedback,
    FinishReason,
    RetryAction,
)
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import SchedulerTimeline, TaskId
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType, TensorFieldSpec
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.host_runtime import CompletionInput
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.shared_tensor import SharedTensor
from pllm.execution.state_execution import StateDiagnostics
from pllm.execution.state_feed import Feed
from pllm.execution.trace import TRACE_SCHEMA_VERSION, TraceSelection, TraceSnapshot

if TYPE_CHECKING:
    from pllm.backends.vllm.mrv1 import Mrv1Runner


from pllm.backends.vllm.position_state import POSITION_ENV
from pllm.backends.vllm.prefix_cache import POLICY_ENV, CachePolicy
from pllm.backends.vllm.runner import TAPS_ENV, BackendWorkerState, WorkloadModelRunner
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_hidden import hidden_binding
from pllm.execution.position_state import PositionAnnouncement

_MP_ENV = "VLLM_ENABLE_V1_MULTIPROCESSING"


_V2_RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"


_WORKER_CLASS = "pllm.worker.GPUWorker"


_SUPPORTED_VLLM_VERSION = "0.27.1"


_BACKEND_ABI_FIELDS = (
    "vllm=0.27.1",
    "mode=in_process|engine_core/tp1/pp1/dp1/non_spec",
    "rows=one_logit_per_request",
    "attachment=shared_plan_v8",
    "attachments=request_training|parameter_overlay|activation_patch",
    "parameter=head_add_fp32/input_reconstruction@1",
    "activation=phase_windows/exact_step_pins/source_readonly@2",
    "model=causal_indexed_layer@5",
    "tensor=fx_affine_scan/aten_projection@1",
)


_REQUEST_CAPABILITIES = (
    "producer/packed_hidden@1",
    "sampler/post_filter_exact@1",
    "sampler/pre_filter_add@1",
)


_OVERLAY_CAPABILITIES = (
    "operand_capture/mlp_down_projection_pair@1",
    "overlay/dense_add@1",
    "overlay/low_rank_add@1",
    "overlay/head_add_fp32@1",
    "operand_capture/projection_input@1",
)


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
            raise RuntimeError("vLLM workload runner dtype is unsupported")


def _model_dtype(dtype: VllmDType) -> Literal["bfloat16", "float16", "float32"]:
    match dtype:
        case VllmDType.BFLOAT16:
            return "bfloat16"
        case VllmDType.FLOAT16:
            return "float16"
        case VllmDType.FLOAT32:
            return "float32"


class BackendCommand(StrEnum):
    PROBE = "probe"
    INSTALL = "install"
    STAGE = "stage"
    FINISH = "finish"
    ABORT = "abort"
    DIAGNOSTICS = "diagnostics"
    TRACE = "trace"
    PROFILE = "profile_tasks"
    OBSERVE = "observe"
    DETACH = "detach"


@dataclass(frozen=True, slots=True)
class BackendProbe:
    """Worker probe policy, shared with installation so capacity and input reservations agree."""

    requirements: BackendRequirements
    publish_lag: int = 1


@dataclass(frozen=True, slots=True)
class BackendInstall:
    attachment: SignedBackendAttachment
    definition: str


@dataclass(frozen=True, slots=True)
class BackendStage:
    identities: tuple[RequestIdentity, ...]
    auxiliary: AuxiliaryInput[SharedTensor] | None
    completion: SharedTensor | None


FeedbackT = TypeVar("FeedbackT", bound=CompletionFeedback)


@dataclass(slots=True)
class VllmLlm(Generic[DataT, ResultT]):
    """Single public handle for signed vLLM workload workloads."""

    _llm: LLM | None
    _runner: WorkloadModelRunner | Mrv1Runner | None
    _deployment: BackendDeployment[DataT, ResultT] | None
    attachment: SignedBackendAttachment
    _facts: BackendCompileFacts
    _closed: bool = False
    _next_prompt_index: int = 0
    _pending_input: ExitStack | None = None

    def _require_llm(self) -> LLM:
        if self._closed or self._llm is None:
            raise RuntimeError("model handle is closed")
        return self._llm

    @property
    def hidden_size(self) -> int:
        return self._facts.probe.hidden_size

    @property
    def vocab_size(self) -> int:
        return self._facts.vocab_size

    @property
    def device(self) -> torch.device:
        return self._facts.device

    def __enter__(self) -> VllmLlm[DataT, ResultT]:
        if self._closed:
            raise RuntimeError("model handle is closed")
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        try:
            self.close()
        except Exception as cleanup:
            if exc is None:
                raise
            exc.add_note(f"model cleanup also failed: {cleanup}")

    def feed(self, name: str) -> Feed:
        """Return the producer handle for a declared externally fed state."""
        self._require_llm()
        if self._runner is None:
            raise ValueError("state feeds need multiprocessing=False; pass the writer to a spawned producer")
        return self.runner.require_session().feed(name)

    def diagnostics(self) -> StateDiagnostics:
        if self._closed:
            raise RuntimeError("model handle is closed")
        return self.worker_state().diagnostics

    @property
    def runner(self) -> WorkloadModelRunner | Mrv1Runner:
        if self._closed:
            raise RuntimeError("model handle is closed")
        if self._runner is None:
            raise RuntimeError("the model runner belongs to the EngineCore child")
        return self._runner

    @property
    def deployment(self) -> BackendDeployment[DataT, ResultT]:
        if self._closed:
            raise RuntimeError("model handle is closed")
        if self._deployment is None:
            raise RuntimeError("the execution and arena belong to the EngineCore child")
        return self._deployment

    def command(self, command: BackendCommand, payload: str = "", *, timeout: float = 30.0) -> str:
        replies = self._require_llm().collective_rpc("workload_command", timeout=timeout, args=(command.value, payload))
        if len(replies) != 1:
            self._shutdown()
            raise RuntimeError("canonical worker quorum differs from the signed single-worker topology")
        return replies[0]

    def worker_state(self) -> BackendWorkerState:
        return decode_boundary(self.command(BackendCommand.DIAGNOSTICS), BackendWorkerState)

    def trace_snapshot(self) -> TraceSnapshot:
        if self._closed:
            raise RuntimeError("vLLM workload handle is closed")
        snapshot = (
            self._runner.trace.snapshot()
            if self._runner is not None
            else decode_boundary(self.command(BackendCommand.TRACE), TraceSnapshot)
        )
        if snapshot.schema_version != TRACE_SCHEMA_VERSION:
            raise RuntimeError("worker trace differs from the original schema")
        return snapshot

    def profile_tasks(self, tasks: tuple[TaskId, ...]) -> None:
        if self._closed:
            raise RuntimeError("vLLM workload handle is closed")
        if self._runner is not None:
            self._runner.trace.profile_tasks(tasks)
        else:
            self.command(BackendCommand.PROFILE, encode_boundary(tasks))

    def observe(self, selection: TraceSelection) -> None:
        """Configure bounded explicit tensor/gradient snapshots and opt-in Kineto profiling."""
        if self._closed:
            raise RuntimeError("vLLM workload handle is closed")
        if self._runner is not None:
            self._runner.trace.configure(selection)
        else:
            self.command(BackendCommand.OBSERVE, encode_boundary(selection))

    def close(self) -> None:
        if self._llm is None and self._pending_input is None:
            return
        try:
            if not self._closed:
                self.command(BackendCommand.DETACH)
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        # Closed to new work immediately; retain ownership if shutdown needs a retry.
        self._closed = True
        if self._llm is not None:
            self._llm.llm_engine.engine_core.shutdown()
        self._llm = self._runner = self._deployment = None
        if self._pending_input is not None:
            self._pending_input.close()
            self._pending_input = None

    def cache_policy(self) -> CachePolicy:
        """Read the installed identity, including idle workload replacements."""
        return self.worker_state().cache_policy

    def generate_feedback(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
        *,
        feedback: Callable[[CompletionInput], FeedbackT | None],
        feedback_group: str,
        max_rounds: int,
        prompt_indices: tuple[int, ...] | None = None,
        use_tqdm: bool = False,
        completion_device: torch.device | None = None,
        edit: Callable[[FeedbackT], StateTensorUpdate[torch.Tensor] | None] | None = None,
        routes: tuple[str, ...] | None = None,
    ) -> Iterator[tuple[CompletionInput, FeedbackT]]:
        """Generate, settle feedback, then retry eligible original prompts within a finite bound.

        The callable runs locally after generate has released backend requests; it is
        never installed on a worker. Each yield is the original completion and result,
        with no retained history. Multi-request barriers are regenerated as a full
        vote group. Closing the iterator leaves no pending generation or transaction.
        Optional edits reuse StateTensorUpdate and require a signed activation workload;
        the feedback adapter returns original tensors, never a worker callback.
        """
        if type(max_rounds) is not int or max_rounds < 1 or type(prompts) is not list or not prompts:
            raise ValueError("feedback generation requires a nonempty prompt list and positive finite round bound")
        texts: list[str] = list(prompts)
        indices: tuple[int, ...] = (
            tuple(range(self._next_prompt_index, self._next_prompt_index + len(texts)))
            if prompt_indices is None
            else prompt_indices
        )
        route_values: list[str] | None = list(routes) if routes is not None else None
        auxiliary: AuxiliaryInput[torch.Tensor] | None = None
        to_worker: Callable[[torch.Tensor], torch.Tensor] = lambda tensor: tensor.to(self.runner.device)
        for attempt in range(max_rounds):
            batch: CompletionInput = self.generate(
                texts,
                sampling_params,
                use_tqdm=use_tqdm,
                feedback_group=feedback_group,
                prompt_indices=indices,
                sample_index=attempt,
                auxiliary=auxiliary,
                completion_device=completion_device,
                routes=None if route_values is None else tuple(route_values),
            )
            result = feedback(batch)
            if result is None:
                raise RuntimeError("completion feedback returned no settled control result")
            entries, controls = batch.completions.entries, result.retry_controls
            if len(controls) != len(entries) or any(
                control.completion is not entry for control, entry in zip(controls, entries, strict=True)
            ):
                raise ValueError("feedback controls must reference the original ordered completions")
            by_request = {entry.identity.request_id: entry for entry in entries}
            covered: set[str] = set()
            ready: set[str] = set()
            for barrier in result.update_barriers:
                members = set(barrier.request_ids)
                if (
                    not members
                    or len(members) != len(barrier.request_ids)
                    or covered.intersection(members)
                    or not members.issubset(by_request)
                    or not 0 <= barrier.ready_count <= len(members)
                    or any(by_request[request].group_id != barrier.group_id for request in members)
                ):
                    raise ValueError("feedback barriers must partition original completion identities")
                covered.update(members)
                if barrier.state is BarrierState.READY and barrier.ready_count > 0 and result.update_count > 0:
                    ready.update(members)
            if covered != by_request.keys():
                raise ValueError("feedback barriers omitted a completion")
            retry = {
                control.completion.identity.request_id
                for control in controls
                if control.action is RetryAction.RETRY
                and (not control.retry_after_update or control.completion.identity.request_id in ready)
            }
            stopped = {
                control.completion.identity.request_id for control in controls if control.action is RetryAction.STOP
            }
            for barrier in result.update_barriers:
                members = set(barrier.request_ids)
                if members.intersection(retry):
                    if members.intersection(stopped):
                        retry.difference_update(members)
                    else:
                        retry.update(members)
            selected: tuple[int, ...] = tuple(
                index for index, entry in enumerate(entries) if entry.identity.request_id in retry
            )
            auxiliary = edit(result) if edit is not None and selected and attempt + 1 < max_rounds else None
            if auxiliary is not None and self._runner is not None:
                validate_auxiliary(auxiliary, self.attachment.inputs.plan, SharedTensor.describe)
                auxiliary = map_auxiliary(auxiliary, to_worker)
            yield batch, result
            if not selected:
                return
            texts = [texts[index] for index in selected]
            indices = tuple(indices[index] for index in selected)
            if route_values is not None:
                route_values = [route_values[index] for index in selected]

    @overload
    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams,
        *,
        use_tqdm: bool = False,
        auxiliary: AuxiliaryInput[torch.Tensor] | None = None,
        sample_index: int = 0,
        feedback_group: None = None,
        prompt_indices: tuple[int, ...] | None = None,
        completion_device: torch.device | None = None,
        routes: tuple[str, ...] | None = None,
    ) -> list[RequestOutput]: ...

    @overload
    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams,
        *,
        use_tqdm: bool = False,
        auxiliary: AuxiliaryInput[torch.Tensor] | None = None,
        sample_index: int = 0,
        feedback_group: str,
        prompt_indices: tuple[int, ...] | None = None,
        completion_device: torch.device | None = None,
        routes: tuple[str, ...] | None = None,
    ) -> CompletionInput: ...

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams,
        *,
        use_tqdm: bool = False,
        auxiliary: AuxiliaryInput[torch.Tensor] | None = None,
        sample_index: int = 0,
        feedback_group: str | None = None,
        prompt_indices: tuple[int, ...] | None = None,
        completion_device: torch.device | None = None,
        routes: tuple[str, ...] | None = None,
    ) -> list[RequestOutput] | CompletionInput:
        if self._closed:
            raise RuntimeError("vLLM workload handle is closed")
        if completion_device is not None and feedback_group is None:
            raise ValueError("completion device requires completion feedback")
        output_device = completion_device or (torch.device("cpu") if self._runner is None else self.runner.device)
        if output_device.type == "cuda":
            if output_device == torch.device("cuda"):
                output_device = torch.device("cuda", self.attachment.inputs.probe.device_ordinal)
            if output_device.index != self.attachment.inputs.probe.device_ordinal:
                raise ValueError("completion IPC currently requires the model's CUDA device")
        elif output_device.type != "cpu":
            raise ValueError("completion output requires CPU or CUDA storage")
        if (
            type(prompts) not in (str, list)
            or (type(prompts) is str and not prompts)
            or (
                type(prompts) is list
                and (not prompts or any(type(prompt) is not str or not prompt for prompt in prompts))
            )
            or type(sampling_params) is not SamplingParams
            or sampling_params.n != 1
            or type(use_tqdm) is not bool
            or type(sample_index) is not int
            or sample_index < 0
            or feedback_group is not None
            and (type(feedback_group) is not str or not feedback_group)
        ):
            raise ValueError("vLLM workload generate requires non-empty text and n=1")
        texts = [prompts] if isinstance(prompts, str) else prompts
        indices = (
            tuple(range(self._next_prompt_index, self._next_prompt_index + len(texts)))
            if prompt_indices is None
            else prompt_indices
        )
        if len(indices) != len(texts) or any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("prompt indices must align with the original input prompts")
        # A route value is user-computed (regex, discriminator, tenant); the signed
        # graph only declares whether any request state keys on it.
        signed_route = any(
            RequestKeyField.ROUTE in state.key_fields for state in self.attachment.inputs.plan.semantic_graph.states
        )
        if routes is not None and (
            not signed_route
            or len(routes) != len(texts)
            or any(type(route) is not str or not route for route in routes)
        ):
            raise ValueError("routes require a signed route-keyed state and one value per prompt")
        if signed_route and routes is None:
            raise ValueError("a route-keyed request state requires per-prompt routes")
        groups: dict[int, str] = {}
        for index, prompt in zip(indices, texts, strict=True):
            if groups.setdefault(index, prompt) != prompt:
                raise ValueError("one prompt identity cannot name different input texts")
        validate_auxiliary(auxiliary, self.attachment.inputs.plan, SharedTensor.describe)
        llm = self._require_llm()
        llm.sleep(level=0, mode="keep")
        request_ids: list[str] = []
        lifetime = ExitStack()
        try:
            request_ids = llm.enqueue(prompts, self.cache_policy().sampling(sampling_params), use_tqdm=use_tqdm)
            identities = tuple(
                RequestIdentity(request_id, index, sample_index, route)
                for request_id, index, route in zip(
                    request_ids, indices, routes if routes is not None else ("",) * len(texts), strict=True
                )
            )
            # Borrow vLLM's exact frontend mapping before completed entries are removed.
            external = {
                llm.llm_engine.output_processor.request_states[identity.request_id].external_req_id: identity
                for identity in identities
            }
            destination = None
            completion_wire = None
            if feedback_group is not None:
                field = TensorFieldSpec(
                    LayoutFieldId(0),
                    (len(identities), self.attachment.inputs.probe.hidden_size),
                    TensorDType.FLOAT32,
                    4,
                )
                with torch.inference_mode(False):
                    if self._runner is None:
                        destination, completion_wire = SharedTensor.allocate(field, lifetime, output_device)
                    else:
                        destination = torch.empty(field.shape, device=output_device, dtype=torch.float32)
            if self._runner is None:
                auxiliary_wire = map_auxiliary(auxiliary, lambda tensor: SharedTensor.export(tensor, lifetime))
                self.command(
                    BackendCommand.STAGE, encode_boundary(BackendStage(identities, auxiliary_wire, completion_wire))
                )
            else:
                self.runner.stage_workload_input(auxiliary, identities, destination)
            self._next_prompt_index = max(self._next_prompt_index, max(indices) + 1)
            llm.wake_up(tags=["scheduling"])
            outputs = llm.wait_for_completion(RequestOutput, use_tqdm=use_tqdm)
            result: list[RequestOutput] | CompletionInput = outputs
            batch = None
            if feedback_group is not None:
                completions: dict[RequestIdentity, Completion] = {}
                for output in outputs:
                    if not output.finished or len(output.outputs) != 1:
                        raise RuntimeError("completion feedback requires final n=1 outputs")
                    identity = external[output.request_id]
                    (completion,) = output.outputs
                    if identity in completions:
                        raise RuntimeError("duplicate completion for a submitted request")
                    completions[identity] = Completion(
                        identity,
                        completion.index,
                        feedback_group,
                        tuple(completion.token_ids),
                        completion.text,
                        FinishReason(completion.finish_reason),
                    )
                if len(completions) != len(identities):
                    raise RuntimeError("completion feedback did not finish every submitted request")
                batch = CompletionBatch(tuple(completions[identity] for identity in identities))
                if self._runner is not None:
                    capture = self.runner.completion_capture
                    if capture is None:
                        raise RuntimeError("completion capture was not started")
                    result = capture.finish(batch)
            self.command(BackendCommand.FINISH)
            if batch is not None and self._runner is None:
                if destination is None:
                    raise RuntimeError("completion output was not allocated")
                result = CompletionInput(
                    batch,
                    destination,
                    CudaStreamReadiness(torch.cuda.current_stream(destination.device))
                    if destination.device.type == "cuda"
                    else None,
                )
            return result
        except Exception as original:
            try:
                llm.llm_engine.abort_request(request_ids, internal=True)
                self.command(BackendCommand.ABORT)
                llm.wake_up(tags=["scheduling"])
            except Exception as cleanup:
                # Borrowed input storage cannot outlive an unacknowledged worker.
                original.add_note(f"worker abort also failed: {cleanup}")
                try:
                    self._shutdown()
                except Exception as shutdown:
                    original.add_note(f"worker shutdown also failed; close() can retry: {shutdown}")
            raise
        finally:
            if self._closed and self._llm is not None:
                self._pending_input = lifetime
            else:
                lifetime.close()


def create_vllm_llm(
    workload: BackendWorkload[DataT, ResultT],
    config: ModelConfig,
    *,
    definition: str | None = None,
) -> VllmLlm[DataT, ResultT]:
    requirements = workload.backend_requirements()
    offload = config.kv_offload
    if requirements.position_states and config.runner is RunnerKind.MRV1:
        raise ValueError("PositionState layer routes require runner=MRV2")
    if config.enable_prefix_caching and (requirements.parameter_targets or requirements.activation_layers):
        raise ValueError("parameter-version overlays do not support shared prefix caching")
    if _VLLM_VERSION != _SUPPORTED_VLLM_VERSION:
        raise RuntimeError("vLLM workload requires vLLM 0.27.1 exactly")
    if config.cuda_graphs and requirements.activation_layers:
        raise ValueError("parameter overlays and activation patches do not run under CUDA graphs yet")
    environment = {
        _MP_ENV: "1" if config.multiprocessing else "0",
        _V2_RUNNER_ENV: "1" if config.runner is RunnerKind.MRV2 else "0",
        TAPS_ENV: json.dumps([[source.layer, source.site.value] for source in requirements.hidden_inputs]),
        POLICY_ENV: config.generated_kv,
        POSITION_ENV: encode_boundary(PositionAnnouncement(requirements.position_states)),
    }
    if config.multiprocessing:
        environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    previous_environment = {name: os.environ.get(name) for name in environment}
    if config.multiprocessing and definition is None:
        raise ValueError("EngineCore requires a closed mathematical definition from the public factory")
    if any(previous_environment[name] not in (None, value) for name, value in environment.items()):
        raise RuntimeError("vLLM workload conflicts with explicit MP, spawn or legacy-runner environment")
    os.environ.update(environment)
    try:
        llm = LLM(
            **graph_options(config, requirements),
            model=config.model,
            revision=config.revision,
            dtype=_model_dtype(config.dtype),
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            distributed_executor_backend="uni",
            worker_cls=_WORKER_CLASS,
            enforce_eager=not config.cuda_graphs,
            max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            gpu_memory_utilization=config.gpu_memory_utilization,
            seed=config.seed,
            trust_remote_code=config.trust_remote_code,
            enable_prefix_caching=config.enable_prefix_caching,
            enable_chunked_prefill=config.enable_chunked_prefill,
            # The supported uni-process, non-speculative decoder enables async by default.
            async_scheduling=config.runner is RunnerKind.MRV2,
            scheduler_cls="pllm.backends.vllm.prefix_cache."
            + ("WorkloadAsyncScheduler" if config.runner is RunnerKind.MRV2 else "WorkloadScheduler"),
            kv_transfer_config=None if offload is None else offload.options(),
        )
    finally:
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name)
            else:
                os.environ[name] = value

    with ExitStack() as construction:
        construction.callback(llm.llm_engine.engine_core.shutdown)
        if config.multiprocessing:
            replies = llm.collective_rpc(
                "workload_command",
                timeout=30.0,
                args=(
                    BackendCommand.PROBE.value,
                    encode_boundary(BackendProbe(requirements, config.execution.publish_lag)),
                ),
            )
            if len(replies) != 1:
                raise RuntimeError("canonical probe requires the exact single-worker quorum")
            facts = workload.specialize(decode_boundary(replies[0], BackendCompileFacts), config.execution)
            attachment = workload.plan_backend(facts)
            handle = VllmLlm[DataT, ResultT](llm, None, None, attachment, facts)
            installed = decode_boundary(
                handle.command(
                    BackendCommand.INSTALL,
                    encode_boundary(BackendInstall(attachment, definition or "")),
                ),
                BackendWorkerState,
            )
            if installed.process_id == os.getpid():
                raise RuntimeError("EngineCore did not install the original signed compilation in its child")
        else:
            match llm.llm_engine.engine_core:
                case InprocClient() as client:
                    pass
                case _:
                    raise RuntimeError("vLLM workload did not construct an in-process client")
            match client.engine_core.model_executor:
                case UniProcExecutor() as executor:
                    pass
                case _:
                    raise RuntimeError("vLLM workload did not construct a uni-process executor")
            match executor.driver_worker.worker:
                case Worker() as worker:
                    pass
                case _:
                    raise RuntimeError("vLLM workload did not construct its worker")
            from pllm.backends.vllm.mrv1 import Mrv1Runner

            match worker.model_runner:
                case WorkloadModelRunner() | Mrv1Runner() as runner:
                    pass
                case _:
                    raise RuntimeError("vLLM workload did not construct its selected runner")
            facts = workload.specialize(
                probe_backend(requirements, runner, publish_lag=config.execution.publish_lag),
                config.execution,
            )
            deployment = workload.compile_backend(facts)
            runner.attach_workload(deployment)
            handle = VllmLlm(llm, runner, deployment, deployment.attachment, facts)
        construction.pop_all()
        return handle


def probe_backend(
    requirements: BackendRequirements,
    runner: WorkloadModelRunner | Mrv1Runner,
    timeline: SchedulerTimeline | None = None,
    *,
    publish_lag: int = 1,
) -> BackendCompileFacts:
    _require_runner(runner)
    if requirements.position_states != runner.position_states:
        raise ValueError("PositionState schema differs from pre-allocation announcement; create a new engine")
    match runner.model:
        case Qwen2ForCausalLM() as model:
            pass
        case _:
            raise RuntimeError("vLLM workload currently requires Qwen2")
    if model.quant_config is not None:
        raise RuntimeError("vLLM workload requires an unquantized model")
    if requirements.model_source is not None and requirements.model_source != (
        runner.model_config.model,
        runner.model_config.revision or "main",
    ):
        raise ValueError("causal training and inference must use the same model source and revision")
    layer_count = len(model.model.layers)
    if any(layer < 0 or layer >= layer_count for layer in requirements.activation_layers):
        raise ValueError("activation layer is outside the backend model")
    hidden_resources = tuple(
        sorted(
            (
                AttachmentHiddenResource(hidden_binding(source.layer, source.site, layer_count), source.phase)
                for source in requirements.hidden_inputs
            ),
            key=lambda resource: (*resource.resource_key, resource.phase is HiddenPhase.DECODE),
        )
    )
    targets = inspect_parameter_targets(model, requirements.parameter_targets, runner.dtype)
    replay_capabilities = (
        ("model/tokenwise_replay@1", TOKEN_CAPTURE_CAPABILITY)
        if runner.runner_kind is RunnerKind.MRV2 and graph_mode(runner.overlay_graph_bytes) is AttachmentGraphMode.EAGER
        else ()
    )
    probe = _base_probe(
        runner,
        model_fingerprint=_digest(
            (
                f"model={runner.model_config.model}",
                f"revision={runner.model_config.revision or ''}",
                "class=vllm.model_executor.models.qwen2.Qwen2ForCausalLM",
                f"hidden_size={model.config.hidden_size}",
                f"layers={layer_count}",
                f"vocab_size={model.config.vocab_size}",
                *(f"hidden={resource.binding.module_path}" for resource in hidden_resources),
                *(f"activation_layer={layer}" for layer in requirements.activation_layers),
                *(f"target={target.target}:{target.input_size}:{target.output_size}" for target in targets),
            )
        ),
        hidden_size=model.config.hidden_size,
        layer_count=layer_count,
        hidden_resources=hidden_resources,
        capabilities=tuple(
            sorted(
                set(
                    (
                        *_REQUEST_CAPABILITIES,
                        *_OVERLAY_CAPABILITIES,
                        *replay_capabilities,
                        f"backend/vllm_{runner.runner_kind.value}@1",
                    )
                )
            )
        ),
    )
    observed = RuntimeFacts(runner.device).timeline(
        (
            "state/dense_page_cuda@1",
            "state/fenced_reclaim@1",
            "transition/full_root@1",
            "optimizer/adamw@1",
            *replay_capabilities,
            *(
                ("compute/cuda_graph@1",)
                if timeline is not None and "compute/cuda_graph@1" in timeline.capabilities
                else ()
            ),
        ),
    )
    required = input_reservations(
        probe,
        observed,
        runner.max_num_reqs,
        model.lm_head.weight.shape[0],
        requirements.sampler_stage,
        requirements.sampler_feedback,
    )
    required = tuple(value + requirements.resident_bytes + runner.overlay_graph_bytes for value in required)
    observed = replace(observed, reserved_bytes=required)
    if timeline is None:
        timeline = observed
    elif (
        timeline.anchors != observed.anchors
        or timeline.capabilities != observed.capabilities
        or timeline.schema_version != observed.schema_version
    ):
        raise ValueError("worker timeline differs from the supported single-device anchors and capabilities")
    elif len(timeline.reserved_bytes) != len(required) or any(
        actual < minimum for actual, minimum in zip(timeline.reserved_bytes, required, strict=True)
    ):
        raise ValueError("worker timeline underfunds the backend input/sampler buffers")
    free_bytes, _total_bytes = torch.cuda.mem_get_info(runner.device)
    # Per scheduled request: the current root, the private root being written, and the root the GPU
    # may still read from the step before, because the host prepares a step while the GPU runs the last.
    # Each lagged step additionally retains its exact input root for readers. No host wait
    # is hidden in this capacity: current + private + GPU reader + publish_lag retained roots.
    capacity = (3 + publish_lag) * runner.max_num_reqs + 2
    facts = BackendCompileFacts(
        replace(probe, targets=targets),
        timeline,
        (
            PageBundleProviderFacts(
                "canonical.page_bundle_cuda@1",
                capacity,
                probe.device_kind,
                probe.device_ordinal,
            ),
        ),
        runner.max_num_reqs,
        free_bytes,
        runner.model_config.seed,
        model.config.vocab_size,
        layer_count,
    )
    return facts


def _require_runner(runner: WorkloadModelRunner | Mrv1Runner) -> None:
    if (
        _VLLM_VERSION != _SUPPORTED_VLLM_VERSION
        or runner.parallel_config.tensor_parallel_size != 1
        or runner.parallel_config.pipeline_parallel_size != 1
        or runner.parallel_config.data_parallel_size != 1
        or runner.parallel_config.decode_context_parallel_size != 1
        or runner.speculative_config is not None
        or runner.lora_config is not None
        or runner.device.type != "cuda"
        or runner.max_num_reqs < 1
        or type(runner.sampler) is not (Sampler if runner.runner_kind is RunnerKind.MRV2 else Mrv1Sampler)
    ):
        raise RuntimeError("vLLM workload requires TP1/PP1/DP1 non-spec execution")


def _base_probe(
    runner: WorkloadModelRunner | Mrv1Runner,
    *,
    model_fingerprint: str,
    hidden_size: int,
    layer_count: int,
    hidden_resources: tuple[AttachmentHiddenResource, ...],
    capabilities: tuple[str, ...],
) -> BackendPlacementProbe:
    device_ordinal = runner.device.index
    if type(device_ordinal) is not int or device_ordinal < 0:
        raise RuntimeError("vLLM workload requires an ordinal CUDA device")
    return BackendPlacementProbe(
        adapter_id=f"canonical.vllm_{runner.runner_kind.value}@1",
        backend_abi_digest=_digest(
            (
                *_BACKEND_ABI_FIELDS,
                f"runner={runner.runner_kind.value}",
                f"cuda_graphs={not runner.model_config.enforce_eager}",
            )
        ),
        runner_fingerprint=_digest(
            (
                runner.runner_kind.value,
                f"device=cuda:{device_ordinal}",
                f"max_num_tokens={runner.max_num_tokens}",
                f"max_num_reqs={runner.max_num_reqs}",
                f"dtype={runner.dtype}",
                f"seed={runner.model_config.seed}",
            )
        ),
        model_fingerprint=model_fingerprint,
        topology=runner.workload_topology,
        graph_mode=graph_mode(runner.overlay_graph_bytes),
        device_kind=StateDeviceKind.CUDA,
        device_ordinal=device_ordinal,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_dtype=_tensor_dtype(runner.dtype),
        hidden_size=hidden_size,
        max_packed_rows=runner.max_num_tokens,
        hidden_resources=hidden_resources,
        capabilities=capabilities,
        causal_domain=CausalModelDomain(runner.max_num_reqs, layer_count, runner.model_config.max_model_len),
        position_states=runner.position_states,
    )


__all__ = [
    "VllmConfig",
    "VllmDType",
    "VllmLlm",
    "create_vllm_llm",
]


def input_reservations(
    probe: BackendPlacementProbe,
    timeline: SchedulerTimeline,
    rows: int,
    vocab: int,
    stage: SamplerStage | None,
    feedback: bool,
) -> tuple[int, ...]:
    hidden = (
        2
        * len({resource.binding for resource in probe.hidden_resources})
        * probe.max_packed_rows
        * probe.hidden_size
        * 4
    )
    # FP32 envelope for original logits, conversion, projection and rowwise arithmetic.
    # Excludes upstream sampling/GEMM library scratch, model weights and KV storage.
    workspace = (
        rows * (36 * vocab + 4 * probe.hidden_size + 8)
        if stage is SamplerStage.PACKED_POST_FILTER_EXACT
        else rows * (12 * vocab + 8)
        if stage is SamplerStage.PACKED_PRE_FILTER_ADD
        else 0
    )
    retained = rows * (4 * vocab + 8) if feedback else 0
    # StepCapture retains its feedback pointers until the next table is prepared.
    # Budget both old and new private copies during handoff, including next forward.
    private_feedback = 2 * retained
    sampler = next(anchor.ordinal for anchor in timeline.anchors if anchor.symbol == "sampler.pre_draw")
    return tuple(
        hidden
        + private_feedback
        + (workspace if anchor.symbol in ("sampler.pre_draw", "sampler.complete") else 0)
        + (retained if anchor.ordinal >= sampler else 0)
        for anchor in timeline.anchors
    )
