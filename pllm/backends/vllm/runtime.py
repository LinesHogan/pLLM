"""Workload-neutral public handle and vLLM engine construction: probing, installation and the VllmLlm API.

Sessions live in session.py and runners in runner.py; this module builds the engine and exposes the handle.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import partial
from hashlib import sha256
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

import torch
from vllm import RequestOutput, SamplingParams
from vllm.entrypoints.llm import LLM

from pllm.authoring import (
    BackendRequirements,
    BackendWorkload,
    DataT,
    HostBatch,
    ResultT,
    StateImport,
    map_host_batch,
)
from pllm.backends.vllm import compat
from pllm.backends.vllm.hidden_capture import CAPTURE_ENV, HiddenCaptureAnnouncement, HiddenCaptureLayout
from pllm.backends.vllm.overlay_graph import (
    GraphOptions,
    graph_mode,
    graph_options,
)
from pllm.backends.vllm.parameter_overlay import (
    inspect_parameter_targets,
)
from pllm.backends.vllm.training import CausalTraining
from pllm.config import ModelConfig
from pllm.contracts.backend_attachment import (
    TOKEN_CAPTURE_CAPABILITY,
    AttachmentGraphMode,
    AttachmentHiddenResource,
    BackendAttachment,
    BackendPlacementProbe,
    CausalModelDomain,
)
from pllm.contracts.codec import decode_boundary, encode_boundary
from pllm.contracts.feedback import (
    BarrierState,
    Completion,
    CompletionBatch,
    CompletionFeedback,
    FinishReason,
    RetryAction,
)
from pllm.contracts.host_batch import validate_host_batch
from pllm.contracts.parameter_overlay import expand_targets
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import SchedulerTimeline, TaskId
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType, TensorFieldSpec
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.host_runtime import CompletionInput
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import model_timeline
from pllm.execution.shared_tensor import SharedTensor
from pllm.execution.state_execution import StateDiagnostics
from pllm.execution.state_feed import Feed
from pllm.execution.trace import TraceSelection, TraceSnapshot

if TYPE_CHECKING:
    from pllm.backends.vllm.mrv1 import Mrv1Runner


from pllm.backends.vllm.architecture import hidden_binding
from pllm.backends.vllm.position_state import POSITION_ENV
from pllm.backends.vllm.prefix_cache import CachePolicy
from pllm.backends.vllm.runner import BackendWorkerState, WorkloadModelRunner
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.execution.position_state import PositionAnnouncement

_logger = logging.getLogger(__name__)

_MP_ENV = "VLLM_ENABLE_V1_MULTIPROCESSING"


_V2_RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"


_WORKER_CLASS = "pllm.worker.GPUWorker"


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
    return {
        torch.float16: TensorDType.FLOAT16,
        torch.bfloat16: TensorDType.BFLOAT16,
        torch.float32: TensorDType.FLOAT32,
    }[dtype]


class BackendCommand(StrEnum):
    PROBE = "probe"
    INSTALL = "install"
    STAGE = "stage"
    FINISH = "finish"
    ABORT = "abort"
    DIAGNOSTICS = "diagnostics"
    TRACE = "trace"
    PROFILE = "profile_tasks"
    SELECT_TRACE = "select_trace"
    DETACH = "detach"


@dataclass(frozen=True, slots=True)
class BackendProbe:
    """What the driver asks the worker to probe; the publish lag sizes page capacity."""

    requirements: BackendRequirements
    publish_lag: int = 1


@dataclass(frozen=True, slots=True)
class BackendInstall:
    """The driver's specialized facts and plan; the worker binds them without probing again."""

    facts: BackendCompileFacts
    attachment: BackendAttachment
    definition: str
    cache_policy: CachePolicy


@dataclass(frozen=True, slots=True)
class BackendStage:
    identities: tuple[RequestIdentity, ...]
    batch: HostBatch[SharedTensor] | None
    completion: SharedTensor | None
    driver_pid: int = field(default_factory=os.getpid)


FeedbackT = TypeVar("FeedbackT", bound=CompletionFeedback)


@dataclass(slots=True)
class VllmLlm(Generic[DataT, ResultT]):
    """Single public handle for signed vLLM workload workloads."""

    _llm: LLM | None
    _runner: WorkloadModelRunner | Mrv1Runner | None
    attachment: BackendAttachment
    _facts: BackendCompileFacts
    _cache_policy: CachePolicy
    _sample_limit: int | None
    _eager_reason: str
    _closed: bool = False
    _next_prompt_index: int = 0
    _pending_input: ExitStack | None = None

    def _require_open(self) -> LLM:
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
        return self._runner.device if self._runner is not None else torch.device("cuda", 0)

    def __enter__(self) -> VllmLlm[DataT, ResultT]:
        self._require_open()
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
        self._require_open()
        if self._runner is None:
            raise ValueError(
                "state feeds need tensor_parallel_size=1 and multiprocessing=False; pass the writer to a spawned producer"
            )
        return self.runner.require_session().feed(name)

    def diagnostics(self) -> StateDiagnostics:
        self._require_open()
        return replace(self._worker_state().diagnostics, eager_reason=self._eager_reason)

    @property
    def runner(self) -> WorkloadModelRunner | Mrv1Runner:
        self._require_open()
        if self._runner is None:
            raise RuntimeError("use tensor_parallel_size=1 and multiprocessing=False to access the runner")
        return self._runner

    def _command(self, command: BackendCommand, payload: str = "", *, timeout: float = 30.0) -> list[str]:
        replies = self._require_open().collective_rpc(
            "workload_command", timeout=timeout, args=(command.value, payload)
        )
        return replies

    def _worker_state(self) -> BackendWorkerState:
        states = [decode_boundary(reply, BackendWorkerState) for reply in self._command(BackendCommand.DIAGNOSTICS)]
        assert all(state.diagnostics.logical_counts() == states[0].diagnostics.logical_counts() for state in states), (
            "TP worker logical counters diverged",
            states,
        )
        return states[0]

    def trace_snapshot(self) -> TraceSnapshot:
        return self.trace_snapshots()[0]

    def trace_snapshots(self) -> tuple[TraceSnapshot, ...]:
        """Return every worker's trace in rank order; trace_snapshot selects the output rank."""
        return tuple(decode_boundary(reply, TraceSnapshot) for reply in self._command(BackendCommand.TRACE))

    def profile_tasks(self, tasks: tuple[TaskId, ...]) -> None:
        self._command(BackendCommand.PROFILE, encode_boundary(tasks))

    def select_trace(self, selection: TraceSelection) -> None:
        """Select what the trace records: bounded tensor/gradient snapshots and opt-in Kineto profiling."""
        self._command(BackendCommand.SELECT_TRACE, encode_boundary(selection))

    def close(self) -> None:
        if self._llm is None and self._pending_input is None:
            return
        try:
            if not self._closed:
                self._command(BackendCommand.DETACH)
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        # Closed to new work immediately; retain ownership if shutdown needs a retry.
        self._closed = True
        if self._llm is not None:
            self._llm.llm_engine.engine_core.shutdown()
        self._llm = self._runner = None
        if self._pending_input is not None:
            self._pending_input.close()
            self._pending_input = None

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
        edit: Callable[[FeedbackT], StateImport[torch.Tensor] | None] | None = None,
        routes: tuple[str, ...] | None = None,
    ) -> Iterator[tuple[CompletionInput, FeedbackT]]:
        """Generate, settle feedback, then retry eligible original prompts within a finite bound.

        The callable runs locally after generate has released backend requests; it is
        never installed on a worker. Each yield is the original completion and result,
        with no retained history. Multi-request barriers are regenerated as a full
        vote group. Closing the iterator leaves no pending generation or transaction.
        Optional edits reuse StateImport and require a signed activation workload;
        the feedback adapter returns original tensors, never a worker callback.
        A callback that returns None has nothing to feed back for its round: the round
        is not yielded (the callback already saw its completions) and nothing retries.
        """
        if max_rounds < 1 or not prompts:
            raise ValueError("feedback generation requires a nonempty prompt list and positive finite round bound")
        # Round k samples indices k*n..(k+1)*n-1; fail before any round trains state.
        if self._sample_limit is not None and max_rounds * sampling_params.n > self._sample_limit:
            raise ValueError(
                f"{max_rounds} feedback rounds of n={sampling_params.n} samples exceed this workload's "
                f"pllm.sample_index(limit={self._sample_limit}); lower max_rounds to at most "
                f"{self._sample_limit // sampling_params.n} or raise the workload's limit"
            )
        texts: list[str] = list(prompts)
        indices: tuple[int, ...] = self._prompt_indices(texts, prompt_indices)
        route_values: list[str] | None = list(routes) if routes is not None else None
        edits: HostBatch[torch.Tensor] | None = None
        for attempt in range(max_rounds):
            batch: CompletionInput = self.generate(
                texts,
                sampling_params,
                use_tqdm=use_tqdm,
                feedback_group=feedback_group,
                prompt_indices=indices,
                sample_index=attempt * sampling_params.n,
                batch=edits,
                completion_device=completion_device,
                routes=None if route_values is None else tuple(route_values),
            )
            result = feedback(batch)
            if result is None:
                return
            entries, controls = batch.completions.entries, result.retry_controls
            ready: set[str] = set()
            for barrier in result.update_barriers:
                members = set(barrier.request_ids)
                if barrier.state is BarrierState.READY and barrier.ready_count > 0 and result.update_count > 0:
                    ready.update(members)
            retry: set[str] = {
                control.completion.identity.request_id
                for control in controls
                if control.action is RetryAction.RETRY
                and (not control.retry_after_update or control.completion.identity.request_id in ready)
            }
            stopped: set[str] = {
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
            edits = edit(result) if edit is not None and selected and attempt + 1 < max_rounds else None
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
        batch: HostBatch[torch.Tensor] | None = None,
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
        batch: HostBatch[torch.Tensor] | None = None,
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
        batch: HostBatch[torch.Tensor] | None = None,
        sample_index: int = 0,
        feedback_group: str | None = None,
        prompt_indices: tuple[int, ...] | None = None,
        completion_device: torch.device | None = None,
        routes: tuple[str, ...] | None = None,
    ) -> list[RequestOutput] | CompletionInput:
        self._require_open()
        if completion_device is not None and feedback_group is None:
            raise ValueError("completion device requires completion feedback")
        texts = [prompts] if isinstance(prompts, str) else prompts
        if not texts or any(not prompt for prompt in texts):
            raise ValueError("provide non-empty prompts")
        if sample_index < 0 or feedback_group is not None and not feedback_group:
            raise ValueError("use a nonnegative sample index and a non-empty feedback group")
        n = sampling_params.n
        if self._sample_limit is not None and sample_index + n > self._sample_limit:
            raise ValueError(
                f"sample indices {sample_index}..{sample_index + n - 1}: this workload reads "
                f"pllm.sample_index(limit={self._sample_limit}); keep sample_index + n at most {self._sample_limit} or "
                "raise the workload's limit"
            )
        indices = self._prompt_indices(texts, prompt_indices)
        # A route value is user-computed (regex, discriminator, tenant); the signed
        # graph only declares whether any request state keys on it.
        signed_route = any(
            RequestKeyField.ROUTE in state.key_fields for state in self.attachment.plan.semantic_graph.states
        )
        if routes is not None and (not signed_route or len(routes) != len(texts) or any(not route for route in routes)):
            raise ValueError("routes require a signed route-keyed state and one value per prompt")
        if signed_route and routes is None:
            raise ValueError("a route-keyed request state requires per-prompt routes")
        validate_host_batch(
            batch,
            self.attachment.plan.semantic_graph,
            self.attachment.plan.tensor_layout_request,
            self.attachment.plan.timeline,
            SharedTensor.describe,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
        )
        llm = self._require_open()
        llm.sleep(level=0, mode="keep")
        request_ids: list[str] = []
        lifetime = ExitStack()
        try:
            # Each of a prompt's n samples is its own request, with its own sample index and request state.
            request_ids = llm.enqueue(
                [prompt for prompt in texts for _ in range(n)],
                [self._cache_policy.sampling(sample) for _ in texts for sample in _one_per_sample(sampling_params)],
                use_tqdm=use_tqdm,
            )
            identities = self._request_identities(request_ids, indices, sample_index, n, routes)
            # Borrow vLLM's exact frontend mapping before completed entries are removed.
            external = {
                llm.llm_engine.output_processor.request_states[identity.request_id].external_req_id: identity
                for identity in identities
            }
            destination = self._stage(identities, batch, feedback_group, completion_device, lifetime)
            self._next_prompt_index = max(self._next_prompt_index, max(indices) + 1)
            llm.wake_up(tags=["scheduling"])
            outputs = llm.wait_for_completion(RequestOutput, use_tqdm=use_tqdm)
            return self._collect(outputs, identities, external, feedback_group, destination, n)
        except Exception as original:
            try:
                llm.llm_engine.abort_request(request_ids, internal=True)
                self._command(BackendCommand.ABORT)
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

    def _prompt_indices(self, texts: list[str], prompt_indices: tuple[int, ...] | None) -> tuple[int, ...]:
        indices = (
            tuple(range(self._next_prompt_index, self._next_prompt_index + len(texts)))
            if prompt_indices is None
            else prompt_indices
        )
        if len(indices) != len(texts) or any(index < 0 for index in indices):
            raise ValueError("prompt indices must align with the original input prompts")
        groups: dict[int, str] = {}
        for index, prompt in zip(indices, texts, strict=True):
            if groups.setdefault(index, prompt) != prompt:
                raise ValueError("one prompt identity cannot name different input texts")
        return indices

    def _request_identities(
        self,
        request_ids: list[str],
        indices: tuple[int, ...],
        sample_index: int,
        n: int,
        routes: tuple[str, ...] | None,
    ) -> tuple[RequestIdentity, ...]:
        """Sample j of each prompt gets sample index `sample_index + j`."""
        samples = (
            (index, sample_index + j, route)
            for index, route in zip(indices, routes if routes is not None else ("",) * len(indices), strict=True)
            for j in range(n)
        )
        return tuple(
            RequestIdentity(request_id, index, sample, route)
            for request_id, (index, sample, route) in zip(request_ids, samples, strict=True)
        )

    def _stage(
        self,
        identities: tuple[RequestIdentity, ...],
        batch: HostBatch[torch.Tensor] | None,
        feedback_group: str | None,
        completion_device: torch.device | None,
        lifetime: ExitStack,
    ) -> torch.Tensor | None:
        output_device = completion_device or (torch.device("cpu") if self._runner is None else self.device)
        if output_device.type == "cuda":
            if output_device == torch.device("cuda"):
                output_device = self.device
            if output_device != self.device:
                raise ValueError("completion IPC currently requires the model's CUDA device")
        elif output_device.type != "cpu":
            raise ValueError("completion output requires CPU or CUDA storage")
        destination = None
        completion_wire = None
        if feedback_group is not None:
            field = TensorFieldSpec(LayoutFieldId(0), (len(identities), self.hidden_size), TensorDType.FLOAT32, 4)
            with torch.inference_mode(False):
                if self._runner is None:
                    destination, completion_wire = SharedTensor.allocate(field, lifetime, output_device)
                else:
                    destination = torch.empty(field.shape, device=output_device, dtype=torch.float32)
        if self._runner is None:
            tp = self._require_open().llm_engine.vllm_config.parallel_config.tensor_parallel_size
            batch_wire = map_host_batch(
                batch, lambda tensor: SharedTensor.export(tensor.cpu() if tp > 1 else tensor, lifetime)
            )
            self._command(
                BackendCommand.STAGE,
                encode_boundary(BackendStage(identities, batch_wire, completion_wire)),
            )
        else:
            batch = map_host_batch(batch, lambda tensor: tensor.to(self.device))
            self.runner.require_session().stage_input(batch, identities, destination)
        return destination

    def _collect(
        self,
        outputs: list[RequestOutput],
        identities: tuple[RequestIdentity, ...],
        external: dict[str, RequestIdentity],
        feedback_group: str | None,
        destination: torch.Tensor | None,
        n: int,
    ) -> list[RequestOutput] | CompletionInput:
        # FINISH validates and synchronizes the staged rows before releasing requests.
        self._command(BackendCommand.FINISH)
        if feedback_group is None:
            return _per_prompt(outputs, n)
        completions: dict[RequestIdentity, Completion] = {}
        for output in outputs:
            identity = external[output.request_id]
            (completion,) = output.outputs
            completions[identity] = Completion(
                identity,
                completion.index,
                feedback_group,
                tuple(completion.token_ids),
                completion.text,
                FinishReason(completion.finish_reason),
            )
        assert destination is not None
        return CompletionInput(
            CompletionBatch(tuple(completions[identity] for identity in identities)),
            destination,
            CudaStreamReadiness(torch.cuda.current_stream(destination.device))
            if destination.device.type == "cuda"
            else None,
        )


def _one_per_sample(params: SamplingParams) -> list[SamplingParams]:
    """vLLM's own split of n samples into requests: n=1 each, and seed + j for sample j when a seed is set."""
    samples: list[SamplingParams] = []
    for j in range(params.n):
        sample = params.clone()
        sample.n = 1
        if params.seed is not None:
            sample.seed = params.seed + j
        samples.append(sample)
    return samples


def _per_prompt(outputs: list[RequestOutput], n: int) -> list[RequestOutput]:
    """One RequestOutput per prompt holding its n completions, as vLLM returns for SamplingParams(n=...)."""
    merged = outputs[::n]
    for start, output in zip(range(0, len(outputs), n), merged, strict=True):
        output.outputs = [replace(sample.outputs[0], index=j) for j, sample in enumerate(outputs[start : start + n])]
    return merged


def vllm_engine(
    config: ModelConfig, workload_options: dict[str, Any], environment: dict[str, str], *, cuda_graphs: bool
) -> LLM:
    """Build the vLLM engine; a workload adds its worker, scheduler and graph options."""
    if config.tensor_parallel_size > 1:
        if cuda_graphs:
            workload_options = {**graph_options(config, BackendRequirements()), **workload_options}
        environment.update(VLLM_USE_V2_MODEL_RUNNER="1", VLLM_WORKER_MULTIPROC_METHOD="spawn")
    previous = {name: os.environ.get(name) for name in environment}
    if any(previous[name] not in (None, value) for name, value in environment.items()):
        raise RuntimeError("vLLM workload conflicts with explicit MP, spawn or legacy-runner environment")
    os.environ.update(environment)
    try:
        return LLM(
            model=config.model,
            revision=config.revision,
            dtype=config.dtype,
            tensor_parallel_size=config.tensor_parallel_size,
            pipeline_parallel_size=config.pipeline_parallel_size,
            data_parallel_size=config.data_parallel_size,
            distributed_executor_backend="mp" if config.tensor_parallel_size > 1 else "uni",
            enforce_eager=not cuda_graphs,
            max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            gpu_memory_utilization=config.gpu_memory_utilization,
            seed=config.seed,
            trust_remote_code=config.trust_remote_code,
            enable_prefix_caching=config.enable_prefix_caching,
            enable_chunked_prefill=config.enable_chunked_prefill,
            kv_transfer_config=None if config.kv_offload is None else config.kv_offload.options(),
            **workload_options,
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name)
            else:
                os.environ[name] = value


def _eager_reason(config: ModelConfig, requirements: BackendRequirements) -> str:
    """Why the workload's model runs without vLLM's CUDA graphs; empty when it can use them."""
    if config.cuda_graphs is False:
        return "ModelConfig(cuda_graphs=False)"
    if config.runner == "mrv1":
        return 'runner="mrv1" runs workloads without CUDA graphs'
    if requirements.activation_layers:
        return "activation patches run outside CUDA graphs"
    if requirements.custom_forward:
        return "@pllm.custom_forward reruns model layers outside CUDA graphs"
    if requirements.instance_adapter:
        return "CUDA-graph overlays read request state only; declare the pllm.model.Adapter state as pllm.RequestState"
    return ""


def create_vllm_llm(
    workload: BackendWorkload[DataT, ResultT],
    config: ModelConfig,
    *,
    definition: str | None = None,
) -> VllmLlm[DataT, ResultT]:
    requirements = workload.backend_requirements()
    compat.admit_parallel(config, requirements)
    if requirements.position_states and config.runner == "mrv1":
        raise ValueError('PositionState and @pllm.custom_forward need runner="mrv2"')
    eager_reason = _eager_reason(config, requirements)
    if config.cuda_graphs is True and eager_reason:
        raise ValueError(f'{eager_reason}; set ModelConfig(cuda_graphs="auto") or cuda_graphs=False')
    if config.cuda_graphs == "auto" and eager_reason:
        _logger.warning("running the model without CUDA graphs: %s", eager_reason)
    environment = {
        _MP_ENV: "1" if config.multiprocessing else "0",
        _V2_RUNNER_ENV: "1" if config.runner == "mrv2" else "0",
        CAPTURE_ENV: encode_boundary(
            HiddenCaptureAnnouncement(requirements.hidden_inputs, config.max_num_seqs, TensorDType(config.dtype))
        ),
        POSITION_ENV: encode_boundary(PositionAnnouncement(requirements.position_states)),
    }
    if config.multiprocessing or config.tensor_parallel_size > 1:
        environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    options: GraphOptions = {} if eager_reason else graph_options(config, requirements)
    # vLLM keys its compiled-model cache by its own config; the hidden taps and position states pllm adds to the
    # forward must enter that key, or an engine loads a model compiled for another workload. pllm's entries are
    # integers (overlay_graph decodes them as dict[str, int]), so the digest enters as one.
    forward = sha256((environment[CAPTURE_ENV] + environment[POSITION_ENV]).encode()).hexdigest()
    options["additional_config"] = {**options.get("additional_config", {}), "pllm_forward": int(forward[:15], 16)}
    llm = vllm_engine(
        config,
        {
            **options,
            "worker_cls": _WORKER_CLASS,
            # The supported uni-process, non-speculative decoder enables async by default.
            "async_scheduling": config.runner == "mrv2",
            "scheduler_cls": "pllm.backends.vllm.prefix_cache."
            + ("WorkloadAsyncScheduler" if config.runner == "mrv2" else "WorkloadScheduler"),
        },
        environment,
        cuda_graphs=not eager_reason,
    )

    with ExitStack() as construction:
        construction.callback(llm.llm_engine.engine_core.shutdown)
        if config.multiprocessing or config.tensor_parallel_size > 1:
            replies = llm.collective_rpc(
                "workload_command",
                timeout=30.0,
                args=(
                    BackendCommand.PROBE.value,
                    encode_boundary(BackendProbe(requirements, config.execution.publish_lag)),
                ),
            )
            probes = [decode_boundary(reply, BackendCompileFacts) for reply in replies]
            budget = min(probe.state_memory_budget_bytes for probe in probes)
            facts = replace(probes[0], state_memory_budget_bytes=budget)
            assert all(replace(probe, state_memory_budget_bytes=budget) == facts for probe in probes), (
                "TP workers disagree on model or planning facts"
            )
            facts, declaration = workload.specialize(facts, config.execution)
            attachment = workload.plan_backend(facts, declaration)
            policy = CachePolicy.for_attachment(attachment, config.generated_kv)
            handle = VllmLlm[DataT, ResultT](
                llm, None, attachment, facts, policy, requirements.sample_limit, eager_reason
            )
            install = encode_boundary(BackendInstall(facts, attachment, definition or "", policy))
            llm.collective_rpc("workload_command", timeout=30.0, args=(BackendCommand.INSTALL.value, install))
        else:
            runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
            facts, declaration = workload.specialize(
                probe_backend(requirements, runner, publish_lag=config.execution.publish_lag),
                config.execution,
                causal=partial(CausalTraining, runner.decoder),
            )
            deployment = workload.bind_backend(facts, workload.plan_backend(facts, declaration), declaration)
            policy = CachePolicy.for_attachment(deployment.attachment, config.generated_kv)
            runner.attach_workload(deployment, policy)
            handle = VllmLlm[DataT, ResultT](
                llm, runner, deployment.attachment, facts, policy, requirements.sample_limit, eager_reason
            )
        construction.pop_all()
        return handle


def probe_backend(
    requirements: BackendRequirements,
    runner: WorkloadModelRunner | Mrv1Runner,
    *,
    publish_lag: int = 1,
) -> BackendCompileFacts:
    decoder = runner.decoder
    model = decoder.model
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
    requested_targets = expand_targets(requirements.parameter_targets, layer_count)
    targets = inspect_parameter_targets(decoder, requested_targets, runner.dtype) if requested_targets else ()
    custom_forward_capabilities = (
        ("model/custom_forward@1", TOKEN_CAPTURE_CAPABILITY)
        if runner.runner_kind == "mrv2" and graph_mode(runner.overlay_graph_bytes) is AttachmentGraphMode.EAGER
        else ()
    )
    probe = _base_probe(
        runner,
        model_fingerprint=_digest(
            (
                f"model={runner.model_config.model}",
                f"revision={runner.model_config.revision or ''}",
                f"class={decoder.identity}",
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
                        *custom_forward_capabilities,
                        f"backend/vllm_{runner.runner_kind}@1",
                    )
                )
            )
        ),
    )
    observed = model_timeline(
        runner.device,
        (
            "state/dense_page_cuda@1",
            "state/fenced_reclaim@1",
            "transition/full_root@1",
            "optimizer/adamw@1",
            *(("backend/tensor_parallel@1",) if runner.parallel_config.tensor_parallel_size > 1 else ()),
            *custom_forward_capabilities,
        ),
    )
    required = input_reservations(
        probe,
        runner.hidden_layouts,
        observed,
        runner.max_num_reqs,
        model.lm_head.num_embeddings_padded,
        requirements.sampler_stage,
        requirements.sampler_feedback,
    )
    required = tuple(value + runner.overlay_graph_bytes for value in required)
    free_bytes, _total_bytes = torch.cuda.mem_get_info(runner.device)
    # Per scheduled request: the current root, the private root being written, and the root the GPU
    # may still read from the step before, because the host prepares a step while the GPU runs the last.
    # Each lagged step additionally retains its exact input root for readers. No host wait
    # is hidden in this capacity: current + private + GPU reader + publish_lag retained roots.
    capacity = (3 + publish_lag) * runner.max_num_reqs + 2
    facts = BackendCompileFacts(
        replace(probe, targets=targets, position_states=requirements.position_states),
        replace(observed, reserved_bytes=required),
        (
            PageBundleProviderFacts(
                "canonical.page_bundle_cuda@1",
                capacity,
                probe.device_kind,
            ),
        ),
        runner.max_num_reqs,
        free_bytes,
        runner.model_config.seed,
        model.config.vocab_size,
        layer_count,
    )
    return facts


def _base_probe(
    runner: WorkloadModelRunner | Mrv1Runner,
    *,
    model_fingerprint: str,
    hidden_size: int,
    layer_count: int,
    hidden_resources: tuple[AttachmentHiddenResource, ...],
    capabilities: tuple[str, ...],
) -> BackendPlacementProbe:
    return BackendPlacementProbe(
        model_fingerprint=model_fingerprint,
        graph_mode=graph_mode(runner.overlay_graph_bytes),
        device_kind=StateDeviceKind.CUDA,
        tensor_dtype=_tensor_dtype(runner.dtype),
        hidden_size=hidden_size,
        max_packed_rows=runner.max_num_tokens,
        hidden_resources=hidden_resources,
        capabilities=capabilities,
        causal_domain=CausalModelDomain(runner.max_num_reqs, layer_count, runner.model_config.max_model_len),
        position_states=runner.position_states,
    )


__all__ = [
    "VllmLlm",
    "create_vllm_llm",
]


def input_reservations(
    probe: BackendPlacementProbe,
    hidden_layouts: tuple[HiddenCaptureLayout, ...],
    timeline: SchedulerTimeline,
    rows: int,
    vocab: int,
    stage: SamplerStage | None,
    feedback: bool,
) -> tuple[int, ...]:
    """Reserve persistent capture storage and sampler scratch across model-step anchors."""
    # Layouts union aliases and phases per module. Slots and capture indices stay
    # resident across steps; task_memory traces consumer tables, gathers and casts.
    hidden = sum(layout.resident_bytes for layout in hidden_layouts)
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
