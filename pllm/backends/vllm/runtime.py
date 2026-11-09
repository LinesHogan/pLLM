"""Workload-neutral public handle and vLLM engine construction: probing, installation and the VllmLlm API.

Sessions live in session.py and runners in runner.py; this module builds the engine and exposes the handle.
"""

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import partial
from hashlib import sha256
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import torch
from vllm import RequestOutput, SamplingParams
from vllm.entrypoints.llm import LLM
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from pllm.authoring import (
    BackendRequirements,
    BackendWorkload,
    DataT,
    HostBatch,
    ResultT,
    StateImport,
    map_host_batch,
)
from pllm.backends.vllm.admission import admit_workload
from pllm.backends.vllm.checkpoint import load_options
from pllm.backends.vllm.compile_facts import compile_facts
from pllm.backends.vllm.hidden_taps import TAPS_ENV, HiddenTapAnnouncement
from pllm.backends.vllm.model_shape import ModelShape
from pllm.backends.vllm.overlay_graph import (
    GraphOptions,
    graph_options,
)
from pllm.backends.vllm.training import CausalTraining
from pllm.config import ExecutionOptions, ModelConfig, ModelDType
from pllm.contracts.backend_attachment import (
    BackendAttachment,
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
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.semantic_graph import TaskId
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType, TensorFieldSpec
from pllm.execution.checkpoint import checkpoint_directory
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.host_runtime import CompletionInput
from pllm.execution.shared_tensor import SharedTensor
from pllm.execution.state_execution import StateDiagnostics
from pllm.execution.state_feed import Feed
from pllm.execution.torch_private import (
    blas_workspace_bytes,
    bytecode_hooks,
    clear_blas_workspaces,
    remove_bytecode_hooks,
)
from pllm.execution.trace import TraceSelection, TraceSnapshot

if TYPE_CHECKING:
    from pllm.backends.vllm.mrv1 import Mrv1Runner


from pllm.backends.vllm.kv_layout import LAYOUT_ENV
from pllm.backends.vllm.position_state import POSITION_ENV
from pllm.backends.vllm.prefix_cache import CachePolicy
from pllm.backends.vllm.runner import BackendWorkerState, WorkloadModelRunner
from pllm.execution.position_state import PositionAnnouncement

_logger = logging.getLogger(__name__)

_MP_ENV = "VLLM_ENABLE_V1_MULTIPROCESSING"


_V2_RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"


_WORKER_CLASS = "pllm.worker.GPUWorker"


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
    samples_per_prompt: int = 1
    driver_pid: int = field(default_factory=os.getpid)


FeedbackT = TypeVar("FeedbackT", bound=CompletionFeedback)


@dataclass(frozen=True, slots=True)
class _Generation:
    """What one generate call produced: vLLM's outputs, each sample's identity (by vLLM's external request id) and
    the final hidden rows when completions were requested."""

    outputs: list[RequestOutput]
    identities: tuple[RequestIdentity, ...]
    external: dict[str, RequestIdentity]
    destination: torch.Tensor | None
    n: int

    def completion_input(self, group_id: str) -> CompletionInput:
        """Every sample as a Completion labelled `group_id`, in prompt-major order, with its final hidden row."""
        assert self.destination is not None
        completions: dict[RequestIdentity, Completion] = {}
        for output in self.outputs:
            identity = self.external[output.request_id]
            (completion,) = output.outputs
            completions[identity] = Completion(
                identity,
                completion.index,
                group_id,
                tuple(completion.token_ids),
                completion.text,
                FinishReason(completion.finish_reason),
            )
        return CompletionInput(
            CompletionBatch(tuple(completions[identity] for identity in self.identities)), self.destination
        )


@dataclass(slots=True)
class VllmLlm(Generic[DataT, ResultT]):
    """The engine handle returned by make_llm, with workload diagnostics, traces and feeds when installed."""

    llm: LLM | None
    _runner: WorkloadModelRunner | Mrv1Runner | None = None
    backend_attachment: BackendAttachment | None = None
    compile_facts: BackendCompileFacts | None = None
    cache_policy: CachePolicy | None = None
    _sample_limit: int | None = None
    _eager_reason: str = ""
    _closed: bool = False
    next_prompt_index: int = 0
    _pending_input: ExitStack | None = None
    compile_hooks: tuple[int, ...] = ()  # Dynamo bytecode hooks the in-process engine registered; see vllm_engine

    def require_llm(self) -> LLM:
        """Return the live vLLM engine; a closed handle cannot execute operations."""
        if self._closed or self.llm is None:
            raise RuntimeError("model handle is closed")
        return self.llm

    @property
    def hidden_size(self) -> int:
        return (
            self.compile_facts.probe.hidden_size
            if self.compile_facts is not None
            else self.require_llm().llm_engine.vllm_config.model_config.get_hidden_size()
        )

    @property
    def vocab_size(self) -> int:
        return (
            self.compile_facts.vocab_size
            if self.compile_facts is not None
            else self.require_llm().llm_engine.vllm_config.model_config.get_vocab_size()
        )

    @property
    def attachment(self) -> BackendAttachment:
        """The installed workload's plan; workload-only operations require one."""
        self.require_llm()
        if self.backend_attachment is None:
            raise ValueError("pass workload= to make_llm to use workload operations")
        return self.backend_attachment

    @property
    def device(self) -> torch.device:
        return self._runner.device if self._runner is not None else torch.device("cuda", 0)

    def __enter__(self) -> VllmLlm[DataT, ResultT]:
        self.require_llm()
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
        self.require_llm()
        if self._runner is None:
            raise ValueError(
                "state feeds need tensor_parallel_size=1 and multiprocessing=False; pass the writer to a spawned producer"
            )
        return self.runner.require_session().feed(name)

    def diagnostics(self) -> StateDiagnostics:
        """Read exact workload health and vLLM's allocated KV blocks/max-length concurrency."""
        self.require_llm()
        return replace(self.worker_state().diagnostics, eager_reason=self._eager_reason)

    @property
    def runner(self) -> WorkloadModelRunner | Mrv1Runner:
        self.require_llm()
        if self._runner is None:
            raise RuntimeError("use tensor_parallel_size=1 and multiprocessing=False to access the runner")
        return self._runner

    def command(self, command: BackendCommand, payload: str = "", *, timeout: float = 30.0) -> list[str]:
        """Send a workload command to every engine worker and collect their replies."""
        self.attachment
        replies = self.require_llm().collective_rpc("workload_command", timeout=timeout, args=(command.value, payload))
        return replies

    def worker_state(self) -> BackendWorkerState:
        """Read exact diagnostics and state snapshots from the engine workers."""
        states = [decode_boundary(reply, BackendWorkerState) for reply in self.command(BackendCommand.DIAGNOSTICS)]
        assert all(state.diagnostics.logical_counts() == states[0].diagnostics.logical_counts() for state in states), (
            "TP worker logical counters diverged",
            states,
        )
        return states[0]

    def trace_snapshot(self) -> TraceSnapshot:
        return self.trace_snapshots()[0]

    def trace_snapshots(self) -> tuple[TraceSnapshot, ...]:
        """Return every worker's trace in rank order; trace_snapshot selects the output rank."""
        return tuple(decode_boundary(reply, TraceSnapshot) for reply in self.command(BackendCommand.TRACE))

    def profile_tasks(self, tasks: tuple[TaskId, ...]) -> None:
        self.command(BackendCommand.PROFILE, encode_boundary(tasks))

    def select_trace(self, selection: TraceSelection) -> None:
        """Select what the trace records: bounded tensor/gradient snapshots and opt-in Kineto profiling."""
        self.command(BackendCommand.SELECT_TRACE, encode_boundary(selection))

    def close(self) -> None:
        if self.llm is None and self._pending_input is None:
            return
        try:
            if not self._closed and self.backend_attachment is not None:
                self.command(BackendCommand.DETACH)
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        # Closed to new work immediately; retain ownership if shutdown needs a retry.
        self._closed = True
        if self.llm is not None:
            self.llm.llm_engine.engine_core.shutdown()
        self.llm = self._runner = None
        if self._pending_input is not None:
            self._pending_input.close()
            self._pending_input = None
        # Process-wide torch.compile state retains a closed engine: the bytecode hooks hold its weights and the
        # cached backends its CUDA graph pools. Closing an in-process engine already ends any other in-process
        # engine (vLLM tears down the shared parallel groups), so resetting the shared compile caches costs no one.
        remove_bytecode_hooks(self.compile_hooks)
        torch.compiler.reset()
        # vLLM's shutdown runs its collection while this handle still owns the runner. Collect model/session
        # cycles only after dropping those owners.
        free_cuda_garbage()

    def generate_feedback(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
        *,
        feedback: Callable[[CompletionInput], FeedbackT | None],
        group_id: str,
        max_rounds: int,
        prompt_indices: tuple[int, ...] | None = None,
        use_tqdm: bool = False,
        completion_device: torch.device | None = None,
        edit: Callable[[FeedbackT], StateImport[torch.Tensor] | None] | None = None,
        routes: tuple[str, ...] | None = None,
    ) -> Iterator[tuple[CompletionInput, FeedbackT]]:
        """Generate completions labelled `group_id`, run `feedback` on them, then regenerate the samples it marks
        RETRY, at most `max_rounds` rounds. A retried sample runs alone (n=1) and keeps its place among its prompt's
        samples: sample j has index k*n + j in round k.

        The callable runs locally after generate has released backend requests; it is
        never installed on a worker. Each yield is the original completion and result,
        with no retained history. Multi-request barriers are regenerated as a full
        vote group. Closing the iterator leaves no pending generation.
        Optional edits return a StateImport, one row per retried sample, for an activation-patch workload;
        the feedback adapter returns original tensors, never a worker callback.
        A callback that returns None has nothing to feed back for its round: the round
        is not yielded (the callback already saw its completions) and nothing retries.
        """
        if max_rounds < 1 or not prompts or not group_id:
            raise ValueError("generate_feedback needs at least one prompt, a non-empty group_id and max_rounds >= 1")
        n = sampling_params.n
        # Sample j of a prompt has index k*n + j in round k; fail before any round trains state.
        if self._sample_limit is not None and max_rounds * n > self._sample_limit:
            raise ValueError(
                f"{max_rounds} feedback rounds of n={n} samples exceed this workload's "
                f"pllm.sample_index(limit={self._sample_limit}); lower max_rounds to at most "
                f"{self._sample_limit // n} or raise the workload's limit"
            )
        texts: list[str] = list(prompts)
        indices: tuple[int, ...] = self.resolve_prompt_indices(texts, prompt_indices)
        text_of = dict(zip(indices, texts, strict=True))
        first_samples: tuple[int, ...] = (0,) * len(texts)
        edits: HostBatch[torch.Tensor] | None = None
        device = self._completion_device(completion_device)
        for attempt in range(max_rounds):
            batch = self._generate(
                texts, sampling_params, use_tqdm, edits, first_samples, indices, routes, device
            ).completion_input(group_id)
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
            retried = [entry.identity for entry in entries if entry.identity.request_id in retry]
            edits = edit(result) if edit is not None and retried and attempt + 1 < max_rounds else None
            yield batch, result
            if not retried:
                return
            # Each retried sample is regenerated alone, as one prompt with n=1, under its next round's index.
            texts = [text_of[identity.prompt_index] for identity in retried]
            indices = tuple(identity.prompt_index for identity in retried)
            first_samples = tuple(identity.sample_index + n for identity in retried)
            if routes is not None:
                routes = tuple(identity.route for identity in retried)
            sampling_params = sampling_params.clone()
            sampling_params.n = 1

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams,
        *,
        use_tqdm: bool = False,
        batch: HostBatch[torch.Tensor] | None = None,
        sample_index: int = 0,
        prompt_indices: tuple[int, ...] | None = None,
        routes: tuple[str, ...] | None = None,
    ) -> list[RequestOutput]:
        """vLLM's generate with the workload running: one RequestOutput per prompt, holding its n samples.

        `batch` feeds the methods that read pllm.batch fields this call; `sample_index` numbers the first sample of
        each prompt; `routes` gives one routing value per prompt for request states keyed by "route".
        """
        if self.backend_attachment is None:
            if batch is not None or sample_index or prompt_indices is not None or routes is not None:
                raise ValueError("batch, sample_index, prompt_indices and routes need workload= in make_llm")
            return self.require_llm().generate(prompts, sampling_params, use_tqdm=use_tqdm)
        texts = [prompts] if isinstance(prompts, str) else prompts
        run = self._generate(
            texts, sampling_params, use_tqdm, batch, (sample_index,) * len(texts), prompt_indices, routes, None
        )
        return _per_prompt(run.outputs, run.n)

    def generate_completions(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams,
        *,
        group_id: str,
        use_tqdm: bool = False,
        batch: HostBatch[torch.Tensor] | None = None,
        sample_index: int = 0,
        prompt_indices: tuple[int, ...] | None = None,
        completion_device: torch.device | None = None,
        routes: tuple[str, ...] | None = None,
    ) -> CompletionInput:
        """Generate like `generate`, returning every sample as a Completion labelled `group_id` together with its
        final hidden row (on `completion_device`, by default the model's device), the input host feedback reads."""
        if not group_id:
            raise ValueError("generate_completions needs a non-empty group_id")
        texts = [prompts] if isinstance(prompts, str) else prompts
        device = self._completion_device(completion_device)
        run = self._generate(
            texts, sampling_params, use_tqdm, batch, (sample_index,) * len(texts), prompt_indices, routes, device
        )
        return run.completion_input(group_id)

    def _completion_device(self, requested: torch.device | None) -> torch.device:
        """Where completion hidden rows land: `requested`, else the model's device (CPU for an EngineCore engine)."""
        if self.backend_attachment is None:
            raise ValueError(
                "completions carry each sample's final hidden row, which only a workload collects; pass "
                "workload=pllm.workloads.DummyWorkload() to make_llm when nothing should train"
            )
        return requested or (torch.device("cpu") if self._runner is None else self.device)

    def _generate(
        self,
        texts: list[str],
        sampling_params: SamplingParams,
        use_tqdm: bool,
        batch: HostBatch[torch.Tensor] | None,
        first_samples: tuple[int, ...],
        prompt_indices: tuple[int, ...] | None,
        routes: tuple[str, ...] | None,
        completion_device: torch.device | None,
    ) -> _Generation:
        """Run one generate call; prompt p's samples get indices first_samples[p] + j. With a `completion_device`,
        also collect each sample's final hidden row there."""
        self.require_llm()
        if not texts or any(not prompt for prompt in texts):
            raise ValueError("provide non-empty prompts")
        if min(first_samples) < 0:
            raise ValueError("use a nonnegative sample index")
        n = sampling_params.n
        if self._sample_limit is not None and max(first_samples) + n > self._sample_limit:
            raise ValueError(
                f"sample index {max(first_samples) + n - 1}: this workload reads "
                f"pllm.sample_index(limit={self._sample_limit}); keep sample_index + n at most {self._sample_limit} or "
                "raise the workload's limit"
            )
        indices = self.resolve_prompt_indices(texts, prompt_indices)
        # A route value is user-computed (regex, discriminator, tenant); the signed
        # graph only declares whether any request state keys on it.
        signed_route = any(
            RequestKeyField.ROUTE in state.key_fields for state in self.attachment.plan.semantic_graph.states
        )
        if routes is not None and (not signed_route or len(routes) != len(texts) or any(not route for route in routes)):
            raise ValueError('routes= needs a request state keyed by "route" and one value per prompt')
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
            prompt_count=len(texts),
            import_fields=self.attachment.spec.import_fields,
        )
        llm = self.require_llm()
        assert self.cache_policy is not None
        llm.sleep(level=0, mode="keep")
        request_ids: list[str] = []
        lifetime = ExitStack()
        try:
            # Each of a prompt's n samples is its own request, with its own sample index and request state.
            request_ids = llm.enqueue(
                [prompt for prompt in texts for _ in range(n)],
                [
                    self.cache_policy.sampling(sample)
                    for first in first_samples
                    for sample in _one_per_sample(sampling_params, first)
                ],
                use_tqdm=use_tqdm,
            )
            identities = self._request_identities(request_ids, indices, first_samples, n, routes)
            # Borrow vLLM's exact frontend mapping before completed entries are removed.
            external = {
                llm.llm_engine.output_processor.request_states[identity.request_id].external_req_id: identity
                for identity in identities
            }
            destination = self._stage(identities, batch, completion_device, lifetime, n)
            self.next_prompt_index = max(self.next_prompt_index, max(indices) + 1)
            llm.wake_up(tags=["scheduling"])
            outputs = llm.wait_for_completion(RequestOutput, use_tqdm=use_tqdm)
            # FINISH validates and synchronizes the staged rows before releasing requests.
            self.command(BackendCommand.FINISH)
            return _Generation(outputs, identities, external, destination, n)
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
            if self._closed and self.llm is not None:
                self._pending_input = lifetime
            else:
                lifetime.close()

    def resolve_prompt_indices(self, texts: list[str], prompt_indices: tuple[int, ...] | None) -> tuple[int, ...]:
        """Assign prompt indices when omitted and validate explicitly supplied indices."""
        indices = (
            tuple(range(self.next_prompt_index, self.next_prompt_index + len(texts)))
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
        first_samples: tuple[int, ...],
        n: int,
        routes: tuple[str, ...] | None,
    ) -> tuple[RequestIdentity, ...]:
        """Sample j of prompt p gets sample index `first_samples[p] + j`."""
        samples = (
            (index, first + j, route)
            for index, first, route in zip(
                indices, first_samples, routes if routes is not None else ("",) * len(indices), strict=True
            )
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
        completion_device: torch.device | None,
        lifetime: ExitStack,
        samples_per_prompt: int,
    ) -> torch.Tensor | None:
        destination = None
        completion_wire = None
        if completion_device is not None:
            output_device = completion_device
            if output_device.type == "cuda":
                if output_device == torch.device("cuda"):
                    output_device = self.device
                if output_device != self.device:
                    raise ValueError("completion IPC currently requires the model's CUDA device")
            elif output_device.type != "cpu":
                raise ValueError("completion output requires CPU or CUDA storage")
            field = TensorFieldSpec(LayoutFieldId(0), (len(identities), self.hidden_size), TensorDType.FLOAT32, 4)
            with torch.inference_mode(False):
                if self._runner is None:
                    destination, completion_wire = SharedTensor.allocate(field, lifetime, output_device)
                else:
                    destination = torch.empty(field.shape, device=output_device, dtype=torch.float32)
        if self._runner is None:
            tp = self.require_llm().llm_engine.vllm_config.parallel_config.tensor_parallel_size
            batch_wire = map_host_batch(
                batch, lambda tensor: SharedTensor.export(tensor.cpu() if tp > 1 else tensor, lifetime)
            )
            self.command(
                BackendCommand.STAGE,
                encode_boundary(BackendStage(identities, batch_wire, completion_wire, samples_per_prompt)),
            )
        else:
            batch = map_host_batch(batch, lambda tensor: tensor.to(self.device))
            self.runner.require_session().stage_input(
                batch, identities, destination, samples_per_prompt=samples_per_prompt
            )
        return destination


def _one_per_sample(params: SamplingParams, first_sample: int) -> list[SamplingParams]:
    """vLLM's own split of n samples into requests: n=1 each, and with a seed, seed + i for sample index i (vLLM's
    seed + j when the first sample index is 0), so distinct samples of one prompt never share a seed."""
    samples: list[SamplingParams] = []
    for j in range(params.n):
        sample = params.clone()
        sample.n = 1
        if params.seed is not None:
            sample.seed = params.seed + first_sample + j
        samples.append(sample)
    return samples


def _per_prompt(outputs: list[RequestOutput], n: int) -> list[RequestOutput]:
    """One RequestOutput per prompt holding its n completions, as vLLM returns for SamplingParams(n=...)."""
    merged = outputs[::n]
    for start, output in zip(range(0, len(outputs), n), merged, strict=True):
        output.outputs = [replace(sample.outputs[0], index=j) for j, sample in enumerate(outputs[start : start + n])]
    return merged


def free_cuda_garbage() -> None:
    """Collect until no cycles remain (finalizers free further cycles), then return cached CUDA memory."""
    while gc.collect():
        pass
    if torch.cuda.is_initialized():
        clear_blas_workspaces()
        torch.cuda.empty_cache()


def vllm_engine(
    config: ModelConfig, workload_options: dict[str, Any], environment: dict[str, str], *, cuda_graphs: bool
) -> tuple[LLM, tuple[int, ...]]:
    """Build the vLLM engine; a workload adds its worker, scheduler and graph options.

    Also returns the Dynamo bytecode hooks an in-process model registered while loading (Qwen3 registers two);
    vLLM never removes them, so the handle's close() does.
    """
    environment = {
        _MP_ENV: "1" if config.multiprocessing else "0",
        _V2_RUNNER_ENV: "1" if config.runner == "mrv2" else "0",
        **environment,
    }
    if config.tensor_parallel_size > 1:
        if cuda_graphs:
            workload_options = {**graph_options(config, BackendRequirements()), **workload_options}
        environment.update(VLLM_USE_V2_MODEL_RUNNER="1", VLLM_WORKER_MULTIPROC_METHOD="spawn")
    previous = {name: os.environ.get(name) for name in environment}
    conflicts = tuple(name for name, value in environment.items() if previous[name] not in (None, value))
    if conflicts:
        raise RuntimeError(
            f"ModelConfig conflicts with {', '.join(conflicts)}; unset these variables or choose matching runner/multiprocessing settings"
        )
    os.environ.update(environment)
    # vLLM's startup profile asserts that free memory does not grow while it measures, and it collects garbage
    # mid-profile: CUDA tensors kept alive only by cycles (say, a caught exception's traceback) must go first.
    free_cuda_garbage()
    hooks = bytecode_hooks()
    try:
        llm = LLM(
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
    return llm, tuple(bytecode_hooks() - hooks)


def create_vllm_llm(
    workload: BackendWorkload[DataT, ResultT],
    config: ModelConfig,
    *,
    definition: str | None = None,
) -> VllmLlm[DataT, ResultT]:
    requirements = workload.backend_requirements()
    eager_reason = admit_workload(config, requirements)
    if config.cuda_graphs == "auto" and eager_reason:
        _logger.warning("running the model without CUDA graphs: %s", eager_reason)
    environment = {
        TAPS_ENV: encode_boundary(HiddenTapAnnouncement.declared(requirements, config)),
        POSITION_ENV: encode_boundary(PositionAnnouncement(requirements.position_states)),
        LAYOUT_ENV: encode_boundary(requirements.kv_layouts),
    }
    if config.multiprocessing or config.tensor_parallel_size > 1:
        environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    options: GraphOptions = {} if eager_reason else graph_options(config, requirements)
    # vLLM keys its compiled-model cache by its own config; the hidden taps and position states pllm adds to the
    # forward must enter that key, or an engine loads a model compiled for another workload. pllm's entries are
    # integers (overlay_graph decodes them as dict[str, int]), so the digest enters as one.
    forward = sha256((environment[TAPS_ENV] + environment[POSITION_ENV] + environment[LAYOUT_ENV]).encode()).hexdigest()
    options["additional_config"] = {**options.get("additional_config", {}), "pllm_forward": int(forward[:15], 16)}
    llm, hooks = vllm_engine(
        config,
        {
            **options,
            **load_options(requirements.checkpoint_names),
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
        # Until the workload handle exists, a plain handle owns the engine and releases it on failure.
        construction.callback(VllmLlm[None, None](llm, compile_hooks=hooks).close)
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
            budget = min(probe.state_page_budget_bytes for probe in probes)
            facts = replace(probes[0], state_page_budget_bytes=budget)
            assert all(replace(probe, state_page_budget_bytes=budget) == facts for probe in probes), (
                "TP workers disagree on model or planning facts"
            )
            facts, declaration = workload.specialize(facts, config.execution)
            attachment = workload.plan_backend(facts, declaration)
            policy = CachePolicy.for_attachment(attachment, config.generated_kv)
            handle = VllmLlm[DataT, ResultT](
                llm, None, attachment, facts, policy, requirements.sample_limit, eager_reason, compile_hooks=hooks
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
            deployment = workload.bind_backend(
                facts,
                workload.plan_backend(facts, declaration),
                declaration,
                map_host=get_accelerator_view_from_cpu_tensor,
            )
            policy = CachePolicy.for_attachment(deployment.attachment, config.generated_kv)
            runner.attach_workload(deployment, policy)
            handle = VllmLlm[DataT, ResultT](
                llm,
                runner,
                deployment.attachment,
                facts,
                policy,
                requirements.sample_limit,
                eager_reason,
                compile_hooks=hooks,
            )
        construction.pop_all()
        return handle


def probe_backend(
    requirements: BackendRequirements,
    runner: WorkloadModelRunner | Mrv1Runner,
    *,
    publish_lag: int = 1,
) -> BackendCompileFacts:
    shape = ModelShape.of_model(runner.decoder, requirements.parameter_targets, runner.dtype)
    free_bytes, _total_bytes = torch.cuda.mem_get_info(runner.device)
    dtypes: dict[torch.dtype, ModelDType] = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    config = ModelConfig(
        runner.model_config.model,
        revision=runner.model_config.revision,
        dtype=dtypes[runner.dtype],
        max_model_len=runner.model_config.max_model_len,
        max_num_seqs=runner.max_num_reqs,
        max_num_batched_tokens=runner.max_num_tokens,
        seed=runner.model_config.seed,
        runner=runner.runner_kind,
        tensor_parallel_size=runner.parallel_config.tensor_parallel_size,
        execution=ExecutionOptions(publish_lag=publish_lag),
    )
    return compile_facts(
        shape,
        requirements,
        config,
        model_identity=runner.decoder.identity,
        hidden_layouts=runner.hidden_layouts,
        overlay_graph_bytes=runner.overlay_graph_bytes,
        memory_bytes=free_bytes,
        blas_workspace_bytes=blas_workspace_bytes(),
        checkpoint=checkpoint_directory(config.model, config.revision) if requirements.checkpoint_names else "",
    )


__all__ = [
    "VllmLlm",
    "create_vllm_llm",
]
