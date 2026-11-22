"""Async per-request engine handle and OpenAI-compatible HTTP serving on vLLM."""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Iterable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field, replace
from itertools import count
from types import TracebackType
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import torch
from fastapi import FastAPI
from uvicorn import Config as UvicornConfig, Server as UvicornServer
from vllm import RequestOutput, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.sampling_params import RequestOutputKind
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.output_processor import RequestOutputCollector
from vllm.v1.engine.parallel_sampling import ParentRequest

from pllm.backends.vllm.compat import admit_parallel, build_openai_server_app, init_openai_server_state
from pllm.backends.vllm.control import (
    CONTROL_KEY,
    BackendCompletions,
    BackendImport,
    BackendOnlineImport,
    BackendTrain,
    completion_hidden,
    import_versions,
    reported_result,
    train_result,
    unwrap_feeds,
)
from pllm.backends.vllm.prefix_cache import CachePolicy, OnlineRequest, read_online
from pllm.backends.vllm.runner import BackendInstall, BackendInstalled, BackendWorkerState
from pllm.backends.vllm.runtime import (
    WORKER_CLASS,
    BackendCommand,
    engine_environment,
    free_cuda_garbage,
    workload_engine_setup,
)
from pllm.config import ModelConfig
from pllm.contracts.backend_attachment import BackendAttachment
from pllm.contracts.codec import decode_boundary, encode_boundary
from pllm.contracts.feedback import Completion, CompletionBatch, FinishReason
from pllm.contracts.host_batch import HostBatch, StateImport
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.semantic_graph import TaskId
from pllm.execution.compile_facts import BackendCompileFacts, BackendRequirements
from pllm.execution.host_runtime import CompletionInput
from pllm.execution.state_execution import StateDiagnostics
from pllm.execution.state_feed import FanoutFeed, Feed
from pllm.execution.step_driver import TrainStep
from pllm.execution.trace import TraceSelection, TraceSnapshot

if TYPE_CHECKING:
    from pllm.workload import Workload

_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Submission:
    """Worker inputs and internal child ids for one external async request."""

    requests: tuple[tuple[str, int], ...]
    prompt_index: int
    identities: dict[int, RequestIdentity] = field(default_factory=dict[int, RequestIdentity])

    def stamp(self, request: EngineCoreRequest, index: int) -> None:
        params = request.sampling_params
        assert params is not None
        online = read_online(params) or OnlineRequest(sample_index=index, prompt_index=self.prompt_index)
        self.identities[index] = RequestIdentity(
            request.request_id, online.prompt_index, online.sample_index, online.route
        )
        params = params.clone()
        params.extra_args = {**(params.extra_args or {}), CONTROL_KEY: self.requests[index][0]}
        request.sampling_params = params


class _WorkloadAsyncLLM(AsyncLLM):
    """AsyncLLM subclass stamping pLLM cache policy and online identity onto every outgoing request."""

    pllm_policy: CachePolicy | None = None
    pllm_sample_limit: int | None = None
    pllm_requires_route: bool = False
    pllm_has_route_state: bool = False
    _pllm_prompt_counter: count[int] | None = None
    _pllm_parent_prompts: dict[str, tuple[int, str]] | None = None
    _pllm_hooked: bool = False

    _submissions: dict[str, _Submission] | None = None

    @asynccontextmanager
    async def submission(
        self, request_id: str, samples: int, update: BackendImport | None, *, capture: bool
    ) -> AsyncGenerator[_Submission]:
        """Stage each child's control input before vLLM allocates its internal request id."""
        self._hook_requests()
        if self._submissions is None:
            self._submissions = {}
        if request_id in self._submissions:
            raise ValueError(f"duplicate online request {request_id}")
        keys = tuple(uuid4().hex for _ in range(samples))
        requests = tuple((key, 0) for key in keys)
        submission = _Submission(requests, self._allocate_prompt_index() if self.pllm_policy is None else 0)
        self._submissions[request_id] = submission
        try:
            if update is not None:
                await self._stage_command(BackendCommand.STAGE_ONLINE_IMPORT, BackendOnlineImport(keys, update))
            if capture:
                await self._stage_command(BackendCommand.STAGE_COMPLETIONS, BackendCompletions(requests))
            yield submission
        except BaseException:
            if capture:
                await self.collective_rpc(
                    "workload_command",
                    timeout=30.0,
                    args=(
                        BackendCommand.FINISH_COMPLETIONS.value,
                        encode_boundary(BackendCompletions(requests, discard=True)),
                    ),
                )
            raise
        finally:
            del self._submissions[request_id]
            if update is not None:
                await self._stage_command(BackendCommand.STAGE_ONLINE_IMPORT, BackendOnlineImport(keys, None))

    async def _stage_command(self, command: BackendCommand, payload: BackendOnlineImport | BackendCompletions) -> None:
        replies = await self.collective_rpc(
            "workload_command", timeout=30.0, args=(command.value, encode_boundary(payload))
        )
        for reply in replies:
            reported_result(reply, type(None))

    def bind_online_policy(
        self,
        *,
        policy: CachePolicy,
        sample_limit: int | None,
        requires_route: bool,
        has_route_state: bool,
    ) -> None:
        self.pllm_policy = policy
        self.pllm_sample_limit = sample_limit
        self.pllm_requires_route = requires_route
        self.pllm_has_route_state = has_route_state
        self._pllm_prompt_counter = count()
        self._pllm_parent_prompts = {}
        self._hook_requests()

    def _hook_requests(self) -> None:
        if not self._pllm_hooked:
            register_request = self.output_processor.add_request

            def add_request(
                request: EngineCoreRequest,
                prompt: str | None,
                parent_req: ParentRequest | None = None,
                request_index: int = 0,
                queue: RequestOutputCollector | None = None,
            ) -> None:
                if request.request_id not in self.output_processor.request_states:
                    self._stamp_request(request, parent_req, request_index)
                    if self._submissions:
                        submission = self._submissions.get(request.external_req_id or request.request_id)
                        if submission is not None:
                            submission.stamp(request, request_index)
                register_request(request, prompt, parent_req, request_index, queue)

            self.output_processor.add_request = add_request
            self._pllm_hooked = True

    def _allocate_prompt_index(self) -> int:
        if self._pllm_prompt_counter is None:
            self._pllm_prompt_counter = count()
        return next(self._pllm_prompt_counter)

    def _stamp_request(
        self,
        request: EngineCoreRequest,
        parent_req: ParentRequest | None,
        index: int,
    ) -> None:
        policy = self.pllm_policy
        if policy is None:
            return
        params = request.sampling_params
        if params is None:
            raise ValueError("online serving supports sampling generation requests only")
        extras = params.extra_args or {}
        raw_route = extras.get("pllm_route", "")
        if not isinstance(raw_route, str):
            raise ValueError("pllm_route in extra_args must be a string")
        if raw_route and not self.pllm_has_route_state:
            raise ValueError('routes= needs a request state keyed by "route" and one value per prompt')
        if self.pllm_requires_route and not raw_route:
            raise ValueError("a route-keyed request state requires per-prompt routes")
        requested = index + 1 if parent_req is None else max(index + 1, parent_req.n)
        if self.pllm_sample_limit is not None and requested > self.pllm_sample_limit:
            raise ValueError(
                f"sample index {requested - 1}: this workload reads pllm.sample_index(limit={self.pllm_sample_limit}); "
                f"keep n at most {self.pllm_sample_limit} or raise the workload's limit"
            )
        if self._pllm_parent_prompts is None:
            self._pllm_parent_prompts = {}
        if parent_req is not None:
            prompt_index, _ = self._pllm_parent_prompts.setdefault(
                parent_req.request_id, (self._allocate_prompt_index(), raw_route)
            )
            if index + 1 == parent_req.n:
                self._pllm_parent_prompts.pop(parent_req.request_id, None)
        else:
            prompt_index = self._allocate_prompt_index()
        request.sampling_params = policy.online_sampling(
            params, OnlineRequest(route=raw_route, sample_index=index, prompt_index=prompt_index)
        )


@dataclass(slots=True)
class AsyncVllmLlm:
    """Async continuous-batching engine handle returned by `pllm.make_async_llm`."""

    engine: _WorkloadAsyncLLM
    _config: ModelConfig
    _requirements: BackendRequirements | None = None
    _definition: str = ""
    _eager_reason: str = ""
    backend_attachment: BackendAttachment | None = None
    compile_facts: BackendCompileFacts | None = None
    cache_policy: CachePolicy | None = None
    _started: bool = False
    _closed: bool = False
    _start_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _request_ids: count[int] = field(default_factory=count)

    @property
    def attachment(self) -> BackendAttachment:
        """The installed workload's plan; workload-only operations require one."""
        self.require_open()
        if self.backend_attachment is None:
            raise ValueError("pass workload= to make_llm to use workload operations")
        return self.backend_attachment

    @property
    def hidden_size(self) -> int:
        self.require_open()
        return (
            self.compile_facts.probe.hidden_size
            if self.compile_facts is not None
            else self.engine.model_config.get_hidden_size()
        )

    @property
    def vocab_size(self) -> int:
        self.require_open()
        return (
            self.compile_facts.vocab_size
            if self.compile_facts is not None
            else self.engine.model_config.get_vocab_size()
        )

    def require_open(self) -> _WorkloadAsyncLLM:
        if self._closed:
            raise RuntimeError("model handle is closed")
        return self.engine

    async def start(self) -> AsyncVllmLlm:
        """Probe workers and install the workload if one was given."""
        async with self._start_lock:
            self.require_open()
            if self._started:
                return self
            if self._requirements is None:
                self._started = True
                return self
            replies = await self.engine.collective_rpc(
                "workload_command",
                timeout=30.0,
                args=(BackendCommand.PROBE.value, encode_boundary(self._requirements)),
            )
            probes = [decode_boundary(reply, BackendCompileFacts) for reply in replies]
            budget = min(probe.state_page_budget_bytes for probe in probes)
            facts = replace(probes[0], state_page_budget_bytes=budget)
            assert all(replace(probe, state_page_budget_bytes=budget) == facts for probe in probes), (
                "TP workers disagree on model or planning facts"
            )
            install = BackendInstall(facts, self._config.execution, self._definition, self._config.generated_kv)
            replies = await self.engine.collective_rpc(
                "workload_command",
                timeout=30.0,
                args=(BackendCommand.INSTALL.value, encode_boundary(install)),
            )
            installed, *others = (reported_result(reply, BackendInstalled) for reply in replies)
            assert all(other == installed for other in others), "TP workers planned different workloads"
            self.backend_attachment = installed.attachment
            self.compile_facts = installed.facts
            self.cache_policy = installed.cache_policy
            has_route = any(
                RequestKeyField.ROUTE in state.key_fields for state in installed.attachment.plan.semantic_graph.states
            )
            self.engine.bind_online_policy(
                policy=installed.cache_policy,
                sample_limit=self._requirements.sample_limit,
                requires_route=has_route,
                has_route_state=has_route,
            )
            self._started = True
            return self

    async def __aenter__(self) -> AsyncVllmLlm:
        return await self.start()

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        try:
            await self.close()
        except Exception as cleanup:
            if exc is None:
                raise
            exc.add_note(f"model cleanup also failed: {cleanup}")

    async def stream(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        *,
        request_id: str | None = None,
        route: str = "",
        state_import: StateImport[torch.Tensor] | None = None,
    ) -> AsyncIterator[RequestOutput]:
        """Submit one asynchronous request and yield RequestOutput chunks as tokens generate."""
        if not prompt:
            raise ValueError("provide non-empty prompts")
        await self.start()
        params = SamplingParams() if sampling_params is None else sampling_params.clone()
        if route:
            params.extra_args = {**(params.extra_args or {}), "pllm_route": route}
        req_id = request_id or f"pllm-{next(self._request_ids)}"
        if state_import is None:
            async for output in self._outputs(prompt, params, req_id):
                yield output
            return
        self.attachment
        async with self.engine.submission(req_id, params.n, BackendImport.of(state_import), capture=False):
            async for output in self._outputs(prompt, params, req_id):
                yield output

    async def _outputs(self, prompt: str, params: SamplingParams, req_id: str) -> AsyncIterator[RequestOutput]:
        finished = False
        try:
            async for output in self.engine.generate(prompt, params, req_id):
                finished = output.finished
                yield output
        finally:
            if not finished and not self._closed:
                await self.engine.abort(req_id)

    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        *,
        request_id: str | None = None,
        route: str = "",
        state_import: StateImport[torch.Tensor] | None = None,
    ) -> RequestOutput:
        """Run one asynchronous request to completion and return its final RequestOutput."""
        params = SamplingParams() if sampling_params is None else sampling_params.clone()
        params.output_kind = RequestOutputKind.FINAL_ONLY
        last: RequestOutput | None = None
        async for output in self.stream(prompt, params, request_id=request_id, route=route, state_import=state_import):
            last = output
        assert last is not None
        return last

    async def generate_completions(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams,
        *,
        group_id: str,
        routes: tuple[str, ...] | None = None,
        completion_device: torch.device | None = None,
    ) -> CompletionInput:
        """Generate each prompt's samples and collect their final hidden rows in prompt order."""
        texts = [prompts] if isinstance(prompts, str) else prompts
        if not group_id or not texts or any(not text for text in texts):
            raise ValueError("provide non-empty prompts and a non-empty group_id")
        if routes is not None and len(routes) != len(texts):
            raise ValueError("provide one route per prompt")
        await self.start()
        async with asyncio.TaskGroup() as group:
            calls = [
                group.create_task(self._completion(prompt, sampling_params, group_id, route))
                for prompt, route in zip(texts, routes or ("",) * len(texts), strict=True)
            ]
        batches = [call.result() for call in calls]
        entries = tuple(entry for batch in batches for entry in batch.completions.entries)
        hidden = torch.cat([batch.hidden for batch in batches])
        return CompletionInput(CompletionBatch(entries), hidden.to(completion_device or torch.device("cpu")))

    async def _completion(self, prompt: str, params: SamplingParams, group_id: str, route: str) -> CompletionInput:
        key = f"pllm-{next(self._request_ids)}"
        async with self.engine.submission(key, params.n, None, capture=True) as submission:
            output = await self.generate(prompt, params, request_id=key, route=route)
            replies = await self.command(
                BackendCommand.FINISH_COMPLETIONS, encode_boundary(BackendCompletions(submission.requests))
            )
            hidden = completion_hidden(replies)
            entries = tuple(
                Completion(
                    submission.identities[sample.index],
                    sample.index,
                    group_id,
                    tuple(sample.token_ids),
                    sample.text,
                    FinishReason(sample.finish_reason),
                )
                for sample in sorted(output.outputs, key=lambda sample: sample.index)
            )
            return CompletionInput(CompletionBatch(entries), hidden)

    async def import_state(self, state_import: StateImport[torch.Tensor], *, route: str = "") -> None:
        """Publish instance or route state while the engine is serving."""
        import_versions(
            await self.command(BackendCommand.IMPORT_STATE, encode_boundary(BackendImport.of(state_import, route)))
        )

    async def train_batch(self, batch: HostBatch[torch.Tensor]) -> TrainStep:
        """Run a host batch update and return its loss and parameter snapshots."""
        return train_result(await self.command(BackendCommand.TRAIN_BATCH, encode_boundary(BackendTrain.of(batch))))

    async def feed(self, name: str) -> Feed | FanoutFeed:
        """Claim the producer endpoint on each worker."""
        return unwrap_feeds(await self.command(BackendCommand.FEED, encode_boundary(name)))

    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Cancel an in-flight request by ID."""
        await self.start()
        await self.engine.abort(request_id)

    async def command(self, command: BackendCommand, payload: str = "", *, timeout: float = 30.0) -> list[str]:
        """Send a workload command to every worker rank asynchronously."""
        await self.start()
        if command not in (BackendCommand.STAGE_COMPLETIONS, BackendCommand.FINISH_COMPLETIONS):
            self.attachment
        return await self.engine.collective_rpc("workload_command", timeout=timeout, args=(command.value, payload))

    async def diagnostics(self) -> StateDiagnostics:
        """Read exact workload health and vLLM KV capacity from workers."""
        replies = await self.command(BackendCommand.DIAGNOSTICS)
        states = [decode_boundary(reply, BackendWorkerState) for reply in replies]
        assert all(state.diagnostics.logical_counts() == states[0].diagnostics.logical_counts() for state in states), (
            "TP worker logical counters diverged",
            states,
        )
        return replace(states[0].diagnostics, eager_reason=self._eager_reason)

    async def trace_snapshots(self) -> tuple[TraceSnapshot, ...]:
        """Return every worker's trace in rank order."""
        return tuple(decode_boundary(reply, TraceSnapshot) for reply in await self.command(BackendCommand.TRACE))

    async def trace_snapshot(self) -> TraceSnapshot:
        return (await self.trace_snapshots())[0]

    async def profile_tasks(self, tasks: tuple[TaskId, ...]) -> None:
        await self.command(BackendCommand.PROFILE, encode_boundary(tasks))

    async def select_trace(self, selection: TraceSelection) -> None:
        await self.command(BackendCommand.SELECT_TRACE, encode_boundary(selection))

    async def close(self) -> None:
        """Detach workload resources and shut down worker processes."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._started and self.backend_attachment is not None:
                await self.engine.collective_rpc(
                    "workload_command", timeout=30.0, args=(BackendCommand.DETACH.value, encode_boundary(True))
                )
        finally:
            self.engine.shutdown()
            free_cuda_garbage()


def _async_engine_args(
    config: ModelConfig,
    engine_options: dict[str, Any],
    *,
    cuda_graphs: bool,
) -> AsyncEngineArgs:
    kv_transfer = (
        None
        if config.kv_offload is None
        else KVTransferConfig(
            kv_connector=config.kv_offload.connector,
            kv_role="kv_both",
            kv_connector_extra_config={
                "cpu_bytes_to_use": config.kv_offload.cpu_bytes,
                "blocks_per_chunk": 1,
                "offload_prompt_only": config.kv_offload.prompt_only,
            },
        )
    )
    return AsyncEngineArgs(
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
        kv_cache_memory_bytes=config.kv_cache_memory_bytes,
        seed=config.seed,
        trust_remote_code=config.trust_remote_code,
        enable_prefix_caching=config.enable_prefix_caching,
        enable_chunked_prefill=config.enable_chunked_prefill,
        kv_transfer_config=kv_transfer,
        **engine_options,
    )


def create_async_llm(
    workload: Workload | None,
    config: ModelConfig,
    *,
    definition: str = "",
) -> AsyncVllmLlm:
    """Build an AsyncVllmLlm handle for continuous per-request serving."""
    config.uses_multiprocessing(async_engine=True)
    if workload is None:
        admit_parallel(config)
        requirements = None
        eager_reason = ""
        cuda_graphs = config.cuda_graphs is not False
        engine_options: dict[str, Any] = {"worker_cls": WORKER_CLASS}
        environment: dict[str, str] = {}
    else:
        requirements = workload.backend_requirements()
        eager_reason, engine_options, environment = workload_engine_setup(config, requirements)
        cuda_graphs = not eager_reason
    with engine_environment(config, environment, async_engine=True):
        engine_args = _async_engine_args(config, engine_options, cuda_graphs=cuda_graphs)
        vllm_config = engine_args.create_engine_config()
        engine = _WorkloadAsyncLLM.from_vllm_config(vllm_config, start_engine_loop=True)
    assert isinstance(engine, _WorkloadAsyncLLM)
    return AsyncVllmLlm(engine, config, requirements, definition, eager_reason)


def _parse_openai_args(
    model: str,
    *,
    host: str,
    port: int,
    served_model_name: str | None,
    api_key: str | None,
    chat_template: str | None,
    uvicorn_log_level: str,
) -> argparse.Namespace:
    cli_tokens = ["serve", model, "--host", host, "--port", str(port), "--uvicorn-log-level", uvicorn_log_level]
    if served_model_name is not None:
        cli_tokens.extend(["--served-model-name", served_model_name])
    if api_key is not None:
        cli_tokens.extend(["--api-key", api_key])
    if chat_template is not None:
        cli_tokens.extend(["--chat-template", chat_template])
    parser = FlexibleArgumentParser()
    ServeSubcommand().subparser_init(parser.add_subparsers())  # pyright: ignore[reportUnknownMemberType]
    args = parser.parse_args(cli_tokens)
    assert args is not None
    args.model = model
    return args


async def build_openai_app(
    llm: AsyncVllmLlm,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    served_model_name: str | None = None,
    api_key: str | None = None,
    chat_template: str | None = None,
    uvicorn_log_level: str = "info",
) -> tuple[FastAPI, AsyncExitStack]:
    """Build vLLM's OpenAI-compatible FastAPI application backed by `llm`."""
    await llm.start()
    args = _parse_openai_args(
        llm.engine.model_config.model,
        host=host,
        port=port,
        served_model_name=served_model_name,
        api_key=api_key,
        chat_template=chat_template,
        uvicorn_log_level=uvicorn_log_level,
    )
    supported_tasks = await llm.engine.get_supported_tasks()
    app = build_openai_server_app(args, supported_tasks, llm.engine.model_config)
    stack = AsyncExitStack()
    await init_openai_server_state(llm.engine, app.state, args, supported_tasks)
    app.state.server = UvicornServer(UvicornConfig(app, host=host, port=port, log_level=uvicorn_log_level))
    return app, stack


async def serve_openai_llm(
    llm: AsyncVllmLlm,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    served_model_name: str | None = None,
    api_key: str | None = None,
    chat_template: str | None = None,
    uvicorn_log_level: str = "info",
) -> None:
    """Start and serve an OpenAI-compatible HTTP endpoint for an initialized `AsyncVllmLlm`."""
    async with llm:
        app, stack = await build_openai_app(
            llm,
            host=host,
            port=port,
            served_model_name=served_model_name,
            api_key=api_key,
            chat_template=chat_template,
            uvicorn_log_level=uvicorn_log_level,
        )
        async with stack:
            server = UvicornServer(UvicornConfig(app, host=host, port=port, log_level=uvicorn_log_level))
            app.state.server = server
            await server.serve()


__all__ = [
    "AsyncVllmLlm",
    "build_openai_app",
    "create_async_llm",
    "serve_openai_llm",
]
