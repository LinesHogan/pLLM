"""Strict supported subset of vLLM's V1 AsyncLLM interface for online serving."""

from collections.abc import AsyncGenerator, Callable, Iterable, Mapping
from typing import Literal, Protocol, overload

from vllm import RequestOutput, SamplingParams
from vllm.config import ModelConfig as VllmModelConfig, VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.tasks import SupportedTask
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.output_processor import RequestOutputCollector
from vllm.v1.engine.parallel_sampling import ParentRequest

class _OutputProcessor(Protocol):
    request_states: Mapping[str, object]
    add_request: Callable[
        [EngineCoreRequest, str | None, ParentRequest | None, int, RequestOutputCollector | None],
        None,
    ]

class AsyncLLM(EngineClient):
    vllm_config: VllmConfig
    model_config: VllmModelConfig
    output_processor: _OutputProcessor

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
    ) -> "AsyncLLM": ...
    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]: ...
    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
    ) -> AsyncGenerator[RequestOutput, None]: ...
    async def abort(self, request_id: str | Iterable[str], internal: bool = False) -> None: ...
    def shutdown(self, timeout: float | None = None) -> None: ...
    @overload
    async def collective_rpc(
        self, method: Literal["workload_command"], *, timeout: float | None = None, args: tuple[str, str] = ...
    ) -> list[str]: ...
    @overload
    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> list[str]: ...
