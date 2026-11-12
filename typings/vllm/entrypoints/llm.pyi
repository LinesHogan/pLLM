"""Strict supported subset of the vLLM 0.27.1 offline LLM boundary."""

from collections.abc import Callable
from typing import Literal, TypeVar, overload

from vllm import RequestOutput, SamplingParams
from vllm.config import CompilationConfig
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine.llm_engine import LLMEngine

_R = TypeVar("_R")

class LLM:
    llm_engine: LLMEngine

    def __init__(
        self,
        model: str,
        *,
        revision: str | None,
        dtype: Literal["bfloat16", "float16", "float32"],
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        data_parallel_size: int,
        distributed_executor_backend: Literal["uni", "mp"],
        worker_cls: str,
        enforce_eager: bool,
        max_model_len: int,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        gpu_memory_utilization: float,
        kv_cache_memory_bytes: int | None = None,
        seed: int,
        trust_remote_code: bool,
        enable_prefix_caching: bool,
        enable_chunked_prefill: bool,
        scheduler_cls: str | None = ...,
        async_scheduling: bool | None = None,
        additional_config: dict[str, int] | None = None,
        compilation_config: CompilationConfig | None = None,
        kv_transfer_config: dict[str, str | dict[str, int | bool]] | None = None,
        attention_config: dict[str, str] | None = None,
    ) -> None: ...
    def get_tokenizer(self) -> TokenizerLike: ...
    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | list[SamplingParams],
        *,
        use_tqdm: bool,
    ) -> list[RequestOutput]: ...
    def enqueue(
        self, prompts: str | list[str], sampling_params: SamplingParams | list[SamplingParams], *, use_tqdm: bool
    ) -> list[str]: ...
    def wait_for_completion(self, output_type: type[RequestOutput], *, use_tqdm: bool) -> list[RequestOutput]: ...
    @overload
    def collective_rpc(
        self, method: Literal["workload_command"], *, timeout: float, args: tuple[str, str]
    ) -> list[str]: ...
    @overload
    def collective_rpc(self, method: Callable[..., _R], *, args: tuple[str, ...]) -> list[_R]: ...
    def sleep(self, level: Literal[0], mode: Literal["keep"]) -> None: ...
    def wake_up(self, tags: list[Literal["scheduling"]]) -> None: ...
