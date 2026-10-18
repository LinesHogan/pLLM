"""Strict supported subset of the vLLM 0.27.1 offline LLM boundary."""

from typing import Literal

from vllm import RequestOutput, SamplingParams
from vllm.v1.engine.llm_engine import LLMEngine


class LLM:
    llm_engine: LLMEngine

    def __init__(
        self,
        model: str,
        *,
        revision: str | None,
        dtype: Literal["bfloat16", "float16", "float32"],
        tensor_parallel_size: Literal[1],
        pipeline_parallel_size: Literal[1],
        data_parallel_size: Literal[1],
        distributed_executor_backend: Literal["uni"],
        worker_cls: str,
        enforce_eager: bool,
        max_model_len: int,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        gpu_memory_utilization: float,
        seed: int,
        trust_remote_code: bool,
        enable_prefix_caching: bool,
        enable_chunked_prefill: Literal[False],
        async_scheduling: bool | None = None,
    ) -> None: ...

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams,
        *,
        use_tqdm: bool,
    ) -> list[RequestOutput]: ...

    def enqueue(self, prompts: str | list[str], sampling_params: SamplingParams,
                *, use_tqdm: bool) -> list[str]: ...

    def wait_for_completion(self, output_type: type[RequestOutput],
                            *, use_tqdm: bool) -> list[RequestOutput]: ...

    def collective_rpc(self, method: Literal["workload_command"], *, timeout: float,
                       args: tuple[str, str]) -> list[str]: ...

    def sleep(self, level: Literal[0], mode: Literal["keep"]) -> None: ...

    def wake_up(self, tags: list[Literal["scheduling"]]) -> None: ...
