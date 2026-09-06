"""Public tLLM API and the sole workload routing boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict, TypeVar, overload

if TYPE_CHECKING:
    from vllm import LLM
    from tllm.canonical.backend_program import BackendWorkload
    from tllm.canonical_vllm.runtime import CanonicalVllmConfig, CanonicalVllmLlm


_DataT = TypeVar("_DataT")
_ResultT = TypeVar("_ResultT")


class _EngineOptions(TypedDict, total=False):
    seed: int
    worker_extension_cls: str
    max_num_seqs: int
    max_num_batched_tokens: int
    max_cudagraph_capture_size: int
    max_lora_rank: int
    max_loras: int
    max_cpu_loras: int


@overload
def make_llm(*, workloads: tuple[BackendWorkload[_DataT, _ResultT], ...] | list[BackendWorkload[_DataT, _ResultT]],
             canonical_config: CanonicalVllmConfig) -> CanonicalVllmLlm[_DataT, _ResultT]: ...


@overload
def make_llm(
    model_name: str | None = None,
    dtype: Literal["auto", "half", "float16", "bfloat16", "float", "float32"] | None = None,
    gpu_memory_utilization: float | None = None, max_model_len: int | None = None,
    enable_prefix_caching: bool | None = None, enforce_eager: bool | None = None,
    seed: int | None = None, worker_extension_cls: str | None = None,
    max_num_seqs: int | None = None, max_num_batched_tokens: int | None = None,
    max_cudagraph_capture_size: int | None = None, enable_lora: bool | None = None,
    max_lora_rank: int | None = None, max_loras: int | None = None, max_cpu_loras: int | None = None,
    *, workloads: None = None, canonical_config: None = None,
) -> LLM: ...


def make_llm(
    model_name: str | None = None,
    dtype: Literal["auto", "half", "float16", "bfloat16", "float", "float32"] | None = None,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    enable_prefix_caching: bool | None = None,
    enforce_eager: bool | None = None,
    seed: int | None = None,
    worker_extension_cls: str | None = None,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    max_cudagraph_capture_size: int | None = None,
    enable_lora: bool | None = None,
    max_lora_rank: int | None = None,
    max_loras: int | None = None,
    max_cpu_loras: int | None = None,
    *,
    workloads: tuple[BackendWorkload[_DataT, _ResultT], ...] | list[BackendWorkload[_DataT, _ResultT]] | None = None,
    canonical_config: CanonicalVllmConfig | None = None,
) -> LLM | CanonicalVllmLlm[_DataT, _ResultT]:
    if workloads is not None:
        if not isinstance(workloads, (tuple, list)) or len(workloads) != 1:
            raise TypeError("canonical make_llm requires one workload")
        from tllm.canonical.backend_program import BackendWorkload

        if not isinstance(workloads[0], BackendWorkload):
            raise TypeError("workload does not declare a supported model-backend program")
        if any(
            value is not None
            for value in (
                model_name, dtype, gpu_memory_utilization, max_model_len,
                enable_prefix_caching, enforce_eager, seed, worker_extension_cls,
                max_num_seqs, max_num_batched_tokens, max_cudagraph_capture_size,
                enable_lora, max_lora_rank, max_loras, max_cpu_loras,
            )
        ):
            raise TypeError("canonical make_llm accepts engine options through canonical_config")
        from tllm.canonical_vllm.runtime import (
            CanonicalVllmConfig,
            create_canonical_vllm_llm,
        )

        if type(canonical_config) is not CanonicalVllmConfig:
            raise TypeError("canonical_config must be an exact CanonicalVllmConfig")
        return create_canonical_vllm_llm(workloads[0], canonical_config)

    if canonical_config is not None:
        raise TypeError("canonical_config requires a workload")
    if (
        model_name is None or dtype is None or gpu_memory_utilization is None
        or max_model_len is None or enable_prefix_caching is None
    ):
        raise TypeError(
            "plain make_llm requires model_name, dtype, gpu_memory_utilization, "
            "max_model_len and enable_prefix_caching"
        )
    from vllm import LLM

    options: _EngineOptions = {}
    if seed is not None:
        options["seed"] = seed
    if worker_extension_cls is not None:
        options["worker_extension_cls"] = worker_extension_cls
    if max_num_seqs is not None:
        options["max_num_seqs"] = max_num_seqs
    if max_num_batched_tokens is not None:
        options["max_num_batched_tokens"] = max_num_batched_tokens
    if max_cudagraph_capture_size is not None:
        options["max_cudagraph_capture_size"] = max_cudagraph_capture_size
    if max_lora_rank is not None:
        options["max_lora_rank"] = max_lora_rank
    if max_loras is not None:
        options["max_loras"] = max_loras
    if max_cpu_loras is not None:
        options["max_cpu_loras"] = max_cpu_loras
    return LLM(
        model=model_name,
        dtype=dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=enable_prefix_caching,
        enforce_eager=bool(enforce_eager),
        enable_lora=bool(enable_lora),
        **options,
    )


__all__ = [
    "make_llm",
]
