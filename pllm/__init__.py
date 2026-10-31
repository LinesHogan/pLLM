"""Public factory for model-backed and host workloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, overload

from pllm import host as host, model as model, state as state  # noqa: E402
from pllm.authoring import AdamW as AdamW
from pllm.config import (
    ExecutionOptions as ExecutionOptions,
    KVOffload as KVOffload,
    ModelConfig as ModelConfig,
)
from pllm.execution.host_runtime import (
    HostRuntimeFacts,
    HostWorkload,
)
from pllm.execution.scan import scan as scan
from pllm.execution.token_program import Layers as Layers, Taps as Taps
from pllm.execution.trace import TraceEvent as TraceEvent
from pllm.usage import (
    SGD as SGD,
    Buffer as Buffer,
    Config as Config,
    Exact as Exact,
    InstanceState as InstanceState,
    Phase as Phase,
    PositionState as PositionState,
    RequestState as RequestState,
    TensorUpdate as TensorUpdate,
    Visibility as Visibility,
    Workload as Workload,
    add_logits as add_logits,
    base_logits as base_logits,
    batch as batch,
    capture as capture,
    causal_model as causal_model,
    differentiate as differentiate,
    export as export,
    guide_sampling as guide_sampling,
    hidden as hidden,
    index as index,
    layer_route as layer_route,
    module as module,
    on as on,
    operand as operand,
    overlay as overlay,
    parameter as parameter,
    result as result,
    sampled_tokens as sampled_tokens,
    samples as samples,
    train as train,
    version as version,
    write as write,
)

if TYPE_CHECKING:
    from vllm import LLM

    from pllm.authoring import BackendWorkload
    from pllm.backends.vllm.runtime import VllmLlm

D = TypeVar("D")
R = TypeVar("R")
C = TypeVar("C")


@overload
def make_llm(config: HostRuntimeFacts, *, workload: HostWorkload[C]) -> C: ...


@overload
def make_llm(config: ModelConfig, *, workload: BackendWorkload[D, R]) -> VllmLlm[D, R]: ...


@overload
def make_llm(config: ModelConfig, *, workload: None = None) -> LLM: ...


def make_llm(
    config: ModelConfig | HostRuntimeFacts, *, workload: BackendWorkload[D, R] | HostWorkload[C] | None = None
) -> LLM | VllmLlm[D, R] | C:
    from pllm.authoring import BackendWorkload
    from pllm.execution.host_runtime import HostWorkload

    if isinstance(config, HostRuntimeFacts):
        if not isinstance(workload, HostWorkload):
            raise TypeError("host execution requires a host workload")
        return workload.compile_host(config)
    if workload is None:
        if config.execution != ExecutionOptions():
            raise ValueError("workload execution options require a workload")
        from vllm import LLM

        return LLM(
            model=config.model,
            dtype=config.dtype,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            distributed_executor_backend="uni",
            worker_cls="auto",
            enable_chunked_prefill=config.enable_chunked_prefill,
            max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            gpu_memory_utilization=config.gpu_memory_utilization,
            seed=config.seed,
            enable_prefix_caching=config.enable_prefix_caching,
            enforce_eager=not config.cuda_graphs,
            trust_remote_code=config.trust_remote_code,
            revision=config.revision,
            kv_transfer_config=None if config.kv_offload is None else config.kv_offload.options(),
        )
    if not isinstance(workload, BackendWorkload):
        raise TypeError("model execution requires a backend workload")
    from pllm.backends.vllm.runtime import create_vllm_llm

    definition = None
    if config.multiprocessing:
        from pllm.workload_transport import encode_workload

        definition = encode_workload(workload)
    return create_vllm_llm(workload, config, definition=definition)


__all__ = [
    "make_llm",
    "scan",
    "Layers",
    "Taps",
    "layer_route",
    "Exact",
    "Visibility",
    "ModelConfig",
    "ExecutionOptions",
    "KVOffload",
    "Workload",
    "Phase",
    "on",
    "write",
    "export",
    "TraceEvent",
    "add_logits",
    "guide_sampling",
    "train",
    "differentiate",
    "overlay",
    "Config",
    "hidden",
    "parameter",
    "module",
    "version",
    "index",
    "samples",
    "result",
    "base_logits",
    "sampled_tokens",
    "operand",
    "capture",
    "causal_model",
    "batch",
    "InstanceState",
    "RequestState",
    "PositionState",
    "AdamW",
    "SGD",
    "Buffer",
    "TensorUpdate",
    "host",
    "model",
    "state",
]
