"""Public factory for model-backed and host workloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, overload

from pllm import host as host, model as model, state as state  # noqa: E402
from pllm.authoring import AdamW as AdamW
from pllm.config import (
    ExecutionOptions as ExecutionOptions,
    ModelConfig as ModelConfig,
)
from pllm.execution.host_runtime import (
    HostRuntimeFacts,
    HostWorkload,
)
from pllm.execution.scan import scan as scan
from pllm.usage import (
    SGD as SGD,
    BaseLogits as BaseLogits,
    Buffer as Buffer,
    Capture as Capture,
    Config as Config,
    Exact as Exact,
    Hidden as Hidden,
    InstanceState as InstanceState,
    Operand as Operand,
    Parameter as Parameter,
    Phase as Phase,
    RequestState as RequestState,
    Result as Result,
    SampledTokens as SampledTokens,
    Samples as Samples,
    StateIndex as StateIndex,
    StateVersion as StateVersion,
    TensorUpdate as TensorUpdate,
    Visibility as Visibility,
    Workload as Workload,
    add_logits as add_logits,
    differentiate as differentiate,
    export as export,
    guide_sampling as guide_sampling,
    hook as hook,
    on as on,
    overlay as overlay,
    train as train,
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
            dtype=config.dtype.value,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            distributed_executor_backend="uni",
            worker_cls="auto",
            enable_chunked_prefill=False,
            max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            gpu_memory_utilization=config.gpu_memory_utilization,
            seed=config.seed,
            enable_prefix_caching=config.enable_prefix_caching,
            enforce_eager=not config.cuda_graphs,
            trust_remote_code=config.trust_remote_code,
            revision=config.revision,
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
    "Exact",
    "Visibility",
    "ModelConfig",
    "ExecutionOptions",
    "Workload",
    "Phase",
    "on",
    "write",
    "hook",
    "export",
    "add_logits",
    "guide_sampling",
    "train",
    "Config",
    "Hidden",
    "Samples",
    "Parameter",
    "Capture",
    "StateIndex",
    "StateVersion",
    "Result",
    "InstanceState",
    "RequestState",
    "BaseLogits",
    "SampledTokens",
    "AdamW",
    "overlay",
    "Operand",
    "Buffer",
    "TensorUpdate",
    "differentiate",
    "SGD",
    "host",
    "model",
    "state",
]
