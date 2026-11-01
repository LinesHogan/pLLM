"""Workload authoring vocabulary and the model entry point; host loops start from pllm.host.make_host."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, overload

from pllm import host as host, model as model, state as state  # noqa: E402
from pllm.authoring import AdamW as AdamW, Dimensions as Dimensions
from pllm.config import (
    ExecutionOptions as ExecutionOptions,
    KVOffload as KVOffload,
    ModelConfig as ModelConfig,
)
from pllm.execution.scan import scan as scan
from pllm.execution.token_program import LayerOutputs as LayerOutputs, Layers as Layers
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
    UpdateOutput as UpdateOutput,
    Visibility as Visibility,
    Workload as Workload,
    add_logits as add_logits,
    base_logits as base_logits,
    batch as batch,
    carry as carry,
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
    sample_index as sample_index,
    sampled_tokens as sampled_tokens,
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


@overload
def make_llm(config: ModelConfig, *, workload: BackendWorkload[D, R]) -> VllmLlm[D, R]: ...


@overload
def make_llm(config: ModelConfig, *, workload: None = None) -> LLM: ...


def make_llm(config: ModelConfig, *, workload: BackendWorkload[D, R] | None = None) -> LLM | VllmLlm[D, R]:
    """A vLLM engine for `config`, running `workload` inside it; plain vLLM `LLM` without a workload."""
    if workload is None:
        if config.execution != ExecutionOptions():
            raise ValueError("workload execution options require a workload")
        from pllm.backends.vllm.compat import admit_parallel
        from pllm.backends.vllm.runtime import vllm_engine

        admit_parallel(config)
        return vllm_engine(config, {"worker_cls": "auto"}, {}, cuda_graphs=config.cuda_graphs is not False)
    from pllm.backends.vllm.runtime import create_vllm_llm

    definition = None
    if config.multiprocessing or config.tensor_parallel_size > 1:
        from pllm.workload_transport import encode_workload

        definition = encode_workload(workload)
    return create_vllm_llm(workload, config, definition=definition)


__all__ = [
    "make_llm",
    "scan",
    "Layers",
    "LayerOutputs",
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
    "sample_index",
    "result",
    "base_logits",
    "sampled_tokens",
    "operand",
    "carry",
    "causal_model",
    "batch",
    "InstanceState",
    "RequestState",
    "PositionState",
    "AdamW",
    "SGD",
    "Dimensions",
    "Buffer",
    "UpdateOutput",
    "host",
    "model",
    "state",
]
