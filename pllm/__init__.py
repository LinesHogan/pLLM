"""pllm: write a workload, then run it inside vLLM.

Membership rule: the top level holds the vocabulary for writing a workload (Workload, every decorator and input
marker, state types and pllm.Field, optimizers, Phase, Attr, UpdateOutput, pllm.scan) and the entry point
make_llm with ModelConfig and ExecutionOptions. Domain objects live in sub-namespaces: pllm.model (a place in,
or a change to, the base model's computation), pllm.host (host loops between generate calls), pllm.io (data
crossing pllm's boundary outside generate's arguments) and pllm.testing (stepping workloads without GPU/vLLM).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, overload

from pllm import host as host, io as io, model as model  # noqa: E402
from pllm.authoring import AdamW as AdamW, Dimensions as Dimensions
from pllm.config import (
    ExecutionOptions as ExecutionOptions,
    KVOffload as KVOffload,
    ModelConfig as ModelConfig,
)
from pllm.execution.scan import scan as scan
from pllm.usage import (
    SGD as SGD,
    Attr as Attr,
    Field as Field,
    InstanceState as InstanceState,
    Phase as Phase,
    PositionState as PositionState,
    RequestState as RequestState,
    UpdateOutput as UpdateOutput,
    Workload as Workload,
    add_logits as add_logits,
    base_logits as base_logits,
    batch as batch,
    carry as carry,
    causal_model as causal_model,
    custom_forward as custom_forward,
    custom_residual as custom_residual,
    depth as depth,
    differentiate as differentiate,
    export as export,
    final_hidden as final_hidden,
    guide_sampling as guide_sampling,
    hidden as hidden,
    index as index,
    layer_logits as layer_logits,
    module as module,
    on as on,
    operand as operand,
    parameter as parameter,
    residual as residual,
    result as result,
    sample_index as sample_index,
    sampled_tokens as sampled_tokens,
    train as train,
    train_in_forward as train_in_forward,
    value as value,
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
    "custom_forward",
    "custom_residual",
    "depth",
    "value",
    "ModelConfig",
    "ExecutionOptions",
    "KVOffload",
    "Workload",
    "Phase",
    "on",
    "write",
    "export",
    "add_logits",
    "guide_sampling",
    "train",
    "differentiate",
    "train_in_forward",
    "Attr",
    "hidden",
    "residual",
    "layer_logits",
    "final_hidden",
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
    "Field",
    "UpdateOutput",
    "host",
    "model",
    "io",
]
