"""pllm: write a workload, then run it inside vLLM.

Membership rule: the top level holds the vocabulary for writing a workload (Workload, every decorator and input
marker, state types, their init= sources and pllm.Field, optimizers, Phase, UpdateOutput, pllm.scan) and the entry point
make_llm and explain with ModelConfig and ExecutionOptions. Domain objects live in sub-namespaces: pllm.model (a place in,
or a change to, the base model's computation), pllm.host (host loops between generate calls), pllm.io (data
crossing pLLM's boundary outside generate's arguments) and pllm.testing (checking and stepping workloads in tests).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pllm import host as host, io as io, model as model  # noqa: E402
from pllm.config import (
    ExecutionOptions as ExecutionOptions,
    KVOffload as KVOffload,
    ModelConfig as ModelConfig,
)
from pllm.execution.declaration import Dimensions as Dimensions
from pllm.execution.scan import scan as scan
from pllm.explanation import explain as explain
from pllm.kernels.adamw import AdamWConfig as AdamW
from pllm.usage import (
    SGD as SGD,
    Computed as Computed,
    Fed as Fed,
    Field as Field,
    FromModule as FromModule,
    Imported as Imported,
    InstanceState as InstanceState,
    ModelCheckpoint as ModelCheckpoint,
    Phase as Phase,
    PositionState as PositionState,
    Pretrained as Pretrained,
    RequestState as RequestState,
    UpdateOutput as UpdateOutput,
    Zeros as Zeros,
    add_logits as add_logits,
    attention as attention,
    base_logits as base_logits,
    batch as batch,
    carry as carry,
    causal_model as causal_model,
    cfg as cfg,
    custom_attention as custom_attention,
    custom_forward as custom_forward,
    custom_residual as custom_residual,
    depth as depth,
    differentiate as differentiate,
    export as export,
    final_hidden as final_hidden,
    guide_sampling as guide_sampling,
    hidden as hidden,
    index as index,
    input_tokens as input_tokens,
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
from pllm.workload import Workload as Workload

if TYPE_CHECKING:
    from pllm.backends.vllm.runtime import VllmLlm
    from pllm.backends.vllm.serving import AsyncVllmLlm


def make_llm(config: ModelConfig, *, workload: Workload | None = None) -> VllmLlm:
    """A context-managed vLLM engine; execution options apply when a workload is installed."""
    if workload is None:
        from pllm.backends.vllm.compat import admit_parallel
        from pllm.backends.vllm.runtime import WORKER_CLASS, VllmLlm, local_runner, vllm_engine

        admit_parallel(config)
        llm, hooks = vllm_engine(config, {"worker_cls": WORKER_CLASS}, {}, cuda_graphs=config.cuda_graphs is not False)
        return VllmLlm(llm, local_runner(llm, config), compile_hooks=hooks)
    from pllm.backends.vllm.runtime import create_vllm_llm

    definition = None
    if config.uses_multiprocessing():
        from pllm.workload_transport import encode_workload

        definition = encode_workload(workload)
    return create_vllm_llm(workload, config, definition=definition)


def make_async_llm(config: ModelConfig, *, workload: Workload | None = None) -> AsyncVllmLlm:
    """An async continuous-batching vLLM engine handle (`async with make_async_llm(...) as llm:`)."""
    from pllm.backends.vllm.serving import create_async_llm

    definition = ""
    if workload is not None:
        from pllm.workload_transport import encode_workload

        definition = encode_workload(workload)
    return create_async_llm(workload, config, definition=definition)


__all__ = [
    "make_llm",
    "make_async_llm",
    "explain",
    "scan",
    "custom_forward",
    "cfg",
    "custom_residual",
    "custom_attention",
    "attention",
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
    "hidden",
    "residual",
    "layer_logits",
    "final_hidden",
    "parameter",
    "module",
    "version",
    "index",
    "input_tokens",
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
    "Zeros",
    "Computed",
    "Pretrained",
    "ModelCheckpoint",
    "FromModule",
    "Fed",
    "Imported",
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
