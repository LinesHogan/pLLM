"""Public factory for model-backed and host workloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, overload

from pllm.config import ExecutionOptions as ExecutionOptions, ModelConfig as ModelConfig
from pllm.execution.host_runtime import (
    HostRuntimeFacts as HostRuntimeFacts, HostWorkload as HostWorkload,
    HostTransactionCoordinator as HostTransactionCoordinator,
    TransactionHostBinding as TransactionHostBinding,
    CompletionInput as CompletionInput, NoHostRequest as NoHostRequest,
)
from pllm.authoring import AdamW as AdamW, HiddenSite as HiddenSite, CausalModel as CausalModel
from pllm.usage import (
    Workload as Workload, Phase as Phase, on as on, hook as hook, export as export, add_logits as add_logits,
    train as train, Config as Config, Hidden as Hidden, Samples as Samples, Parameter as Parameter,
    Result as Result, InstanceState as InstanceState, RequestState as RequestState,
    StateIndex as StateIndex, StateVersion as StateVersion, guide_sampling as guide_sampling,
    BaseLogits as BaseLogits, SampledTokens as SampledTokens,
    overlay as overlay, Operand as Operand, Buffer as Buffer, TensorUpdate as TensorUpdate,
    differentiate as differentiate,
    Activation as Activation,
    Recurrence as Recurrence,
    transaction as transaction,
    CompletionWorkload as CompletionWorkload,
    CausalAdapterWorkload as CausalAdapterWorkload,
    SGD as SGD, Batch as Batch,
)
from pllm.contracts.parameter_overlay import ParameterOverlayOperator as ParameterOverlayOperator, ParameterTarget as ParameterTarget
from pllm.contracts.request_identity import RequestKeyField as RequestKeyField
from pllm.contracts.semantic_graph import ConditionalLayerReplay as ConditionalLayerReplay, StateScope as StateScope
from pllm.contracts.backend_attachment import ActivationPatchOperator as ActivationPatchOperator
from pllm.contracts.auxiliary import (StateTensorUpdate as StateTensorUpdate, CausalTrainBatch as CausalTrainBatch,
    WeightedCEBatch as WeightedCEBatch)
from pllm.execution.codec import load_tensor_fields as load_tensor_fields
from pllm.execution.tensor_program import InvocationOutput as InvocationOutput
from pllm.execution.tensor_binding import TensorResult as TensorResult
from pllm.execution.ptts.page_bundle import PagePublication as PagePublication
from pllm.contracts.state_ref import StateRef as StateRef
from pllm.kernels.paged_linear import TensorRows as TensorRows, PagedRows as PagedRows

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


def make_llm(config: ModelConfig | HostRuntimeFacts, *,
             workload: BackendWorkload[D, R] | HostWorkload[C] | None = None) -> LLM | VllmLlm[D, R] | C:
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
        return LLM(model=config.model, dtype=config.dtype.value,
            tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
            distributed_executor_backend="uni", worker_cls="auto", enable_chunked_prefill=False,
            max_model_len=config.max_model_len, max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            gpu_memory_utilization=config.gpu_memory_utilization, seed=config.seed,
            enable_prefix_caching=config.enable_prefix_caching, enforce_eager=True,
            trust_remote_code=config.trust_remote_code, revision=config.revision)
    if not isinstance(workload, BackendWorkload):
        raise TypeError("model execution requires a backend workload")
    from pllm.backends.vllm.runtime import create_vllm_llm
    definition = None
    if config.multiprocessing:
        from pllm.worker import encode_workload
        definition = encode_workload(workload)
    return create_vllm_llm(workload, config, definition=definition)


__all__ = ["make_llm", "ModelConfig", "ExecutionOptions", "HostRuntimeFacts", "Workload", "Phase", "on", "hook",
    "export", "add_logits", "guide_sampling", "train", "Config", "Hidden", "Samples", "Parameter", "StateIndex", "StateVersion", "Result",
    "InstanceState", "RequestState", "BaseLogits", "SampledTokens", "AdamW", "HiddenSite",
    "overlay", "Operand", "Buffer", "TensorUpdate", "ParameterOverlayOperator", "ParameterTarget", "RequestKeyField", "ConditionalLayerReplay", "differentiate",
    "Activation", "ActivationPatchOperator", "StateScope", "StateTensorUpdate", "load_tensor_fields", "Recurrence",
    "transaction", "HostWorkload", "HostTransactionCoordinator", "TransactionHostBinding", "CompletionInput", "NoHostRequest",
    "CompletionWorkload", "InvocationOutput", "PagePublication", "StateRef", "CausalAdapterWorkload", "CausalModel", "CausalTrainBatch",
    "TensorRows", "PagedRows", "SGD", "Batch", "WeightedCEBatch", "TensorResult"]
