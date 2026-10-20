"""Public factory for model-backed and host workloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, overload

from pllm.authoring import AdamW as AdamW, CausalModel as CausalModel, HiddenSite as HiddenSite
from pllm.config import ExecutionOptions as ExecutionOptions, ModelConfig as ModelConfig
from pllm.contracts.auxiliary import (
    CausalTrainBatch as CausalTrainBatch,
    StateTensorUpdate as StateTensorUpdate,
    WeightedCEBatch as WeightedCEBatch,
)
from pllm.contracts.backend_attachment import (
    ActivationPatchOperator as ActivationPatchOperator,
    DecodeRefresh as DecodeRefresh,
    DecodeWindow as DecodeWindow,
    MissingPolicy as MissingPolicy,
)
from pllm.contracts.parameter_overlay import (
    ParameterModuleKind as ParameterModuleKind,
    ParameterOverlayOperator as ParameterOverlayOperator,
    ParameterRole as ParameterRole,
    ParameterTarget as ParameterTarget,
)
from pllm.contracts.request_identity import RequestKeyField as RequestKeyField
from pllm.contracts.semantic_graph import ConditionalLayerReplay as ConditionalLayerReplay, StateScope as StateScope
from pllm.contracts.state_ref import StateRef as StateRef
from pllm.execution.codec import load_tensor_fields as load_tensor_fields
from pllm.execution.host_runtime import (
    CompletionInput as CompletionInput,
    HostRuntimeFacts as HostRuntimeFacts,
    HostTransactionCoordinator as HostTransactionCoordinator,
    HostWorkload as HostWorkload,
    NoHostRequest as NoHostRequest,
    TransactionHostBinding as TransactionHostBinding,
)
from pllm.execution.ptts.page_bundle import PagePublication as PagePublication
from pllm.execution.scan import scan as scan
from pllm.execution.tensor_binding import TensorResult as TensorResult
from pllm.execution.tensor_frame import Tensors as Tensors
from pllm.execution.tensor_program import InvocationOutput as InvocationOutput
from pllm.learning import (
    CausalAdapterWorkload as CausalAdapterWorkload,
    CompletionWorkload as CompletionWorkload,
    transaction as transaction,
)
from pllm.usage import (
    SGD as SGD,
    Activation as Activation,
    BaseLogits as BaseLogits,
    Batch as Batch,
    Buffer as Buffer,
    Capture as Capture,
    Config as Config,
    Hidden as Hidden,
    InstanceState as InstanceState,
    Operand as Operand,
    Overlay as Overlay,
    Parameter as Parameter,
    Phase as Phase,
    Recurrence as Recurrence,
    RequestState as RequestState,
    Result as Result,
    SampledTokens as SampledTokens,
    Samples as Samples,
    StateIndex as StateIndex,
    StateVersion as StateVersion,
    TensorUpdate as TensorUpdate,
    Workload as Workload,
    add_logits as add_logits,
    differentiate as differentiate,
    export as export,
    guide_sampling as guide_sampling,
    hook as hook,
    on as on,
    overlay as overlay,
    train as train,
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
    "ModelConfig",
    "ExecutionOptions",
    "HostRuntimeFacts",
    "Workload",
    "Phase",
    "on",
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
    "HiddenSite",
    "overlay",
    "Operand",
    "Buffer",
    "TensorUpdate",
    "ParameterOverlayOperator",
    "ParameterTarget",
    "RequestKeyField",
    "ConditionalLayerReplay",
    "differentiate",
    "Activation",
    "Overlay",
    "ActivationPatchOperator",
    "DecodeWindow",
    "DecodeRefresh",
    "MissingPolicy",
    "StateScope",
    "ParameterModuleKind",
    "ParameterRole",
    "Tensors",
    "StateTensorUpdate",
    "load_tensor_fields",
    "Recurrence",
    "transaction",
    "HostWorkload",
    "HostTransactionCoordinator",
    "TransactionHostBinding",
    "CompletionInput",
    "NoHostRequest",
    "CompletionWorkload",
    "InvocationOutput",
    "PagePublication",
    "StateRef",
    "CausalAdapterWorkload",
    "CausalModel",
    "CausalTrainBatch",
    "SGD",
    "Batch",
    "WeightedCEBatch",
    "TensorResult",
]
