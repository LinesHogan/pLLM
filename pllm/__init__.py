"""Public factory for model-backed and host workloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, overload

from pllm.authoring import AdamW as AdamW
from pllm.authoring import CausalModel as CausalModel
from pllm.authoring import HiddenSite as HiddenSite
from pllm.config import ExecutionOptions as ExecutionOptions
from pllm.config import ModelConfig as ModelConfig
from pllm.contracts.auxiliary import CausalTrainBatch as CausalTrainBatch
from pllm.contracts.auxiliary import StateTensorUpdate as StateTensorUpdate
from pllm.contracts.auxiliary import WeightedCEBatch as WeightedCEBatch
from pllm.contracts.backend_attachment import ActivationPatchOperator as ActivationPatchOperator
from pllm.contracts.backend_attachment import DecodeRefresh as DecodeRefresh
from pllm.contracts.backend_attachment import DecodeWindow as DecodeWindow
from pllm.contracts.backend_attachment import MissingPolicy as MissingPolicy
from pllm.contracts.parameter_overlay import ParameterModuleKind as ParameterModuleKind
from pllm.contracts.parameter_overlay import ParameterOverlayOperator as ParameterOverlayOperator
from pllm.contracts.parameter_overlay import ParameterRole as ParameterRole
from pllm.contracts.parameter_overlay import ParameterTarget as ParameterTarget
from pllm.contracts.request_identity import RequestKeyField as RequestKeyField
from pllm.contracts.semantic_graph import ConditionalLayerReplay as ConditionalLayerReplay
from pllm.contracts.semantic_graph import StateScope as StateScope
from pllm.contracts.state_ref import StateRef as StateRef
from pllm.execution.codec import load_tensor_fields as load_tensor_fields
from pllm.execution.host_runtime import (
    CompletionInput as CompletionInput,
)
from pllm.execution.host_runtime import (
    HostRuntimeFacts as HostRuntimeFacts,
)
from pllm.execution.host_runtime import (
    HostTransactionCoordinator as HostTransactionCoordinator,
)
from pllm.execution.host_runtime import (
    HostWorkload as HostWorkload,
)
from pllm.execution.host_runtime import (
    NoHostRequest as NoHostRequest,
)
from pllm.execution.host_runtime import (
    TransactionHostBinding as TransactionHostBinding,
)
from pllm.execution.ptts.page_bundle import PagePublication as PagePublication
from pllm.execution.tensor_binding import TensorResult as TensorResult
from pllm.execution.tensor_frame import Tensors as Tensors
from pllm.execution.tensor_program import InvocationOutput as InvocationOutput
from pllm.learning import (
    CausalAdapterWorkload as CausalAdapterWorkload,
)
from pllm.learning import (
    CompletionWorkload as CompletionWorkload,
)
from pllm.learning import (
    transaction as transaction,
)
from pllm.usage import (
    SGD as SGD,
)
from pllm.usage import (
    Activation as Activation,
)
from pllm.usage import (
    BaseLogits as BaseLogits,
)
from pllm.usage import (
    Batch as Batch,
)
from pllm.usage import (
    Buffer as Buffer,
)
from pllm.usage import (
    Capture as Capture,
)
from pllm.usage import (
    Config as Config,
)
from pllm.usage import (
    Hidden as Hidden,
)
from pllm.usage import (
    InstanceState as InstanceState,
)
from pllm.usage import (
    Operand as Operand,
)
from pllm.usage import (
    Overlay as Overlay,
)
from pllm.usage import (
    Parameter as Parameter,
)
from pllm.usage import (
    Phase as Phase,
)
from pllm.usage import (
    Recurrence as Recurrence,
)
from pllm.usage import (
    RequestState as RequestState,
)
from pllm.usage import (
    Result as Result,
)
from pllm.usage import (
    SampledTokens as SampledTokens,
)
from pllm.usage import (
    Samples as Samples,
)
from pllm.usage import (
    StateIndex as StateIndex,
)
from pllm.usage import (
    StateVersion as StateVersion,
)
from pllm.usage import (
    TensorUpdate as TensorUpdate,
)
from pllm.usage import (
    Workload as Workload,
)
from pllm.usage import (
    add_logits as add_logits,
)
from pllm.usage import (
    differentiate as differentiate,
)
from pllm.usage import (
    export as export,
)
from pllm.usage import (
    guide_sampling as guide_sampling,
)
from pllm.usage import (
    hook as hook,
)
from pllm.usage import (
    on as on,
)
from pllm.usage import (
    overlay as overlay,
)
from pllm.usage import (
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
