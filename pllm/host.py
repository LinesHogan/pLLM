"""Host-side learning: transactional host workloads and completion/causal-adapter training.

Pass `HostRuntimeFacts` to `pllm.make_llm` for host workloads.
"""

from pllm.authoring import CausalModel as CausalModel
from pllm.contracts.auxiliary import CausalTrainBatch as CausalTrainBatch, WeightedCEBatch as WeightedCEBatch
from pllm.execution.host_runtime import (
    CompletionInput as CompletionInput,
    HostRuntimeFacts as HostRuntimeFacts,
    HostTransactionCoordinator as HostTransactionCoordinator,
    HostWorkload as HostWorkload,
    NoHostRequest as NoHostRequest,
    TransactionHostBinding as TransactionHostBinding,
)
from pllm.learning import (
    CausalAdapterWorkload as CausalAdapterWorkload,
    CompletionWorkload as CompletionWorkload,
    transaction as transaction,
)
from pllm.usage import Batch as Batch

__all__ = [
    "transaction",
    "HostWorkload",
    "HostTransactionCoordinator",
    "TransactionHostBinding",
    "HostRuntimeFacts",
    "CompletionInput",
    "NoHostRequest",
    "CompletionWorkload",
    "CausalAdapterWorkload",
    "CausalModel",
    "CausalTrainBatch",
    "WeightedCEBatch",
    "Batch",
]
