"""Host-side learning: transactional host workloads and completion/causal-adapter training.

Pass `HostRuntimeFacts` to `pllm.make_llm` for host workloads. Completion, reward, retry and barrier types
are the feedback a host service exchanges with the framework.
"""

from pllm.authoring import CausalModel as CausalModel
from pllm.contracts.auxiliary import CausalTrainBatch as CausalTrainBatch, WeightedCEBatch as WeightedCEBatch
from pllm.contracts.feedback import (
    BarrierState as BarrierState,
    Completion as Completion,
    CompletionBatch as CompletionBatch,
    ControlReason as ControlReason,
    FinishReason as FinishReason,
    RetryAction as RetryAction,
    RetryControl as RetryControl,
    RewardOutcome as RewardOutcome,
    UpdateBarrier as UpdateBarrier,
)
from pllm.execution.host_runtime import (
    CompletionInput as CompletionInput,
    HostReceipt as HostReceipt,
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
from pllm.usage import batch as batch

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
    "batch",
    "BarrierState",
    "Completion",
    "CompletionBatch",
    "ControlReason",
    "FinishReason",
    "RetryAction",
    "RetryControl",
    "RewardOutcome",
    "UpdateBarrier",
    "HostReceipt",
]
