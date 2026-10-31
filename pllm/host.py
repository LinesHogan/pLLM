"""Host loops: generate -> score -> feed back, with pllm-managed training between generate calls.

A host algorithm is a feedback callable `CompletionInput -> feedback` passed to `VllmLlm.generate_feedback`.
When it trains tensor state, that state is an ordinary `pllm.Workload` whose `@pllm.train` reads host batches
(`pllm.batch(...)`); `make_host(HostConfig(...), workload)` compiles it once, and the callable steps it.
Completion, reward, retry and barrier types are what the callable returns to the loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pllm.contracts.auxiliary import (
    CausalTrainBatch as CausalTrainBatch,
    WeightedCEBatch as WeightedCEBatch,
    WeightedRowBatch as WeightedRowBatch,
)
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
from pllm.execution.host_runtime import CompletionInput as CompletionInput, HostConfig as HostConfig

if TYPE_CHECKING:
    from pllm.execution.state_execution import StateExecution
    from pllm.execution.step_input import TensorStepInput
    from pllm.execution.tensor_binding import TensorOutput
    from pllm.execution.tensor_frame import TensorFrame
    from pllm.usage import Workload


def make_host(config: HostConfig, workload: Workload) -> StateExecution[TensorStepInput, TensorFrame, TensorOutput]:
    """Compile `workload` without a model; each `step(TensorStepInput.of_batch(batch))` trains on one host batch.

    Admit the instance state with `admit(())` before the first step and release it with `retire(())`.
    """
    return workload.declare_host(config).compile()


__all__ = [
    "make_host",
    "HostConfig",
    "CompletionInput",
    "WeightedCEBatch",
    "WeightedRowBatch",
    "CausalTrainBatch",
    "BarrierState",
    "Completion",
    "CompletionBatch",
    "ControlReason",
    "FinishReason",
    "RetryAction",
    "RetryControl",
    "RewardOutcome",
    "UpdateBarrier",
]
