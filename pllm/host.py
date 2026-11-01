"""Host loops: generate -> score -> feed back, with pllm-managed training between generate calls.

Membership rule: a value is only meaningful in a host loop, not inside a model step: make_host and HostConfig,
the step result a host step returns, the host batches its training reads, and the completion, reward, retry and
barrier types a feedback callable exchanges with VllmLlm.generate_feedback.

A host algorithm is a feedback callable `CompletionInput -> feedback` passed to `VllmLlm.generate_feedback`.
When it trains tensor state, that state is an ordinary `pllm.Workload` whose `@pllm.train` reads host batches
(`pllm.batch(...)`); `make_host(HostConfig(...), workload)` compiles it once, and the callable steps it.
Completion, reward, retry and barrier types are what the callable returns to the loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from pllm.contracts.host_batch import (
    CausalTrainBatch as CausalTrainBatch,
    WeightedCEBatch as WeightedCEBatch,
    WeightedRowBatch as WeightedRowBatch,
)
from pllm.execution.host_runtime import CompletionInput as CompletionInput, HostConfig as HostConfig
from pllm.execution.step_input import StepInput as StepInput
from pllm.execution.tensor_binding import StepResult as StepResult

if TYPE_CHECKING:
    from pllm.execution.state_execution import StateExecution
    from pllm.execution.tensor_binding import TensorOutput
    from pllm.execution.tensor_frame import TensorFrame
    from pllm.usage import Workload


def make_host(config: HostConfig, workload: Workload) -> StateExecution[StepInput, TensorFrame, TensorOutput]:
    """Compile `workload` without a model; each `step(StepInput.of_batch(batch))` trains on one host batch and
    returns a StepResult.

    Admit the instance state with `admit(())` before the first step and release it with `retire(())`.
    """
    return workload.compile_host(config)


__all__ = [
    "make_host",
    "HostConfig",
    "StepInput",
    "StepResult",
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
