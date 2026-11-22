"""Host loops: generate -> score -> feed back, with pLLM-managed training between generate calls.

Membership rule: a value is only meaningful in a host loop, not inside a model step: make_host, HostConfig and the
HostTrainer it returns, the host batches training reads, and the completion, reward, retry and barrier types a
feedback callable exchanges with VllmLlm.generate_feedback.

A host algorithm is a feedback callable `CompletionInput -> feedback` passed to `VllmLlm.generate_feedback`.
State the model reads while generating trains through `llm.generate(..., batch=...)`. State only host code reads
(a reward head, an edit the callable computes itself) is an ordinary `pllm.Workload` whose `@pllm.train` reads
host batches (`pllm.batch(...)`): `make_host(HostConfig(...), workload)` compiles it once, and the callable calls
`train(batch)`. Completion, reward, retry and barrier types are what the callable returns to the loop.
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING

import torch

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
    HostBatch,
    WeightedCEBatch as WeightedCEBatch,
    WeightedRowBatch as WeightedRowBatch,
)
from pllm.execution.host_runtime import CompletionInput as CompletionInput, HostConfig as HostConfig
from pllm.execution.step_driver import TrainStep as TrainStep
from pllm.execution.step_input import StepInput
from pllm.execution.tensor_declaration import StepResult

if TYPE_CHECKING:
    from pllm.execution.state_execution import StateDiagnostics, StateExecution
    from pllm.workload import Workload


class HostTrainer:
    """A workload's instance state, trained on host batches without a model.

    Each `train(batch)` runs on the caller's current CUDA stream, so the batch only has to be ready there.
    `close()` (or leaving a `with` block) releases the state.
    """

    def __init__(self, execution: StateExecution) -> None:
        self._execution = execution
        execution.admit((), self._stream())

    def _stream(self) -> torch.cuda.Stream | None:
        device = self._execution.device
        return torch.cuda.current_stream(device) if device.type == "cuda" else None

    def train(self, batch: HostBatch[torch.Tensor]) -> TrainStep:
        """One optimizer step of the workload's `@pllm.train` method on `batch`."""
        result = self._execution.step(StepInput.of_batch(batch), self._stream())
        assert isinstance(result, StepResult)
        return TrainStep(result.loss, result.parameters)

    @property
    def version(self) -> int:
        """The trained state's newest published version: 0 at admission, one more per step that published."""
        (latest,) = self._execution.latest(())
        return latest.version_id

    def diagnostics(self) -> StateDiagnostics:
        """Exact counts of the steps so far (`loss_count`, `update_count`, `trained_rows`, ...), as on an engine."""
        return self._execution.diagnostics()

    def close(self) -> None:
        self._execution.retire((), wait=True)
        self._execution.close()

    def __enter__(self) -> HostTrainer:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()


def make_host(config: HostConfig, workload: Workload) -> HostTrainer:
    """Compile `workload` without a model and admit its instance state; `train(batch)` steps it on one host batch."""
    return HostTrainer(workload.compile_host(config))


__all__ = [
    "make_host",
    "HostConfig",
    "HostTrainer",
    "TrainStep",
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
