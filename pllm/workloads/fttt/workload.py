"""FTTT on the host: judge each completion, train a low-rank edit on the failed ones, and return the edit a retry
applies. Pass `FTTTWorkload.run` as generate_feedback's `feedback=` and `FTTTResult.state_update` as its `edit=`."""

from __future__ import annotations

from dataclasses import replace

import torch

from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.host import CompletionInput
from pllm.workloads.fttt.edit import FTTTEdit
from pllm.workloads.fttt.host_service import FTTTFeedbackService
from pllm.workloads.fttt.types import FTTTConfig, FTTTQueuedFeedback, FTTTResult


class FTTTWorkload:
    """One feedback service and the edit it trains; steps run on the caller's current CUDA stream."""

    def __init__(
        self,
        feedback: FTTTFeedbackService,
        *,
        hidden_size: int,
        device: torch.device = torch.device("cpu"),
        initialization_seed: int = 0,
        max_input_rows: int | None = None,
    ) -> None:
        if not 0 < feedback.config.rank <= hidden_size:
            raise ValueError(f"FTTT rank {feedback.config.rank} must be in 1..hidden_size ({hidden_size})")
        self.feedback = feedback
        self.hidden_size, self.device, self.max_input_rows = hidden_size, device, max_input_rows
        self.edit = FTTTEdit.initial(feedback.config, hidden_size, device, initialization_seed)
        # Steps may run on different streams; each orders after the last write to the edit.
        self._written = CudaStreamReadiness(torch.cuda.current_stream(device)) if device.type == "cuda" else None

    @property
    def config(self) -> FTTTConfig:
        return self.feedback.config

    def run(self, batch: CompletionInput) -> FTTTResult | None:
        """Judge `batch`, queue its feedback, and run one edit step on the oldest queued feedback."""
        self.feedback.enqueue(batch)
        return self.step()

    def step(self) -> FTTTResult | None:
        """One edit step on the oldest queued feedback, None when nothing is queued. A failure restores the edit
        and drops the queue."""
        queued = self.feedback.begin()
        if queued is None:
            return None
        snapshot = self.edit.snapshot()
        try:
            result = self._train(queued)
            self.feedback.commit(queued, result.intent)
        except BaseException:
            self.edit = snapshot
            self.feedback.abort(queued)
            raise
        return result

    def _train(self, queued: FTTTQueuedFeedback) -> FTTTResult:
        hidden = queued.batch.hidden
        if (
            hidden.shape[1] != self.hidden_size
            or hidden.dtype != torch.float32
            or hidden.device != self.device
            or self.max_input_rows is not None
            and len(hidden) > self.max_input_rows
        ):
            limit = "" if self.max_input_rows is None else f" and at most {self.max_input_rows} rows"
            raise ValueError(f"FTTT needs float32 completion rows of width {self.hidden_size} on {self.device}{limit}")
        if self._written is not None:
            stream = torch.cuda.current_stream(self.device)
            self._written.order_on(stream)
            if queued.batch.readiness is not None:
                queued.batch.readiness.order_on(stream)
        loss = None
        with torch.inference_mode(False), torch.enable_grad():
            if queued.selected_rows:
                rows = torch.tensor(queued.selected_rows, dtype=torch.long, device=hidden.device)
                edit_input, loss = self.edit.train(
                    self.config, hidden.detach().index_select(0, rows), queued.intent.feedback_records
                )
            else:
                edit_input = hidden.new_empty((0, hidden.shape[1] + self.config.feedback_feature_size + 1))
            delta = self.edit.delta(edit_input)
        if self._written is not None:
            self._written = CudaStreamReadiness(torch.cuda.current_stream(self.device))
        intent = replace(queued.intent, adapter_version=self.edit.version)
        losses = () if loss is None else (loss,)
        return FTTTResult(intent.transaction_id, delta, intent, losses, len(losses), self.edit.version)


__all__ = ["FTTTWorkload"]
