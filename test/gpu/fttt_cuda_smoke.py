"""FTTT on CUDA: a completion captured on one stream trains an edit consumed on another."""

from __future__ import annotations

import torch

from pllm.contracts.request_identity import RequestIdentity
from pllm.host import Completion, CompletionBatch, CompletionInput, FinishReason, HostConfig, RetryAction
from pllm.workloads.fttt import FTTTConfig, FTTTFeedback, FTTTFeedbackQueue, FTTTJudge, FTTTJudgeResult


class _CudaNegativeJudge(FTTTJudge):
    def judge(self, completion: Completion) -> FTTTJudgeResult:
        return FTTTJudgeResult(False, 0.0, "wrong arithmetic carry", (11, 12), "incorrect", "cuda_exact", "")


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("canonical FTTT CUDA smoke requires CUDA")
    device = torch.device("cuda", 0)
    producer_stream = torch.cuda.Stream(device=device)
    consumer_stream = torch.cuda.Stream(device=device)
    config = FTTTConfig(rank=2, learning_rate=0.05, feedback_feature_size=3, max_retries=2)
    service = FTTTFeedbackQueue(config, _CudaNegativeJudge())
    workload = FTTTFeedback(service, HostConfig(device, hidden_size=4, seed=7, max_input_rows=1))
    identity = RequestIdentity("cuda-feedback-0", 0, 0)
    completion = Completion(identity, 0, "cuda-round-0", (1, 2), "wrong arithmetic", FinishReason.STOP)
    with torch.cuda.stream(producer_stream):
        hidden_rows = torch.linspace(-1.0, 1.0, steps=4, device=device, dtype=torch.float32).reshape(1, 4)
    with torch.cuda.stream(consumer_stream):
        result = workload(CompletionInput.capture(CompletionBatch((completion,)), hidden_rows, producer_stream))
    if result is None:
        raise RuntimeError("canonical FTTT did not execute its queued transaction")
    if (
        result.edit_delta.device != device
        or result.edit_delta.shape != (1, 4)
        or not bool(torch.isfinite(result.edit_delta).all().item())
        or float(result.edit_delta.abs().max().item()) == 0.0
        or result.update_count != 1
        or result.state_version != 1
        or result.intent.retry_controls[0].action is not RetryAction.RETRY
        or service.lifecycle("cuda-round-0", 0).adapter_version != 1
        or len(result.losses) != 1
    ):
        raise RuntimeError("canonical FTTT CUDA lifecycle evidence is incomplete")
    print(f"FTTT CUDA: update_count={result.update_count} state_version={result.state_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
