"""One real-CUDA stream/receipt proof for the canonical FTTT graph."""

from __future__ import annotations

import torch

from tllm.canonical.workloads.fttt import (
    FTTTCompileOptions,
    FTTTCompletion,
    FTTTCompletionBatch,
    FTTTConfig,
    FTTTExecuted,
    FTTTFeedbackService,
    FTTTFinishReason,
    FTTTInstanceKey,
    FTTTJudgeResult,
    FTTTJudgeService,
    FTTTRetryAction,
    FTTTSession,
    FTTTSpecialization,
    FTTTWorkload,
)
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.semantic_graph import (
    AnchorSymbol,
    SchedulerTimeline,
    TimelineAnchor,
)
from tllm.contracts.tensor_layout import TensorDType


class _CudaNegativeJudge(FTTTJudgeService):
    def judge(self, completion: FTTTCompletion) -> FTTTJudgeResult:
        return FTTTJudgeResult(
            False,
            0.0,
            "wrong arithmetic carry",
            (11, 12),
            "incorrect",
            "cuda_exact",
            "",
        )


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("canonical FTTT CUDA smoke requires CUDA")
    device = torch.device("cuda", 0)
    producer_stream = torch.cuda.Stream(device=device)
    consumer_stream = torch.cuda.Stream(device=device)
    config = FTTTConfig(
        rank=2,
        learning_rate=0.05,
        feedback_feature_size=3,
        max_retries=2,
    )
    workload = FTTTWorkload(config)
    compiled = workload.compile(
        FTTTSpecialization(
            hidden_size=4,
            tensor_dtype=TensorDType.FLOAT32,
            device=device,
            initialization_seed=7,
        ),
        FTTTCompileOptions(
            SchedulerTimeline(
                anchors=(
                    TimelineAnchor(AnchorSymbol("token.commit"), 0),
                    TimelineAnchor(AnchorSymbol("host.feedback.ready"), 1),
                    TimelineAnchor(AnchorSymbol("step.end"), 2),
                    TimelineAnchor(AnchorSymbol("next.prepare"), 3),
                ),
                capabilities=("compute/cuda@1", "host/service@1"),
            )
        ),
    )
    execution = compiled.execution
    service = FTTTFeedbackService(config, _CudaNegativeJudge())
    session = FTTTSession(service, execution)
    identity = RequestIdentity("cuda-feedback-0", 0, 0)
    completion = FTTTCompletion(
        identity,
        0,
        "cuda-round-0",
        (1, 2),
        "wrong arithmetic",
        FTTTFinishReason.STOP,
    )
    with torch.cuda.stream(producer_stream):
        hidden_rows = torch.linspace(
            -1.0,
            1.0,
            steps=4,
            device=device,
            dtype=torch.float32,
        ).reshape(1, 4)
    if not service.enqueue_cuda(
        FTTTCompletionBatch((completion,)),
        hidden_rows,
        producer_stream,
    ):
        raise RuntimeError("negative CUDA feedback did not enqueue training")
    with torch.cuda.stream(consumer_stream):
        polled = session.run_next()
    match polled:
        case FTTTExecuted(result=result):
            pass
        case _:
            raise RuntimeError("canonical FTTT CUDA transaction did not execute")
    diagnostics = execution.diagnostics()
    if (
        result.edit_delta.device != device
        or result.edit_delta.shape != (1, 4)
        or not bool(torch.isfinite(result.edit_delta).all().item())
        or float(result.edit_delta.abs().max().item()) == 0.0
        or result.update_count != 1
        or result.state_version != 1
        or result.intent.retry_controls[0].action is not FTTTRetryAction.RETRY
        or service.instance_state(FTTTInstanceKey("cuda-feedback-0", 0)).adapter_version
        != 1
        or diagnostics.loss_count != 1
        or diagnostics.update_count != 1
        or diagnostics.fallback_count != 0
    ):
        raise RuntimeError("canonical FTTT CUDA lifecycle evidence is incomplete")
    print(
        {
            "loss_count": diagnostics.loss_count,
            "update_count": diagnostics.update_count,
            "state_version": diagnostics.state_version,
            "delta_max": float(result.edit_delta.abs().max().item()),
            "fallback_count": diagnostics.fallback_count,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
