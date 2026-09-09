"""Nominal typed judge implementations for the FTTT host boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pllm.contracts.feedback import Completion, ControlReason

from pllm.workloads.fttt.types import (
    FTTTJudgeResult,
)


class FTTTJudgeService(ABC):
    """Cold host-service implementation selected outside the task graph."""

    @abstractmethod
    def judge(self, completion: Completion) -> FTTTJudgeResult:
        raise NotImplementedError


class FTTTDefaultJudge(FTTTJudgeService):
    def judge(self, completion: Completion) -> FTTTJudgeResult:
        text = completion.text.strip().lower()
        negative = ("incorrect", "wrong", "failed", "failure", "error")
        positive = ("correct", "passed", "success", "solved")
        is_negative = any(marker in text for marker in negative)
        if any(marker in text for marker in positive) and not is_negative:
            return FTTTJudgeResult(
                True,
                1.0,
                "correct",
                (),
                "correct",
                "fttt_default",
                ControlReason.CORRECT_OUTPUT.value,
            )
        if completion.finish_reason.value != "stop":
            feedback = (
                "Generation did not finish cleanly: "
                + completion.finish_reason.value
                + "."
            )
        else:
            feedback = "Your answer is incorrect."
        return FTTTJudgeResult(
            False,
            0.0,
            feedback,
            (),
            "incorrect",
            "fttt_default",
            "",
        )


__all__ = ["FTTTDefaultJudge", "FTTTJudgeService"]
