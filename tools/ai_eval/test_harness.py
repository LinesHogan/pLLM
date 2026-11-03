"""Check harness acceptance against hidden references and verify author filesystem isolation."""

import subprocess
from pathlib import Path
from types import ModuleType

import pytest
import torch

from tools.ai_eval import oracles, reference, run, score


@pytest.mark.parametrize("task", run.TASKS)
def test_reference_acceptance(task: str, tmp_path: Path) -> None:
    """Every task must be solvable before a failed Kimi submission can be called a regression."""
    submission = ModuleType("reference_submission")
    submission.__dict__["Workload"] = reference.WORKLOADS[task]
    submission.__dict__["direction"] = reference.direction
    submission.__dict__["vote"] = reference.vote
    submission.__dict__["feedback"] = reference.feedback
    result = score.evaluate(task, tmp_path, submission)
    assert result["score"] == 4, result["failures"]
    assert result["explain_tier"] in ("graph", "eager")


def zero_direction(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """A deliberately incorrect population contrast."""
    return torch.zeros_like(positive[0])


def test_oracle_rejects_plausible_wrong_math() -> None:
    """A valid workload declaration alone cannot earn mathematical acceptance."""
    submission = ModuleType("wrong_submission")
    submission.__dict__["Workload"] = reference.CAA
    submission.__dict__["direction"] = zero_direction
    with pytest.raises(AssertionError):
        oracles.check_math("caa", submission)


def test_author_cannot_read_host_sources(tmp_path: Path) -> None:
    """A real mount namespace must hide a sentinel, runtime, oracle, tests and agent history."""
    destination = tmp_path / "caa"
    destination.mkdir()
    workspace, _ = run.prepare(destination, "HEAD", authenticate=False)
    sentinel = tmp_path / "hidden-oracle-secret"
    sentinel.write_text("must not be visible")
    command = run.sandbox(destination, "gpt-6.1-sol")
    command = command[: command.index("/kimi", command.index("--clearenv"))]
    # A non-agent shell probes the exact namespace the production Kimi command would receive.
    script = (
        "test -f /workspace/TASK.md && test -f /workspace/README.md "
        "&& test ! -e /workspace/pllm && test ! -e /workspace/test "
        "&& test ! -e /workspace/pllm && test ! -e /home/agent/.codex/auth.json "
        f"&& test ! -e {sentinel}"
    )
    subprocess.run([*command, "/bin/sh", "-c", script], cwd=workspace, check=True)
