"""Check harness acceptance against hidden references and verify author filesystem isolation."""

import subprocess
from pathlib import Path
from types import ModuleType

import pytest
import torch

from tools.ai_eval import oracles, reference, run, score


@pytest.mark.parametrize("task", run.TASKS)
def test_reference_acceptance(task: str, tmp_path: Path) -> None:
    """Every task must be solvable before a failed author submission can be called a regression."""
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


@pytest.mark.parametrize("agent_name", run.AGENTS)
def test_author_cannot_read_host_sources(tmp_path: Path, agent_name: str) -> None:
    """A real mount namespace must hide a sentinel, runtime, oracle, tests and agent history."""
    destination = tmp_path / "caa"
    destination.mkdir()
    workspace, _ = run.prepare(destination, "HEAD")
    sentinel = tmp_path / "hidden-oracle-secret"
    sentinel.write_text("must not be visible")
    command = run.sandbox(destination, run.AGENTS[agent_name], run.AGENTS[agent_name].model, authenticate=False)
    command = command[: command.index("/author", command.index("--clearenv"))]
    # A non-agent shell probes the exact namespace the production author command would receive.
    script = (
        "test -f /workspace/TASK.md && test -f /workspace/README.md "
        "&& test ! -e /workspace/pllm && test ! -e /workspace/test "
        "&& test ! -e /workspace/pllm && test ! -e /home/agent/.codex/auth.json "
        "&& test ! -e /home/agent/.gemini/codex-cli/conversation_summaries.db "
        f"&& test ! -e {sentinel}"
    )
    subprocess.run([*command, "/bin/sh", "-c", script], cwd=workspace, check=True)


@pytest.mark.parametrize(
    ("agent", "transcript", "count", "failure"),
    [
        ("codex", '{"event":"step_update","step_update":{"state":"DONE","step_type":"agent_response"}}', 1, ""),
        ("codex", '{"event":"step_update","step_update":{"text_delta":"HTTP 429"}}', 0, ""),
        ("kimi", '{"role":"assistant","content":"quota exhausted"}', 1, ""),
        ("codex", '{"event":"error","error":{"message":"HTTP 429 quota exhausted"}}', 0, "HTTP 429"),
        ("kimi", "authentication failed: invalid token", 0, "authentication failed"),
        ("codex", "dial tcp: connection refused", 0, "dial tcp"),
        ("codex", 'HTTP 429\n{"event":"result","result":{"status":"SUCCESS","num_turns":2}}', 0, ""),
    ],
)
def test_transcript_evidence(agent: str, transcript: str, count: int, failure: str, tmp_path: Path) -> None:
    """Count provider-specific complete turns and distinguish service errors from author text."""
    path = tmp_path / "transcript.jsonl"
    path.write_text(transcript)
    assert run.turns(path, run.AGENTS[agent]) == count
    assert run.provider_failure(path) == failure


def test_provider_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One fresh retry preserves both transcripts and never assigns provider failure a zero score."""
    import argparse
    import json

    attempts: list[Path] = []
    backoffs: list[int] = []

    def unavailable(command: list[str], cwd: Path, output: Path, timeout: int) -> tuple[int, float]:
        assert not (cwd / "submission.py").exists()
        (cwd / "submission.py").write_text("# partial output must not reach the retry")
        output.write_text('{"event":"error","error":{"message":"HTTP 429"}}\n')
        attempts.append(output)
        return 1, 2.0

    monkeypatch.setattr(run, "execute", unavailable)
    monkeypatch.setattr(run.time, "sleep", backoffs.append)
    args = argparse.Namespace(agent="codex", model="gpt-6.1-sol", revision="HEAD", timeout=10, gpu=False)
    result = run.run_task("caa", tmp_path, args)
    assert result["category"] == "INFRA" and result["score"] is None
    assert result["wall_seconds"] == 4.0
    assert backoffs == [10] and len(attempts) == 2
    assert all(path.is_file() for path in attempts)
    assert json.loads((tmp_path / "caa" / "result.json").read_text())["score"] is None
    assert not (tmp_path / "caa" / "home").exists()
    assert not (tmp_path / "caa" / "workspace").exists()
