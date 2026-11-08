"""Run documentation-only agent authoring sessions, then score their submissions outside the sandbox."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.ai_eval.records import Metadata, Score, TaskResult  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PYTHON = Path("/workspace/pllm/.venv/bin/python")
PROMPT = "Read TASK.md, README.md and relevant doc/ documents. Implement TASK.md in submission.py. Work autonomously."


def agy_turns(event: dict[str, Any]) -> int:
    """Count completed model responses, including calls, even if the session later times out."""
    step = event.get("step_update", {})
    return int(step.get("state") == "DONE" and step.get("step_type") == "agent_response")


def kimi_turns(event: dict[str, Any]) -> int:
    """Count Kimi's complete assistant messages, including tool calls."""
    return int(event.get("role") == "assistant")


@dataclass(frozen=True)
class Agent:
    """CLI-specific launch, authentication and complete-message transcript format."""

    binary: Path
    model: str
    auth_files: tuple[str, ...]
    arguments: tuple[str, ...]
    count_turns: Callable[[dict[str, Any]], int]

    def argv(self, model: str) -> list[str]:
        """Build a non-interactive prompt command inside the sandbox."""
        return ["/author", "--model", model, "--output-format", "stream-json", *self.arguments, PROMPT]


AGENTS = {
    "codex": Agent(
        Path("/home/user/.local/bin/codex"),
        "gpt-6.1-sol",
        (".codex/auth.json",),
        ("--dangerously-skip-permissions", "--add-dir", "/workspace", "-p"),
        agy_turns,
    ),
    "kimi": Agent(
        Path("/home/user/.codex/auth.json"),
        "gpt-6.1-sol",
        (
            ".codex/auth.json",
            ".codex/auth.json",
            ".codex/auth.json",
            ".codex/auth.json",
            ".codex/auth.json",
        ),
        ("--skills-dir", "/home/agent/empty-skills", "--prompt"),
        kimi_turns,
    ),
}
TASKS = ("caa", "dola", "iti", "prompt_lora", "esamp", "memory", "looped_depth", "ttrl")


def execute(command: list[str], cwd: Path, output: Path, timeout: int) -> tuple[int, float]:
    """Capture a bounded process and kill its complete process group on timeout."""
    start = time.monotonic()
    with output.open("w") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = 124
    return code, round(time.monotonic() - start, 2)


def prepare(destination: Path, revision: str) -> tuple[Path, str]:
    """Export only committed docs and the task into a clean author workspace."""
    workspace = destination / "workspace"
    workspace.mkdir(parents=True)
    archive = subprocess.check_output(["git", "archive", revision, "README.md", "doc/", ":(exclude)doc/dev"], cwd=ROOT)
    subprocess.run(["tar", "-x", "-C", str(workspace)], input=archive, check=True)
    task = destination.name
    shutil.copyfile(HERE / "specs" / f"{task}.md", workspace / "TASK.md")
    shutil.copyfile(workspace / "TASK.md", destination / "task.md")
    home = destination / "home"
    home.mkdir(mode=0o700)
    (home / "empty-skills").mkdir()
    digest = hashlib.sha256()
    for path in sorted(workspace.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(workspace)).encode())
            digest.update(path.read_bytes())
    return workspace, digest.hexdigest()


def sandbox(destination: Path, agent: Agent, model: str, *, authenticate: bool = True) -> list[str]:
    """Mount an allowlist; the host repo, venv, tests and hidden oracles have no reachable path."""
    command = ["bwrap", "--die-with-parent", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--new-session"]
    for path in ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc"):
        command += ["--ro-bind", path, path]
    command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/run"]
    command += ["--bind", str(destination / "workspace"), "/workspace"]
    command += ["--bind", str(destination / "home"), "/home/agent"]
    if authenticate:
        for name in agent.auth_files:
            source = Path.home() / name
            if source.exists():
                target = destination / "home" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                command += ["--ro-bind", str(source), f"/home/agent/{name}"]
    command += ["--ro-bind", str(agent.binary), "/author", "--chdir", "/workspace", "--clearenv"]
    for key, value in {"HOME": "/home/agent", "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "TERM": "dumb"}.items():
        command += ["--setenv", key, value]
    return command + agent.argv(model)


def turns(path: Path, agent: Agent) -> int:
    """Count complete assistant events, including tool-call messages, excluding transport deltas/retries."""
    count = 0
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        count += agent.count_turns(event)
    return count


def provider_failure(path: Path) -> str:
    """Recognize provider errors in CLI diagnostics, never in assistant prose or tool output."""
    pattern = re.compile(
        r"(?:HTTP(?:/\S+)?[ :]+(?:429|401|403|5\d\d)|"
        r"\b(?:429|401|403)\b|"
        r"quota.{0,40}(?:exceeded|exhausted)|resource_exhausted|rate.?limit|"
        r"unauthenticated|unauthorized|authentication failed|invalid_grant|invalid.{0,20}(?:token|api.?key)|"
        r"not logged in|please (?:log|sign) in|"
        r"connection (?:refused|reset|error|timed out)|network (?:error|unreachable)|"
        r"dial tcp|no such host|TLS handshake|context deadline exceeded)",
        re.IGNORECASE,
    )
    failure = ""
    for line in path.read_text().splitlines():
        try:
            event: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            event = {}
        kind = event.get("event", event.get("type"))
        if kind == "result":
            result = event.get("result", event)
            if result.get("status") == "SUCCESS" or result.get("is_error") is False:
                return ""
        elif event and kind != "error" and "error" not in event:
            continue
        match = pattern.search(line)
        if match:
            failure = match.group(0)
    return failure


def author(destination: Path, args: argparse.Namespace) -> tuple[int, float, int, str, str]:
    """Run a fresh attempt, retrying provider infrastructure failure once after backoff."""
    agent = AGENTS[args.agent]
    seconds = 0.0
    count = 0
    for attempt in (1, 2):
        workspace, digest = prepare(destination, args.revision)
        transcript = destination / f"transcript-{attempt}.jsonl"
        code, elapsed = execute(sandbox(destination, agent, args.model), workspace, transcript, args.timeout)
        seconds += elapsed
        count += turns(transcript, agent)
        failure = provider_failure(transcript)
        shutil.copyfile(transcript, destination / "transcript.jsonl")
        if not failure or attempt == 2:
            return code, round(seconds, 2), count, digest, failure
        print(f"{destination.name}: INFRA ({failure}); retrying in 10s", flush=True)
        shutil.rmtree(destination / "home")
        shutil.rmtree(workspace)
        time.sleep(10)
    raise AssertionError("two attempts always return")


def run_task(task: str, output: Path, args: argparse.Namespace) -> TaskResult:
    """Author without oracle feedback, save evidence and independently score completed sessions."""
    destination = output / task
    destination.mkdir()
    try:
        code, seconds, count, digest, failure = author(destination, args)
        workspace = destination / "workspace"
        for name in ("submission.py", "notes.md"):
            path = workspace / name
            if path.is_file() and not path.is_symlink():
                shutil.copyfile(path, destination / name)
        if failure:
            scoring: Score = {
                "score": None,
                "imports_declares": False,
                "check": False,
                "oracle_cpu": False,
                "explain_tier": "unavailable",
                "failures": {"provider": failure},
                "category": "INFRA",
            }
            (destination / "score.json").write_text(json.dumps(scoring, indent=2) + "\n")
        else:
            score_task(task, destination, args.gpu)
            scoring = json.loads((destination / "score.json").read_text())
        result: TaskResult = {
            **scoring,
            "task": task,
            "author_exit": code,
            "wall_seconds": seconds,
            "author_turns": count,
            "input_sha256": digest,
            "transcript": f"{task}/transcript.jsonl",
        }
        (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"{task}: {result['category']} {result['score']}/4, {seconds}s, {count} assistant turns", flush=True)
        return result
    finally:
        shutil.rmtree(destination / "home", ignore_errors=True)
        shutil.rmtree(destination / "workspace", ignore_errors=True)


def score_task(task: str, destination: Path, gpu: bool) -> None:
    """Use the trusted repo interpreter; optional CUDA execution always takes the shared GPU lock."""
    command = [str(PYTHON), str(HERE / "score.py"), task, str(destination)]
    if gpu:
        command = ["flock", "/tmp/pllm-gpu.lock", "taskset", "-c", "0-15", *command, "--gpu"]
    code, _ = execute(command, ROOT, destination / "scoring.log", 180)
    if code or not (destination / "score.json").exists():
        result: Score = {
            "score": 0,
            "imports_declares": False,
            "check": False,
            "oracle_cpu": False,
            "explain_tier": "unavailable",
            "failures": {"harness": f"scorer exited {code}; see scoring.log"},
            "category": "unclassified",
        }
        (destination / "score.json").write_text(json.dumps(result, indent=2) + "\n")


def report(output: Path, metadata: Metadata, results: list[TaskResult]) -> None:
    """Save machine-readable evidence and a compact human-readable table with relative transcript links."""
    (output / "results.json").write_text(json.dumps({"metadata": metadata, "tasks": results}, indent=2) + "\n")
    lines = [
        "# AI-usability evaluation",
        "",
        f"Base commit: `{metadata['base_commit']}`; agent: `{metadata['agent']}`; model: `{metadata['model']}`.",
        "",
        "| Task | Score /4 | Declares | check | CPU oracle | Tier | Seconds | Turns | Exit | Category | Transcript |",
        "|---|---:|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for result in results:
        cells = [
            result["task"],
            "INFRA" if result["score"] is None else str(result["score"]),
            *("pass" if result[k] else "FAIL" for k in ("imports_declares", "check", "oracle_cpu")),
            result["explain_tier"],
            str(result["wall_seconds"]),
            str(result["author_turns"]),
            str(result["author_exit"]),
            result["category"],
            f"[JSONL]({result['transcript']})",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Turns use the selected CLI transcript format; provider INFRA retries once after 10s. "
        "Turns/time sum attempts (backoff excluded). "
        "check skips are preserved in score.json and are not oracle passes. GPU was "
        + ("enabled." if metadata["gpu"] else "not run."),
        "",
        "Failure categories require evidence review; see review.md.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Evaluate a pinned revision without giving the author agent access to the evaluator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--agent", choices=AGENTS, default="codex")
    parser.add_argument("--model", help="default: gpt-6.1-sol (codex), gpt-6.1-sol (kimi)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--gpu", action="store_true", help="also compare tensor oracles on CUDA (no engine)")
    args = parser.parse_args()
    agent = AGENTS[args.agent]
    args.model = args.model or agent.model
    if args.jobs < 1 or args.timeout < 1:
        parser.error("use positive --jobs and --timeout values")
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("--output must be a new directory; prior evidence is never overwritten")
    commit = subprocess.check_output(["git", "rev-parse", args.revision], cwd=ROOT, text=True).strip()
    runtime = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    # Runtime is this checkout; reject comparison against a different pllm tree.
    subprocess.run(
        ["git", "diff", "--exit-code", commit, "--", "pllm", "rust", "doc", "README.md"], cwd=ROOT, check=True
    )
    args.revision = commit
    args.output.mkdir(parents=True)
    metadata: Metadata = {
        "base_commit": commit,
        "runtime_commit": runtime,
        "model": args.model,
        "gpu": args.gpu,
        "agent": args.agent,
        "author_version": subprocess.check_output([str(agent.binary), "--version"], text=True).strip(),
        "timeout_seconds": args.timeout,
        "jobs": args.jobs,
        "python": str(PYTHON),
        "isolation": "bubblewrap allowlist; docs/task only; no repo/runtime source or scoring feedback",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_task, task, args.output, args) for task in args.tasks]
        results = [future.result() for future in futures]
    report(args.output, metadata, results)


if __name__ == "__main__":
    main()
