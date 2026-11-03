"""Run documentation-only Kimi authoring sessions, then score their submissions outside the sandbox."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.ai_eval.records import Metadata, Score, TaskResult  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PYTHON = Path("/workspace/pllm/.venv/bin/python")
KIMI = Path("/home/user/.codex/auth.json")
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


def prepare(destination: Path, revision: str, *, authenticate: bool = True) -> tuple[Path, str]:
    """Export only committed docs and the task into a clean author workspace."""
    workspace = destination / "workspace"
    workspace.mkdir(parents=True)
    archive = subprocess.check_output(["git", "archive", revision, "README.md", "doc/"], cwd=ROOT)
    subprocess.run(["tar", "-x", "-C", str(workspace)], input=archive, check=True)
    task = destination.name
    shutil.copyfile(HERE / "specs" / f"{task}.md", workspace / "TASK.md")
    shutil.copyfile(workspace / "TASK.md", destination / "task.md")
    home = destination / "home"
    home.mkdir(mode=0o700)
    config = home / ".codex"
    config.mkdir(mode=0o700)
    # Copy only authentication/configuration; no prior sessions, skills, workspace history or logs.
    source = KIMI.parent.parent if authenticate else config
    for name in ("config.toml", "device_id", "region"):
        if (source / name).exists():
            shutil.copy2(source / name, config / name)
    for name in ("credentials", "oauth"):
        if (source / name).exists():
            shutil.copytree(source / name, config / name)
    (home / "empty-skills").mkdir()
    digest = hashlib.sha256()
    for path in sorted(workspace.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(workspace)).encode())
            digest.update(path.read_bytes())
    return workspace, digest.hexdigest()


def sandbox(destination: Path, model: str) -> list[str]:
    """Mount an allowlist; the host repo, venv, tests and hidden oracles have no reachable path."""
    command = ["bwrap", "--die-with-parent", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--new-session"]
    for path in ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc"):
        command += ["--ro-bind", path, path]
    command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/run"]
    command += ["--bind", str(destination / "workspace"), "/workspace"]
    command += ["--bind", str(destination / "home"), "/home/agent"]
    command += ["--ro-bind", str(KIMI), "/kimi", "--chdir", "/workspace", "--clearenv"]
    for key, value in {"HOME": "/home/agent", "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "TERM": "dumb"}.items():
        command += ["--setenv", key, value]
    prompt = (
        "Read TASK.md, README.md and relevant doc/ documents. Implement TASK.md in submission.py. Work autonomously."
    )
    return command + [
        "/kimi",
        "--model",
        model,
        "--skills-dir",
        "/home/agent/empty-skills",
        "--output-format",
        "stream-json",
        "--prompt",
        prompt,
    ]


def turns(path: Path) -> int:
    """Count complete assistant events, including tool-call messages, excluding transport deltas/retries."""
    count = 0
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        count += event.get("role") == "assistant"
    return count


def run_task(task: str, output: Path, args: argparse.Namespace) -> TaskResult:
    """Author once without oracle feedback, save the submission and independently score it."""
    destination = output / task
    destination.mkdir()
    try:
        workspace, digest = prepare(destination, args.revision)
        code, seconds = execute(
            sandbox(destination, args.model), workspace, destination / "transcript.jsonl", args.timeout
        )
        submission = workspace / "submission.py"
        if submission.is_file() and not submission.is_symlink():
            shutil.copyfile(submission, destination / "submission.py")
        if (workspace / "notes.md").is_file() and not (workspace / "notes.md").is_symlink():
            shutil.copyfile(workspace / "notes.md", destination / "notes.md")
        score_task(task, destination, args.gpu)
        scoring: Score = json.loads((destination / "score.json").read_text())
        result: TaskResult = {
            **scoring,
            "task": task,
            "author_exit": code,
            "wall_seconds": seconds,
            "kimi_turns": turns(destination / "transcript.jsonl"),
            "input_sha256": digest,
            "transcript": f"{task}/transcript.jsonl",
        }
        (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"{task}: {result['score']}/4, {seconds}s, {result['kimi_turns']} assistant turns", flush=True)
        return result
    finally:
        # Never version credentials. Remove even when scoring fails or authoring times out.
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
        "# Kimi AI-usability evaluation",
        "",
        f"Base commit: `{metadata['base_commit']}`; model: `{metadata['model']}`.",
        "",
        "| Task | Score /4 | Declares | check | CPU oracle | Tier | Seconds | Turns | Exit | Transcript |",
        "|---|---:|---|---|---|---|---:|---:|---:|---|",
    ]
    for result in results:
        cells = [
            result["task"],
            str(result["score"]),
            *("pass" if result[k] else "FAIL" for k in ("imports_declares", "check", "oracle_cpu")),
            result["explain_tier"],
            str(result["wall_seconds"]),
            str(result["kimi_turns"]),
            str(result["author_exit"]),
            f"[JSONL]({result['transcript']})",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Turns count assistant messages (including tool calls); one CLI session per task. "
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
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--gpu", action="store_true", help="also compare tensor oracles on CUDA (no engine)")
    args = parser.parse_args()
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
        "kimi_version": subprocess.check_output([str(KIMI), "--version"], text=True).strip(),
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
