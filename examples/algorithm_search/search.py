"""Run feedback-guided sandbox authoring or replay a scored algorithm-search ledger."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tools.ai_eval.run import AGENTS, execute, prepare, provider_failure, sandbox  # noqa: E402


def records(path: Path) -> list[dict[str, Any]]:
    """Read a ledger in order; a new search starts with no records."""
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def score(candidate: Path, directory: Path, ledger: Path) -> dict[str, Any]:
    """Score in a fresh process; candidate errors become feedback, never search crashes."""
    output = directory / "score.json"
    command = [sys.executable, str(HERE / "score.py"), str(candidate), "--output", str(output), "--known", str(ledger)]
    code, _ = execute(command, ROOT, directory / "score.log", 7200)
    result = json.loads(output.read_text()) if output.exists() else {"source": candidate.read_text()}
    if code or not output.exists():
        return {**result, "eligible": False, "failure": f"scorer exited {code}; see {directory / 'score.log'}"}
    return result


def retain(ledger: Path, result: dict[str, Any]) -> None:
    """Keep a single score per fingerprint; attempts and duplicates retain their own score.json."""
    if "duplicate_of" not in result:
        with ledger.open("a") as stream:
            stream.write(json.dumps(result, default=str) + "\n")


def best(ledger: Path) -> str:
    """Select highest gated accuracy, then lowest explain FLOPs; otherwise start from vanilla."""
    passing = [row for row in records(ledger) if row["eligible"]]
    if not passing:
        return (HERE / "candidate.py").read_text()
    return max(passing, key=lambda row: (row["accuracy"], -row["flops"]))["source"]


def replay(ledger: Path, destination: Path) -> None:
    """Re-score stored sources without an agent, requiring exact accuracy and reporting timing samples."""
    for index, record in enumerate(records(ledger)):
        if "source" not in record:
            continue
        directory = destination / str(index)
        directory.mkdir(parents=True)
        candidate = directory / "candidate.py"
        candidate.write_text(record["source"])
        result = score(candidate, directory, directory / "no-ledger.jsonl")
        pins = (
            "model_revision",
            "dataset_revision",
            "dataset_sha256",
            "question_ids",
            "seeds",
            "benchmark_options",
            "quality_options",
            "versions",
            "gpu_type",
        )
        if any(result.get(key) != record.get(key) for key in pins):
            raise RuntimeError(f"replay environment or scoring inputs changed: {directory / 'score.json'}")
        if result.get("fingerprint") != record.get("fingerprint") or bool(result.get("failure")) != bool(
            record.get("failure")
        ):
            raise RuntimeError(f"replay gate or fingerprint changed: {directory / 'score.json'}")
        if result.get("accuracy") != record.get("accuracy") or result.get("answers") != record.get("answers"):
            raise RuntimeError(f"greedy answers changed: {directory / 'score.json'}")
        if "round_rates" not in record:
            continue
        rates, previous = result["round_rates"], record["round_rates"]
        print(
            json.dumps(
                {
                    "fingerprint": result["fingerprint"],
                    "accuracy": result["accuracy"],
                    "throughput": result["throughput"],
                    "reference_throughput": record["throughput"],
                    "throughput_change": result["throughput"] / record["throughput"] - 1,
                    "round_rates": rates,
                    "reference_round_rates": previous,
                }
            ),
            flush=True,
        )


def propose(directory: Path, ledger: Path, revision: str, args: argparse.Namespace) -> dict[str, Any]:
    """Reuse ai_eval's docs export, Agent and bubblewrap runner, adding scored feedback."""
    destination = directory / "dola"  # prepare exports the ai_eval task; replace it with the search task below.
    destination.mkdir()
    workspace, _ = prepare(destination, revision)
    shutil.copyfile(HERE / "task.md", workspace / "TASK.md")
    for seed in ("dola.py", "early_exit.py"):
        shutil.copyfile(HERE / seed, workspace / seed)
    (workspace / "candidate.py").write_text(best(ledger))
    (workspace / "leaderboard.jsonl").write_text(ledger.read_text())
    agent = AGENTS[args.agent]
    model = args.model or agent.model
    # Kimi refreshes OAuth under a file lock; keep those writes in the private sandbox home.
    for name in agent.auth_files:
        source, target = Path.home() / name, destination / "home" / name
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copyfile(source, target)
    command = sandbox(destination, agent, model, authenticate=False)
    command[-1] = (
        "Read TASK.md, README.md, relevant doc/ and leaderboard.jsonl. "
        f"Submit one candidate.py variant within {args.timeout} seconds. "
        "Finish after editing it; scoring runs next and later rounds receive its feedback."
    )
    transcript = directory / "transcript.jsonl"
    code, seconds = execute(command, workspace, transcript, args.timeout)
    failure = provider_failure(transcript)
    if not failure:
        failure = next(
            (
                line
                for line in transcript.read_text().splitlines()
                if line.startswith("error:") and "authorization grant is invalid" in line
            ),
            "",
        )
    metadata: dict[str, Any] = {
        "agent": args.agent,
        "agent_model": model,
        "agent_version": subprocess.check_output([str(agent.binary), "--version"], text=True).strip(),
        "author_exit": code,
        "author_seconds": seconds,
        "docs_revision": revision,
    }
    if failure or code:
        return {
            **metadata,
            "eligible": False,
            "category": "INFRA" if failure else "AUTHOR",
            "failure": failure or f"author exited {code}",
        }
    candidate = directory / "candidate.py"
    proposal = workspace / "candidate.py"
    if not proposal.is_file() or proposal.is_symlink():
        return {**metadata, "eligible": False, "category": "AUTHOR", "failure": "write a regular candidate.py"}
    candidate.write_text(proposal.read_text())
    return {**score(candidate, directory, ledger), **metadata}


def main() -> None:
    """Score the three seeds, then run fresh feedback rounds, or replay an existing ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--agent", choices=AGENTS, default="codex")
    parser.add_argument("--model", help="author model; defaults to the ai_eval Agent model")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", type=Path, default=Path("outputs/algorithm_search"))
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.rounds < 1 or args.timeout < 1:
        parser.error("use positive --rounds and --timeout")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.replay:
        replay(args.replay.resolve(), output)
        return
    ledger = output / "ledger.jsonl"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for seed in ("candidate", "dola", "early_exit"):
        directory = output / seed
        directory.mkdir()
        candidate = directory / "candidate.py"
        shutil.copyfile(HERE / f"{seed}.py", candidate)
        retain(
            ledger, {**score(candidate, directory, ledger), "agent": "seed", "agent_model": None, "agent_version": None}
        )
    for index in range(args.rounds):
        directory = output / f"round-{index + 1}"
        directory.mkdir()
        result = propose(directory, ledger, revision, args)
        (directory / "score.json").write_text(json.dumps(result, indent=2) + "\n")
        retain(ledger, result)
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in ("fingerprint", "category", "eligible", "accuracy", "throughput_ratio", "duplicate_of")
                }
            ),
            flush=True,
        )
    if not any(row["eligible"] for row in records(ledger)):
        raise RuntimeError(f"no candidate passed all gates; inspect {ledger}")
    (output / "candidate.py").write_text(best(ledger))


if __name__ == "__main__":
    main()
