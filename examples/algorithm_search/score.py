"""Score search candidates in fresh CPU and locked GPU processes; used by search.py."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
import urllib.request
from dataclasses import asdict, replace
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CUDA_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
if "--gpu" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402
from vllm import SamplingParams  # noqa: E402

import pllm  # noqa: E402
from pllm.backends.vllm.compile_facts import compile_facts  # noqa: E402
from pllm.backends.vllm.model_shape import ModelShape  # noqa: E402
from pllm.benchmark import BenchmarkOptions, compare  # noqa: E402
from pllm.execution.checkpoint import checkpoint_directory  # noqa: E402
from pllm.execution.freeze import fingerprint  # noqa: E402
from pllm.testing import check, check_on_gpu  # noqa: E402
from tools.ai_eval.score import load_submission  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DATA_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
DATA_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
OPTIONS = BenchmarkOptions(gpu_memory_utilization=0.7, cuda_graph_pool_budget_bytes=8 << 30)


def config() -> pllm.ModelConfig:
    """Use the aligned benchmark geometry with pinned weights and automatic graph admission."""
    return replace(
        OPTIONS.model_config(),
        revision=MODEL_REVISION,
        cuda_graphs="auto",
        execution=pllm.ExecutionOptions(cuda_graph_pool_budget_bytes=OPTIONS.cuda_graph_pool_budget_bytes),
    )


def identity(workload: pllm.Workload) -> str:
    """Use pllm's signed declaration, which includes method code and workload values."""
    requirements = workload.backend_requirements()
    shape = ModelShape.of_config(config(), requirements.parameter_targets)
    facts = compile_facts(
        shape,
        requirements,
        config(),
        model_identity=shape.architecture,
        hidden_layouts=(),
        overlay_graph_bytes=0,
        memory_bytes=1 << 40,
    )
    return workload.build(facts).graph.execution_parameters_digest


def cpu(candidate: Path, known: Path, result: dict[str, Any]) -> None:
    """Import, deduplicate, explain and enforce the CPU contract gate before leasing a GPU."""
    result.update(
        {
            "source": candidate.read_text(),
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "dataset": "openai/grade-school-math/test",
            "dataset_revision": DATA_REVISION,
            "dataset_sha256": DATA_SHA256,
            "question_ids": list(range(200)),
            "seeds": {"model": 7, "sampling": 7, "contracts": "pllm.testing fixed script"},
            "versions": {"torch": torch.__version__, "vllm": version("vllm"), "python": sys.version},
            "benchmark_options": asdict(OPTIONS),
            "quality_options": {"temperature": 0, "max_tokens": 512, "max_model_len": 2048, "batch": 8},
            "eligible": False,
        }
    )
    submission = load_submission(candidate)
    workload: object = submission.WORKLOAD
    neutral: object = submission.NEUTRAL
    if not isinstance(workload, pllm.Workload) or not isinstance(neutral, pllm.Workload):
        raise ValueError("define WORKLOAD and NEUTRAL as pllm.Workload instances")
    result["fingerprint"] = fingerprint((identity(workload), identity(neutral)))
    for line in known.read_text().splitlines() if known.exists() else ():
        record = json.loads(line)
        if record.get("fingerprint") == result["fingerprint"]:
            result.update(record, duplicate_of=record["fingerprint"], source=candidate.read_text())
            return
    report = pllm.explain(workload, config())
    result.update(explain=str(report), explain_details=asdict(report))
    estimates = [task.estimate for member in report.members for task in member.tasks]
    base_flops = report.model.base_forward_flops_per_token_estimate
    assert base_flops is not None
    result["flops"] = base_flops + sum(
        (cost.matmul_flops_estimate + cost.projection_flops_estimate) / cost.rows
        for cost in estimates
        if cost is not None
    )
    if report.budget_errors:
        raise ValueError(report.budget_errors)
    result["check"] = str(check(workload, config(), neutral=neutral))


def answer(text: str) -> str:
    """Normalize the last number, the usual GSM8K exact numeric match convention."""
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return str(Decimal(numbers[-1].replace(",", "")).normalize()) if numbers else ""


def accuracy(workload: pllm.Workload) -> tuple[float, list[str]]:
    """Generate the pinned first 200 questions greedily with a fixed Qwen chat prompt."""
    url = (
        f"https://raw.githubusercontent.com/openai/grade-school-math/{DATA_REVISION}/grade_school_math/data/test.jsonl"
    )
    data = urllib.request.urlopen(url, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != DATA_SHA256:
        raise ValueError("GSM8K content changed; restore the pinned dataset")
    questions = [json.loads(line) for line in data.splitlines()[:200]]
    prompts = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        + row["question"]
        + "\nSolve step by step. End with the numeric answer."
        + "<|im_end|>\n<|im_start|>assistant\n"
        for row in questions
    ]
    quality_config = replace(config(), max_model_len=2048, max_num_seqs=8)
    with pllm.make_llm(quality_config, workload=workload) as llm:
        outputs = llm.generate(prompts, SamplingParams(temperature=0, seed=7, max_tokens=512))
        answers = [answer(output.outputs[0].text) for output in outputs]
    correct = sum(value == answer(row["answer"]) for value, row in zip(answers, questions, strict=True))
    return correct / len(questions), answers


def throughput(workload: pllm.Workload, eager: bool, log: Path) -> dict[str, Any]:
    """Reuse benchmark.compare's aligned measurements, retaining all three round rates."""
    # The revision is pinned even though BenchmarkOptions has no revision field.
    options = replace(OPTIONS, model=checkpoint_directory(MODEL, MODEL_REVISION), cuda_graphs=not eager)
    with log.open("w") as stream, contextlib.redirect_stdout(stream):
        compare(workload, options)
        if eager:
            compare(
                load_submission(Path(__file__).with_name("candidate.py")).WORKLOAD, replace(options, cuda_graphs=True)
            )
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.startswith("{")]
    candidate = next(row for row in rows if row.get("implementation") == "tensor")
    vanilla = next(row for row in rows if row.get("implementation") == "vanilla_graph")
    return {
        "throughput": candidate["output_tokens_per_second"],
        "round_rates": candidate["round_rates"],
        "vanilla_round_rates": vanilla["round_rates"],
        "throughput_ratio": candidate["output_tokens_per_second"] / vanilla["output_tokens_per_second"],
    }


def gpu(candidate: Path, result: dict[str, Any], output: Path) -> None:
    """Run real-engine contracts, then quality and aligned throughput under the caller's GPU lock."""
    submission = load_submission(candidate)
    result["gpu_type"] = torch.cuda.get_device_name()
    result["check_on_gpu"] = str(check_on_gpu(submission.WORKLOAD, config(), neutral=submission.NEUTRAL))
    result["accuracy"], result["answers"] = accuracy(submission.WORKLOAD)
    result.update(
        throughput(
            submission.WORKLOAD, bool(result["explain_details"]["eager_reason"]), output.with_suffix(".benchmark.txt")
        )
    )
    result["eligible"] = result["throughput_ratio"] >= 0.9


def main() -> None:
    """Persist gate failures and diagnostics, including when a GPU subprocess fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, default=Path("score.json"))
    parser.add_argument("--known", type=Path, default=Path("ledger.jsonl"))
    parser.add_argument("--gpu", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    result: dict[str, Any] = json.loads(output.read_text()) if args.gpu else {"eligible": False}
    try:
        if args.gpu:
            gpu(args.candidate.resolve(), result, output)
        else:
            cpu(args.candidate.resolve(), args.known, result)
    except Exception:
        result.update(eligible=False, failure=traceback.format_exc())
    output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    if not args.gpu and "check" in result and "failure" not in result and "duplicate_of" not in result:
        environment = dict(os.environ)
        if CUDA_DEVICES is None:
            environment.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            environment["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICES
        environment.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "1")
        environment.setdefault("OMP_NUM_THREADS", "1")
        command = [
            "flock",
            "/tmp/pllm-gpu.lock",
            "taskset",
            "-c",
            "0-15",
            sys.executable,
            str(Path(__file__).resolve()),
            str(args.candidate.resolve()),
            "--gpu",
            "--output",
            str(output),
        ]
        completed = subprocess.run(command, env=environment, check=False)
        if completed.returncode:
            result.update(eligible=False, failure=f"GPU process exited {completed.returncode}")
            output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(output)


if __name__ == "__main__":
    main()
