"""Trusted subprocess scorer for AI submissions; never mounted into the authoring sandbox."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if "--gpu" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pllm  # noqa: E402
from pllm import testing  # noqa: E402
from tools.ai_eval import oracles, reference  # noqa: E402
from tools.ai_eval.records import Score  # noqa: E402


def config(directory: Path) -> pllm.ModelConfig:
    """Write a local tiny model config; scoring never fetches weights or a remote config."""
    model = directory / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 3,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "vocab_size": 16,
                "max_position_embeddings": 512,
                "rms_norm_eps": 1e-6,
                "torch_dtype": "float32",
                "tie_word_embeddings": False,
            }
        )
        + "\n"
    )
    return pllm.ModelConfig(
        str(model),
        dtype="float32",
        max_num_seqs=3,
        max_num_batched_tokens=16,
        execution=pllm.ExecutionOptions(publish_lag=0),
    )


def load_submission(path: Path) -> ModuleType:
    """Import a candidate through its real module name so pllm can fingerprint its functions."""
    spec = importlib.util.spec_from_file_location("ai_eval_submission", path)
    if spec is None or spec.loader is None:
        raise ValueError("write submission.py with class Workload")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate(task: str, directory: Path, submission: ModuleType, gpu: bool = False) -> Score:
    """Score independent gates and retain complete tracebacks and check skips as evidence."""
    result: Score = {
        "score": 0,
        "imports_declares": False,
        "check": False,
        "oracle_cpu": False,
        "explain_tier": "unavailable",
        "oracle_gpu": "not run",
        "failures": {},
        "category": "unreviewed",
    }
    model_config = config(directory)
    try:
        workload = submission.Workload()
        assert isinstance(workload, pllm.Workload), "export class Workload(pllm.Workload)"
    except Exception:
        result["failures"]["import"] = traceback.format_exc()
        result["score"] = 0
        return result
    for stage in ("imports_declares", "check", "oracle_cpu", "explain"):
        try:
            if stage == "imports_declares":
                oracles.declarations(task, workload, model_config)
                result["imports_declares"] = True
            elif stage == "check":
                result["check_details"] = asdict(testing.check(workload, model_config))
                result["check"] = True
            elif stage == "oracle_cpu":
                oracles.check_math(task, submission)
                result["oracle_cpu"] = True
            else:
                report = pllm.explain(workload, model_config)
                (directory / "explain.txt").write_text(str(report) + "\n")
                assert not report.budget_errors, report.budget_errors
                result["explain_tier"] = "eager" if report.eager_reason else "graph"
                result["eager_reason"] = report.eager_reason
        except Exception:
            result["failures"][stage] = traceback.format_exc()
    if gpu:
        try:
            oracles.check_math(task, submission, "cuda")
            result["oracle_gpu"] = "pass (tensor math only)"
        except Exception:
            result["oracle_gpu"] = "FAIL"
            result["failures"]["oracle_gpu"] = traceback.format_exc()
    result["score"] = sum((result["imports_declares"], result["check"], result["oracle_cpu"]))
    result["score"] += result["explain_tier"] != "unavailable"
    return result


def main() -> None:
    """Write a score even for import failures; crashes/timeouts remain runner errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=reference.WORKLOADS)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--reference", action="store_true", help="validate evaluator using a hidden reference")
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    try:
        if args.reference:
            submission = ModuleType("reference_submission")
            submission.__dict__["Workload"] = reference.WORKLOADS[args.task]
            submission.__dict__["direction"] = reference.direction
            submission.__dict__["vote"] = reference.vote
            submission.__dict__["feedback"] = reference.feedback
        else:
            submission = load_submission(directory / "submission.py")
        result = evaluate(args.task, directory, submission, args.gpu)
    except Exception:
        result = {
            "score": 0,
            "imports_declares": False,
            "check": False,
            "oracle_cpu": False,
            "explain_tier": "unavailable",
            "category": "unreviewed",
            "failures": {"import": traceback.format_exc()},
        }
    (directory / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("check_details",)}, indent=2))


if __name__ == "__main__":
    main()
