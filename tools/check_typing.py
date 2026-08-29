#!/usr/bin/env python3
"""Typing gate: strict tier must stay at zero errors; basic tier only decreases.

Usage (from the repo root, venv activated)::

    python tools/check_typing.py            # check both tiers
    python tools/check_typing.py --rebase   # record current basic tier as the new baseline

Tiers:
  - strict: ``pyrightconfig.strict.json`` over core contracts plus the workload
    DSL, typed kit/runtime bindings, migrated native workloads, and their public
    composition fixture. Hard gate: 0 errors, 0 warnings.
  - basic: ``pyrightconfig.json`` over the rest of tllm/ (vllm_patch and
    backends excluded as the sanctioned getattr quarantine). Errors are
    bucketed per directory and compared against ``tools/typing_baseline.json``;
    any bucket exceeding its baseline count (or any new bucket) fails the gate.

See ``docs/typing_rules.md`` (untracked) for the rationale.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STRICT_CONFIG = "pyrightconfig.strict.json"
BASIC_CONFIG = "pyrightconfig.json"
BASELINE_PATH = REPO_ROOT / "tools" / "typing_baseline.json"
AUTHOR_SURFACE = (
    Path("tllm/workload/__init__.py"),
    Path("tllm/workload/frontend.py"),
    Path("tllm/workload/config.py"),
    Path("tllm/workload/staging.py"),
    Path("tllm/workload/host.py"),
    Path("tllm/kit/backend.py"),
    Path("tllm/kit/workload.py"),
    Path("tllm/consumers/dummy/workload.py"),
    Path("tllm/consumers/esamp/workload.py"),
    Path("tllm/consumers/override/workload.py"),
    Path("tllm/consumers/reft/workload.py"),
    Path("tllm/consumers/ttrl/workload.py"),
    Path("tllm/consumers/recirculation/workload.py"),
    Path("tllm/consumers/recirculation/stage_definition.py"),
    Path("tllm/consumers/recirculation/stage_binding.py"),
)


def _author_surface_violations() -> list[str]:
    violations: list[str] = []
    for relative in AUTHOR_SURFACE:
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for line_number, line in enumerate(source.splitlines(), start=1):
            if "# type: ignore" in line or "pyright: ignore" in line:
                violations.append(f"{relative}:{line_number}: type-ignore is forbidden")
        tree = ast.parse(source, filename=str(relative))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "Any":
                violations.append(f"{relative}:{node.lineno}: Any is forbidden")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"getattr", "setattr"}
            ):
                violations.append(
                    f"{relative}:{node.lineno}: {node.func.id} is forbidden"
                )
    return violations


def _run_pyright(config: str) -> dict[str, object]:
    pyright = shutil.which("pyright")
    if pyright is None:
        print("error: `pyright` not found on PATH; activate the project venv first", file=sys.stderr)
        raise SystemExit(2)
    proc = subprocess.run(
        [pyright, "--project", config, "--outputjson"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"error: pyright did not emit JSON for {config}:\n{proc.stdout}\n{proc.stderr}", file=sys.stderr)
        raise SystemExit(2) from None


def _bucket_errors(report: dict[str, object]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for diag in report.get("generalDiagnostics", []):  # type: ignore[union-attr]
        if diag["severity"] != "error":
            continue
        parts = str(diag["file"]).split("/tllm/")[-1].split("/")
        bucket = "tllm/" + "/".join(parts[:2]) if len(parts) > 1 else str(diag["file"])
        counts[bucket] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebase", action="store_true", help="record the current basic tier as the new baseline")
    args = parser.parse_args()

    strict_report = _run_pyright(STRICT_CONFIG)
    strict_summary = strict_report.get("summary", {})  # type: ignore[assignment]
    strict_errors = int(strict_summary.get("errorCount", -1))  # type: ignore[union-attr]
    strict_warnings = int(strict_summary.get("warningCount", -1))  # type: ignore[union-attr]
    if strict_errors or strict_warnings:
        print(f"FAIL strict tier ({STRICT_CONFIG}): {strict_errors} errors, {strict_warnings} warnings (must be 0/0)")
        return 1
    print(f"OK   strict tier ({STRICT_CONFIG}): 0 errors, 0 warnings")

    author_violations = _author_surface_violations()
    if author_violations:
        print("FAIL author surface typing rules:")
        print("\n".join(f"  {violation}" for violation in author_violations))
        return 1
    print("OK   author surface: no Any, reflection, or type-ignore escapes")

    basic_counts = _bucket_errors(_run_pyright(BASIC_CONFIG))
    total = sum(basic_counts.values())

    if args.rebase:
        baseline = {
            "comment": "Basic-tier pyright baseline (only-decrease). Regenerate with `python tools/check_typing.py --rebase`.",
            "total": total,
            "buckets": dict(sorted(basic_counts.items())),
        }
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        print(f"rebased {BASELINE_PATH.relative_to(REPO_ROOT)}: total {total}")
        return 0

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    baseline_buckets: dict[str, int] = baseline["buckets"]
    violations = [
        f"  {bucket}: {count} > baseline {baseline_buckets.get(bucket, 0)}"
        for bucket, count in sorted(basic_counts.items())
        if count > baseline_buckets.get(bucket, 0)
    ]
    if violations:
        print(f"FAIL basic tier ({BASIC_CONFIG}): {total} errors; buckets exceeding the only-decrease baseline:")
        print("\n".join(violations))
        print("fix the new errors, or run `python tools/check_typing.py --rebase` only when the total went down")
        return 1
    decreased = int(baseline["total"]) - total
    print(f"OK   basic tier ({BASIC_CONFIG}): {total} errors (baseline {baseline['total']}, decreased by {max(decreased, 0)})")
    if decreased > 0:
        print("hint: baseline improved — consider `python tools/check_typing.py --rebase` to lock in the progress")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
