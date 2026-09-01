#!/usr/bin/env python3
"""Typing gate: strict tier must stay at zero errors; basic tier only decreases.

Usage (from the repo root, venv activated)::

    python tools/check_typing.py            # check both tiers
    python tools/check_typing.py --rebase   # lock in a strictly non-regressing baseline

Tiers:
  - strict: ``pyrightconfig.strict.json`` over core contracts plus the workload
    DSL, typed kit/runtime bindings, migrated native workloads, and their public
    composition fixture. Hard gate: 0 errors, 0 warnings.
  - basic: ``pyrightconfig.json`` over the rest of tllm/ (vllm_patch and
    backends excluded as the sanctioned getattr quarantine). Errors are
    bucketed per directory and compared against ``tools/typing_baseline.json``;
    any bucket exceeding its baseline count (or any new bucket) fails the gate.

Dynamic-boundary allowlist:
  - ``worker_extension._runner_specialization_facts`` may probe the two
    version-dependent vLLM token-limit attributes. It returns a closed typed
    facts object or fails closed; reflection cannot escape that function.

All other workload/native-runtime files reject ``Any``, reflection, and type
ignore escapes through the AST gate below.
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
CONFIG_KIND_NEGATIVE_FIXTURE = Path("test/typing/config_kind_negative.py")
WORKLOAD_SURFACE = tuple(
    path.relative_to(REPO_ROOT)
    for path in sorted((REPO_ROOT / "tllm" / "workload").glob("*.py"))
)
AUTHOR_SURFACE = (*WORKLOAD_SURFACE,
    Path("tllm/kit/backend.py"),
    Path("tllm/kit/workload.py"),
    Path("tllm/consumers/esamp/workload.py"),
    Path("tllm/consumers/fttt/feedback.py"),
    Path("tllm/consumers/fttt/workload.py"),
    Path("tllm/consumers/override/workload.py"),
    Path("tllm/consumers/reft/workload.py"),
    Path("tllm/consumers/ttrl/workload.py"),
    Path("tllm/consumers/tlm/workload.py"),
    Path("tllm/consumers/recirculation/workload.py"),
    Path("tllm/consumers/recirculation/stage_definition.py"),
    Path("tllm/consumers/recirculation/stage_binding.py"),
    Path("tllm/backends/vllm/typed_layer_replay.py"),
    Path("tllm/runtime/declaration_adapter.py"),
    Path("tllm/runtime/backend_region_binders.py"),
    Path("tllm/runtime/mp/worker_extension.py"),
    Path("tllm/runtime/parameter_overlay.py"),
    Path("tllm/runtime/packed_hidden_snapshot.py"),
    Path("tllm/runtime/packed_phase_row_staging.py"),
    Path("tllm/runtime/runtime.py"),
    Path("tllm/runtime/workload_bindings.py"),
    Path("tllm/runtime/workload_execution.py"),
    Path("tllm/runtime/workload_plugins.py"),
)

FORBIDDEN_COMPATIBILITY_NAMES = frozenset(
    {
        "NativeWorkloadResources",
        "ReconstructedWorkload",
        "workload_output_sinks",
        "workload_provided_inputs",
        "workload_state_snapshots",
        "_config_kind",
    }
)

# Exact foreign-object probes. Entries are (path, enclosing function, call).
DYNAMIC_BOUNDARY_ALLOWLIST = frozenset(
    {
        (
            Path("tllm/backends/vllm/typed_layer_replay.py"),
            "_resolve_qwen_like_model",
            "getattr",
        ),
        (
            Path("tllm/backends/vllm/typed_layer_replay.py"),
            "_resolve_qwen_like_model",
            "hasattr",
        ),
        (
            Path("tllm/backends/vllm/typed_layer_replay.py"),
            "_metadata_replace_kwargs",
            "getattr",
        ),
        (
            Path("tllm/backends/vllm/typed_layer_replay.py"),
            "_metadata_replace_kwargs",
            "hasattr",
        ),
        (
            Path("tllm/backends/vllm/typed_layer_replay.py"),
            "_admit_typed_layer_replay",
            "getattr",
        ),
        (
            Path("tllm/backends/vllm/typed_layer_replay.py"),
            "install_typed_layer_replay",
            "setattr",
        ),
        (
            Path("tllm/backends/vllm/typed_layer_replay.py"),
            "signed_ordered_row_fusion_policy",
            "getattr",
        ),
        (
            Path("tllm/backends/vllm/typed_layer_replay.py"),
            "abort_typed_layer_replay",
            "getattr",
        ),
        (
            Path("tllm/backends/vllm/typed_layer_replay.py"),
            "restore_typed_layer_replay",
            "getattr",
        ),
        (
            Path("tllm/backends/vllm/typed_layer_replay.py"),
            "restore_typed_layer_replay",
            "setattr",
        ),
        (
            Path("tllm/runtime/mp/worker_extension.py"),
            "_runner_specialization_facts",
            "getattr",
        ),
        (
            Path("tllm/runtime/workload_plugins.py"),
            "_resolve_factory",
            "getattr",
        ),
        (
            Path("tllm/runtime/workload_plugins.py"),
            "_resolve_factory",
            "importlib.import_module",
        ),
    }
)

# Basic-tier warnings are not mixed into the historical error baseline. These
# two exact legacy categories are separately only-decrease; new files/rules or
# a larger count fail. ``--rebase`` deliberately cannot update this map.
BASIC_WARNING_BASELINE = {
    ("tllm/consumers/_common/bundle_inputs.py", "reportInvalidTypeVarUse"): 1,
    ("tllm/consumers/esamp/__init__.py", "reportUnsupportedDunderAll"): 9,
}


class _TypingRuleVisitor(ast.NodeVisitor):
    def __init__(self, relative: Path) -> None:
        self.relative = relative
        self.function = "<module>"
        self.class_name = "<module>"
        self.violations: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        wraps_runtime_type_check = any(
            isinstance(descendant, ast.Call)
            and isinstance(descendant.func, ast.Name)
            and descendant.func.id == "isinstance"
            for descendant in ast.walk(node)
        )
        if node.name.startswith("_is_") or (
            node.name.startswith("_require_") and wraps_runtime_type_check
        ):
            self.violations.append(
                f"{self.relative}:{node.lineno}: type-check wrapper helpers are forbidden"
            )
        if self.class_name == "TllmRuntime" and node.name == "local_workload":
            parameter = node.args.args[1] if len(node.args.args) > 1 else None
            parameter_type = (
                ast.unparse(parameter.annotation)
                if parameter is not None and parameter.annotation is not None
                else ""
            )
            return_type = ast.unparse(node.returns) if node.returns is not None else ""
            if parameter_type != "ConfiguredWorkloadT" or return_type != "ConfiguredWorkloadT":
                self.violations.append(
                    f"{self.relative}:{node.lineno}: local_workload must preserve the "
                    "runtime workload-set TypeVar"
                )
        previous = self.function
        self.function = node.name
        self.generic_visit(node)
        self.function = previous

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        previous = self.class_name
        self.class_name = node.name
        self.generic_visit(node)
        self.class_name = previous

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if (
            self.class_name == "WorkloadExecutionBindings"
            and isinstance(node.target, ast.Name)
            and node.target.id in {"provided_inputs", "output_sinks"}
        ):
            self.violations.append(
                f"{self.relative}:{node.lineno}: execution callback maps must use "
                "RuntimePortRegistry"
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "Any":
            self.violations.append(
                f"{self.relative}:{node.lineno}: Any is forbidden"
            )
        if node.id in FORBIDDEN_COMPATIBILITY_NAMES:
            self.violations.append(
                f"{self.relative}:{node.lineno}: obsolete compatibility surface "
                f"{node.id} is forbidden"
            )

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in {
            "getattr",
            "setattr",
            "hasattr",
        }:
            key = (self.relative, self.function, node.func.id)
            if key not in DYNAMIC_BOUNDARY_ALLOWLIST:
                self.violations.append(
                    f"{self.relative}:{node.lineno}: {node.func.id} is forbidden"
                )
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "importlib"
            and node.func.attr == "import_module"
        ):
            key = (self.relative, self.function, "importlib.import_module")
            if key not in DYNAMIC_BOUNDARY_ALLOWLIST:
                self.violations.append(
                    f"{self.relative}:{node.lineno}: dynamic import is forbidden"
                )
        self.generic_visit(node)


def _author_surface_violations() -> list[str]:
    violations: list[str] = []
    for relative in AUTHOR_SURFACE:
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for line_number, line in enumerate(source.splitlines(), start=1):
            if "# type: ignore" in line or "pyright: ignore" in line:
                violations.append(f"{relative}:{line_number}: type-ignore is forbidden")
        tree = ast.parse(source, filename=str(relative))
        visitor = _TypingRuleVisitor(relative)
        visitor.visit(tree)
        violations.extend(visitor.violations)
    return violations


def _run_pyright(
    config: str, paths: tuple[Path, ...] = ()
) -> dict[str, object]:
    pyright = shutil.which("pyright")
    if pyright is None:
        print("error: `pyright` not found on PATH; activate the project venv first", file=sys.stderr)
        raise SystemExit(2)
    proc = subprocess.run(
        [
            pyright,
            "--project",
            config,
            *(str(path) for path in paths),
            "--outputjson",
        ],
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


def _warning_counts(report: dict[str, object]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for diag in report.get("generalDiagnostics", []):  # type: ignore[union-attr]
        if diag["severity"] != "warning":
            continue
        relative = str(diag["file"]).split(f"{REPO_ROOT}/")[-1]
        counts[(relative, str(diag.get("rule", "<unknown>")))] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebase",
        action="store_true",
        help="lock in a basic-tier baseline only when no bucket regresses",
    )
    args = parser.parse_args()

    strict_report = _run_pyright(STRICT_CONFIG)
    strict_summary = strict_report.get("summary", {})  # type: ignore[assignment]
    strict_errors = int(strict_summary.get("errorCount", -1))  # type: ignore[union-attr]
    strict_warnings = int(strict_summary.get("warningCount", -1))  # type: ignore[union-attr]
    if strict_errors or strict_warnings:
        print(f"FAIL strict tier ({STRICT_CONFIG}): {strict_errors} errors, {strict_warnings} warnings (must be 0/0)")
        return 1
    print(f"OK   strict tier ({STRICT_CONFIG}): 0 errors, 0 warnings")

    negative_report = _run_pyright(
        STRICT_CONFIG, (CONFIG_KIND_NEGATIVE_FIXTURE,)
    )
    negative_diagnostics = negative_report.get("generalDiagnostics", [])
    rejected_lines = {
        int(diagnostic["range"]["start"]["line"]) + 1
        for diagnostic in negative_diagnostics
        if diagnostic["severity"] == "error"
        and str(diagnostic["file"]).endswith(str(CONFIG_KIND_NEGATIVE_FIXTURE))
    }
    expected_rejections = {6, 10, 11, 12, 13, 14}
    if not expected_rejections.issubset(rejected_lines):
        print(
            "FAIL config-kind negative fixture: expected static rejections on "
            f"lines {sorted(expected_rejections)}, got {sorted(rejected_lines)}"
        )
        return 1
    print("OK   config-kind negative fixture: all mismatches rejected statically")

    author_violations = _author_surface_violations()
    if author_violations:
        print("FAIL author surface typing rules:")
        print("\n".join(f"  {violation}" for violation in author_violations))
        return 1
    print("OK   author surface: no Any, reflection, or type-ignore escapes")

    basic_report = _run_pyright(BASIC_CONFIG)
    warning_counts = _warning_counts(basic_report)
    warning_regressions = [
        f"  {path} [{rule}]: {count} > baseline "
        f"{BASIC_WARNING_BASELINE.get((path, rule), 0)}"
        for (path, rule), count in sorted(warning_counts.items())
        if count > BASIC_WARNING_BASELINE.get((path, rule), 0)
    ]
    if warning_regressions:
        print(f"FAIL basic tier ({BASIC_CONFIG}): warning baseline regressions:")
        print("\n".join(warning_regressions))
        return 1
    basic_counts = _bucket_errors(basic_report)
    total = sum(basic_counts.values())
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    baseline_buckets: dict[str, int] = baseline["buckets"]

    if args.rebase:
        regressions = [
            f"  {bucket}: {count} > baseline {baseline_buckets.get(bucket, 0)}"
            for bucket, count in sorted(basic_counts.items())
            if count > baseline_buckets.get(bucket, 0)
        ]
        if total > int(baseline["total"]) or regressions:
            print("FAIL refusing to rebase an increased basic typing baseline:")
            print("\n".join(regressions))
            return 1
        baseline = {
            "comment": "Basic-tier pyright baseline (only-decrease). --rebase refuses total or per-bucket regressions.",
            "total": total,
            "buckets": dict(sorted(basic_counts.items())),
        }
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        print(f"rebased {BASELINE_PATH.relative_to(REPO_ROOT)}: total {total}")
        return 0

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
    print(
        f"OK   basic warnings: {sum(warning_counts.values())} "
        f"(only-decrease baseline {sum(BASIC_WARNING_BASELINE.values())})"
    )
    if decreased > 0:
        print("hint: baseline improved — consider `python tools/check_typing.py --rebase` to lock in the progress")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
