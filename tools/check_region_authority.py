#!/usr/bin/env python3
"""Fail-closed authority gate for the Region v1 production path.

The Region executor is intentionally generic.  Workload names, task-level
partitioning, compiler-policy selection, and executor-string grammars belong
to the signed Rust decision or to leaf workload/backend adapters, never to the
physical lowering path.  This checker scans only the small, production-owned
surface listed below.  It deliberately excludes tests and compatibility
adapters, where named fixtures are legitimate.

Usage: ``.venv/bin/python tools/check_region_authority.py``.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

# These files form the generic authority boundary.  Do not broaden this list
# to legacy compatibility modules: that would turn this focused hard gate into
# a migration inventory and encourage exemptions.
AUTHORITY_FILES = (
    Path("tllm/canonical/execution.py"),
    Path("tllm/canonical/state_execution.py"),
    Path("tllm/canonical/host_runtime.py"),
    Path("tllm/canonical_vllm/sampler.py"),
)

# These are native binding surfaces outside the generic executor.  They must
# obey the same no-redecision rules, but intentionally are not searched for
# workload names: worker statistics and registered backend primitives may use
# those names without acquiring physical scheduling authority.
NATIVE_BINDING_FILES = (
    Path("tllm/backends/vllm/typed_layer_replay.py"),
)

PHYSICAL_FILES = frozenset(
    {
        Path("tllm/canonical/execution.py"),
        Path("tllm/canonical/state_execution.py"),
    }
)

COMPILER_DISPATCH_FILES = frozenset(
    {
        *PHYSICAL_FILES,
    }
)

SAMPLER_FILES = frozenset(
    {
        Path("tllm/canonical_vllm/sampler.py"),
    }
)

CONSUMER_NAMES = (
    "esamp",
    "dummy",
    "recirculation",
    "ttrl",
    "fttt",
    "gtta",
    "override",
    "tlm",
)


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    rule: str
    text: str

    def format(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.text.strip()}"


def _production_rust(source: str) -> str:
    """Drop the test module before applying production authority rules."""
    marker = "#[cfg(test)]"
    return source.split(marker, 1)[0]


def _line_violations(path: Path, source: str) -> list[Violation]:
    if path.suffix == ".rs":
        source = _production_rust(source)
    violations: list[Violation] = []
    consumer_pattern = re.compile(
        r"tllm\.consumers|(?<![A-Za-z0-9_])(?:"
        + "|".join(re.escape(name) for name in CONSUMER_NAMES)
        + r")(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )
    dispatch_patterns = (
        ("physical path may not re-decide compile safety", r"\.compile_safe\b"),
        ("physical path may not read raw compile policy", r"\.compile_policy\b"),
    )
    physical_patterns = (
        (
            "physical path may not inspect Region task membership",
            r"\bregion\.tasks\b|\bexecution_region\.tasks\b",
        ),
        (
            "physical path may not parse executor strings",
            r"\bexecutor\s*\.\s*(split|partition|rsplit)\s*\(|"
            r"\bimplementation_id\s*\.\s*(split|partition|rsplit)\s*\(",
        ),
        (
            "native Region path may not inspect legacy adapter ownership",
            r"\.legacy_consumer\b",
        ),
        (
            "physical path may not silently retry with eager",
            r"try:[\s\S]{0,240}torch\.compile[\s\S]{0,240}"
            r"except[^\n]*:[\s\S]{0,240}\beager\b",
        ),
        (
            "physical path may not derive materialization refcounts",
            r"len\s*\(\s*binding\.consumer_regions\b|"
            r"\binitial_refcounts\b|\b_remaining_result_uses\b",
        ),
    )
    for line_number, line in enumerate(source.splitlines(), start=1):
        if consumer_pattern.search(line):
            violations.append(
                Violation(path, line_number, "consumer authority leaked into Region core", line)
            )
        patterns = ()
        if path in COMPILER_DISPATCH_FILES:
            patterns += dispatch_patterns
        if path in PHYSICAL_FILES:
            patterns += physical_patterns[:3] + (physical_patterns[4],)
        if path in SAMPLER_FILES:
            patterns += (
                (
                    "sampler bridge may not infer policy from workload config",
                    r"\bprovider\s*\.\s*config\b|"
                    r"getattr\s*\(\s*provider\s*,\s*[\"']config[\"']",
                ),
            )
        for rule, pattern in patterns:
            if re.search(pattern, line):
                violations.append(Violation(path, line_number, rule, line))
    if path in PHYSICAL_FILES:
        eager_retry_rule, eager_retry_pattern = physical_patterns[3]
        for match in re.finditer(eager_retry_pattern, source):
            line_number = source.count("\n", 0, match.start()) + 1
            line = source.splitlines()[line_number - 1]
            violations.append(
                Violation(path, line_number, eager_retry_rule, line)
            )
    return violations


def _native_binding_violations(path: Path, source: str) -> list[Violation]:
    """Apply physical no-redecision checks to native worker/backend binders."""

    violations: list[Violation] = []
    patterns = (
        ("native binding may not re-decide compile safety", r"\.compile_safe\b"),
        ("native binding may not read raw compile policy", r"\.compile_policy\b"),
        (
            "native binding may not inspect Region task membership",
            r"\bregion\.tasks\b|\bexecution_region\.tasks\b",
        ),
        (
            "native binding may not parse executor strings",
            r"\bexecutor\s*\.\s*(split|partition|rsplit)\s*\(|"
            r"\bimplementation_id\s*\.\s*(split|partition|rsplit)\s*\(",
        ),
        (
            "native binding may not derive materialization refcounts",
            r"len\s*\(\s*binding\.consumer_regions\b|"
            r"\binitial_refcounts\b|\b_remaining_result_uses\b",
        ),
    )
    for line_number, line in enumerate(source.splitlines(), start=1):
        for rule, pattern in patterns:
            if re.search(pattern, line):
                violations.append(Violation(path, line_number, rule, line))
    return violations


def collect(root: Path = REPO_ROOT) -> list[Violation]:
    violations: list[Violation] = []
    for relative_path in AUTHORITY_FILES:
        path = root / relative_path
        if not path.is_file():
            violations.append(
                Violation(relative_path, 0, "authority file is missing", "")
            )
            continue
        violations.extend(
            _line_violations(relative_path, path.read_text(encoding="utf-8")))
    for relative_path in NATIVE_BINDING_FILES:
        path = root / relative_path
        if not path.is_file():
            violations.append(
                Violation(relative_path, 0, "native binding file is missing", "")
            )
            continue
        violations.extend(
            _native_binding_violations(
                relative_path, path.read_text(encoding="utf-8")
            )
        )
    return violations


def main() -> int:
    violations = collect()
    if not violations:
        print(
            "OK Region authority gate "
            "("
            f"{len(AUTHORITY_FILES) + len(NATIVE_BINDING_FILES)} "
            "production files)"
        )
        return 0
    print("FAIL Region authority gate:", file=sys.stderr)
    for violation in violations:
        print(violation.format(), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
