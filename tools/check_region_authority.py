#!/usr/bin/env python3
"""Check execution and native planning authority without workload-specific decisions.

Retained scheduler, ownership, sampler and worker rules cover the live Python/Rust
boundaries. FX scan classification stays independent of workload identities.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# These files form the generic authority boundary.  Do not broaden this list
# to legacy compatibility modules: that would turn this focused hard gate into
# a migration inventory and encourage exemptions.
AUTHORITY_FILES = (
    Path("pllm/execution/scan.py"),
    Path("pllm/execution/scan_analysis.py"),
    Path("pllm/execution/scan_execution.py"),
    Path("rust/pllm_core/src/planning/version_hazards.rs"),
    Path("rust/pllm_core/src/planning/state_representation.rs"),
    Path("rust/pllm_core/src/planning/scheduler.rs"),
    Path("pllm/execution/custom_forward.py"),
    Path("pllm/execution/execution.py"),
    Path("pllm/execution/module_graph.py"),
    Path("pllm/execution/paged_access.py"),
    Path("pllm/execution/tensor_borrow.py"),
    Path("pllm/execution/state_execution.py"),
    Path("pllm/authoring.py"),
    Path("pllm/backends/vllm/runtime.py"),
    Path("pllm/backends/vllm/session.py"),
    Path("pllm/backends/vllm/runner.py"),
    Path("pllm/execution/host_runtime.py"),
    Path("pllm/backends/vllm/sampler.py"),
)

# These are native binding surfaces outside the generic executor.  They must
# obey the same no-redecision rules, but intentionally are not searched for
# workload names: worker statistics and registered backend primitives may use
# those names without acquiring physical scheduling authority.
NATIVE_BINDING_FILES = (
    Path("pllm/worker.py"),
    Path("pllm/backends/vllm/custom_forward.py"),
    Path("pllm/backends/vllm/activation.py"),
    Path("pllm/backends/vllm/parameter_overlay.py"),
)

PHYSICAL_FILES = frozenset(
    {
        Path("pllm/execution/execution.py"),
        Path("pllm/execution/state_execution.py"),
    }
)

SAMPLER_FILES = frozenset(
    {
        Path("pllm/backends/vllm/sampler.py"),
    }
)

CONSUMER_NAMES = (
    "inplace_ttt",
    "esamp",
    "dummy",
    "recirculation",
    "reward_head",
    "fttt",
    "override",
    "tlm",
)

CONSUMER_PATTERN = re.compile(
    r"pllm\.consumers|(?<![A-Za-z0-9_])(?:"
    + "|".join(re.escape(name) for name in CONSUMER_NAMES)
    + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

REFCOUNT_NAMES = ("initial_refcounts", "_remaining_result_uses")


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    rule: str
    text: str

    def format(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.text.strip()}"


def _consumer_hit(text: str) -> bool:
    return CONSUMER_PATTERN.search(text) is not None


def _attribute_on(node: ast.AST, attributes: tuple[str, ...], names: tuple[str, ...]) -> bool:
    """Attribute access `name.attribute` for the given attribute and receiver names."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr in attributes
        and isinstance(node.value, ast.Name)
        and node.value.id in names
    )


class _PythonAuthority(ast.NodeVisitor):
    """One AST pass applying the no-redecision rules the file's category selects."""

    def __init__(self, path: Path, lines: list[str]) -> None:
        self.path = path
        self.lines = lines
        self.violations: list[Violation] = []
        physical = path in PHYSICAL_FILES
        self.native = path in NATIVE_BINDING_FILES
        self.prefix = "native binding" if self.native else "physical path"
        self.check_consumers = path in AUTHORITY_FILES
        self.check_redecision = physical or self.native
        self.check_eager = physical
        self.check_sampler = path in SAMPLER_FILES

    def _add(self, node: ast.expr | ast.stmt, rule: str) -> None:
        self.violations.append(Violation(self.path, node.lineno, rule, self.lines[node.lineno - 1]))

    def visit_Expr(self, node: ast.Expr) -> None:
        # Bare strings are documentation; everything else is visited normally.
        if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if self.check_consumers and _consumer_hit(node.id):
            self._add(node, "consumer authority leaked into execution core")
        if self.check_redecision and node.id in REFCOUNT_NAMES:
            self._add(node, f"{self.prefix} may not derive materialization refcounts")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self.check_consumers and (_consumer_hit(node.attr) or _attribute_on(node, ("consumers",), ("pllm",))):
            self._add(node, "consumer authority leaked into execution core")
        if self.check_redecision:
            if node.attr == "compile_safe":
                self._add(node, f"{self.prefix} may not re-decide compile safety")
            if node.attr == "compile_policy":
                self._add(node, f"{self.prefix} may not read raw compile policy")
            if node.attr in REFCOUNT_NAMES:
                self._add(node, f"{self.prefix} may not derive materialization refcounts")
        if self.check_sampler and _attribute_on(node, ("config",), ("provider",)):
            self._add(node, "sampler bridge may not infer policy from workload config")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self.check_redecision and _attribute_on(
            node.func, ("split", "partition", "rsplit"), ("executor", "implementation_id")
        ):
            self._add(node, f"{self.prefix} may not parse executor strings")
        if (
            self.check_sampler
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and (
                len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "provider"
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "config"
            )
        ):
            self._add(node, "sampler bridge may not infer policy from workload config")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if self.check_consumers and isinstance(node.value, str) and _consumer_hit(node.value):
            self._add(node, "consumer authority leaked into execution core")

    def visit_Try(self, node: ast.Try) -> None:
        if self.check_eager:
            compiled = any(
                _attribute_on(inner.func, ("compile",), ("torch",)) if isinstance(inner, ast.Call) else False
                for statement in node.body
                for inner in ast.walk(statement)
            )
            if compiled and any(
                (isinstance(inner, ast.Name) and inner.id == "eager")
                or (isinstance(inner, ast.Attribute) and inner.attr == "eager")
                or (isinstance(inner, ast.Constant) and inner.value == "eager")
                for handler in node.handlers
                for inner in ast.walk(handler)
            ):
                self._add(node, "physical path may not silently retry with eager")
        self.generic_visit(node)


def _rust_violations(path: Path, source: str) -> list[Violation]:
    """Consumer-name scan over the production half of a Rust planning file."""
    production = source.split("#[cfg(test)]", 1)[0]
    return [
        Violation(path, number, "consumer authority leaked into execution core", line)
        for number, line in enumerate(production.splitlines(), start=1)
        if _consumer_hit(line)
    ]


def collect(root: Path = REPO_ROOT) -> list[Violation]:
    violations: list[Violation] = []
    for relative_path in (*AUTHORITY_FILES, *NATIVE_BINDING_FILES):
        path = root / relative_path
        if not path.is_file():
            violations.append(Violation(relative_path, 0, "authority file is missing", ""))
            continue
        source = path.read_text(encoding="utf-8")
        if relative_path.suffix == ".rs":
            violations.extend(_rust_violations(relative_path, source))
            continue
        visitor = _PythonAuthority(relative_path, source.splitlines())
        visitor.visit(ast.parse(source, filename=str(relative_path)))
        violations.extend(visitor.violations)
    return violations


def main() -> int:
    violations = collect()
    if not violations:
        print(f"OK Execution authority gate ({len(AUTHORITY_FILES) + len(NATIVE_BINDING_FILES)} production files)")
        return 0
    print("FAIL Execution authority gate:", file=sys.stderr)
    for violation in violations:
        print(violation.format(), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
