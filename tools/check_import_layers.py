#!/usr/bin/env python3
"""Reject runtime imports against the reviewed package dependency direction.

Top-level public modules and TYPE_CHECKING bodies are exempt. Function-local
imports follow the same rules as module imports; there is no migration baseline.
"""

from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = REPO_ROOT / "pllm"

# importer package -> allowed internal targets (besides itself).
RULES: dict[str, frozenset[str]] = {
    "contracts": frozenset(),
    # `config` holds plain option values with no package dependencies.
    "execution": frozenset({"config", "contracts", "kernels"}),
    "kernels": frozenset({"contracts"}),
    # Workloads may also use the public grouped API modules (pllm.host, pllm.model, pllm.io).
    "workloads": frozenset({"authoring", "config", "contracts", "execution", "kernels", "host", "model", "io"}),
    "backends": frozenset({"authoring", "config", "contracts", "execution"}),
}


def _scan_file(path: Path) -> list[str]:
    relative = path.relative_to(PKG_ROOT)
    if len(relative.parts) == 1:
        return []
    importer = relative.parts[0]
    if importer not in RULES:
        return [f"{path}:1: undeclared pllm package {importer}"]
    package = ".".join(("pllm", *relative.parent.parts))
    allowed = RULES[importer] | {importer, "_core"}
    violations: list[str] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.If) and (
            isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
            or isinstance(node.test, ast.Attribute)
            and isinstance(node.test.value, ast.Name)
            and node.test.value.id == "typing"
            and node.test.attr == "TYPE_CHECKING"
        ):
            for child in node.orelse:
                visit(child)
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            else:
                target = resolve_name("." * node.level + (node.module or ""), package)
                targets = [target]
                if target == "pllm":
                    targets = [
                        f"pllm.{alias.name}"
                        for alias in node.names
                        if (PKG_ROOT / alias.name).is_dir() or (PKG_ROOT / f"{alias.name}.py").is_file()
                    ]
            for target in targets:
                parts = target.split(".")
                if importer in ("contracts", "execution") and parts[0] in ("vllm", "transformers"):
                    violations.append(
                        f"{path}:{node.lineno}: {importer} must use backend contracts instead of importing {target}"
                    )
                if len(parts) > 1 and parts[0] == "pllm" and parts[1] not in allowed:
                    violations.append(f"{path}:{node.lineno}: {importer} cannot import {target}")
                if importer in ("contracts", "execution") and target.startswith("torch._inductor"):
                    violations.append(f"{path}:{node.lineno}: Inductor internals belong in the tensor compiler backend")
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    return violations


def main() -> int:
    violations = [violation for path in sorted(PKG_ROOT.rglob("*.py")) for violation in _scan_file(path)]
    print("\n".join(violations) if violations else "OK import layers: zero runtime violations")
    return int(bool(violations))


if __name__ == "__main__":
    raise SystemExit(main())
