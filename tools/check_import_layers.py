#!/usr/bin/env python3
"""Import-layer gate: cross-package imports must follow the declared direction.

Usage (from the repo root, venv activated)::

    python tools/check_import_layers.py            # check against the baseline
    python tools/check_import_layers.py --rebase   # record current violations as the baseline
    python tools/check_import_layers.py --list     # print every violating import

Target direction (a package may import only the packages listed in RULES)::

    core ← contracts ← canonical
                     ← ports ← common
                         ← producers ← workgraph ← runtime
    backends (contracts/ports/workgraph only; runtime types arrive via protocols)
    consumers / workflows: leaves, may import anything below

The workload compiler has two deliberately narrow bridges, declared in
``EXACT_MODULE_EDGES`` rather than at package scope:

    workload.lowering / workload.region_compiler → schedule.contracts
    schedule.coordinator → workload frontend/lowering

This keeps the frontend independent from the compiler, permits the typed IR
adapter at its intended boundary, and prevents a new workload or schedule
module from acquiring runtime or legacy-consumer authority by accident.

Two tiers, mirroring tools/check_typing.py:
  - module-level violations: the red line; any new one fails the gate.
  - function-local imports: a ledger; still circular at call time, so they are
    counted separately and must only decrease until R3 eliminates them.

Imports under ``if TYPE_CHECKING:`` are erased at runtime and exempt.

``runtime -> backends`` is exempt only for ``tllm.backends.protocol`` (the
sanctioned protocol seam); imports of concrete backend drivers fail.

Files directly under ``tllm/`` (the package facade) are exempt. Every new
subpackage of ``tllm`` must be declared in RULES — undeclared packages fail.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = REPO_ROOT / "tllm"
BASELINE_PATH = REPO_ROOT / "tools" / "import_layers_baseline.json"

# importer package -> allowed internal targets (besides itself).
RULES: dict[str, frozenset[str]] = {
    "core": frozenset(),
    "contracts": frozenset({"core"}),
    # Clean-room native core. It consumes only closed contracts and Rust FFI;
    # legacy runtime/workload/workgraph layers must never become dependencies.
    "canonical": frozenset({"contracts", "core"}),
    # Clean in-process vLLM attachment.  It may consume signed canonical
    # plans/contracts and Rust FFI, but never the legacy runtime/backends.
    "canonical_vllm": frozenset({"canonical", "contracts", "core"}),
    "ports": frozenset({"contracts", "core"}),
    "common": frozenset({"contracts", "ports", "core"}),
    "producers": frozenset({"contracts", "ports", "core"}),
    "workgraph": frozenset({"contracts", "ports", "producers", "core"}),
    "runtime": frozenset({"contracts", "ports", "common", "producers", "workgraph", "core"}),
    # Program compiler: lowers frontend declarations into static IR.
    "program": frozenset({"contracts", "ports", "common", "producers", "workgraph", "core"}),
    # The workload frontend and schedule middle-end communicate only through
    # the source-module-specific bridges below.  Neither package gets a broad
    # cross-package allowance: that would hide an import cycle or permit the
    # frontend to take runtime/legacy execution authority.
    "schedule": frozenset(),
    "workload": frozenset(),
    # Backend drivers are adapters *above* the runtime: they may import it,
    # but the runtime must never import a concrete driver (only the pure
    # protocol module, via SANCTIONED_EDGES).
    "backends": frozenset({"contracts", "ports", "workgraph", "runtime", "core"}),
    # Author-facing ergonomics kit: composes the public DSL facade, the
    # runtime plugin contracts, and common codecs.  Nothing below imports it,
    # so it can never become a hidden dependency of the compiler or runtime.
    "kit": frozenset({"contracts", "common", "ports", "runtime", "workload", "core"}),
    "consumers": frozenset(
        {
            "contracts", "ports", "common", "producers", "workgraph",
            "runtime", "program", "kit", "core",
        }
    ),
    "workflows": None,  # leaf entry points: may import anything
    # Script helpers used by workflows/starter/gpu scripts.
    "util": None,
}

# Any package may import the pure backend contract module (it has zero
# internal imports); the runtime may touch nothing else under backends/.
SANCTIONED_EDGES: frozenset[tuple[str, str]] = frozenset(
    {
        ("runtime", "tllm.backends.protocol"),
        ("producers", "tllm.backends.protocol"),
        ("workgraph", "tllm.backends.protocol"),
        # Consumer binders reach concrete backend drivers only through the
        # protocol seam (runner binding -> driver), never via driver modules.
        ("consumers", "tllm.backends.protocol"),
    }
)

# Source-module-specific compiler integration edges.  These are intentionally
# not package rules: e.g. allowing ``workload -> schedule`` would let the DSL
# frontend call the planner, while allowing ``schedule -> workload`` would let
# arbitrary planner code reach user executors.  Keep the coordinator as the
# sole composition root and schedule.contracts as the sole reverse dependency.
EXACT_MODULE_EDGES: frozenset[tuple[str, str]] = frozenset(
    {
        # Closed locator enums are pure wire vocabulary shared by the DSL and
        # the signed schedule contract. Keep this narrower than a package-wide
        # dependency so neither layer can acquire general contract authority.
        ("tllm.schedule.contracts", "tllm.contracts.hidden_locator"),
        ("tllm.schedule.contracts", "tllm.contracts.packed_hidden"),
        ("tllm.schedule.contracts", "tllm.contracts.request_identity"),
        ("tllm.workload.frontend", "tllm.contracts.hidden_locator"),
        ("tllm.workload.frontend", "tllm.contracts.request_identity"),
        ("tllm.workload.ir", "tllm.contracts.hidden_locator"),
        ("tllm.workload.ir", "tllm.contracts.request_identity"),
        ("tllm.workload.staging", "tllm.contracts.workload_ports"),
        # Workload frontend adapters.
        ("tllm.workload.compat", "tllm.program.decl"),
        ("tllm.workload.compat", "tllm.contracts.request_identity"),
        ("tllm.workload.compat", "tllm.workgraph.frontend"),
        ("tllm.workload", "tllm.contracts.request_identity"),
        ("tllm.workload.lowering", "tllm.schedule.contracts"),
        ("tllm.workload.region_compiler", "tllm.schedule.contracts"),
        # Compiler composition and mechanical physical lowering.  The frozen
        # physical-plan vocabulary lives in the native contracts layer; the
        # schedule middle-end never touches the legacy workgraph stack.
        ("tllm.schedule.coordinator", "tllm.contracts.physical_plan"),
        ("tllm.schedule.coordinator", "tllm.workload.frontend"),
        ("tllm.schedule.coordinator", "tllm.workload.lowering"),
        ("tllm.schedule.physical", "tllm.contracts.physical_plan"),
        ("tllm.schedule.timeline", "tllm.backends.protocol"),
        # Backends export the typed main-workload timeline, but never import
        # the coordinator or workload executors.
        ("tllm.backends.timeline", "tllm.schedule.timeline"),
        # Runtime consumes signed compiler output and local artifacts.  It is
        # intentionally a consumer of these packages, never their dependency.
        ("tllm.runtime.mp.control", "tllm.schedule"),
        ("tllm.runtime.mp.worker_extension", "tllm.schedule"),
        ("tllm.runtime.mp.worker_extension", "tllm.schedule.contracts"),
        ("tllm.runtime.mp.worker_extension", "tllm.schedule.timeline"),
        ("tllm.runtime.residual_runtime", "tllm.schedule"),
        ("tllm.runtime.state", "tllm.schedule"),
        ("tllm.runtime.workload_execution", "tllm.schedule"),
        ("tllm.runtime.workload_execution", "tllm.schedule.contracts"),
        ("tllm.runtime.workload_execution", "tllm.schedule.physical"),
        ("tllm.runtime.workload_execution", "tllm.workload.frontend"),
        ("tllm.runtime.workload_execution", "tllm.workload.lowering"),
        ("tllm.runtime.workload_execution", "tllm.workload.region_compiler"),
        # Authority enum check at native execution bind time (function-local).
        ("tllm.runtime.workload_execution", "tllm.workload"),
        ("tllm.runtime.workload_plugins", "tllm.schedule"),
        ("tllm.runtime.workload_plugins", "tllm.workload"),
        ("tllm.runtime.workload_plugins", "tllm.workload.lowering"),
        ("tllm.runtime.runtime", "tllm.schedule"),
        ("tllm.runtime.runtime", "tllm.workload"),
        ("tllm.runtime.declaration_adapter", "tllm.workload"),
        # Leaf workload implementations may use only the public DSL facade.
        # Public migration routing boundary; the deleted Dummy implementation
        # cannot be imported back into the clean-room core.
        ("tllm.consumers.dummy", "tllm.canonical.workloads.dummy"),
        ("tllm.consumers.esamp.workload", "tllm.workload"),
        ("tllm.consumers.override.workload", "tllm.workload"),
        ("tllm.consumers.recirculation.native_backend", "tllm.workload"),
        ("tllm.consumers.recirculation.workload", "tllm.workload"),
        ("tllm.consumers.reft.workload", "tllm.workload"),
        ("tllm.consumers.ttrl.workload", "tllm.workload"),
        ("tllm.consumers.tlm.workload", "tllm.workload"),
    }
)

# Compiled Rust artifact and other top-level extension modules are not packages.
NON_PACKAGE_MODULES: frozenset[str] = frozenset({"_core"})


@dataclass(frozen=True)
class Violation:
    importer: str
    module: str
    target_pkg: str
    level: str  # "module" | "local"
    file: str
    line: int

    @property
    def bucket(self) -> str:
        return f"{self.importer}->{self.target_pkg}"


def _module_parts(path: Path) -> list[str]:
    rel = path.relative_to(PKG_ROOT).with_suffix("")
    parts = ["tllm", *rel.parts]
    if parts[-1] == "__init__":
        parts.pop()
    return parts


def _resolve_from(current: list[str], level: int, module: str | None) -> list[str] | None:
    """Resolve a relative `from ... import` to absolute module parts."""
    if level == 0:
        return module.split(".") if module else None
    # level 1 = current package; the file itself is one part deep.
    base = current[: len(current) - level]
    if not base or base[0] != "tllm":
        return None
    return base + (module.split(".") if module else [])


def _is_type_checking_block(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _is_allowed_target(
    importer: str,
    source_module: str,
    target_pkg: str,
    target_module: str,
) -> bool:
    """Return whether one resolved cross-package import follows the layer rules."""
    allowed = RULES.get(importer, "undeclared")
    if allowed is None:
        return True
    if allowed != "undeclared" and target_pkg in allowed:
        return True
    if (source_module, target_module) in EXACT_MODULE_EDGES:
        return True
    return (importer, target_module) in SANCTIONED_EDGES


def _scan_file(path: Path) -> list[Violation]:
    rel_parts = path.relative_to(PKG_ROOT).parts
    if len(rel_parts) == 1:
        return []  # package facade (tllm/__init__.py etc.) is exempt
    importer = rel_parts[0]
    allowed = RULES.get(importer, "undeclared")
    if allowed is None:
        return []  # leaf packages may import anything
    current = _module_parts(path)
    source_module = ".".join(current)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    violations: list[Violation] = []

    def visit(node: ast.AST, in_function: bool, type_only: bool = False) -> None:
        for child in ast.iter_child_nodes(node):
            if type_only and isinstance(child, (ast.Import, ast.ImportFrom)):
                continue  # erased at runtime
            targets: list[list[str] | None] = []
            if isinstance(child, ast.Import):
                targets = [alias.name.split(".") for alias in child.names]
            elif isinstance(child, ast.ImportFrom):
                targets = [_resolve_from(current, child.level, child.module)]
            else:
                visit(
                    child,
                    in_function or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)),
                    type_only or _is_type_checking_block(child),
                )
                continue
            for parts in targets:
                if parts is None or len(parts) < 2 or parts[0] != "tllm":
                    continue
                target_pkg = parts[1]
                if target_pkg == importer or target_pkg in NON_PACKAGE_MODULES:
                    continue
                if allowed == "undeclared":
                    violations.append(
                        Violation(importer, ".".join(parts), target_pkg, "undeclared", str(path), child.lineno)
                    )
                    continue
                module_name = ".".join(parts)
                if _is_allowed_target(importer, source_module, target_pkg, module_name):
                    continue
                violations.append(
                    Violation(
                        importer,
                        module_name,
                        target_pkg,
                        "local" if in_function else "module",
                        str(path.relative_to(REPO_ROOT)),
                        child.lineno,
                    )
                )

    visit(tree, False)
    return violations


def _collect() -> list[Violation]:
    violations: list[Violation] = []
    for path in sorted(PKG_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        violations.extend(_scan_file(path))
    return violations


def _buckets(violations: list[Violation], level: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in violations:
        if v.level == level:
            counts[v.bucket] = counts.get(v.bucket, 0) + 1
    return dict(sorted(counts.items()))


def _rules_snapshot() -> dict[str, list[str] | str]:
    snapshot: dict[str, list[str] | str] = {
        pkg: ("leaf (any)" if allowed is None else sorted(allowed))
        for pkg, allowed in sorted(RULES.items())
    }
    snapshot["exact_module_edges"] = [
        f"{source} -> {target}"
        for source, target in sorted(EXACT_MODULE_EDGES)
    ]
    snapshot["sanctioned_edges"] = [
        f"{source} -> {target}"
        for source, target in sorted(SANCTIONED_EDGES)
    ]
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebase", action="store_true", help="record current violations as the new baseline")
    parser.add_argument("--list", action="store_true", help="print every violating import")
    args = parser.parse_args()

    violations = _collect()
    undeclared = [v for v in violations if v.level == "undeclared"]
    if undeclared:
        seen = sorted({v.importer for v in undeclared})
        print(f"error: undeclared tllm subpackage(s) {seen}; add them to RULES in {Path(__file__).name}", file=sys.stderr)
        return 2

    module_buckets = _buckets(violations, "module")
    local_buckets = _buckets(violations, "local")

    if args.list:
        for v in violations:
            print(f"{v.level:6s} {v.file}:{v.line}  {v.bucket}  ({v.module})")
        print(f"module-level: {sum(module_buckets.values())}, local: {sum(local_buckets.values())}")
        return 0

    if args.rebase:
        if BASELINE_PATH.exists():
            previous = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
            for current, old in (
                (module_buckets, previous["module_level"]),
                (local_buckets, previous["local_level"]),
            ):
                if any(
                    count > old.get(bucket, 0)
                    for bucket, count in current.items()
                ):
                    print(
                        "error: --rebase cannot increase import violations",
                        file=sys.stderr,
                    )
                    return 1
        baseline = {
            "comment": "Cross-package import baseline (only-decrease). Regenerate with `python tools/check_import_layers.py --rebase`.",
            "rules": _rules_snapshot(),
            "module_level": module_buckets,
            "local_level": local_buckets,
        }
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        print(
            f"rebased {BASELINE_PATH.relative_to(REPO_ROOT)}: "
            f"module-level {sum(module_buckets.values())}, local {sum(local_buckets.values())}"
        )
        return 0

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    live_rules = _rules_snapshot()
    if baseline.get("rules") != live_rules:
        print(
            "FAIL import-layer rule snapshot differs from the reviewed baseline"
        )
        print(
            "audit the rule change, then run "
            "`python tools/check_import_layers.py --rebase`"
        )
        return 1
    failed = False
    for label, current, base in (
        ("module-level", module_buckets, baseline["module_level"]),
        ("local", local_buckets, baseline["local_level"]),
    ):
        over = [
            f"  {bucket}: {count} > baseline {base.get(bucket, 0)}"
            for bucket, count in sorted(current.items())
            if count > base.get(bucket, 0)
        ]
        if over:
            print(f"FAIL {label} imports ({sum(current.values())} total):")
            print("\n".join(over))
            print("run `python tools/check_import_layers.py --list` to see the offending imports")
            failed = True
        else:
            decreased = sum(base.values()) - sum(current.values())
            print(f"OK   {label} imports: {sum(current.values())} (baseline {sum(base.values())}, decreased by {max(decreased, 0)})")
    if failed:
        return 1
    if sum(module_buckets.values()) + sum(local_buckets.values()) < sum(baseline["module_level"].values()) + sum(
        baseline["local_level"].values()
    ):
        print("hint: baseline improved — consider `python tools/check_import_layers.py --rebase` to lock in the progress")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
