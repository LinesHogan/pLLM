"""One strict typing gate for the complete maintained Python implementation."""

import ast
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Adapters to dynamically typed upstream APIs bypass the source gate. The author's own Python objects
# (workload classes, settings, arbitrary checked values) are such an upstream for the authoring modules.
UNTYPED_BOUNDARIES = {
    ROOT / "pllm/execution/torch_private.py",
    ROOT / "pllm/execution/module_graph.py",
    ROOT / "pllm/contracts/codec.py",
    ROOT / "pllm/usage.py",
    ROOT / "pllm/workload.py",
    ROOT / "pllm/execution/method_dependencies.py",
    ROOT / "pllm/workload_transport.py",
    ROOT / "pllm/cli.py",
}
KERNELS = ROOT / "pllm/kernels"
# Pyright follows the compat module's release constants per release, so each major seam gets a config.
_VLLM_VERSION = version("vllm")
PROJECT = (
    "pyrightconfig.json"
    if _VLLM_VERSION.startswith("0.27.")
    else "pyrightconfig.vllm034.json"
    if _VLLM_VERSION.startswith(("0.33.", "0.34."))
    else "pyrightconfig.vllm032.json"
)
FORBIDDEN = {"Any", "object", "cast", "getattr", "setattr", "hasattr"}


def idiomatic(node: ast.Name, parents: dict[ast.AST, ast.AST]) -> bool:
    """`object.__setattr__` in frozen dataclasses, `__eq__(other: object)`, and erased generic arguments."""
    parent = parents.get(node)
    if node.id == "object" and isinstance(parent, ast.Attribute):
        return True
    if node.id == "object" and isinstance(parent, ast.arg):
        function = parents.get(parents.get(parent, parent), parent)
        return isinstance(function, ast.FunctionDef) and function.name in ("__eq__", "__ne__")
    if node.id == "Any" and isinstance(parent, ast.Tuple):
        parent = parents.get(parent)
    return node.id == "Any" and isinstance(parent, ast.Subscript)


def main() -> int:
    violations: list[str] = []
    for path in sorted((ROOT / "pllm").rglob("*.py")):
        if path in UNTYPED_BOUNDARIES or path.is_relative_to(KERNELS):
            continue
        source = path.read_text()
        tree = ast.parse(source)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in FORBIDDEN and not idiomatic(node, parents):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.id} outside untyped API boundaries")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "importlib"
                    and node.func.attr == "import_module"
                )
            ):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: dynamic type-path import")
        for index, line in enumerate(source.splitlines(), 1):
            # A scoped `pyright: ignore[rule]` names the one upstream gap it accepts; blanket escapes are rejected.
            if "# type: ignore" in line or re.search(r"pyright: ignore(?!\[)", line):
                violations.append(f"{path.relative_to(ROOT)}:{index}: type-ignore escape")
    if violations:
        print("\n".join(violations))
        return 1
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pyright",
            "--pythonpath",
            sys.executable,
            "--warnings",
            "--project",
            PROJECT,
        ],
        cwd=ROOT,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
