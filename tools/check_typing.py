"""One strict typing gate for the complete maintained Python implementation."""

import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CODEC = ROOT / "pllm/execution/codec.py"
COLD_AUTHORING = ROOT / "pllm/usage.py"
FORBIDDEN = {"Any", "object", "cast", "getattr", "setattr", "hasattr"}


def main() -> int:
    violations: list[str] = []
    for path in sorted((ROOT / "pllm").rglob("*.py")):
        if path == CODEC:
            continue
        source = path.read_text()
        for node in ast.walk(ast.parse(source)):
            if path != COLD_AUTHORING and isinstance(node, ast.Name) and node.id in FORBIDDEN:
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.id} outside codec/cold authoring")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and (
                isinstance(node.func.value, ast.Name) and node.func.value.id == "importlib"
                and node.func.attr == "import_module"):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: dynamic type-path import")
        for index, line in enumerate(source.splitlines(), 1):
            if "# type: ignore" in line or "pyright: ignore" in line:
                violations.append(f"{path.relative_to(ROOT)}:{index}: type-ignore escape")
    if violations:
        print("\n".join(violations))
        return 1
    return subprocess.run([sys.executable, "-m", "pyright", "--warnings", "--project", "pyrightconfig.json"], cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
