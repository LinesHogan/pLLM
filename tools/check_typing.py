"""One strict typing gate for the complete maintained Python implementation."""

import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CODEC = ROOT / "pllm/execution/codec.py"
CONTRACT_CODEC = ROOT / "pllm/contracts/codec.py"
GRAPH_ADAPTER = ROOT / "pllm/execution/module_graph.py"
COMPILER_ADAPTER = ROOT / "pllm/backends/torch_inductor.py"
COLD_AUTHORING = {ROOT / "pllm/usage.py", ROOT / "pllm/execution/freeze.py", ROOT / "pllm/learning.py",
    # The by-reference transport boundary: class coordinates resolve through getattr/importlib.
    ROOT / "pllm/workload_transport.py", ROOT / "pllm/contracts/workload_reference.py",
    # Worker RPC erases the reconstructed workload's generic payload types.
    ROOT / "pllm/worker.py"}
DYNAMIC_IMPORT = {CODEC, ROOT / "pllm/workload_transport.py"}
FORBIDDEN = {"Any", "object", "cast", "getattr", "setattr", "hasattr"}


def main() -> int:
    violations: list[str] = []
    for path in sorted((ROOT / "pllm").rglob("*.py")):
        # FX graph arguments and Module.forward are dynamically typed upstream.
        # Keep those checks at the graph import boundary, outside scheduling.
        # The shared contract codec is the sole dynamic schema adapter; source
        # contracts and semantic rules themselves retain the ordinary strict gate.
        # Inductor lowering artifacts also expose dynamic upstream types;
        # only this pinned backend inspects them, through a typed result ABI.
        if path in (CODEC, CONTRACT_CODEC, GRAPH_ADAPTER, COMPILER_ADAPTER):
            continue
        source = path.read_text()
        for node in ast.walk(ast.parse(source)):
            if path not in COLD_AUTHORING and isinstance(node, ast.Name) and node.id in FORBIDDEN:
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.id} outside codec/cold authoring")
            if path not in DYNAMIC_IMPORT and isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and (
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
