"""Check that the tracked documentation still matches the code; run by full_verify and by doc authors.

Three checks over README.md, doc/ and examples/:
1. every `pllm.X` name and `from pllm... import X` in Python code resolves against the importable package;
2. every relative markdown link points at an existing file;
3. notebook code cells tagged "cpu-check" (cell metadata `tags`) run in order on CPU, one process per notebook.

Notebooks without any "cpu-check" cell are pending a rewrite and are skipped unless `--all` is given;
the report lists them. Exit status is 1 when any check fails.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[1]
CPU_TAG = "cpu-check"
FENCED_PYTHON = re.compile(r"^```(?:python|py)\n(.*?)^```", re.S | re.M)
FENCED_ANY = re.compile(r"^```.*?^```", re.S | re.M)
INLINE_CODE = re.compile(r"`[^`\n]*`")
LINK = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
REFERENCE_LINK = re.compile(r"^\[[^\]]+\]:\s*(\S+)", re.M)
EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|#)", re.I)


class CellMetadata(TypedDict, total=False):
    tags: list[str]


class Cell(TypedDict):
    """The fields of an nbformat 4 cell this tool reads."""

    cell_type: str
    source: str | list[str]
    metadata: CellMetadata


@dataclass(frozen=True)
class Block:
    """One piece of Python code from a document, with where it starts for the report."""

    where: str
    source: str


@dataclass(frozen=True)
class Notebook:
    """The cells of one .ipynb file that the checks read."""

    path: Path
    code: list[Block]
    markdown: list[str]
    cpu_cells: list[Block]


def read_notebook(path: Path) -> Notebook:
    cells: list[Cell] = json.loads(path.read_text())["cells"]
    code: list[Block] = []
    markdown: list[str] = []
    cpu: list[Block] = []
    for index, cell in enumerate(cells):
        lines = cell["source"]  # nbformat allows one string or a list of lines
        source = lines if isinstance(lines, str) else "".join(lines)
        if cell["cell_type"] == "markdown":
            markdown.append(source)
            code.extend(markdown_blocks(source, f"cell {index} "))
        elif cell["cell_type"] == "code":
            code.append(Block(f"cell {index}", source))
            if CPU_TAG in cell["metadata"].get("tags", []):
                cpu.append(Block(f"cell {index}", source))
    return Notebook(path, code, markdown, cpu)


def markdown_blocks(text: str, prefix: str = "") -> list[Block]:
    """The fenced Python blocks of a markdown text, each named by the line of its opening fence."""
    blocks: list[Block] = []
    for match in FENCED_PYTHON.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        blocks.append(Block(f"{prefix}line {line}", match.group(1)))
    return blocks


def resolve(dotted: str) -> bool:
    """Whether `dotted` names a module, or an attribute reachable from one (submodules imported on demand)."""
    parts = dotted.split(".")
    value: object = importlib.import_module(parts[0])
    for depth, part in enumerate(parts[1:], start=2):
        if hasattr(value, part):
            value = getattr(value, part)
        elif isinstance(value, ModuleType):
            try:
                value = importlib.import_module(".".join(parts[:depth]))
            except ModuleNotFoundError:
                return False
        elif isinstance(value, type) and any(part in vars(klass).get("__annotations__", {}) for klass in value.__mro__):
            return True  # a dataclass field without a class-level default
        else:
            return False
    return True


def attribute_chain(node: ast.Attribute) -> tuple[str, str] | None:
    """The root name and dotted tail of `a.b.c`, or None when the chain starts at an expression."""
    names: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        names.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    return current.id, ".".join(reversed(names))


def pllm_names(tree: ast.Module) -> dict[str, str]:
    """Local names this code binds to pllm objects through imports, mapped to their dotted pllm path."""
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "pllm":
                    names[alias.asname or "pllm"] = alias.name if alias.asname else "pllm"
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "pllm":
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return names


def missing_symbols(blocks: list[Block]) -> list[str]:
    """Unresolvable pllm references in `blocks`; imports in earlier blocks stay visible to later ones."""
    trees: list[tuple[Block, ast.Module]] = []
    failures: list[str] = []
    for block in blocks:
        try:
            trees.append((block, ast.parse(block.source)))
        except SyntaxError as error:
            failures.append(f"{block.where}: does not parse ({error.msg}, line {error.lineno})")
    names: dict[str, str] = {}
    for _, tree in trees:
        names.update(pllm_names(tree))
    for block, tree in trees:
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        wanted = set(pllm_names(tree).values())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and not isinstance(parents.get(node), ast.Attribute):
                chain = attribute_chain(node)
                if chain is not None and chain[0] in names:
                    wanted.add(f"{names[chain[0]]}.{chain[1]}")
        missing: list[str] = []
        for name in sorted(wanted):  # sorted puts `a.B` before `a.B.C`; report only the first missing link
            if not any(name.startswith(f"{shorter}.") for shorter in missing) and not resolve(name):
                missing.append(name)
        failures.extend(f"{block.where}: {name} does not exist" for name in missing)
    return failures


def broken_links(path: Path, texts: list[str]) -> list[str]:
    failures: list[str] = []
    for text in texts:
        prose = INLINE_CODE.sub("", FENCED_ANY.sub("", text))
        for target in LINK.findall(prose) + REFERENCE_LINK.findall(prose):
            if EXTERNAL.match(target):
                continue
            file = target.split("#", 1)[0]
            if not (path.parent / file).exists():
                failures.append(f"link {target} does not exist")
    return failures


# Runs each cell in one shared namespace under its own file name, so a traceback names the failing cell.
CELL_RUNNER = """import json, sys
namespace = {"__name__": "__main__"}
for where, source in json.load(sys.stdin):
    exec(compile(source, f"<{where}>", "exec"), namespace)
"""
CELL_FRAME = re.compile(r'File "<(cell \d+)>", line (\d+)')


def run_cpu_cells(notebook: Notebook) -> list[str]:
    cells = json.dumps([(block.where, block.source) for block in notebook.cpu_cells])
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT)}
    completed = subprocess.run(
        [sys.executable, "-c", CELL_RUNNER],
        input=cells,
        cwd=notebook.path.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode == 0:
        return []
    error = (completed.stderr.strip().splitlines() or ["no output"])[-1]
    frames = CELL_FRAME.findall(completed.stderr)
    where = f"{frames[-1][0]} line {frames[-1][1]}" if frames else "setup"
    return [f"{where}: {CPU_TAG} cell fails on CPU: {error}"]


def check_file(path: Path, include_untagged: bool) -> tuple[list[str], bool]:
    """Failures in one document, and whether it was checked at all."""
    if path.suffix == ".py":
        return missing_symbols([Block("file", path.read_text())]), True
    if path.suffix == ".md":
        text = path.read_text()
        return missing_symbols(markdown_blocks(text)) + broken_links(path, [text]), True
    notebook = read_notebook(path)
    if not notebook.cpu_cells and not include_untagged:
        return [], False
    failures = missing_symbols(notebook.code) + broken_links(path, notebook.markdown)
    if notebook.cpu_cells:
        failures += run_cpu_cells(notebook)
    return failures, True


def documents() -> list[Path]:
    paths = [ROOT / "README.md", *(ROOT / "doc").rglob("*.md"), *(ROOT / "doc").rglob("*.ipynb")]
    paths += (ROOT / "examples").rglob("*.py")
    return sorted(path for path in paths if ".ipynb_checkpoints" not in path.parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="also check notebooks that have no cpu-check cell")
    include_untagged = parser.parse_args().all
    sys.path.insert(0, str(ROOT))
    failed = 0
    skipped: list[str] = []
    paths = documents()
    for path in paths:
        failures, checked = check_file(path, include_untagged)
        name = str(path.relative_to(ROOT))
        if not checked:
            skipped.append(name)
        for failure in failures:
            print(f"{name}: {failure}")
        failed += bool(failures)
    if skipped:
        print(f"skipped {len(skipped)} notebooks without {CPU_TAG} cells (pending rewrite): {', '.join(skipped)}")
    print(f"check_docs: {len(paths) - len(skipped)} documents checked, {failed} with failures")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
