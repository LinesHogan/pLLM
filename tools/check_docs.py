"""Check that the tracked documentation still matches the code; run by full_verify and by doc authors.

Checks over README.md, doc/ and examples/:
1. every `pllm.X` name and `from pllm... import X` in Python code resolves against the importable package;
2. every relative markdown link points at an existing file and heading (including notebook markdown);
3. notebook code cells tagged "cpu-check" (cell metadata `tags`) run in order on CPU, one process per notebook,
   and print their stored output, compared as in 4 (`--write-outputs` stores what they print instead);
   untagged cells (loading a model, generating, CUDA) are only symbol-checked and counted per notebook in the summary;
4. the Python blocks of README.md and doc/{agent,guides,howto,reference,explain}/*.md run in order, one process per
   file: on CPU, or with `--gpu` also the agent blocks fenced ```python gpu, on the GPU. Intentionally partial
   blocks must contain `# check-docs: skip: <reason>`; each opt-out is reported in the summary. A ```text block
   right after an executed block shows its output and must match what it prints, where a line `...` stands for
   any number of lines, `...` within a line for any text, and memory addresses and times in s or ms always match;
   a text block that is not that output is fenced ```text not-output;
5. doc/agent/reference.md names every public pllm name in its task sections, API_GROUPS files every public name
   under one task, and every generated block equals what this tool renders from its source (`--write-generated`
   rewrites them): from the code, the signature list of reference.md, the API page doc/reference/api.md (each
   name's full docstring) and the field and default columns of doc/reference/config.md; from
   doc/reference/errors.toml, the error tables of doc/reference/errors.md (Chinese) and doc/agent/pitfalls.md
   (English);
6. error messages quoted in backticks in the first table column of a section about rejections or errors still
   exist in pllm's Python or Rust sources: every piece between `...` markers (which stand for text that
   depends on the user's code) must appear in one string literal;
7. every throughput (tok/s), time in ms, and percentage on a line that compares with native vLLM, stated in the
   prose of README.md and doc/ (outside doc/dev/), appears in a result of doc/reference/measurements.toml, so each
   measured number has one source; a line whose numbers are not measurements ends with
   `<!-- check-docs: not measured -->`.

8. README.md and doc/ outside doc/dev/ are written for users: a word from JARGON appears there only if the
   glossary (doc/reference/glossary.md) explains it in a bold term. Contributor vocabulary (PTTS, receipts,
   evidence numbers, milestones, local paths, commit hashes) has no glossary entry, so it always fails.

The essays in doc/explain/highlights/ are dated articles: each names the commit and setup it was written against
and is not kept current, so their code does not run and their vocabulary and numbers are not checked.

Notebooks without any "cpu-check" cell are pending a rewrite and are skipped unless `--all` is given;
the report lists them. Exit status is 1 when any check fails.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import cache, partial
from pathlib import Path
from types import ModuleType
from typing import Any, NotRequired, TypedDict
from urllib.parse import unquote, urlsplit

from github_slugger import GithubSlugger
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
CPU_TAG = "cpu-check"
SKIP = re.compile(r"^\s*# check-docs: skip: (\S[^\n]*)$", re.M)
MARKDOWN = MarkdownIt(options_update={"inline_definitions": True})
AGENT_DOCS = ROOT / "doc" / "agent"
EXECUTED_DOCS = (
    AGENT_DOCS,
    *(ROOT / "doc" / name for name in ("guides", "howto", "reference", "explain")),
)
REFERENCE = AGENT_DOCS / "reference.md"
PUBLIC_NAMESPACES = ("pllm", "pllm.model", "pllm.host", "pllm.io", "pllm.testing", "pllm.benchmark")
ERRORS = ROOT / "doc" / "reference" / "errors.toml"
API = ROOT / "doc" / "reference" / "api.md"
CONFIG = ROOT / "doc" / "reference" / "config.md"
CONFIG_CLASSES = (
    "pllm.ModelConfig",
    "pllm.ExecutionOptions",
    "pllm.KVOffload",
    "pllm.host.HostConfig",
    "pllm.benchmark.BenchmarkOptions",
)
# The API page's sections, by what a user wants to do; every public name is filed under exactly one.
API_GROUPS = (
    ("引擎与负载", ("make_llm", "explain", "Workload", "ModelConfig", "ExecutionOptions", "KVOffload")),
    (
        "读什么：输入标记",
        (
            "hidden",
            "residual",
            "final_hidden",
            "layer_logits",
            "attention",
            "input_tokens",
            "sample_index",
            "base_logits",
            "sampled_tokens",
            "result",
            "parameter",
            "module",
            "version",
            "index",
            "carry",
            "depth",
            "value",
            "operand",
            "batch",
            "causal_model",
        ),
    ),
    (
        "存什么：状态、初值与优化器",
        (
            "RequestState",
            "InstanceState",
            "PositionState",
            "Field",
            "Dimensions",
            "Zeros",
            "Computed",
            "Pretrained",
            "ModelCheckpoint",
            "FromModule",
            "Fed",
            "Imported",
            "AdamW",
            "SGD",
        ),
    ),
    (
        "结果去哪：效果与时机",
        ("export", "add_logits", "guide_sampling", "write", "train", "differentiate", "UpdateOutput", "on", "Phase"),
    ),
    (
        "改模型与 forward",
        (
            "model.ActivationPatch",
            "model.Adapter",
            "model.Projection",
            "train_in_forward",
            "custom_residual",
            "custom_attention",
            "model.PagedKV",
            "custom_forward",
            "model.Layers",
            "model.LayerOutputs",
            "scan",
        ),
    ),
    (
        "宿主与负载之间的数据：导出、导入与投喂",
        (
            "io.TraceEvent",
            "io.TraceSelection",
            "io.TraceSnapshot",
            "io.StateImport",
            "io.Feed",
            "io.checkpoint_directory",
            "io.checkpoint_tensors",
            "io.load_tensor_fields",
        ),
    ),
    (
        "宿主循环与批训练",
        (
            "host.make_host",
            "host.HostConfig",
            "host.HostTrainer",
            "host.TrainStep",
            "host.WeightedRowBatch",
            "host.CausalTrainBatch",
            "host.WeightedCEBatch",
            "model.CausalModel",
            "host.CompletionInput",
            "host.Completion",
            "host.CompletionBatch",
            "host.RewardOutcome",
            "host.RetryControl",
            "host.RetryAction",
            "host.UpdateBarrier",
            "host.BarrierState",
            "host.ControlReason",
            "host.FinishReason",
        ),
    ),
    (
        "检查正确性",
        (
            "testing.check",
            "testing.check_on_gpu",
            "testing.check_distribution",
            "testing.CheckResult",
            "testing.ContractError",
            "testing.DistributionResult",
            "testing.DistanceSummary",
            "testing.TokenDistance",
        ),
    ),
    (
        "在测试里手动单步",
        (
            "testing.compile_execution",
            "testing.backend_facts",
            "testing.scheduled_step",
            "testing.step_input",
            "testing.run_step",
            "testing.run_guided_step",
            "testing.hidden_binding",
            "testing.final_hidden_binding",
            "testing.RequestIdentity",
            "testing.SamplerFeedback",
        ),
    ),
    (
        "测吞吐",
        ("benchmark.throughput", "benchmark.compare", "benchmark.BenchmarkOptions", "benchmark.BenchmarkResult"),
    ),
)
ERROR_SECTION = re.compile(r"rejection|error|pitfall|报错|错误", re.I)
# Words a reader of the user documents may not know: each needs a glossary entry. The contributor vocabulary
# after them (PTTS, receipts, evidence numbers, milestones, local paths, commit hashes) belongs in doc/dev/.
JARGON = re.compile(
    r"准入|规划器|旁路|锚点|Inductor|StepCapture"
    r"|PTTS|receipt|签署|证据 \d+|(?<![A-Za-z0-9_])E\d(?![A-Za-z0-9_])|已移除|/home/"
    r"|(?<![A-Za-z0-9_])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,12}(?![A-Za-z0-9_])"
)
GLOSSARY = ROOT / "doc" / "reference" / "glossary.md"
DEV_DOCS = ROOT / "doc" / "dev"
ESSAYS = ROOT / "doc" / "explain" / "highlights"
MEASUREMENTS = ROOT / "doc" / "reference" / "measurements.toml"
FIGURE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:\\text\{\s*)?(\\?%|tok/s|tokens?/s|ms)(?![\w/])")
NATIVE = re.compile(r"原生|native|vanilla", re.I)
NOT_MEASURED = "<!-- check-docs: not measured -->"
# Printed text that changes from run to run, compared as `…`: memory addresses and wall-clock times.
VOLATILE = re.compile(r"0x[0-9a-f]{4,}|\b\d+(?:\.\d+)? ?(?:ms|s)\b")


class Run(TypedDict):
    """The part of a measurements.toml run this tool reads: result name to stated figures."""

    results: dict[str, str]


class ErrorRow(TypedDict):
    """One message of errors.toml, with why it is raised and how to fix it in English and in Chinese."""

    message: str
    why: str
    fix: str
    why_zh: str
    fix_zh: str


class ErrorSection(TypedDict):
    """A group of related messages of errors.toml, titled in both languages."""

    en: str
    zh: str
    note: NotRequired[str]
    note_zh: NotRequired[str]
    error: list[ErrorRow]


class CellMetadata(TypedDict, total=False):
    tags: list[str]


class Output(TypedDict, total=False):
    output_type: str
    name: str
    text: str | list[str]


class Cell(TypedDict):
    """The fields of an nbformat 4 cell this tool reads."""

    cell_type: str
    source: str | list[str]
    metadata: CellMetadata
    outputs: NotRequired[list[Output]]


@dataclass(frozen=True)
class Block:
    """One piece of Python code from a document, with where it starts for the report."""

    where: str
    source: str
    gpu: bool = False
    output: str | None = None  # what the document shows it printing, if it shows anything

    @property
    def skip_reason(self) -> str:
        """The explicit reason this partial example opts out of execution, or an empty string."""
        match = SKIP.search(self.source)
        return match.group(1) if match else ""


@dataclass(frozen=True)
class Notebook:
    """The cells of one .ipynb file that the checks read."""

    path: Path
    code: list[Block]
    markdown: list[str]
    cpu_cells: list[Block]
    untagged_cells: list[Block]


def joined(text: str | list[str]) -> str:
    """nbformat stores text as one string or as a list of lines."""
    return text if isinstance(text, str) else "".join(text)


def read_notebook(path: Path) -> Notebook:
    cells: list[Cell] = json.loads(path.read_text())["cells"]
    code: list[Block] = []
    markdown: list[str] = []
    cpu: list[Block] = []
    untagged: list[Block] = []
    for index, cell in enumerate(cells):
        source = joined(cell["source"])
        if cell["cell_type"] == "markdown":
            markdown.append(source)
            code.extend(markdown_blocks(source, f"cell {index} "))
        elif cell["cell_type"] == "code":
            code.append(Block(f"cell {index}", source))
            if CPU_TAG in cell["metadata"].get("tags", []):
                outputs = cell.get("outputs", [])
                stdout = "".join(joined(output.get("text", "")) for output in outputs if output.get("name") == "stdout")
                cpu.append(Block(f"cell {index}", source, output=stdout))
            else:
                untagged.append(Block(f"cell {index}", source))
    return Notebook(path, code, markdown, cpu, untagged)


def markdown_blocks(text: str, prefix: str = "") -> list[Block]:
    """The fenced Python blocks of a markdown text, each named by the line of its opening fence and carrying the
    ```text block right after it as its output."""
    blocks: list[Block] = []
    tokens = MARKDOWN.parse(text)
    for token, after in zip(tokens, [*tokens[1:], None]):
        info = token.info.split()
        if token.type == "fence" and info and info[0] in ("python", "py"):
            assert token.map is not None
            output = (
                after.content if after is not None and after.type == "fence" and after.info.strip() == "text" else None
            )
            blocks.append(Block(f"{prefix}line {token.map[0] + 1}", token.content, "gpu" in info[1:], output))
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
    """Local names this code binds to pLLM objects through imports, mapped to their dotted pLLM path."""
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
    """Unresolvable pLLM references in `blocks`; imports in earlier blocks stay visible to later ones."""
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


def heading_anchors(texts: list[str]) -> set[str]:
    """GitHub heading IDs, with duplicate suffixes shared across a notebook's markdown cells."""
    anchors: set[str] = set()
    slugger = GithubSlugger()
    tokens = MARKDOWN.parse("\n\n".join(texts))
    for opening, inline in zip(tokens, tokens[1:]):
        if opening.type != "heading_open":
            continue
        title = "".join(
            child.content for child in inline.children or [] if child.type in ("text", "code_inline", "image")
        )
        anchors.add(slugger.slug(title))
    return anchors


def document_markdown(path: Path) -> list[str]:
    """The Markdown rendered by a plain document or notebook."""
    return read_notebook(path).markdown if path.suffix == ".ipynb" else [path.read_text()]


def markdown_targets(texts: list[str]) -> set[str]:
    """Link and image destinations, including unused reference definitions, outside code blocks."""
    targets: set[str] = set()
    for token in MARKDOWN.parse("\n\n".join(texts)):
        if token.type == "definition":
            targets.add(str(token.meta["url"]))
        for child in token.children or []:
            if child.type in ("link_open", "image"):
                target = child.attrGet("href" if child.type == "link_open" else "src")
                assert isinstance(target, str)
                targets.add(target)
    return targets


def broken_links(path: Path, texts: list[str], anchors: dict[Path, set[str]]) -> list[str]:
    """Missing relative link targets and heading fragments; external URLs are left alone."""
    failures: list[str] = []
    for target in sorted(markdown_targets(texts)):
        link = urlsplit(target)
        if link.scheme or link.netloc:
            continue
        file = (path.parent / unquote(link.path)).resolve() if link.path else path.resolve()
        if not file.exists():
            failures.append(f"link {target} does not exist")
        elif link.fragment and file.suffix in (".md", ".ipynb"):
            if file not in anchors:
                anchors[file] = heading_anchors(document_markdown(file))
            if unquote(link.fragment) not in anchors[file]:
                failures.append(f"link {target} has no heading anchor {unquote(link.fragment)}")
    return failures


@cache
def source_strings() -> str:
    """Every string literal of pLLM's Python and Rust sources, joined by NUL; f-string values also become NUL, so a
    quoted piece of a message never spans text that depends on the user's code."""
    texts: list[str] = []
    for path in (ROOT / "pllm").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                texts.append(node.value)
            elif isinstance(node, ast.JoinedStr):
                texts.append(
                    "\0".join(str(part.value) if isinstance(part, ast.Constant) else "" for part in node.values)
                )
    for path in (ROOT / "rust" / "pllm_core" / "src").rglob("*.rs"):
        # A backslash at a line end continues a Rust string literal without the next line's indentation.
        texts.append(re.sub(r"\\\n\s*", "", path.read_text()))
    return "\0".join(texts)


def quoted_messages(text: str) -> list[tuple[int, str]]:
    """(line, message) for backticked text in the first table column of a section about rejections or errors."""
    messages: list[tuple[int, str]] = []
    level = 0  # heading level of the enclosing error section, 0 outside one
    fenced = False
    for number, line in enumerate(text.splitlines(), start=1):
        fenced ^= line.startswith("```")
        heading = re.match(r"(#+) ", line)
        if heading and not fenced:
            depth = len(heading.group(1))
            if ERROR_SECTION.search(line):
                level = depth
            elif depth <= level:
                level = 0
        elif level and not fenced and line.startswith("|") and not re.match(r"\|\s*-", line):
            first_cell = re.split(r"(?<!\\)\|", line)[1]
            messages.extend((number, span) for span in re.findall(r"`([^`]+)`", first_cell))
    return messages


def stale_messages(text: str) -> list[str]:
    """Quoted error messages with a piece that no pLLM source string contains."""
    corpus = source_strings()
    return [
        f"line {number}: quoted message `{message}` is not in pLLM's sources; copy the current text and write ... "
        "for the parts that depend on the user's code"
        for number, message in quoted_messages(text)
        if any(piece.strip() not in corpus for piece in message.split("...") if piece.strip())
    ]


def figures(text: str) -> set[tuple[float, str]]:
    """The (value, unit) of each throughput, ms time and percentage in `text`; `5,894 tok/s` equals `5894 token/s`."""
    return {
        (float(number.replace(",", "")), "tok/s" if unit.endswith("/s") else unit.lstrip("\\"))
        for number, unit in FIGURE.findall(text)
    }


@cache
def measured_figures() -> frozenset[tuple[float, str]]:
    """Every figure some result of measurements.toml states."""
    runs: list[Run] = tomllib.loads(MEASUREMENTS.read_text())["run"]
    return frozenset(figure for run in runs for value in run["results"].values() for figure in figures(value))


def unmeasured_figures(text: str) -> list[str]:
    """Figures stated in the prose of `text` that no recorded run holds; a percentage counts only on a line that
    compares with native vLLM, since other percentages (shares of a step, thresholds) are not measurements."""
    failures: list[str] = []
    fenced = False
    for number, line in enumerate(text.splitlines(), start=1):
        fenced ^= line.lstrip().startswith("```")
        if fenced or NOT_MEASURED in line:
            continue
        stated = figures(line) - measured_figures()
        failures.extend(
            f"line {number}: {value:g} {unit} is not a result in doc/reference/measurements.toml; "
            f"quote a recorded run, or end the line with {NOT_MEASURED}"
            for value, unit in sorted(stated)
            if unit != "%" or NATIVE.search(line)
        )
    return failures


def user_document(path: Path) -> bool:
    """Whether `path` is prose for users that must stay true: not doc/dev/, not a dated essay, not example code."""
    return path.suffix != ".py" and not path.is_relative_to(DEV_DOCS) and not path.is_relative_to(ESSAYS)


@cache
def glossary_terms() -> tuple[str, ...]:
    """The bold terms the glossary explains."""
    return tuple(re.findall(r"\*\*([^*]+)\*\*", GLOSSARY.read_text()))


def unexplained_words(path: Path) -> list[str]:
    """JARGON words in a user document that no glossary term contains."""
    hits: dict[str, list[str]] = {}
    for index, text in enumerate(document_markdown(path)):
        where = f"cell {index} line " if path.suffix == ".ipynb" else "line "
        for number, line in enumerate(text.splitlines(), start=1):
            for match in JARGON.finditer(line):
                if not any(match.group() in term for term in glossary_terms()):
                    hits.setdefault(match.group(), []).append(f"{where}{number}")
    return [
        f"{', '.join(places)}: {word} is not explained in doc/reference/glossary.md; add an entry there, or say it in "
        "plain words (contributor vocabulary belongs in doc/dev/)"
        for word, places in sorted(hits.items())
    ]


# Runs each cell in one shared namespace under its own file name, so a traceback names the failing cell, and
# prints what each cell printed as one JSON object on the last line.
# Cells are registered in linecache, as Jupyter does, so inspect.getsource works (Triton's @jit needs it).
CELL_RUNNER = """import contextlib, io, json, linecache, sys
namespace = {"__name__": "__main__"}
printed = {}
for where, source in json.load(sys.stdin):
    name = f"<{where}>"
    linecache.cache[name] = (len(source), None, source.splitlines(True), name)
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
        exec(compile(source, name, "exec"), namespace)
    printed[where] = stdout.getvalue()
print(json.dumps(printed))
"""
CELL_FRAME = re.compile(r'File "<((?:cell|line) \d+)>", line (\d+)')


def run_cells(directory: Path, blocks: list[Block], *, gpu: bool = False) -> tuple[list[str], dict[str, str]]:
    """Run `blocks` in order in one fresh process in `directory`; without `gpu` CUDA is hidden.
    Returns the failure, if any, and what each block printed when all of them ran."""
    if not blocks:
        return [], {}
    cells = json.dumps([(block.where, block.source) for block in blocks])
    # vLLM's log handler would otherwise write into whichever cell's stdout is captured when it is configured.
    environment = {**os.environ, "PYTHONPATH": str(ROOT), "VLLM_CONFIGURE_LOGGING": "0"}
    if not gpu:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [sys.executable, "-c", CELL_RUNNER],
        input=cells,
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if completed.returncode == 0:
        return [], json.loads(completed.stdout.splitlines()[-1])
    error = (completed.stderr.strip().splitlines() or ["no output"])[-1]
    frames = CELL_FRAME.findall(completed.stderr)
    where = f"{frames[-1][0]} (+{frames[-1][1]})" if frames else "setup"
    return [f"{where}: code fails {'on GPU' if gpu else 'on CPU'}: {error}"], {}


def normalized(text: str) -> list[str]:
    """The lines of printed text without trailing spaces, with the parts that change between runs as `…`."""
    return [VOLATILE.sub("…", line.rstrip()) for line in text.strip("\n").splitlines()]


def output_mismatches(blocks: list[Block], printed: dict[str, str]) -> list[str]:
    """Blocks whose shown output differs from what they printed. A shown line `...` stands for any number of lines,
    and `...` within a line for any text, as in quoted error messages."""
    failures: list[str] = []
    for block in blocks:
        if block.output is None or block.where not in printed:
            continue
        shown, actual = normalized(block.output), normalized(printed[block.where])
        pattern = "".join("(?:.*\n)*" if line == "..." else f"{line_pattern(line)}\n" for line in shown)
        if not re.fullmatch(pattern, "".join(f"{line}\n" for line in actual)):
            failures.append(f"{block.where}: the output shown for it is stale: {first_difference(shown, actual)}")
    return failures


def line_pattern(line: str) -> str:
    """A shown output line as a regular expression in which `...` stands for any text."""
    return ".*".join(map(re.escape, line.split("...")))


def first_difference(shown: list[str], actual: list[str]) -> str:
    """The first shown line no longer printed, or else the first printed line not shown, for the report."""
    patterns = [line_pattern(line) for line in shown]
    for line, pattern in zip(shown, patterns):
        if not any(re.fullmatch(pattern, printed) for printed in actual):
            return f"{line!r} is no longer printed"
    for printed in actual:
        if not any(re.fullmatch(pattern, printed) for pattern in patterns):
            return f"{printed!r} is printed but not shown"
    return "the lines are printed in another order"


def store_outputs(path: Path, printed: dict[str, str]) -> None:
    """Make what each executed notebook cell printed its only stored output (`--write-outputs`)."""
    document = json.loads(path.read_text())
    for index, cell in enumerate(document["cells"]):
        if (text := printed.get(f"cell {index}")) is not None:
            stream = {"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)}
            cell["outputs"] = [stream] if text else []
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False, sort_keys=True) + "\n")  # as Jupyter writes


class _Default:
    """A parameter default as the signature list prints it: long values are elided."""

    def __init__(self, value: object) -> None:
        text = repr(value)
        self.text = text if len(text) <= 40 else "..."

    def __repr__(self) -> str:
        return self.text


def public_names() -> list[tuple[str, Callable[..., object]]]:
    """Every public value of the documented namespaces as (qualified name, value), sub-namespaces excluded."""
    names: list[tuple[str, Callable[..., object]]] = []
    for namespace in PUBLIC_NAMESPACES:
        module = importlib.import_module(namespace)
        for name in module.__all__:
            value: object = getattr(module, name)
            if not isinstance(value, ModuleType):
                assert callable(value), name  # every public value is a class or a function
                names.append((f"{namespace}.{name}", value))
    return names


def signature(name: str, value: Callable[..., object]) -> str:
    """`name(parameters)` for callables, `name: A | B` for enums."""
    is_class = inspect.isclass(value)
    if is_class and issubclass(value, Enum):
        return f"{name}: {' | '.join(member.name for member in value)}"
    head = inspect.signature(value)
    parameters = [
        parameter.replace(default=_Default(parameter.default))
        if parameter.default is not inspect.Parameter.empty
        else parameter
        for parameter in head.parameters.values()
    ]
    head = head.replace(parameters=parameters)
    if is_class:
        head = head.replace(return_annotation=inspect.Signature.empty)
    return f"{name}{head}"


def docstring(value: Callable[..., object]) -> str:
    """The value's own docstring: for a class, not an inherited one nor the dataclass-generated `Name(fields)`."""
    if not inspect.isclass(value):
        return inspect.cleandoc(value.__doc__ or "")
    text = inspect.cleandoc(vars(value).get("__doc__") or "")
    return "" if text.startswith(f"{value.__name__}(") else text


def signature_line(name: str, value: Callable[..., object]) -> str:
    """The signature with the docstring's first paragraph beneath, as the agent reference lists it."""
    summary = docstring(value).split("\n\n")[0].replace("\n", " ")
    return f"{signature(name, value)}\n    {summary}" if summary else signature(name, value)


def render_signatures(_: str) -> str:
    """The generated block of reference.md: one text list of signatures per namespace."""
    parts: list[str] = []
    for namespace in PUBLIC_NAMESPACES:
        lines = [signature_line(name, value) for name, value in public_names() if name.rsplit(".", 1)[0] == namespace]
        parts.append(f"### {namespace}\n\n```text\n" + "\n".join(lines) + "\n```\n")
    return "\n".join(parts)


def table_cell(text: str) -> str:
    """Text made safe for one markdown table cell."""
    return text.replace("|", "\\|")


def render_errors(chinese: bool) -> str:
    """The error tables of errors.toml: Chinese ones as sections of errors.md, English ones inside pitfalls.md."""
    sections: list[ErrorSection] = tomllib.loads(ERRORS.read_text())["section"]
    parts: list[str] = []
    for section in sections:
        title, note = (section["zh"], section.get("note_zh")) if chinese else (section["en"], section.get("note"))
        lines = [f"{'##' if chinese else '###'} {title}", ""]
        if note:
            lines += [note, ""]
        lines += ["| 报错（原文） | 为什么 | 改法 |" if chinese else "| Message | Why | Fix |", "| --- | --- | --- |"]
        for row in section["error"]:
            why, fix = (row["why_zh"], row["fix_zh"]) if chinese else (row["why"], row["fix"])
            lines.append(f"| `{table_cell(row['message'])}` | {table_cell(why)} | {table_cell(fix)} |")
        parts.append("\n".join(lines) + "\n")
    return "\n".join(parts)


def render_api(_: str) -> str:
    """The API page: each public name's signature and full docstring, in the sections of API_GROUPS."""
    values = dict(public_names())
    parts: list[str] = []
    for title, names in API_GROUPS:
        parts.append(f"## {title}\n")
        for name in (f"pllm.{short}" for short in names if f"pllm.{short}" in values):
            text = docstring(values[name])
            parts.append(
                f"### `{name}`\n\n```text\n{signature(name, values[name])}\n```\n" + (f"\n{text}\n" if text else "")
            )
    return "\n".join(parts)


def ungrouped_names() -> list[str]:
    """Public names API_GROUPS does not file exactly once, and names it files that are not public."""
    filed = [f"pllm.{short}" for _, names in API_GROUPS for short in names]
    public = [name for name, _ in public_names()]
    return [
        f"public name {name} is filed {filed.count(name)} times in API_GROUPS; file it under one section"
        for name in public
        if filed.count(name) != 1
    ] + [f"API_GROUPS files {name}, which is not a public name" for name in filed if name not in public]


def cells(line: str) -> list[str]:
    """The cells of one markdown table row."""
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", line)[1:-1]]


def default_text(field: dataclasses.Field[Any]) -> str:
    """A field's default as config.md shows it: 必填 when there is none, `Name()` for a default dataclass value,
    byte counts also in MiB."""
    assert field.default_factory is dataclasses.MISSING, field.name  # no config field uses a factory
    value = field.default
    if value is dataclasses.MISSING:
        return "必填"
    if dataclasses.is_dataclass(value) and value == type(value)():
        return f"`{type(value).__name__}()`"
    text = f'`"{value}"`' if isinstance(value, str) else f"`{value!r}`"
    if field.name.endswith("_bytes") and isinstance(value, int) and value and value % (1 << 20) == 0:
        return f"{text}（{value >> 20} MiB）"
    return text


def render_config(dotted: str, current: str) -> str:
    """A config dataclass's table: the field and default columns from the code, the other columns kept from the
    current table by field name (empty for a new field)."""
    module, name = dotted.rsplit(".", 1)
    fields = dataclasses.fields(getattr(importlib.import_module(module), name))
    header, separator, *rows = current.strip("\n").splitlines()
    written = {cells(row)[0].strip("`"): cells(row)[2:] for row in rows}
    blank = [""] * (len(cells(header)) - 2)
    lines = [header, separator]
    for field in fields:
        lines.append(
            "| " + " | ".join([f"`{field.name}`", default_text(field), *written.get(field.name, blank)]) + " |"
        )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class Generated:
    """A block of a document that this tool writes from code or data, between two comment markers. `render`
    receives the block's current text, so a generated table can keep the columns people write by hand."""

    path: Path
    name: str
    render: Callable[[str], str]

    @property
    def begin(self) -> str:
        return f"<!-- {self.name}: generated by tools/check_docs.py --write-generated; do not edit -->\n"

    @property
    def end(self) -> str:
        return f"<!-- end of generated {self.name} -->"

    def split(self, text: str) -> tuple[str, str, str] | None:
        """The text before, inside and after the block, or None when the document lacks its markers."""
        start, end = text.find(self.begin), text.find(self.end)
        if start < 0 or end < start:
            return None
        return text[: start + len(self.begin)], text[start + len(self.begin) : end], text[end:]


SIGNATURES = Generated(REFERENCE, "signatures", render_signatures)
GENERATED = (
    SIGNATURES,
    Generated(ROOT / "doc" / "reference" / "errors.md", "errors", lambda _: render_errors(chinese=True)),
    Generated(AGENT_DOCS / "pitfalls.md", "errors", lambda _: render_errors(chinese=False)),
    Generated(API, "api", render_api),
    *(
        Generated(CONFIG, f"{dotted.rsplit('.', 1)[1]} fields", partial(render_config, dotted))
        for dotted in CONFIG_CLASSES
    ),
)


def stale_blocks(path: Path, text: str) -> list[str]:
    """Generated blocks of `path` that are missing or differ from what their source renders now."""
    failures: list[str] = []
    for block in GENERATED:
        if block.path != path:
            continue
        parts = block.split(text)
        if parts is None:
            failures.append(f"the generated {block.name} block is missing; add its markers {block.begin.strip()}")
        elif parts[1] != block.render(parts[1]):
            failures.append(f"the generated {block.name} block is stale; run tools/check_docs.py --write-generated")
    return failures


def write_generated() -> None:
    """Rewrite every generated block from its source (`--write-generated`)."""
    for block in GENERATED:
        parts = block.split(block.path.read_text())
        assert parts is not None, f"{block.path} has no {block.name} markers"
        block.path.write_text(parts[0] + block.render(parts[1]) + parts[2])


def reference_failures(text: str) -> list[str]:
    """Public names the task sections of reference.md never mention."""
    parts = SIGNATURES.split(text)
    prose = text if parts is None else parts[0] + parts[2]
    return [
        f"public name {name} is missing from the task sections; add it where an agent would look for it"
        for name, _ in public_names()
        if not re.search(rf"(?<![\w.]){re.escape(name)}(?![\w])", prose)
    ]


def runs_code(path: Path) -> bool:
    """Whether the Python blocks of a markdown document are executed."""
    return path == ROOT / "README.md" or path.parent in EXECUTED_DOCS


def check_file(
    path: Path, include_untagged: bool, gpu: bool, anchors: dict[Path, set[str]], *, write_outputs: bool
) -> tuple[list[str], bool]:
    """Failures in one document, and whether it was checked at all."""
    if path.suffix == ".py":
        return missing_symbols([Block("file", path.read_text())]), True
    if path.suffix == ".md":
        text = path.read_text()
        blocks = markdown_blocks(text)
        failures = missing_symbols(blocks) + broken_links(path, [text], anchors) + stale_messages(text)
        if user_document(path):
            failures += unmeasured_figures(text) + unexplained_words(path)
        if runs_code(path):
            use_gpu = gpu and path.parent == AGENT_DOCS
            executed = [block for block in blocks if not block.skip_reason and (use_gpu or not block.gpu)]
            run_failures, printed = run_cells(path.parent, executed, gpu=use_gpu)
            failures += run_failures + output_mismatches(executed, printed)
            if path.parent != AGENT_DOCS:
                failures += [
                    f"{block.where}: use a check-docs: skip reason for GPU examples"
                    for block in blocks
                    if block.gpu and not block.skip_reason
                ]
        failures += stale_blocks(path, text)
        if path == REFERENCE:
            failures += reference_failures(text)
        if path == API:
            failures += ungrouped_names()
        return failures, True
    notebook = read_notebook(path)
    failures = broken_links(path, notebook.markdown, anchors) + unexplained_words(path)
    failures += [
        f"markdown cell {index} {failure}"
        for index, text in enumerate(notebook.markdown)
        for failure in unmeasured_figures(text)
    ]
    if not notebook.cpu_cells and not include_untagged:
        return failures, False
    failures += missing_symbols(notebook.code)
    run_failures, printed = run_cells(notebook.path.parent, notebook.cpu_cells)
    failures += run_failures
    if write_outputs and printed:
        store_outputs(path, printed)
    else:
        failures += [
            f"{failure}; store the new output with --write-outputs"
            for failure in output_mismatches(notebook.cpu_cells, printed)
        ]
    return failures, True


def documents() -> list[Path]:
    paths = [ROOT / "README.md", *(ROOT / "doc").rglob("*.md"), *(ROOT / "doc").rglob("*.ipynb")]
    paths += (ROOT / "examples").rglob("*.py")
    return sorted(path for path in paths if ".ipynb_checkpoints" not in path.parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="also check notebooks that have no cpu-check cell")
    parser.add_argument("--gpu", action="store_true", help="also run the doc/agent blocks fenced ```python gpu")
    parser.add_argument("--write-generated", action="store_true", help="rewrite every generated block from its source")
    parser.add_argument("--write-outputs", action="store_true", help="store what cpu-check notebook cells print")
    arguments = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if arguments.write_generated:
        write_generated()
    failed = 0
    skipped: list[str] = []
    opt_outs: list[str] = []
    unexecuted: dict[str, Notebook] = {}
    anchors: dict[Path, set[str]] = {}
    paths = documents()
    for path in paths:
        failures, checked = check_file(
            path, arguments.all, arguments.gpu, anchors, write_outputs=arguments.write_outputs
        )
        name = str(path.relative_to(ROOT))
        if path.suffix == ".md" and runs_code(path):
            opt_outs.extend(
                f"{name}:{block.where}: {block.skip_reason}"
                for block in markdown_blocks(path.read_text())
                if block.skip_reason
            )
        if path.suffix == ".ipynb" and (notebook := read_notebook(path)).untagged_cells:
            unexecuted[name] = notebook
        if not checked:
            skipped.append(name)
        for failure in failures:
            print(f"{name}: {failure}")
        failed += bool(failures)
    if skipped:
        print(f"skipped {len(skipped)} notebooks without {CPU_TAG} cells (pending rewrite): {', '.join(skipped)}")
    untagged = sum(len(notebook.untagged_cells) for notebook in unexecuted.values())
    print(f"notebook cells not executed (untagged: they load a model or need a GPU): {untagged}")
    for name, notebook in unexecuted.items():
        cells = len(notebook.untagged_cells) + len(notebook.cpu_cells)
        print(f"  {name}: {len(notebook.untagged_cells)} of {cells} code cells")
    print(f"execution opt-outs: {len(opt_outs)}")
    for opt_out in opt_outs:
        print(f"  {opt_out}")
    print(f"check_docs: {len(paths) - len(skipped)} documents checked, {failed} with failures")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
