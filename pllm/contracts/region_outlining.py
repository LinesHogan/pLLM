"""Exact extraction of adjacent tensor calls, before physical scheduling."""

from dataclasses import dataclass
from collections.abc import Iterator

from pllm.contracts.region_program import (
    Block, Call, Function, If, Map, Program, Region, Scan, TensorType, While,
)
from pllm.contracts.tensor_access import TensorAccessGraph


@dataclass(frozen=True, slots=True)
class CallOutline:
    # Pairs (operation index, child region): loop body 0; if then/else 0/1;
    # while condition/body 0/1. Empty path addresses the entry block.
    region_path: tuple[int, ...]
    first: int
    count: int


@dataclass(frozen=True, slots=True)
class CallOutlineEvidence:
    site: CallOutline
    # Baseline obtained by exact tensor inlining, before numerical rewrites.
    # The local binder reconstructs this graph from original imported calls.
    access: TensorAccessGraph


@dataclass(frozen=True, slots=True)
class OutlinedProgram:
    program: Program
    function: Function
    # Original calls and their exact frontier. Tensor graph composition must
    # replay this body, not accept an opaque replacement function.
    body: Block


def call_outline_sites(program: Program, *, functions: frozenset[str] | None = None) -> Iterator[CallOutline]:
    """Enumerate contiguous pure tensor groups, retaining all control boundaries."""
    eligible = {f.name for f in program.functions if not f.resources
                and all(isinstance(t, TensorType) for t in (*f.inputs, *f.outputs))
                and (functions is None or f.name in functions)}

    def walk(region: Region, path: tuple[int, ...]) -> Iterator[CallOutline]:
        ops = region.block.operations
        first = 0
        while first < len(ops):
            end = first
            while end < len(ops):
                op = ops[end]
                if not isinstance(op, Call) or op.function not in eligible:
                    break
                end += 1
            for count in range(end - first, 1, -1):
                for start in range(first, end - count + 1):
                    yield CallOutline(path, start, count)
            first = max(first + 1, end)
        for index, op in enumerate(ops):
            children = ((op.body,) if isinstance(op, (Scan, Map)) else
                        (op.then_region, op.else_region) if isinstance(op, If) else
                        (op.condition, op.body) if isinstance(op, While) else ())
            for child, nested in enumerate(children):
                yield from walk(nested, (*path, index, child))

    yield from walk(program.entry, ())


