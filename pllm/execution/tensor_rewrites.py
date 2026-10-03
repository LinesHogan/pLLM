"""Guarded algebraic candidates over the imported functional ATen graph.

This pass retains the source declaration and export ABI. It does not select a
plan, warn, benchmark on user state, or obtain any runtime effect capability.
"""

from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Generic, TypeVar

import torch

from pllm.contracts.tensor_rewrite import MatmulAssociation, MatmulCost, MatmulReassociation, TensorRewrite
from pllm.execution.codec import (
    boundary_digest, matmul_node_arguments, replace_matmul_nodes,
    tensor_graph_outputs, tensor_graph_structure_digest, tensor_graph_expression_digest, tensor_node_metadata,
)
from pllm.execution.exported_cell import ExportedCell, TensorSignature
from pllm.execution.tensor_batching import batching_steps

RewriteT = TypeVar("RewriteT", bound=TensorRewrite, covariant=True)


@dataclass(frozen=True, slots=True)
class TensorRewriteCandidate(Generic[RewriteT]):
    cell: ExportedCell
    rewrite: RewriteT


@dataclass(frozen=True, slots=True)
class TensorRewritePath(Generic[RewriteT]):
    cell: ExportedCell
    rewrites: tuple[RewriteT, ...]


def _matrix_signature(node: torch.fx.Node) -> TensorSignature | None:
    tensor = tensor_node_metadata(node)
    if tensor is None or tensor.layout != torch.strided or tensor.ndim != 2:
        return None
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return None
    if tensor.device.type not in ("cpu", "cuda") or tensor.is_conj() or tensor.is_neg():
        return None
    # Dynamic tensor shapes require correlated guards, not int(SymInt) samples.
    if any(type(d) is not int for d in (*tensor.shape, *tensor.stride())):
        return None
    return TensorSignature.of(tensor)


def _candidate(cell: ExportedCell, outer: torch.fx.Node, association: MatmulAssociation
               ) -> TensorRewriteCandidate[MatmulReassociation] | None:
    outer_args = matmul_node_arguments(outer)
    if outer_args is None:
        return None
    to_right = association is MatmulAssociation.LEFT_TO_RIGHT
    inner = outer_args[0] if to_right else outer_args[1]
    inner_args = matmul_node_arguments(inner)
    if (inner_args is None or tuple(inner.users) != (outer,)
            or sum(arg is inner for arg in outer_args) != 1):
        return None
    a, b, c = (*inner_args, outer_args[1]) if to_right else (outer_args[0], *inner_args)
    sa, sb, sc, si, so = (_matrix_signature(n) for n in (a, b, c, inner, outer))
    if sa is None or sb is None or sc is None or si is None or so is None:
        return None
    if any(s.dtype != sa.dtype or s.device != sa.device for s in (sb, sc, si, so)):
        return None
    m, k = sa.shape
    n, p = sc.shape
    if sb.shape != (k, n) or si.shape != ((m, n) if to_right else (k, p)) or so.shape != (m, p):
        return None
    item_bytes = torch.empty((), dtype=sa.dtype).element_size()
    left = MatmulCost(2 * (m * k * n + m * n * p), m * n * item_bytes)
    right = MatmulCost(2 * (k * n * p + m * k * p), k * p * item_bytes)
    operands = (a.name, b.name, c.name)
    graph, replacement = replace_matmul_nodes(cell.graph, inner.name, outer.name, operands,
                                               right_associated=to_right)
    result_node = next(node for node in graph.graph.nodes if node.name == replacement[1])
    if _matrix_signature(result_node) != so:
        return None  # Layout is part of the exported ABI, even for internal users.
    record = MatmulReassociation(cell.digest, boundary_digest(str(cell.graph.graph)),
        boundary_digest(str(graph.graph)), association, (inner.name, outer.name), replacement,
        operands, (m, k, n, p), str(sa.dtype), str(sa.device),
        left if to_right else right, right if to_right else left)

    def functional(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        values = tensor_graph_outputs(graph, args)
        return tuple(values[i] for i in cell.output_positions)

    return TensorRewriteCandidate(replace(cell, graph=graph, function=functional,
        transformations=(*cell.transformations, f"matmul reassociation: {association.value}")), record)


def _steps(cell: ExportedCell) -> Iterator[TensorRewriteCandidate[MatmulReassociation]]:
    """Single-site, rank-2 reassociations without choosing a winner.

    Matrix associativity preserves real arithmetic, but changes floating-point
    evaluation (including possible overflow). Every candidate therefore requires
    the numerical-transform policy. FLOPs and the internal product size are
    analytical facts, not guarantees of speed, peak memory or numerical error.
    """
    for outer in cell.graph.graph.nodes:
        for association in MatmulAssociation:
            candidate = _candidate(cell, outer, association)
            if candidate is not None:
                yield candidate


def _all_steps(cell: ExportedCell) -> Iterator[TensorRewriteCandidate[TensorRewrite]]:
    yield from _steps(cell)
    for transformed, record in batching_steps(cell):
        yield TensorRewriteCandidate(transformed, record)


def replay_tensor_rewrites(source: ExportedCell, records: tuple[TensorRewrite, ...]) -> ExportedCell:
    """Replay mixed algebra/batching steps against their exact predecessor graph."""
    return _replay(source, records, _all_steps)


def _replay(source: ExportedCell, records: tuple[RewriteT, ...],
            steps: Callable[[ExportedCell], Iterator[TensorRewriteCandidate[RewriteT]]]) -> ExportedCell:
    if source.transformations:
        raise ValueError("tensor rewrite replay requires the original exported cell")
    current = source
    for record in records:
        if source.digest != record.source_digest:
            raise ValueError("tensor rewrite belongs to a different source")
        for candidate in steps(current):
            if candidate.rewrite == record:
                current = candidate.cell
                break
        else:
            raise ValueError("tensor rewrite evidence does not match the source graph and prerequisites")
    return current


def explore_tensor_combinations(sources: tuple[ExportedCell, ...], *, max_depth: int = 2,
                                max_candidates: int = 16) -> tuple[tuple[TensorRewrite, ...], ...]:
    """Joint bounded search over reassociation and independent GEMM batching.

    Changing only one call can leave another call's large temporary unchanged.
    Keep combinations until the region's actual lifetimes and completed time
    can be assessed. Function order is canonical; commuting rewrites do not
    create duplicate candidates. Intermediates can be locally more expensive:
    later rules and region-level lifetimes determine the complete candidate's
    cost. Already visited graph structures are excluded, preventing inverse
    cycles from creating candidates just by giving their nodes new names.
    """
    ordered = tuple(sorted(sources, key=lambda cell: cell.digest))
    return tuple(tuple(record for path in state for record in path.rewrites)
                 for state in _explore(ordered, max_depth, max_candidates, _all_steps))


def _explore(sources: tuple[ExportedCell, ...], max_depth: int,
             max_candidates: int, steps: Callable[[ExportedCell], Iterator[TensorRewriteCandidate[RewriteT]]]
             ) -> Iterator[tuple[TensorRewritePath[RewriteT], ...]]:
    if any(source.transformations for source in sources):
        raise ValueError("rewrite search requires the original exported cell")
    if len({source.digest for source in sources}) != len(sources):
        raise ValueError("rewrite search requires unique source function identities")
    if (type(max_depth) is not int or max_depth < 0 or type(max_candidates) is not int or max_candidates < 0):
        raise ValueError("rewrite search requires nonnegative depth and candidate limits")
    if not sources or max_depth == 0 or max_candidates == 0:
        return
    initial_keys = tuple(tensor_graph_structure_digest(source.graph) for source in sources)
    original_expressions = tuple(tensor_graph_expression_digest(source.graph) for source in sources)
    seen = {initial_keys}
    initial: tuple[TensorRewritePath[RewriteT], ...] = tuple(TensorRewritePath(source, ()) for source in sources)
    pending = deque(((initial, initial_keys, 0),))
    produced = 0
    while pending:
        state, keys, depth = pending.popleft()
        if depth >= max_depth:
            continue
        for index, parent in enumerate(state):
            for candidate in steps(parent.cell):
                key = tensor_graph_structure_digest(candidate.cell.graph)
                next_keys = (*keys[:index], key, *keys[index + 1:])
                if next_keys in seen:
                    continue
                seen.add(next_keys)
                if tensor_graph_expression_digest(candidate.cell.graph) == original_expressions[index]:
                    continue  # No mathematical change; do not select an inverse cycle.
                path = TensorRewritePath(candidate.cell, (*parent.rewrites, candidate.rewrite))
                next_state = (*state[:index], path, *state[index + 1:])
                yield next_state
                produced += 1
                if produced == max_candidates:
                    return
                pending.append((next_state, next_keys, depth + 1))
