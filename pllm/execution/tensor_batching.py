"""Independent GEMM batching over source tensor mathematics, before selection."""

from collections.abc import Iterator
from dataclasses import replace

import torch

from pllm.contracts.tensor_rewrite import MatmulBatching
from pllm.execution.codec import (
    batch_matmul_nodes,
    boundary_digest,
    matmul_node_arguments,
    tensor_graph_outputs,
    tensor_graph_result_nodes,
    tensor_node_metadata,
)
from pllm.execution.exported_cell import ExportedCell, TensorSignature


def _signature(node: torch.fx.Node) -> TensorSignature | None:
    value = tensor_node_metadata(node)
    if (
        value is None
        or value.layout != torch.strided
        or value.ndim != 2
        or value.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        or value.device.type not in ("cpu", "cuda")
        or value.is_conj()
        or value.is_neg()
        or any(type(d) is not int or d <= 0 for d in value.shape)
        or any(type(s) is not int for s in value.stride())
    ):
        return None
    return TensorSignature.of(value)


def batching_steps(cell: ExportedCell) -> Iterator[tuple[ExportedCell, MatmulBatching]]:
    """Group shape-compatible independent products; retain all original users.

    Every operand must already be available before the first grouped product.
    Groups contain at most eight products per rewrite; further disjoint groups
    can be explored under the existing total rewrite/candidate budget. There
    is no local FLOP or timing filter: packing and retained backing storage are
    charged from the resulting access graph by the original region selector.
    """
    nodes = tuple(cell.graph.graph.nodes)
    for start, first in enumerate(nodes):
        pair = matmul_node_arguments(first)
        if pair is None:
            continue
        left, right, output = (_signature(node) for node in (*pair, first))
        if left is None or right is None or output is None:
            continue
        m, k = left.shape
        n = right.shape[1]
        if (
            right.shape != (k, n)
            or output.shape != (m, n)
            or any(spec.dtype != left.dtype or spec.device != left.device for spec in (right, output))
        ):
            continue
        available = set(nodes[:start])
        products: list[torch.fx.Node] = []
        operands: list[tuple[str, str]] = []
        for node in nodes[start:]:
            args = matmul_node_arguments(node)
            if args is None or any(argument not in available for argument in args):
                continue
            specs = tuple(_signature(argument) for argument in (*args, node))
            if any(
                spec is None or (spec.shape, spec.dtype, spec.device) != (want.shape, want.dtype, want.device)
                for spec, want in zip(specs, (left, right, output), strict=True)
            ):
                continue
            products.append(node)
            operands.append((args[0].name, args[1].name))
            if len(products) == 8:
                break
        if len(products) < 2:
            continue
        graph, replacement = batch_matmul_nodes(cell.graph, tuple(node.name for node in products))
        results = tensor_graph_result_nodes(graph)
        if any(
            (value := tensor_node_metadata(results[position])) is None or TensorSignature.of(value) != guard
            for position, guard in zip(cell.output_positions, cell.output_signatures, strict=True)
        ):
            continue
        record = MatmulBatching(
            cell.digest,
            boundary_digest(str(cell.graph.graph)),
            boundary_digest(str(graph.graph)),
            tuple(node.name for node in products),
            tuple(operands),
            replacement,
            (len(products), m, k, n),
            str(left.dtype),
            str(left.device),
            len(products) * (m * k + k * n) * torch.empty((), dtype=left.dtype).element_size(),
        )

        # Bind each yielded graph now; a late-bound closure over the generator's
        # next group would execute mathematics different from its evidence.
        yield _with_graph(cell, graph, len(products)), record


def _with_graph(cell: ExportedCell, graph: torch.fx.GraphModule, count: int) -> ExportedCell:
    def functional(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        values = tensor_graph_outputs(graph, args)
        return tuple(values[i] for i in cell.output_positions)

    return replace(
        cell,
        graph=graph,
        function=functional,
        transformations=(*cell.transformations, f"independent matmul batching ({count} products)"),
    )
