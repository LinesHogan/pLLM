"""Source-verified call outlining and local functional tensor graph composition."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch._subclasses.fake_tensor import FakeTensor

from pllm import _core
from pllm.contracts.region_outlining import CallOutline, OutlinedProgram
from pllm.contracts.region_program import Call, Program
from pllm.execution.codec import (
    TensorGraphCall, decode_boundary, encode_boundary, inline_tensor_graphs,
    tensor_graph_outputs, tensor_graph_result_nodes, tensor_node_metadata,
)
from pllm.execution.exported_cell import ExportedCell, TensorSignature
from pllm.execution.region_native import TensorSource, is_original_region_source
from pllm.execution.region_storage import import_region_functions
from pllm.execution.tensor_access import import_tensor_access_graph


def outline_region_program(program: Program, site: CallOutline) -> OutlinedProgram:
    """Derive the original calls and escaping values with the native authority."""
    return decode_boundary(_core.outline_region_program(encode_boundary(program), encode_boundary(site)), OutlinedProgram)


@dataclass(frozen=True, slots=True)
class OutlinedRegion:
    source: Program
    site: CallOutline
    outlined: OutlinedProgram
    cell: ExportedCell


def outline_region_calls(program: Program, sources: Mapping[str, TensorSource], site: CallOutline) -> OutlinedRegion:
    """Replay exact inlining with all escaping versions and original layout guards.

    The returned graph is inspectable and can be rewritten across former call
    boundaries. It is not a fused backend kernel or a selected physical plan.
    Storage and scheduling must be rederived from its transformed Program.
    """
    if any(not is_original_region_source(cell) for cell in sources.values()):
        raise ValueError("call outlining requires original source implementations")
    import_region_functions(program, sources)
    outlined = outline_region_program(program, site)
    calls: list[TensorGraphCall] = []
    frontier = {v.name for v in outlined.body.arguments}
    examples: dict[str, torch.fx.Node] = {}
    for op in outlined.body.operations:
        if not isinstance(op, Call):
            raise ValueError("outlined body must consist of original tensor calls")
        cell = sources[op.function]
        if not isinstance(cell, ExportedCell):
            raise ValueError("resource borrows and opaque kernels cannot be absorbed into a tensor graph")
        placeholders = tuple(n for n in cell.graph.graph.nodes if n.op == "placeholder")
        for name, node in zip(op.inputs, placeholders, strict=True):
            if name in frontier:
                examples.setdefault(name, node)
        calls.append(TensorGraphCall(cell.graph, op.inputs, tuple(v.name for v in op.outputs), cell.output_positions))
    inputs: list[tuple[str, FakeTensor]] = []
    for value in outlined.body.arguments:
        sample = tensor_node_metadata(examples[value.name])
        if sample is None:
            raise ValueError("outlined frontier requires original tensor metadata")
        inputs.append((value.name, sample))
    graph = inline_tensor_graphs(tuple(inputs), tuple(calls), outlined.body.yields)

    def signature(node: torch.fx.Node) -> TensorSignature:
        sample = tensor_node_metadata(node)
        if sample is None:
            raise ValueError("outlined graph has no tensor metadata")
        return TensorSignature.of(sample)

    def functional(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tensor_graph_outputs(graph, args)

    cell = ExportedCell(outlined.function.name, tuple(v.name for v in outlined.body.arguments), outlined.body.yields,
        tuple(signature(n) for n in graph.graph.nodes if n.op == "placeholder"),
        tuple(signature(n) for n in tensor_graph_result_nodes(graph)), (), tuple(range(len(outlined.body.yields))),
        graph, functional, ())
    if cell.declaration != outlined.function:
        raise ValueError("composed tensor graph differs from the outlined source ABI")
    import_tensor_access_graph(cell)
    return OutlinedRegion(program, site, outlined, cell)
