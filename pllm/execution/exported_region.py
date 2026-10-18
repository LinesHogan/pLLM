"""Import original export graphs with backend-bound external tensor kernels."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch.export.graph_signature import ExportGraphSignature, InputKind, OutputKind, TensorArgument

from pllm.contracts.region_program import Block, Call, Function, Program, Region, TensorType, Value, verify
from pllm.execution.codec import (
    check_static_export,
    functionalize_export,
    tensor_graph_result_nodes,
    tensor_node_metadata,
)
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.module_graph import TensorGraphPartition, partition_tensor_graph
from pllm.execution.region_native import FunctionalKernelPrimitive, NativePrimitive, TensorSource


@dataclass(frozen=True, slots=True)
class ExportedRegion:
    program: Program
    functions: Mapping[str, TensorSource]
    # Keep Torch's normalized signature: callers use its original lifted-state
    # targets and mutation-output positions, rather than guessing from names.
    signature: ExportGraphSignature


def import_exported_region(
    program: torch.export.ExportedProgram,
    *,
    bind_kernel: Callable[[TensorGraphPartition], NativePrimitive | FunctionalKernelPrimitive],
) -> ExportedRegion:
    """Keep original ATen fragments and delegate opaque fragments to a backend.

    All lifted parameters and constants remain explicit input ports. Functional
    mutation outputs retain export order; publication belongs to the caller.
    No local function closes over a model, parameter, or resource owner.
    """
    program = functionalize_export(program)
    program.validate()
    check_static_export(program)
    graph, signature = program.graph_module, program.graph_signature
    placeholders = tuple(node for node in graph.graph.nodes if node.op == "placeholder")

    def type_of(node: torch.fx.Node) -> TensorType:
        tensor = tensor_node_metadata(node)
        if (
            tensor is None
            or tensor.layout != torch.strided
            or any(type(size) is not int for size in (*tensor.shape, *tensor.stride()))
        ):
            raise ValueError("export region requires static strided tensor ports")
        return TensorType(tuple(tensor.shape), str(tensor.dtype), str(tensor.device))

    names: dict[str, str] = {}
    arguments: list[Value] = []
    for node, spec in zip(placeholders, signature.input_specs, strict=True):
        if not isinstance(spec.arg, TensorArgument) or spec.kind not in (
            InputKind.PARAMETER,
            InputKind.BUFFER,
            InputKind.CONSTANT_TENSOR,
            InputKind.USER_INPUT,
        ):
            raise ValueError("export region requires explicit lifted tensor inputs")
        name = spec.target if spec.target is not None else spec.arg.name
        names[node.name] = name
        arguments.append(Value(name, type_of(node)))
    if len({value.name for value in arguments}) != len(arguments):
        raise ValueError("export region input targets must be distinct")
    if any(
        not isinstance(spec.arg, TensorArgument)
        or spec.kind not in (OutputKind.USER_OUTPUT, OutputKind.USER_INPUT_MUTATION, OutputKind.BUFFER_MUTATION)
        for spec in signature.output_specs
    ):
        raise ValueError("export region requires tensor results and explicit mutation outputs")
    nodes = {node.name: node for node in graph.graph.nodes}
    operations: list[Call] = []
    functions: dict[str, TensorSource] = {}
    used = set(names.values())
    for partition in partition_tensor_graph(graph):
        if partition.opaque:
            cell: TensorSource = bind_kernel(partition)
            if cell.implementation is None or cell.declaration.resources:
                raise ValueError("opaque tensor fragments require a pure backend kernel contract")
        else:
            examples: list[torch.Tensor] = []
            for name in partition.inputs:
                tensor = tensor_node_metadata(nodes[name])
                if tensor is None:
                    raise ValueError("export partition lost its tensor input metadata")
                examples.append(tensor)
            cell = ExportedCell.from_export(torch.export.export(partition.graph, tuple(examples), strict=False))
        expected = Function(
            cell.declaration.name,
            tuple(type_of(nodes[name]) for name in partition.inputs),
            tuple(type_of(nodes[name]) for name in partition.outputs),
        )
        if cell.declaration != expected:
            raise ValueError("backend kernel changed its original graph frontier")
        results: list[Value] = []
        for name in partition.outputs:
            result = name
            while result in used:
                result += "_"
            used.add(result)
            names[name] = result
            results.append(Value(result, type_of(nodes[name])))
        operations.append(Call(tuple(names[name] for name in partition.inputs), tuple(results), cell.declaration.name))
        functions[cell.declaration.name] = cell
    result = Program(
        Region(
            Block(
                tuple(arguments),
                tuple(operations),
                tuple(names[node.name] for node in tensor_graph_result_nodes(graph)),
            )
        ),
        tuple(cell.declaration for cell in functions.values()),
    )
    verify(result)
    return ExportedRegion(result, MappingProxyType(functions), signature)
