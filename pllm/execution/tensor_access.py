"""Import actual export storage aliases; ask the original AER for memory costs."""

import torch

from pllm import _core
from pllm.contracts.tensor_access import (
    TensorAccess,
    TensorAccessGraph,
    TensorAccessKind,
    TensorMemoryProfile,
    TensorStorage,
    TensorStorageView,
)
from pllm.execution.codec import (
    boundary_digest,
    decode_boundary,
    encode_boundary,
    refresh_tensor_metadata,
    tensor_graph_result_nodes,
    tensor_node_metadata,
)
from pllm.execution.exported_cell import ExportedCell, TensorSignature

# This is an access-model vocabulary, not an ATen evaluator. Unlisted operators
# keep their storage facts, but their traffic stays explicitly unknown.
_VIEWS = frozenset(
    {
        "aten.alias.default",
        "aten.detach.default",
        "aten.permute.default",
        "aten.t.default",
        "aten.transpose.int",
        "aten.slice.Tensor",
        "aten.select.int",
        "aten.view.default",
        "aten.view.dtype",
        "aten._unsafe_view.default",
        "aten.reshape.default",
        "aten.squeeze.default",
        "aten.squeeze.dim",
        "aten.unsqueeze.default",
        "aten.expand.default",
        "aten.as_strided.default",
    }
)
_DENSE = frozenset(
    {
        "aten.mm.default",
        "aten.matmul.default",
        "aten.bmm.default",
        "aten.linear.default",
        "aten.add.Tensor",
        "aten.add.Scalar",
        "aten.sub.Tensor",
        "aten.sub.Scalar",
        "aten.mul.Tensor",
        "aten.mul.Scalar",
        "aten.div.Tensor",
        "aten.div.Scalar",
        "aten.neg.default",
        "aten.abs.default",
        "aten.relu.default",
        "aten.sigmoid.default",
        "aten.silu.default",
        "aten.tanh.default",
        "aten.exp.default",
        "aten.sqrt.default",
        "aten.square.default",
        "aten.pow.Tensor_Scalar",
        "aten.ones_like.default",
        "aten.rsqrt.default",
        "aten.sum.default",
        "aten.sum.dim_IntList",
        "aten.mean.default",
        "aten.mean.dim",
        "aten.clone.default",
        "aten._to_copy.default",
        "aten.cat.default",
        "aten.stack.default",
    }
)


def import_tensor_access_graph(cell: ExportedCell) -> TensorAccessGraph:
    """Read guarded fake metadata without allocating or evaluating user tensors.

    Storage identities are local import facts, never data pointers sent to Rust.
    Compiled/fused executables need their own lowering-derived access model.
    """
    if "torch.compile" in cell.transformations:
        raise ValueError("compiled tensor kernels need a backend access model")
    placeholders = tuple(n for n in cell.graph.graph.nodes if n.op == "placeholder")
    if len(placeholders) != len(cell.input_signatures):
        raise ValueError("tensor access graph differs from its input ABI")
    for node, signature in zip(placeholders, cell.input_signatures, strict=True):
        meta = tensor_node_metadata(node)
        if meta is None or TensorSignature.of(meta) != signature:
            raise ValueError("tensor access metadata differs from its input ABI")
    graph = refresh_tensor_metadata(cell.graph)
    handles: list[torch.UntypedStorage] = []
    storages: list[TensorStorage] = []
    values: list[TensorStorageView] = []
    operations: list[TensorAccess] = []
    inputs: list[str] = []
    storage_ids: dict[str, int] = {}
    for node in graph.graph.nodes:
        if node.op == "output":
            continue
        tensor = tensor_node_metadata(node)
        if (
            tensor is None
            or tensor.layout != torch.strided
            or any(type(d) is not int for d in (*tensor.shape, *tensor.stride()))
        ):
            raise ValueError(f"tensor access import needs static strided tensor metadata at {node.name}")
        offset = tensor.storage_offset()
        if type(offset) is not int:
            raise ValueError("tensor access import needs a static storage offset")
        storage = tensor.untyped_storage()
        if storage not in handles:
            handles.append(storage)
            storages.append(TensorStorage(str(tensor.device), storage.nbytes()))
        storage_id = handles.index(storage)
        values.append(
            TensorStorageView(
                node.name, storage_id, tuple(tensor.shape), tuple(tensor.stride()), offset, str(tensor.dtype)
            )
        )
        if node.op == "placeholder":
            inputs.append(node.name)
        else:
            arguments = tuple(n.name for n in node.all_input_nodes)
            operator = str(node.target)
            aliases = any(storage_ids[name] == storage_id for name in arguments)
            kind = (
                TensorAccessKind.VIEW
                if operator in _VIEWS and aliases
                else TensorAccessKind.DENSE
                if (operator in _DENSE or operator in _VIEWS) and not aliases
                else TensorAccessKind.UNKNOWN
            )
            operations.append(
                TensorAccess(
                    node.name,
                    operator,
                    kind,
                    arguments,
                    tuple(_argument(value) for value in node.args),
                    tuple((key, _argument(value)) for key, value in sorted(node.kwargs.items())),
                )
            )
        storage_ids[node.name] = storage_id
    raw_outputs = tensor_graph_result_nodes(graph)
    if len(cell.output_positions) != len(cell.output_signatures) or any(
        i < 0 or i >= len(raw_outputs) for i in cell.output_positions
    ):
        raise ValueError("tensor access graph differs from its output ABI")
    for position, signature in zip(cell.output_positions, cell.output_signatures, strict=True):
        meta = tensor_node_metadata(raw_outputs[position])
        if meta is None or TensorSignature.of(meta) != signature:
            raise ValueError("tensor access metadata differs from its output ABI")
    return TensorAccessGraph(
        cell.digest,
        boundary_digest(str(cell.graph.graph)),
        tuple(storages),
        tuple(values),
        tuple(inputs),
        tuple(raw_outputs[i].name for i in cell.output_positions),
        tuple(operations),
    )


def _argument(value: object) -> str:
    return "$" + value.name if isinstance(value, torch.fx.Node) else repr(value)


def profile_tensor_memory(cell: ExportedCell) -> TensorMemoryProfile:
    """AER derives alias-aware lifetimes and logical access volumes from imports."""
    graph = import_tensor_access_graph(cell)
    return decode_boundary(_core.profile_tensor_memory(encode_boundary(graph)), TensorMemoryProfile)


def validate_tensor_memory(cell: ExportedCell, profile: TensorMemoryProfile) -> None:
    """Reimport source facts, then independently check the native memory witness."""
    _core.validate_tensor_memory(encode_boundary(import_tensor_access_graph(cell)), encode_boundary(profile))
