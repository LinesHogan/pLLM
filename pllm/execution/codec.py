"""Torch graph/storage adapters and compatibility exports of the shared codec."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from uuid import UUID

import torch
from torch import _decomp
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

from pllm.contracts.codec import (
    boundary_digest as boundary_digest,
    boundary_scalar as _primitive,
    decode_boundary as decode_boundary,
    encode_boundary as encode_boundary,
)


class _TensorHooks(Protocol):
    def register_hook(self, hook: Callable[[torch.Tensor], torch.Tensor]) -> torch.utils.hooks.RemovableHandle: ...


def register_gradient_hook(tensor: torch.Tensor, hook: Callable[[torch.Tensor], torch.Tensor]) -> None:
    cast(_TensorHooks, tensor).register_hook(hook)


TensorFunction = Callable[..., tuple[torch.Tensor, ...]]


class _FunctionalExport(Protocol):
    def run_decompositions(self, decomp_table: dict[object, object]) -> torch.export.ExportedProgram: ...

    @property
    def range_constraints(self) -> dict[object, object]: ...


class _TensorDecompositions(Protocol):
    def get_decompositions(self, aten_ops: tuple[OpOverload, ...]) -> dict[object, object]: ...


class _TensorCompiler(Protocol):
    def compile(self, function: TensorFunction, *, fullgraph: bool, dynamic: bool) -> TensorFunction: ...


def check_static_export(program: torch.export.ExportedProgram) -> None:
    if cast(_FunctionalExport, program).range_constraints:
        raise ValueError("tensor cell import requires static export shapes; symbolic constraints cannot be discarded")


def functionalize_export(program: torch.export.ExportedProgram) -> torch.export.ExportedProgram:
    """Functionalize mutations; retain an already functional core ATen graph.

    Apart from redundant tracing, AOT's RNG preservation initializes CUDA even
    for fake inputs. Pure source import must work in a driver without a device
    execution context. The same ATen check is applied again by the importer.
    """
    _check_export_metadata(program)
    _normalize_transposes(program)
    try:
        for node in program.graph.nodes:
            if node.op == "call_function":
                target: object = node.target
                check_functional_aten(target)
                if _needs_decomposition(node):
                    break
            elif node.op not in ("placeholder", "output"):
                break
        else:
            return program
    except ValueError:
        pass
    # Tuple-valued splits otherwise leave Python getitem nodes in a functional
    # export (notably SwiGLU and QKV projections). Use PyTorch's alias-preserving
    # view decompositions, so the existing tensor-only importer sees the actual
    # storage views. Do not admit arbitrary Python callables into the graph.
    splits = (torch.ops.aten.split.Tensor, torch.ops.aten.split_with_sizes.default)
    decompositions = cast(_TensorDecompositions, _decomp).get_decompositions(splits)
    result = cast(_FunctionalExport, program).run_decompositions(decompositions)
    _check_export_metadata(result)
    return result


def _normalize_transposes(program: torch.export.ExportedProgram) -> None:
    """Lower NumPy's axis reversal as the exact view, without another AOT export."""
    changed = False
    for node in program.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.numpy_T.default:
            operand = node.all_input_nodes[0]
            tensor = tensor_node_metadata(operand)
            if tensor is not None:
                node.target = torch.ops.aten.permute.default
                node.args, node.kwargs = (operand, list(reversed(range(tensor.ndim)))), {}
                changed = True
    if changed:
        program.graph_module.recompile()


def _needs_decomposition(node: torch.fx.Node) -> bool:
    # These pure views expose their complete backing through fake metadata.
    # Rank-two matmul is already a supported GEMM boundary. Other composites
    # still decompose, so hidden temporary allocations are not erased.
    operator = node.target
    assert isinstance(operator, OpOverload)
    if operator in (
        torch.ops.aten.detach.default,
        torch.ops.aten.t.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.permute.default,
    ):
        return False
    if operator == torch.ops.aten.matmul.default:
        operands = tuple(tensor_node_metadata(n) for n in node.all_input_nodes)
        if len(operands) == 2 and all(t is not None and t.ndim == 2 for t in operands):
            return False
    return torch.Tag.core not in operator.tags


def _check_export_metadata(program: torch.export.ExportedProgram) -> None:
    # Export inserts metadata assertions for tensor-to-tensor conversions.
    # Prove them on the guarded static metadata before importing tensor math.
    for node in tuple(program.graph.nodes):
        if node.op == "call_function" and node.target == torch.ops.aten._assert_tensor_metadata.default:
            sample = tensor_node_metadata(node.all_input_nodes[0])
            assert sample is not None
            with sample.fake_mode:
                torch.ops.aten._assert_tensor_metadata.default(
                    *torch.fx.map_arg(node.args, tensor_node_metadata), **node.kwargs
                )
            program.graph.erase_node(node)
    program.graph_module.recompile()


def check_functional_aten(target: object) -> None:
    if not isinstance(target, OpOverload):
        raise ValueError(f"export requires ATen operators, got {target}")
    schema: object = getattr(cast(object, target), "_schema")
    if not isinstance(schema, torch.FunctionSchema):
        raise ValueError("export operator has no supported schema")
    if target.namespace != "aten" or schema.is_mutable:
        raise ValueError(f"export contains a nonfunctional or external operator: {target}")
    if torch.Tag.nondeterministic_seeded in target.tags:
        raise ValueError("export RNG needs explicit instance-addressed inputs")


def tensor_graph_outputs(graph: torch.fx.GraphModule, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    values: object = graph(*args)
    if not isinstance(values, tuple):
        raise ValueError("export requires a flat tensor result tuple")
    result: list[torch.Tensor] = []
    for value in cast(tuple[object, ...], values):
        if not isinstance(value, torch.Tensor):
            raise ValueError("export outputs must be tensors")
        result.append(value)
    return tuple(result)


def compile_tensor_function(function: TensorFunction) -> TensorFunction:
    return cast(_TensorCompiler, torch).compile(function, fullgraph=True, dynamic=False)


class _TorchRandom(Protocol):
    def fork_rng(self, devices: list[int]) -> AbstractContextManager[None]: ...


def preserve_torch_rng(devices: list[int]) -> AbstractContextManager[None]:
    """Typed adapter for PyTorch's CPU/device RNG isolation context."""
    return cast(_TorchRandom, torch.random).fork_rng(devices)


def matmul_node_arguments(node: torch.fx.Node) -> tuple[torch.fx.Node, torch.fx.Node] | None:
    """Narrow FX's dynamic ATen arguments; matrix shape rules live in the pass."""
    if node.op != "call_function" or node.target not in (torch.ops.aten.mm.default, torch.ops.aten.matmul.default):
        return None
    args: tuple[object, ...] = node.args
    if node.kwargs or len(args) != 2:
        return None
    left, right = args
    if not isinstance(left, torch.fx.Node) or not isinstance(right, torch.fx.Node):
        return None
    return left, right


def tensor_node_metadata(node: torch.fx.Node) -> FakeTensor | None:
    value: object = node.meta.get("val")
    return value if isinstance(value, FakeTensor) else None


def tensor_graph_result_nodes(graph: torch.fx.GraphModule) -> tuple[torch.fx.Node, ...]:
    output = next(node for node in graph.graph.nodes if node.op == "output")
    values: object = output.args[0]
    if not isinstance(values, tuple) or not all(isinstance(v, torch.fx.Node) for v in cast(tuple[object, ...], values)):
        raise ValueError("tensor access import requires flat tensor graph outputs")
    return cast(tuple[torch.fx.Node, ...], values)


def refresh_tensor_metadata(graph: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Replay only fake tensor semantics on a private graph, including aliases."""
    result = deepcopy(graph)
    inputs: list[FakeTensor] = []
    for node in result.graph.nodes:
        node.meta.pop("tensor_meta", None)
        if node.op == "placeholder":
            value = tensor_node_metadata(node)
            if value is None:
                raise ValueError("tensor metadata propagation needs fake input tensors")
            inputs.append(value)
    FakeTensorProp(result, inputs[0].fake_mode if inputs else None).propagate_dont_convert_inputs(*inputs)
    return result


@dataclass(frozen=True, slots=True)
class TensorGraphCall:
    graph: torch.fx.GraphModule
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    output_positions: tuple[int, ...]


def inline_tensor_graphs(
    inputs: tuple[tuple[str, FakeTensor], ...], calls: tuple[TensorGraphCall, ...], outputs: tuple[str, ...]
) -> torch.fx.GraphModule:
    """Copy functional operators in order, then prove internal layout guards.

    This never executes user Python or allocates real tensors. The resulting
    FX graph still uses the original ATen kernels; backend fusion is separate.
    """
    graph = torch.fx.Graph()
    values: dict[str, torch.fx.Node] = {}
    mode = FakeTensorMode()
    for index, (name, sample) in enumerate(inputs):
        node = graph.placeholder(f"arg_{index}")
        with mode:
            node.meta["val"] = torch.empty_strided(
                tuple(sample.shape), tuple(sample.stride()), dtype=sample.dtype, device=sample.device
            )
        values[name] = node
    guards: list[tuple[str, FakeTensor]] = []
    for call in calls:
        placeholders = tuple(n for n in call.graph.graph.nodes if n.op == "placeholder")
        bindings = {node: values[name] for node, name in zip(placeholders, call.inputs, strict=True)}
        for node, mapped in bindings.items():
            sample = tensor_node_metadata(node)
            if sample is None:
                raise ValueError("call composition requires tensor input metadata")
            guards.append((mapped.name, sample))
        for node in call.graph.graph.nodes:
            if node.op == "call_function":
                check_functional_aten(node.target)
                if node.is_impure():
                    raise ValueError("call composition contains an undeclared effect")
                bindings[node] = graph.node_copy(node, lambda argument: bindings[argument])
            elif node.op not in ("placeholder", "output"):
                raise ValueError("call composition requires functional tensor graphs")
        raw = tensor_graph_result_nodes(call.graph)
        for name, position in zip(call.outputs, call.output_positions, strict=True):
            original = raw[position]
            mapped = bindings[original]
            values[name] = mapped
            sample = tensor_node_metadata(original)
            if sample is None:
                raise ValueError("call composition requires tensor output metadata")
            guards.append((mapped.name, sample))
    graph.output(tuple(values[name] for name in outputs))
    result = refresh_tensor_metadata(torch.fx.GraphModule(torch.nn.Module(), graph))
    nodes = {n.name: n for n in result.graph.nodes}
    for name, expected in guards:
        actual = tensor_node_metadata(nodes[name])
        if actual is None or (tuple(actual.shape), tuple(actual.stride()), actual.dtype, actual.device) != (
            tuple(expected.shape),
            tuple(expected.stride()),
            expected.dtype,
            expected.device,
        ):
            raise ValueError("composed graph does not satisfy an original call layout guard")
    return result


def replace_matmul_nodes(
    graph: torch.fx.GraphModule,
    inner_name: str,
    outer_name: str,
    operands: tuple[str, str, str],
    *,
    right_associated: bool,
) -> tuple[torch.fx.GraphModule, tuple[str, str]]:
    """Mechanically rebuild a checked two-product site on a private FX graph."""
    result = deepcopy(graph)
    nodes = {n.name: n for n in result.graph.nodes}
    inner, outer = nodes[inner_name], nodes[outer_name]
    inner_target, outer_target = inner.target, outer.target
    if (
        not isinstance(inner_target, OpOverload)
        or not isinstance(outer_target, OpOverload)
        or inner_target not in (torch.ops.aten.mm.default, torch.ops.aten.matmul.default)
        or outer_target not in (torch.ops.aten.mm.default, torch.ops.aten.matmul.default)
    ):
        raise ValueError("matmul rewrite requires two imported matrix products")
    inner_operator = cast(OpOverload, inner_target)
    outer_operator = cast(OpOverload, outer_target)
    a, b, c = (nodes[name] for name in operands)
    if tuple(inner.users) != (outer,):
        raise ValueError("matmul rewrite cannot erase a shared intermediate")

    def product(left: torch.fx.Node, right: torch.fx.Node, target: OpOverload) -> torch.fx.Node:
        lv, rv = tensor_node_metadata(left), tensor_node_metadata(right)
        if lv is None or rv is None:
            raise ValueError("matmul rewrite requires exported fake tensor metadata")
        node = result.graph.call_function(target, (left, right))
        # Evaluate metadata only. Do not allocate real tensors or carry over
        # stale intermediate shapes/ATen traces from the deleted product.
        with lv.fake_mode:
            node.meta["val"] = torch.mm(lv, rv)
        return node

    with result.graph.inserting_before(outer):
        first = product(b, c, inner_operator) if right_associated else product(a, b, inner_operator)
        final = product(a, first, outer_operator) if right_associated else product(first, c, outer_operator)
    outer.replace_all_uses_with(final)
    result.graph.erase_node(outer)
    result.graph.erase_node(inner)
    result.graph.lint()
    result.recompile()
    # Replacing the final product also changes storage aliases of downstream
    # views. Updating only the two new products leaves their metadata stale.
    return refresh_tensor_metadata(result), (first.name, final.name)


def batch_matmul_nodes(
    graph: torch.fx.GraphModule, names: tuple[str, ...]
) -> tuple[torch.fx.GraphModule, tuple[str, ...]]:
    """Pack independent rank-two products on a private imported FX graph.

    Requiring every operand before the first product makes this local rewrite
    topologically valid without an extra graph scheduler. Shape/type/layout
    guards and original-source evidence are checked by the mathematical pass.
    """
    result = deepcopy(graph)
    nodes = {node.name: node for node in result.graph.nodes}
    if len(names) < 2 or len(set(names)) != len(names) or any(name not in nodes for name in names):
        raise ValueError("matrix batching requires distinct source products")
    products = tuple(nodes[name] for name in names)
    ordered = tuple(result.graph.nodes)
    if tuple(sorted(products, key=ordered.index)) != products:
        raise ValueError("matrix batching must preserve the source product order")
    arguments: list[tuple[torch.fx.Node, torch.fx.Node]] = []
    available = set(ordered[: ordered.index(products[0])])
    for product in products:
        pair = matmul_node_arguments(product)
        if pair is None or any(value not in available for value in pair):
            raise ValueError("matrix batching operands must be available before all source products")
        arguments.append(pair)
    with result.graph.inserting_before(products[0]):
        left = result.graph.call_function(torch.ops.aten.stack.default, ([pair[0] for pair in arguments], 0))
        right = result.graph.call_function(torch.ops.aten.stack.default, ([pair[1] for pair in arguments], 0))
        batch = result.graph.call_function(torch.ops.aten.bmm.default, (left, right))
        outputs = tuple(
            result.graph.call_function(torch.ops.aten.select.int, (batch, 0, index)) for index in range(len(products))
        )
    for product, output in zip(products, outputs, strict=True):
        product.replace_all_uses_with(output)
        result.graph.erase_node(product)
    result.graph.lint()
    result.recompile()
    return refresh_tensor_metadata(result), tuple(node.name for node in (left, right, batch, *outputs))


class _SliceBounds(Protocol):
    start: object
    stop: object
    step: object


def _tensor_graph_search_key(graph: torch.fx.GraphModule, *, ordered: bool) -> str:
    references: dict[torch.fx.Node, object] = {}

    def argument(value: object) -> object:
        if isinstance(value, torch.fx.Node):
            return references[value]
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            return {"float": value.hex()}
        if type(value) is complex:
            return {"complex": (value.real.hex(), value.imag.hex())}
        if isinstance(value, (tuple, list)):
            return {
                "tuple" if isinstance(value, tuple) else "list": tuple(
                    argument(v) for v in cast(tuple[object, ...] | list[object], value)
                )
            }
        if isinstance(value, dict):
            fields = cast(dict[object, object], value)
            if any(type(key) is not str for key in fields):
                raise ValueError("tensor graph keyword arguments need string keys")
            return {"dict": {cast(str, key): argument(v) for key, v in fields.items()}}
        if isinstance(value, slice):
            bounds = cast(_SliceBounds, value)
            return {"slice": (argument(bounds.start), argument(bounds.stop), argument(bounds.step))}
        if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
            return {"torch_literal": str(value)}
        raise ValueError("unsupported literal in tensor rewrite search key")

    descriptors: list[object] = []
    for node in graph.graph.nodes:
        descriptor = (node.op, str(node.target), argument(node.args), argument(node.kwargs))
        references[node] = {"node": len(descriptors)} if ordered else {"expression": boundary_digest(descriptor)}
        descriptors.append(descriptor)
    return boundary_digest(tuple(descriptors) if ordered else tuple(sorted(boundary_digest(d) for d in descriptors)))


def tensor_graph_structure_digest(graph: torch.fx.GraphModule) -> str:
    """Search key retaining operation order, exact sharing and the output ABI.

    Generated node names are ignored. This is not algebraic evidence and never
    substitutes for imported storage/access facts or their source digests.
    """
    return _tensor_graph_search_key(graph, ordered=True)


def tensor_graph_expression_digest(graph: torch.fx.GraphModule) -> str:
    """Prune paths that return to source expressions, ignoring independent order.

    This includes all operations with multiplicity and ordered output values.
    It is only a pruning key within the guarded private-intermediate rewrite
    family, not an alias analysis or a general equivalence checker.
    """
    return _tensor_graph_search_key(graph, ordered=False)


def load_tensor_fields(path: Path, fields: tuple[str, ...]) -> tuple[torch.Tensor, ...]:
    """Decode a weights-only checkpoint into the caller's explicit tensor field order."""
    payload: object = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid tensor checkpoint: {path.name}")
    values = cast(dict[object, object], payload)
    tensors: list[torch.Tensor] = []
    for field in fields:
        value = values.get(field)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"invalid tensor checkpoint field {field}: {path.name}")
        tensors.append(value)
    return tuple(tensors)


class _CudaDevice(Protocol):
    uuid: object


class _CudaAllocator(Protocol):
    def memory_snapshot(self, mempool_id: tuple[int, int], include_traces: bool) -> list[dict[str, object]]: ...
    def get_device_properties(self, device: torch.device) -> _CudaDevice: ...


def cuda_device_uuid(device: torch.device) -> str:
    return str(UUID(str(cast(_CudaAllocator, torch.cuda).get_device_properties(device).uuid)))


CudaStorageHandle = tuple[int, bytes, int, int, bytes, int, bytes, bool]


class _CudaStorage(Protocol):
    def _share_cuda_(self) -> CudaStorageHandle: ...

    @staticmethod
    def _new_shared_cuda(
        device: int,
        handle: bytes,
        size: int,
        offset: int,
        counter: bytes,
        counter_offset: int,
        event: bytes,
        wait: bool,
    ) -> torch.UntypedStorage: ...

    def share(self) -> CudaStorageHandle:
        return self._share_cuda_()

    def open(self, handle: CudaStorageHandle) -> torch.UntypedStorage:
        return self._new_shared_cuda(*handle)


def share_cuda_storage(tensor: torch.Tensor) -> CudaStorageHandle:
    return _CudaStorage.share(cast(_CudaStorage, tensor.untyped_storage()))


def open_cuda_storage(handle: CudaStorageHandle) -> torch.UntypedStorage:
    return _CudaStorage.open(cast(_CudaStorage, torch.UntypedStorage), handle)


def cuda_graph_pool_bytes(pool: tuple[int, int], device: int) -> int:
    """Decode PyTorch allocator facts at the foreign-data boundary, only in cold planning."""
    if torch.cuda.get_allocator_backend() != "native" or pool == (0, 0):
        raise ValueError("CUDA graph residency requires its captured native allocator pool")
    segments = cast(_CudaAllocator, torch.cuda).memory_snapshot(pool, False)
    total = 0
    for segment in segments:
        if _primitive(segment["device"], int) == device:
            size = _primitive(segment["total_size"], int)
            if size < 0:
                raise ValueError("CUDA graph pool size must be nonnegative")
            total += size
    return total


@dataclass(frozen=True, slots=True)
class CudaProfileTotals:
    kernel_count: int
    kernel_microseconds: float
    cuda_api_microseconds: float
    cuda_enqueue_api_microseconds: float
    cuda_wait_api_microseconds: float
    gpu_activity_microseconds: float
    memcpy_bytes: int
    cuda_launch_api_count: int
    cuda_graph_launch_api_count: int


def _span_totals(spans: dict[tuple[str, str], list[tuple[float, float]]]) -> tuple[int, float]:
    count = 0
    total = 0.0
    for intervals in spans.values():
        end = float("-inf")
        for start, stop in sorted(intervals):
            count += start >= end
            total += max(0.0, stop - max(start, end))
            end = max(end, stop)
    return count, total


def cuda_profile_totals(path: Path) -> CudaProfileTotals:
    """Decode Kineto activity, merging nested API spans per thread and GPU spans per device.

    Enqueue API time is not total Python submission time and calls may block internally.
    Launch counts merge nested runtime/driver calls on each host thread; graph
    launches are a subset, not the number of device kernels executed by a graph.
    Kernel duration sums may overlap; GPU activity includes memcpy/memset without overlap.
    Memcpy bytes exclude kernel-internal loads/stores and all allocator figures.
    Foreign trace metadata is deliberately not a closed PLLM wire contract.
    """
    payload = cast(dict[str, object], json.loads(path.read_text()))
    events = cast(list[dict[str, object]], payload["traceEvents"])
    kernels, kernel_us, transfer = 0, 0.0, 0
    spans: dict[str, dict[tuple[str, str], list[tuple[float, float]]]] = {
        kind: {} for kind in ("api", "enqueue", "wait", "gpu", "launch", "graph")
    }
    for event in events:
        category = event.get("cat")
        if event.get("ph") != "X" or category not in (
            "kernel",
            "cuda_runtime",
            "cuda_driver",
            "gpu_memcpy",
            "gpu_memset",
        ):
            continue
        duration = _primitive(event["dur"], float)
        start = _primitive(event["ts"], float)
        device_event = category in ("kernel", "gpu_memcpy", "gpu_memset")
        key = (str(event["pid"]), "device" if device_event else str(event["tid"]))
        interval = (start, start + duration)
        spans["gpu" if device_event else "api"].setdefault(key, []).append(interval)
        if category == "kernel":
            kernels += 1
            kernel_us += duration
        elif category == "gpu_memcpy":
            args = cast(dict[str, object], event["args"])
            transfer += _primitive(args["bytes"], int)
        elif not device_event:
            name = _primitive(event["name"], str).lower()
            if "synchronize" in name:
                spans["wait"].setdefault(key, []).append(interval)
            elif "launch" in name or "async" in name and ("memcpy" in name or "memset" in name):
                spans["enqueue"].setdefault(key, []).append(interval)
                if "launch" in name:
                    spans["launch"].setdefault(key, []).append(interval)
                    if "graphlaunch" in name:
                        spans["graph"].setdefault(key, []).append(interval)
    totals = {key: _span_totals(value) for key, value in spans.items()}
    return CudaProfileTotals(
        kernels,
        kernel_us,
        totals["api"][1],
        totals["enqueue"][1],
        totals["wait"][1],
        totals["gpu"][1],
        transfer,
        totals["launch"][0],
        totals["graph"][0],
    )
