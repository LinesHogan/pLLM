"""A decoder layer scan with explicit residual carry and selected layer outputs.

One external call executes one original layer. KV storage, attention metadata
and failure recovery belong to the backend invocation that binds these calls.
"""

from collections.abc import Callable

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from pllm.contracts.region_program import (
    AccessMode, AccessRange, AffineIndex, BankType, Block, BorrowedResource, Call, Constant, ExternalResource, Function, If,
    IndexTerm, Op, Parameter, Program, ResourceAccess, Region, Scalar, ScalarKind, ScalarType, Scan, Select, SizeParameter, TensorType, Value,
)
from pllm.contracts.backend_attachment import CausalModelDomain
from pllm.contracts.region_composition import RegionFunctionBody, inline_function_bodies
from pllm.contracts.region_imports import NativeKernelImport, RegionTensorImport
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.execution.codec import boundary_digest
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_native import source_region_functions
from pllm.execution.region_resources import ExternalFunction, PagedResourceBinding, RegionPageOwner, RegionResources
from pllm.execution.region_values import Bank, Data


LayerForward = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]
RowLayerForward = Callable[[int, int, int, int, torch.Tensor, torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]


def index_cell(argument: int) -> AccessRange:
    return AccessRange(AffineIndex(0, (IndexTerm(argument),)), AffineIndex(1, (IndexTerm(argument),)))


def kv_resource(domain: CausalModelDomain) -> BorrowedResource:
    return BorrowedResource("kv", (domain.requests, domain.layers, domain.positions))


def decoder_normalize_function(tensor: TensorType, layers: int) -> Function:
    """The model endpoint's signature is independent of a replay algorithm."""
    identity = boundary_digest(tensor, layers)
    return Function("decoder.final_norm@1:" + identity, (ScalarType.INDEX, tensor, tensor), (tensor,),
        ("context", "weights"), (ResourceAccess("context", AccessMode.READ, (index_cell(0),)),
        ResourceAccess("weights", AccessMode.READ, (AccessRange(AffineIndex(layers), AffineIndex(layers + 1)),))))


def decoder_layer_function(tensor: TensorType, domain: CausalModelDomain, *, residual: bool) -> Function:
    """The model layer signature is independent of taps and replay ranges."""
    index = ScalarType.INDEX
    positions = TensorType((1,), "torch.int64", tensor.device)
    accesses = (ResourceAccess("kv", AccessMode.READ,
        (index_cell(1), index_cell(3), AccessRange(AffineIndex(), AffineIndex(0, (IndexTerm(2),))))),
        ResourceAccess("kv", AccessMode.WRITE, (index_cell(1), index_cell(3), index_cell(2))),
        ResourceAccess("weights", AccessMode.READ, (index_cell(3),)),
        ResourceAccess("context", AccessMode.READ, (index_cell(0),)))
    identity = boundary_digest(tensor, domain)
    name = "decoder.layer_with_residual@1:" if residual else "decoder.layer_entry@1:"
    return Function(name + identity, (index, index, index, index, positions, tensor, *((tensor,) if residual else ())),
        (tensor, tensor), ("context", "kv", "weights"), accesses)


def decoder_region_workspace_bytes(hidden: int, width: int) -> int:
    # At most two taps, entry/carry roots, call-local input/output copies and
    # branch results; independent of layer count. Positions are int64 scalars.
    # This is additional to the original model/kernel/attention reservation.
    return 16 * hidden * width + 64


class DecoderSource:
    """A composable layer body; constructing it creates no nested executor."""

    def __init__(self, tensor: TensorType, layers: tuple[int, ...], domain: CausalModelDomain, *, taps: tuple[int, ...] = (),
                 weights: BorrowedResource | None = None, context: ExternalResource | BorrowedResource = ExternalResource("model"),
                 source_imports: tuple[RegionTensorImport, ...] = (),
                 function_bodies: tuple[RegionFunctionBody, ...] = ()) -> None:
        if (not layers or tuple(sorted(set(layers))) != layers or any(type(i) is not int or not 0 <= i < domain.layers for i in layers)
            or len(taps) > 2 or len(set(taps)) != len(taps) or any(i not in layers for i in taps)):
            raise ValueError("decoder region needs a positive layer count and at most two distinct layer taps")
        if len(tensor.shape) != 2 or tensor.shape[0] != 1 or tensor.shape[1] < 1:
            raise ValueError("decoder region requires a single token's hidden matrix")
        dtypes = {"torch.float16": torch.float16, "torch.bfloat16": torch.bfloat16,
                  "torch.float32": torch.float32, "torch.float64": torch.float64}
        if tensor.dtype not in dtypes:
            raise ValueError("decoder region requires floating hidden states")
        def add(hidden: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return (hidden + residual,)
        addition = None
        if taps:
            # Only tapped layers need this graph; layouts suffice on the driver.
            with FakeTensorMode():
                example = torch.empty(tensor.shape, dtype=dtypes[tensor.dtype], device=tensor.device)
                addition = ExportedCell.capture(add,
                    ("hidden", "residual"), ("sum",), (example, example.clone()))
        index = ScalarType.INDEX
        positions = TensorType((1,), "torch.int64", tensor.device)
        layer_ids = BankType(index, len(layers))
        if weights is not None and weights.domain != (domain.layers + 1,):
            raise ValueError("decoder weight family requires one cell per layer and final normalization")
        resources = (context.name, "kv", *(() if weights is None else (weights.name,)))
        identity = boundary_digest(tensor, layers, domain, taps, weights, context)
        accesses = (ResourceAccess("kv", AccessMode.READ,
            (index_cell(1), index_cell(3), AccessRange(AffineIndex(), AffineIndex(0, (IndexTerm(2),))))),
            ResourceAccess("kv", AccessMode.WRITE, (index_cell(1), index_cell(3), index_cell(2))),
            *(() if weights is None else (ResourceAccess(weights.name, AccessMode.READ, (index_cell(3),)),)),
            *((ResourceAccess(context.name, AccessMode.READ, (index_cell(0),)),) if isinstance(context, BorrowedResource) else ()))
        self.kv = kv_resource(domain)
        self.first = Function("decoder.first_layer@6:" + identity, (index, index, index, index, positions, tensor),
            (tensor, tensor), resources, accesses)
        self.next = Function("decoder.next_layer@6:" + identity, (index, index, index, index, positions, tensor, tensor),
            (tensor, tensor), resources, accesses)
        if weights is not None and weights.name == "weights" and isinstance(context, BorrowedResource) and context.name == "context":
            self.first = decoder_layer_function(tensor, domain, residual=False)
            self.next = decoder_layer_function(tensor, domain, residual=True)

        def tapped(prefix: str, ordinal: str, hidden: str, residual: str, old: tuple[str, ...]) -> tuple[tuple[Op, ...], tuple[str, ...]]:
            if addition is None:
                return (), ()
            operations: list[Op] = []
            results: list[str] = []
            args = (Value("hidden", tensor), Value("residual", tensor), Value("old", tensor))
            present = Region(Block(args, (Call(("hidden", "residual"), (Value("sum", tensor),), addition.digest),), ("sum",)))
            absent = Region(Block(args, (), ("old",)))
            for n, (layer, previous) in enumerate(zip(taps, old, strict=True)):
                target, condition, result = (f"{prefix}_{name}{n}" for name in ("target", "matches", "tap"))
                operations.extend((Constant((), (Value(target, index),), layer),
                    Scalar((ordinal, target), (Value(condition, ScalarType.BOOL),), ScalarKind.EQUAL),
                    If((condition, hidden, residual, previous), (Value(result, tensor),), present, absent)))
                results.append(result)
            return tuple(operations), tuple(results)

        coordinates = (Value("row", index), Value("request", index), Value("token", index), Value("layer_ids", layer_ids), Value("positions", positions))
        entry = (*coordinates, Value("seed", tensor))
        initial_taps, initial_names = tapped("initial", "first_layer", "hidden0", "residual0", ("seed",) * len(taps))
        operations: list[Op] = [
            Constant((), (Value("first_layer", index),), layers[0]),
            Call(("row", "request", "token", "first_layer", "positions", "seed"),
                (Value("hidden0", tensor), Value("residual0", tensor)), self.first.name),
            *initial_taps,
        ]
        returned = ("hidden0", "residual0", *initial_names)
        declarations = [self.first]
        if len(layers) > 1:
            carry = (Value("hidden", tensor), Value("residual", tensor),
                     *(Value(f"tap{i}", tensor) for i in range(len(taps))))
            tap_ops, tap_names = tapped("next", "layer", "hidden1", "residual1", tuple(f"tap{i}" for i in range(len(taps))))
            layer_index: tuple[Op, ...] = (
                Constant((), (Value("base", index),), layers[0] + 1),
                Scalar(("index", "base"), (Value("layer", index),), ScalarKind.ADD),
            ) if layers == tuple(range(layers[0], layers[0] + len(layers))) else (
                Constant((), (Value("one", index),), 1),
                Scalar(("index", "one"), (Value("ordinal", index),), ScalarKind.ADD),
                Select(("layer_ids", "ordinal"), (Value("layer", index),)),
            )
            body = Region(Block((Value("index", index), *carry, *coordinates), (
                *layer_index,
                Call(("row", "request", "token", "layer", "positions", "hidden", "residual"),
                     (Value("hidden1", tensor), Value("residual1", tensor)), self.next.name),
                *tap_ops,
            ), ("hidden1", "residual1", *tap_names)))
            results = tuple(Value("final_" + value.name, value.type) for value in carry)
            operations.append(Scan(("hidden0", "residual0", *initial_names, *tuple(value.name for value in coordinates)),
                                   results, len(layers) - 1, body, invariants=len(coordinates)))
            returned = tuple(value.name for value in results)
            declarations.append(self.next)
        if addition is not None:
            declarations.append(addition.declaration)
        self.program = Program(Region(Block(entry, tuple(operations), returned)), tuple(declarations),
            (context, self.kv, *(() if weights is None else (weights,))))
        self.program = inline_function_bodies(self.program, function_bodies) if function_bodies else self.program
        self.functions = source_region_functions(self.program.functions, {} if addition is None else {addition.digest: addition}, source_imports)
        self.layers = layers

    def externals(self, forward: RowLayerForward, kernel_imports: tuple[NativeKernelImport, ...] = ()) -> tuple[ExternalFunction, ...]:
        def invoke(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
            row, request, token, ordinal, *values = inputs
            position, value, *rest = values
            residual = None if not rest else rest[0]
            if (type(row) is not int or type(request) is not int or type(token) is not int
                or type(ordinal) is not int or ordinal not in self.layers or not isinstance(position, torch.Tensor)
                or not isinstance(value, torch.Tensor) or residual is not None and not isinstance(residual, torch.Tensor)):
                raise ValueError("decoder layer operands differ from the original region")
            return forward(row, request, token, ordinal, position, value, residual)
        kernels = {kernel.declaration.name: kernel for kernel in kernel_imports}
        return tuple(ExternalFunction(function, invoke, kernels.get(function.name)) for function in (self.first, self.next)
                     if function.name in self.functions)


class DecoderRegion:
    """Standalone binding of the same source used in composed model programs."""

    def __init__(self, tensor: TensorType, layers: tuple[int, ...], domain: CausalModelDomain, *, taps: tuple[int, ...] = ()) -> None:
        self.source = DecoderSource(tensor, layers, domain, taps=taps)
        source = self.source.program
        block = source.entry.block
        # The standalone layer segment has one request and token per invocation.
        # Keep these as bounded symbols in the same source, so KV access maps
        # retain their coordinates across every layer without recompilation.
        coordinates = (SizeParameter("request", 0, domain.requests - 1),
                       SizeParameter("token", 0, domain.positions - 1))
        self.program = Program(Region(Block(
            tuple(value for value in block.arguments if value.name not in ("request", "token")),
            (*(Parameter((), (Value(parameter.name, ScalarType.INDEX),), parameter.name) for parameter in coordinates),
             *block.operations), block.yields)), source.functions, source.resources, coordinates)
        schedule = analyze_region_program(self.program, optimize=False).reference_schedule
        self.execution = RegionExecution(self.program, lower_region_schedule(self.program, schedule),
            lower_region_bindings(self.program, schedule), self.source.functions)

    def run(self, owner: RegionPageOwner, original: PackedPhaseRow, positions: torch.Tensor, hidden: torch.Tensor,
            forward: LayerForward, *, kv: PagedResourceBinding) -> tuple[torch.Tensor, ...]:
        def layer(row: int, request: int, token: int, ordinal: int, position: torch.Tensor, value: torch.Tensor,
                  residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
            if (row, request, token) != (original.packed_row_index, original.request_ordinal, original.token_offset):
                raise ValueError("decoder call differs from its original packed KV coordinates")
            return forward(ordinal, position, value, residual)
        sizes = {"request": original.request_ordinal, "token": original.token_offset}
        resources = RegionResources(self.program, owner, (), self.source.externals(layer), sizes=sizes, borrows=(kv,))
        result = self.execution.run({"row": original.packed_row_index,
            "layer_ids": Bank(self.source.layers), "positions": positions, "seed": hidden}, pages=resources, sizes=sizes).values
        tensors: list[torch.Tensor] = []
        for value in result:
            if not isinstance(value, torch.Tensor):
                raise RuntimeError("decoder region returned a non-tensor result")
            tensors.append(value)
        return tuple(tensors)
