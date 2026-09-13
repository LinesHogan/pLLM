"""A decoder layer scan with explicit residual carry and selected layer outputs.

One external call executes one original layer. KV storage, attention metadata
and failure recovery belong to the backend invocation that binds these calls.
"""

from collections.abc import Callable

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from pllm.contracts.region_program import (
    AccessMode, AccessRange, AffineIndex, BankType, Block, BorrowedResource, Call, Coordinate, ExternalResource, Function,
    IndexTerm, Op, Parameter, Program, ResourceAccess, Region, ScalarType, Scan, Select, SizeParameter, TensorType, Value,
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

        coordinates = (Value("row", index), Value("request", index), Value("token", index), Value("layer_ids", layer_ids), Value("positions", positions))
        entry = (*coordinates, Value("seed", tensor))
        contiguous = layers == tuple(range(layers[0], layers[0] + len(layers)))
        operations: list[Op] = [
            Coordinate((), (Value("first_layer", index),), AffineIndex(layers[0])),
            Call(("row", "request", "token", "first_layer", "positions", "seed"),
                (Value("hidden0", tensor), Value("residual0", tensor)), self.first.name),
        ]
        # Tap hits are constructor-known positions: a tap on the first layer is
        # computed at entry, and any other tap starts as the seed passthrough.
        # No Constant/EQUAL/If predicates are materialized for static facts.
        current = ["hidden0", "residual0"]
        for n, layer in enumerate(taps):
            name = f"initial_tap{n}"
            if layer == layers[0]:
                assert addition is not None
                operations.append(Call(("hidden0", "residual0"), (Value(name, tensor),), addition.digest))
                current.append(name)
            else:
                current.append("seed")
        returned = tuple(current)
        declarations = [self.first]
        if len(layers) > 1:
            declarations.append(self.next)
            hits: dict[int, int] = {}
            for n, layer in enumerate(taps):
                position = layers.index(layer)
                if position:
                    hits[position - 1] = n
            slot_names = ("hidden", "residual", *(f"tap{n}" for n in range(len(taps))))

            def layer_ops(cursor: str | None, offset: int, prefix: str = "") -> tuple[Op, ...]:
                # "layer" for a segment-local cursor with the segment's global
                # offset folded into the affine constant; cursor=None is an
                # unrolled single iteration whose position is a constant.
                layer, ordinal = f"{prefix}layer", f"{prefix}ordinal"
                terms = () if cursor is None else (IndexTerm(0),)
                inputs = () if cursor is None else (cursor,)
                if contiguous:
                    return (Coordinate(inputs, (Value(layer, index),),
                        AffineIndex(layers[0] + 1 + offset, terms)),)
                return (Coordinate(inputs, (Value(ordinal, index),), AffineIndex(offset + 1, terms)),
                    Select(("layer_ids", ordinal), (Value(layer, index),)))

            total = len(layers) - 1
            i = 0
            segment = 0
            while i < total:
                if i in hits:
                    # Unrolled tap iteration: hidden/residual advance once and
                    # the hit tap slot becomes this layer's addition result.
                    prefix = f"u{segment}_"
                    hidden, residual = f"{prefix}hidden", f"{prefix}residual"
                    operations.extend(layer_ops(None, i, prefix))
                    operations.append(Call(("row", "request", "token", f"{prefix}layer", "positions", current[0], current[1]),
                        (Value(hidden, tensor), Value(residual, tensor)), self.next.name))
                    n = hits[i]
                    tap = f"u{segment}_tap{n}"
                    assert addition is not None
                    operations.append(Call((hidden, residual), (Value(tap, tensor),), addition.digest))
                    current = [hidden, residual, *current[2:]]
                    current[2 + n] = tap
                    i += 1
                else:
                    j = i
                    while j < total and j not in hits:
                        j += 1
                    # A tap-free run: tap slots pass through the body unchanged.
                    body = Region(Block((Value("index", index), *(Value(name, tensor) for name in slot_names), *coordinates), (
                        *layer_ops("index", i),
                        Call(("row", "request", "token", "layer", "positions", "hidden", "residual"),
                             (Value("hidden1", tensor), Value("residual1", tensor)), self.next.name),
                    ), ("hidden1", "residual1", *slot_names[2:])))
                    outputs = tuple(Value(f"s{segment}_{name}", tensor) for name in slot_names)
                    operations.append(Scan((*current, *tuple(value.name for value in coordinates)),
                                           outputs, j - i, body, invariants=len(coordinates)))
                    current = [value.name for value in outputs]
                    i = j
                segment += 1
            returned = tuple(current)
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
