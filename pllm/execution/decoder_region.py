"""A decoder layer scan with explicit residual carry and selected layer outputs.

One external call executes one original layer. KV storage, attention metadata
and failure recovery belong to the backend invocation that binds these calls.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from pllm.contracts.backend_attachment import ActivationPatchBinding, ActivationPatchOperator, CausalModelDomain
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.region_composition import (
    RegionFunctionBody,
    inline_function_bodies,
    inline_region,
    select_function_bodies,
)
from pllm.contracts.region_imports import NativeKernelImport, RegionTensorImport
from pllm.contracts.region_program import (
    AccessMode,
    AccessRange,
    AffineIndex,
    BankType,
    Block,
    BorrowedResource,
    Call,
    Coordinate,
    ExternalResource,
    Function,
    If,
    IndexTerm,
    Op,
    Parameter,
    Program,
    Region,
    ResourceAccess,
    ScalarType,
    Scan,
    Select,
    SizeParameter,
    TensorType,
    Value,
)
from pllm.contracts.tensor_layout import TensorLayoutRequest
from pllm.execution.activation import low_rank_affine_delta
from pllm.execution.codec import boundary_digest
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.exported_resources import borrow_exported_inputs
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_native import RegionFunction, source_region_functions
from pllm.execution.region_resources import ExternalFunction, PagedResourceBinding, RegionPageOwner, RegionResources
from pllm.execution.region_values import Bank, Data

LayerForward = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]
RowLayerForward = Callable[
    [int, int, int, int, torch.Tensor, torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]
]


@dataclass(frozen=True, slots=True)
class DecoderActivation:
    binding: ActivationPatchBinding
    fields: tuple[TensorType, ...]

    def resource(self, max_rows: int) -> BorrowedResource:
        return BorrowedResource(f"activation.{self.binding.layer}", (max_rows,))


def decoder_activations(
    bindings: tuple[ActivationPatchBinding, ...], layouts: TensorLayoutRequest, resource: str, device: str
) -> tuple[DecoderActivation, ...]:
    if not bindings:
        return ()
    (layout,) = (entry.layout for entry in layouts.layouts if entry.resource == resource)
    fields = {field.field_id: TensorType(field.shape, "torch." + field.dtype.value, device) for field in layout.fields}
    return tuple(DecoderActivation(binding, tuple(fields[field] for field in binding.fields)) for binding in bindings)


def activation_selected_function(activation: DecoderActivation) -> Function:
    return Function(
        "activation.selected@1:" + boundary_digest(activation),
        (ScalarType.INDEX,),
        (ScalarType.BOOL,),
        ("context",),
        (ResourceAccess("context", AccessMode.READ, (index_cell(0),)),),
    )


def _activation_body(
    tensor: TensorType, domain: CausalModelDomain, context: BorrowedResource, activations: tuple[DecoderActivation, ...]
) -> tuple[RegionFunctionBody, dict[str, RegionFunction]]:
    """A source-level layer dispatch; the backend supplies only immutable row predicates and fields."""
    resources = tuple(activation.resource(context.domain[0]) for activation in activations)
    index = ScalarType.INDEX
    declaration = Function(
        "decoder.activation@1:" + boundary_digest(tensor, domain, activations),
        (index, index, tensor),
        (tensor,),
        (context.name, *(resource.name for resource in resources)),
        tuple(ResourceAccess(resource.name, AccessMode.READ, (index_cell(0),)) for resource in (context, *resources)),
    )
    arguments = (Value("row", index), Value("layer", index), Value("hidden", tensor))
    functions: dict[str, RegionFunction] = {}
    variants: list[RegionFunctionBody] = []
    for layer in range(domain.layers):
        activation = next((item for item in activations if item.binding.layer == layer), None)
        if activation is None:
            variants.append(
                RegionFunctionBody(
                    declaration, Program(Region(Block(arguments, (), ("hidden",))), (), (context, *resources))
                )
            )
            continue
        dtypes = {
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
        }
        dtype = dtypes[tensor.dtype]
        convert = activation.fields[0].dtype != tensor.dtype

        def add(hidden: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return (hidden + (torch.ops.aten._to_copy.default(delta, dtype=dtype) if convert else delta),)

        def affine(
            hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, rotation: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            return (hidden + low_rank_affine_delta(hidden, weight, bias, rotation),)

        with FakeTensorMode():
            examples = tuple(
                torch.empty(type_.shape, dtype=dtypes[type_.dtype], device=type_.device)
                for type_ in (tensor, *activation.fields)
            )
            if activation.binding.operator is ActivationPatchOperator.ADD:
                cell = ExportedCell.capture(add, ("hidden", "delta"), ("patched",), examples)
            else:
                cell = ExportedCell.capture(affine, ("hidden", "weight", "bias", "rotation"), ("patched",), examples)
        body, local = borrow_exported_inputs(
            cell,
            activation.resource(context.domain[0]),
            {name: ordinal for ordinal, name in enumerate(cell.inputs[1:])},
        )
        functions.update(local)
        selected = activation_selected_function(activation)
        inputs = (Value("hidden", tensor), Value("row", index))
        patched = inline_region(body.program, inputs, "patch")
        yes = Region(Block(inputs, patched.operations, tuple(value.name for value in patched.results)))
        no = Region(Block(inputs, (), ("hidden",)))
        variants.append(
            RegionFunctionBody(
                declaration,
                Program(
                    Region(
                        Block(
                            arguments,
                            (
                                Call(("row",), (Value("selected", ScalarType.BOOL),), selected.name),
                                If(("selected", "hidden", "row"), (Value("patched", tensor),), yes, no),
                            ),
                            ("patched",),
                        )
                    ),
                    (selected, *body.program.functions),
                    (context, *resources),
                ),
            )
        )
    return select_function_bodies(tuple(variants), selector="layer"), functions


def index_cell(argument: int) -> AccessRange:
    return AccessRange(AffineIndex(0, (IndexTerm(argument),)), AffineIndex(1, (IndexTerm(argument),)))


def kv_resource(domain: CausalModelDomain) -> BorrowedResource:
    return BorrowedResource("kv", (domain.requests, domain.layers, domain.positions))


def decoder_normalize_function(tensor: TensorType, layers: int) -> Function:
    """The model endpoint's signature is independent of a replay algorithm."""
    identity = boundary_digest(tensor, layers)
    return Function(
        "decoder.final_norm@1:" + identity,
        (ScalarType.INDEX, tensor, tensor),
        (tensor,),
        ("context", "weights"),
        (
            ResourceAccess("context", AccessMode.READ, (index_cell(0),)),
            ResourceAccess("weights", AccessMode.READ, (AccessRange(AffineIndex(layers), AffineIndex(layers + 1)),)),
        ),
    )


def decoder_layer_function(tensor: TensorType, domain: CausalModelDomain, *, residual: bool) -> Function:
    """The model layer signature is independent of taps and replay ranges."""
    index = ScalarType.INDEX
    positions = TensorType((1,), "torch.int64", tensor.device)
    accesses = (
        ResourceAccess(
            "kv",
            AccessMode.READ,
            (index_cell(1), index_cell(3), AccessRange(AffineIndex(), AffineIndex(0, (IndexTerm(2),)))),
        ),
        ResourceAccess("kv", AccessMode.WRITE, (index_cell(1), index_cell(3), index_cell(2))),
        ResourceAccess("weights", AccessMode.READ, (index_cell(3),)),
        ResourceAccess("context", AccessMode.READ, (index_cell(0),)),
    )
    identity = boundary_digest(tensor, domain)
    name = "decoder.layer_with_residual@1:" if residual else "decoder.layer_entry@1:"
    return Function(
        name + identity,
        (index, index, index, index, positions, tensor, *((tensor,) if residual else ())),
        (tensor, tensor),
        ("context", "kv", "weights"),
        accesses,
    )


def decoder_region_workspace_bytes(hidden: int, width: int) -> int:
    # At most two taps, entry/carry roots, call-local input/output copies and
    # branch results; independent of layer count. Positions are int64 scalars.
    # This is additional to the original model/kernel/attention reservation.
    return 16 * hidden * width + 64


class DecoderSource:
    """A composable layer body; constructing it creates no nested executor."""

    def __init__(
        self,
        tensor: TensorType,
        layers: tuple[int, ...],
        domain: CausalModelDomain,
        *,
        taps: tuple[int, ...] = (),
        weights: BorrowedResource | None = None,
        context: ExternalResource | BorrowedResource = ExternalResource("model"),
        source_imports: tuple[RegionTensorImport, ...] = (),
        function_bodies: tuple[RegionFunctionBody, ...] = (),
        activations: tuple[DecoderActivation, ...] = (),
        overlay: BorrowedResource | None = None,
    ) -> None:
        if (
            not layers
            or tuple(sorted(set(layers))) != layers
            or any(type(i) is not int or not 0 <= i < domain.layers for i in layers)
            or len(set(taps)) != len(taps)
            or any(i not in layers for i in taps)
        ):
            raise ValueError("decoder region needs a positive layer count and distinct taps within its slice")
        if len(tensor.shape) != 2 or tensor.shape[0] != 1 or tensor.shape[1] < 1:
            raise ValueError("decoder region requires a single token's hidden matrix")
        dtypes = {
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
        }
        if tensor.dtype not in dtypes:
            raise ValueError("decoder region requires floating hidden states")

        def add(hidden: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return (hidden + residual,)

        addition = None
        if taps:
            # Only tapped layers need this graph; layouts suffice on the driver.
            with FakeTensorMode():
                example = torch.empty(tensor.shape, dtype=dtypes[tensor.dtype], device=tensor.device)
                addition = ExportedCell.capture(add, ("hidden", "residual"), ("sum",), (example, example.clone()))
        index = ScalarType.INDEX
        positions = TensorType((1,), "torch.int64", tensor.device)
        layer_ids = BankType(index, len(layers))
        if weights is not None and weights.domain != (domain.layers + 1,):
            raise ValueError("decoder weight family requires one cell per layer and final normalization")
        resources = (
            context.name,
            "kv",
            *(() if weights is None else (weights.name,)),
            *(() if overlay is None else (overlay.name,)),
        )
        identity = boundary_digest(tensor, layers, domain, taps, weights, context, overlay)
        accesses = (
            ResourceAccess(
                "kv",
                AccessMode.READ,
                (index_cell(1), index_cell(3), AccessRange(AffineIndex(), AffineIndex(0, (IndexTerm(2),)))),
            ),
            ResourceAccess("kv", AccessMode.WRITE, (index_cell(1), index_cell(3), index_cell(2))),
            *(() if weights is None else (ResourceAccess(weights.name, AccessMode.READ, (index_cell(3),)),)),
            *(
                (ResourceAccess(context.name, AccessMode.READ, (index_cell(0),)),)
                if isinstance(context, BorrowedResource)
                else ()
            ),
        )
        if overlay is not None:
            from pllm.execution.token_program import overlay_accesses

            accesses += overlay_accesses(overlay)
        self.kv = kv_resource(domain)
        self.first = Function(
            "decoder.first_layer@6:" + identity,
            (index, index, index, index, positions, tensor),
            (tensor, tensor),
            resources,
            accesses,
        )
        self.next = Function(
            "decoder.next_layer@6:" + identity,
            (index, index, index, index, positions, tensor, tensor),
            (tensor, tensor),
            resources,
            accesses,
        )
        if (
            weights is not None
            and weights.name == "weights"
            and isinstance(context, BorrowedResource)
            and context.name == "context"
        ):
            self.first = decoder_layer_function(tensor, domain, residual=False)
            self.next = decoder_layer_function(tensor, domain, residual=True)

        patch_body = None
        patch_functions: dict[str, RegionFunction] = {}
        patch_resources: tuple[BorrowedResource, ...] = ()
        if activations:
            if not isinstance(context, BorrowedResource):
                raise ValueError("activation source requires the original packed-row context")
            patch_body, patch_functions = _activation_body(tensor, domain, context, activations)
            patch_resources = tuple(item.resource(context.domain[0]) for item in activations)

        def patched(hidden: str, layer: str) -> tuple[Op, ...]:
            return (
                ()
                if patch_body is None
                else (Call(("row", layer, "raw_" + hidden), (Value(hidden, tensor),), patch_body.declaration.name),)
            )

        def result_name(hidden: str) -> str:
            return hidden if patch_body is None else "raw_" + hidden

        contiguous = layers == tuple(range(layers[0], layers[0] + len(layers)))
        coordinates = (
            Value("row", index),
            Value("request", index),
            Value("token", index),
            *(() if contiguous else (Value("layer_ids", layer_ids),)),
            Value("positions", positions),
        )
        entry = (*coordinates, Value("seed", tensor))
        operations: list[Op] = [
            Coordinate((), (Value("first_layer", index),), AffineIndex(layers[0])),
            Call(
                ("row", "request", "token", "first_layer", "positions", "seed"),
                (Value(result_name("hidden0"), tensor), Value("residual0", tensor)),
                self.first.name,
            ),
            *patched("hidden0", "first_layer"),
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
                    return (Coordinate(inputs, (Value(layer, index),), AffineIndex(layers[0] + 1 + offset, terms)),)
                return (
                    Coordinate(inputs, (Value(ordinal, index),), AffineIndex(offset + 1, terms)),
                    Select(("layer_ids", ordinal), (Value(layer, index),)),
                )

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
                    operations.append(
                        Call(
                            ("row", "request", "token", f"{prefix}layer", "positions", current[0], current[1]),
                            (Value(result_name(hidden), tensor), Value(residual, tensor)),
                            self.next.name,
                        )
                    )
                    operations.extend(patched(hidden, f"{prefix}layer"))
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
                    body = Region(
                        Block(
                            (Value("index", index), *(Value(name, tensor) for name in slot_names), *coordinates),
                            (
                                *layer_ops("index", i),
                                Call(
                                    ("row", "request", "token", "layer", "positions", "hidden", "residual"),
                                    (Value(result_name("hidden1"), tensor), Value("residual1", tensor)),
                                    self.next.name,
                                ),
                                *patched("hidden1", "layer"),
                            ),
                            ("hidden1", "residual1", *slot_names[2:]),
                        )
                    )
                    outputs = tuple(Value(f"s{segment}_{name}", tensor) for name in slot_names)
                    operations.append(
                        Scan(
                            (*current, *tuple(value.name for value in coordinates)),
                            outputs,
                            j - i,
                            body,
                            invariants=len(coordinates),
                        )
                    )
                    current = [value.name for value in outputs]
                    i = j
                segment += 1
            returned = tuple(current)
        if addition is not None:
            declarations.append(addition.declaration)
            patch_functions[addition.digest] = addition
        if patch_body is not None:
            declarations.append(patch_body.declaration)
        self.program = Program(
            Region(Block(entry, tuple(operations), returned)),
            tuple(declarations),
            (
                context,
                self.kv,
                *(() if weights is None else (weights,)),
                *patch_resources,
                *(() if overlay is None else (overlay,)),
            ),
        )
        if patch_body is not None:
            self.program = inline_function_bodies(self.program, (patch_body,))
        self.program = inline_function_bodies(self.program, function_bodies) if function_bodies else self.program
        self.functions = source_region_functions(self.program.functions, patch_functions, source_imports)
        self.layers = layers

    def externals(
        self, forward: RowLayerForward, kernel_imports: tuple[NativeKernelImport, ...] = ()
    ) -> tuple[ExternalFunction, ...]:
        def invoke(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
            row, request, token, ordinal, *values = inputs
            position, value, *rest = values
            residual = None if not rest else rest[0]
            if (
                type(row) is not int
                or type(request) is not int
                or type(token) is not int
                or type(ordinal) is not int
                or ordinal not in self.layers
                or not isinstance(position, torch.Tensor)
                or not isinstance(value, torch.Tensor)
                or residual is not None
                and not isinstance(residual, torch.Tensor)
            ):
                raise ValueError("decoder layer operands differ from the original region")
            return forward(row, request, token, ordinal, position, value, residual)

        kernels = {kernel.declaration.name: kernel for kernel in kernel_imports}
        return tuple(
            ExternalFunction(function, invoke, kernels.get(function.name))
            for function in (self.first, self.next)
            if function.name in self.functions
        )


class DecoderRegion:
    """Standalone binding of the same source used in composed model programs."""

    def __init__(
        self, tensor: TensorType, layers: tuple[int, ...], domain: CausalModelDomain, *, taps: tuple[int, ...] = ()
    ) -> None:
        self.source = DecoderSource(tensor, layers, domain, taps=taps)
        source = self.source.program
        block = source.entry.block
        # The standalone layer segment has one request and token per invocation.
        # Keep these as bounded symbols in the same source, so KV access maps
        # retain their coordinates across every layer without recompilation.
        coordinates = (
            SizeParameter("request", 0, domain.requests - 1),
            SizeParameter("token", 0, domain.positions - 1),
        )
        self.program = Program(
            Region(
                Block(
                    tuple(value for value in block.arguments if value.name not in ("request", "token")),
                    (
                        *(
                            Parameter((), (Value(parameter.name, ScalarType.INDEX),), parameter.name)
                            for parameter in coordinates
                        ),
                        *block.operations,
                    ),
                    block.yields,
                )
            ),
            source.functions,
            source.resources,
            coordinates,
        )
        schedule = analyze_region_program(self.program, optimize=False).reference_schedule
        self.execution = RegionExecution(
            self.program,
            lower_region_schedule(self.program, schedule),
            lower_region_bindings(self.program, schedule),
            self.source.functions,
        )

    def run(
        self,
        owner: RegionPageOwner,
        original: PackedPhaseRow,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        forward: LayerForward,
        *,
        kv: PagedResourceBinding,
    ) -> tuple[torch.Tensor, ...]:
        def layer(
            row: int,
            request: int,
            token: int,
            ordinal: int,
            position: torch.Tensor,
            value: torch.Tensor,
            residual: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if (row, request, token) != (original.packed_row_index, original.request_ordinal, original.token_offset):
                raise ValueError("decoder call differs from its original packed KV coordinates")
            return forward(ordinal, position, value, residual)

        sizes = {"request": original.request_ordinal, "token": original.token_offset}
        resources = RegionResources(self.program, owner, (), self.source.externals(layer), sizes=sizes, borrows=(kv,))
        result = self.execution.run(
            {
                "row": original.packed_row_index,
                **(
                    {"layer_ids": Bank(self.source.layers)}
                    if any(value.name == "layer_ids" for value in self.program.entry.block.arguments)
                    else {}
                ),
                "positions": positions,
                "seed": hidden,
            },
            pages=resources,
            sizes=sizes,
        ).values
        tensors: list[torch.Tensor] = []
        for value in result:
            if not isinstance(value, torch.Tensor):
                raise RuntimeError("decoder region returned a non-tensor result")
            tensors.append(value)
        return tuple(tensors)
