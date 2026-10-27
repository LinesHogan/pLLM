"""Extract carry-independent coefficients and readouts from functional FX scan steps."""

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch._functorch.compile_utils import fx_graph_cse  # pyright: ignore[reportUnknownVariableType]
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Graph, GraphModule, Node
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.node import Argument, Target
from torch.utils._python_dispatch import _disable_current_modes  # pyright: ignore[reportPrivateUsage]

Kind = Literal["scalar", "diagonal", "left", "right"]
Coefficient = Node | float
Specs = tuple[tuple[torch.Size, tuple[int, ...], torch.dtype, torch.device], ...]


def stage(value: torch.Tensor, order: tuple[int, ...], capacity: int) -> torch.Tensor:
    """Materialize a shared GEMM batch layout while retaining the author's axis order."""
    axes = (0, *(axis + 1 for axis in order))
    ordered = value.permute(axes)
    if len(value) == capacity:
        packed = ordered.contiguous()
    else:
        packed = value.new_zeros((capacity, *ordered.shape[1:]))
        packed[: len(value)].copy_(ordered)
    return packed.permute(tuple(axes.index(axis) for axis in range(value.ndim)))


@dataclass(frozen=True)
class Linear:
    """One scalar, pointwise or one-sided matrix action, optionally on transposed state."""

    kind: Kind = "scalar"
    coefficient: Coefficient = 1.0
    transposed: bool = False


@dataclass(frozen=True)
class Affine:
    """A symbolic linear map and its independent offset; None denotes an exact zero."""

    linear: Linear | None
    bias: Node | None


@dataclass(frozen=True)
class ScanAnalysis:
    """Pure coefficient and exclusive-state readout graphs; no invocation storage."""

    kind: Literal["additive", "diagonal", "left", "right"]
    coefficients: GraphModule
    readout: GraphModule
    orders: tuple[tuple[int, ...], ...] = ()
    scale: float = 1.0
    copy_readout: bool = True

    def batched(self, samples: tuple[torch.Tensor, ...], sequences: int) -> "ScanAnalysis":
        """Lower vmap once on shape-only operands, preserving shared preprocessing nodes."""
        orders = self.matrix_orders(sequences)
        samples = (
            samples[0],
            *(stage(t, order, len(t)) for t, order in zip(samples[1 : 1 + sequences], orders, strict=True)),
            *samples[1 + sequences :],
        )
        dimensions = (None, *(0 for _ in range(sequences)), *(None for _ in samples[1 + sequences :]))
        coefficients = make_fx(torch.func.vmap(self.coefficients, in_dims=dimensions))(*samples)
        coefficients = GraphModule(coefficients, fx_graph_cse(coefficients.graph))
        _, _, *cuts = coefficients(*samples)
        states = samples[0].new_empty((samples[1].shape[0], *samples[0].shape))
        readout = make_fx(torch.func.vmap(self.readout))(states, *cuts)
        result: torch.Tensor = readout(states, *cuts)
        storage = result.untyped_storage()
        copy = storage.nbytes() != result.numel() * result.element_size() or any(
            storage == value.untyped_storage() for value in (states, *cuts)
        )
        return ScanAnalysis(self.kind, coefficients, readout, orders, self.scale, copy)

    def matrix_orders(self, sequences: int) -> tuple[tuple[int, ...], ...]:
        """Infer packing from independent transpose chains feeding matrix batch axes."""
        inputs = [n for n in self.coefficients.graph.nodes if n.op == "placeholder"][1 : 1 + sequences]
        orders = {node: tuple(range(node.meta["val"].ndim)) for node in inputs}
        views = {node: (node, order) for node, order in orders.items()}
        for node in self.coefficients.graph.nodes:
            if (
                node.target in (torch.ops.aten.t.default, torch.ops.aten.transpose.int)
                and node.all_input_nodes[0] in views
            ):
                root, order = views[node.all_input_nodes[0]]
                axes = list(order)
                a, b = (-1, -2) if node.target == torch.ops.aten.t.default else (node.args[1], node.args[2])
                assert isinstance(a, int) and isinstance(b, int)
                axes[a], axes[b] = axes[b], axes[a]
                views[node] = root, tuple(axes)
            elif node.target in (torch.ops.aten.bmm.default, torch.ops.aten.matmul.default):
                for operand in node.all_input_nodes:
                    if operand in views:
                        root, order = views[operand]
                        if len(order) > 2:
                            orders[root] = (*order[:-2], *sorted(order[-2:]))
        return tuple(orders[node] for node in inputs)


class Unsupported(Exception):
    """The dependent slice does not close under the supported action algebra."""


class _Extract:
    def __init__(self, source: GraphModule, sample: torch.Tensor) -> None:
        self.source, self.sample = source, sample
        self.graph = Graph()
        self.independent: dict[Node, Node] = {}
        self.values: dict[Node, Affine] = {}
        self.refused: set[Node] = set()

    def call(self, target: Target, *args: Argument, **kwargs: Argument) -> Node:
        if target in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default, torch.ops.aten.matmul.default):
            a, scale_a = self.scaled(args[0])
            b, scale_b = self.scaled(args[1])
            product = self.graph.call_function(target, (a, b), kwargs)  # pyright: ignore[reportArgumentType]
            return (
                product if scale_a * scale_b == 1 else self.call(torch.ops.aten.mul.Tensor, product, scale_a * scale_b)
            )
        if target in (torch.ops.aten.t.default, torch.ops.aten.transpose.int):
            value, scale = self.scaled(args[0])
            if scale != 1:
                return self.call(torch.ops.aten.mul.Tensor, self.call(target, value, *args[1:]), scale)
        if target == torch.ops.aten.mul.Tensor and isinstance(args[1], (int, float)):
            value, scale = self.scaled(args[0])
            factor = scale * args[1]
            if isinstance(value, Node) and factor == 1:
                return value
            args = (value, factor)
        return self.graph.call_function(target, args, kwargs)  # pyright: ignore[reportArgumentType]

    @staticmethod
    def scaled(value: Argument) -> tuple[Argument, float]:
        if (
            isinstance(value, Node)
            and value.target == torch.ops.aten.mul.Tensor
            and isinstance(value.args[1], (int, float))
        ):
            return value.args[0], float(value.args[1])
        return value, 1.0

    def multiply(self, a: Coefficient, b: Coefficient) -> Coefficient:
        if isinstance(a, float) and isinstance(b, float):
            return a * b
        return self.call(torch.ops.aten.mul.Tensor, a, b)

    def transpose(self, value: Coefficient) -> Coefficient:
        return value if isinstance(value, float) else self.call(torch.ops.aten.transpose.int, value, -1, -2)

    def matrix(self, linear: Linear, side: Kind) -> Coefficient:
        if linear.kind == side:
            return linear.coefficient
        if linear.kind != "scalar":
            raise Unsupported
        axis = -1 if side == "right" else -2
        width = self.sample.shape[-3 - axis if linear.transposed else axis]
        eye = self.call(torch.ops.aten.eye.default, width, dtype=self.sample.dtype, device=self.sample.device)
        return self.multiply(eye, linear.coefficient)

    def add_maps(self, a: Linear | None, b: Linear | None, alpha: float) -> Linear | None:
        if b is None:
            return a
        if a is None:
            return Linear(b.kind, self.multiply(b.coefficient, alpha), b.transposed)
        if a.transposed != b.transposed:
            raise Unsupported
        side = b.kind if a.kind == "scalar" else a.kind
        if b.kind not in ("scalar", side):
            raise Unsupported
        aa, bb = (
            (self.matrix(value, side) for value in (a, b))
            if side in ("left", "right")
            else (a.coefficient, b.coefficient)
        )
        coefficient = (
            aa + alpha * bb
            if isinstance(aa, float) and isinstance(bb, float)
            else self.call(torch.ops.aten.add.Tensor, aa, bb, alpha=alpha)
        )
        return Linear(side, coefficient, a.transposed)

    def value(self, arg: Argument) -> Affine:
        if isinstance(arg, (int, float)):
            return Affine(
                None,
                self.call(
                    torch.ops.aten.scalar_tensor.default, arg, dtype=self.sample.dtype, device=self.sample.device
                ),
            )
        if not isinstance(arg, Node):
            raise Unsupported
        return self.values[arg]

    def extract(self, node: Node) -> Affine:
        op, args = node.target, node.args
        a = self.value(args[0])
        if op in (torch.ops.aten.add.Tensor, torch.ops.aten.sub.Tensor):
            b = self.value(args[1])
            alpha = float(node.kwargs.get("alpha", 1)) * (-1 if op == torch.ops.aten.sub.Tensor else 1)  # pyright: ignore[reportArgumentType]
            linear = self.add_maps(a.linear, b.linear, alpha)
            bias = a.bias
            if b.bias is not None:
                bias = (
                    self.call(torch.ops.aten.mul.Tensor, b.bias, alpha)
                    if bias is None
                    else self.call(torch.ops.aten.add.Tensor, bias, b.bias, alpha=alpha)
                )
            return Affine(linear, bias)
        if op in (torch.ops.aten.neg.default, torch.ops.aten.mul.Tensor, torch.ops.aten.div.Tensor):
            if op == torch.ops.aten.neg.default:
                factor: Coefficient = -1.0
                factor_shape: Coefficient = factor
            else:
                if a.linear is None and op == torch.ops.aten.mul.Tensor:
                    a = self.value(args[1])
                    factor_arg = args[0]
                else:
                    factor_arg = args[1]
                if isinstance(factor_arg, Node):
                    if self.values[factor_arg].linear is not None:
                        raise Unsupported
                    factor = self.independent[factor_arg]
                elif isinstance(factor_arg, (int, float)):
                    factor = float(factor_arg)
                else:
                    raise Unsupported
                factor_shape = factor
                if op == torch.ops.aten.div.Tensor:
                    factor = (
                        1 / factor
                        if isinstance(factor, float)
                        else self.call(torch.ops.aten.reciprocal.default, factor)
                    )
            assert a.linear is not None
            kind = a.linear.kind
            if isinstance(factor_shape, Node):
                metadata: torch.Tensor = factor_shape.meta["val"]
                if metadata.numel() != 1:
                    if kind not in ("scalar", "diagonal"):
                        raise Unsupported
                    kind = "diagonal"
            return Affine(
                Linear(kind, self.multiply(a.linear.coefficient, factor), a.linear.transposed),
                None if a.bias is None else self.call(torch.ops.aten.mul.Tensor, a.bias, factor),
            )
        if op in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default, torch.ops.aten.matmul.default):
            b = self.value(args[1])
            side: Kind = "right" if a.linear is not None else "left"
            dependent, independent = (a, b) if side == "right" else (b, a)
            if independent.linear is not None or independent.bias is None or dependent.linear is None:
                raise Unsupported
            linear = dependent.linear
            coefficient = independent.bias
            if linear.kind == "scalar":
                coefficient = self.call(torch.ops.aten.mul.Tensor, coefficient, linear.coefficient)
            elif linear.kind == side:
                operands = (linear.coefficient, coefficient) if side == "right" else (coefficient, linear.coefficient)
                coefficient = self.call(op, *operands)
            else:
                raise Unsupported
            bias = (
                None
                if dependent.bias is None
                else self.call(
                    op, *((dependent.bias, independent.bias) if side == "right" else (independent.bias, dependent.bias))
                )
            )
            return Affine(Linear(side, coefficient, linear.transposed), bias)
        if op in (
            torch.ops.aten.view.default,
            torch.ops.aten.expand.default,
            torch.ops.aten._unsafe_view.default,
            torch.ops.aten.clone.default,
            torch.ops.aten._to_copy.default,
        ):
            before: torch.Tensor = node.all_input_nodes[0].meta["val"]
            after: torch.Tensor = node.meta["val"]
            if before.shape == after.shape and before.dtype == after.dtype and before.device == after.device:
                return a
            raise Unsupported
        if op in (torch.ops.aten.t.default, torch.ops.aten.transpose.int, torch.ops.aten.permute.default):
            if op == torch.ops.aten.permute.default:
                metadata: torch.Tensor = node.all_input_nodes[0].meta["val"]
                axes = node.args[1]
                if axes == list(range(metadata.ndim)):
                    return a
                if axes != [*range(metadata.ndim - 2), metadata.ndim - 1, metadata.ndim - 2]:
                    raise Unsupported
            if op == torch.ops.aten.transpose.int:
                metadata: torch.Tensor = node.all_input_nodes[0].meta["val"]
                assert isinstance(args[1], int) and isinstance(args[2], int)
                if {args[1] % metadata.ndim, args[2] % metadata.ndim} != {
                    metadata.ndim - 1,
                    metadata.ndim - 2,
                }:  # pyright: ignore[reportArgumentType]
                    raise Unsupported
            assert a.linear is not None
            side = "right" if a.linear.kind == "left" else "left" if a.linear.kind == "right" else a.linear.kind
            linear = Linear(side, self.transpose(a.linear.coefficient), not a.linear.transposed)  # pyright: ignore[reportArgumentType]
            return Affine(linear, None if a.bias is None else self.call(op, a.bias, *args[1:]))
        raise Unsupported

    def run(self) -> ScanAnalysis | None:
        placeholders = [node for node in self.source.graph.nodes if node.op == "placeholder"]
        carry = placeholders[0]
        for node in self.source.graph.nodes:
            if node is carry:
                self.independent[node] = self.graph.placeholder(node.name)
                self.values[node] = Affine(Linear(), None)
            elif (
                node.op in ("placeholder", "get_attr")
                or node.op == "call_function"
                and all(self.values[n].linear is None for n in node.all_input_nodes)
            ):
                copied = self.graph.node_copy(node, self.independent.__getitem__)
                self.independent[node] = copied
                self.values[node] = Affine(None, copied)
            elif node.op == "call_function":
                try:
                    if any(n in self.refused for n in node.all_input_nodes):
                        raise Unsupported
                    self.values[node] = self.extract(node)
                except (Unsupported, KeyError):
                    # Unsupported pure readout nodes need not disqualify the carry.
                    self.refused.add(node)
                    self.values[node] = Affine(Linear(), None)
            elif node.op == "output":
                outputs: tuple[Argument, ...] = node.args[0]  # pyright: ignore[reportAssignmentType]
                assert isinstance(outputs, tuple)
                updated: Argument = outputs[0]
                output: Argument = outputs[1]
                assert isinstance(updated, Node) and isinstance(output, Node)
                return None if updated in self.refused else self.finish(carry, updated, output)
        raise AssertionError("FX graph has no output")

    def finish(self, carry: Node, updated: Node, output: Node) -> ScanAnalysis | None:
        value = self.values[updated]
        linear = value.linear
        metadata: torch.Tensor = updated.meta["val"]
        if (
            linear is None
            or linear.transposed
            or metadata.shape != self.sample.shape
            or metadata.dtype != self.sample.dtype
            or metadata.stride() != self.sample.stride()
        ):
            return None
        kind = (
            "additive"
            if linear.kind == "scalar" and linear.coefficient == 1.0
            else "diagonal"
            if linear.kind == "scalar"
            else linear.kind
        )
        read = Graph()
        mapping = {carry: read.placeholder("carry")}
        cuts: list[Node] = []

        def copy(node: Node) -> Node:
            if node not in mapping:
                if node in self.independent:
                    cuts.append(self.independent[node])
                    mapping[node] = read.placeholder(node.name)
                else:
                    mapping[node] = read.node_copy(node, copy)
            return mapping[node]

        read.output(copy(output))
        bias = value.bias
        if bias is None:
            bias = self.call(
                torch.ops.aten.zeros.default,
                list(self.sample.shape),
                dtype=self.sample.dtype,
                device=self.sample.device,
            )
        scale = 1.0
        if kind == "additive" and bias.target == torch.ops.aten.mul.Tensor and isinstance(bias.args[1], (int, float)):
            unscaled = bias.args[0]
            if (
                isinstance(unscaled, Node)
                and "val" in unscaled.meta
                and unscaled.meta["val"].dtype == self.sample.dtype
            ):
                bias, scale = unscaled, float(bias.args[1])
        coefficient = linear.coefficient
        if isinstance(coefficient, float):
            # Identity needs no coefficient payload or CUDA scalar fill.
            coefficient = self.call(
                torch.ops.aten.full.default,
                [0] if kind == "additive" else [],
                coefficient,
                dtype=self.sample.dtype,
                device=self.sample.device,
            )
        self.graph.output((coefficient, bias, *cuts))
        self.graph.eliminate_dead_code()
        return ScanAnalysis(kind, GraphModule(self.source, self.graph), GraphModule(self.source, read), scale=scale)  # pyright: ignore[reportArgumentType]


def analyze(step: Callable[..., tuple[torch.Tensor, torch.Tensor]], specs: Specs) -> ScanAnalysis | None:
    """Trace shape-only inputs, reject effects, and normalize the returned carry map."""
    with _disable_current_modes(), torch.inference_mode(False), FakeTensorMode(allow_non_fake_inputs=True):
        samples = tuple(
            torch.empty_strided(shape, stride, dtype=dtype, device=device) for shape, stride, dtype, device in specs
        )
        source = make_fx(lambda *args: step(*args), tracing_mode="fake")(*samples)
        if any(value.requires_grad for value in (*source.parameters(), *source.buffers())):
            return None
        for node in source.graph.nodes:
            if node.op == "call_function":
                target = node.target
                if target is not operator.getitem and (
                    not isinstance(target, OpOverload)
                    or target._schema.is_mutable  # pyright: ignore[reportPrivateUsage]
                    or torch.Tag.nondeterministic_seeded in target.tags
                ):  # pyright: ignore[reportPrivateUsage]
                    return None
        return _Extract(source, samples[0]).run()
