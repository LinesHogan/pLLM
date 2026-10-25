"""Trace user layer routes with Torch and lower their segments and tensor cells for TokenwiseRegion."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from pllm.contracts.backend_attachment import CausalModelDomain
from pllm.contracts.region_composition import inline_region
from pllm.contracts.region_program import (
    Block,
    BorrowedResource,
    Call,
    Function,
    Op,
    Program,
    Region,
    ResourceType,
    ScalarType,
    TensorType,
    Value,
)
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.codec import tensor_decompositions, tensor_node_metadata
from pllm.execution.decoder_region import DecoderActivation, DecoderSource
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_native import RegionFunction
from pllm.execution.shared_tensor import tensor_dtype

Tokens = Literal["prefill", "decode", "all"]


@torch.library.custom_op("pllm::layers", mutates_args=())
def _layers(hidden: torch.Tensor, start: int, stop: int) -> list[torch.Tensor]:
    raise RuntimeError("call Layers only inside a @pllm.layer_route method")


def _fake_layers(hidden: torch.Tensor, start: int, stop: int) -> list[torch.Tensor]:
    return [torch.empty_like(hidden) for _ in range(start, stop)]


_layers.register_fake(_fake_layers)  # pyright: ignore[reportUnknownMemberType]


@dataclass(frozen=True)
class Taps:
    """Post-layer residual streams indexed by absolute layer, negative from the model end."""

    values: tuple[torch.Tensor, ...]
    start: int
    layer_count: int

    def __getitem__(self, layer: int) -> torch.Tensor:
        layer = layer if layer >= 0 else self.layer_count + layer
        if not self.start <= layer < self.start + len(self.values):
            raise ValueError("index a tap inside the Layers slice that produced it")
        return self.values[layer - self.start]


@dataclass(frozen=True)
class Layers:
    """Run original model slices, writing the current token's KV; position stays on device."""

    seed: torch.Tensor
    position: torch.Tensor
    layer_count: int

    def __call__(self, start: int = 0, stop: int | None = None, hidden: torch.Tensor | None = None) -> Taps:
        stop = self.layer_count if stop is None else stop
        if not 0 <= start < stop <= self.layer_count or hidden is None and start != 0:
            raise ValueError("use 0 <= start < stop <= layer_count and supply hidden when start > 0")
        return Taps(tuple(_layers(self.seed if hidden is None else hidden, start, stop)), start, self.layer_count)


@dataclass(frozen=True)
class Segment:
    """One model slice and just the post-layer values consumed by the route."""

    start: int
    stop: int
    input: str
    taps: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class Cell:
    """A maximal ATen fragment between model slices, retaining its exported source."""

    source: ExportedCell


@dataclass(frozen=True)
class Output:
    """The residual stream to normalize for sampling."""

    value: str


def _tensor(node: torch.fx.Node) -> torch.Tensor:
    value = tensor_node_metadata(node)
    assert value is not None
    return value


def _cell(nodes: list[torch.fx.Node]) -> Cell:
    inputs = tuple(dict.fromkeys(arg for node in nodes for arg in node.all_input_nodes if arg not in nodes))
    outputs = tuple(node for node in nodes if any(user not in nodes for user in node.users))
    graph = torch.fx.Graph()
    mapped = {node: graph.placeholder(node.name) for node in inputs}
    for node in nodes:
        mapped[node] = graph.node_copy(node, mapped.__getitem__)
    graph.output(tuple(mapped[node] for node in outputs))
    return Cell(
        ExportedCell.from_export(
            torch.export.export(torch.fx.GraphModule({}, graph), tuple(_tensor(node) for node in inputs), strict=False),
            inputs=tuple(node.name for node in inputs),
            outputs=tuple(node.name for node in outputs),
        )
    )


@dataclass(frozen=True)
class TokenProgram:
    """Ordered model slices, exported tensor fragments and the sampled residual stream."""

    steps: tuple[Segment | Cell | Output, ...]
    seed: str
    position: str

    @classmethod
    def capture(
        cls, function: Callable[[Layers], torch.Tensor | None], tensor: TensorType, layer_count: int
    ) -> "TokenProgram":
        """Trace on fake tensors; Torch rejects data-dependent Python control flow."""

        def invoke(seed: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
            result = function(Layers(seed, position, layer_count))
            if not isinstance(result, torch.Tensor):
                raise ValueError("return a final-layer residual-stream tensor from your layer_route method")
            return result

        with FakeTensorMode():
            seed = torch.empty(
                tensor.shape, dtype=tensor_dtype(TensorDType(tensor.dtype.removeprefix("torch."))), device=tensor.device
            )
            position = torch.empty((1,), dtype=torch.int64, device=tensor.device)
            graph = make_fx(
                invoke,
                decomposition_table=tensor_decompositions((torch.ops.aten.linalg_vector_norm.default,)),
                tracing_mode="fake",
            )(seed, position)
            steps: list[Segment | Cell | Output] = []
            pending: list[torch.fx.Node] = []
            placeholders = tuple(node.name for node in graph.graph.nodes if node.op == "placeholder")
            for node in graph.graph.nodes:
                if node.target == torch.ops.pllm.layers.default or node.op == "output":
                    if pending:
                        steps.append(_cell(pending))
                        pending = []
                    if node.op == "output":
                        result = node.args[0]
                        assert isinstance(result, torch.fx.Node)
                        if _tensor(result).shape != seed.shape or _tensor(result).dtype != seed.dtype:
                            raise ValueError("return a residual stream with the model hidden shape and dtype")
                        steps.append(Output(result.name))
                    else:
                        hidden, start, stop = node.args
                        assert isinstance(hidden, torch.fx.Node) and type(start) is int and type(stop) is int
                        steps.append(
                            Segment(
                                start,
                                stop,
                                hidden.name,
                                tuple(
                                    (start + projection.args[1], projection.name)
                                    for projection in node.users
                                    if projection.users and type(projection.args[1]) is int
                                ),
                            )
                        )
                elif node.op != "placeholder" and not any(
                    arg.target == torch.ops.pllm.layers.default for arg in node.all_input_nodes
                ):
                    pending.append(node)
        return cls(tuple(steps), *placeholders)

    def lower(
        self,
        tensor: TensorType,
        domain: CausalModelDomain,
        max_rows: int,
        activations: tuple[DecoderActivation, ...],
    ) -> tuple[Program, dict[str, RegionFunction]]:
        """Compose DecoderSource slices with their exported cells in source order."""
        positions = TensorType((1,), "torch.int64", tensor.device)
        arguments = tuple(Value(name, ScalarType.INDEX) for name in ("row", "request", "offset")) + (
            Value("positions", positions),
            Value("seed", tensor),
        )
        values = {self.seed: arguments[-1], self.position: arguments[-2]}
        operations: list[Op] = []
        functions: dict[str, RegionFunction] = {}
        declarations: dict[str, Function] = {}
        weights, context = BorrowedResource("weights", (domain.layers + 1,)), BorrowedResource("context", (max_rows,))
        resources: dict[str, ResourceType] = {resource.name: resource for resource in (weights, context)}
        result = arguments[-1]
        for ordinal, step in enumerate(self.steps):
            if isinstance(step, Segment):
                source = DecoderSource(
                    tensor,
                    tuple(range(step.start, step.stop)),
                    domain,
                    taps=tuple(layer for layer, _ in step.taps),
                    weights=weights,
                    context=context,
                    activations=activations,
                )
                segment = inline_region(source.program, (*arguments[:4], values[step.input]), f"segment{ordinal}")
                operations.extend(segment.operations)
                values.update((name, value) for (_, name), value in zip(step.taps, segment.results[2:], strict=True))
                functions.update(source.functions)
                declarations.update((function.name, function) for function in source.program.functions)
                resources.update((resource.name, resource) for resource in source.program.resources)
            elif isinstance(step, Cell):
                cell = step.source
                results = tuple(
                    Value(name, type_) for name, type_ in zip(cell.outputs, cell.declaration.outputs, strict=True)
                )
                operations.append(Call(tuple(values[name].name for name in cell.inputs), results, cell.digest))
                values.update((value.name, value) for value in results)
                functions[cell.digest] = cell
                declarations[cell.digest] = cell.declaration
            else:
                result = values[step.value]
        return Program(
            Region(Block(arguments, tuple(operations), (result.name,))),
            tuple(declarations.values()),
            tuple(resources.values()),
        ), functions
