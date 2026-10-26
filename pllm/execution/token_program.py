"""Trace layer routes into model slices and functional tensor cells for direct execution."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal

import torch
from torch._dynamo.source import LocalSource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from pllm.contracts.region_program import TensorType
from pllm.execution.codec import check_functional_aten, tensor_decompositions, tensor_node_metadata
from pllm.execution.position_state import position_save
from pllm.execution.shared_tensor import region_dtype

Tokens = Literal["prefill", "decode", "all"]
Over = Literal["token", "span"]
KV = Literal["commit", "peek"]


@torch.library.custom_op("pllm::layers", mutates_args=())
def _layers(hidden: torch.Tensor, start: int, stop: int, kv: str) -> list[torch.Tensor]:
    raise RuntimeError("call Layers only inside a @pllm.layer_route method")


def _fake_layers(hidden: torch.Tensor, start: int, stop: int, kv: str) -> list[torch.Tensor]:
    return [torch.empty_like(hidden) for _ in range(start, stop)]


_layers.register_fake(_fake_layers)  # pyright: ignore[reportUnknownMemberType]


@torch.library.custom_op("pllm::apply_updates", mutates_args=())
def _apply_updates(hidden: torch.Tensor) -> None:
    raise RuntimeError("call apply_updates only inside a @pllm.layer_route method")


def _fake_updates(hidden: torch.Tensor) -> None:
    pass


_apply_updates.register_fake(_fake_updates)  # pyright: ignore[reportUnknownMemberType]


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
    """Public model slices and position state used inside layer_route methods."""

    seed: torch.Tensor
    position: torch.Tensor
    layer_count: int
    generated: torch.Tensor
    entries: dict[str, torch.Tensor] = field(repr=False)
    over: Over = "token"

    def load(self, name: str, offset: int = -1) -> torch.Tensor:
        """Read the previous token's state, even after a save in this token."""
        if self.over == "span":
            raise ValueError("use layers.load only in over=token routes; spans have no serial predecessor")
        if offset != -1:
            raise ValueError("use offset=-1 for PositionState")
        if name not in self.entries:
            raise ValueError(f"declare PositionState {name!r} before loading or saving it")
        return self.entries[name].clone()

    def save(self, name: str, value: torch.Tensor) -> None:
        """Save a snapshot at this position; the last save wins."""
        if name not in self.entries:
            raise ValueError(f"declare PositionState {name!r} before saving it")
        entry = self.entries[name]
        if value.shape != entry.shape or value.dtype != entry.dtype or value.device != entry.device:
            raise ValueError(f"save {name} with shape {tuple(entry.shape)} and dtype {entry.dtype}")
        position_save(value.clone(), name)

    def __call__(
        self, start: int = 0, stop: int | None = None, hidden: torch.Tensor | None = None, *, kv: KV = "commit"
    ) -> Taps:
        """Commit causal K/V, or peek at earlier committed positions plus fresh own K/V.

        A peek leaves the cache untouched and requires an earlier commit to every
        layer in this slice within the same route.
        """
        if kv not in ("commit", "peek"):
            raise ValueError("set layers(kv=) to commit or peek")
        stop = self.layer_count if stop is None else stop
        if not 0 <= start < stop <= self.layer_count or hidden is None and start != 0:
            raise ValueError("use 0 <= start < stop <= layer_count and supply hidden when start > 0")
        value = self.seed if hidden is None else hidden
        if value.shape != self.seed.shape or value.dtype != self.seed.dtype or value.device != self.seed.device:
            raise ValueError("supply a hidden tensor with the model shape, dtype and device")
        return Taps(tuple(_layers(value, start, stop, kv)), start, self.layer_count)

    def apply_updates(self) -> None:
        """Apply operands captured since the previous boundary to this token's private overlay."""
        _apply_updates(self.seed)


@dataclass(frozen=True)
class Segment:
    """One model slice and just the post-layer values consumed by the route."""

    start: int
    stop: int
    input: str
    taps: tuple[tuple[int, str], ...]
    kv: KV = "commit"


@dataclass(frozen=True)
class Cell:
    """A functional ATen fragment whose row batching preserves per-position tensor math."""

    graph: torch.fx.GraphModule
    function: Callable[..., tuple[torch.Tensor, ...]]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class ApplyUpdates:
    """An explicit ordered transition boundary with no tensor return."""


@dataclass(frozen=True)
class Output:
    """The residual stream to normalize for sampling."""

    value: str


@dataclass(frozen=True)
class Save:
    """An SSA snapshot selected for the current position's final store."""

    name: str
    value: str


def _tensor(node: torch.fx.Node) -> torch.Tensor:
    value = tensor_node_metadata(node)
    assert value is not None
    return value


def _cell(nodes: list[torch.fx.Node], over: Over) -> Cell:
    # Recompute symbolic sizes in each fragment so route boundaries carry tensors only.
    needed = set(nodes)
    for node in reversed(tuple(nodes[0].graph.nodes)):
        if node in needed:
            needed.update(
                arg for arg in node.all_input_nodes if arg.op == "call_function" and tensor_node_metadata(arg) is None
            )
    nodes = [node for node in nodes[0].graph.nodes if node in needed]
    inputs = tuple(dict.fromkeys(arg for node in nodes for arg in node.all_input_nodes if arg not in nodes))
    outputs = tuple(
        node
        for node in nodes
        if tensor_node_metadata(node) is not None and any(user not in nodes for user in node.users)
    )
    graph = torch.fx.Graph()
    mapped = {node: graph.placeholder(node.name) for node in inputs}
    for node in nodes:
        mapped[node] = graph.node_copy(node, mapped.__getitem__)
    graph.output(tuple(mapped[node] for node in outputs))
    for node in nodes:
        if node.op != "call_function":
            raise ValueError("layer route cells need functional ATen operations with explicit tensor inputs")
        check_functional_aten(node.target)
    module = torch.fx.GraphModule({}, graph)
    input_rows = tuple(bool(_tensor(node).ndim and _tensor(node).shape[0] == 1) for node in inputs)
    output_rows = tuple(bool(_tensor(node).ndim and _tensor(node).shape[0] == 1) for node in outputs)

    def per_row(_seed: torch.Tensor, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return module(*args)

    mapped_function = torch.vmap(per_row)

    def batched(seed: torch.Tensor, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # vmap preserves the single-position meaning of reductions and reshapes;
        # the model segments still call native [B, H] layers once per wave.
        results = mapped_function(
            seed, *(value.unsqueeze(1) if row else value for value, row in zip(args, input_rows, strict=True))
        )
        return tuple(value.squeeze(1) if row else value for value, row in zip(results, output_rows, strict=True))

    return Cell(
        module,
        per_row if over == "span" else batched,
        tuple(node.name for node in inputs),
        tuple(node.name for node in outputs),
    )


@dataclass(frozen=True)
class TokenProgram:
    """Ordered model slices, functional tensor fragments and the sampled residual stream."""

    steps: tuple[Segment | Cell | ApplyUpdates | Save | Output, ...]
    seed: str
    position: str
    generated: str
    entries: tuple[tuple[str, str], ...]
    prompt_pass_is_identity: bool = False
    over: Over = "token"

    @classmethod
    def capture(
        cls,
        function: Callable[[Layers], torch.Tensor | None],
        tensor: TensorType,
        layer_count: int,
        names: tuple[str, ...] = (),
        over: Over = "token",
    ) -> "TokenProgram":
        """Trace on fake tensors; Torch rejects data-dependent Python control flow."""

        def invoke(
            seed: torch.Tensor, position: torch.Tensor, generated: torch.Tensor, *entries: torch.Tensor
        ) -> torch.Tensor:
            result = function(
                Layers(seed, position, layer_count, generated, dict(zip(names, entries, strict=True)), over)
            )
            if not isinstance(result, torch.Tensor):
                raise ValueError("return a final-layer residual-stream tensor from your layer_route method")
            return result

        shapes = ShapeEnv()
        rows = (
            shapes.create_symintnode(shapes.create_symbol(2, LocalSource("rows")), hint=2, source=LocalSource("rows"))
            if over == "span"
            else tensor.shape[0]
        )
        with FakeTensorMode(shape_env=shapes):
            shape = (rows, *tensor.shape[1:])
            seed = torch.empty(shape, dtype=region_dtype(tensor.dtype), device=tensor.device)
            position = torch.empty((shape[0], 1), dtype=torch.int64, device=tensor.device)
            generated = torch.empty((shape[0], 1), dtype=torch.bool, device=tensor.device)
            graph = make_fx(
                torch.func.functionalize(invoke),
                decomposition_table=tensor_decompositions((torch.ops.aten.linalg_vector_norm.default,)),
                tracing_mode="fake",
            )(seed, position, generated, *(torch.empty_like(seed) for _ in names))
            steps: list[Segment | Cell | ApplyUpdates | Save | Output] = []
            pending: list[torch.fx.Node] = []
            committed: set[int] = set()
            placeholders = tuple(node.name for node in graph.graph.nodes if node.op == "placeholder")
            for node in graph.graph.nodes:
                if node.target == torch.ops.higher_order.while_loop:
                    raise ValueError("layer routes do not support traced while_loop; use a bounded for-loop")
                if (
                    node.target
                    in (
                        torch.ops.pllm.layers.default,
                        torch.ops.pllm.apply_updates.default,
                        torch.ops.pllm.position_save.default,
                    )
                    or node.op == "output"
                ):
                    if pending:
                        steps.append(_cell(pending, over))
                        pending = []
                    if node.target == torch.ops.pllm.position_save.default:
                        value, name = node.args
                        assert isinstance(value, torch.fx.Node) and isinstance(name, str)
                        steps.append(Save(name, value.name))
                    elif node.target == torch.ops.pllm.apply_updates.default:
                        steps.append(ApplyUpdates())
                    elif node.op == "output":
                        result = node.args[0]
                        assert isinstance(result, torch.fx.Node)
                        if (
                            _tensor(result).shape != seed.shape
                            or _tensor(result).dtype != seed.dtype
                            or _tensor(result).device != seed.device
                        ):
                            raise ValueError("return a residual stream with the model hidden shape and dtype")
                        steps.append(Output(result.name))
                    else:
                        hidden, start, stop, kv = node.args
                        assert isinstance(hidden, torch.fx.Node) and type(start) is int and type(stop) is int
                        assert kv in ("commit", "peek")
                        if kv == "peek" and not set(range(start, stop)) <= committed:
                            raise ValueError("commit every layer in a slice before calling it with kv=peek")
                        if kv == "commit":
                            committed.update(range(start, stop))
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
                                kv,
                            )
                        )
                elif node.op != "placeholder" and not any(
                    arg.target == torch.ops.pllm.layers.default for arg in node.all_input_nodes
                ):
                    pending.append(node)
        program = cls(
            tuple(steps),
            placeholders[0],
            placeholders[1],
            placeholders[2],
            tuple(zip(names, placeholders[3:], strict=True)),
            over=over,
        )

        return replace(program, prompt_pass_is_identity=program._prompt_identity(layer_count))

    @property
    def role_sensitive(self) -> bool:
        """Generated roles and predecessor storage affect prefix-cache identity."""
        return bool(self.entries) or any(
            isinstance(step, Cell) and self.generated in step.inputs for step in self.steps
        )

    def _prompt_identity(self, layer_count: int) -> bool:
        """Prove the prompt branch is one complete decoder pass with final-tap saves."""
        aliases: dict[str, str | bool] = {self.generated: False}
        segments: list[Segment] = []
        saved: list[str | bool] = []
        result: str | bool = ""
        for step in self.steps:
            if isinstance(step, Cell):
                values: dict[torch.fx.Node, str | bool] = {}
                for node in step.graph.graph.nodes:
                    if node.op == "placeholder":
                        values[node] = aliases[node.name] if node.name in aliases else node.name
                    elif node.target in (torch.ops.aten.clone.default, torch.ops.aten.detach.default):
                        values[node] = values[node.all_input_nodes[0]]
                    elif node.target == torch.ops.aten.where.self and values[node.all_input_nodes[0]] is False:
                        values[node] = values[node.all_input_nodes[2]]
                    elif node.op == "output":
                        aliases.update(zip(step.outputs, (values[arg] for arg in node.all_input_nodes), strict=True))
                    else:
                        values[node] = node.name
            elif isinstance(step, Segment):
                if aliases.get(step.input, step.input) != self.seed or step.start != 0:
                    return False
                segments.append(step)
            elif isinstance(step, Save):
                saved.append(aliases.get(step.value, step.value))
            elif isinstance(step, Output):
                result = aliases.get(step.value, step.value)
            else:
                return False
        if len(segments) != 1 or segments[0].stop != layer_count:
            return False
        final = dict(segments[0].taps).get(segments[0].stop - 1)
        return result == final and all(value == final for value in saved)
