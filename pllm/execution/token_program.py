"""Trace @pllm.custom_forward methods into model slices and functional tensor cells for direct execution."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from operator import getitem
from typing import Literal

import torch
from torch._dynamo.source import LocalSource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from pllm.execution.model_sites import StateInput
from pllm.execution.module_graph import ModuleCall
from pllm.execution.position_state import position_save
from pllm.execution.torch_private import check_functional_aten, tensor_decompositions, tensor_node_metadata

Tokens = Literal["prefill", "decode", "all"]
Over = Literal["token", "span"]
KV = Literal["commit", "peek"]


@dataclass(frozen=True)
class TokenTensor:
    """Static seed metadata for custom-forward tracing without allocating model storage."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    device: str


@torch.library.custom_op("pllm::layers", mutates_args=())
def _layers(hidden: torch.Tensor, start: int, stop: int, kv: str, active: torch.Tensor | None) -> list[torch.Tensor]:
    raise RuntimeError("call Layers only inside a @pllm.custom_forward method")


def _fake_layers(
    hidden: torch.Tensor, start: int, stop: int, kv: str, active: torch.Tensor | None
) -> list[torch.Tensor]:
    return [torch.empty_like(hidden) for _ in range(start, stop)]


_layers.register_fake(_fake_layers)  # pyright: ignore[reportUnknownMemberType]


@torch.library.custom_op("pllm::apply_updates", mutates_args=())
def _apply_updates(hidden: torch.Tensor) -> None:
    raise RuntimeError("call apply_updates only inside a @pllm.custom_forward method")


def _fake_updates(hidden: torch.Tensor) -> None:
    pass


_apply_updates.register_fake(_fake_updates)  # pyright: ignore[reportUnknownMemberType]


@dataclass(frozen=True)
class LayerOutputs:
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
    """Public model slices and position state used inside custom_forward methods."""

    # The hidden entering layer 0: the token embeddings.
    embeddings: torch.Tensor
    position: torch.Tensor
    layer_count: int
    generated: torch.Tensor
    entries: dict[str, torch.Tensor] = field(repr=False)
    over: Over = "token"

    def load(self, name: str, offset: int = -1) -> torch.Tensor:
        """Read the previous token's state, even after a save in this token."""
        if self.over == "span" or offset != -1 or name not in self.entries:
            raise ValueError(
                f'use layers.load({name!r}, offset=-1) in over="token" custom forwards with a declared PositionState {name!r} '
                "(spans have no serial predecessor)"
            )
        return self.entries[name].clone()

    def save(self, name: str, value: torch.Tensor) -> None:
        """Save a snapshot at this position; the last save wins."""
        entry = self.entries.get(name)
        if entry is None or value.shape != entry.shape or value.dtype != entry.dtype or value.device != entry.device:
            expected = "" if entry is None else f" with shape {tuple(entry.shape)} and dtype {entry.dtype}"
            raise ValueError(f"declare PositionState {name!r} and save {name}{expected}")
        position_save(value.clone(), name)

    def __call__(
        self,
        start: int = 0,
        stop: int | None = None,
        hidden: torch.Tensor | None = None,
        *,
        kv: KV = "commit",
        active: torch.Tensor | None = None,
    ) -> LayerOutputs:
        """Commit causal K/V, or peek at earlier committed positions plus fresh own K/V.

        A peek leaves the cache untouched and requires an earlier commit to every
        layer in this slice within the same custom forward.
        active is a bool [rows, 1] tensor in token mode. False rows keep hidden
        at every layer. On a first commit their K/V is propagated through each
        skipped layer's own norm, QKV and RoPE; later skips retain the last commit.
        Peek never writes, including skipped rows.
        """
        if active is not None:
            if self.over != "token":
                raise ValueError('active= needs over="token": a span method sees the whole prompt chunk')
            if (
                active.dtype != torch.bool
                or active.shape != self.embeddings[:, :1].shape
                or active.device != self.embeddings.device
            ):
                raise ValueError("pass active as a bool tensor [rows, 1] computed in this method")
        if kv not in ("commit", "peek"):
            raise ValueError(
                "skipped tokens cannot be left out of a layer's attention: skipped rows get K/V propagated; pass kv='commit' or 'peek'"
            )
        stop = self.layer_count if stop is None else stop
        value = self.embeddings if hidden is None else hidden
        if (
            not 0 <= start < stop <= self.layer_count
            or (hidden is None and start != 0)
            or value.shape != self.embeddings.shape
            or value.dtype != self.embeddings.dtype
            or value.device != self.embeddings.device
        ):
            raise ValueError(
                "use 0 <= start < stop <= layer_count, kv in ('commit', 'peek'), "
                "and supply a hidden tensor with the model shape, dtype and device when start > 0"
            )
        return LayerOutputs(tuple(_layers(value, start, stop, kv, active)), start, self.layer_count)

    def apply_updates(self) -> None:
        """Apply operands captured since the previous boundary to this token's private overlay."""
        _apply_updates(self.embeddings)


@dataclass(frozen=True)
class Segment:
    """One model slice and just the post-layer values consumed by the program."""

    start: int
    stop: int
    input: str
    taps: tuple[tuple[int, str], ...]
    kv: KV = "commit"
    active: str | None = None
    propagate: tuple[int, ...] = ()


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


def _cell(nodes: list[torch.fx.Node], over: Over, invariants: tuple[str, ...]) -> Cell:
    # Recompute symbolic sizes in each fragment so program boundaries carry tensors only.
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
        assert node.op == "call_function"
        # Module parameters are unpacked by split followed by tuple indexing.
        if node.target is not getitem:
            check_functional_aten(node.target)
    module = torch.fx.GraphModule({}, graph)
    input_rows = tuple(
        node.name not in invariants and bool(_tensor(node).ndim and _tensor(node).shape[0] == 1) for node in inputs
    )
    output_rows = tuple(bool(_tensor(node).ndim and _tensor(node).shape[0] == 1) for node in outputs)

    def per_row(_seed: torch.Tensor, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return module(*args)

    mapped_function = torch.vmap(per_row, in_dims=(0, *(None if node.name in invariants else 0 for node in inputs)))

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
    parameters: tuple[tuple[str, StateInput], ...] = ()
    prompt_pass_is_identity: bool = False
    over: Over = "token"

    @classmethod
    def trace(
        cls,
        function: Callable[..., torch.Tensor | None],
        tensor: TokenTensor,
        layer_count: int,
        names: tuple[str, ...] = (),
        over: Over = "token",
        parameters: tuple[tuple[str, StateInput, torch.dtype], ...] = (),
    ) -> "TokenProgram":
        """Trace on fake tensors; Torch rejects data-dependent Python control flow."""

        def invoke(
            seed: torch.Tensor, position: torch.Tensor, generated: torch.Tensor, *entries: torch.Tensor
        ) -> torch.Tensor:
            result = function(
                Layers(
                    seed, position, layer_count, generated, dict(zip(names, entries[: len(names)], strict=True)), over
                ),
                **{
                    name: value if source.module is None else ModuleCall(source.module, value, False)
                    for (name, source, _), value in zip(parameters, entries[len(names) :], strict=True)
                },
            )
            if (
                not isinstance(result, torch.Tensor)
                or result.shape != seed.shape
                or result.dtype != seed.dtype
                or result.device != seed.device
            ):
                raise ValueError("return a residual-stream tensor with the model hidden shape, dtype and device")
            return result

        shapes = ShapeEnv()
        rows = (
            shapes.create_symintnode(shapes.create_symbol(2, LocalSource("rows")), hint=2, source=LocalSource("rows"))
            if over == "span"
            else tensor.shape[0]
        )
        with FakeTensorMode(shape_env=shapes):
            shape = (rows, *tensor.shape[1:])
            seed = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
            position = torch.empty((shape[0], 1), dtype=torch.int64, device=tensor.device)
            generated = torch.empty((shape[0], 1), dtype=torch.bool, device=tensor.device)
            arguments = (
                seed,
                position,
                generated,
                *(torch.empty_like(seed) for _ in names),
                *(torch.empty(source.view.shape, dtype=dtype, device=tensor.device) for _, source, dtype in parameters),
            )
            graph = make_fx(
                invoke,
                decomposition_table=tensor_decompositions((torch.ops.aten.linalg_vector_norm.default,)),
                tracing_mode="fake",
            )(*arguments)
            # Composite ATen decompositions (notably cosine_similarity) can introduce
            # mutations after dispatch functionalization, so functionalize their graph.
            graph = make_fx(torch.func.functionalize(graph), tracing_mode="fake")(*arguments)
            steps: list[Segment | Cell | ApplyUpdates | Save | Output] = []
            pending: list[torch.fx.Node] = []
            committed: set[int] = set()
            placeholders = tuple(node.name for node in graph.graph.nodes if node.op == "placeholder")
            invariants = placeholders[3 + len(names) :]
            for node in graph.graph.nodes:
                if node.target == torch.ops.higher_order.while_loop:
                    raise ValueError("@pllm.custom_forward does not support traced while_loop; use a bounded for-loop")
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
                        steps.append(_cell(pending, over, invariants))
                        pending = []
                    if node.target == torch.ops.pllm.position_save.default:
                        steps.append(Save(str(node.args[1]), node.all_input_nodes[0].name))
                    elif node.target == torch.ops.pllm.apply_updates.default:
                        steps.append(ApplyUpdates())
                    elif node.op == "output":
                        steps.append(Output(node.all_input_nodes[0].name))
                    else:
                        _, start, stop, kv, active = node.args
                        assert type(start) is int and type(stop) is int and kv in ("commit", "peek")
                        if kv == "peek" and not set(range(start, stop)) <= committed:
                            raise ValueError("commit every layer in a slice before calling it with kv=peek")
                        propagate = tuple(
                            layer for layer in range(start, stop) if kv == "commit" and layer not in committed
                        )
                        if kv == "commit":
                            committed.update(range(start, stop))
                        steps.append(
                            Segment(
                                start,
                                stop,
                                node.all_input_nodes[0].name,
                                tuple(
                                    (start + projection.args[1], projection.name)
                                    for projection in node.users
                                    if projection.users and type(projection.args[1]) is int
                                ),
                                kv,
                                active.name if isinstance(active, torch.fx.Node) else None,
                                propagate if active is not None else (),
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
            tuple(zip(names, placeholders[3 : 3 + len(names)], strict=True)),
            parameters=tuple(
                (key, replace(source, module=None)) for key, (_, source, _) in zip(invariants, parameters, strict=True)
            ),
            over=over,
        )

        return replace(program, prompt_pass_is_identity=program._prompt_identity(layer_count))

    @property
    def adaptive(self) -> bool:
        """Whether any slice selects active rows on the device."""
        return any(isinstance(step, Segment) and step.active is not None for step in self.steps)

    @property
    def role_sensitive(self) -> bool:
        """Generated roles and predecessor storage affect prefix-cache identity."""
        for step in self.steps:
            match step:
                case Cell(inputs=inputs) if self.generated in inputs:
                    return True
                case _:
                    pass
        return bool(self.entries)

    def _prompt_identity(self, layer_count: int) -> bool:
        """Prove the prompt branch is one complete decoder pass with final-tap saves."""
        aliases: dict[str, str | bool] = {self.generated: False}
        segments: list[Segment] = []
        saved: list[str | bool] = []
        result: str | bool = ""
        for step in self.steps:
            match step:
                case Cell():
                    values: dict[torch.fx.Node, str | bool] = {}
                    for node in step.graph.graph.nodes:
                        if node.op == "placeholder":
                            values[node] = aliases[node.name] if node.name in aliases else node.name
                        elif node.target in (torch.ops.aten.clone.default, torch.ops.aten.detach.default):
                            values[node] = values[node.all_input_nodes[0]]
                        elif node.target == torch.ops.aten.where.self and values[node.all_input_nodes[0]] is False:
                            values[node] = values[node.all_input_nodes[2]]
                        elif node.op == "output":
                            aliases.update(
                                zip(step.outputs, (values[arg] for arg in node.all_input_nodes), strict=True)
                            )
                        else:
                            values[node] = node.name
                case Segment():
                    if step.active is not None or aliases.get(step.input, step.input) != self.seed or step.start != 0:
                        return False
                    segments.append(step)
                case Save():
                    saved.append(aliases.get(step.value, step.value))
                case Output():
                    result = aliases.get(step.value, step.value)
                case ApplyUpdates():
                    return False
        if len(segments) != 1 or segments[0].stop != layer_count:
            return False
        final = dict(segments[0].taps).get(segments[0].stop - 1)
        return result == final and all(value == final for value in saved)
