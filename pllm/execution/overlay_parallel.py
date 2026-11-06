"""Prove down-overlay updates commute with input-feature slicing before backend installation.

Authoring calls this with global and rank-local fake tensors. Only a conservative
set of ATen operations is admitted; execution adds no checks or collectives.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import prod

import torch
from torch._ops import OpOverload
from torch.fx import Node
from torch.fx.experimental.proxy_tensor import make_fx

from pllm.contracts.parameter_overlay import ProjectionDelta
from pllm.execution.binding_base import Field
from pllm.execution.declaration import Dimensions
from pllm.execution.tensor_frame import UpdateOutput


def _axis(whole: torch.Tensor, local: torch.Tensor, size: int) -> int | None:
    axes = tuple(i for i, (a, b) in enumerate(zip(whole.shape, local.shape, strict=True)) if a != b)
    if not axes:
        return None
    if len(axes) != 1 or whole.shape[axes[0]] != local.shape[axes[0]] * size:
        raise ValueError("TP down updates must preserve one contiguous input-feature shard")
    return axes[0]


def _reject(operation: str) -> ValueError:
    return ValueError(
        f"TP down update is not shard-local at {operation}; its update/gradient needs other ranks' shards "
        "or an unproved operation. Use a shard-local update or tensor_parallel_size=1"
    )


def _live_nodes(nodes: tuple[Node, ...]) -> tuple[Node, ...]:
    live = {node for node in nodes if node.op == "output" or node.is_impure()}
    for node in reversed(nodes):
        if node in live and not str(node.target).startswith(
            ("aten.new_", "aten.zeros_like", "aten.ones_like", "aten.full_like")
        ):
            live.update(node.all_input_nodes)
    return tuple(node for node in nodes if node in live)


def _check_graphs(whole: tuple[Node, ...], local: tuple[Node, ...], size: int) -> None:
    if len(whole) != len(local):
        raise _reject("shape-dependent control flow")
    axes: dict[Node, int | None] = {}
    structural = {"t", "transpose", "permute", "unsqueeze", "squeeze", "detach", "clone", "alias", "_to_copy"}
    for a, b in zip(whole, local, strict=True):
        label = str(a.target)
        op = a.target
        if a.op != b.op or a.target != b.target:
            raise _reject(label)
        if isinstance(op, OpOverload) and torch.Tag.nondeterministic_seeded in op.tags:
            raise _reject("random updates; use deterministic shard-local math")
        value: torch.Tensor | tuple[torch.Tensor, ...] | None = a.meta.get("val")
        part: torch.Tensor | tuple[torch.Tensor, ...] | None = b.meta.get("val")
        if not isinstance(value, torch.Tensor) or not isinstance(part, torch.Tensor):
            continue
        axis = _axis(value, part, size)
        axes[a] = axis
        if a.op != "call_function":
            continue
        inputs = tuple(node for node in a.all_input_nodes if axes.get(node) is not None)
        if not inputs and axis is None:
            if str(a.args) != str(b.args) or str(a.kwargs) != str(b.kwargs):
                raise _reject(label)
            continue
        if not isinstance(op, OpOverload):
            raise _reject(label)
        name = label.split(".")[1]
        if axis is None:
            raise _reject(label)
        if name in ("view", "_unsafe_view", "reshape"):
            source = inputs[0] if inputs else None
            if source is None:
                raise _reject(label)
            tensor: torch.Tensor = source.meta["val"]
            source_axis = axes[source]
            assert source_axis is not None
            if prod(tensor.shape[:source_axis]) != prod(value.shape[:axis]):
                raise _reject(label)
        elif (
            name in structural
            or name in ("mm", "bmm", "addmm", "sum", "mean", "slice", "copy_", "expand")
            or torch.Tag.pointwise in op.tags
        ):
            if str(a.args) != str(b.args) or str(a.kwargs) != str(b.kwargs):
                raise _reject(label)
            if name == "slice" and inputs and a.args[1] == axes[inputs[0]]:
                if a.args[2:] != (0, 9223372036854775807):
                    raise _reject(label)
        elif name not in ("zeros", "zeros_like", "ones", "ones_like", "full", "full_like"):
            raise _reject(label)


@dataclass(frozen=True)
class _Update:
    function: Callable[..., UpdateOutput | tuple[torch.Tensor, torch.Tensor] | torch.Tensor]
    arguments: Mapping[str, str]
    fields: tuple[str, ...]
    applied_fields: tuple[str, ...]
    gradient: bool
    chunked: bool

    def __call__(self, *values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        bound = dict(zip(self.arguments, values, strict=True))
        if self.gradient:
            parameters = {key: bound[key] for key, source in self.arguments.items() if source in self.applied_fields}
            constants = {key: value for key, value in bound.items() if key not in parameters}

            def loss(parameters: dict[str, torch.Tensor]) -> torch.Tensor:
                result = self.function(**parameters, **constants)
                assert isinstance(result, torch.Tensor)
                return result.sum()

            return tuple(torch.func.grad(loss)(parameters).values())
        result = self.function(**bound)
        if self.chunked:
            assert isinstance(result, tuple)
            return (result[0],)
        assert isinstance(result, UpdateOutput)
        return (
            *(bound[key] for key, source in self.arguments.items() if source in self.fields),
            result.changed,
            result.update_count,
            result.trained_rows,
        )


def admit_update(
    function: Callable[..., UpdateOutput | tuple[torch.Tensor, torch.Tensor] | torch.Tensor],
    fields: Mapping[str, Field],
    effect: ProjectionDelta,
    dimensions: Dimensions,
    arguments: Mapping[str, str],
    invariants: Mapping[str, tuple[torch.Tensor, ...]],
    *,
    gradient: bool,
    chunked: bool,
) -> None:
    """Reject an update whose fields or gradients depend on another input-feature shard.

    Trace just the updated fields, pruning the projection readout. In particular a
    reconstruction error contracts the input shard before updating either factor,
    while a target/input outer product preserves the input-feature axis.
    """
    if dimensions.tensor_parallel_size == 1:
        return
    update = _Update(function, arguments, tuple(fields), effect.applied_fields, gradient, chunked)
    for index, (inputs, outputs) in enumerate(dimensions.target_shapes):
        samples: list[tuple[torch.Tensor, ...]] = []
        for size in (1, dimensions.tensor_parallel_size):
            widths = {
                "input": (3, inputs // size),
                "target": (3, outputs),
                "weight": (outputs, inputs // size),
                "embedding": (3, dimensions.hidden_size),
                "prompt_end": (),
            }
            values = {
                key: torch.empty(
                    field.size(inputs // size, outputs),
                    device="meta",
                    dtype=dimensions.dtype.torch if field.dtype in (None, "auto") else field.dtype,
                )
                for key, field in fields.items()
            }
            values.update(
                {key: torch.empty(shape, dtype=dimensions.dtype.torch, device="meta") for key, shape in widths.items()}
            )
            values.update({key: tensors[index] for key, tensors in invariants.items()})
            samples.append(tuple(values[source] for source in arguments.values()))

        graphs = [make_fx(update, tracing_mode="fake")(*sample).graph for sample in samples]
        for graph in graphs:
            graph.eliminate_dead_code()
        _check_graphs(
            _live_nodes(tuple(graphs[0].nodes)), _live_nodes(tuple(graphs[1].nodes)), dimensions.tensor_parallel_size
        )
