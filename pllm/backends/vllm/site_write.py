"""Expose terminal adds for vLLM residual norms and in-place value writes.

ModelSites uses these eager FX bodies only to remove the terminal add. No compiler
backend or tensor arithmetic is changed; callers own the output ports.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, fx
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.proxy import TraceError

from pllm.execution.model_sites import PatchInput, Place, ResidualRewrite, SiteValues
from pllm.execution.module_graph import ModuleCall
from pllm.execution.torch_private import tensor_node_metadata

_Argument = Tensor | ModuleCall | int | Callable[[Tensor], Tensor]


class SiteWrite:
    """One sublayer's shape-specialized write, with a removable same-dtype terminal add."""

    def __init__(self, rewrite: ResidualRewrite, place: Place, values: SiteValues, samples: tuple[Tensor, ...]) -> None:
        self.rewrite, self.place, self.values = rewrite, place, values
        self._bodies: dict[tuple[tuple[int, ...], ...], Callable[..., tuple[Tensor, ...]] | None] = {}
        self.identity = False
        self._trace(samples)

    def _arguments(self) -> tuple[_Argument, ...]:
        depth = self.rewrite.depth(self.place)
        return tuple(source.value(self.values, depth) for _, source in self.rewrite.inputs)

    def _call(self, tensors: tuple[Tensor, ...], arguments: tuple[_Argument, ...]) -> Tensor:
        return self.rewrite.write(
            *tensors, **dict(zip((name for name, _ in self.rewrite.inputs), arguments, strict=True))
        )

    def _trace(self, samples: tuple[Tensor, ...]) -> None:
        key = self._key(samples)
        self._bodies[key] = None
        if any(isinstance(source, PatchInput) for _, source in self.rewrite.inputs):
            return
        arguments = self._arguments()
        # Modules and patch callables carry their own tensors; do not lift those
        # into graph constants. Their ordinary write still uses the output ports.
        if any(not isinstance(value, (Tensor, int)) for value in arguments):
            return
        body = make_fx(
            lambda *args: self._call(args[: len(samples)], args[len(samples) :]),
            tracing_mode="fake",
            _allow_non_fake_inputs=True,
        )(*samples, *arguments)
        output = next(node for node in body.graph.nodes if node.op == "output")
        add = output.args[0]
        if len(samples) == 1 and isinstance(add, fx.Node) and self._conversion(add, samples[0]):
            source = add.args[0]
            assert isinstance(source, fx.Node)
            self._store(key, body, output, add, (source,))
            return
        inputs = tuple(node for node in body.graph.nodes if node.op == "placeholder")
        if len(samples) == 1 and add is inputs[0] and len(tuple(body.graph.nodes)) == len(inputs) + 1:
            self.identity = self._shape_independent(len(samples), arguments)
            return
        if not isinstance(add, fx.Node) or add.target != torch.ops.aten.add.Tensor or add.kwargs:
            return
        left, right = add.args
        if not isinstance(left, fx.Node) or not isinstance(right, fx.Node):
            return
        tensors = tuple(tensor_node_metadata(node) for node in (left, right, add))
        if any(
            value is None or value.shape != samples[0].shape or value.dtype != samples[0].dtype for value in tensors
        ):
            return
        self.identity = (
            len(samples) == 2
            and {left, right} == set(inputs[:2])
            and len(tuple(body.graph.nodes)) == len(inputs) + 2
            and self._shape_independent(len(samples), arguments)
        )
        self._store(key, body, output, add, (left, right))

    @staticmethod
    def _conversion(node: fx.Node, sample: Tensor) -> bool:
        value = tensor_node_metadata(node)
        return (
            node.target == torch.ops.aten._to_copy.default
            and value is not None
            and value.shape == sample.shape
            and value.dtype == sample.dtype
            and value.device == sample.device
        )

    def _store(
        self,
        key: tuple[tuple[int, ...], ...],
        body: fx.GraphModule,
        output: fx.Node,
        terminal: fx.Node,
        operands: tuple[fx.Node, ...],
    ) -> None:
        output.args = (operands,)
        body.graph.erase_node(terminal)
        body.recompile()
        self._bodies[key] = body

    def _shape_independent(self, count: int, arguments: tuple[_Argument, ...]) -> bool:
        # A concrete row count alone cannot prove identity at every capture size.
        # Proxy tracing rejects Python branches on tensor shapes or values.
        if count == 1:
            function: Callable[[Tensor], Tensor] | Callable[[Tensor, Tensor], Tensor] = lambda x: self._call(
                (x,), arguments
            )
        else:
            function = lambda x, y: self._call((x, y), arguments)
        try:
            body = fx.symbolic_trace(function)
        except (TraceError, RuntimeError):
            return False
        return len(tuple(body.graph.nodes)) == 2 * count

    @staticmethod
    def _key(tensors: tuple[Tensor, ...]) -> tuple[tuple[int, ...], ...]:
        return tuple(part for tensor in tensors for part in (tuple(tensor.shape), tuple(tensor.stride())))

    def operands(self, *tensors: Tensor) -> tuple[Tensor, ...] | None:
        """Return a conversion source or same-dtype add operands for the caller's port."""
        key = self._key(tensors)
        if key not in self._bodies:
            self._trace(tensors)
        body = self._bodies[key]
        return None if body is None else body(*tensors, *self._arguments())
