"""Expose terminal adds for vLLM residual norms and in-place value writes.

ModelSites exposes terminal adds to native norms and compiles the remaining authored
math together with its output ports, independently of vLLM's model compiler.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch
from torch import Tensor, fx
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.proxy import TraceError

from pllm.execution.model_sites import PatchInput, Place, ResidualRewrite, SiteValues, activation_write
from pllm.execution.module_graph import ModuleCall
from pllm.execution.torch_private import compile_math, compiling_method, set_tuple_output, tensor_node_metadata

_Argument = Tensor | ModuleCall | int | Callable[[Tensor], Tensor]


_Key = tuple[tuple[int, ...], ...]


def _method_ports(
    function: Callable[..., Tensor],
    names: tuple[str, ...],
    tensors: tuple[Tensor, ...],
    arguments: tuple[_Argument, ...],
    ports: tuple[Tensor, ...],
) -> None:
    # Dynamo installs bytecode globals here, outside the frozen workload's globals.
    result = function(*tensors, **dict(zip(names, arguments, strict=True)))
    (port,) = ports
    if result.shape != port.shape or result.dtype != port.dtype:
        raise ValueError(f"return the model port's shape and dtype {port.dtype}; cast at the end of the method")
    port.copy_(result)


def compile_site_method(rewrite: ResidualRewrite, *, read: bool = False) -> Callable[..., None]:
    """Share one fullgraph callable across a method's depths and model capture sizes."""
    function = rewrite.read if read else rewrite.write
    assert function is not None
    inputs = rewrite.read_inputs if read else rewrite.inputs
    return compile_math(partial(_method_ports, function, tuple(name for name, _ in inputs)), dynamic=True)


def _copy_ports(operands: tuple[Tensor, ...], ports: tuple[Tensor, ...]) -> None:
    if len(ports) == 1 and len(operands) == 2:
        torch.add(*operands, out=ports[0])
    else:
        for port, operand in zip(ports, operands, strict=True):
            port.copy_(operand.reshape_as(port))


class SiteBody:
    """One authored write or read, fused into its model ports with an optional terminal add."""

    def __init__(
        self,
        rewrite: ResidualRewrite,
        place: Place,
        values: SiteValues,
        samples: tuple[Tensor, ...],
        *,
        compiled: Callable[..., None] | None = None,
        read: bool = False,
    ) -> None:
        self.rewrite, self.place, self.values = rewrite, place, values
        self.compiled, self.read = compiled, read
        self.inputs = rewrite.read_inputs if read else rewrite.inputs
        self.function = rewrite.read if read else rewrite.write
        assert self.function is not None
        self._bodies: dict[_Key, tuple[Callable[..., tuple[Tensor, ...]], bool]] = {}
        self.identity = False
        with compiling_method("model-site", rewrite.name):
            self._trace(samples)
            if (
                compiled is not None
                and self.function is not activation_write
                and (not self.identity or place[0] == "final")
            ):
                self.emit(samples, (samples[0].new_empty(self._shape(samples[0])),))

    def _shape(self, sample: Tensor) -> tuple[int, ...]:
        return (len(sample), sample.shape[-1]) if self.read or self.place[0] == "final" else tuple(sample.shape)

    def _arguments(self) -> tuple[_Argument, ...]:
        depth = self.rewrite.depth(self.place)
        return tuple(source.value(self.values, depth) for _, source in self.inputs)

    def _call(self, tensors: tuple[Tensor, ...], arguments: tuple[_Argument, ...]) -> Tensor:
        function = self.function
        assert function is not None
        return function(*tensors, **dict(zip((name for name, _ in self.inputs), arguments, strict=True)))

    def _outputs(self, tensors: tuple[Tensor, ...], arguments: tuple[_Argument, ...]) -> tuple[Tensor, ...]:
        result = self._call(tensors, arguments)
        sample = tensors[0]
        expected = self._shape(sample)
        if result.shape != expected or result.dtype != sample.dtype:
            raise ValueError(
                f"{self.rewrite.name}: return {tuple(expected)} {sample.dtype} (the model dtype); "
                "compute in float32 inside the method and cast at the end"
            )
        return (result,)

    def _trace(self, samples: tuple[Tensor, ...]) -> None:
        self._bodies[self._key(samples)] = self._outputs, False
        if any(isinstance(value, ModuleCall) or callable(value) for value in self._arguments()):
            return
        if self.compiled is not None:
            self._trace_math(samples)
            return
        if self.read or self.rewrite.streams > 1:
            return
        try:
            self._trace_math(samples)
        except (RuntimeError, NotImplementedError):
            # The numerical opt-out also admits tensor bodies that cannot be traced.
            pass

    def _trace_math(self, samples: tuple[Tensor, ...]) -> None:
        key = self._key(samples)
        arguments = self._arguments()
        trace: Callable[[tuple[Tensor, ...], tuple[_Argument, ...]], tuple[Tensor, ...]] = lambda tensors, args: (
            self._outputs(tensors, args)
        )
        body = make_fx(trace, tracing_mode="fake", _allow_non_fake_inputs=True)(samples, arguments)
        output = next(node for node in body.graph.nodes if node.op == "output")
        (add,) = output.args[0]
        self._bodies[key] = body, False
        if self.read or self.rewrite.streams > 1:
            return
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
        key: _Key,
        body: fx.GraphModule,
        output: fx.Node,
        terminal: fx.Node,
        operands: tuple[fx.Node, ...],
    ) -> None:
        set_tuple_output(body, output, operands)
        body.graph.erase_node(terminal)
        body.recompile()
        self._bodies[key] = body, True

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

    def split(self, *tensors: Tensor) -> bool:
        """Whether native normalization can consume this write's terminal add."""
        if self.function is activation_write:
            return True
        if self.compiled is not None:
            return False
        key = self._key(tensors)
        if key not in self._bodies:
            self._trace(tensors)
        return self._bodies[key][1]

    def emit(self, tensors: tuple[Tensor, ...], ports: tuple[Tensor, ...]) -> None:
        """Run the author's body into its caller's ports, warming each shape before capture."""
        if self.function is activation_write:
            stream, branch = tensors
            ((_, source),) = self.inputs
            assert isinstance(source, PatchInput)
            patched = source.value(self.values, self.rewrite.depth(self.place))(branch)
            _copy_ports((stream, patched), ports)
            return
        if self.compiled is not None:
            with compiling_method("model-site", self.rewrite.name + (".read" if self.read else "")):
                self.compiled(tensors, self._arguments(), ports)
            return
        key = self._key(tensors)
        if key not in self._bodies:
            self._trace(tensors)
        _copy_ports(self._bodies[key][0](tensors, self._arguments()), ports)
