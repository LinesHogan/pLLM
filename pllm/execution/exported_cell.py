"""Cold export and functional state-output binding for an iteration's mathematics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Lock
import warnings

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.export.graph_signature import InputKind, OutputKind, TensorArgument

from pllm.contracts.region_program import Function, TensorType
from pllm.execution.codec import (
    boundary_digest, check_functional_aten, check_static_export, compile_tensor_function,
    functionalize_export, tensor_graph_outputs, tensor_graph_result_nodes, tensor_node_metadata,
)
from pllm.execution.freeze import fingerprint


CellFunction = Callable[..., tuple[torch.Tensor, ...]]


class NumericalTransformWarning(UserWarning):
    """The selected plan may change floating-point rounding, not algorithm semantics."""


_warned: set[str] = set()
_warning_lock = Lock()


def warn_numerical_transforms(signature: str, transformations: tuple[str, ...]) -> None:
    if not transformations:
        return
    with _warning_lock:
        if signature in _warned:
            return
        _warned.add(signature)
    warnings.warn(
        "pLLM selected " + ", ".join(transformations)
        + "; these transformations may change floating-point results, including rounding and overflow. "
        "Set ExecutionOptions(allow_numerical_transforms=False) to retain the original numerical path.",
        NumericalTransformWarning, stacklevel=3,
    )


class _Cell(torch.nn.Module):
    def __init__(self, function: CellFunction) -> None:
        super().__init__()
        self.function = function

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.function(*args)


@dataclass(frozen=True, slots=True)
class TensorSignature:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def of(cls, tensor: torch.Tensor) -> TensorSignature:
        return cls(tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)

    def check(self, tensor: torch.Tensor) -> None:
        if TensorSignature.of(tensor) != self:
            raise ValueError("tensor differs from its exported shape, stride, dtype or device guard")


def _check_distinct_inputs(tensors: tuple[torch.Tensor, ...]) -> None:
    # The first import supports distinct input storage. Cloning aliased examples
    # separately would silently change their mutation semantics before export.
    # Metadata-only export has storage identities but no usable data pointers.
    # Preserve its alias check instead of treating every fake pointer as zero.
    storage = [(str(t.device), t.untyped_storage() if isinstance(t, FakeTensor) else t.untyped_storage().data_ptr())
               for t in tensors if t.numel()]
    if len(set(storage)) != len(storage):
        raise ValueError("exported cell inputs must have distinct storage; aliased input binding is not supported yet")


@dataclass(frozen=True, slots=True)
class ExportedCell:
    digest: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    input_signatures: tuple[TensorSignature, ...]
    output_signatures: tuple[TensorSignature, ...]
    mutations: tuple[tuple[str, str], ...]
    # Functional export places mutation results before the public return tuple.
    output_positions: tuple[int, ...]
    graph: torch.fx.GraphModule
    function: CellFunction
    transformations: tuple[str, ...]

    @classmethod
    def from_export(cls, program: torch.export.ExportedProgram, *, inputs: tuple[str, ...] | None = None,
                    outputs: tuple[str, ...] | None = None) -> ExportedCell:
        """Import an existing graph, with lifted state passed through its ports.

        Inputs follow ``graph_signature.input_specs``; parameter/buffer/constant
        ports default to their original targets. Outputs include functional
        mutation results in export order, followed by user results. The caller
        binds the required versions and publishes updates through their owner.
        This never invokes ``program.module()`` or copies its parameter storage.
        """
        return cls._from_functional_export(functionalize_export(program), inputs, outputs)

    @classmethod
    def _from_functional_export(cls, program: torch.export.ExportedProgram, inputs: tuple[str, ...] | None,
                                outputs: tuple[str, ...] | None) -> ExportedCell:
        program.validate()
        check_static_export(program)
        signature, graph = program.graph_signature, program.graph_module
        input_defaults: list[str] = []
        for spec in signature.input_specs:
            if (not isinstance(spec.arg, TensorArgument)
                or spec.kind not in (InputKind.USER_INPUT, InputKind.PARAMETER, InputKind.BUFFER, InputKind.CONSTANT_TENSOR)):
                raise ValueError("export cell inputs must be explicit tensor ports, including lifted state")
            input_defaults.append(spec.target if spec.target is not None else spec.arg.name)
        inputs = tuple(input_defaults) if inputs is None else inputs
        outputs = tuple(f"result_{i}" for i in range(len(signature.output_specs))) if outputs is None else outputs
        if (len(inputs) != len(signature.input_specs) or len(outputs) != len(signature.output_specs)
            or len(set(inputs)) != len(inputs) or len(set(outputs)) != len(outputs)):
            raise ValueError("exported cell ports must have unique names and match the complete graph signature")
        mutations: list[tuple[str, str]] = []
        for ordinal, spec in enumerate(signature.output_specs):
            if not isinstance(spec.arg, TensorArgument) or spec.kind not in (
                OutputKind.USER_OUTPUT, OutputKind.USER_INPUT_MUTATION, OutputKind.BUFFER_MUTATION):
                raise ValueError("export cell outputs must be tensors with explicit state updates")
            if spec.kind is OutputKind.USER_OUTPUT:
                continue
            targets = tuple(inputs[i] for i, source in enumerate(signature.input_specs)
                if isinstance(source.arg, TensorArgument) and (
                    spec.kind is OutputKind.USER_INPUT_MUTATION and source.kind is InputKind.USER_INPUT and source.arg.name == spec.target
                    or spec.kind is OutputKind.BUFFER_MUTATION and source.kind is InputKind.BUFFER and source.target == spec.target))
            if len(targets) != 1:
                raise ValueError("export state update must identify exactly one original input port")
            mutations.append((targets[0], outputs[ordinal]))
        for node in graph.graph.nodes:
            if node.op not in ("placeholder", "call_function", "output"):
                raise ValueError(f"unsupported exported cell node: {node.op}")
            if node.op == "call_function":
                check_functional_aten(node.target)
                if node.is_impure():
                    raise ValueError(f"exported cell contains an undeclared effect: {node.target}")

        def tensor_signature(node: torch.fx.Node) -> TensorSignature:
            tensor = tensor_node_metadata(node)
            if tensor is None or tensor.layout != torch.strided or any(type(d) is not int for d in (*tensor.shape, *tensor.stride())):
                raise ValueError("export cell requires static strided tensor metadata")
            return TensorSignature.of(tensor)

        arguments = tuple(tensor_signature(node) for node in graph.graph.nodes if node.op == "placeholder")
        results = tuple(tensor_signature(node) for node in tensor_graph_result_nodes(graph))
        if len(arguments) != len(inputs) or len(results) != len(outputs):
            raise ValueError("export graph ports differ from its original signature")

        def functional(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return tensor_graph_outputs(graph, args)

        digest = boundary_digest("export-cell@2", str(graph.graph), str(signature), torch.__version__,
            tuple((s.shape, s.stride, str(s.dtype), str(s.device)) for s in (*arguments, *results)), inputs, outputs)
        return cls(digest, inputs, outputs, arguments, results, tuple(mutations), tuple(range(len(outputs))), graph, functional, ())

    @classmethod
    def capture(cls, function: CellFunction, inputs: tuple[str, ...], outputs: tuple[str, ...],
                examples: tuple[torch.Tensor, ...]) -> ExportedCell:
        if len(inputs) != len(examples) or len(set(inputs)) != len(inputs) or len(set(outputs)) != len(outputs):
            raise ValueError("exported cell ports must have unique names and complete tensor examples")
        _check_distinct_inputs(examples)
        # Export executes Python: use private examples, never arena borrows or user-owned inputs.
        samples = tuple(t.detach().clone() for t in examples)
        with torch.no_grad():
            program = functionalize_export(torch.export.export(_Cell(function), samples, strict=False))
        signature = program.graph_signature
        if any(spec.kind is not InputKind.USER_INPUT for spec in signature.input_specs):
            raise ValueError("iteration cells must receive parameters, buffers and tensor constants through explicit input ports")
        if signature.buffers_to_mutate:
            raise ValueError("iteration buffer writes require explicit managed state input ports")
        if len(signature.user_inputs) != len(inputs):
            raise ValueError("iteration cell input structure must be a flat tensor tuple")
        positions = tuple(i for i, spec in enumerate(signature.output_specs) if spec.kind is OutputKind.USER_OUTPUT)
        if len(positions) != len(outputs):
            raise ValueError("iteration cell output tuple differs from its declared output ports")
        input_names = dict(zip(signature.user_inputs, inputs, strict=True))
        output_nodes = {spec.arg.name: outputs[i] for i, spec in enumerate(
            spec for spec in signature.output_specs if spec.kind is OutputKind.USER_OUTPUT)}
        mutations: list[tuple[str, str]] = []
        for result, target in signature.user_inputs_to_mutate.items():
            if result not in output_nodes:
                raise ValueError("a mutated cell input must also be returned as an explicit state output")
            mutations.append((input_names[target], output_nodes[result]))
        imported = cls._from_functional_export(program, inputs, None)
        def functional(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
            values = imported.function(*args)
            return tuple(values[i] for i in positions)
        # The authoring wrapper keeps its public result tuple and explicit-return
        # requirement. Shape facts and graph validation use the same import path.
        return replace(imported, digest=boundary_digest(fingerprint(function), imported.digest, outputs, positions),
            outputs=outputs, output_signatures=tuple(imported.output_signatures[i] for i in positions),
            mutations=tuple(mutations), output_positions=positions, function=functional)

    def compiled(self, examples: tuple[torch.Tensor, ...]) -> ExportedCell:
        """Compile a chosen tensor block after structural scheduling, on private inputs.

        This constructs a candidate. The plan selector owns the numerical policy
        and warns only if it actually selects the candidate.
        """
        self(*examples)  # Check all import guards before compiling.
        executable = compile_tensor_function(self.function)
        with torch.no_grad():
            executable(*tuple(t.detach().clone() for t in examples))
        return replace(self, function=executable, transformations=(*self.transformations, "torch.compile"))

    @property
    def declaration(self) -> Function:
        """Import into a structured region; the original tensor graph stays available."""
        return Function(self.digest,
            tuple(TensorType(t.shape, str(t.dtype), str(t.device)) for t in self.input_signatures),
            tuple(TensorType(t.shape, str(t.dtype), str(t.device)) for t in self.output_signatures))

    @property
    def output_bytes(self) -> int:
        from math import prod
        return sum(prod(t.shape) * torch.empty((), dtype=t.dtype).element_size() for t in self.output_signatures)

    def __call__(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        _check_distinct_inputs(args)
        return self.invoke(args)

    def invoke(self, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Invoke normalized immutable values, which may share backing storage.

        Capture and the direct authoring entry retain their no-alias guard:
        aliasing there could change Python's in-place mutation semantics. Here
        all writes are already explicit functional outputs; a repeated value
        denotes the same old version, not a second mutable argument.
        """
        if len(args) != len(self.inputs):
            raise ValueError("iteration cell input arity differs from its exported signature")
        for spec, tensor in zip(self.input_signatures, args, strict=True):
            spec.check(tensor)
        with torch.no_grad():
            return self.function(*args)
