"""Compile imported mathematics through a backend with explicit allocation facts."""

from math import prod

import torch

from pllm.contracts.region_imports import CompiledTensorImport
from pllm.contracts.tensor_compilation import TensorCompiler, UnsupportedTensorKernel
from pllm.backends.torch_inductor import InductorTensorCompiler
from pllm.execution.exported_cell import ExportedCell, TensorSignature
from pllm.execution.region_native import CompiledPrimitive
from pllm.execution.tensor_access import import_tensor_access_graph


# Default backend wiring lives here; the optimizer only consumes TensorCompiler.
DEFAULT_TENSOR_COMPILER: TensorCompiler = InductorTensorCompiler()


def _contiguous(spec: TensorSignature) -> bool:
    stride = 1
    for size, actual in zip(spec.shape[::-1], spec.stride[::-1], strict=True):
        if actual != stride:
            return False
        stride *= max(1, size)
    return True


def supports_tensor_compilation(source: ExportedCell, *, compiler: TensorCompiler = DEFAULT_TENSOR_COMPILER) -> bool:
    """A prefilter, not proof that the backend's eventual lowering is supported."""
    if "torch.compile" in source.transformations:
        return False
    specs = (*source.input_signatures, *source.output_signatures)
    if not specs or any(prod(s.shape) == 0 or any(stride < 0 for stride in s.stride) for s in specs):
        return False
    # Inputs are readonly borrows: retain their actual strides instead of
    # materializing transposed weights. Outputs still use the canonical-copy ABI.
    if any(not _contiguous(spec) for spec in source.output_signatures):
        return False
    if len({s.device for s in specs}) != 1:
        return False
    return compiler.supports(source.graph)


def compile_tensor_cell(source: ExportedCell, *, compiler: TensorCompiler = DEFAULT_TENSOR_COMPILER) -> CompiledPrimitive:
    """Compile during planning with read-only inputs and explicit output copies.

    Every result is a fresh root, including duplicate returns, while imported
    compiler allocations cover raw output storage. Input borrows preserve the
    original frame's storage lifetimes without another layer of private copies.
    """
    if not supports_tensor_compilation(source, compiler=compiler):
        raise UnsupportedTensorKernel("tensor compiler requires supported static layouts and a feasible backend implementation")
    access = import_tensor_access_graph(source)
    code = compiler.compile(source.graph)
    implementation = CompiledTensorImport(access, code.backend, code.code_digest, code.allocations)

    def invoke(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        raw = code.function(*tensors)
        result = tuple(raw[i] for i in source.output_positions)
        for value, signature in zip(result, source.output_signatures, strict=True):
            signature.check(value)
        return result

    return CompiledPrimitive(source, implementation, invoke)
