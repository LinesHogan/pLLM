"""Audited vLLM kernel providers for source-declared region endpoints."""

from dataclasses import dataclass

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm import __version__ as vllm_version, _custom_ops, envs
from vllm.ir.ops import fused_add_rms_norm
from vllm.model_executor.layers.layernorm import RMSNorm

from pllm.contracts.region_imports import FunctionalKernelImport, NativeKernelImport
from pllm.contracts.region_program import Function, TensorType
from pllm.execution.codec import (
    boundary_digest,
    refresh_tensor_metadata,
    tensor_graph_result_nodes,
    tensor_node_metadata,
)
from pllm.execution.decoder_region import decoder_normalize_function
from pllm.execution.exported_cell import TensorSignature
from pllm.execution.module_graph import (
    TensorGraphPartition,
    functionalized_kernel_operators,
    tensor_fragment_allocations,
)
from pllm.execution.region_native import FunctionalKernelPrimitive

_NORM_FORWARD = RMSNorm.forward


def bind_vllm_tensor_kernel(partition: TensorGraphPartition) -> FunctionalKernelPrimitive:
    """Bind an original functionalized CUDA fragment, without rebuilding its math.

    The current provider covers vLLM's SwiGLU and fused-add RMSNorm kernels.
    They use supplied result buffers and have no hidden global/library workspace.
    Torch's functionalization allocations remain explicit in the fragment.
    """
    operators = functionalized_kernel_operators(partition.graph)
    if not partition.opaque or any(
        name not in ("_C.silu_and_mul.default", "_C.fused_add_rms_norm.default") for name in operators
    ):
        raise ValueError("vLLM tensor fragment has no audited kernel implementation")
    graph = refresh_tensor_metadata(partition.graph)

    def signature(node: torch.fx.Node) -> TensorSignature:
        value = tensor_node_metadata(node)
        if value is None or value.device.type != "cuda" or not value.is_contiguous():
            raise ValueError("vLLM tensor kernel requires contiguous CUDA tensor ports")
        return TensorSignature.of(value)

    inputs = tuple(signature(node) for node in graph.graph.nodes if node.op == "placeholder")
    outputs = tuple(signature(node) for node in tensor_graph_result_nodes(graph))
    devices = {str(value.device) for value in (*inputs, *outputs)}
    if len(devices) != 1:
        raise ValueError("vLLM tensor kernel requires one original execution device")
    (device,) = devices
    types = tuple(TensorType(value.shape, str(value.dtype), str(value.device)) for value in (*inputs, *outputs))
    digest = boundary_digest("vllm-export-kernel@1", vllm_version, torch.__version__, str(graph.graph), types)
    declaration = Function("vllm.export@1:" + digest, types[: len(inputs)], types[len(inputs) :])
    implementation = FunctionalKernelImport(
        declaration, "vllm.export-kernel@1:" + digest, device, tensor_fragment_allocations(graph)
    )
    return FunctionalKernelPrimitive.bind(graph, implementation)


def _original_norm(norm: RMSNorm) -> bool:
    return (
        type(norm) is RMSNorm
        and not envs.VLLM_BATCH_INVARIANT
        and norm.forward == _NORM_FORWARD.__get__(norm, RMSNorm)
        and norm.variance_size_override is None
        and norm.has_weight
        and norm.pass_weight_add
    )


@dataclass(frozen=True, slots=True)
class VllmRmsNormKernel:
    norm: RMSNorm
    implementation: NativeKernelImport
    epsilon: float
    weight: torch.Tensor

    @classmethod
    def inspect(cls, norm: RMSNorm, tensor: TensorType, layers: int) -> "VllmRmsNormKernel | None":
        if (
            not _original_norm(norm)
            or norm.weight.device.type != "cuda"
            or norm.weight.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or tensor.shape != (1, norm.hidden_size)
            or tuple(norm.weight.shape) != (norm.hidden_size,)
            or not norm.weight.is_contiguous()
            or tensor.dtype != str(norm.weight.dtype)
            or tensor.device != str(norm.weight.device)
        ):
            return None
        # Dispatch inspects layouts and dtype; no model tensor is executed or
        # copied while constructing the signed backend catalog.
        with FakeTensorMode():
            sample = torch.empty(tensor.shape, dtype=norm.weight.dtype, device=norm.weight.device)
            provider = fused_add_rms_norm.dispatch(sample, sample, norm.weight, norm.variance_epsilon)
        if provider.provider != "vllm_c" or not provider.inplace:
            return None
        declaration = decoder_normalize_function(tensor, layers)
        epsilon = float(norm.variance_epsilon)
        identity = boundary_digest(vllm_version, str(torch.__version__), declaration, epsilon)
        # vllm.kernels.vllm_c.fused_add_rms_norm directly calls the in-place
        # CUDA op: no raw output backing or global library scratch is allocated.
        # The NativePrimitive wrapper separately funds two private inputs and
        # its independent result copy; model weights stay borrowed upstream.
        implementation = NativeKernelImport(declaration, "vllm.fused_add_rms_norm@1:" + identity, tensor.device, ())
        return cls(norm, implementation, epsilon, norm.weight)

    def run(self, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        tensor = self.implementation.declaration.inputs[1]
        if (
            not isinstance(tensor, TensorType)
            or not _original_norm(self.norm)
            or self.norm.variance_epsilon != self.epsilon
            or self.norm.weight is not self.weight
            or any(
                (tuple(value.shape), str(value.dtype), str(value.device)) != (tensor.shape, tensor.dtype, tensor.device)
                for value in (hidden, residual)
            )
            or (tuple(self.norm.weight.shape), str(self.norm.weight.dtype), str(self.norm.weight.device))
            != ((tensor.shape[-1],), tensor.dtype, tensor.device)
            or not hidden.is_contiguous()
            or not residual.is_contiguous()
            or not self.norm.weight.is_contiguous()
            or fused_add_rms_norm.dispatch(hidden, residual, self.norm.weight, self.epsilon).provider != "vllm_c"
        ):
            raise ValueError("model normalization changed after its kernel contract was bound")
        _custom_ops.fused_add_rms_norm(hidden, residual, self.norm.weight, self.epsilon)
        return hidden


def qwen_region_kernels(norm: RMSNorm, tensor: TensorType, layers: int) -> tuple[NativeKernelImport, ...]:
    kernel = VllmRmsNormKernel.inspect(norm, tensor, layers)
    return () if kernel is None else (kernel.implementation,)
