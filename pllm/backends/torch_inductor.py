"""Pinned Torch/Inductor implementation and its closed allocation contract.

This provider owns compiler configuration, template selection and inspection of
version-specific lowering artifacts. Scheduling consumes only its result ABI.
"""

from collections.abc import Generator
from contextlib import AbstractContextManager
from copy import deepcopy
from typing import Protocol, cast

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

from pllm.contracts.region_imports import KernelAllocation
from pllm.contracts.tensor_compilation import CompiledTensorCode, UnsupportedTensorKernel
from pllm.execution.codec import (
    boundary_digest, encode_boundary, preserve_torch_rng, tensor_node_metadata, tensor_graph_result_nodes,
)


class _InductorConfig(Protocol):
    def patch(self, changes: dict[str, object]) -> AbstractContextManager[None]: ...


_INDUCTOR_GEMM = frozenset({"aten.mm.default", "aten.matmul.default", "aten.bmm.default", "aten.addmm.default", "aten.linear.default"})


def supports_inductor_tensor_graph(source: torch.fx.GraphModule) -> bool:
    """Cheap backend feasibility check; final lowering must supply allocations.

    Operator support is determined by Inductor and the resulting implementation,
    not a second pointwise/reduction whitelist in the middle-end.
    """
    ports = (*(node for node in source.graph.nodes if node.op == "placeholder"), *tensor_graph_result_nodes(source))
    for node in ports:
        tensor = tensor_node_metadata(node)
        if (tensor is None or tensor.device.type not in ("cpu", "cuda")
            or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)):
            return False
    operations = tuple(node for node in source.graph.nodes if node.op == "call_function")
    if any(str(node.target) in _INDUCTOR_GEMM for node in operations):
        for node in source.graph.nodes:
            if node.op == "placeholder":
                tensor = tensor_node_metadata(node)
                if tensor is None or tensor.device.type != "cuda" or tensor.dtype == torch.float64:
                    return False
    return bool(operations)


def compile_inductor_tensor_graph(source: torch.fx.GraphModule) -> CompiledTensorCode:
    """Pinned Inductor adapter: pointwise/reduction allocations after codegen.

    Dynamic library types and compiler hooks stay in this backend adapter. No
    tensor memory number is obtained by timing or profiling a sample execution.
    Inputs are borrowed read-only. Offset-one fake inputs force code generation
    that accepts unaligned pointers; runtime alignment copies are rejected.
    """
    from sympy import Integer
    from torch._dynamo.convert_frame import compile_lock
    from torch._inductor import config
    from torch._inductor.choices import InductorChoices
    from torch._inductor.codegen.common import KernelTemplate
    from torch._inductor.compile_fx import compile_fx_inner
    from torch._inductor.decomposition import select_decomp_table
    from torch._inductor.exc import InductorError
    from torch._inductor.graph import GraphLowering
    from torch._inductor.ir import (
        ComputedBuffer, ConcatKernel, NonOwningLayout, Pointwise, Reduction, TritonTemplateBuffer, MultiTemplateBuffer,
    )
    from torch._inductor.kernel_inputs import KernelInputs
    from torch._inductor.kernel_template_choice import KernelTemplateChoice
    from torch._inductor.output_code import CompiledFxGraph
    from torch._inductor.scheduler import BaseSchedulerNode, NopKernelSchedulerNode, SchedulerNode
    from torch._inductor.select_algorithm import ExternKernelChoice, TritonTemplate, TritonTemplateCaller
    from torch._inductor.utils import BoxedBool
    from torch._inductor.virtualized import V
    from torch.fx.experimental.proxy_tensor import make_fx

    graph = deepcopy(source)
    examples: list[FakeTensor] = []
    for node in graph.graph.nodes:
        node.meta.pop("tensor_meta", None)
        if node.op == "placeholder":
            value = tensor_node_metadata(node)
            if value is None:
                raise UnsupportedTensorKernel("Inductor import needs static tensor metadata")
            # TensorSignature guards layouts, not pointers or storage offsets.
            # Even assume_aligned_inputs=False infers alignment from offset-zero
            # examples in the pinned Torch version. Use deliberately unaligned
            # fake storage, without changing the source mathematical import.
            extent = sum((size - 1) * stride for size, stride in zip(value.shape, value.stride())) + 2
            with value.fake_mode:
                sample = torch.empty((extent,), dtype=value.dtype, device=value.device).as_strided(
                    value.shape, value.stride(), storage_offset=1)
            if not isinstance(sample, FakeTensor):
                raise UnsupportedTensorKernel("compiler input normalization requires fake tensors")
            examples.append(sample)
    FakeTensorProp(graph, examples[0].fake_mode if examples else None).propagate_dont_convert_inputs(*examples)
    expected = tuple(tensor_node_metadata(n) for n in tensor_graph_result_nodes(graph))
    captured: list[GraphLowering] = []

    class TensorChoices(InductorChoices):
        def _finalize_template_configs(self, template_choices: dict[str, Generator[KernelTemplateChoice, None, None]],
                kernel_inputs: KernelInputs, templates: list[KernelTemplate | ExternKernelChoice], op_name: str,
                kwarg_overrides: dict[str, dict[str, object]] | None = None) -> list[KernelTemplateChoice]:
            # One deterministic native Triton choice, without a second timing
            # search or a library-workspace fallback. pLLM compares completed
            # executions, including this template and its fused epilogue.
            for template in templates:
                if not isinstance(template, TritonTemplate) or template.name not in ("mm", "bmm"):
                    continue
                for index, choice in enumerate(template_choices[template.uid]):
                    if index == 4:
                        break
                    choice.layout = kernel_inputs.output_layout(flexible=False)
                    if isinstance(choice.choice, TritonTemplateCaller):
                        return [choice]
            raise UnsupportedTensorKernel("GEMM has no supported deterministic Triton template")

    def inspect(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        lowered = V.graph
        for group in nodes:
            for node in group.get_nodes():
                if isinstance(node, NopKernelSchedulerNode) and isinstance(node.node, ConcatKernel):
                    continue  # Allocation-only root; its producers fill disjoint views.
                if not isinstance(node, SchedulerNode):
                    raise UnsupportedTensorKernel("compiled allocation adapter requires pointwise/reduction scheduler nodes")
                operation = node.node
                if operation is None:
                    raise UnsupportedTensorKernel("compiled scheduler node has no tensor operation")
                if isinstance(operation, TritonTemplateBuffer) and not isinstance(operation, MultiTemplateBuffer):
                    if operation.mutated_inputs:
                        raise UnsupportedTensorKernel("GEMM template has extra mutation/workspace outputs")
                    continue
                if not isinstance(operation, ComputedBuffer) or not (
                        isinstance(operation.data, Pointwise)
                        or isinstance(operation.data, Reduction) and operation.data.reduction_type == "sum"):
                    raise UnsupportedTensorKernel("compiled allocation adapter requires pointwise/sum-reduction kernels without external calls")
        captured.append(lowered)
        return nodes

    patches: dict[str, object] = {
        "fx_graph_cache": False, "fx_graph_remote_cache": False,
        "inplace_buffers": False, "allow_buffer_reuse": False, "memory_planning": False,
        "assume_aligned_inputs": False,
        # Fusion must retain the source operators' low-precision rounding
        # boundaries, including updates to a loop-carried value.
        "emulate_precision_casts": True,
        "max_autotune": False, "max_autotune_pointwise": False, "max_autotune_gemm": False,
        "combo_kernels": False, "graph_partition": False, "wrap_inductor_compiled_regions": False,
        "triton.cudagraphs": False, "triton.autotune_pointwise": False, "triton.autotune_at_compile_time": True,
        "post_grad_custom_pre_pass": None, "post_grad_custom_post_pass": None, "_pre_fusion_custom_pass": None,
    }
    if any(str(node.target) in _INDUCTOR_GEMM
           for node in graph.graph.nodes if node.op == "call_function"):
        if any(value.device.type != "cuda" or value.dtype == torch.float64 for value in examples):
            raise UnsupportedTensorKernel("compiled GEMM requires CUDA float16, bfloat16 or float32 inputs")
        patches.update({"max_autotune_gemm": True, "max_autotune_gemm_backends": "TRITON",
                        "benchmark_epilogue_fusion": False, "autotune_fallback_to_aten": False})
    devices = sorted({t.device.index for t in examples if t.device.type == "cuda"})
    # Use Torch's own cold compilation lock for its scoped global config hooks.
    try:
        with (compile_lock, cast(_InductorConfig, config).patch({**patches, "_post_fusion_custom_pass": inspect}),
              V.set_choices_handler(TensorChoices()),
              torch.no_grad(), preserve_torch_rng(devices)):
            # Normalize to Inductor's ATen dialect under the same scoped
            # config/RNG guard. Only private fake inputs are traced; the
            # original graph and its output guards remain the source contract.
            graph = make_fx(graph, decomposition_table=select_decomp_table(), tracing_mode="fake")(*examples)
            artifact = compile_fx_inner(graph, examples, is_inference=True, cudagraphs=BoxedBool(False))
    except InductorError as error:
        cause: BaseException | None = error.inner_exception
        seen: set[int] = set()
        while cause is not None and id(cause) not in seen:
            if isinstance(cause, UnsupportedTensorKernel):
                raise cause from error
            seen.add(id(cause))
            cause = cause.__cause__ or cause.__context__
        raise
    if not isinstance(artifact, CompiledFxGraph) or len(captured) != 1:
        raise UnsupportedTensorKernel("Inductor allocation capture did not produce one compiled graph")
    lowered, = captured
    if (artifact.mutated_inputs or artifact.constants or artifact.torchbind_constants
            or lowered.wrapper_code.allocated_workspaces or artifact.inputs_to_check
            or getattr(lowered, "_defers_input_alignment")):
        raise UnsupportedTensorKernel("compiled tensor block has mutation, hidden constants, alignment copies or opaque workspace")

    def integer(value: object) -> int:
        if type(value) is int:
            return value
        if isinstance(value, Integer):
            return int(value)
        raise UnsupportedTensorKernel("compiled allocation layout must be static, not a sampled size")

    allocations: list[KernelAllocation] = []
    for buffer in lowered.buffers:
        if buffer.get_name() in lowered.removed_buffers:
            continue
        # Mean lowers to a sum and scaling. Split reductions remain separate
        # ComputedBuffers, so their partial sums enter the allocation envelope.
        template = isinstance(buffer, TritonTemplateBuffer) and not isinstance(buffer, MultiTemplateBuffer)
        if not template and not isinstance(buffer, ConcatKernel) and (not isinstance(buffer, ComputedBuffer) or not (
                isinstance(buffer.data, Pointwise)
                or isinstance(buffer.data, Reduction) and buffer.data.reduction_type == "sum")):
            raise UnsupportedTensorKernel("compiled allocation has no supported tensor lowering")
        if isinstance(buffer.get_output_spec(), NonOwningLayout):
            continue  # Its owning concat/output root is already in lowered.buffers.
        storage_size = integer(lowered.get_allocation_storage_size(buffer))
        allocations.append(KernelAllocation(buffer.get_name(), str(buffer.get_device()),
            storage_size * torch.empty((), dtype=buffer.get_dtype()).element_size()))
    if len(lowered.graph_outputs) != len(expected):
        raise UnsupportedTensorKernel("compiled tensor output arity differs from source")
    for value, guard in zip(lowered.graph_outputs, expected, strict=True):
        if guard is None or (tuple(integer(d) for d in value.get_size()), tuple(integer(d) for d in value.get_stride()),
                value.get_dtype(), value.get_device()) != (tuple(guard.shape), tuple(guard.stride()), guard.dtype, guard.device):
            raise UnsupportedTensorKernel("compiled tensor output layout differs from source")

    def invoke(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        values: object = artifact(list(args))
        if not isinstance(values, (tuple, list)):
            raise RuntimeError("compiled tensor graph returned a non-tensor structure")
        tensors: list[torch.Tensor] = []
        for value in cast(tuple[object, ...] | list[object], values):
            if not isinstance(value, torch.Tensor):
                raise RuntimeError("compiled tensor graph returned a non-tensor value")
            tensors.append(value)
        return tuple(tensors)

    return CompiledTensorCode(invoke, encode_boundary(("inductor-tensor-readonly@6", torch.__version__, patches,
                                                      "first-valid-mm-bmm-template-up-to-four@1")),
        boundary_digest(artifact.source_code), tuple(allocations))


class InductorTensorCompiler:
    """Stateless provider; all compilation settings are scoped by Torch's lock."""

    def supports(self, source: torch.fx.GraphModule) -> bool:
        return supports_inductor_tensor_graph(source)

    def compile(self, source: torch.fx.GraphModule) -> CompiledTensorCode:
        return compile_inductor_tensor_graph(source)
