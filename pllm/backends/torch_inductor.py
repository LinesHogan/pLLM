"""Pinned Torch/Inductor implementation and its graph-derived capacity contract.

This provider owns compiler configuration, deterministic template selection and
guards on version-specific execution artifacts. Scheduling consumes only its
result ABI. Capacity facts come from the normalized export graph itself — the
layout span of every tensor value, an explicit charge for reduction temporaries
and a named slack fraction — never from timing, profiling or code-generation
internals. Inductor features stay enabled; the account over-counts what fusion
eliminates and covers what code generation invents.
"""

import math
import operator
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
    boundary_digest,
    encode_boundary,
    preserve_torch_rng,
    tensor_graph_result_nodes,
    tensor_node_metadata,
)


class _InductorConfig(Protocol):
    def patch(self, changes: dict[str, object]) -> AbstractContextManager[None]: ...


_INDUCTOR_GEMM = frozenset(
    {"aten.mm.default", "aten.matmul.default", "aten.bmm.default", "aten.addmm.default", "aten.linear.default"}
)

# Slack added to the graph-derived account for code-generation side effects the
# graph cannot name: concat packing slack, reduction semaphores and allocator
# rounding. Split-reduction partial buffers are bounded separately by charging
# each reduction operation's inputs (a split never exceeds the reduced element
# count), and the deterministic plain mm/bmm template carries no workspace.
# Named and versioned into the import identity so worker re-derivation and the
# driver compute the same account.
COMPILER_SLACK_FRACTION = 0.05

# Reductions whose temporary partial buffers the account charges explicitly.
_REDUCTION_PACKETS = frozenset({torch.ops.aten.sum, torch.ops.aten.mean})

# Operations accepted after ATen normalization: pointwise math, metadata views,
# cat/stack, sum/mean reductions and the plain GEMM family. Anything else can
# lower to an external library call or an opaque workspace that a graph account
# cannot see; those graphs stay on their original execution path instead.
_ALLOWED_PACKETS = frozenset(
    packet
    for names in (
        # Pointwise binary and comparison.
        (
            "add",
            "sub",
            "mul",
            "div",
            "pow",
            "remainder",
            "fmod",
            "maximum",
            "minimum",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
            "logical_and",
            "logical_or",
            "logical_xor",
            "where",
        ),
        # Pointwise unary and casts.
        (
            "neg",
            "abs",
            "exp",
            "log",
            "sqrt",
            "rsqrt",
            "reciprocal",
            "sigmoid",
            "silu",
            "relu",
            "gelu",
            "tanh",
            "erf",
            "clamp",
            "floor",
            "ceil",
            "round",
            "trunc",
            "sign",
            "sin",
            "cos",
            "isnan",
            "isinf",
            "to",
            "clone",
            "contiguous",
            "detach",
            "type_as",
        ),
        # Metadata views and packing (a view never allocates; counting it anyway is
        # a deliberate over-count, never an omission).
        (
            "view",
            "reshape",
            "permute",
            "transpose",
            "t",
            "expand",
            "broadcast_to",
            "slice",
            "select",
            "squeeze",
            "unsqueeze",
            "narrow",
            "as_strided",
            "flatten",
            "repeat",
            "cat",
            "stack",
        ),
        # Bounded reductions and the plain GEMM family.
        ("sum", "mean", "mm", "addmm", "bmm", "matmul", "linear"),
    )
    for packet in (getattr(torch.ops.aten, name) for name in names)
) | frozenset(
    {
        # ATen decompositions emit pointwise prims for dtype conversion.
        torch.ops.prims.convert_element_type,
    }
)


def supports_inductor_tensor_graph(source: torch.fx.GraphModule) -> bool:
    """Cheap backend feasibility check; final lowering must supply allocations.

    Operator support is determined by Inductor and the resulting implementation,
    not a second pointwise/reduction whitelist in the middle-end.
    """
    ports = (*(node for node in source.graph.nodes if node.op == "placeholder"), *tensor_graph_result_nodes(source))
    for node in ports:
        tensor = tensor_node_metadata(node)
        if (
            tensor is None
            or tensor.device.type not in ("cpu", "cuda")
            or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        ):
            return False
    operations = tuple(node for node in source.graph.nodes if node.op == "call_function")
    if any(str(node.target) in _INDUCTOR_GEMM for node in operations):
        for node in source.graph.nodes:
            if node.op == "placeholder":
                tensor = tensor_node_metadata(node)
                if tensor is None or tensor.device.type != "cuda" or tensor.dtype == torch.float64:
                    return False
    return bool(operations)


def _span_bytes(value: torch.Tensor) -> int:
    """Layout span of one tensor value; views of larger storage stay bounded."""
    if value.numel() == 0:
        return 0
    elements = sum((size - 1) * stride for size, stride in zip(value.shape, value.stride())) + 1
    return elements * value.element_size()


def _graph_allocations(graph: torch.fx.GraphModule) -> tuple[KernelAllocation, ...]:
    """Graph-derived capacity account for one normalized graph.

    Every tensor value is charged its layout span as if it materialized — the
    over-count covers fusion being disabled or partial, and views are counted
    knowingly. Each sum/mean operation additionally charges its tensor inputs:
    split-reduction partial buffers never exceed the reduced element count.
    The named slack covers the remainder (concat packing, semaphores, rounding).
    """
    allocations: list[KernelAllocation] = []
    subtotal = 0
    for node in graph.graph.nodes:
        if node.op != "call_function":
            continue
        packet = getattr(node.target, "overloadpacket", None)
        if packet not in _ALLOWED_PACKETS and node.target is not operator.getitem:
            raise UnsupportedTensorKernel(
                f"normalized operation {node.target} stays outside the pointwise/sum-reduction/GEMM subset"
            )
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            continue
        bytes_ = _span_bytes(value)
        subtotal += bytes_
        allocations.append(KernelAllocation(node.name, str(value.device), bytes_))
        if packet in _REDUCTION_PACKETS:
            for index, argument in enumerate(node.all_input_nodes):
                argument_value = argument.meta.get("val")
                if not isinstance(argument_value, torch.Tensor):
                    continue
                charge = _span_bytes(argument_value)
                subtotal += charge
                allocations.append(
                    KernelAllocation(f"{node.name}#reduction{index}", str(argument_value.device), charge)
                )
    slack = math.ceil(subtotal * COMPILER_SLACK_FRACTION)
    if slack:
        device = allocations[-1].device
        allocations.append(KernelAllocation(f"compiler-slack@{COMPILER_SLACK_FRACTION}", device, slack))
    return tuple(allocations)


def compile_inductor_tensor_graph(source: torch.fx.GraphModule) -> CompiledTensorCode:
    """Pinned Inductor adapter: deterministic choice, graph-derived account.

    Dynamic library types and compiler hooks stay in this backend adapter. No
    tensor memory number is obtained by timing, profiling or reading
    code-generation allocations. Inputs are borrowed read-only. Offset-one fake
    inputs force code generation that accepts unaligned pointers; runtime
    alignment copies are rejected.
    """
    from torch._dynamo.convert_frame import compile_lock
    from torch._inductor import config
    from torch._inductor.choices import InductorChoices
    from torch._inductor.codegen.common import KernelTemplate
    from torch._inductor.compile_fx import compile_fx_inner
    from torch._inductor.decomposition import select_decomp_table
    from torch._inductor.exc import InductorError
    from torch._inductor.kernel_inputs import KernelInputs
    from torch._inductor.kernel_template_choice import KernelTemplateChoice
    from torch._inductor.output_code import CompiledFxGraph
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
                    value.shape, value.stride(), storage_offset=1
                )
            if not isinstance(sample, FakeTensor):
                raise UnsupportedTensorKernel("compiler input normalization requires fake tensors")
            examples.append(sample)
    FakeTensorProp(graph, examples[0].fake_mode if examples else None).propagate_dont_convert_inputs(*examples)

    class TensorChoices(InductorChoices):
        def _finalize_template_configs(
            self,
            template_choices: dict[str, Generator[KernelTemplateChoice, None, None]],
            kernel_inputs: KernelInputs,
            templates: list[KernelTemplate | ExternKernelChoice],
            op_name: str,
            kwarg_overrides: dict[str, dict[str, object]] | None = None,
        ) -> list[KernelTemplateChoice]:
            # One deterministic native Triton choice, without a second timing
            # search or a library-workspace fallback. Plain mm/bmm templates
            # carry no workspace, which the graph account relies on; an extern
            # library call would own workspace the graph cannot name. pLLM
            # compares completed executions, including this template and its
            # fused epilogue.
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

    patches: dict[str, object] = {
        "fx_graph_cache": False,
        "fx_graph_remote_cache": False,
        "assume_aligned_inputs": False,
        # Fusion must retain the source operators' low-precision rounding
        # boundaries, including updates to a loop-carried value.
        "emulate_precision_casts": True,
        "max_autotune": False,
        "max_autotune_pointwise": False,
        "max_autotune_gemm": False,
        "combo_kernels": False,
        "graph_partition": False,
        "wrap_inductor_compiled_regions": False,
        # CUDA graph pools are allocator state the graph account cannot name.
        "triton.cudagraphs": False,
        "triton.autotune_pointwise": False,
        "triton.autotune_at_compile_time": True,
        "post_grad_custom_pre_pass": None,
        "post_grad_custom_post_pass": None,
        "_pre_fusion_custom_pass": None,
    }
    if any(str(node.target) in _INDUCTOR_GEMM for node in graph.graph.nodes if node.op == "call_function"):
        if any(value.device.type != "cuda" or value.dtype == torch.float64 for value in examples):
            raise UnsupportedTensorKernel("compiled GEMM requires CUDA float16, bfloat16 or float32 inputs")
        patches.update(
            {
                "max_autotune_gemm": True,
                "max_autotune_gemm_backends": "TRITON",
                "benchmark_epilogue_fusion": False,
                "autotune_fallback_to_aten": False,
            }
        )
    devices = sorted({t.device.index for t in examples if t.device.type == "cuda"})
    # Use Torch's own cold compilation lock for its scoped global config hooks.
    try:
        with (
            compile_lock,
            cast(_InductorConfig, config).patch({**patches, "_post_fusion_custom_pass": None}),
            V.set_choices_handler(TensorChoices()),
            torch.no_grad(),
            preserve_torch_rng(devices),
        ):
            # Normalize to Inductor's ATen dialect under the same scoped
            # config/RNG guard. Only private fake inputs are traced; the
            # original graph and its output guards remain the source contract.
            graph = make_fx(graph, decomposition_table=select_decomp_table(), tracing_mode="fake")(*examples)
            allocations = _graph_allocations(graph)
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
    if not isinstance(artifact, CompiledFxGraph):
        raise UnsupportedTensorKernel("Inductor compilation did not produce one compiled graph")
    # Execution convention guards: the region borrows inputs read-only and
    # accounts every output copy itself, so hidden mutation, captured
    # constants or alignment copies break the contract rather than the account.
    if artifact.mutated_inputs or artifact.constants or artifact.torchbind_constants or artifact.inputs_to_check:
        raise UnsupportedTensorKernel("compiled tensor block has mutation, hidden constants or alignment copies")

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

    return CompiledTensorCode(
        invoke,
        encode_boundary(
            (
                "inductor-tensor-readonly@7",
                torch.__version__,
                patches,
                "first-valid-mm-bmm-template-up-to-four@1",
                f"graph-account-slack@{COMPILER_SLACK_FRACTION}",
            )
        ),
        boundary_digest(artifact.source_code),
        tuple(allocations),
    )


class InductorTensorCompiler:
    """Stateless provider; all compilation settings are scoped by Torch's lock."""

    def supports(self, source: torch.fx.GraphModule) -> bool:
        return supports_inductor_tensor_graph(source)

    def compile(self, source: torch.fx.GraphModule) -> CompiledTensorCode:
        return compile_inductor_tensor_graph(source)
