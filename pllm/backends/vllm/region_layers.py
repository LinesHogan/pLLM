"""Resource contracts for partitions captured from the original vLLM layer.

Attention owns the original KV accesses. The feed-forward suffix only reads
weights and prepared context. This adapter introduces no executor or page owner.
"""

from dataclasses import dataclass, replace
from collections.abc import Mapping
from types import MappingProxyType

import torch
from torch.export.graph_signature import OutputKind
from vllm import ir
from vllm.model_executor.models.qwen2 import Qwen2Attention, Qwen2DecoderLayer
from vllm.model_executor.layers.layernorm import RMSNorm

from pllm.contracts.backend_attachment import CausalModelDomain
from pllm.contracts.region_composition import RegionFunctionBody, inline_function_bodies, inline_region, select_function_bodies
from pllm.contracts.region_imports import RegionTensorImport, region_import_name
from pllm.contracts.region_program import (
    AccessMode, Block, BorrowedResource, Call, Function, Program, Region,
    ResourceAccess, ScalarType, TensorType, Value,
)
from pllm.execution.codec import boundary_digest
from pllm.execution.decoder_region import decoder_layer_function, index_cell, kv_resource
from pllm.execution.module_graph import (
    ModuleCallBinding, ModuleExport, ModuleGraphPartition, capture_module_export, capture_module_partition,
    exported_parameter_fields, module_has_execution_hooks,
)
from pllm.execution.exported_region import import_exported_region
from pllm.execution.exported_resources import borrow_exported_inputs
from pllm.execution.region_native import TensorSource
from pllm.execution.region_storage import import_region_functions
from pllm.backends.vllm.region_kernels import bind_vllm_tensor_kernel


_LAYER_FORWARD = Qwen2DecoderLayer.forward


def decoder_partition_bodies(tensor: TensorType, domain: CausalModelDomain, max_rows: int,
                             graph_digest: str) -> tuple[RegionFunctionBody, ...]:
    """Bind captured prefix/suffix ports to the original decoder resource ABI."""
    resources = (BorrowedResource("context", (max_rows,)), kv_resource(domain), BorrowedResource("weights", (domain.layers + 1,)))
    index = ScalarType.INDEX
    identity = boundary_digest(tensor, domain, max_rows, graph_digest)
    feed_forward = Function("decoder.readonly_suffix@1:" + identity, (index, index, tensor, tensor), (tensor, tensor),
        ("context", "weights"), (ResourceAccess("context", AccessMode.READ, (index_cell(0),)),
                                ResourceAccess("weights", AccessMode.READ, (index_cell(1),))))
    result: list[RegionFunctionBody] = []
    for residual in (False, True):
        declaration = decoder_layer_function(tensor, domain, residual=residual)
        attention = Function(("decoder.stateful_prefix_with_residual@1:" if residual else "decoder.stateful_prefix_entry@1:") + identity,
            declaration.inputs, declaration.outputs, declaration.resources, declaration.accesses)
        argument_names = ("row", "request", "token", "layer", "positions", "hidden", *(("residual",) if residual else ()))
        arguments = tuple(Value(name, type_) for name, type_ in zip(argument_names, declaration.inputs, strict=True))
        program = Program(Region(Block(arguments, (
            Call(tuple(value.name for value in arguments),
                (Value("attention", tensor), Value("carry", tensor)),
                attention.name),
            Call(("row", "layer", "attention", "carry"),
                (Value("output", tensor), Value("residual_out", tensor)),
                feed_forward.name),
        ), ("output", "residual_out"))), (attention, feed_forward), resources)
        body = RegionFunctionBody(declaration, program)
        body.validate()
        result.append(body)
    return tuple(result)


def _original_layer(layer: Qwen2DecoderLayer) -> bool:
    if (type(layer) is not Qwen2DecoderLayer or type(layer.self_attn) is not Qwen2Attention
        or type(layer.input_layernorm) is not RMSNorm
        or layer.forward != _LAYER_FORWARD.__get__(layer, Qwen2DecoderLayer)):
        return False
    # Existing layer-level captures and parameter/activation adapters retain
    # their whole-layer endpoint until their effects have explicit source bodies.
    return all(module.forward == type(module).forward.__get__(module, type(module))
        and not module_has_execution_hooks(module)
        for module in layer.modules())


@dataclass(frozen=True, slots=True)
class VllmDecoderBodies:
    layers: tuple[Qwen2DecoderLayer, ...]
    definitions: tuple[RegionFunctionBody, ...]
    calls: tuple[ModuleCallBinding, ...]
    graphs: tuple[tuple[ModuleGraphPartition, ModuleGraphPartition], ...]
    imports: tuple[RegionTensorImport, ...]
    functions: Mapping[str, TensorSource]
    suffix: RegionFunctionBody
    exports: tuple[ModuleExport, ...]

    @classmethod
    def inspect(cls, layers: tuple[Qwen2DecoderLayer, ...], tensor: TensorType, domain: CausalModelDomain,
                max_rows: int) -> "VllmDecoderBodies | None":
        if len(layers) != domain.layers or not all(_original_layer(layer) for layer in layers):
            return None
        # These are external resource facts, not an alternate layer forward.
        # FX reads the actual call order and frontier from the original method.
        graphs = tuple((capture_module_partition(layer, stateful=("self_attn",),
                            readonly=("input_layernorm", "post_attention_layernorm", "mlp"), none_inputs=("residual",)),
                        capture_module_partition(layer, stateful=("self_attn",),
                            readonly=("input_layernorm", "post_attention_layernorm", "mlp"))) for layer in layers)
        if any(len(graph.frontier) != 2 for pair in graphs for graph in pair):
            raise ValueError("captured decoder partition differs from its two-tensor frontier ABI")
        if any(str(first.suffix.graph) != str(next_.suffix.graph) for first, next_ in graphs):
            raise ValueError("decoder input specializations require different readonly suffixes")
        digest = boundary_digest(tuple(tuple(graph.digest for graph in pair) for pair in graphs))
        definitions = decoder_partition_bodies(tensor, domain, max_rows, digest)
        suffix = definitions[0].program.functions[1]
        weights = BorrowedResource("weights", (domain.layers + 1,))
        dtype = {"torch.float16": torch.float16, "torch.bfloat16": torch.bfloat16,
                 "torch.float32": torch.float32}.get(tensor.dtype)
        if dtype is None:
            raise ValueError("decoder source requires a supported floating tensor dtype")
        examples = tuple(torch.empty(tensor.shape, dtype=dtype, device=tensor.device) for _ in range(2))
        exports: list[ModuleExport] = []
        source_bodies: list[RegionFunctionBody] = []
        imports: dict[str, RegionTensorImport] = {}
        functions: dict[str, TensorSource] = {}
        for layer, (first, _) in zip(layers, graphs, strict=True):
            # Trace through the backend's own dispatch layer. Dynamo records
            # the selected original CUDA ops and makes parameter aliases
            # explicit before export lifts them into tensor ports.
            # vLLM executes this source under inference_mode. Capture the same
            # dispatch state instead of dropping Torch's runtime mode guards.
            with torch.inference_mode(), ir.enable_torch_wrap(False):
                # Loader subclasses add checkpoint-loading behavior. Their
                # detached views retain the same storage and original math,
                # while Torch functional_call avoids tracing those subclasses.
                captured = capture_module_export(first.suffix, examples,
                    parameters={name: value.detach() for name, value in first.suffix.named_parameters()})
            exports.append(captured)
            imported = import_exported_region(captured.program, bind_kernel=bind_vllm_tensor_kernel)
            if imported.signature.buffers_to_mutate:
                raise ValueError("decoder readonly suffix cannot update original buffers")
            fields = exported_parameter_fields(captured.program, tuple(layer.parameters()))
            borrowed, local = borrow_exported_inputs(imported, weights, fields)
            positions = tuple(i for i, spec in enumerate(imported.signature.output_specs) if spec.kind is OutputKind.USER_OUTPUT)
            if len(positions) != 2:
                raise ValueError("decoder suffix requires its two original output tensors")
            arguments = tuple(Value(name, type_) for name, type_ in zip(
                ("row", "layer", "hidden", "residual"), suffix.inputs, strict=True))
            _, ordinal, hidden, residual = arguments
            inlined = inline_region(borrowed.program, (hidden, residual, ordinal), "suffix")
            body = RegionFunctionBody(suffix, Program(Region(Block(arguments, inlined.operations,
                tuple(inlined.results[i].name for i in positions))),
                borrowed.program.functions, (BorrowedResource("context", (max_rows,)), weights)))
            body.validate()
            local_imports = import_region_functions(body.program, local)
            for imported in local_imports:
                name = region_import_name(imported)
                if name in imports and imports[name] != imported:
                    raise ValueError("decoder suffixes disagree on an imported implementation")
                imports[name] = imported
                functions.setdefault(name, local[name])
            source_bodies.append(body)
        if not source_bodies:
            raise ValueError("decoder source has no tensor suffix")
        source_body = select_function_bodies(tuple(source_bodies), selector="layer")
        definitions = tuple(replace(body, program=inline_function_bodies(body.program, (source_body,))) for body in definitions)
        return cls(layers, definitions,
            tuple(ModuleCallBinding.bind(module) for layer in layers for module in layer.modules()), graphs,
            tuple(imports[name] for name in sorted(imports)), MappingProxyType(functions), source_body, tuple(exports))

    def validate(self, layers: tuple[Qwen2DecoderLayer, ...]) -> None:
        if len(layers) != len(self.layers) or any(actual is not original or not _original_layer(actual)
                for actual, original in zip(layers, self.layers, strict=True)):
            raise ValueError("model layer changed after binding its source definition")
        modules = tuple(module for layer in layers for module in layer.modules())
        if len(modules) != len(self.calls):
            raise ValueError("model modules changed after binding their source definition")
        for binding, module in zip(self.calls, modules, strict=True):
            binding.validate(module)
        with torch.inference_mode(), ir.enable_torch_wrap(False):
            for exported in self.exports:
                exported.validate()

    def attention(self, ordinal: int, positions: torch.Tensor, hidden: torch.Tensor,
                  residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        graph = self.graphs[ordinal][int(residual is not None)]
        hidden_out, residual_out = graph.run_prefix((positions, hidden, residual))
        return hidden_out, residual_out
