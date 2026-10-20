"""Capture Qwen2 calls and bind causal prefill chunks to the runner's model and KV pages."""

import operator
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import groupby
from math import ceil
from typing import Generic

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import GraphModule, Node
from vllm import ir
from vllm.forward_context import ForwardContext, get_forward_context, override_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2Model
from vllm.sequence import IntermediateTensors

from pllm.authoring import DataT, ResultT, TensorStepInput
from pllm.backends.vllm.region_attention import attention_metadata, validate_attention_resources
from pllm.backends.vllm.region_layers import original_layer
from pllm.backends.vllm.window_scheduler import WindowAdmission, WindowBatch
from pllm.contracts.backend_attachment import BackendPlacementProbe
from pllm.contracts.packed_rows import PackedPhaseRowBatch
from pllm.contracts.parameter_overlay import ParameterOverlayFieldBinding
from pllm.contracts.region_imports import BorrowedTensorImport, KernelAllocation, NativeKernelImport
from pllm.contracts.region_program import BorrowedResource, Function, TensorType
from pllm.contracts.semantic_graph import AnchorId, TaskId, TaskSourceRegion
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.codec import boundary_digest
from pllm.execution.cuda_stream import upload
from pllm.execution.execution import TaskGraphInvocation
from pllm.execution.memory import MemoryTrace
from pllm.execution.model_scan import ModelScanSpan
from pllm.execution.module_graph import ModuleCallBinding, capture_module_calls, capture_module_partition
from pllm.execution.region_resources import RegionPageOwner
from pllm.execution.region_streams import RegionStreamBindings
from pllm.execution.shared_tensor import tensor_dtype
from pllm.execution.state_execution import StateExecution, StateTaskContext
from pllm.execution.tensor_borrow import MutableTensorView, PagedAllocationView, ReadonlyTensorCell, ReadonlyTensorView
from pllm.execution.trace import MAX_SNAPSHOT_BYTES


class _LayerMemory(MemoryTrace):
    """Original tensor allocation trace plus an explicit native-kernel audit."""

    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        operator = str(func)
        allowed = {
            "prim.device.default",
            "_C.rms_norm.default",
            "_C.fused_add_rms_norm.default",
            "_C.silu_and_mul.default",
            "_C.rotary_embedding.default",
            "vllm.unified_kv_cache_update.default",
            "vllm.unified_attention_with_output.default",
        }
        if not operator.startswith("aten.") and operator not in allowed:
            raise ValueError(f"ModelScan needs a native allocation/stream contract for {operator}")
        if operator == "aten._local_scalar_dense.default":
            raise ValueError("ModelScan layer endpoints cannot inspect device scalars on the host")
        return super().__torch_dispatch__(func, types, args, kwargs)


def _adapter_mlp(layer: Qwen2DecoderLayer, scale: float) -> GraphModule:
    source = capture_module_calls(layer.mlp)
    graph = torch.fx.Graph()
    mapped: dict[Node, Node] = {}
    a, b = graph.placeholder("adapter_a"), graph.placeholder("adapter_b")
    inputs: list[Node] = []
    projection: Node | None = None
    operand: Node | None = None
    for node in source.graph.nodes:
        if node.op == "output":
            assert operand is not None and len(inputs) == 1
            output = node.args[0]
            assert isinstance(output, Node)
            graph.output((mapped[output], operand, inputs[0]))
            continue
        copied = graph.node_copy(node, mapped.__getitem__)
        mapped[node] = copied
        if node.op == "placeholder":
            inputs.append(copied)
        if node.op == "call_module" and node.target == "down_proj":
            projection = node
            assert isinstance(node.args[0], Node)
            operand = mapped[node.args[0]]
        if node.op == "call_function" and node.target is operator.getitem and node.args == (projection, 0):
            assert operand is not None
            at = graph.call_method("t", (a,))
            bt = graph.call_method("t", (b,))
            down = graph.call_function(torch.matmul, (operand, at))
            up = graph.call_function(torch.matmul, (down, bt))
            delta = graph.call_function(operator.mul, (up, scale))
            mapped[node] = graph.call_function(operator.add, (copied, delta))
    graph.lint()
    return GraphModule(source, graph)


def _adapter_layer(layer: Qwen2DecoderLayer, scale: float, *, entry: bool) -> GraphModule:
    source = capture_module_calls(layer, none_inputs=("residual",) if entry else ())
    source.add_module("_adapter_mlp", _adapter_mlp(layer, scale))
    graph = torch.fx.Graph()
    mapped: dict[Node, Node] = {}
    a, b = graph.placeholder("adapter_a"), graph.placeholder("adapter_b")
    operands: tuple[Node, Node] | None = None
    for node in source.graph.nodes:
        if node.op == "call_module" and node.target == "mlp":
            args = torch.fx.map_arg(node.args, mapped.__getitem__)
            call = graph.call_module("_adapter_mlp", (a, b, *args))
            mapped[node] = graph.call_function(operator.getitem, (call, 0))
            operands = (
                graph.call_function(operator.getitem, (call, 1)),
                graph.call_function(operator.getitem, (call, 2)),
            )
        elif node.op == "output":
            assert operands is not None
            outputs = torch.fx.map_arg(node.args[0], mapped.__getitem__)
            graph.output((*outputs, *operands))
        else:
            mapped[node] = graph.node_copy(node, mapped.__getitem__)
    graph.lint()
    return GraphModule(source, graph)


@dataclass(frozen=True)
class ModelScanSource:
    """Original layer calls, with explicit effects only at the declared projections."""

    graphs: tuple[GraphModule, ...]
    suffix: GraphModule
    bindings: tuple[ModuleCallBinding, ...]

    @classmethod
    def capture(cls, model: Qwen2Model, effects: tuple[ParameterOverlayFieldBinding, ...]) -> "ModelScanSource":
        if not all(original_layer(layer) for layer in model.layers):
            raise ValueError("ModelScan needs the original Qwen2 layer implementations without implicit hooks")
        names = tuple(f"layers.{i}" for i in range(len(model.layers)))
        partition = capture_module_partition(
            model, stateful=names, readonly=("norm",), none_inputs=("input_ids", "intermediate_tensors")
        )
        calls = tuple(node.target for node in partition.prefix.graph.nodes if node.op == "call_module")
        if calls != names or len(partition.frontier) != 1:
            raise ValueError("ModelScan needs a complete layer chain and its original final normalization")
        targets = {effect.target.layer: effect for effect in effects}
        graphs = tuple(
            _adapter_layer(layer, targets[i].scale, entry=i == 0)
            if i in targets
            else capture_module_calls(layer, none_inputs=("residual",) if i == 0 else ())
            for i, layer in enumerate(model.layers)
        )
        return cls(
            graphs, partition.suffix, tuple(ModuleCallBinding.bind(module) for module in tuple(model.modules())[1:])
        )

    def imports(
        self,
        model: Qwen2Model,
        sources: tuple[TaskSourceRegion, ...],
        probe: BackendPlacementProbe,
    ) -> tuple[NativeKernelImport | BorrowedTensorImport, ...]:
        """Audit actual Qwen calls and charge the native allocations absent from fake execution."""
        assert probe.causal_domain is not None
        validate_attention_resources(
            tuple(layer.self_attn.attn for layer in model.layers), hidden_size=probe.hidden_size
        )
        result: list[NativeKernelImport | BorrowedTensorImport] = []
        for source in sources:
            for function in source.program.functions:
                if function.name.startswith("train."):
                    continue
                if function.name.startswith("weight."):
                    result.append(BorrowedTensorImport(function, 0))
                    continue
                layer = int(function.name.split(".")[-1])
                attention = model.layers[layer].self_attn.attn
                if attention.calculate_kv_scales or attention.query_quant is not None or attention.use_direct_call:
                    raise ValueError("ModelScan windows need unquantized opaque FA2 without adaptive KV scales")
                result.append(self._import(function, layer, attention, probe.causal_domain.positions))
        return tuple(result)

    def _import(
        self,
        function: Function,
        layer: int,
        attention: Attention,
        positions: int,
    ) -> NativeKernelImport:
        types = function.inputs[4:]
        hidden = types[0]
        assert isinstance(hidden, TensorType)
        dtype = tensor_dtype(TensorDType(hidden.dtype.removeprefix("torch.")))
        graph = self.graphs[layer]
        peaks: list[int] = []
        with torch.no_grad(), ir.enable_torch_wrap(False), FakeTensorMode(allow_non_fake_inputs=True) as mode:
            parameters: dict[str, torch.Tensor] = {
                name: mode.from_tensor(value.detach()) for name, value in graph.named_parameters()
            }
            parameters.update((name, mode.from_tensor(value.detach())) for name, value in graph.named_buffers())
            for rows in range(1, hidden.shape[0] + 1):
                h = torch.empty((rows, hidden.shape[1]), dtype=dtype, device=hidden.device)
                r = None if layer == 0 else torch.empty_like(h)
                offsets = torch.empty(rows, dtype=torch.int64, device=hidden.device)
                factors: list[torch.Tensor] = []
                for type_ in types[2:]:
                    assert isinstance(type_, TensorType)
                    factors.append(
                        torch.empty(
                            type_.shape,
                            dtype=tensor_dtype(TensorDType(type_.dtype.removeprefix("torch."))),
                            device=type_.device,
                        )
                    )

                def run() -> tuple[torch.Tensor, ...]:
                    values = torch.func.functional_call(graph, parameters, (*factors, offsets, h, r))
                    return tuple(_pad(value, hidden.shape[0]) for value in values)

                with _LayerMemory() as memory:
                    result = run()
                peaks.append(memory.peak)
                del result
        # FA2 varlen_fwd uses the current stream. Its native LSE and optional
        # single-query GQA split buffers are not visible to Torch fake tracing.
        # Pinned source: flash-attention 28e862d, flash_api.cpp:345-369, 650-835.
        heads, width = attention.num_heads, attention.head_size
        tile = 256 if width <= 64 else 128 if width <= 128 else 64
        splits = min(128, ceil(positions / tile))
        rounded = ceil(width / (32 if width <= 128 else 64)) * (32 if width <= 128 else 64)
        fa2 = 8 * heads * hidden.shape[0] + 2 * heads * width * dtype.itemsize
        fa2 += (splits * heads * (rounded + 1) * 4) if splits > 1 else 0
        diagnostics = positions * (2 * attention.num_kv_heads * width * dtype.itemsize + 40) + 9 * MAX_SNAPSHOT_BYTES
        allocations = (
            KernelAllocation("original_tensor_calls", hidden.device, max(peaks)),
            KernelAllocation("fa2_and_metadata", hidden.device, fa2 + 2048),
            KernelAllocation(
                "blas",
                hidden.device,
                torch.backends.cuda.cublas_workspace_size() + torch.backends.cuda.cublaslt_workspace_size(),
            ),
            KernelAllocation("observations", hidden.device, diagnostics),
        )
        return NativeKernelImport(
            function,
            "vllm.qwen2.fa2.scan@3:" + boundary_digest(str(graph.graph), allocations),
            hidden.device,
            allocations,
            host_reads=False,
        )

    def validate(self) -> None:
        """Retain the source modules and their original forward bindings."""
        for binding in self.bindings:
            binding.validate(binding.module)


class VllmModelScan(Generic[DataT, ResultT]):
    """Bind one native engine grant to the selected shared source window."""

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        execution: StateExecution[TensorStepInput, DataT, ResultT],
        anchor: AnchorId,
        effects: tuple[ParameterOverlayFieldBinding, ...],
        admission: WindowAdmission,
    ) -> None:
        self.model, self.execution, self.anchor = model.model, execution, anchor
        self.admission, self.chunk_size = admission, admission.chunk_size
        self.source = ModelScanSource.capture(self.model, effects)
        self.attentions = tuple(layer.self_attn.attn for layer in self.model.layers)
        validate_attention_resources(self.attentions, hidden_size=model.config.hidden_size)
        self._weights = ReadonlyTensorView(
            BorrowedResource("weights", (len(self.model.layers),)),
            tuple(
                ReadonlyTensorCell.bind(
                    (i,),
                    (layer.mlp.down_proj.weight, *layer.parameters())
                    + tuple(tensor for tensor in layer.buffers() if tensor is not self.attentions[i].kv_cache),
                )
                for i, layer in enumerate(self.model.layers)
            ),
        )
        self.original = self.model.forward
        self._spans: tuple[ModelScanSpan, ...] = ()
        self._invocation: TaskGraphInvocation[StateTaskContext[DataT]] | None = None
        self._context: ForwardContext | None = None
        self._positions: torch.Tensor | None = None
        self._hidden: torch.Tensor | None = None
        self._output: torch.Tensor | None = None
        self._batch: WindowBatch | None = None
        self._sequences: dict[tuple[int, int], torch.Tensor] = {}
        self.admissions: deque[WindowBatch] = deque(maxlen=256)
        self._auxiliary = (torch.cuda.Stream(device=execution.device),)
        for stream in self._auxiliary:
            for tensor in (*self.model.parameters(), *self.model.buffers(), *(a.kv_cache for a in self.attentions)):
                if tensor.is_cuda:
                    tensor.record_stream(stream)
        self.model.forward = self.model_forward

    @property
    def spans(self) -> tuple[ModelScanSpan, ...]:
        return self._spans

    def accept(self, batch: WindowBatch) -> None:
        """Retain the original allocator's immutable witness for the pending step."""
        if batch.grants and batch.admission != self.admission:
            raise ValueError("engine admission differs from the installed source window")
        self._batch = batch
        if batch.grants:
            self.admissions.append(batch)

    def borrow_streams(self, owner: RegionPageOwner, lanes: int) -> RegionStreamBindings:
        assert owner.pin.stream is not None and lanes == 2
        return RegionStreamBindings(owner.pin.stream, self._auxiliary)

    def inputs(self, request: int, chunk: int) -> tuple[torch.Tensor, ...]:
        assert self._hidden is not None
        span = self.spans[request]
        start = span.row.packed_row_index
        return tuple(
            _pad(self._hidden[start + i : start + min(i + chunk, span.length)], chunk)
            for i in range(0, span.length, chunk)
        )

    def weights_view(self, resource: BorrowedResource) -> ReadonlyTensorView:
        assert resource == self._weights.resource
        return self._weights

    def input_view(self, resource: BorrowedResource) -> ReadonlyTensorView:
        assert self._positions is not None and self._context is not None
        cells: list[ReadonlyTensorCell] = []
        for request in range(resource.domain[0]):
            for layer, attention in enumerate(self.attentions):
                tensors: tuple[torch.Tensor, ...] = ()
                if request < len(self.spans):
                    span = self.spans[request]
                    begin, end = span.row.packed_row_index, span.row.packed_row_index + span.length
                    metadata, _, slots = attention_metadata(self._context, attention)
                    tensors = (
                        self._positions[begin:end],
                        metadata.block_table[span.row.request_ordinal : span.row.request_ordinal + 1],
                        slots[begin:end],
                        *(value for (r, _), value in self._sequences.items() if r == request),
                    )
                cells.append(ReadonlyTensorCell.bind((request, layer), tensors))
        return ReadonlyTensorView(resource, tuple(cells))

    def kv_view(self, resource: BorrowedResource) -> MutableTensorView:
        assert self._batch is not None
        grants = {grant.request_id: grant for grant in self._batch.grants}
        kernel_block = self.attentions[0].kv_cache.shape[2]
        if self._batch.block_size % kernel_block or any(a.kv_cache.shape[2] != kernel_block for a in self.attentions):
            raise ValueError("KV allocation grants and kernel block sizes must share an exact subdivision")
        factor = self._batch.block_size // kernel_block
        blocks = tuple(
            tuple(block * factor + i for block in grants[span.row.identity.request_id].blocks for i in range(factor))
            for span in self.spans
        )
        return PagedAllocationView(resource, tuple(a.kv_cache for a in self.attentions), blocks)

    def prepare(self, rows: PackedPhaseRowBatch, invocation: TaskGraphInvocation[StateTaskContext[DataT]]) -> None:
        """Validate exact chunk alignment, original request identities and prompt tails once."""
        self.clear()
        assert self._batch is not None
        grants = {grant.request_id: grant for grant in self._batch.grants}
        spans: list[ModelScanSpan] = []
        for ordinal, group in groupby(rows.rows, key=lambda row: row.request_ordinal):
            packed = tuple(group)
            first = packed[0]
            grant = grants[first.identity.request_id]
            if (
                ordinal != len(spans)
                or first.token_offset != grant.prefix
                or len(packed) != grant.tokens
                or any(
                    row.identity != first.identity
                    or row.phase is not first.phase
                    or row.token_offset != first.token_offset + i
                    or row.packed_row_index != first.packed_row_index + i
                    for i, row in enumerate(packed)
                )
            ):
                raise ValueError("model rows must match the original engine allocation grant")
            span = ModelScanSpan(first, len(packed), grant.prompt_length)
            chunk = self.admission.chunk_size
            if span.learning and (
                grant.prefix % chunk
                or grant.tokens > chunk * self.admission.chunks
                or grant.tokens % chunk
                and grant.prefix + grant.tokens != grant.prompt_length
            ):
                raise ValueError("admit complete chunks within the signed window bound, plus only a true prompt tail")
            if not span.learning and span.length != 1:
                raise ValueError("frozen decode requires one original token per request")
            spans.append(span)
        if not spans or len(spans) != len(grants):
            raise ValueError("prepared model spans must cover every admitted request exactly")
        self._spans, self._invocation = tuple(spans), invocation
        invocation.context.model_scan = self

    def model_forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if self._invocation is None:
            return self.original(input_ids, positions, intermediate_tensors, inputs_embeds)
        if (
            intermediate_tensors is not None
            or positions.ndim != 1
            or positions.numel() != sum(s.length for s in self.spans)
        ):
            raise ValueError("ModelScan needs one unsharded position per admitted token")
        self.source.validate()
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.model.embed_input_ids(input_ids)
        self._positions, self._hidden, self._context = positions, inputs_embeds, get_forward_context()
        chunks = tuple(
            (request, span.row.token_offset + start, min(chunk, span.length - start))
            for request, span in enumerate(self.spans)
            for chunk in (self.chunk_size if span.learning else 1,)
            for start in range(0, span.length, chunk)
        )
        sequences = upload([[0, rows, start + rows] for _, start, rows in chunks], inputs_embeds.device, torch.int32)
        self._sequences = {(request, start): sequences[i] for i, (request, start, _) in enumerate(chunks)}
        for stream in self._auxiliary:
            for tensor in (positions, sequences):
                tensor.record_stream(stream)
            for attention in self.attentions:
                metadata, _, slots = attention_metadata(self._context, attention)
                metadata.block_table.record_stream(stream)
                slots.record_stream(stream)
        self.execution.execute_anchor(self._invocation, self.anchor)
        assert self._output is not None
        return self._output

    def forward(
        self,
        request: int,
        layer: int,
        start: int,
        rows: int,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        parameters: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        assert self._context is not None and self._positions is not None and self._hidden is not None
        span = self.spans[request]
        begin = span.row.packed_row_index + start - span.row.token_offset
        stop, length = begin + rows, start + rows
        sequence = self._sequences[request, start]
        attention = self.attentions[layer]
        base, _, mapping = attention_metadata(self._context, attention)
        slot = mapping[begin:stop]
        prefill = span.learning
        metadata = replace(
            base,
            num_actual_tokens=rows,
            max_query_len=rows,
            max_seq_len=length,
            query_start_loc=sequence[:2].to(base.query_start_loc.dtype),
            seq_lens=sequence[2:].to(base.seq_lens.dtype),
            block_table=base.block_table[span.row.request_ordinal : span.row.request_ordinal + 1],
            slot_mapping=slot,
            num_decode_reqs=int(not prefill),
            num_prefill_reqs=int(prefill),
            num_decode_tokens=0 if prefill else rows,
            num_prefill_tokens=rows if prefill else 0,
            scheduler_metadata=None,
            prefix_scheduler_metadata=None,
        )
        context = replace(
            self._context,
            attn_metadata={attention.layer_name: metadata},
            slot_mapping={attention.layer_name: slot},
            batch_descriptor=None,
            is_padding=None,
        )
        with override_forward_context(context):
            outputs = self.source.graphs[layer](
                *parameters, self._positions[begin:stop], hidden[:rows], None if layer == 0 else residual[:rows]
            )
        self.observe_kv(request, layer, start, length)
        return tuple(_pad(value, hidden.shape[0]) for value in outputs)

    def observe_kv(self, request: int, layer: int, start: int, stop: int) -> None:
        assert self._invocation is not None and self._context is not None
        owner = self._invocation.context
        if owner.trace.observing:
            span, attention = self.spans[request], self.attentions[layer]
            metadata, cache, _ = attention_metadata(self._context, attention)
            positions = torch.arange(stop, device=cache.device)
            blocks = metadata.block_table[span.row.request_ordinal, positions // cache.shape[2]].long()
            owner.trace.observe(
                owner.invocation_id,
                TaskId(0),
                f"{layer}.kv",
                cache[blocks, :, positions % cache.shape[2], :],
                row=span.row,
                iteration=start // self.chunk_size if span.learning else 0,
            )

    def finish(
        self,
        outputs: tuple[tuple[torch.Tensor, ...], ...],
        residuals: tuple[tuple[torch.Tensor, ...], ...],
        chunks: tuple[int, ...],
    ) -> None:
        values: list[torch.Tensor] = []
        for span, hidden, residual, chunk in zip(self.spans, outputs, residuals, chunks, strict=True):
            for i, (h, r) in enumerate(zip(hidden, residual, strict=True)):
                rows = min(chunk, span.length - i * chunk)
                values.append(self.source.suffix((h[:rows], r[:rows])))
        self._output = torch.cat(values)

    def clear(self) -> None:
        self._spans, self._invocation, self._context = (), None, None
        self._positions = self._hidden = self._output = None
        self._sequences.clear()

    def close(self) -> None:
        self.clear()
        self.model.forward = self.original


def _pad(value: torch.Tensor, rows: int) -> torch.Tensor:
    """Extend only the region value layout; padded rows never reach model kernels."""
    return torch.nn.functional.pad(value, (0, 0, 0, rows - value.shape[0]))
