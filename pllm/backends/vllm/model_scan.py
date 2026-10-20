"""Capture Qwen2 calls and bind causal prefill chunks to the runner's model and KV pages."""

import operator
from dataclasses import dataclass, replace
from typing import Generic

import torch
from torch.fx import GraphModule, Node
from vllm.forward_context import ForwardContext, get_forward_context, override_forward_context
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2Model
from vllm.sequence import IntermediateTensors

from pllm.authoring import DataT, ResultT, TensorStepInput
from pllm.backends.vllm.region_attention import attention_metadata, validate_attention_resources
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch
from pllm.contracts.parameter_overlay import ParameterOverlayFieldBinding
from pllm.contracts.region_program import BorrowedResource
from pllm.contracts.semantic_graph import AnchorId, TaskId
from pllm.execution.cuda_stream import upload
from pllm.execution.execution import TaskGraphInvocation
from pllm.execution.module_graph import ModuleCallBinding, capture_module_calls, capture_module_partition
from pllm.execution.state_execution import StateExecution, StateTaskContext
from pllm.execution.tensor_borrow import TensorResourceView


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

    def validate(self) -> None:
        """Retain the source modules and their original forward bindings."""
        for binding in self.bindings:
            binding.validate(binding.module)


class VllmModelScan(Generic[DataT, ResultT]):
    """Model input/output handoff; the signed task owns all adapter advancement."""

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        execution: StateExecution[TensorStepInput, DataT, ResultT],
        anchor: AnchorId,
        effects: tuple[ParameterOverlayFieldBinding, ...],
        chunk_size: int,
    ) -> None:
        self.model, self.execution, self.anchor = model.model, execution, anchor
        self.chunk_size = chunk_size
        self.source = ModelScanSource.capture(self.model, effects)
        self.attentions = tuple(layer.self_attn.attn for layer in self.model.layers)
        validate_attention_resources(self.attentions, hidden_size=model.config.hidden_size)
        self.original = self.model.forward
        self._rows: tuple[PackedPhaseRow, ...] = ()
        self._invocation: TaskGraphInvocation[StateTaskContext[DataT]] | None = None
        self._context: ForwardContext | None = None
        self._positions: torch.Tensor | None = None
        self._hidden: torch.Tensor | None = None
        self._output: torch.Tensor | None = None
        self.model.forward = self.model_forward

    @property
    def row(self) -> PackedPhaseRow:
        """The request and phase whose model inputs were prepared."""
        return self._rows[0]

    @property
    def length(self) -> int:
        """Number of original scheduled token rows."""
        return len(self._rows)

    def weight(self, layer: int) -> torch.Tensor:
        """Borrow the selected original frozen projection."""
        return self.model.layers[layer].mlp.down_proj.weight.detach()

    def inputs(self, chunk: int) -> tuple[torch.Tensor, ...]:
        """Pack the embedding frontier without including padding in any model call."""
        assert self._hidden is not None
        return tuple(_pad(self._hidden[start : start + chunk], chunk) for start in range(0, self.length, chunk))

    def kv_view(self, resource: BorrowedResource) -> TensorResourceView:
        """Borrow disjoint full layer caches; causal order is retained within each."""
        return TensorResourceView(resource, tuple(attention.kv_cache for attention in self.attentions))

    def prepare(self, rows: PackedPhaseRowBatch, invocation: TaskGraphInvocation[StateTaskContext[DataT]]) -> None:
        """Reject unsupported lifecycle cases before running any model cell."""
        self.clear()
        if (
            not rows.rows
            or len({row.identity for row in rows.rows}) != 1
            or len({row.phase for row in rows.rows}) != 1
            or tuple(row.packed_row_index for row in rows.rows) != tuple(range(len(rows.rows)))
            or tuple(row.token_offset for row in rows.rows)
            != tuple(range(rows.rows[0].token_offset, rows.rows[0].token_offset + len(rows.rows)))
            or rows.rows[0].phase is HiddenPhase.PREFILL
            and rows.rows[0].token_offset != 0
        ):
            raise ValueError("ModelScan currently needs one complete fresh prompt or one request's decode step")
        self._rows, self._invocation = rows.rows, invocation
        invocation.context.model_scan = self

    def model_forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """Hand the ordinary model call to its already selected region task."""
        if self._invocation is None:
            return self.original(input_ids, positions, intermediate_tensors, inputs_embeds)
        if intermediate_tensors is not None or positions.ndim != 1 or positions.numel() != self.length:
            raise ValueError("ModelScan needs one unsharded position per scheduled row")
        self.source.validate()
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.model.embed_input_ids(input_ids)
        self._positions, self._hidden, self._context = positions, inputs_embeds, get_forward_context()
        self.execution.execute_anchor(self._invocation, self.anchor)
        assert self._output is not None
        return self._output

    def forward(
        self,
        layer: int,
        start: int,
        rows: int,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        parameters: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Use genuine causal chunk metadata over the runner's original cache pages."""
        assert self._context is not None and self._positions is not None and self._hidden is not None
        stop = start + rows
        size, length = rows, self.row.token_offset + stop
        sequence = upload([0, size, length], self._hidden.device, torch.int32)
        metadata, slots = {}, {}
        prefill = self.row.phase is HiddenPhase.PREFILL
        for attention in (self.attentions[layer],):
            base, _, mapping = attention_metadata(self._context, attention)
            slot = mapping[start:stop]
            metadata[attention.layer_name] = replace(
                base,
                num_actual_tokens=size,
                max_query_len=size,
                query_start_loc=sequence[:2].to(base.query_start_loc.dtype),
                max_seq_len=length,
                seq_lens=sequence[2:].to(base.seq_lens.dtype),
                block_table=base.block_table[self.row.request_ordinal : self.row.request_ordinal + 1],
                slot_mapping=slot,
                num_decode_reqs=int(not prefill),
                num_prefill_reqs=int(prefill),
                num_decode_tokens=0 if prefill else size,
                num_prefill_tokens=size if prefill else 0,
                scheduler_metadata=None,
                prefix_scheduler_metadata=None,
            )
            slots[attention.layer_name] = slot
        context = replace(
            self._context, attn_metadata=metadata, slot_mapping=slots, batch_descriptor=None, is_padding=None
        )
        with override_forward_context(context):
            outputs = self.source.graphs[layer](
                *parameters, self._positions[start:stop], hidden[:rows], None if layer == 0 else residual[:rows]
            )
        self.observe_kv(layer, start, stop)
        return tuple(_pad(value, hidden.shape[0]) for value in outputs)

    def observe_kv(self, layer: int, start: int, stop: int) -> None:
        """Record each completed cell's causal prefix through asynchronous tracing."""
        assert self._invocation is not None and self._context is not None
        owner = self._invocation.context
        if owner.trace.observing:
            attention = self.attentions[layer]
            metadata, cache, _ = attention_metadata(self._context, attention)
            positions = torch.arange(self.row.token_offset + stop, device=cache.device)
            blocks = metadata.block_table[self.row.request_ordinal, positions // cache.shape[2]].long()
            owner.trace.observe(
                owner.invocation_id,
                TaskId(0),
                f"{layer}.kv",
                cache[blocks, :, positions % cache.shape[2], :],
                row=self.row,
                iteration=start // self.chunk_size,
            )

    def finish(self, outputs: tuple[torch.Tensor, ...], residuals: tuple[torch.Tensor, ...], chunk: int) -> None:
        """Normalize each valid chunk with the original module, then return packed rows."""
        self._output = torch.cat(
            tuple(
                self.source.suffix((h[: min(chunk, self.length - i * chunk)], r[: min(chunk, self.length - i * chunk)]))
                for i, (h, r) in enumerate(zip(outputs, residuals, strict=True))
            )
        )

    def clear(self) -> None:
        """Release references to this invocation's inputs and outputs."""
        self._rows, self._invocation, self._context = (), None, None
        self._positions = self._hidden = self._output = None

    def close(self) -> None:
        """Remove the model handoff and restore the original runner call."""
        self.clear()
        self.model.forward = self.original


def _pad(value: torch.Tensor, rows: int) -> torch.Tensor:
    """Extend only the region value layout; padded rows never reach model kernels."""
    return torch.nn.functional.pad(value, (0, 0, 0, rows - value.shape[0]))
