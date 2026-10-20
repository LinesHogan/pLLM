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
from pllm.contracts.semantic_graph import AnchorId, TaskId
from pllm.execution.cuda_stream import upload
from pllm.execution.execution import TaskGraphInvocation
from pllm.execution.module_graph import ModuleCallBinding, capture_module_calls
from pllm.execution.state_execution import StateExecution, StateTaskContext


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


def _adapter_layer(layer: Qwen2DecoderLayer, scale: float) -> GraphModule:
    source = capture_module_calls(layer, none_inputs=("residual",))
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
    """The original complete model graph with one explicit projection effect."""

    model: Qwen2Model
    graph: GraphModule
    bindings: tuple[ModuleCallBinding, ...]

    @classmethod
    def capture(cls, model: Qwen2Model, scale: float) -> "ModelScanSource":
        """Retain original layer order, embedding and final normalization."""
        bindings = tuple(ModuleCallBinding.bind(module) for module in tuple(model.modules())[1:])
        source = capture_module_calls(model, none_inputs=("input_ids", "intermediate_tensors"))
        source.add_module("_adapter_layer", _adapter_layer(model.layers[0], scale))
        graph = torch.fx.Graph()
        mapped: dict[Node, Node] = {}
        a, b = graph.placeholder("adapter_a"), graph.placeholder("adapter_b")
        operands: tuple[Node, Node] | None = None
        for node in source.graph.nodes:
            if node.op == "call_module" and node.target == "layers.0":
                args = torch.fx.map_arg(node.args, mapped.__getitem__)
                call = graph.call_module("_adapter_layer", (a, b, *args))
                mapped[node] = call
                operands = (
                    graph.call_function(operator.getitem, (call, 2)),
                    graph.call_function(operator.getitem, (call, 3)),
                )
            elif node.op == "output":
                assert operands is not None
                graph.output((torch.fx.map_arg(node.args[0], mapped.__getitem__), *operands))
            else:
                mapped[node] = graph.node_copy(node, mapped.__getitem__)
        graph.lint()
        return cls(model, GraphModule(source, graph), bindings)

    def validate(self) -> None:
        """Check that the source still calls the same original modules."""
        for binding in self.bindings:
            binding.validate(binding.module)


class VllmModelScan(Generic[DataT, ResultT]):
    """Model input/output handoff; the signed task owns all adapter advancement."""

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        execution: StateExecution[TensorStepInput, DataT, ResultT],
        anchor: AnchorId,
        scale: float,
        chunk_size: int,
    ) -> None:
        self.model, self.execution, self.anchor = model.model, execution, anchor
        self.chunk_size = chunk_size
        self.source = ModelScanSource.capture(self.model, scale)
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

    @property
    def weight(self) -> torch.Tensor:
        """Borrow the original frozen projection for local reconstruction."""
        return self.model.layers[0].mlp.down_proj.weight.detach()

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
        self, start: int, stop: int, parameters: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use genuine causal chunk metadata over the runner's original cache pages."""
        assert self._context is not None and self._positions is not None and self._hidden is not None
        size, length = stop - start, self.row.token_offset + stop
        sequence = upload([0, size, length], self._hidden.device, torch.int32)
        metadata, slots = {}, {}
        prefill = self.row.phase is HiddenPhase.PREFILL
        for attention in self.attentions:
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
            hidden, x, target = self.source.graph(
                *parameters, None, self._positions[start:stop], None, self._hidden[start:stop]
            )
        self.observe_kv(start, stop)
        return hidden, x, target

    def observe_kv(self, start: int, stop: int) -> None:
        """Expose complete causal prefixes through the existing opt-in tensor trace."""
        assert self._invocation is not None and self._context is not None
        owner = self._invocation.context
        if not owner.trace.observing:
            return
        length = self.row.token_offset + stop
        for ordinal, attention in enumerate(self.attentions):
            metadata, cache, _ = attention_metadata(self._context, attention)
            positions = torch.arange(length, device=cache.device)
            blocks = metadata.block_table[self.row.request_ordinal, positions // cache.shape[2]].long()
            values = cache[blocks, :, positions % cache.shape[2], :]
            owner.trace.observe(
                owner.invocation_id,
                TaskId(0),
                f"kv.{ordinal}",
                values,
                row=self.row,
                iteration=start // self.chunk_size,
            )

    def finish(self, outputs: tuple[torch.Tensor, ...]) -> None:
        """Return ordinary packed hidden states to the runner's unchanged sampler."""
        self._output = torch.cat(outputs, dim=0)

    def clear(self) -> None:
        """Release references to this invocation's inputs and outputs."""
        self._rows, self._invocation, self._context = (), None, None
        self._positions = self._hidden = self._output = None

    def close(self) -> None:
        """Remove the model handoff and restore the original runner call."""
        self.clear()
        self.model.forward = self.original
