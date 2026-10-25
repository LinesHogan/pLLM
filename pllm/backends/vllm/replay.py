"""Qwen2/FA2 replay bindings and owned per-token CUDA capture for the region executor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Generic

import torch
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

from pllm.authoring import DataT, ResultT, TensorStepInput
from pllm.backends.vllm.activation import VllmActivationPatch
from pllm.backends.vllm.parameter_overlay import ExactParameterOverlay
from pllm.backends.vllm.region_attention import (
    TokenAttentionBuffers,
    attention_paged_access,
    validate_attention_resources,
)
from pllm.backends.vllm.region_kernels import VllmRmsNormKernel
from pllm.backends.vllm.region_layers import VllmDecoderBodies
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch
from pllm.contracts.region_composition import RegionFunctionBody
from pllm.contracts.region_imports import NativeKernelImport, RegionTensorImport
from pllm.contracts.region_program import AccessMode, BorrowedResource, TensorType
from pllm.contracts.semantic_graph import AnchorId, SemanticTask, TokenProgramEffect
from pllm.execution.codec import boundary_digest
from pllm.execution.execution import TaskGraphInvocation
from pllm.execution.paged_access import PagedAccessPlan, PagedResourceView
from pllm.execution.region_execution import RegionInvocation, RegionTokenCapture
from pllm.execution.region_resources import (
    ExternalFunction,
    HostResourceBinding,
    PagedResourceBinding,
    PinnedResourceBinding,
    ReadonlyResourceBinding,
    RegionPageOwner,
)
from pllm.execution.region_values import Data
from pllm.execution.replay import TokenwiseReplayPrimitive, TokenwiseReplayTask
from pllm.execution.state_execution import StateExecution, StateTaskContext
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView


@dataclass(frozen=True, slots=True)
class _TokenContextView(Generic[DataT, ResultT]):
    resource: BorrowedResource
    replay: VllmTokenwiseReplay[DataT, ResultT]
    rows: tuple[PackedPhaseRow, ...]

    @property
    def namespace(self) -> int:
        return id(self.replay)

    def validate(self) -> None:
        assert self.replay.rows is self.rows

    def check(self, mode: AccessMode, ranges: tuple[tuple[int, int], ...]) -> None:
        self.validate()


class VllmTokenwiseReplay(TokenwiseReplayPrimitive, Generic[DataT, ResultT]):
    """Model kernels bound to the original signed task and invocation."""

    token_capture = True

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        task: SemanticTask,
        execution: StateExecution[TensorStepInput, DataT, ResultT],
        anchor: AnchorId,
        *,
        source_imports: tuple[RegionTensorImport, ...] = (),
        function_bodies: tuple[RegionFunctionBody, ...] = (),
        activation: VllmActivationPatch | None = None,
        overlay: ExactParameterOverlay | None = None,
    ) -> None:
        if not isinstance(task.replay, TokenProgramEffect) or task.replay.layer_count != len(model.model.layers):
            raise ValueError("tokenwise replay differs from the specialized model")
        validate_attention_resources(
            tuple(layer.self_attn.attn for layer in model.model.layers), hidden_size=model.config.hidden_size
        )
        self.model, self.effect = model.model, task.replay
        self.projections = overlay
        self.implementation_id = next(
            item.implementation_id
            for item in execution.arena.plan.schedule_physical.tasks
            if execution.arena.plan.semantic_graph.tasks[item.task_id].symbol == task.symbol
        )
        self._capacity = (
            TokenwiseReplayTask[DataT]
            .from_implementation(execution.implementation(self.implementation_id))
            .region.domain.positions
        )
        self.activation = activation
        tensor = TensorType(
            (1, model.config.hidden_size), str(self.model.norm.weight.dtype), str(self.model.norm.weight.device)
        )
        self._normalize_kernel = (
            VllmRmsNormKernel.inspect(self.model.norm, tensor, len(self.model.layers)) if source_imports else None
        )
        self._layer_bodies: VllmDecoderBodies | None = None
        if function_bodies:
            source = (
                TokenwiseReplayTask[DataT].from_implementation(execution.implementation(self.implementation_id)).region
            )
            self._layer_bodies = VllmDecoderBodies.inspect(
                tuple(self.model.layers), tensor, source.domain, source.context.domain[0]
            )
        if self.function_bodies != function_bodies:
            raise ValueError("local model source differs from the signed backend definitions")
        local_imports = (*self.kernel_imports, *(() if self._layer_bodies is None else self._layer_bodies.imports))
        if local_imports != source_imports:
            raise ValueError("local model sources differ from the signed backend catalog")
        if self._layer_bodies is not None:
            implementation = TokenwiseReplayTask[DataT].from_implementation(
                execution.implementation(self.implementation_id)
            )
            missing = {
                function.name for function in implementation.region.program.functions
            } - implementation.region.functions.keys()
            local_functions = {
                name: function for name, function in self._layer_bodies.functions.items() if name in missing
            }
            execution.rebind_implementation(
                replace(implementation, region=implementation.region.bind_functions(local_functions))
            )
        self.execution, self.anchor = execution, anchor
        self._entitlement = next(
            fact for fact in execution.arena.plan.timeline.task_resources if fact.task == task.symbol
        )
        self.original = self.model.forward
        self._rows: tuple[PackedPhaseRow, ...] = ()
        self._invocation: TaskGraphInvocation[StateTaskContext[DataT]] | None = None
        self._hidden: torch.Tensor | None = None
        self._positions: torch.Tensor | None = None
        self._output: torch.Tensor | None = None
        self._captures: dict[tuple[bool, tuple[bool, ...]], RegionTokenCapture] = {}
        self._buffers: TokenAttentionBuffers | None = None
        self._identity = ""
        self._seed = torch.empty(tensor.shape, dtype=self.model.norm.weight.dtype, device=self.model.norm.weight.device)
        self._position = torch.empty(1, dtype=torch.int64, device=self._seed.device)
        self._activation_fields: dict[int, tuple[torch.Tensor, ...]] = {}
        self._activation_rows: tuple[tuple[tuple[torch.Tensor, ...], ...], ...] = ()
        self._access: PagedAccessPlan | None = None
        self._stream: torch.cuda.Stream | None = None
        self.model.forward = self.forward

    @property
    def kernel_imports(self) -> tuple[NativeKernelImport, ...]:
        return () if self._normalize_kernel is None else (self._normalize_kernel.implementation,)

    @property
    def function_bodies(self) -> tuple[RegionFunctionBody, ...]:
        return () if self._layer_bodies is None else self._layer_bodies.definitions

    def model_endpoints(self) -> tuple[ExternalFunction, ...]:
        provider = self._layer_bodies
        if provider is None:
            return ()
        provider.validate(tuple(self.model.layers))

        def attention(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
            _row, _request, _token, ordinal, positions, hidden, *rest = inputs
            residual = rest[0] if rest else None
            assert type(ordinal) is int and isinstance(positions, torch.Tensor) and isinstance(hidden, torch.Tensor)
            assert residual is None or isinstance(residual, torch.Tensor)
            return provider.attention(ordinal, positions, hidden, residual)

        first, next_ = provider.definitions
        return (
            ExternalFunction(first.program.functions[0], attention),
            ExternalFunction(next_.program.functions[0], attention),
        )

    @property
    def rows(self) -> tuple[PackedPhaseRow, ...]:
        return self._rows

    def bind_kv(self, resource: BorrowedResource, owner: RegionPageOwner) -> PagedResourceBinding:
        if self._invocation is None or owner is not self._invocation.context or self._access is None:
            raise RuntimeError("tokenwise KV proof requires its original model invocation")
        view = PagedResourceView(
            resource,
            self._access,
            tuple((ordinal, layer.self_attn.attn.layer_name) for ordinal, layer in enumerate(self.model.layers)),
        )
        return PagedResourceBinding(view, owner)

    def bind_weights(self, resource: BorrowedResource, owner: RegionPageOwner) -> ReadonlyResourceBinding:
        if self._invocation is None or owner is not self._invocation.context:
            raise RuntimeError("tokenwise weights require their original model invocation")
        modules = (*self.model.layers, self.model.norm)
        cells = tuple(
            ReadonlyTensorCell.bind((ordinal,), tuple(module.parameters())) for ordinal, module in enumerate(modules)
        )
        return ReadonlyResourceBinding(ReadonlyTensorView(resource, cells), owner)

    def bind_context(self, resource: BorrowedResource, owner: RegionPageOwner) -> HostResourceBinding:
        if self._invocation is None or owner is not self._invocation.context:
            raise RuntimeError("tokenwise context requires its original model invocation")
        return HostResourceBinding(_TokenContextView(resource, self, self._rows), owner)

    def bind_activation(self, resource: BorrowedResource, layer: int, owner: RegionPageOwner) -> PinnedResourceBinding:
        if self._invocation is None or owner is not self._invocation.context or self.activation is None:
            raise RuntimeError("activation fields require their original model invocation")
        return self.activation.bind_pinned(resource, layer, owner)

    def activation_selected(self, row: int, layer: int) -> bool:
        assert self.activation is not None
        return bool(
            self._activation_rows[row][tuple(item.binding.layer for item in self.activation.layers).index(layer)]
        )

    def prepare(self, rows: PackedPhaseRowBatch, invocation: TaskGraphInvocation[StateTaskContext[DataT]]) -> None:
        self.clear()
        self._rows, self._invocation = rows.rows, invocation

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if self._invocation is None:
            return self.original(input_ids, positions, intermediate_tensors, inputs_embeds)
        if intermediate_tensors is not None or positions.ndim != 1 or positions.numel() != len(self._rows):
            raise ValueError("tokenwise replay requires one original position per packed row")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("tokenwise replay requires model input IDs or embeddings")
            inputs_embeds = self.model.embed_input_ids(input_ids)
        self._hidden, self._positions = inputs_embeds, positions
        self._access = attention_paged_access(
            get_forward_context(), tuple(layer.self_attn.attn for layer in self.model.layers), self._rows
        )
        self.execution.execute_anchor(self._invocation, self.anchor)
        if self._output is None:
            raise RuntimeError("signed tokenwise task did not return normal hidden states")
        return self._output

    def run_region(self, invocation: RegionInvocation) -> tuple[torch.Tensor, ...]:
        """Stage packed rows and submit source-derived token graphs on the owner stream."""
        assert self._hidden is not None and self._positions is not None and self._invocation is not None
        invocation.overlay = self.overlay
        bindings = self._prepare_attention(invocation)
        self._prepare_activations(invocation)
        signatures = tuple(
            (
                enabled,
                (tuple(bool(fields) for fields in self._activation_rows[index]) if self._activation_rows else ())
                + (() if self.overlay is None else (self._rows[index].phase.value == "decode",)),
            )
            for index, enabled in enumerate(invocation.eligible)
        )
        outputs = torch.empty_like(self._hidden)
        for index, (row, signature) in enumerate(zip(self._rows, signatures, strict=True)):
            invocation.pages.before_statement()
            self._seed.copy_(self._hidden[index : index + 1])
            self._position.copy_(self._positions[index : index + 1])
            TokenAttentionBuffers.refresh(bindings, row)
            if self.activation is not None:
                for handle, fields in zip(self.activation.layers, self._activation_rows[index], strict=True):
                    for target, field in zip(self._activation_fields.get(handle.binding.layer, ()), fields):
                        target.copy_(field)
            if self.overlay is not None:
                assert self.projections is not None
                self.overlay.stage(index, signature[0])
                self.projections.prepare_token(row, self.overlay)
            capture = self._captures.get(signature) or self._capture_token(invocation, row, signature)
            capture.graph.replay()
            if self.overlay is not None:
                self.overlay.collect(capture.overlay_results, capture.overlay_pending)
            outputs[index : index + 1].copy_(capture.output)
            if signature[0]:
                assert capture.observation is not None
                invocation.observe(index, capture.observation)
                invocation.record(index)
        return tuple(outputs[index : index + 1] for index in range(len(self._rows)))

    def _prepare_attention(
        self, invocation: RegionInvocation
    ) -> tuple[tuple[FlashAttentionMetadata, torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        """Bind stable metadata and invalidate graphs when their source or backing changes."""
        assert self._hidden is not None
        attentions = tuple(layer.self_attn.attn for layer in self.model.layers)
        context = get_forward_context()
        stream = torch.cuda.current_stream(self._hidden.device)
        if self._stream is not None and self._stream != stream:
            stream.wait_stream(self._stream)
        self._stream = stream
        identity = boundary_digest(
            invocation.execution.capture_signature,
            self.execution.arena.plan.semantic_graph.execution_parameters_digest,
            None if self.overlay_storage is None else self.overlay_storage.slab.data_ptr(),
            tuple(
                (tensor.data_ptr(), tuple(tensor.shape), tensor.stride(), str(tensor.dtype), str(tensor.device))
                for tensor in (*tuple(item.kv_cache for item in attentions), *tuple(self.model.parameters()))
            ),
        )
        if identity != self._identity:
            stream.synchronize()
            self._captures.clear()
            self._buffers = TokenAttentionBuffers(context, attentions, self._capacity)
            self._identity = identity
        assert self._buffers is not None
        lengths = torch.tensor(
            [row.token_offset + 1 for row in self._rows], dtype=torch.int32, device=self._hidden.device
        )
        bindings = self._buffers.bind(context, attentions, lengths)
        return bindings

    def _prepare_activations(self, invocation: RegionInvocation) -> None:
        """Bind exact pins once and route captured field reads through owned staging."""
        self._activation_rows = () if self.activation is None else self.activation.prepare_token_rows()
        if self.activation is not None:
            for ordinal, handle in enumerate(self.activation.layers):
                fields = next((items[ordinal] for items in self._activation_rows if items[ordinal]), ())
                if fields and handle.binding.layer not in self._activation_fields:
                    self._activation_fields[handle.binding.layer] = tuple(torch.empty_like(field) for field in fields)
        invocation.pages.stage_token_tensors(
            {f"activation.{layer}": fields for layer, fields in self._activation_fields.items()}
        )

    def _capture_token(
        self, invocation: RegionInvocation, row: PackedPhaseRow, signature: tuple[bool, tuple[bool, ...]]
    ) -> RegionTokenCapture:
        """Capture one source variant, restoring only its cold writable KV slots."""
        assert self._buffers is not None and self._hidden is not None
        instances = invocation.execution.prepare_token_body(
            invocation, row.request_ordinal, row.token_offset - invocation.sizes[f"prefix{row.request_ordinal}"]
        )
        assert self._access is not None
        attentions = tuple(layer.self_attn.attn for layer in self.model.layers)
        slots = tuple((attention.kv_cache, *self._access.slot(attention.layer_name, row)) for attention in attentions)
        saved = tuple(cache[block, :, offset, :].clone() for cache, block, offset in slots)
        try:
            with override_forward_context(self._buffers.context):
                capture = RegionTokenCapture(invocation, instances, self._hidden.device)
        finally:
            for (cache, block, offset), value in zip(slots, saved, strict=True):
                cache[block, :, offset, :].copy_(value)
        if self.overlay is not None:
            self.overlay.reset()
        resident = sum(item.resident_bytes for item in self._captures.values()) + capture.resident_bytes
        if resident > self._entitlement.resident_bytes:
            raise ValueError(
                f"token capture needs {resident} resident bytes; increase capture_memory_bytes above {self._entitlement.resident_bytes}"
            )
        self._captures[signature] = capture
        return capture

    def begin(self, row: PackedPhaseRow) -> tuple[torch.Tensor, torch.Tensor]:
        return self._position, self._seed

    def layer(
        self,
        row: PackedPhaseRow,
        ordinal: int,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.layers[ordinal].forward(positions, hidden, residual)

    def normalize(self, row: PackedPhaseRow, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if self._normalize_kernel is not None:
            if self.model.norm is not self._normalize_kernel.norm:
                raise ValueError("model replaced the bound normalization module")
            return self._normalize_kernel.run(hidden, residual)
        normal, _ = self.model.norm.forward(hidden, residual)
        return normal

    def commit(self, row: PackedPhaseRow, record: Callable[[], None]) -> None:
        record()

    def finish(self, outputs: tuple[torch.Tensor, ...]) -> None:
        if len(outputs) != len(self._rows):
            raise RuntimeError("tokenwise outputs require every prepared row to commit")
        self._output = torch.cat(outputs, dim=0)
        self._access = None

    def clear(self) -> None:
        self._rows, self._invocation = (), None
        self._hidden, self._positions, self._output = None, None, None
        self._activation_rows = ()
        self._access = None

    def close(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()
        self._captures.clear()
        self.clear()
        self.model.forward = self.original
