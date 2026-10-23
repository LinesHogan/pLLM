"""Eager Qwen2/FA2 current-slot replay; plans and parameter pins stay upstream."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Generic

import torch
from vllm.forward_context import ForwardContext, get_forward_context, override_forward_context
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

from pllm.authoring import DataT, ResultT, TensorStepInput
from pllm.backends.vllm.activation import VllmActivationPatch
from pllm.backends.vllm.overlay_graph import PinnedParameterOverlay
from pllm.backends.vllm.region_attention import (
    attention_paged_access,
    select_attention_metadata,
    validate_attention_resources,
)
from pllm.backends.vllm.region_kernels import VllmRmsNormKernel
from pllm.backends.vllm.region_layers import VllmDecoderBodies
from pllm.contracts.backend_attachment import CausalModelDomain
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch
from pllm.contracts.region_composition import RegionFunctionBody
from pllm.contracts.region_imports import NativeKernelImport, RegionTensorImport
from pllm.contracts.region_program import AccessMode, BorrowedResource, TensorType
from pllm.contracts.semantic_graph import AnchorId, ConditionalLayerReplay, SemanticTask, TokenwiseLayerReplay
from pllm.contracts.state_ref import StateRef
from pllm.execution.decoder_region import DecoderRegion
from pllm.execution.execution import TaskGraphInvocation
from pllm.execution.paged_access import PagedAccessPlan, PagedResourceView, validate_paged_views
from pllm.execution.ptts.page_bundle import PageBundleArena, PagePin
from pllm.execution.region_resources import (
    ExternalFunction,
    HostResourceBinding,
    PagedResourceBinding,
    PinnedResourceBinding,
    ReadonlyResourceBinding,
    RegionPageOwner,
)
from pllm.execution.region_streams import RegionStreamBindings
from pllm.execution.region_values import Data
from pllm.execution.replay import TOKENWISE_REPLAY_IMPLEMENTATION, TokenwiseReplayPrimitive, TokenwiseReplayTask
from pllm.execution.state_execution import StateExecution, StateTaskContext
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView

_DecoderForward = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True, slots=True)
class _ReplayInput:
    row: PackedPhaseRow
    positions: torch.Tensor
    carry: torch.Tensor
    metadata: dict[str, FlashAttentionMetadata]
    slots: tuple[tuple[torch.Tensor, int, int], ...]
    original_kv: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class _TokenReplayInput:
    context: ForwardContext
    backups: tuple[tuple[torch.Tensor, int, int, torch.Tensor], ...]


class VllmLayerReplay:
    """Re-runs the signed ascending layer range per selected token; the hook sits
    on the range's first layer only, deeper layers are replayed by direct call."""

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        effect: ConditionalLayerReplay,
        overlay: PinnedParameterOverlay,
        domain: CausalModelDomain,
    ) -> None:
        layers = model.model.layers
        if not effect.layers or tuple(sorted(set(effect.layers))) != effect.layers or effect.layers[-1] >= len(layers):
            raise ValueError("conditional replay requires a sorted unique in-range layer set")
        self.effect, self.overlay = effect, overlay
        self.layers = tuple(layers[index] for index in effect.layers)
        validate_attention_resources(
            tuple(layer.self_attn.attn for layer in self.layers), hidden_size=model.config.hidden_size
        )
        self.originals: tuple[_DecoderForward, ...] = tuple(layer.forward for layer in self.layers)
        tensor = TensorType(
            (1, model.config.hidden_size), str(model.model.norm.weight.dtype), str(model.model.norm.weight.device)
        )
        self.region = DecoderRegion(tensor, effect.layers, domain)
        self._owner: RegionPageOwner | None = None
        self._kv: PagedResourceBinding | None = None
        self._rows: tuple[PackedPhaseRow, ...] = ()
        self._all_rows: tuple[PackedPhaseRow, ...] = ()
        self._inputs: tuple[_ReplayInput, ...] = ()
        self._forward_context: ForwardContext | None = None
        self._materialized = False
        self.layers[0].forward = self.forward

    @property
    def rows(self) -> tuple[PackedPhaseRow, ...]:
        return self._rows

    def prepare(self, rows: PackedPhaseRowBatch, owner: RegionPageOwner) -> None:
        self.clear()
        self._owner = owner
        self._all_rows = rows.rows
        self._rows = tuple(
            row
            for index, row in enumerate(rows.rows)
            if index + 1 == len(rows.rows) or rows.rows[index + 1].identity != row.identity
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._rows:
            return self.originals[0](positions, hidden_states, residual)
        context = get_forward_context()
        access = attention_paged_access(context, tuple(layer.self_attn.attn for layer in self.layers), self._all_rows)
        if self._owner is None:
            raise RuntimeError("conditional KV binding requires its original invocation")
        view = PagedResourceView(
            self.region.source.kv,
            access,
            tuple(
                (ordinal, layer.self_attn.attn.layer_name)
                for ordinal, layer in zip(self.effect.layers, self.layers, strict=True)
            ),
        )
        validate_paged_views((view,), self._owner.arena.allocation_ranges)
        self._kv = PagedResourceBinding(view, self._owner)
        prepared: list[_ReplayInput] = []
        # Decoder residual addition is in-place; own only the selected token inputs.
        for row in self._rows:
            index = row.packed_row_index
            carry = hidden_states[index : index + 1].detach()
            carry = carry.clone() if residual is None else carry + residual[index : index + 1]
            position = positions[index : index + 1]
            length = int(position.item()) + 1
            sequence = torch.tensor([0, 1, length], dtype=torch.int32, device=carry.device)
            metadata: dict[str, FlashAttentionMetadata] = {}
            slots: list[tuple[torch.Tensor, int, int]] = []
            originals: list[torch.Tensor] = []
            for layer in self.layers:
                selected, cache, block, offset = select_attention_metadata(
                    context, layer.self_attn.attn, row, length, sequence, access
                )
                metadata[layer.self_attn.attn.layer_name] = selected
                slots.append((cache, block, offset))
                originals.append(cache[block, :, offset, :])
            prepared.append(_ReplayInput(row, position, carry, metadata, tuple(slots), tuple(originals)))
        output = self.originals[0](positions, hidden_states, residual)
        # Views follow the in-place KV writes; deeper layers write their slot only
        # after this hook returns. Clone at the first replay step — after the normal
        # forward finished, before any replay write — so restore returns the
        # successful normal token, not the pre-forward or replayed KV.
        self._inputs = tuple(prepared)
        self._forward_context = context
        return output

    def seed(self, row: PackedPhaseRow) -> torch.Tensor:
        return next(item.carry for item in self._inputs if item.row is row)

    def step(
        self, row: PackedPhaseRow, carry: torch.Tensor, arena: PageBundleArena, pin: PagePin, ref: StateRef
    ) -> torch.Tensor:
        if self._forward_context is None or self._owner is None or self._owner.arena is not arena or self._kv is None:
            raise RuntimeError("replay requires its original normal forward context")
        original_context = self._forward_context
        if not self._materialized:
            self._inputs = tuple(
                replace(item, original_kv=tuple(original.clone() for original in item.original_kv))
                for item in self._inputs
            )
            self._materialized = True
        item = next(item for item in self._inputs if item.row is row)

        def forward(
            index: int, positions: torch.Tensor, hidden: torch.Tensor, residual: torch.Tensor | None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            ordinal = self.effect.layers.index(index)
            layer = self.layers[ordinal]
            name = layer.self_attn.attn.layer_name
            metadata = item.metadata[name]
            context = replace(
                original_context,
                attn_metadata={name: metadata},
                slot_mapping={name: metadata.slot_mapping},
                batch_descriptor=None,
                is_padding=None,
            )
            self.overlay.prepare_replay(row, index, arena, pin, ref)
            try:
                with override_forward_context(context):
                    return self.originals[ordinal](positions, hidden, residual)
            finally:
                self.overlay.clear()

        hidden, residual = self.region.run(self._owner, row, item.positions, carry, forward, kv=self._kv)
        return hidden + residual

    def restore(self) -> None:
        for item in self._inputs:
            for (cache, block, offset), original in zip(item.slots, item.original_kv, strict=True):
                cache[block, :, offset, :].copy_(original)

    def clear(self) -> None:
        self._rows, self._inputs, self._forward_context = (), (), None
        self._all_rows = ()
        self._owner = None
        self._kv = None
        self._materialized = False

    def close(self) -> None:
        self.clear()
        self.layers[0].forward = self.originals[0]


@dataclass(frozen=True, slots=True)
class _TokenContextView(Generic[DataT, ResultT]):
    resource: BorrowedResource
    replay: VllmTokenwiseReplay[DataT, ResultT]
    namespace: int
    rows: tuple[PackedPhaseRow, ...]

    def __post_init__(self) -> None:
        if any(row.packed_row_index != i for i, row in enumerate(self.rows)):
            raise ValueError("context proof requires the original consecutive packed indices")
        self.validate()

    def validate(self) -> None:
        self.replay.validate_context(self.resource, self.namespace, self.rows)

    def check(self, mode: AccessMode, ranges: tuple[tuple[int, int], ...]) -> None:
        self.validate()
        if len(ranges) != 1 or not 0 <= ranges[0][0] <= ranges[0][1] <= len(self.replay.rows):
            raise ValueError("context access exceeds the original packed rows")
        if mode is AccessMode.READ:
            self.replay.check_prepared_context(ranges[0])


class VllmTokenwiseReplay(TokenwiseReplayPrimitive, Generic[DataT, ResultT]):
    """Model kernels bound to the original signed task and invocation."""

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
    ) -> None:
        if not isinstance(task.replay, TokenwiseLayerReplay) or task.replay.layer_count != len(model.model.layers):
            raise ValueError("tokenwise replay differs from the specialized model")
        validate_attention_resources(
            tuple(layer.self_attn.attn for layer in model.model.layers), hidden_size=model.config.hidden_size
        )
        self.model, self.effect = model.model, task.replay
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
                TokenwiseReplayTask[DataT]
                .from_implementation(execution.implementation(TOKENWISE_REPLAY_IMPLEMENTATION))
                .region
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
                execution.implementation(TOKENWISE_REPLAY_IMPLEMENTATION)
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
        self.original = self.model.forward
        self._rows: tuple[PackedPhaseRow, ...] = ()
        self._invocation: TaskGraphInvocation[StateTaskContext[DataT]] | None = None
        self._hidden: torch.Tensor | None = None
        self._positions: torch.Tensor | None = None
        self._output: torch.Tensor | None = None
        self._tokens: dict[PackedPhaseRow, _TokenReplayInput] = {}
        self._access: PagedAccessPlan | None = None
        self._auxiliary_streams: tuple[torch.cuda.Stream, ...] = ()
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
            row, request, token, ordinal, positions, hidden, *rest = inputs
            residual = rest[0] if rest else None
            if (
                type(row) is not int
                or type(request) is not int
                or type(token) is not int
                or type(ordinal) is not int
                or not isinstance(positions, torch.Tensor)
                or not isinstance(hidden, torch.Tensor)
                or residual is not None
                and not isinstance(residual, torch.Tensor)
                or self._access is None
            ):
                raise ValueError("attention operands differ from the prepared model source")
            original = self._rows[row]
            if (request, token) != (original.request_ordinal, original.token_offset) or original not in self._tokens:
                raise ValueError("attention requires its original token preparation")
            self._access.slot(self.model.layers[ordinal].self_attn.attn.layer_name, original)
            with override_forward_context(self._tokens[original].context):
                return provider.attention(ordinal, positions, hidden, residual)

        first, next_ = provider.definitions
        return (
            ExternalFunction(first.program.functions[0], attention),
            ExternalFunction(next_.program.functions[0], attention),
        )

    def borrow_streams(self, owner: RegionPageOwner, lanes: int) -> RegionStreamBindings:
        if self._invocation is None or self._invocation.context is not owner or owner.pin.stream is None:
            raise ValueError("model streams require their original CUDA invocation and pin")
        if type(lanes) is not int or not 2 <= lanes <= 8:
            raise ValueError("model stream binding requires two to eight physical lanes")
        stream = owner.pin.stream
        if any(auxiliary.device != stream.device for auxiliary in self._auxiliary_streams):
            raise ValueError("model auxiliary streams belong to another device")
        available = tuple(auxiliary for auxiliary in self._auxiliary_streams if auxiliary != stream)
        if len(available) < lanes - 1:
            added = tuple(torch.cuda.Stream(device=stream.device) for _ in range(lanes - 1 - len(available)))
            self._auxiliary_streams += added
            available += added
        result = RegionStreamBindings(stream, available[: lanes - 1])
        result.validate_available()
        return result

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
        return HostResourceBinding(_TokenContextView(resource, self, id(self._tokens), self._rows), owner)

    def bind_activation(self, resource: BorrowedResource, layer: int, owner: RegionPageOwner) -> PinnedResourceBinding:
        if self._invocation is None or owner is not self._invocation.context or self.activation is None:
            raise RuntimeError("activation fields require their original model invocation")
        return self.activation.bind_pinned(resource, layer, owner)

    def activation_selected(self, row: int, layer: int) -> bool:
        if self._invocation is None or self.activation is None or self._rows[row] not in self._tokens:
            raise RuntimeError("activation selection requires its prepared token")
        return self.activation.selected(row, layer)

    def validate_context(self, resource: BorrowedResource, namespace: int, rows: tuple[PackedPhaseRow, ...]) -> None:
        if (
            self._invocation is None
            or namespace != id(self._tokens)
            or rows is not self._rows
            or len(resource.domain) != 1
            or len(rows) > resource.domain[0]
        ):
            raise ValueError("context proof differs from its original packed registry")

    def check_prepared_context(self, span: tuple[int, int]) -> None:
        if any(self._rows[i] not in self._tokens for i in range(*span)):
            raise RuntimeError("context read requires its original prepared row")

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

    def begin(self, row: PackedPhaseRow) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, positions = self._hidden, self._positions
        if hidden is None or positions is None or self._output is not None or self._access is None:
            raise RuntimeError("tokenwise replay requires its original model inputs")
        if row in self._tokens:
            raise RuntimeError("tokenwise preparation repeated before commit")
        context = get_forward_context()
        index = row.packed_row_index
        position = positions[index : index + 1]
        length = int(position.item()) + 1
        if length != row.token_offset + 1:
            raise ValueError("tokenwise replay position differs from original packed identity")
        # Every layer consumes the same immutable singleton sequence, not a per-layer copy.
        sequence = torch.tensor([0, 1, length], dtype=torch.int32, device=hidden.device)
        metadata_by_layer: dict[str, FlashAttentionMetadata] = {}
        slots_by_layer: dict[str, torch.Tensor] = {}
        backups: list[tuple[torch.Tensor, int, int, torch.Tensor]] = []
        for layer in self.model.layers:
            attention = layer.self_attn.attn
            selected, cache, block, offset = select_attention_metadata(
                context, attention, row, length, sequence, self._access
            )
            metadata_by_layer[attention.layer_name] = selected
            slots_by_layer[attention.layer_name] = selected.slot_mapping
            backups.append((cache, block, offset, cache[block, :, offset, :].clone()))
        selected = replace(
            context,
            attn_metadata=metadata_by_layer,
            slot_mapping=slots_by_layer,
            batch_descriptor=None,
            is_padding=None,
        )
        self._tokens[row] = _TokenReplayInput(selected, tuple(backups))
        return position, hidden[index : index + 1]

    def layer(
        self,
        row: PackedPhaseRow,
        ordinal: int,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if row not in self._tokens or self._positions is None or self._invocation is None or self._access is None:
            raise RuntimeError("tokenwise layer requires its prepared token")
        self._access.slot(self.model.layers[ordinal].self_attn.attn.layer_name, row)
        with override_forward_context(self._tokens[row].context):
            return self.model.layers[ordinal].forward(positions, hidden, residual)

    def normalize(self, row: PackedPhaseRow, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if row not in self._tokens:
            raise RuntimeError("tokenwise normalization requires its original preparation")
        if self._normalize_kernel is not None:
            if self.model.norm is not self._normalize_kernel.norm:
                raise ValueError("model replaced the bound normalization module")
            return self._normalize_kernel.run(hidden, residual)
        normal, _ = self.model.norm.forward(hidden, residual)
        return normal

    def commit(self, row: PackedPhaseRow, record: Callable[[], None]) -> None:
        if row not in self._tokens:
            raise RuntimeError("tokenwise commit requires its original preparation")
        # A failed observation/record leaves this row's recovery data intact.
        # Successful rows release only their own metadata and backups.
        record()
        del self._tokens[row]

    def restore(self) -> None:
        for item in reversed(tuple(self._tokens.values())):
            for cache, block, offset, saved in item.backups:
                cache[block, :, offset, :].copy_(saved)
        self._tokens.clear()

    def finish(self, outputs: tuple[torch.Tensor, ...]) -> None:
        if self._tokens or len(outputs) != len(self._rows):
            raise RuntimeError("tokenwise outputs require every prepared row to commit")
        self._output = torch.cat(outputs, dim=0)
        self._access = None

    def clear(self) -> None:
        self._rows, self._invocation = (), None
        self._hidden, self._positions, self._output = None, None, None
        self._tokens.clear()
        self._access = None

    def close(self) -> None:
        for stream in self._auxiliary_streams:
            stream.synchronize()
        self._auxiliary_streams = ()
        self.clear()
        self.model.forward = self.original
