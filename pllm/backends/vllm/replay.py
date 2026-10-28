"""Direct batched layer-route execution under vLLM's own model and FA2 context."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Generic

import torch
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

from pllm.authoring import DataT, ResultT
from pllm.backends.vllm.activation import VllmActivationPatch
from pllm.backends.vllm.parameter_overlay import ExactParameterOverlay
from pllm.backends.vllm.position_state import PositionCacheLayer, PositionCacheProvider
from pllm.backends.vllm.region_attention import (
    BatchAttentionBuffers,
    attention_metadata,
    attention_mode,
    attention_paged_access,
    validate_attention_resources,
)
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch
from pllm.contracts.semantic_graph import AnchorId, SemanticTask, TokenProgramEffect
from pllm.execution.activation import low_rank_affine_delta
from pllm.execution.cuda_stream import PinnedStaging, upload
from pllm.execution.overlay_task import RouteOperand, RouteOverlay
from pllm.execution.replay import TokenwiseReplayPrimitive, TokenwiseReplayTask
from pllm.execution.route_executor import execute_route, ordinary_pass
from pllm.execution.state_execution import StateExecution, StateTaskContext
from pllm.execution.step_capture import capture_graph
from pllm.execution.step_input import TensorStepInput
from pllm.execution.tensor_frame import OverlayTargetResult
from pllm.execution.token_program import KV, Save, TokenProgram

Signature = tuple[bool, tuple[bool, ...], bool]


class RouteWave(Generic[DataT, ResultT]):
    """Stable inputs, page indices and captured outputs for an exact active width."""

    def __init__(self, replay: VllmTokenwiseReplay[DataT, ResultT], width: int, signature: Signature) -> None:
        self.replay, self.signature = replay, signature
        tensor = replay.model.norm.weight
        self.seed = tensor.new_empty((width, tensor.numel()))
        self.positions = torch.empty(width, device=tensor.device, dtype=torch.long)
        self.staging = PinnedStaging()
        self.indices = torch.empty((3, width), device=tensor.device, dtype=torch.long)
        self.attention = BatchAttentionBuffers(get_forward_context(), replay.attentions, replay.domain.positions, width)
        self.position = PositionCacheProvider(
            replay.position_layers,
            tuple(value for _, value in replay.effect.position_states),
            replay.allocation,
            self.seed,
        )
        self.fields: dict[int, tuple[torch.Tensor, ...]] = {}
        if replay.activation is not None:
            for ordinal, handle in enumerate(replay.activation.layers):
                if signature[1][ordinal]:
                    fields = next(items[ordinal] for items in replay.activation_rows if items[ordinal])
                    self.fields[handle.binding.layer] = tuple(
                        field.new_empty((width, *field.shape)) for field in fields
                    )
        self.routes: tuple[RouteOverlay, ...] = ()
        self.graph: torch.cuda.CUDAGraph | None = None
        self.output = self.observation = self.seed
        self.results: tuple[tuple[tuple[OverlayTargetResult, ...], tuple[RouteOperand, ...]], ...] = ()
        self.lanes = tuple(torch.tensor([i], device=tensor.device) for i in range(width))

    def stage(self, rows: tuple[PackedPhaseRow, ...]) -> None:
        """Queue complete wave input copies without reading any device scalar."""
        replay = self.replay
        assert replay.hidden is not None and replay.positions is not None
        host = self.staging.take(self.indices.numel()).view_as(self.indices)
        host.copy_(
            torch.tensor(
                [(row.packed_row_index, row.request_ordinal, int(row.phase.value == "decode")) for row in rows]
            ).T
        )
        self.staging.copy_to(self.indices)
        packed, requests, generated = self.indices
        self.seed.copy_(replay.hidden.index_select(0, packed))
        self.positions.copy_(replay.positions.index_select(0, packed))
        bindings = self.attention.bind(get_forward_context(), replay.attentions, replay.positions + 1)
        self.attention.refresh(bindings, packed, requests)
        self.position.stage(get_forward_context(), packed, requests, self.positions, generated)
        if replay.activation is not None:
            for ordinal, handle in enumerate(replay.activation.layers):
                for column, target in enumerate(self.fields.get(handle.binding.layer, ())):
                    for lane, row in enumerate(rows):
                        target[lane].copy_(replay.activation_rows[row.packed_row_index][ordinal][column])
        if replay.overlay is not None:
            parent = replay.overlay
            if not self.routes:
                self.routes = tuple(RouteOverlay(parent.math, parent.batch, parent.entry, parent.data) for _ in rows)
            for route, row in zip(self.routes, rows, strict=True):
                route.batch, route.entry, route.data = parent.batch, parent.entry, parent.data
                route.ordinary = []
                route.stage(row.packed_row_index, self.signature[0])
            assert replay.projections is not None
            replay.projections.prepare_wave(rows, self.routes, self.lanes)

    def layer(
        self, ordinal: int, hidden: torch.Tensor, residual: torch.Tensor | None, kv: KV = "commit"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Call the intact native layer and apply any declared post-layer patch."""
        with attention_mode(self.replay.attentions[ordinal], self.positions, kv):
            hidden, residual = self.replay.model.layers[ordinal](self.positions, hidden, residual)
        fields = self.fields.get(ordinal, ())
        if fields:
            if len(fields) == 1:
                hidden = hidden + fields[0].to(hidden.dtype)
            else:
                weight, bias, rotation = fields
                inputs = hidden.unsqueeze(1)
                delta = (
                    torch.bmm(inputs, weight.transpose(1, 2)).squeeze(1) + bias - torch.bmm(inputs, rotation).squeeze(1)
                )
                hidden = hidden + torch.bmm(delta.unsqueeze(1), rotation.transpose(1, 2)).squeeze(1)
        assert residual is not None
        return hidden, residual

    def body(self) -> torch.Tensor:
        """All route boundaries, including gate changes and state stores, are captured."""
        for route in self.routes:
            route.reset()
        generated, *entries = self.position.entry(())
        if self.signature[0]:
            hidden, saved = execute_route(
                self.replay.program["decode" if self.signature[2] else "prefill"],
                self.seed,
                self.positions,
                generated,
                tuple(entries),
                self.layer,
                self.apply_updates,
            )
        else:
            hidden, saved = ordinary_pass(self.seed, self.replay.effect.layer_count, self.layer), tuple(entries)
        self.position.exit(saved)
        self.observation = hidden
        self.results = tuple((tuple(route.results), route.captures()) for route in self.routes)
        return self.replay.model.norm(hidden)

    def apply_updates(self) -> None:
        for route in self.routes:
            route.apply_updates()

    def capture(self) -> None:
        """Restore writable KV and PositionState pages after every cold execution."""
        cells = tuple(
            (attention.kv_cache, self.attention.metadata[attention.layer_name].slot_mapping)
            for attention in self.replay.attentions
        )
        saved = tuple(cache[slots // cache.shape[2], :, slots % cache.shape[2], :].clone() for cache, slots in cells)
        self.position.backup()

        def restore() -> None:
            for (cache, slots), value in zip(cells, saved, strict=True):
                cache[slots // cache.shape[2], :, slots % cache.shape[2], :] = value
            self.position.reset()

        try:
            with override_forward_context(self.attention.context):
                self.graph, self.output, resident = capture_graph(
                    self.seed.device, self.body, restore=restore, pool=self.replay.pool
                )
            if resident > self.replay.resident_bytes:
                raise ValueError(
                    f"route graphs need {resident} bytes; increase capture_memory_bytes above {self.replay.resident_bytes}"
                )
        finally:
            restore()


class VllmTokenwiseReplay(TokenwiseReplayPrimitive, Generic[DataT, ResultT]):
    """Run ordinary spans natively and dependent route positions in cross-request waves."""

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        task: SemanticTask,
        execution: StateExecution[TensorStepInput, DataT, ResultT],
        anchor: AnchorId,
        *,
        activation: VllmActivationPatch | None = None,
        overlay: ExactParameterOverlay | None = None,
        position_layers: tuple[PositionCacheLayer, ...] = (),
        allocation_block_size: int = 16,
    ) -> None:
        assert isinstance(task.replay, TokenProgramEffect)
        self.model, self.effect = model.model, task.replay
        self.execution, self.anchor, self.activation, self.projections = execution, anchor, activation, overlay
        implementation = next(
            item
            for item in execution.arena.plan.schedule.tasks
            if execution.arena.plan.semantic_graph.tasks[item.task_id].symbol == task.symbol
        )
        binding = TokenwiseReplayTask[DataT].from_implementation(
            execution.implementation(implementation.implementation_id)
        )
        self.program, self.domain = binding.program, binding.domain
        self.resident_bytes = next(
            fact.resident_bytes for fact in execution.arena.plan.timeline.task_resources if fact.task == task.symbol
        )
        self.position_layers, self.allocation = position_layers, allocation_block_size
        self.position = PositionCacheProvider(
            position_layers,
            tuple(value for _, value in self.effect.position_states),
            self.allocation,
            self.model.norm.weight.reshape(1, -1),
        )
        self.attentions = tuple(layer.self_attn.attn for layer in self.model.layers)
        validate_attention_resources(self.attentions, hidden_size=model.config.hidden_size)
        self.original = self.model.forward
        self._rows: tuple[PackedPhaseRow, ...] = ()
        self.context: StateTaskContext[DataT] | None = None
        self.hidden: torch.Tensor | None = None
        self.positions: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.observations: torch.Tensor | None = None
        self.captures: dict[tuple[int, Signature], RouteWave[DataT, ResultT]] = {}
        self.activation_rows: tuple[tuple[tuple[torch.Tensor, ...], ...], ...] = ()
        self.pool = torch.cuda.graph_pool_handle()
        self.identity: tuple[int, ...] = ()
        self.stream: torch.cuda.Stream | None = None
        self.model.forward = self.forward

    @property
    def rows(self) -> tuple[PackedPhaseRow, ...]:
        return self._rows

    def prepare(self, rows: PackedPhaseRowBatch, context: StateTaskContext[DataT]) -> None:
        self.clear()
        self._rows, self.context = rows.rows, context

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if self.context is None:
            return self.original(input_ids, positions, intermediate_tensors, inputs_embeds)
        if intermediate_tensors is not None or positions.ndim != 1 or positions.numel() != len(self.rows):
            raise ValueError("layer routes need one position per packed row without pipeline parallelism")
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.model.embed_input_ids(input_ids)
        self.hidden, self.positions = inputs_embeds, positions
        attention_paged_access(get_forward_context(), self.attentions, self.rows)
        self.execution.execute_anchor(self.context, self.anchor)
        assert self.output is not None
        return self.output

    def run(
        self, program: dict[str, TokenProgram], effect: TokenProgramEffect, observe: Callable[[int, torch.Tensor], None]
    ) -> None:
        assert self.hidden is not None and self.positions is not None
        stream = torch.cuda.current_stream(self.hidden.device)
        if self.stream is not None and self.stream != stream:
            stream.wait_stream(self.stream)
        self.stream = stream
        identity = tuple(
            tensor.data_ptr()
            for tensor in (
                *tuple(self.model.parameters()),
                *(item.kv_cache for item in self.attentions),
                *(item.kv_cache for item in self.position_layers),
            )
        )
        identity += BatchAttentionBuffers.mapping_groups(get_forward_context(), self.attentions)
        if identity != self.identity:
            stream.synchronize()
            self.captures.clear()
            self.identity = identity
        self.activation_rows = () if self.activation is None else self.activation.prepare_token_rows()
        self.output = torch.empty_like(self.hidden)
        self.observations = torch.empty_like(self.hidden)
        ordinary = tuple(
            (effect.rows != "all" and row.phase.value != effect.rows)
            or (
                row.phase.value == "prefill"
                and program[row.phase.value].prompt_pass_is_identity
                and self.overlay is None
            )
            for row in self.rows
        )
        requests = [
            list(row for row in self.rows if row.request_ordinal == request)
            for request in dict.fromkeys(row.request_ordinal for row in self.rows)
        ]
        while any(requests):
            spans: dict[TokenProgram | None, list[PackedPhaseRow]] = {}
            groups: dict[Signature, list[PackedPhaseRow]] = {}
            for request in requests:
                while request and (
                    ordinary[request[0].packed_row_index] or program[request[0].phase.value].over == "span"
                ):
                    row = request.pop(0)
                    spans.setdefault(None if ordinary[row.packed_row_index] else program[row.phase.value], []).append(
                        row
                    )
                if request:
                    row = request.pop(0)
                    signature = (
                        effect.rows == "all" or row.phase.value == effect.rows,
                        tuple(bool(fields) for fields in self.activation_rows[row.packed_row_index])
                        if self.activation_rows
                        else (),
                        row.phase.value == "decode",
                    )
                    groups.setdefault(signature, []).append(row)
            for selected_program, span in spans.items():
                self.span(tuple(span), selected_program)
            for signature, rows in groups.items():
                selected = tuple(rows)
                key = len(rows), signature
                if key not in self.captures:
                    self.captures[key] = RouteWave(self, len(rows), signature)
                captured = self.captures[key]
                captured.stage(selected)
                if captured.graph is None:
                    captured.capture()
                assert captured.graph is not None
                captured.graph.replay()
                self.output.index_copy_(0, captured.indices[0], captured.output)
                self.observations.index_copy_(0, captured.indices[0], captured.observation)
                for route, (results, pending) in zip(captured.routes, captured.results, strict=True):
                    route.collect(results, pending)
                    assert self.overlay is not None
                    self.overlay.ordinary.extend(route.ordinary)
        for row in self.rows:
            if effect.rows == "all" or row.phase.value == effect.rows:
                observe(row.packed_row_index, self.observations[row.packed_row_index : row.packed_row_index + 1])

    def span(self, rows: tuple[PackedPhaseRow, ...], program: TokenProgram | None = None) -> None:
        """One native varlen pass, committing the last touched position of every page."""
        assert self.hidden is not None and self.positions is not None and self.output is not None
        context = get_forward_context()
        packed = upload(tuple(row.packed_row_index for row in rows), self.hidden.device)
        groups = tuple(
            tuple(row for row in rows if row.request_ordinal == request)
            for request in dict.fromkeys(row.request_ordinal for row in rows)
        )
        requests = upload(tuple(group[0].request_ordinal for group in groups), self.hidden.device)
        lengths = tuple(len(group) for group in groups)
        starts = [0]
        for length in lengths:
            starts.append(starts[-1] + length)
        if rows != self.rows:
            metadata: dict[str, FlashAttentionMetadata] = {}
            for attention in self.attentions:
                source, _, slots = attention_metadata(context, attention)
                metadata[attention.layer_name] = replace(
                    source,
                    num_actual_tokens=len(rows),
                    max_query_len=max(lengths),
                    query_start_loc=upload(starts, self.hidden.device, torch.int32),
                    seq_lens=upload(
                        tuple(group[-1].token_offset + 1 for group in groups), self.hidden.device, torch.int32
                    ),
                    block_table=source.block_table.index_select(0, requests),
                    slot_mapping=slots.index_select(0, packed),
                    num_decode_reqs=0,
                    num_decode_tokens=0,
                    num_prefill_reqs=len(groups),
                    num_prefill_tokens=len(rows),
                    scheduler_metadata=None,
                    prefix_scheduler_metadata=None,
                )
            context = replace(
                context,
                attn_metadata=metadata,
                slot_mapping={name: item.slot_mapping for name, item in metadata.items()},
                batch_descriptor=None,
                is_padding=None,
            )
        routes: tuple[RouteOverlay, ...] = ()
        if self.overlay is not None:
            parent = self.overlay
            routes = tuple(RouteOverlay(parent.math, parent.batch, parent.entry, parent.data) for _ in groups)
            for route, group in zip(routes, groups, strict=True):
                route.stage(group[0].packed_row_index, program is not None)
                route.reset()
            assert self.projections is not None
            self.projections.prepare_wave(
                tuple(group[0] for group in groups),
                routes,
                tuple(upload(tuple(range(start, stop)), self.hidden.device) for start, stop in zip(starts, starts[1:])),
            )
        positions = self.positions.index_select(0, packed)

        def layer(
            ordinal: int, hidden: torch.Tensor, residual: torch.Tensor | None, kv: KV
        ) -> tuple[torch.Tensor, torch.Tensor]:
            with attention_mode(self.attentions[ordinal], positions, kv):
                hidden, residual = self.model.layers[ordinal](positions, hidden, residual)
            if self.activation is not None:
                for index, handle in enumerate(self.activation.layers):
                    if handle.binding.layer == ordinal:
                        for lane, row in enumerate(rows):
                            fields = self.activation_rows[row.packed_row_index][index]
                            if fields:
                                delta = (
                                    fields[0].to(hidden.dtype)
                                    if len(fields) == 1
                                    else low_rank_affine_delta(hidden[lane : lane + 1], *fields)
                                )
                                hidden[lane : lane + 1].add_(delta)
            assert residual is not None
            return hidden, residual

        def apply_updates() -> None:
            for route in routes:
                route.apply_updates()

        seed = self.hidden.index_select(0, packed)
        saved: tuple[torch.Tensor, ...] = ()
        with override_forward_context(context):
            if program is None:
                hidden = ordinary_pass(seed, self.effect.layer_count, layer)
            else:
                hidden, saved = execute_route(
                    program,
                    seed,
                    positions,
                    upload(tuple(row.phase.value == "decode" for row in rows), seed.device, torch.bool).reshape(-1, 1),
                    tuple(torch.empty_like(seed) for _ in program.entries),
                    layer,
                    apply_updates,
                )
            self.output.index_copy_(0, packed, self.model.norm(hidden))
            assert self.observations is not None
            self.observations.index_copy_(0, packed, hidden)
        for route in routes:
            route.collect(tuple(route.results), route.captures())
            assert self.overlay is not None
            self.overlay.ordinary.extend(route.ordinary)
        self.save_span(
            rows,
            groups,
            requests,
            hidden,
            {
                name: value
                for (name, _), value in zip(() if program is None else program.entries, saved, strict=True)
                if program is not None and any(isinstance(step, Save) and step.name == name for step in program.steps)
            },
        )

    def save_span(
        self,
        rows: tuple[PackedPhaseRow, ...],
        groups: tuple[tuple[PackedPhaseRow, ...], ...],
        requests: torch.Tensor,
        hidden: torch.Tensor,
        saved_values: dict[str, torch.Tensor],
    ) -> None:
        """Snapshot predecessor values before scattering page tails in a native span."""
        if not self.position_layers:
            return
        provider = PositionCacheProvider(
            self.position_layers,
            tuple(value for _, value in self.effect.position_states),
            self.allocation,
            hidden[: len(groups)],
        )
        starts = upload(tuple(group[0].packed_row_index for group in groups), hidden.device)
        offsets = upload(tuple(group[0].token_offset for group in groups), hidden.device)
        provider.stage(get_forward_context(), starts, requests, offsets, offsets.new_zeros(offsets.shape))
        entries = provider.entry(())[1:]
        tails: dict[tuple[int, int], int] = {}
        saved: dict[int, int] = {}
        sources: dict[tuple[int, int], int] = {}
        for index, row in enumerate(rows):
            page = row.request_ordinal, row.token_offset // self.allocation
            tails[page] = index
            if row.phase.value == "prefill" and self.effect.rows in ("all", "prefill"):
                saved[row.request_ordinal] = index
            sources[page] = saved.get(row.request_ordinal, -1)
        indices = upload(tuple(sources.values()), hidden.device)
        packed = upload(tuple(rows[index].packed_row_index for index in tails.values()), hidden.device)
        request_indices = upload(
            tuple(
                tuple(group[0].request_ordinal for group in groups).index(rows[index].request_ordinal)
                for index in tails.values()
            ),
            hidden.device,
        )
        context = get_forward_context()
        assert isinstance(context.slot_mapping, dict)
        program = self.program.get("prefill")
        for cache, (name, _), entry in zip(self.position_layers, self.effect.position_states, entries, strict=True):
            slots = context.slot_mapping[cache.layer_name].index_select(0, packed) // self.allocation
            values = (
                torch.where(
                    (indices >= 0).unsqueeze(1),
                    hidden.index_select(0, indices.clamp_min(0)),
                    entry.index_select(0, request_indices),
                )
                if program is not None
                and program.prompt_pass_is_identity
                and any(isinstance(step, Save) and step.name == name for step in program.steps)
                else entry.index_select(0, request_indices)
            )
            if name in saved_values:
                values = saved_values[name].index_select(0, upload(tuple(tails.values()), hidden.device))
            cache.kv_cache.view(-1, hidden.shape[1])[slots] = values

    def begin(self, row: PackedPhaseRow) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("CUDA routes use batched waves")

    def layer(
        self,
        row: PackedPhaseRow,
        ordinal: int,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        kv: KV = "commit",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with attention_mode(self.attentions[ordinal], positions, kv):
            return self.model.layers[ordinal](positions, hidden, residual)

    def normalize(self, row: PackedPhaseRow, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.model.norm(hidden + residual)

    def commit(self, row: PackedPhaseRow, record: Callable[[], None]) -> None:
        record()

    def finish(self, outputs: tuple[torch.Tensor, ...]) -> None:
        self.output = torch.cat(outputs)

    def clear(self) -> None:
        self._rows, self.context = (), None
        self.hidden, self.positions, self.output, self.observations = None, None, None, None
        self.activation_rows = ()

    def close(self) -> None:
        if self.stream is not None:
            self.stream.synchronize()
        self.captures.clear()
        self.clear()
        self.model.forward = self.original
