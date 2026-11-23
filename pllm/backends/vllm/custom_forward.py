"""Direct batched custom-forward execution under vLLM's own model and FA2 context."""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import torch
from vllm.distributed.parallel_state import GraphCaptureContext, get_tp_group
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.sequence import IntermediateTensors

from pllm.backends.vllm.activation import VllmActivationPatch
from pllm.backends.vllm.adaptive_depth import AdaptiveSlice
from pllm.backends.vllm.architecture import KvLM
from pllm.backends.vllm.custom_forward_attention import (
    BatchAttentionBuffers,
    attention_mode,
    attention_paged_access,
    span_context,
    validate_attention_resources,
)
from pllm.backends.vllm.model_sites import ModelSites
from pllm.backends.vllm.parameter_overlay import ParameterOverlay
from pllm.backends.vllm.position_state import PositionCacheLayer, PositionCacheProvider
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch
from pllm.contracts.semantic_graph import AnchorId, SemanticTask, TokenProgramEffect
from pllm.execution.cuda_stream import PinnedStaging, upload
from pllm.execution.custom_forward import CustomForwardPrimitive, CustomForwardTask
from pllm.execution.model_sites import StateInput
from pllm.execution.overlay_task import ProgramOperand, ProgramOverlay
from pllm.execution.state_execution import StateTaskContext
from pllm.execution.step_capture import capture_graph
from pllm.execution.step_driver import StepDriver
from pllm.execution.tensor_frame import OverlayTargetResult
from pllm.execution.token_program import KV, Save, TokenProgram
from pllm.execution.token_program_executor import Stream, add, execute_token_program, ordinary_pass

Signature = tuple[bool, tuple[bool, ...], bool]


class ProgramWave:
    """Stable inputs, page indices and captured outputs for an exact active width."""

    def __init__(self, owner: VllmCustomForward, width: int, signature: Signature) -> None:
        self.custom_forward, self.signature = owner, signature
        tensor = owner.model.norm.weight
        self.seed = tensor.new_empty((width, tensor.numel()))
        self.positions = torch.empty(width, device=tensor.device, dtype=torch.long)
        self.staging = PinnedStaging()
        self.indices = torch.empty((3, width), device=tensor.device, dtype=torch.long)
        self.attention = BatchAttentionBuffers(get_forward_context(), owner.attentions, owner.domain.positions, width)
        self.adaptive = AdaptiveSlice(self) if owner.effect.adaptive else None
        self.position = PositionCacheProvider(
            owner.position_layers,
            tuple(value for _, value in owner.effect.position_states),
            owner.allocation,
            self.seed,
        )
        self.row_overlays: tuple[ProgramOverlay, ...] = ()
        self.graph: torch.cuda.CUDAGraph | None = None
        self.output = self.observation = self.seed
        self.results: tuple[tuple[tuple[OverlayTargetResult, ...], tuple[ProgramOperand, ...]], ...] = ()
        lanes = torch.arange(width, device=tensor.device)
        self.lanes = tuple(lanes[i : i + 1] for i in range(width))

    def stage(self, rows: Sequence[PackedPhaseRow]) -> None:
        """Queue complete wave input copies without reading any device scalar."""
        owner = self.custom_forward
        assert owner.hidden is not None and owner.positions is not None
        host = self.staging.take(self.indices.numel()).reshape(self.indices.shape)
        host[:] = np.asarray(
            [(row.packed_row_index, row.request_ordinal, int(row.role.value == "decode")) for row in rows],
            dtype=np.int64,
        ).T
        self.staging.copy_to(self.indices)
        packed, requests, generated = self.indices
        self.seed.copy_(owner.hidden.index_select(0, packed))
        self.positions.copy_(owner.positions.index_select(0, packed))
        bindings = self.attention.bind(get_forward_context(), owner.attentions, owner.positions + 1)
        self.attention.refresh(bindings, packed, requests)
        self.position.stage(get_forward_context(), packed, requests, self.positions, generated)
        if owner.overlay is not None:
            parent = owner.overlay
            if not self.row_overlays:
                self.row_overlays = tuple(
                    ProgramOverlay(parent.math, parent.batch, parent.entry, parent.data) for _ in rows
                )
            for row_overlay, row in zip(self.row_overlays, rows, strict=True):
                row_overlay.batch, row_overlay.entry, row_overlay.data = parent.batch, parent.entry, parent.data
                row_overlay.ordinary = []
                row_overlay.stage(row.packed_row_index, self.signature[0])
            assert owner.projections is not None
            owner.projections.prepare_wave(rows, self.row_overlays, self.lanes)

    def layer(
        self, ordinal: int, hidden: torch.Tensor, residual: torch.Tensor | None, kv: KV = "commit"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Call the intact native layer; custom residuals (activation patches among them) apply at its norms."""
        with attention_mode(self.custom_forward.attentions[ordinal], self.positions, kv):
            hidden, residual = self.custom_forward.model.layers[ordinal](self.positions, hidden, residual)
        assert residual is not None
        return hidden, residual

    def body(self) -> torch.Tensor:
        """All program boundaries, including gate changes and state stores, are captured."""
        for row_overlay in self.row_overlays:
            row_overlay.reset()
        generated, *entries = self.position.entry(())
        with self.custom_forward.sites_rows(self.indices[0]) as stream:
            if self.signature[0]:
                hidden, saved = execute_token_program(
                    self.custom_forward.program["decode" if self.signature[2] else "prefill"],
                    self.seed,
                    self.positions,
                    generated,
                    tuple(entries),
                    self.layer,
                    self.apply_updates,
                    stream,
                    self.adaptive,
                    parameters=self.custom_forward.parameters,
                )
            else:
                count = self.custom_forward.effect.layer_count
                hidden, saved = ordinary_pass(self.seed, count, self.layer, stream), tuple(entries)
            self.position.exit(saved)
            self.observation = hidden
            self.results = tuple(
                (tuple(row_overlay.results), row_overlay.operands()) for row_overlay in self.row_overlays
            )
            return self.custom_forward.model.norm(hidden)

    def apply_updates(self) -> None:
        for row_overlay in self.row_overlays:
            row_overlay.apply_updates()

    def capture(self) -> None:
        """Restore writable KV and PositionState pages after every cold execution."""
        cells = tuple(
            (attention.kv_cache, self.attention.metadata[attention.layer_name].slot_mapping)
            for attention in self.custom_forward.attentions
        )
        saved = tuple(cache[slots // cache.shape[2], :, slots % cache.shape[2], :].clone() for cache, slots in cells)
        self.position.backup()

        def restore() -> None:
            for (cache, slots), value in zip(cells, saved, strict=True):
                cache[slots // cache.shape[2], :, slots % cache.shape[2], :] = value
            self.position.reset()

        try:
            # Native decoder all-reduces must register every captured buffer on all TP ranks.
            with (
                override_forward_context(self.attention.context),
                get_tp_group().graph_capture(GraphCaptureContext(torch.cuda.current_stream(self.seed.device))),
            ):
                self.graph, self.output, resident = capture_graph(
                    self.seed.device, self.body, restore=restore, pool=self.custom_forward.pool
                )
            resident += sum(
                wave.adaptive.resident_bytes
                for wave in self.custom_forward.captures.values()
                if wave.adaptive is not None
            )
            if resident > self.custom_forward.resident_bytes:
                raise ValueError(
                    f"custom-forward CUDA graphs need {resident} bytes; raise ExecutionOptions.cuda_graph_pool_budget_bytes above {self.custom_forward.resident_bytes}"
                )
        finally:
            restore()


class VllmCustomForward(CustomForwardPrimitive):
    """Run ordinary spans natively and dependent program positions in cross-request waves."""

    def __init__(
        self,
        model: KvLM,
        task: SemanticTask,
        driver: StepDriver,
        anchor: AnchorId,
        *,
        activation: VllmActivationPatch | None = None,
        sites: ModelSites | None = None,
        projections: ParameterOverlay | None = None,
        position_layers: tuple[PositionCacheLayer, ...] = (),
        allocation_block_size: int = 16,
    ) -> None:
        assert task.custom_forward is not None
        self.model, self.effect = model.model, task.custom_forward
        self.driver, self.anchor, self.activation, self.projections = driver, anchor, activation, projections
        self.sites = sites
        execution = driver.execution
        implementation = execution.arena.plan.scheduled(task.symbol)
        binding = CustomForwardTask.from_implementation(execution.implementation(implementation.implementation_id))
        self.program, self.domain = binding.program, binding.domain
        self.resident_bytes = next(
            fact.resident_bytes for fact in execution.arena.plan.timeline.task_resources if fact.task == task.symbol
        )
        self.position_layers, self.allocation = position_layers, allocation_block_size
        self.attentions = tuple(layer.self_attn.attn for layer in self.model.layers)
        validate_attention_resources(self.attentions, hidden_size=model.config.hidden_size)
        self.original = self.model.forward
        self._rows: Sequence[PackedPhaseRow] = ()
        self.context: StateTaskContext | None = None
        self.hidden: torch.Tensor | None = None
        self.positions: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.observations: torch.Tensor | None = None
        self.captures: dict[tuple[int, Signature], ProgramWave] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.parameters: dict[StateInput, torch.Tensor] = {}
        self.identity: tuple[int, ...] = ()
        self.stream: torch.cuda.Stream | None = None
        execution.rebind_implementation(replace(binding, primitive=self))
        self.model.forward = self.forward

    @property
    def rows(self) -> Sequence[PackedPhaseRow]:
        return self._rows

    @contextmanager
    def sites_rows(self, packed: torch.Tensor, active: torch.Tensor | None = None) -> Generator[Stream]:
        """Select each layer call's device rows for state fields and patch masks.

        Compacted waves retain their parent's homogeneous patch presence; device indices
        still gather each selected request's fields. Return the hooks' completed stream.
        """
        readers = () if self.projections is None else self.projections.readers
        previous_rows = tuple(reader.call_rows for reader in readers)
        for reader in readers:
            reader.call_rows = packed
        try:
            with self.residual_rows(packed, active) as stream:
                yield stream
        finally:
            for reader, previous in zip(readers, previous_rows, strict=True):
                reader.call_rows = previous

    @contextmanager
    def residual_rows(self, packed: torch.Tensor, active: torch.Tensor | None) -> Generator[Stream]:
        """Bind the custom residual hooks within the shared per-call state scope."""
        if self.sites is None:
            yield add
            return
        sites = self.sites
        previous = sites.rows, sites.active
        sites.rows, sites.active = packed, active
        taps = () if sites.taps is None else tuple(sites.taps.taps.values())
        if taps:
            lanes = torch.arange(len(packed), device=packed.device, dtype=torch.long)
            if active is not None:
                lanes = lanes.masked_fill(~active.flatten(), -1)
            sites.wave_inverse.fill_(-1)
            sites.wave_inverse.index_copy_(0, packed, lanes)
        for tap in taps:
            tap.rows, tap.active, tap.inverse = packed, active, sites.wave_inverse
        try:
            yield sites.stream_after
        finally:
            sites.rows, sites.active = previous
            for tap in taps:
                tap.rows, tap.active = previous
                tap.inverse = None

    def prepare(self, rows: PackedPhaseRowBatch, context: StateTaskContext) -> None:
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
            raise ValueError("@pllm.custom_forward needs one position per packed row without pipeline parallelism")
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.model.embed_input_ids(input_ids)
        self.hidden, self.positions = inputs_embeds, positions
        attention_paged_access(get_forward_context(), self.attentions, self.rows)
        self.driver.anchor(self.context, self.anchor)
        assert self.output is not None
        return self.output

    def run(
        self,
        program: dict[str, TokenProgram],
        effect: TokenProgramEffect,
        observe: Callable[[int, torch.Tensor], None],
        parameters: Mapping[StateInput, torch.Tensor],
    ) -> None:
        assert self.hidden is not None and self.positions is not None
        stream = torch.cuda.current_stream(self.hidden.device)
        if self.stream is not None and self.stream != stream:
            stream.wait_stream(self.stream)
        self.stream = stream
        for source, value in parameters.items():
            if source not in self.parameters:
                self.parameters[source] = torch.empty_like(value)
            self.parameters[source].copy_(value)
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
        self.output = torch.empty_like(self.hidden)
        self.observations = torch.empty_like(self.hidden)
        records = tuple(self.rows)  # inspect each packed row once; indexing self.rows builds a record per access
        ordinary = tuple(
            (effect.rows != "all" and row.role.value != effect.rows)
            or (
                row.role.value == "prefill" and program[row.role.value].prompt_pass_is_identity and self.overlay is None
            )
            for row in records
        )
        requests = [
            list(row for row in records if row.request_ordinal == request)
            for request in dict.fromkeys(row.request_ordinal for row in records)
        ]
        while any(requests):
            spans: dict[TokenProgram | None, list[PackedPhaseRow]] = {}
            groups: dict[Signature, list[PackedPhaseRow]] = {}
            for request in requests:
                while request and (
                    ordinary[request[0].packed_row_index] or program[request[0].role.value].over == "span"
                ):
                    row = request.pop(0)
                    spans.setdefault(None if ordinary[row.packed_row_index] else program[row.role.value], []).append(
                        row
                    )
                if request:
                    row = request.pop(0)
                    signature = (
                        effect.rows == "all" or row.role.value == effect.rows,
                        () if self.activation is None else self.activation.presence(row.packed_row_index),
                        row.role.value == "decode",
                    )
                    groups.setdefault(signature, []).append(row)
            for selected_program, span in spans.items():
                self.span(tuple(span), selected_program)
            for signature, rows in groups.items():
                selected = tuple(rows)
                key = len(rows), signature
                if key not in self.captures:
                    self.captures[key] = ProgramWave(self, len(rows), signature)
                captured = self.captures[key]
                captured.stage(selected)
                if captured.graph is None:
                    captured.capture()
                assert captured.graph is not None
                captured.graph.replay()
                self.output.index_copy_(0, captured.indices[0], captured.output)
                self.observations.index_copy_(0, captured.indices[0], captured.observation)
                for row_overlay, (results, pending) in zip(captured.row_overlays, captured.results, strict=True):
                    row_overlay.collect(results, pending)
                    assert self.overlay is not None
                    self.overlay.ordinary.extend(row_overlay.ordinary)
        for row in records:
            if effect.rows == "all" or row.role.value == effect.rows:
                observe(row.packed_row_index, self.observations[row.packed_row_index : row.packed_row_index + 1])

    def span(self, rows: Sequence[PackedPhaseRow], program: TokenProgram | None = None) -> None:
        """One native varlen pass, committing the last touched position of every page."""
        assert self.hidden is not None and self.positions is not None and self.output is not None
        context = get_forward_context()
        packed = upload(tuple(row.packed_row_index for row in rows), self.hidden.device)
        groups = tuple(
            tuple(row for row in rows if row.request_ordinal == request)
            for request in dict.fromkeys(row.request_ordinal for row in rows)
        )
        requests = upload(tuple(group[0].request_ordinal for group in groups), self.hidden.device)
        starts = [0]
        for group in groups:
            starts.append(starts[-1] + len(group))
        if rows != self.rows:
            context = span_context(context, self.attentions, groups, packed, requests, starts)
        row_overlays: tuple[ProgramOverlay, ...] = ()
        if self.overlay is not None:
            parent = self.overlay
            row_overlays = tuple(ProgramOverlay(parent.math, parent.batch, parent.entry, parent.data) for _ in groups)
            for row_overlay, group in zip(row_overlays, groups, strict=True):
                row_overlay.stage(group[0].packed_row_index, program is not None)
                row_overlay.reset()
            assert self.projections is not None
            self.projections.prepare_wave(
                tuple(group[0] for group in groups),
                row_overlays,
                tuple(upload(tuple(range(start, stop)), self.hidden.device) for start, stop in zip(starts, starts[1:])),
            )
        positions = self.positions.index_select(0, packed)

        def layer(
            ordinal: int, hidden: torch.Tensor, residual: torch.Tensor | None, kv: KV
        ) -> tuple[torch.Tensor, torch.Tensor]:
            with attention_mode(self.attentions[ordinal], positions, kv):
                hidden, residual = self.model.layers[ordinal](positions, hidden, residual)
            assert residual is not None
            return hidden, residual

        def apply_updates() -> None:
            for row_overlay in row_overlays:
                row_overlay.apply_updates()

        seed = self.hidden.index_select(0, packed)
        saved: tuple[torch.Tensor, ...] = ()
        with (
            override_forward_context(context),
            self.sites_rows(packed) as stream,
        ):
            if program is None:
                hidden = ordinary_pass(seed, self.effect.layer_count, layer, stream)
            else:
                hidden, saved = execute_token_program(
                    program,
                    seed,
                    positions,
                    upload(tuple(row.role.value == "decode" for row in rows), seed.device, torch.bool).reshape(-1, 1),
                    tuple(torch.empty_like(seed) for _ in program.entries),
                    layer,
                    apply_updates,
                    stream,
                    parameters=self.parameters,
                )
            self.output.index_copy_(0, packed, self.model.norm(hidden))
            assert self.observations is not None
            self.observations.index_copy_(0, packed, hidden)
        for row_overlay in row_overlays:
            row_overlay.collect(tuple(row_overlay.results), row_overlay.operands())
            assert self.overlay is not None
            self.overlay.ordinary.extend(row_overlay.ordinary)
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
        rows: Sequence[PackedPhaseRow],
        groups: tuple[Sequence[PackedPhaseRow], ...],
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
            if row.role.value == "prefill" and self.effect.rows in ("all", "prefill"):
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

    def clear(self) -> None:
        self._rows, self.context = (), None
        self.hidden, self.positions, self.output, self.observations = None, None, None, None

    def close(self) -> None:
        if self.stream is not None:
            self.stream.synchronize()
        self.captures.clear()
        self.clear()
        self.model.forward = self.original
