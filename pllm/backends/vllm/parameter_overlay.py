"""Parameter overlays: projection geometry checks at probing, the eager Exact projection-task driver,
the pinned reader (the custom op that reads PTTS pools inside captured graphs) and the hybrid of both.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.model_executor.layers.linear import LinearBase

from pllm.backends.vllm.architecture import Decoder, LinearOutput, Mlp, frozen_weight
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch
from pllm.contracts.parameter_overlay import (
    CapturedParameterOperand,
    ParameterOverlayFieldBinding,
    ParameterOverlayOperator,
    ParameterTarget,
    ParameterTargetShape,
)
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import StateResource
from pllm.contracts.state_ref import StateRef
from pllm.execution.cuda_stream import CudaStreamReadiness, PinnedStaging
from pllm.execution.overlay_task import ProjectionTransition, RouteOverlay
from pllm.execution.parameter_projection import (
    ROWS_PER_BLOCK,
    prepare_projection_read,
    projection_block_rows,
    projection_blocks,
    uses_pinned_projection,
)
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.ptts.slot_pools import Slots
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.state_execution import DataT, StateTaskContext


def inspect_parameter_targets(
    decoder: Decoder, targets: tuple[ParameterTarget, ...], dtype: torch.dtype
) -> tuple[ParameterTargetShape, ...]:
    specs: list[ParameterTargetShape] = []
    for index, target in enumerate(targets):
        module, offset, size, _ = decoder.projection(target)
        if (
            target in targets[:index]
            or module.tp_size != 1
            or not module.return_bias
            or module.params_dtype is not dtype
            or offset + size > module.output_size_per_partition
        ):
            raise ValueError("parameter overlay requires unique, exact unsharded projections matching model dtype")
        specs.append(ParameterTargetShape(target, module.input_size_per_partition, size))
    return tuple(specs)


Slice = tuple[ParameterOverlayFieldBinding, int, int]


def base_weights(module: LinearBase, slices: tuple[Slice, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    """Each slice's frozen projection rows, the overlay function's operand("weight").

    A quantized projection is dequantized into one dense copy only when some slice reads it;
    slices whose function never reads the weight get no rows.
    """
    reading = any(binding.reads_weight for binding, _, _ in slices)
    weight = frozen_weight(module).detach() if reading else None
    empty = torch.empty((0, module.input_size_per_partition), dtype=module.params_dtype, device=device)
    return tuple(
        weight[offset : offset + size] if weight is not None and binding.reads_weight else empty
        for binding, offset, size in slices
    )


@dataclass(frozen=True, slots=True)
class _PreparedRows:
    source: PackedPhaseRow
    ref: StateRef
    rows: torch.Tensor
    binding: ParameterOverlayFieldBinding
    offset: int
    size: int
    weight: torch.Tensor
    prompt_end: torch.Tensor


class _ExactProjection:
    def __init__(self, module: LinearBase, mlp: Mlp | None, slices: tuple[Slice, ...], device: torch.device) -> None:
        self.module, self.mlp = module, mlp
        self.bindings = tuple(
            (*item, weight) for item, weight in zip(slices, base_weights(module, slices, device), strict=True)
        )
        self.original_forward = module.forward
        self.original_mlp_forward = None if mlp is None else mlp.forward
        self.prepared: tuple[_PreparedRows, ...] = ()
        self.captures: tuple[CapturedParameterOperand, ...] = ()
        self.mlp_input: torch.Tensor | None = None
        self.total_rows = 0
        self.apply_count = self.updated_apply_count = 0
        self.transition: ProjectionTransition | None = None
        self.routes: tuple[RouteOverlay, ...] = ()

    def captures_binding(self, binding: ParameterOverlayFieldBinding) -> bool:
        return self.original_mlp_forward is not None or binding.operator is ParameterOverlayOperator.HEAD_ADD

    def forward_mlp(self, value: torch.Tensor) -> torch.Tensor:
        assert self.original_mlp_forward is not None
        self.mlp_input = value.detach()
        try:
            return self.original_mlp_forward(value)
        finally:
            self.mlp_input = None

    def forward(self, hidden: torch.Tensor) -> LinearOutput:
        base = self.original_forward(hidden)
        if not self.prepared:
            return base
        assert hidden.shape[0] == self.total_rows and not self.captures
        # The unquantized TP1 projection owns this fresh output, not an input or state page.
        match base:
            case (output, *_):
                pass
            case output:
                pass
        # A replay span can contain both phases of one request. Finish its prompt
        # transition (including prompt-end flush) before reading its decode state.
        phases = tuple(
            phase
            for phase in (HiddenPhase.PREFILL, HiddenPhase.DECODE)
            if any(item.source.phase is phase for item in self.prepared)
        )
        for phase in phases:
            prepared = tuple(item for item in self.prepared if item.source.phase is phase)
            self.forward_segment(hidden, output, prepared, complete=phase is phases[-1])
        return base

    def forward_segment(
        self, hidden: torch.Tensor, output: torch.Tensor, prepared: tuple[_PreparedRows, ...], *, complete: bool
    ) -> None:
        inputs = tuple(hidden.index_select(0, item.rows) for item in prepared)
        captures = tuple(
            self.capture(item, selected)
            for item, selected in zip(prepared, inputs, strict=True)
            if self.captures_binding(item.binding)
        )
        if self.routes:
            residuals = {
                key: value
                for route in self.routes
                for key, value in route.run(
                    tuple(
                        operand
                        for operand in captures
                        if operand.identity == route.data.step.rows.rows[route.row].identity
                    )
                ).items()
            }
        else:
            assert self.transition is not None
            residuals = self.transition.run(captures, complete=complete)
        for item, selected in zip(prepared, inputs, strict=True):
            delta = (residuals[item.source.identity, item.binding.target] * item.binding.scale).to(output.dtype)
            columns = output[:, item.offset : item.offset + item.size]
            columns.index_add_(0, item.rows, delta)
            self.apply_count += selected.shape[0]
            if item.ref.version_id > 0:
                self.updated_apply_count += selected.shape[0]
        if not self.routes:
            self.captures += captures

    def capture(self, item: _PreparedRows, selected: torch.Tensor) -> CapturedParameterOperand:
        if item.binding.operator is ParameterOverlayOperator.HEAD_ADD:
            desired = selected
        else:
            assert self.mlp_input is not None and self.mlp_input.shape[0] == self.total_rows
            desired = self.mlp_input.index_select(0, item.rows)
        return CapturedParameterOperand(
            item.source.identity,
            item.binding.target,
            item.source.phase,
            selected.detach(),
            desired.detach(),
            item.weight,
            item.prompt_end,
        )

    def clear(self) -> None:
        self.prepared, self.captures = (), ()
        self.mlp_input = None
        self.total_rows = 0
        self.transition = None
        self.routes = ()


class ExactParameterOverlay:
    """Eager Exact projection-task driver; pinned reads use the shared tensor op."""

    def __init__(
        self, decoder: Decoder, bindings: tuple[ParameterOverlayFieldBinding, ...], device: torch.device
    ) -> None:
        modules: dict[int, tuple[LinearBase, Mlp | None, list[Slice]]] = {}
        for binding in bindings:
            module, offset, size, mlp = decoder.projection(binding.target)
            modules.setdefault(id(module), (module, mlp, []))[2].append((binding, offset, size))
        self._modules = tuple(
            _ExactProjection(module, mlp, tuple(slices), device) for module, mlp, slices in modules.values()
        )
        self.device = device
        self._staging = PinnedStaging()
        for handle in self._modules:
            handle.module.forward = handle.forward
            if handle.mlp is not None:
                handle.mlp.forward = handle.forward_mlp

    @property
    def apply_count(self) -> int:
        return sum(handle.apply_count for handle in self._modules)

    @property
    def updated_apply_count(self) -> int:
        return sum(handle.updated_apply_count for handle in self._modules)

    def reserve(self) -> None:
        """Exact reads stage nothing ahead of the invocation."""

    def prepare(
        self,
        rows: PackedPhaseRowBatch,
        context: StateTaskContext[DataT],
        directory: RequestStateDirectory,
        transition: ProjectionTransition | None = None,
    ) -> None:
        assert transition is not None
        grouped: dict[tuple[RequestIdentity, HiddenPhase], list[PackedPhaseRow]] = {}
        for row in rows.rows:
            grouped.setdefault((row.identity, row.phase), []).append(row)
        table = torch.empty((len(rows.rows) + len(grouped),), dtype=torch.long, device=self.device)
        host = self._staging.take(table.numel()).numpy()
        host[: len(rows.rows)] = [row.packed_row_index for source in grouped.values() for row in source]
        host[len(rows.rows) :] = [
            source[-1].token_offset == source[-1].prompt_length - 1 for source in grouped.values()
        ]
        self._staging.copy_to(table)
        indices = torch.split(table[: len(rows.rows)], [len(source) for source in grouped.values()])
        prompt_ends = table[len(rows.rows) :].bool().unbind()
        requests = tuple(
            (source[0], context.selection.ref_for(identity), index, prompt_end)
            for ((identity, _), source), index, prompt_end in zip(grouped.items(), indices, prompt_ends, strict=True)
        )
        for handle in self._modules:
            handle.transition = transition
            handle.prepared = tuple(
                _PreparedRows(source, ref, index, binding, offset, size, weight, prompt_end)
                for source, ref, index, prompt_end in requests
                for binding, offset, size, weight in handle.bindings
            )
            handle.total_rows = rows.total_rows

    def prepare_wave(
        self, rows: tuple[PackedPhaseRow, ...], routes: tuple[RouteOverlay, ...], indices: tuple[torch.Tensor, ...]
    ) -> None:
        """Use one projection call across the wave and request-local update operands."""
        for row, route, index in zip(rows, routes, indices, strict=True):
            route.prompt_end.fill_(row.token_offset + index.numel() == row.prompt_length)
        for handle in self._modules:
            handle.routes = routes
            handle.prepared = tuple(
                _PreparedRows(
                    row,
                    route.batch.after_refs[route.batch.selection.state_indices[route.row]],
                    index,
                    binding,
                    offset,
                    size,
                    weight,
                    route.prompt_end,
                )
                for row, route, index in zip(rows, routes, indices, strict=True)
                for binding, offset, size, weight in handle.bindings
            )
            handle.captures, handle.total_rows = (), sum(index.numel() for index in indices)

    def take_captures(self) -> tuple[CapturedParameterOperand, ...]:
        for handle in self._modules:
            if handle.routes:
                continue
            expected = sum(handle.captures_binding(item.binding) for item in handle.prepared)
            assert len(handle.captures) == expected
            handle.captures = ()
        return ()

    def clear(self) -> None:
        for handle in self._modules:
            handle.clear()

    def close(self) -> None:
        self.clear()
        for handle in self._modules:
            handle.module.forward = handle.original_forward
            if handle.mlp is not None and handle.original_mlp_forward is not None:
                handle.mlp.forward = handle.original_mlp_forward


def capacities(maximum: int) -> tuple[int, ...]:
    """Finite power-of-two capacity classes, ending at the declared bound."""
    return tuple(sorted({min(1 << i, maximum) for i in range(maximum.bit_length() + 1)}))


_READERS: dict[int, "_ProjectionReader"] = {}


@torch.library.custom_op("pllm::pinned_projection", mutates_args={"output", "captured"})
def pinned_projection(
    hidden: torch.Tensor,
    pools: list[torch.Tensor],
    table: torch.Tensor,
    output: torch.Tensor,
    captured: torch.Tensor,
) -> None:
    """Execute the same prepared tensor callable eagerly or during upstream capture."""
    _READERS[table.data_ptr()].emit(hidden, pools, table, output, captured)


def _projection_meta(
    hidden: torch.Tensor,
    pools: list[torch.Tensor],
    table: torch.Tensor,
    output: torch.Tensor,
    captured: torch.Tensor,
) -> None:
    return None


pinned_projection.register_fake(_projection_meta)  # pyright: ignore[reportUnknownMemberType]


class _ProjectionReader:
    """A target's stable tensor ports; the invocation supplies all pin-authorized indices."""

    def __init__(
        self,
        decoder: Decoder,
        binding: ParameterOverlayFieldBinding,
        arena: PageBundleArena,
        rows: int,
        requests: int,
        budget: int,
    ) -> None:
        self.module, self.offset, self.width, self.mlp = decoder.projection(binding.target)
        self.forward_base = self.module.forward
        self.mlp_base = None if self.mlp is None else self.mlp.forward
        self.binding, self.rows, self.requests = binding, rows, requests
        self.enabled = True
        self.pools = [view.read(arena.field_pool(binding.resource, view.field), pooled=True) for view in binding.fields]
        self.table = torch.zeros(
            projection_blocks(rows, requests) * (ROWS_PER_BLOCK + len(binding.fields)) + rows,
            device=self.pools[0].device,
            dtype=torch.long,
        )
        self.table[-rows:].fill_(-1)
        width = self.module.input_size_per_partition
        self.captured = self.pools[0].new_zeros((rows, width), dtype=self.module.params_dtype)
        self.desired = self.captured.new_empty((rows, self.width)) if self.mlp is not None else self.captured
        (self.weight,) = base_weights(self.module, ((binding, self.offset, self.width),), self.captured.device)
        self.staging = PinnedStaging(capacity=self.table.numel(), slots=2)
        self.host = torch.empty(0, dtype=torch.long)
        self.programs: dict[int, Callable[..., tuple[torch.Tensor, ...]]] = {}
        for count in capacities(rows):
            with torch.inference_mode(False), FakeTensorMode(allow_non_fake_inputs=True):
                examples = (
                    self.captured.new_empty((count, self.captured.shape[1])),
                    *self.ports(self.table, count),
                    *self.pools,
                )
            self.programs[count] = prepare_projection_read(examples, binding.operator, budget)
            # Warm every row bucket against the actual readonly pools before graph capture.
            self.read(self.captured[:count], self.pools, self.table, count)
        _READERS[self.table.data_ptr()] = self
        self.module.forward = self.forward
        if self.mlp is not None:
            self.mlp.forward = self.forward_mlp

    def reserve(self) -> None:
        self.host = self.staging.take(self.table.numel())

    def ports(self, table: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        blocks = projection_blocks(count, min(count, self.requests))
        width = projection_block_rows(count, self.requests)
        maximum = projection_blocks(self.rows, self.requests)
        return (
            table[: maximum * len(self.pools)].view(maximum, len(self.pools))[:blocks],
            table[maximum * len(self.pools) : maximum * len(self.pools) + blocks * ROWS_PER_BLOCK].view(
                blocks, ROWS_PER_BLOCK
            )[:, :width],
            table[-self.rows :][:count],
        )

    def forward_mlp(self, value: torch.Tensor) -> torch.Tensor:
        assert self.mlp_base is not None
        if self.enabled:
            self.desired[: value.shape[0]].copy_(value)
        return self.mlp_base(value)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.nn.Parameter | None]:
        base = self.forward_base(hidden)
        assert isinstance(base, tuple)
        if self.enabled:
            pinned_projection(hidden, self.pools, self.table, base[0], self.captured)
        return base

    def emit(
        self,
        hidden: torch.Tensor,
        pools: list[torch.Tensor],
        table: torch.Tensor,
        output: torch.Tensor,
        captured: torch.Tensor,
    ) -> None:
        count = next(size for size in capacities(self.rows) if size >= hidden.shape[0])
        padded = torch.nn.functional.pad(hidden, (0, 0, 0, count - hidden.shape[0]))
        result = self.read(padded, pools, table, count)
        output[:, self.offset : self.offset + self.width].add_(
            (result[: hidden.shape[0]] * self.binding.scale).to(output.dtype)
        )
        captured[: hidden.shape[0]].copy_(hidden)

    def read(self, hidden: torch.Tensor, pools: list[torch.Tensor], table: torch.Tensor, count: int) -> torch.Tensor:
        values = (hidden, *self.ports(table, count), *pools)
        return self.programs[count](*values)[0]

    def prepare(self, field_slots: Slots, starts: Slots, total: int) -> None:
        host = self.host.numpy()
        maximum = projection_blocks(self.rows, self.requests)
        slots = host[: maximum * len(self.pools)].reshape(maximum, len(self.pools))
        positions = host[maximum * len(self.pools) : -self.rows].reshape(maximum, ROWS_PER_BLOCK)
        owners = host[-self.rows :]
        owners.fill(-1)
        count = next(size for size in capacities(self.rows) if size >= total)
        width = projection_block_rows(count, self.requests)
        lengths = np.diff(np.append(starts, total))
        blocks = (lengths + width - 1) // width
        first_blocks = np.cumsum(blocks) - blocks
        requests = np.repeat(np.arange(len(starts)), lengths)
        offsets = np.arange(total) - starts[requests]
        block_rows = first_blocks[requests] + offsets // width
        selected = field_slots[:, tuple(view.field for view in self.binding.fields)]
        slots[:] = selected[0]
        slots[: int(blocks.sum())] = np.repeat(selected, blocks, axis=0)
        positions[block_rows, offsets % width] = np.arange(total)
        owners[:total] = block_rows * width + offsets % width
        self.staging.copy_to(self.table)

    def close(self) -> None:
        self.module.forward = self.forward_base
        if self.mlp is not None and self.mlp_base is not None:
            self.mlp.forward = self.mlp_base
        _READERS.pop(self.table.data_ptr())


class PinnedParameterOverlay:
    """The sole pinned reader and capture path, for eager, replay and graph placement."""

    def __init__(
        self,
        decoder: Decoder,
        bindings: tuple[ParameterOverlayFieldBinding, ...],
        arena: PageBundleArena,
        rows: int,
        requests: int,
        budget: int,
        *,
        phases: tuple[HiddenPhase, ...],
    ) -> None:
        self.readers = tuple(_ProjectionReader(decoder, binding, arena, rows, requests, budget) for binding in bindings)
        self.phases = phases
        self.sources: tuple[PackedPhaseRow, ...] = ()
        self.starts: Slots = np.empty(0, dtype=np.int64)
        self.total_rows = 0
        self.stream: torch.cuda.Stream | None = None
        self.readiness: CudaStreamReadiness | None = None
        self.apply_count = self.updated_apply_count = 0

    def reserve(self) -> None:
        """Acquire every host table before the invocation issues any GPU work."""
        for reader in self.readers:
            reader.reserve()

    def prepare(
        self,
        rows: PackedPhaseRowBatch,
        context: StateTaskContext[DataT],
        directory: RequestStateDirectory,
        transition: ProjectionTransition | None = None,
    ) -> None:
        assert transition is None
        self.stream = context.pin.stream
        assert self.stream is not None
        if self.readiness is not None:
            self.readiness.order_on(self.stream)
        ordinals = np.fromiter((row.request_ordinal for row in rows.rows), np.int64, rows.total_rows)
        self.starts = np.flatnonzero(np.append(True, ordinals[1:] != ordinals[:-1]))
        self.sources = tuple(rows.rows[int(index)] for index in self.starts)
        self.total_rows = rows.total_rows
        for reader in self.readers:
            versions = context.read_versions(
                tuple(row.identity for row in self.sources), StateResource(reader.binding.resource), directory
            )
            pins = tuple(pin for pin, _ in versions)
            refs = StateRefs.of(tuple(ref for _, ref in versions))
            _, slots = context.arena.slots(pins, refs)
            self.updated_apply_count += int(np.diff(np.append(self.starts, rows.total_rows)) @ (refs.versions > 0))
            for value in (*reader.pools, reader.table, reader.captured, reader.desired):
                value.record_stream(self.stream)
            reader.prepare(slots, self.starts, rows.total_rows)
        self.apply_count += len(rows.rows) * len(self.readers)

    def take_captures(self) -> tuple[CapturedParameterOperand, ...]:
        if not self.sources or not self.phases:
            return ()
        spans = tuple(
            (row, first, last)
            for row, first, last in zip(
                self.sources, self.starts.tolist(), np.append(self.starts[1:], self.total_rows).tolist(), strict=True
            )
            if row.phase in self.phases
        )
        results: list[CapturedParameterOperand] = []
        for reader in self.readers if spans else ():
            for row, first, last in spans:
                hidden = reader.captured[first:last].clone()
                results.append(
                    CapturedParameterOperand(
                        row.identity,
                        reader.binding.target,
                        row.phase,
                        hidden,
                        hidden if reader.mlp is None else reader.desired[first:last].clone(),
                        reader.weight,
                        hidden.new_full((), row.token_offset + last - first == row.prompt_length, dtype=torch.bool),
                    )
                )
        return tuple(results)

    def clear(self) -> None:
        if self.sources:
            assert self.stream is not None
            self.readiness = CudaStreamReadiness(self.stream)
        self.sources = ()

    def close(self) -> None:
        for reader in reversed(self.readers):
            reader.close()


class HybridParameterOverlay:
    """Exact training uses its eager driver; inactive batches read through the pinned op."""

    def __init__(
        self, pinned: PinnedParameterOverlay, exact: ExactParameterOverlay, phases: tuple[HiddenPhase, ...]
    ) -> None:
        self.pinned, self.exact, self.phases = pinned, exact, phases

    @property
    def apply_count(self) -> int:
        return self.pinned.apply_count + self.exact.apply_count

    @property
    def updated_apply_count(self) -> int:
        return self.pinned.updated_apply_count + self.exact.updated_apply_count

    def reserve(self) -> None:
        """Reserve bounded staging before the invocation issues GPU work."""
        self.pinned.reserve()

    def prepare(
        self,
        rows: PackedPhaseRowBatch,
        context: StateTaskContext[DataT],
        directory: RequestStateDirectory,
        transition: ProjectionTransition | None = None,
    ) -> None:
        """Select the same eligibility used by dispatch from the labelled row facts."""
        training = any(row.phase in self.phases for row in rows.rows)
        for reader in self.pinned.readers:
            reader.enabled = not training
        if training:
            self.exact.prepare(rows, context, directory, transition)
        else:
            self.pinned.prepare(rows, context, directory)

    def take_captures(self) -> tuple[CapturedParameterOperand, ...]:
        """Exact consumed training operands in forward; inactive reads produce no task operands."""
        return self.exact.take_captures()

    def clear(self) -> None:
        """Release invocation views through their existing owners."""
        self.exact.clear()
        self.pinned.clear()

    def close(self) -> None:
        """Unwind the stable wrappers in reverse installation order."""
        self.exact.close()
        self.pinned.close()


def parameter_overlay(
    decoder: Decoder,
    bindings: tuple[ParameterOverlayFieldBinding, ...],
    arena: PageBundleArena,
    rows: int,
    requests: int,
    budget: int,
    phases: tuple[HiddenPhase, ...],
    *,
    in_forward: bool,
    graphs: bool,
    routed: bool = False,
) -> PinnedParameterOverlay | ExactParameterOverlay | HybridParameterOverlay:
    """Install each projection wrapper once, before compiling or capturing the model."""
    if routed or not uses_pinned_projection(in_forward=in_forward, graphs=graphs):
        return ExactParameterOverlay(decoder, bindings, arena.device)
    pinned = PinnedParameterOverlay(
        decoder, bindings, arena, rows, requests, budget, phases=() if in_forward else phases
    )
    if not in_forward:
        return pinned
    return HybridParameterOverlay(pinned, ExactParameterOverlay(decoder, bindings, arena.device), phases)
