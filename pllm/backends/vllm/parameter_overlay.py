"""Parameter overlays: projection geometry checks at probing, the eager Exact projection-task driver,
the pinned reader (the custom op that reads PTTS pools inside captured graphs) and the hybrid of both.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.model_executor.layers.linear import LinearBase, RowParallelLinear

from pllm.backends.vllm.architecture import Decoder, LinearOutput, Mlp, frozen_weight
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch
from pllm.contracts.parameter_overlay import (
    ParameterOverlayFieldBinding,
    ParameterOverlayOperator,
    Projection,
    ProjectionOperands,
    ProjectionShape,
)
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import StateResource
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.cuda_stream import CudaStreamReadiness, PinnedStaging
from pllm.execution.overlay_batch import DecodeOverlay
from pllm.execution.overlay_task import OverlayTransition, ProgramOverlay, ProjectionTransition
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
from pllm.execution.tensor_frame import UpdateOutput
from pllm.execution.torch_private import blas_workspace_bytes


def inspect_parameter_targets(
    decoder: Decoder, targets: tuple[Projection, ...], dtype: torch.dtype
) -> tuple[ProjectionShape, ...]:
    specs: list[ProjectionShape] = []
    for index, target in enumerate(targets):
        module, offset, size, _ = decoder.projection(target)
        if (
            target in targets[:index]
            or (module.tp_size != 1 and target.name != "down_proj")
            or not module.return_bias
            or module.params_dtype is not dtype
            or offset + size > module.output_size_per_partition
        ):
            raise ValueError("parameter overlay needs unique q/v or row-parallel down projections matching model dtype")
        specs.append(
            ProjectionShape(
                target, module.input_size_per_partition, size, module.tp_size if target.name == "down_proj" else 1
            )
        )
    return tuple(specs)


def _row_parallel(module: LinearBase) -> bool:
    """Whether the projection computes a rank-local partial output."""
    return isinstance(module, RowParallelLinear) and module.tp_size > 1


class ProjectionHook:
    """Put a down delta inside vLLM's GEMM, before its own row all-reduce.

    TP1 keeps the established forward hook. TP>1 intercepts only quant_method.apply;
    input slicing, rank-zero bias, reduction and return-bias handling stay upstream.
    """

    def __init__(self, module: LinearBase) -> None:
        self.module = module
        self.forward = module.forward
        self.apply = module.quant_method.apply if _row_parallel(module) else None
        self.callback: Callable[[torch.Tensor], LinearOutput] = self.forward
        self.bias: torch.Tensor | None = None

    def base(self, hidden: torch.Tensor) -> LinearOutput:
        if self.apply is None:
            return self.forward(hidden)
        # Accumulating the correction in BF16 before the reduction adds a rounding
        # on every rank. Keep the partial sum wide and round once after vLLM sums it.
        return self.apply(self.module, hidden, self.bias).float(), None

    def partial(self, layer: LinearBase, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        self.bias = bias
        output = self.callback(x)
        assert isinstance(output, tuple)
        return output[0]

    def reduced(self, hidden: torch.Tensor) -> LinearOutput:
        output = self.forward(hidden)
        assert isinstance(output, tuple)
        return output[0].to(self.module.params_dtype), output[1]

    def install(self, callback: Callable[[torch.Tensor], LinearOutput]) -> None:
        self.callback = callback
        if self.apply is None:
            self.module.forward = callback
        else:
            self.module.quant_method.apply = self.partial
            self.module.forward = self.reduced

    def close(self) -> None:
        self.module.forward = self.forward
        if self.apply is not None:
            self.module.quant_method.apply = self.apply


Slice = tuple[ParameterOverlayFieldBinding, int, int]


def base_weight(module: LinearBase, reading: bool, device: torch.device) -> torch.Tensor:
    """Borrow or dequantize operand("weight") for this read, only when its function uses it."""
    if reading:
        if not _row_parallel(module):
            return frozen_weight(module).detach()
        method = module.quant_method
        apply = method.apply
        # vLLM's identity dequantizer calls apply; it must borrow the base, without overlay hooks.
        method.apply = partial(type(method).apply, method)
        try:
            return frozen_weight(module).detach()
        finally:
            method.apply = apply
    return torch.empty((0, module.input_size_per_partition), dtype=module.params_dtype, device=device)


@dataclass(frozen=True, slots=True)
class _PreparedRows:
    source: PackedPhaseRow
    ref: StateRef
    rows: torch.Tensor
    binding: ParameterOverlayFieldBinding
    offset: int
    size: int
    prompt_end: torch.Tensor


class _EmbeddingRows:
    """Own X0 rows before the first norm can reuse their storage, including during graph replay."""

    def __init__(self, decoder: Decoder, rows: int, device: torch.device, *, enabled: bool) -> None:
        norm = decoder.model.model.layers[0].input_layernorm
        self.values = torch.empty(
            (rows, norm.weight.numel()) if enabled else (0, 0), device=device, dtype=norm.weight.dtype
        )
        self.hook = norm.register_forward_pre_hook(self.capture) if enabled else None

    def capture(self, module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        """Copy the embedding stream once per model invocation, in its packed row order."""
        self.values[: len(args[0])].copy_(args[0])

    def close(self) -> None:
        """Remove the model hook when the overlay closes."""
        if self.hook is not None:
            self.hook.remove()


class _ExactProjection:
    def __init__(
        self, module: LinearBase, mlp: Mlp | None, slices: tuple[Slice, ...], embedding: _EmbeddingRows
    ) -> None:
        self.embedding = embedding
        self.module, self.mlp = module, mlp
        self.bindings = slices
        self.hook = ProjectionHook(module)
        self.original_forward = self.hook.base
        self.original_mlp_forward = None if mlp is None else mlp.forward
        self.prepared: tuple[_PreparedRows, ...] = ()
        self.operand_count = 0
        self.mlp_input: torch.Tensor | None = None
        self.total_rows = 0
        self.apply_count = self.updated_apply_count = 0
        self.transition: ProjectionTransition | None = None
        self.row_overlays: tuple[ProgramOverlay, ...] = ()

    def reads_operands(self, binding: ParameterOverlayFieldBinding) -> bool:
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
        assert hidden.shape[0] == self.total_rows and not self.operand_count
        # The GEMM owns this fresh partial output, not an input or state page.
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
        weight = base_weight(self.module, any(binding.reads_weight for binding, _, _ in self.bindings), hidden.device)
        for phase in phases:
            prepared = tuple(item for item in self.prepared if item.source.phase is phase)
            self.forward_segment(hidden, output, prepared, weight, complete=phase is phases[-1])
        return base

    def forward_segment(
        self,
        hidden: torch.Tensor,
        output: torch.Tensor,
        prepared: tuple[_PreparedRows, ...],
        weight: torch.Tensor,
        *,
        complete: bool,
    ) -> None:
        inputs = tuple(hidden.index_select(0, item.rows) for item in prepared)
        operands = tuple(
            self.operands_of(item, selected, weight)
            for item, selected in zip(prepared, inputs, strict=True)
            if self.reads_operands(item.binding)
        )
        if self.row_overlays:
            residuals = {
                key: value
                for row_overlay in self.row_overlays
                for key, value in row_overlay.run(
                    tuple(
                        operand
                        for operand in operands
                        if operand.identity == row_overlay.data.step.rows.identity(row_overlay.row)
                    )
                ).items()
            }
        else:
            assert self.transition is not None
            residuals = self.transition.run(operands, complete=complete)
        for item, selected in zip(prepared, inputs, strict=True):
            delta = residuals[item.source.identity, item.binding.target] * item.binding.scale
            columns = output[:, item.offset : item.offset + item.size]
            columns.index_copy_(0, item.rows, (columns.index_select(0, item.rows) + delta).to(output.dtype))
            self.apply_count += selected.shape[0]
            if item.ref.version_id > 0:
                self.updated_apply_count += selected.shape[0]
        if not self.row_overlays:
            self.operand_count += len(operands)

    def operands_of(self, item: _PreparedRows, selected: torch.Tensor, weight: torch.Tensor) -> ProjectionOperands:
        if item.binding.operator is ParameterOverlayOperator.HEAD_ADD:
            desired = selected
        else:
            assert self.mlp_input is not None and self.mlp_input.shape[0] == self.total_rows
            desired = self.mlp_input.index_select(0, item.rows)
        return ProjectionOperands(
            item.source.identity,
            item.binding.target,
            item.source.phase,
            selected.detach(),
            desired.detach(),
            weight[item.offset : item.offset + item.size] if item.binding.reads_weight else weight[:0],
            item.prompt_end,
            self.embedding.values.index_select(0, item.rows)
            if item.binding.reads_embedding
            else selected.new_empty((0, 0)),
        )

    def clear(self) -> None:
        self.prepared, self.operand_count = (), 0
        self.mlp_input = None
        self.total_rows = 0
        self.transition = None
        self.row_overlays = ()


class ExactParameterOverlay:
    """Eager Exact projection-task driver; pinned reads use the shared tensor op."""

    def __init__(
        self, decoder: Decoder, bindings: tuple[ParameterOverlayFieldBinding, ...], device: torch.device, rows: int
    ) -> None:
        self.embedding = _EmbeddingRows(
            decoder, rows, device, enabled=any(binding.reads_embedding for binding in bindings)
        )
        modules: dict[int, tuple[LinearBase, Mlp | None, list[Slice]]] = {}
        for binding in bindings:
            module, offset, size, mlp = decoder.projection(binding.target)
            modules.setdefault(id(module), (module, mlp, []))[2].append((binding, offset, size))
        self._modules = tuple(
            _ExactProjection(module, mlp, tuple(slices), self.embedding) for module, mlp, slices in modules.values()
        )
        self.device = device
        self.operand_staging = PinnedStaging()
        for handle in self._modules:
            handle.hook.install(handle.forward)
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
        table = torch.empty((rows.total_rows + len(grouped),), dtype=torch.long, device=self.device)
        host = self.operand_staging.take(table.numel())
        host[: rows.total_rows] = [row.packed_row_index for source in grouped.values() for row in source]
        host[rows.total_rows :] = [
            source[-1].token_offset == source[-1].prompt_length - 1 for source in grouped.values()
        ]
        self.operand_staging.copy_to(table)
        indices = torch.split(table[: rows.total_rows], [len(source) for source in grouped.values()])
        prompt_ends = table[rows.total_rows :].bool().unbind()
        requests = tuple(
            (source[0], index, prompt_end)
            for (_, source), index, prompt_end in zip(grouped.items(), indices, prompt_ends, strict=True)
        )
        for handle in self._modules:
            handle.transition = transition
            handle.prepared = tuple(
                _PreparedRows(
                    source,
                    context.selection.ref_for(source.identity, StateResource(binding.resource)),
                    index,
                    binding,
                    offset,
                    size,
                    prompt_end,
                )
                for source, index, prompt_end in requests
                for binding, offset, size in handle.bindings
            )
            handle.total_rows = rows.total_rows

    def prepare_wave(
        self,
        rows: Sequence[PackedPhaseRow],
        row_overlays: tuple[ProgramOverlay, ...],
        indices: tuple[torch.Tensor, ...],
    ) -> None:
        """Use one projection call across the wave and request-local update operands."""
        for row, row_overlay, index in zip(rows, row_overlays, indices, strict=True):
            row_overlay.prompt_end.fill_(row.token_offset + index.numel() == row.prompt_length)
        for handle in self._modules:
            handle.row_overlays = row_overlays
            handle.prepared = tuple(
                _PreparedRows(
                    row,
                    row_overlay.batch.after_refs[int(row_overlay.batch.selection.state_indices[row_overlay.row])],
                    index,
                    binding,
                    offset,
                    size,
                    row_overlay.prompt_end,
                )
                for row, row_overlay, index in zip(rows, row_overlays, indices, strict=True)
                for binding, offset, size in handle.bindings
            )
            handle.operand_count, handle.total_rows = 0, sum(index.numel() for index in indices)

    def take_operands(self) -> tuple[ProjectionOperands, ...]:
        for handle in self._modules:
            if handle.row_overlays:
                continue
            expected = sum(handle.reads_operands(item.binding) for item in handle.prepared)
            assert handle.operand_count == expected
            handle.operand_count = 0
        return ()

    def clear(self) -> None:
        for handle in self._modules:
            handle.clear()

    def close(self) -> None:
        self.clear()
        self.embedding.close()
        for handle in self._modules:
            handle.hook.close()
            if handle.mlp is not None and handle.original_mlp_forward is not None:
                handle.mlp.forward = handle.original_mlp_forward


def capacities(maximum: int) -> tuple[int, ...]:
    """Finite power-of-two capacity classes, ending at the declared bound."""
    return tuple(sorted({min(1 << i, maximum) for i in range(maximum.bit_length() + 1)}))


def uniform_decode(prefill: bool, counts: tuple[int, ...]) -> bool:
    """Whether every scheduled request contributes one decode row, without a prefill or replay span."""
    return not prefill and all(count == 1 for count in counts)


_READERS: dict[int, "ProjectionReader"] = {}


@torch.library.custom_op("pllm::pinned_projection", mutates_args={"output", "captured", "desired_capture", "pools"})
def pinned_projection(
    hidden: torch.Tensor,
    desired: torch.Tensor,
    pools: list[torch.Tensor],
    table: torch.Tensor,
    output: torch.Tensor,
    captured: torch.Tensor,
    desired_capture: torch.Tensor,
) -> None:
    """Execute the same prepared tensor callable eagerly or during upstream capture."""
    _READERS[table.data_ptr()].emit(hidden, desired, pools, table, output, captured, desired_capture)


def _projection_meta(
    hidden: torch.Tensor,
    desired: torch.Tensor,
    pools: list[torch.Tensor],
    table: torch.Tensor,
    output: torch.Tensor,
    captured: torch.Tensor,
    desired_capture: torch.Tensor,
) -> None:
    return None


pinned_projection.register_fake(_projection_meta)  # pyright: ignore[reportUnknownMemberType]


class ProjectionReader:
    """A target's stable tensor ports; the invocation supplies all pin-authorized indices."""

    def __init__(
        self,
        decoder: Decoder,
        binding: ParameterOverlayFieldBinding,
        arena: PageBundleArena,
        rows: int,
        requests: int,
        budget: int,
        *,
        math: OverlayTransition | None = None,
        in_program: bool = False,
    ) -> None:
        self.module, self.offset, self.width, self.mlp = decoder.projection(binding.target)
        self.hook = ProjectionHook(self.module)
        self.forward_base = self.hook.base
        self.mlp_base = None if self.mlp is None else self.mlp.forward
        self.binding, self.rows, self.requests = binding, rows, requests
        self.in_program = in_program
        self.call_rows: torch.Tensor | None = None
        self.enabled = True
        self.pools = [view.read(arena.field_pool(binding.resource, view.field), pooled=True) for view in binding.fields]
        self.table = torch.zeros(
            projection_blocks(rows, requests) * (ROWS_PER_BLOCK + len(binding.fields)) + rows,
            device=self.pools[0].device,
            dtype=torch.long,
        )
        self.table[-rows:].fill_(-1)
        inputs = self.module.input_size_per_partition
        ports = requests if math is not None else rows
        self.captured = self.pools[0].new_zeros((ports, inputs), dtype=self.module.params_dtype)
        self.desired = self.captured.new_empty((ports, self.width) if self.mlp is not None else (0, 0))
        self.mlp_input: torch.Tensor | None = None
        self.staging = PinnedStaging(capacity=self.table.numel(), slots=2)
        self.host = np.empty(0, dtype=np.int64)
        self.decode: DecodeOverlay | None = None
        self.embedding = self.captured.new_empty((0, 0))
        self.programs: dict[int, Callable[..., tuple[torch.Tensor, ...]]] = {}
        blas_bytes = blas_workspace_bytes()
        counts = (
            ()
            if math is not None and HiddenPhase.DECODE in math.phases
            else capacities(requests if math is not None else rows)
        )
        for count in counts:
            with torch.inference_mode(False), FakeTensorMode(allow_non_fake_inputs=True):
                examples = (
                    self.captured.new_empty((count, self.captured.shape[1])),
                    *self.ports(self.table, count),
                    *self.pools,
                )
            self.programs[count] = prepare_projection_read(
                examples, binding.operator, budget, blas_workspace_bytes=blas_bytes
            )
            # Warm every row bucket against the actual readonly pools before graph capture.
            self.read(self.captured[:count], self.pools, self.table, count)
        _READERS[self.table.data_ptr()] = self
        self.hook.install(self.forward)
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
        self.mlp_input = value
        try:
            return self.mlp_base(value)
        finally:
            self.mlp_input = None

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.nn.Parameter | None]:
        base = self.forward_base(hidden)
        assert isinstance(base, tuple)
        if self.enabled:
            buffers = self.pools if self.decode is None else self.decode.buffers
            desired = hidden if self.mlp is None else self.mlp_input
            assert desired is not None
            pinned_projection(hidden, desired, buffers, self.table, base[0], self.captured, self.desired)
        return base

    def emit(
        self,
        hidden: torch.Tensor,
        desired: torch.Tensor,
        pools: list[torch.Tensor],
        table: torch.Tensor,
        output: torch.Tensor,
        captured: torch.Tensor,
        desired_capture: torch.Tensor,
    ) -> None:
        captured[: len(hidden)].copy_(hidden)
        if self.mlp is not None:
            desired_capture[: len(hidden)].copy_(desired)
        if self.decode is not None:
            weight = base_weight(self.module, self.binding.reads_weight, hidden.device)
            result = self.decode.run(
                hidden,
                captured[: len(hidden)] if self.mlp is None else desired_capture[: len(hidden)],
                self.embedding,
                weight[self.offset : self.offset + self.width],
            )
            columns = output[:, self.offset : self.offset + self.width]
            columns.copy_((columns + result * self.binding.scale).to(output.dtype))
            return
        if self.call_rows is None:
            count = next(size for size in self.programs if size >= hidden.shape[0])
            padded = torch.nn.functional.pad(hidden, (0, 0, 0, count - hidden.shape[0]))
            result = self.read(padded, pools, table, count)
        else:
            result = self.read_rows(hidden, pools, table, self.call_rows)
        output[:, self.offset : self.offset + self.width].add_(
            (result[: hidden.shape[0]] * self.binding.scale).to(output.dtype)
        )

    def read(self, hidden: torch.Tensor, pools: list[torch.Tensor], table: torch.Tensor, count: int) -> torch.Tensor:
        values = (hidden, *self.ports(table, count), *pools)
        return self.programs[count](*values)[0]

    def read_rows(
        self, hidden: torch.Tensor, pools: list[torch.Tensor], table: torch.Tensor, packed: torch.Tensor
    ) -> torch.Tensor:
        """Read the pinned delta for every actual projection call, in its device-selected row order.

        Waves and adaptive buckets use one block per lane. Long spans retain the standard request blocks so
        gathering a large dense adapter does not copy its matrix once per token. Both use projection_read's
        existing arithmetic and bounded scratch; graph replay never reads an index on the host.
        """
        slots, rows, owners = self.ports(table, self.rows)
        owners = owners.index_select(0, packed)
        lanes = torch.arange(len(hidden), device=hidden.device)
        if len(hidden) <= self.requests:
            slots = slots.index_select(0, owners.clamp_min(0) // rows.shape[1])
            rows = lanes[:, None]
            owners = lanes.masked_fill(owners < 0, -1)
        else:
            inverse = torch.zeros(self.rows, device=hidden.device, dtype=torch.long)
            inverse.index_copy_(0, packed, lanes)
            rows = inverse.index_select(0, rows.flatten()).view_as(rows)
        return self.programs[self.rows](hidden, slots, rows, owners, *pools)[0]

    def prepare(self, field_slots: Slots, starts: Slots, total: int) -> None:
        host = self.host
        maximum = projection_blocks(self.rows, self.requests)
        slots = host[: maximum * len(self.pools)].reshape(maximum, len(self.pools))
        positions = host[maximum * len(self.pools) : -self.rows].reshape(maximum, ROWS_PER_BLOCK)
        owners = host[-self.rows :]
        owners.fill(-1)
        # A cached wave can replay under different packed batch sizes; its table geometry must stay fixed.
        count = self.rows if self.in_program else next(size for size in capacities(self.rows) if size >= total)
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
        self.hook.close()
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
        math: OverlayTransition | None = None,
        in_program: bool = False,
    ) -> None:
        self.readers = tuple(
            ProjectionReader(decoder, binding, arena, rows, requests, budget, math=math, in_program=in_program)
            for binding in bindings
        )
        self.phases = phases
        self.embedding = _EmbeddingRows(
            decoder,
            rows,
            arena.device,
            enabled=bool(phases or math) and any(binding.reads_embedding for binding in bindings),
        )
        self.math = math
        self.requests = requests
        self.transition: ProjectionTransition | None = None
        if math is not None and HiddenPhase.DECODE in math.phases:
            width = len(math.buffers)
            for reader in self.readers:
                target = math.targets.index(reader.binding.target)
                pools = tuple(
                    arena.field_pool(str(math.resource), LayoutFieldId(target * width + i)) for i in range(width)
                )
                invariants = {
                    name: views[target].read(arena.field_pool(resource, views[target].field), pooled=True)[0].clone()
                    for name, (resource, views) in math.invariants.items()
                }
                reader.decode = DecodeOverlay(
                    math,
                    target,
                    pools,
                    invariants,
                    requests,
                    arena.plan.timeline.tensor_policy,
                    next(
                        state.key_fields
                        for state in arena.plan.semantic_graph.states
                        if state.resource == math.resource
                    ),
                )
                reader.embedding = self.embedding.values
        self.sources: Sequence[PackedPhaseRow] = ()
        self.starts: Slots = np.empty(0, dtype=np.int64)
        self.total_rows = 0
        self.stream: torch.cuda.Stream | None = None
        self.readiness: CudaStreamReadiness | None = None
        self.apply_count = self.updated_apply_count = 0

    def reserve(self) -> None:
        """Acquire every host table before the invocation issues any GPU work."""
        for reader in self.readers:
            if reader.decode is None:
                reader.reserve()
            else:
                reader.decode.reserve()

    def prepare_decode(self, rows: PackedPhaseRowBatch, transition: ProjectionTransition) -> None:
        """Open private pages and stage every decode target before replay starts."""
        assert self.math is not None
        self.transition = transition
        self.apply_count += rows.total_rows * len(self.readers)
        batch = transition.prepare(self.math).for_resource(self.math.resource)
        identities = tuple(dict.fromkeys(rows.identities))
        self.sources = rows.inspect(np.arange(len(identities), dtype=np.int64))
        self.total_rows = rows.total_rows
        self.updated_apply_count += int(
            (batch.selection.refs.versions[list(batch.selection.state_indices)] > 0).sum()
        ) * len(self.readers)
        for reader in self.readers:
            assert reader.decode is not None
            reader.decode.prepare(batch, identities)

    def prepare(
        self,
        rows: PackedPhaseRowBatch,
        context: StateTaskContext[DataT],
        directory: RequestStateDirectory,
        transition: ProjectionTransition | None = None,
    ) -> None:
        self.stream = context.pin.stream
        assert self.stream is not None
        if self.readiness is not None:
            self.readiness.order_on(self.stream)
        if self.math is not None and HiddenPhase.DECODE in self.math.phases:
            assert transition is not None
            self.prepare_decode(rows, transition)
            return
        assert transition is None
        ordinals = rows.columns[1]
        self.starts = np.flatnonzero(np.append(True, ordinals[1:] != ordinals[:-1]))
        self.sources = rows.inspect(self.starts)
        self.total_rows = rows.total_rows
        for reader in self.readers:
            versions = context.read_versions(
                tuple(row.identity for row in self.sources), StateResource(reader.binding.resource), directory
            )
            pins = tuple(pin for pin, _ in versions)
            refs = StateRefs.of(tuple(ref for _, ref in versions))
            _, slots = context.arena.slots(pins, np.arange(len(pins), dtype=np.int64), refs)
            self.updated_apply_count += int(np.diff(np.append(self.starts, rows.total_rows)) @ (refs.versions > 0))
            for value in (*reader.pools, reader.table, reader.captured, reader.desired):
                value.record_stream(self.stream)
            reader.prepare(slots, self.starts, rows.total_rows)
        self.embedding.values.record_stream(self.stream)
        self.apply_count += rows.total_rows * len(self.readers)

    def complete_decode(self) -> None:
        """Settle replay's stable results through the ordinary projection task."""
        assert self.transition is not None
        operands: list[ProjectionOperands] = []
        updates: dict[tuple[RequestIdentity, Projection], UpdateOutput] = {}
        for reader in self.readers:
            assert reader.decode is not None
            count = self.requests
            for index, row in enumerate(self.sources):
                result = reader.decode.outputs[count][index]
                updates[row.identity, reader.binding.target] = result
                hidden = reader.captured[index : index + 1]
                operands.append(
                    ProjectionOperands(
                        row.identity,
                        reader.binding.target,
                        row.phase,
                        hidden,
                        hidden if reader.mlp is None else reader.desired[index : index + 1],
                        base_weight(reader.module, reader.binding.reads_weight, hidden.device)[
                            reader.offset : reader.offset + reader.width
                        ],
                        hidden.new_zeros((), dtype=torch.bool),
                        reader.embedding[index : index + 1]
                        if reader.binding.reads_embedding
                        else hidden.new_empty((0, 0)),
                    )
                )
        self.transition.complete(tuple(operands), updates)

    def take_operands(self) -> tuple[ProjectionOperands, ...]:
        if self.transition is not None:
            self.complete_decode()
            return ()
        if not self.sources or not self.phases:
            return ()
        spans = tuple(
            (row, first, last)
            for row, first, last in zip(
                self.sources, self.starts.tolist(), np.append(self.starts[1:], self.total_rows).tolist(), strict=True
            )
            if row.phase in self.phases
        )
        results: list[ProjectionOperands] = []
        for reader in self.readers if spans else ():
            weight = base_weight(reader.module, reader.binding.reads_weight, reader.captured.device)
            for row, first, last in spans:
                hidden = reader.captured[first:last].clone()
                results.append(
                    ProjectionOperands(
                        row.identity,
                        reader.binding.target,
                        row.phase,
                        hidden,
                        hidden if reader.mlp is None else reader.desired[first:last].clone(),
                        weight[reader.offset : reader.offset + reader.width],
                        hidden.new_full((), row.token_offset + last - first == row.prompt_length, dtype=torch.bool),
                        self.embedding.values[first:last].clone()
                        if reader.binding.reads_embedding
                        else hidden.new_empty((0, 0)),
                    )
                )
        return tuple(results)

    def clear(self) -> None:
        self.transition = None
        if self.sources and self.stream is not None:
            self.readiness = CudaStreamReadiness(self.stream)
        self.sources = ()

    def close(self) -> None:
        self.embedding.close()
        for reader in reversed(self.readers):
            reader.close()


class HybridParameterOverlay:
    """Eager prefill and device-counted decode share one exact projection-task lifecycle."""

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
        training = not uniform_decode(
            bool((rows.columns[3] == 1).any()),
            tuple(np.unique(rows.columns[4], return_counts=True)[1].tolist()),
        )
        for reader in self.pinned.readers:
            reader.enabled = not training
        if training:
            self.exact.prepare(rows, context, directory, transition)
        else:
            self.pinned.prepare(rows, context, directory, transition if HiddenPhase.DECODE in self.phases else None)

    def take_operands(self) -> tuple[ProjectionOperands, ...]:
        """Exact consumed training operands in forward; inactive reads produce no task operands."""
        self.pinned.take_operands()
        return self.exact.take_operands()

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
    in_program: bool = False,
    math: OverlayTransition | None = None,
) -> PinnedParameterOverlay | ExactParameterOverlay | HybridParameterOverlay:
    """Install each projection wrapper once, before compiling or capturing the model."""
    if (in_program and math is not None) or not uses_pinned_projection(in_forward=in_forward, graphs=graphs):
        return ExactParameterOverlay(decoder, bindings, arena.device, rows)
    pinned = PinnedParameterOverlay(
        decoder,
        bindings,
        arena,
        rows,
        requests,
        budget,
        phases=() if in_forward else phases,
        math=math if in_forward else None,
        in_program=in_program,
    )
    if not in_forward:
        return pinned
    return HybridParameterOverlay(pinned, ExactParameterOverlay(decoder, bindings, arena.device, rows), phases)
