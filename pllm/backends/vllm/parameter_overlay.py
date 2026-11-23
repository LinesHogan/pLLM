"""Parameter overlays at vLLM projections, installed once by VllmSession and VllmCustomForward.

Each target projection gets one ProjectionReader hook. Training steps without CUDA graphs run the overlay task
eagerly inside the forward (exact segments); otherwise the hook reads PTTS pools through one custom op that vLLM
can capture. ParameterOverlay owns the readers' per-invocation lifecycle.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.model_executor.layers.linear import LinearBase, QKVParallelLinear, RowParallelLinear

from pllm.backends.vllm import compat
from pllm.backends.vllm.architecture import Decoder, LinearOutput, frozen_weight
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
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.cuda_stream import CudaStreamReadiness, PinnedStaging
from pllm.execution.layer_route import RouteSlotGather, gate_owners_by_layer, gated_dense_add, gated_low_rank_add
from pllm.execution.overlay_batch import DecodeOverlay
from pllm.execution.overlay_task import OverlayTransition, ProgramOverlay, ProjectionTask
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
from pllm.execution.state_execution import StateTaskContext
from pllm.execution.tensor_frame import UpdateOutput
from pllm.execution.torch_private import blas_workspace_bytes


def inspect_parameter_targets(
    decoder: Decoder, targets: tuple[Projection, ...], dtype: torch.dtype
) -> tuple[ProjectionShape, ...]:
    """Inspect fused output slices and vLLM's feature/head partitions before installation."""
    specs: list[ProjectionShape] = []
    for index, target in enumerate(targets):
        module, offset, size, _ = decoder.projection(target)
        if (
            target in targets[:index]
            or not module.return_bias
            or module.params_dtype is not dtype
            or offset + size > module.output_size_per_partition
        ):
            raise ValueError("parameter overlay needs unique attention or dense MLP projections matching model dtype")
        row = target.name == "down_proj"
        partitions = module.tp_size
        if isinstance(module, QKVParallelLinear) and target.name in ("k_proj", "v_proj"):
            partitions //= module.num_kv_head_replicas
        specs.append(
            ProjectionShape(
                target,
                module.input_size_per_partition,
                size,
                input_partitions=partitions if row else 1,
                output_partitions=1 if row else partitions,
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


def capacities(maximum: int) -> tuple[int, ...]:
    """Finite power-of-two capacity classes, ending at the declared bound."""
    return tuple(sorted({min(1 << i, maximum) for i in range(maximum.bit_length() + 1)}))


def uniform_decode(prefill: bool, counts: tuple[int, ...]) -> bool:
    """Whether every scheduled request contributes one decode row, without a prefill or replay span."""
    return not prefill and all(count == 1 for count in counts)


# Pinned readers by the address of their index table; see pinned_projection.
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
    """Execute the same prepared tensor callable eagerly or during upstream capture.

    A custom op takes only tensors and scalars, and vLLM's compiled (and AOT-cached) model graph bakes scalars in,
    so the reader travels as `table`: a runtime graph input whose address finds the reader without specializing
    the graph on one, as model_site does for sessions. A pinned reader registers in ProjectionReader.__init__ and
    leaves in close(); it owns the table meanwhile, so the address cannot be reused while registered.
    """
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


@dataclass(frozen=True, slots=True)
class _Segment:
    """One request's packed rows of one phase in this invocation."""

    source: PackedPhaseRow
    rows: torch.Tensor
    prompt_end: torch.Tensor
    updated: bool


class ProjectionReader:
    """One target's hook: exact segments run the overlay task in the forward, otherwise the pinned custom op reads.

    Only a pinned reader owns stable tensor ports; the invocation supplies all pin-authorized indices.
    """

    def __init__(
        self,
        decoder: Decoder,
        binding: ParameterOverlayFieldBinding,
        arena: PageBundleArena,
        embedding: torch.Tensor,
        rows: int,
        requests: int,
        budget: int,
        *,
        pinned: bool,
        math: OverlayTransition | None,
        in_program: bool,
        captures_operands: bool = False,
    ) -> None:
        self.module, self.offset, self.width, self.mlp = decoder.projection(binding.target)
        self.hook = ProjectionHook(self.module)
        self.forward_base = self.hook.base
        self.mlp_base = None if self.mlp is None else self.mlp.forward
        self.binding, self.rows, self.requests = binding, rows, requests
        self.pinned, self.in_program, self.embedding = pinned, in_program, embedding
        self.captures_operands = captures_operands
        self.mlp_input: torch.Tensor | None = None
        self.apply_count = self.updated_apply_count = 0
        # This invocation's exact segments and their consumer: the step's task, or a custom forward's lanes.
        self.segments: tuple[_Segment, ...] = ()
        self.task: ProjectionTask | None = None
        self.row_overlays: tuple[ProgramOverlay, ...] = ()
        self.call_rows: torch.Tensor | None = None
        self.decode: DecodeOverlay | None = None
        self.programs: dict[int, Callable[..., tuple[torch.Tensor, ...]]] = {}
        ports = (requests if math is not None else rows) if pinned else 0
        self.pools = [view.read(arena.field_pool(binding.resource, view.field), pooled=True) for view in binding.fields]
        self.table = torch.zeros(
            projection_blocks(rows, requests) * (ROWS_PER_BLOCK + len(binding.fields)) + rows if pinned else 0,
            device=embedding.device,
            dtype=torch.long,
        )
        self.table[-rows:].fill_(-1)
        inputs = self.module.input_size_per_partition
        self.captured = embedding.new_zeros((ports, inputs), dtype=self.module.params_dtype)
        self.desired = self.captured.new_empty((ports, self.width) if self.mlp is not None else (0, 0))
        self.routed_mid = (
            torch.empty((rows, self.pools[0].shape[1]), dtype=torch.float32, device=embedding.device)
            if pinned
            and binding.route is not None
            and binding.operator is ParameterOverlayOperator.LOW_RANK_ADD
            and embedding.is_cuda
            and self.module.params_dtype != torch.float64
            and self.pools[0].shape[1] <= 256
            else None
        )
        self.staging = PinnedStaging(capacity=self.table.numel(), slots=2 if pinned else 0)
        self.host = np.empty(0, dtype=np.int64)
        if pinned:
            self.prepare_programs(budget, math)
            _READERS[self.table.data_ptr()] = self
        self.hook.install(self.forward)
        if self.mlp is not None:
            self.mlp.forward = self.forward_mlp

    def prepare_programs(self, budget: int, math: OverlayTransition | None) -> None:
        """Fund and warm every row bucket against the actual readonly pools before graph capture."""
        blas_bytes = blas_workspace_bytes()
        counts = (
            ()
            if math is not None and HiddenPhase.DECODE in math.phases
            else capacities(self.requests if math is not None else self.rows)
        )
        for count in counts:
            with torch.inference_mode(False), FakeTensorMode(allow_non_fake_inputs=True):
                examples = (
                    self.captured.new_empty((count, self.captured.shape[1])),
                    *self.ports(self.table, count),
                    *self.pools,
                )
            self.programs[count] = prepare_projection_read(
                examples, self.binding.operator, budget, blas_workspace_bytes=blas_bytes
            )
            self.read(self.captured[:count], self.pools, self.table, count)

    def reserve(self) -> None:
        if self.decode is None:
            self.host = self.staging.take(self.table.numel())
        else:
            self.decode.reserve()

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

    def weight(self, device: torch.device) -> torch.Tensor:
        """This target's rows of operand("weight"), borrowed only when the function reads it."""
        return base_weight(self.module, self.binding.reads_weight, device)[self.offset : self.offset + self.width]

    def operands(
        self,
        row: PackedPhaseRow,
        select: Callable[[torch.Tensor], torch.Tensor],
        hidden: torch.Tensor,
        desired: torch.Tensor | None,
        weight: torch.Tensor,
        prompt_end: torch.Tensor,
    ) -> ProjectionOperands:
        """One request's operands, selected by `select` from this call's input, MLP input and embedding rows."""
        selected = select(hidden)
        return ProjectionOperands(
            row.identity,
            self.binding.target,
            row.phase,
            selected,
            selected if desired is None else select(desired),
            weight,
            prompt_end,
            select(self.embedding) if self.binding.reads_embedding else selected.new_empty((0, 0)),
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
        if self.segments:
            self.run_segments(hidden.detach(), base[0])
        elif self.pinned:
            desired = hidden if self.mlp is None else self.mlp_input
            assert desired is not None
            buffers = self.pools if self.decode is None else self.decode.buffers
            pinned_projection(hidden, desired, buffers, self.table, base[0], self.captured, self.desired)
        return base

    def run_segments(self, hidden: torch.Tensor, output: torch.Tensor) -> None:
        """Run the overlay on this call's rows and add the residuals into the GEMM's fresh output."""
        weight = self.weight(hidden.device)
        desired = None if self.mlp_input is None else self.mlp_input.detach()
        # A replay span can contain both phases of one request. Finish its prompt
        # transition (including prompt-end flush) before reading its decode state.
        phases = tuple(
            phase
            for phase in (HiddenPhase.PREFILL, HiddenPhase.DECODE)
            if any(segment.source.phase is phase for segment in self.segments)
        )
        columns = output[:, self.offset : self.offset + self.width]
        for phase in phases:
            segments = tuple(segment for segment in self.segments if segment.source.phase is phase)
            operands = tuple(
                self.operands(
                    segment.source,
                    partial(torch.index_select, dim=0, index=segment.rows),
                    hidden,
                    desired,
                    weight,
                    segment.prompt_end,
                )
                for segment in segments
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
                assert self.task is not None
                residuals = self.task.run(operands, complete=phase is phases[-1])
            for segment in segments:
                delta = residuals[segment.source.identity, self.binding.target] * self.binding.scale
                columns.index_copy_(0, segment.rows, (columns.index_select(0, segment.rows) + delta).to(output.dtype))
                self.apply_count += len(segment.rows)
                self.updated_apply_count += len(segment.rows) if segment.updated else 0

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
        if self.captures_operands or self.decode is not None:
            captured[: len(hidden)].copy_(hidden)
            if self.mlp is not None:
                desired_capture[: len(hidden)].copy_(desired)
        if self.decode is not None:
            result = self.decode.run(
                hidden,
                captured[: len(hidden)] if self.mlp is None else desired_capture[: len(hidden)],
                self.embedding,
                self.weight(hidden.device),
            )
            columns = output[:, self.offset : self.offset + self.width]
            columns.copy_((columns + result * self.binding.scale).to(output.dtype))
            return
        if self.binding.route is not None and hidden.is_cuda and hidden.dtype != torch.float64:
            count = (
                self.rows
                if self.call_rows is not None
                else next(size for size in self.programs if size >= hidden.shape[0])
            )
            slots, rows_table, owners = self.ports(table, count)
            active_owners = (
                owners[: hidden.shape[0]] if self.call_rows is None else owners.index_select(0, self.call_rows)
            )
            if self.routed_mid is not None:
                gated_low_rank_add(
                    hidden,
                    slots,
                    active_owners,
                    pools[0],
                    pools[1],
                    self.routed_mid[: hidden.shape[0]],
                    output,
                    self.offset,
                    self.width,
                    rows_table.shape[1],
                    self.binding.scale,
                )
                return
            if self.binding.operator is ParameterOverlayOperator.DENSE_ADD:
                gated_dense_add(
                    hidden,
                    slots,
                    active_owners,
                    pools[0],
                    output,
                    self.offset,
                    self.width,
                    rows_table.shape[1],
                    self.binding.scale,
                )
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

    def clear(self) -> None:
        self.segments, self.task, self.row_overlays = (), None, ()

    def close(self) -> None:
        self.hook.close()
        if self.mlp is not None and self.mlp_base is not None:
            self.mlp.forward = self.mlp_base
        if self.pinned:
            _READERS.pop(self.table.data_ptr())


class ParameterOverlay:
    """All target readers of one workload's overlay and their per-invocation lifecycle.

    In-forward training without CUDA graphs, and every custom-forward program step, runs exact segments. With
    graphs, in-forward training runs exact segments on prefill and mixed steps and replays the pinned reads on
    uniform decode steps (training decode rows inside the graph when the overlay trains on decode). Any other
    overlay reads pinned on every step and hands forward_pinned training operands to the task after the forward.
    """

    def __init__(
        self,
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
    ) -> None:
        """Install each projection wrapper once, before compiling or capturing the model."""
        program = in_program and math is not None
        self.pinned = not program and uses_pinned_projection(in_forward=in_forward, graphs=graphs)
        self.exact = program or in_forward
        self.math = math if in_forward else None
        # forward_pinned training reads its operands back from the pinned ports after the forward.
        self.phases = () if in_forward else phases
        self.embedding = _EmbeddingRows(
            decoder, rows, arena.device, enabled=any(binding.reads_embedding for binding in bindings)
        )
        self.readers = tuple(
            ProjectionReader(
                decoder,
                binding,
                arena,
                self.embedding.values,
                rows,
                requests,
                budget,
                pinned=self.pinned,
                math=self.math,
                in_program=in_program,
                captures_operands=bool(self.phases),
            )
            for binding in bindings
        )
        if self.pinned and self.math is not None and HiddenPhase.DECODE in self.math.phases:
            self.install_decode(arena, self.math, requests)
        self.requests, self.device = requests, arena.device
        self._route_gather = (
            RouteSlotGather(rows, arena.device) if any(binding.route is not None for binding in bindings) else None
        )
        self._route_buffers: dict[tuple[str, int, bool], torch.Tensor] = {}
        self._rows_capacity = rows
        self.operand_staging = PinnedStaging()
        self.task: ProjectionTask | None = None
        self.sources: Sequence[PackedPhaseRow] = ()
        self.starts: Slots = np.empty(0, dtype=np.int64)
        self.total_rows = 0
        self.stream: torch.cuda.Stream | None = None
        self.readiness: CudaStreamReadiness | None = None

    def install_decode(self, arena: PageBundleArena, math: OverlayTransition, requests: int) -> None:
        """Give each reader its device-counted decode transition over the task's private slots."""
        width = len(math.buffers)
        key_fields = next(
            state.key_fields for state in arena.plan.semantic_graph.states if state.resource == math.resource
        )
        immutable = {
            str(state.resource)
            for state in arena.plan.semantic_graph.states
            if state.immutable and str(state.resource) in arena.owned_resources
        }
        mutable = frozenset(name for name, (resource, _) in math.invariants.items() if resource not in immutable)
        batch_invariant = compat.batch_invariant_enabled()
        for reader in self.readers:
            target = math.targets.index(reader.binding.target)
            pools = tuple(arena.field_pool(str(math.resource), LayoutFieldId(target * width + i)) for i in range(width))
            invariants = {
                name: (
                    view := views[target].read(arena.field_pool(resource, views[target].field), pooled=True)[0],
                    view if resource in immutable else view.clone(),
                )[1]
                for name, (resource, views) in math.invariants.items()
            }
            reader.decode = DecodeOverlay(
                math,
                target,
                pools,
                invariants,
                requests,
                arena.plan.timeline.tensor_policy,
                key_fields,
                mutable_invariants=mutable,
                batch_invariant=batch_invariant,
            )

    @property
    def apply_count(self) -> int:
        return sum(reader.apply_count for reader in self.readers)

    @property
    def updated_apply_count(self) -> int:
        return sum(reader.updated_apply_count for reader in self.readers)

    def reserve(self) -> None:
        """Acquire every host table before the invocation issues any GPU work."""
        if self.pinned:
            for reader in self.readers:
                reader.reserve()

    def prepare(
        self,
        rows: PackedPhaseRowBatch,
        context: StateTaskContext,
        directory: RequestStateDirectory,
        task: ProjectionTask | None = None,
    ) -> None:
        """Select exact segments or pinned reads from the labelled row facts, as vLLM's graph dispatch does."""
        if self.exact and not (
            self.pinned
            and uniform_decode(
                bool((rows.phases == 1).any()),
                tuple(np.unique(rows.identity_indices, return_counts=True)[1].tolist()),
            )
        ):
            assert task is not None
            self.prepare_segments(rows, context, task)
        else:
            self.prepare_pinned(rows, context, directory, task)

    def prepare_pinned(
        self,
        rows: PackedPhaseRowBatch,
        context: StateTaskContext,
        directory: RequestStateDirectory,
        task: ProjectionTask | None,
    ) -> None:
        """Stage the pinned ports, or the in-graph decode transition, after the previous invocation's reads."""
        self.stream = context.pin.stream
        assert self.stream is not None
        if self.readiness is not None:
            self.readiness.order_on(self.stream)
        if self.readers[0].decode is None:
            self.prepare_reads(rows, context, directory)
        else:
            assert task is not None
            self.prepare_decode(rows, task)

    def prepare_segments(self, rows: PackedPhaseRowBatch, context: StateTaskContext, task: ProjectionTask) -> None:
        """Upload each request's packed rows per phase and its prompt-end flag for the eager forward."""
        grouped: dict[tuple[RequestIdentity, HiddenPhase], list[PackedPhaseRow]] = {}
        for row in rows.rows:
            grouped.setdefault((row.identity, row.phase), []).append(row)
        table = torch.empty((rows.total_rows + len(grouped),), dtype=torch.long, device=self.device)
        with self.operand_staging.fill(table) as host:
            host[: rows.total_rows] = [row.packed_row_index for source in grouped.values() for row in source]
            host[rows.total_rows :] = [
                source[-1].token_offset == source[-1].prompt_length - 1 for source in grouped.values()
            ]
        indices = torch.split(table[: rows.total_rows], [len(source) for source in grouped.values()])
        prompt_ends = table[rows.total_rows :].bool().unbind()
        for reader in self.readers:
            resource = StateResource(reader.binding.resource)
            reader.task = task
            reader.segments = tuple(
                _Segment(source[0], index, prompt_end, context.selection.ref_for(identity, resource).version_id > 0)
                for ((identity, _), source), index, prompt_end in zip(
                    grouped.items(), indices, prompt_ends, strict=True
                )
            )

    def prepare_wave(
        self,
        rows: Sequence[PackedPhaseRow],
        row_overlays: tuple[ProgramOverlay, ...],
        indices: tuple[torch.Tensor, ...],
    ) -> None:
        """Use one projection call across the wave and request-local update operands."""
        for row, row_overlay, index in zip(rows, row_overlays, indices, strict=True):
            row_overlay.prompt_end.fill_(row.token_offset + index.numel() == row.prompt_length)
        for reader in self.readers:
            reader.row_overlays = row_overlays
            reader.segments = tuple(
                _Segment(
                    row,
                    index,
                    row_overlay.prompt_end,
                    row_overlay.batch.after_refs[
                        int(row_overlay.batch.selection.state_indices[row_overlay.row])
                    ].version_id
                    > 0,
                )
                for row, row_overlay, index in zip(rows, row_overlays, indices, strict=True)
            )

    def prepare_decode(self, rows: PackedPhaseRowBatch, task: ProjectionTask) -> None:
        """Open private pages and stage every decode target before replay starts."""
        assert self.math is not None
        self.task = task
        prepared, written = task.prepare(self.math)
        batch = prepared.for_resource(self.math.resource)
        identities = tuple(dict.fromkeys(rows.identities))
        self.sources = rows.inspect(np.arange(len(identities), dtype=np.int64))
        updated = int((batch.selection.refs.versions[list(batch.selection.state_indices)] > 0).sum())
        for reader in self.readers:
            assert reader.decode is not None
            reader.apply_count += rows.total_rows
            reader.updated_apply_count += updated
            reader.decode.prepare(batch, identities, written)

    def prepare_reads(
        self, rows: PackedPhaseRowBatch, context: StateTaskContext, directory: RequestStateDirectory
    ) -> None:
        """Pin each request's readable version and upload the readers' block tables."""
        assert self.stream is not None
        ordinals = rows.request_ordinals
        self.starts = np.flatnonzero(np.append(True, ordinals[1:] != ordinals[:-1]))
        self.sources = rows.inspect(self.starts)
        self.total_rows = rows.total_rows
        routed_layers: dict[tuple[str, int, bool], torch.Tensor] = {}
        packed_range = np.arange(rows.total_rows, dtype=np.int64)
        for reader in self.readers:
            route = reader.binding.route
            latest = route is not None and route.refresh == "step"
            versions = context.read_versions(
                tuple(row.identity for row in self.sources),
                StateResource(reader.binding.resource),
                directory,
                latest=latest,
            )
            pins = tuple(pin for pin, _ in versions)
            refs = StateRefs.of(tuple(ref for _, ref in versions))
            if latest:
                pin = context.pin_for(refs)
                pins = (pin,) * len(refs)
                for row, ref in zip(self.sources, refs, strict=True):
                    context.injected_refs.setdefault(row.identity, set()).add(ref)
            _, slots = context.arena.slots(pins, np.arange(len(pins), dtype=np.int64), refs)
            reader.apply_count += rows.total_rows
            reader.updated_apply_count += int(np.diff(np.append(self.starts, rows.total_rows)) @ (refs.versions > 0))
            for value in (*reader.pools, reader.table, reader.captured, reader.desired):
                value.record_stream(self.stream)
            if reader.routed_mid is not None:
                reader.routed_mid.record_stream(self.stream)
            reader.prepare(slots, self.starts, rows.total_rows)
            if route is not None and rows.total_rows > 0:
                assert self._route_gather is not None and reader.binding.target.layer is not None
                route_key = (route.resource, int(route.field.field), latest)
                if route_key not in routed_layers:
                    if route_key not in self._route_buffers:
                        self._route_buffers[route_key] = torch.full(
                            (self._rows_capacity,), -1, dtype=torch.long, device=self.device
                        )
                    buf = self._route_buffers[route_key]
                    buf.record_stream(self.stream)
                    self._route_gather.gather(
                        rows,
                        packed_range,
                        StateResource(route.resource),
                        route.field,
                        context,
                        directory,
                        latest=latest,
                        out=buf,
                    )
                    routed_layers[route_key] = buf
                gate_owners_by_layer(
                    reader.table[-reader.rows :],
                    routed_layers[route_key],
                    rows.total_rows,
                    reader.binding.target.layer,
                )
        self.embedding.values.record_stream(self.stream)

    def take_operands(self) -> tuple[ProjectionOperands, ...]:
        """Settle in-graph decode results, or return forward_pinned training operands for the task."""
        assert all(reader.task is None or not reader.task.pending for reader in self.readers)
        if self.task is not None:
            self.complete_decode(self.task)
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
            weight = reader.weight(reader.captured.device)
            for row, first, last in spans:
                results.append(
                    reader.operands(
                        row,
                        lambda value: value[first:last].clone(),
                        reader.captured,
                        None if reader.mlp is None else reader.desired,
                        weight,
                        reader.captured.new_full(
                            (), row.token_offset + last - first == row.prompt_length, dtype=torch.bool
                        ),
                    )
                )
        return tuple(results)

    def complete_decode(self, task: ProjectionTask) -> None:
        """Settle replay's stable results through the ordinary projection task."""
        operands: list[ProjectionOperands] = []
        updates: dict[tuple[RequestIdentity, Projection], UpdateOutput] = {}
        for reader in self.readers:
            assert reader.decode is not None
            weight = reader.weight(reader.captured.device)
            prompt_end = reader.captured.new_zeros((), dtype=torch.bool)
            for index, row in enumerate(self.sources):
                updates[row.identity, reader.binding.target] = reader.decode.outputs[self.requests][index]
                operands.append(
                    reader.operands(
                        row,
                        lambda value: value[index : index + 1],
                        reader.captured,
                        None if reader.mlp is None else reader.desired,
                        weight,
                        prompt_end,
                    )
                )
        task.complete(tuple(operands), updates)

    def clear(self) -> None:
        for reader in self.readers:
            reader.clear()
        self.task = None
        if self.sources and self.stream is not None:
            self.readiness = CudaStreamReadiness(self.stream)
        self.sources = ()

    def close(self) -> None:
        """Unwind the wrappers in reverse installation order."""
        self.clear()
        self.embedding.close()
        for reader in reversed(self.readers):
            reader.close()
