"""Parameter projection integration for vLLM 0.27.1's deferred graph startup.

Depends on GPUModelRunner.load_model/profile_run/capture_model, the capture call
from Worker.compile_or_warm_up_model, Qwen2Model.do_not_compile,
ModelCudaGraphManager construction, VllmConfig.compute_hash and
aot_compile_hash_factors at the first deferred Qwen2Model call, and
EagerAdaptor.compile for source FX bodies inside the upstream CUDA graph wrappers.
Hybrid dispatch uses GPUModelRunner.update_requests and ModelCudaGraphManager.dispatch
before prepare_inputs, and Qwen2Model.do_not_compile to bypass compiled training bodies.
The upstream manager owns all capture/replay loops; PTTS owns every request version.
"""

from typing import Protocol, TypedDict

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode, VllmConfig
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM, Qwen2Model
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, ModelCudaGraphManager
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from pllm.authoring import BackendRequirements
from pllm.backends.torch_inductor import InductorTensorCompiler
from pllm.backends.vllm.parameter_overlay import ExactParameterOverlay
from pllm.config import ModelConfig, RunnerKind
from pllm.contracts.backend_attachment import AttachmentGraphMode, SignedBackendAttachment
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch
from pllm.contracts.parameter_overlay import CapturedParameterOperand, ParameterOverlayFieldBinding
from pllm.contracts.semantic_graph import StateResource
from pllm.contracts.tensor_policy import TensorPolicy
from pllm.execution.codec import cuda_graph_pool_bytes, decode_boundary, encode_boundary
from pllm.execution.cuda_stream import CudaStreamReadiness, PinnedStaging
from pllm.execution.overlay_task import ProjectionTransition
from pllm.execution.parameter_projection import (
    ROWS_PER_BLOCK,
    prepare_projection_read,
    projection_block_rows,
    projection_blocks,
)
from pllm.execution.ptts.page_bundle import PageBundleArena, Slots
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.region_execution import RegionInstructionExecution
from pllm.execution.state_execution import DataT, StateTaskContext

_BUDGET = "pllm_pinned_projection_v2_bytes"
_READERS: dict[int, "_ProjectionReader"] = {}


class _GraphConfiguration(Protocol):
    @property
    def additional_config(self) -> object: ...


def _graph_budget(config: _GraphConfiguration) -> int:
    value = decode_boundary(encode_boundary(config.additional_config), dict[str, int]).get(_BUDGET, 0)
    if value < 0:
        raise ValueError("Overlay graph memory must be a nonnegative byte budget")
    return value


def capacities(maximum: int) -> tuple[int, ...]:
    """Finite power-of-two capacity classes, ending at the declared bound."""
    return tuple(sorted({min(1 << i, maximum) for i in range(maximum.bit_length() + 1)}))


class GraphOptions(TypedDict, total=False):
    """The `LLM(...)` arguments that switch on deferred overlay compilation."""

    additional_config: dict[str, int]
    compilation_config: CompilationConfig


def graph_options(config: ModelConfig, requirements: BackendRequirements) -> GraphOptions:
    """Announce deferred overlay compilation before vLLM profiles the model."""
    if not config.cuda_graphs or not requirements.parameter_targets:
        return {}
    if config.runner is not RunnerKind.MRV2:
        raise ValueError("Graph overlays need the MRv2 runner")
    return {
        "additional_config": {_BUDGET: config.execution.capture_memory_bytes},
        "compilation_config": CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            # Capture the model's source CUDA operators; the shared region selects overlay kernels.
            backend="eager",
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=list(capacities(config.max_num_batched_tokens)),
            max_cudagraph_capture_size=config.max_num_batched_tokens,
        ),
    }


def graph_mode(budget: int) -> AttachmentGraphMode:
    """Describe only the intervention placement; ESamp's external tasks remain unchanged."""
    return AttachmentGraphMode.CUDA_GRAPH if budget > 0 else AttachmentGraphMode.EAGER


class _HybridGraphManager(ModelCudaGraphManager):
    """Keep upstream replay ownership; training batches use the unpadded source model."""

    def __init__(
        self,
        config: VllmConfig,
        device: torch.device,
        mode: CUDAGraphMode,
        decode_query_len: int,
        phases: tuple[HiddenPhase, ...],
    ) -> None:
        super().__init__(config, device, mode, decode_query_len)
        self.phases = phases
        self.eager = False

    def dispatch(
        self, num_reqs: int, num_tokens: int, uniform_token_count: int | None, num_active_loras: int
    ) -> BatchExecutionDescriptor:
        """Use an unpadded source batch whenever an authored phase trains."""
        if self.eager:
            return BatchExecutionDescriptor(CUDAGraphMode.NONE, num_tokens, num_reqs)
        return super().dispatch(num_reqs, num_tokens, uniform_token_count, num_active_loras)


class OverlayGraphRunner(GPUModelRunner):
    """Defer first compilation/capture until the ordinary workload installation completes."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, device)
        self.projection_model: Qwen2Model | None = None
        self.overlay_graph_bytes = _graph_budget(vllm_config)

    def load_model(self, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights)  # pyright: ignore[reportUnknownMemberType]
        if graph_mode(self.overlay_graph_bytes) is AttachmentGraphMode.CUDA_GRAPH:
            assert isinstance(self.model, Qwen2ForCausalLM)
            self.projection_model = self.model.model
            self.projection_model.do_not_compile = True

    def capture_model(self) -> int:
        if self.projection_model is not None and self.projection_model.do_not_compile:
            return 0
        return super().capture_model()

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        """Choose placement from CPU counters after upstream request updates."""
        super().update_requests(scheduler_output)
        manager = self.cudagraph_manager
        if isinstance(manager, _HybridGraphManager) and scheduler_output.num_scheduled_tokens:
            indices = np.fromiter(
                (self.req_states.req_id_to_index[key] for key in scheduler_output.num_scheduled_tokens), np.int64
            )
            start = self.req_states.num_computed_tokens_np[indices]
            end = start + np.fromiter(scheduler_output.num_scheduled_tokens.values(), np.int64)
            prompt = self.req_states.prompt_len.np[indices]
            manager.eager = bool(
                (HiddenPhase.PREFILL in manager.phases and (start < prompt).any())
                or (HiddenPhase.DECODE in manager.phases and (end > prompt).any())
            )
            assert self.projection_model is not None
            self.projection_model.do_not_compile = manager.eager

    def finish_overlay_graphs(self, attachment: SignedBackendAttachment, phases: tuple[HiddenPhase, ...]) -> None:
        """Capture only the bound model; dummy tables have no request ownership or live destinations."""
        if self.projection_model is None:
            return
        # AOT loads before inspecting wrappers: the complete bound plan must enter its key.
        self.vllm_config.additional_config = {
            _BUDGET: self.overlay_graph_bytes,
            "pllm_pinned_projection_plan": attachment.physical.bundle_signature,
        }
        manager = self.cudagraph_manager
        assert manager is not None
        self.cudagraph_manager = (
            _HybridGraphManager(self.vllm_config, self.device, manager.cudagraph_mode, manager.decode_query_len, phases)
            if attachment.inputs.plan.semantic_graph.updates_in_forward
            else ModelCudaGraphManager(self.vllm_config, self.device, manager.cudagraph_mode, manager.decode_query_len)
        )
        self.projection_model.do_not_compile = False
        self.capture_model()
        used = cuda_graph_pool_bytes(current_platform.get_global_graph_pool(), self.device.index or 0)
        if used > self.overlay_graph_bytes:
            raise ValueError(f"Overlay model graphs need {used} bytes; increase execution.capture_memory_bytes")


@torch.library.custom_op("pllm::pinned_projection", mutates_args={"output", "captured"})
def pinned_projection(
    hidden: torch.Tensor,
    pools: list[torch.Tensor],
    table: torch.Tensor,
    output: torch.Tensor,
    captured: torch.Tensor,
) -> None:
    """Execute the same prepared tensor region eagerly or during upstream capture."""
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
        model: Qwen2ForCausalLM,
        binding: ParameterOverlayFieldBinding,
        arena: PageBundleArena,
        rows: int,
        requests: int,
        budget: int,
        policy: TensorPolicy,
    ) -> None:
        from pllm.backends.vllm.parameter_overlay import locate_parameter

        self.module, self.offset, self.width, self.mlp = locate_parameter(model, binding.target)
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
        self.captured = self.module.weight.new_zeros((rows, self.module.weight.shape[1]))
        self.desired = self.module.weight.new_empty((rows, self.width)) if self.mlp is not None else self.captured
        self.staging = PinnedStaging(capacity=self.table.numel(), slots=2)
        self.host = torch.empty(0, dtype=torch.long)
        self.programs: dict[int, RegionInstructionExecution] = {}
        for count in capacities(rows):
            with torch.inference_mode(False), FakeTensorMode(allow_non_fake_inputs=True):
                examples = (
                    self.captured.new_empty((count, self.captured.shape[1])),
                    *self.ports(self.table, count),
                    *self.pools,
                )
            self.programs[count] = prepare_projection_read(
                examples,
                binding.operator,
                budget
                + sum(
                    pool.untyped_storage().nbytes()
                    for pool in {pool.untyped_storage().data_ptr(): pool for pool in self.pools}.values()
                ),
                policy,
                compiler=InductorTensorCompiler(autotune_at_compile_time=False),
            )
            # Warm with actual readonly pools; compiler-created fake arena copies can exceed residency.
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
        names = ("hidden", "slots", "rows", "owners", *(f"pool{i}" for i in range(len(pools))))
        result = self.programs[count].run(dict(zip(names, values, strict=True))).values[0]
        assert isinstance(result, torch.Tensor)
        return result

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
        model: Qwen2ForCausalLM,
        bindings: tuple[ParameterOverlayFieldBinding, ...],
        arena: PageBundleArena,
        rows: int,
        requests: int,
        budget: int,
        policy: TensorPolicy,
        *,
        phases: tuple[HiddenPhase, ...],
    ) -> None:
        self.readers = tuple(
            _ProjectionReader(model, binding, arena, rows, requests, budget, policy) for binding in bindings
        )
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
            resource = StateResource(reader.binding.resource)
            selection = context.selection.for_resource(resource)
            if selection.keys == (None,):
                pins = tuple(directory.inference_pin(row.identity, self.stream) for row in self.sources)
                refs = StateRefs.of(
                    tuple(next(ref for ref in pin.refs if context.arena.root(ref).resource == resource) for pin in pins)
                )
            else:
                # Resolve every request at once: decode replays prepare this table before each graph launch.
                index = dict(
                    zip((row.identity for row in context.selection.rows), selection.state_indices, strict=True)
                )
                entry = tuple(context.selection.refs[index[row.identity]] for row in self.sources)
                refs = StateRefs.of(context.current_owned_refs(entry))
                published = tuple(ref for ref, original in zip(refs, entry, strict=True) if ref != original)
                borrowed = context.borrow_published(published) if published else context.pin
                pins = tuple(
                    context.pin if ref == original else borrowed for ref, original in zip(refs, entry, strict=True)
                )
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
            weight = reader.module.weight[reader.offset : reader.offset + reader.width].detach()
            for row, first, last in spans:
                hidden = reader.captured[first:last].clone()
                results.append(
                    CapturedParameterOperand(
                        row.identity,
                        reader.binding.target,
                        row.phase,
                        hidden,
                        hidden if reader.mlp is None else reader.desired[first:last].clone(),
                        weight,
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
        """Select the same eligibility used by dispatch from Rust-planned row facts."""
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
    model: Qwen2ForCausalLM,
    bindings: tuple[ParameterOverlayFieldBinding, ...],
    arena: PageBundleArena,
    rows: int,
    requests: int,
    budget: int,
    policy: TensorPolicy,
    phases: tuple[HiddenPhase, ...],
    *,
    in_forward: bool,
    graphs: bool,
    routed: bool = False,
) -> PinnedParameterOverlay | ExactParameterOverlay | HybridParameterOverlay:
    """Install each projection wrapper once, before compiling or capturing the model."""
    if routed or in_forward and not graphs:
        return ExactParameterOverlay(model, bindings)
    pinned = PinnedParameterOverlay(
        model, bindings, arena, rows, requests, budget, policy, phases=() if in_forward else phases
    )
    return HybridParameterOverlay(pinned, ExactParameterOverlay(model, bindings), phases) if in_forward else pinned
