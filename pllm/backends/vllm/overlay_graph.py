"""Pinned projection integration for vLLM 0.27.1's graph startup and descriptors.

Depends on GPUModelRunner.load_model/profile_run/capture_model, the capture call
from Worker.compile_or_warm_up_model, Qwen2Model.do_not_compile,
ModelCudaGraphManager._init_candidates/run_pw_graph, and ForwardContext's batch descriptor.
The upstream manager owns all capture/replay loops; PTTS owns every request version.
"""

from dataclasses import replace
from typing import Protocol

import numpy as np
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context, override_forward_context
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM, Qwen2Model
from vllm.platforms import current_platform
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadata
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from pllm.authoring import BackendRequirements
from pllm.config import ModelConfig, RunnerKind
from pllm.contracts.backend_attachment import AttachmentGraphMode
from pllm.contracts.packed_rows import PackedPhaseRowBatch
from pllm.contracts.parameter_overlay import CapturedParameterOperand, ParameterOverlayFieldBinding
from pllm.execution.codec import cuda_graph_pool_bytes, decode_boundary, encode_boundary
from pllm.execution.cuda_stream import CudaStreamReadiness, PinnedStaging
from pllm.execution.overlay_task import ProjectionTransition
from pllm.execution.parameter_projection import prepare_head_read
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.region_execution import RegionInstructionExecution
from pllm.execution.state_execution import DataT, StateTaskContext

_BUDGET = "pllm_pinned_head_graph_v1_bytes"
_READERS: dict[int, "PinnedGraphOverlay"] = {}


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


def graph_options(config: ModelConfig, requirements: BackendRequirements) -> dict[str, object]:
    """Announce deferred overlay compilation before vLLM profiles the model."""
    if not config.cuda_graphs or not requirements.parameter_targets:
        return {}
    if not config.execution.allow_numerical_transforms:
        raise ValueError(
            "Graph checkpoint needs numerical transforms for the compiled head read; use eager for source arithmetic"
        )
    if config.runner is not RunnerKind.MRV2:
        raise ValueError("Graph overlays need the MRv2 runner")
    return {
        "additional_config": {_BUDGET: config.execution.capture_memory_bytes},
        "compilation_config": CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=list(capacities(config.max_num_batched_tokens)),
            max_cudagraph_capture_size=config.max_num_batched_tokens,
        ),
    }


def graph_mode(budget: int) -> AttachmentGraphMode:
    """Describe only the intervention placement; ESamp's external tasks remain unchanged."""
    return AttachmentGraphMode.CUDA_GRAPH if budget > 0 else AttachmentGraphMode.EAGER


def _request_capacity() -> int:
    metadata = get_forward_context().attn_metadata
    assert isinstance(metadata, dict)
    attention = next(iter(metadata.values()))
    assert isinstance(attention, FlashAttentionMetadata)
    return attention.seq_lens.shape[0]


class CapacityGraphManager(ModelCudaGraphManager):
    """Use the upstream request-capacity field in both full and piecewise graph keys."""

    def _init_candidates(self) -> None:
        super()._init_candidates()
        for key, descriptors in self._candidates.items():
            self._candidates[key] = [
                replace(desc, num_reqs=size)
                for desc in descriptors
                for size in (
                    capacities(min(desc.num_tokens, self.max_num_reqs))
                    if desc.cg_mode == CUDAGraphMode.PIECEWISE
                    else (desc.num_reqs,)
                )
            ]
        for mode in self._capture_descs:
            self._capture_descs[mode] = sorted(
                {desc for group in self._candidates.values() for desc in group if desc.cg_mode == mode},
                key=lambda desc: (desc.num_tokens, desc.num_reqs or 0),
                reverse=True,
            )

    def run_pw_graph(self, model: nn.Module, model_inputs: dict[str, object]) -> object:
        context = get_forward_context()
        descriptor = context.batch_descriptor
        assert isinstance(descriptor, BatchDescriptor)
        with override_forward_context(
            replace(context, batch_descriptor=replace(descriptor, num_reqs=_request_capacity()))
        ):
            return super().run_pw_graph(model, model_inputs)


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

    def finish_overlay_graphs(self) -> None:
        """Capture only the bound model; dummy tables have no request ownership or live destinations."""
        if self.projection_model is None:
            return
        manager = self.cudagraph_manager
        assert manager is not None
        self.cudagraph_manager = CapacityGraphManager(
            self.vllm_config,
            self.device,
            manager.cudagraph_mode,
            manager.decode_query_len,
        )
        self.projection_model.do_not_compile = False
        self.capture_model()
        used = cuda_graph_pool_bytes(current_platform.get_global_graph_pool(), self.device.index or 0)
        if used > self.overlay_graph_bytes:
            raise ValueError(f"Overlay model graphs need {used} bytes; increase execution.capture_memory_bytes")


@torch.library.custom_op("pllm::pinned_head_projection", mutates_args={"output", "captured"})
def pinned_head_projection(
    hidden: torch.Tensor,
    pool: torch.Tensor,
    table: torch.Tensor,
    output: torch.Tensor,
    captured: torch.Tensor,
) -> None:
    """Emit a prepared shared-region read; no request owner runs inside the custom op."""
    _READERS[pool.data_ptr()].emit(hidden, pool, table, output, captured)


def _pinned_head_projection_meta(
    hidden: torch.Tensor,
    pool: torch.Tensor,
    table: torch.Tensor,
    output: torch.Tensor,
    captured: torch.Tensor,
) -> None:
    return None


pinned_head_projection.register_fake(_pinned_head_projection_meta)  # pyright: ignore[reportUnknownMemberType]


class PinnedGraphOverlay:
    """One pinned head projection, its stable input table, and captured training operands."""

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        binding: ParameterOverlayFieldBinding,
        arena: PageBundleArena,
        rows: int,
        requests: int,
        budget: int,
    ) -> None:
        from pllm.backends.vllm.parameter_overlay import locate_parameter

        attention = model.model.layers[binding.target.layer].self_attn.attn
        if not isinstance(attention.impl, FlashAttentionImpl) or attention.impl.vllm_flash_attn_version != 2:
            raise ValueError("Graph checkpoint needs the FlashAttention2 backend")
        self.module, self.offset, self.width, _ = locate_parameter(model, binding.target)
        self.forward_base = self.module.forward
        self.binding, self.rows, self.requests, self.budget = binding, rows, requests, budget
        self.pool = arena.field_pool(binding.resource, binding.fields[0])
        self.table = torch.zeros(requests + requests * rows + rows, device=self.pool.device, dtype=torch.long)
        self.table[:requests].fill_(-1)
        self.table[-rows:].fill_(-1)
        self.captured = torch.empty(
            (rows, self.module.weight.shape[1]), device=self.pool.device, dtype=self.module.weight.dtype
        )
        self.staging = PinnedStaging()
        self.stream: torch.cuda.Stream | None = None
        self.readiness: CudaStreamReadiness | None = None
        self.sources: PackedPhaseRowBatch | None = None
        self.programs: dict[tuple[int, int], RegionInstructionExecution] = {}
        self.apply_count = self.updated_apply_count = 0
        self._prepare_programs()
        _READERS[self.pool.data_ptr()] = self
        self.module.forward = self.forward

    def _prepare_programs(self) -> None:
        for count in capacities(self.rows):
            for size in capacities(min(count, self.requests)):
                with torch.inference_mode(False), FakeTensorMode():
                    hidden = torch.empty((count, self.width), device=self.pool.device, dtype=self.module.weight.dtype)
                    slots = torch.empty((size,), device=self.pool.device, dtype=torch.long)
                    rows = torch.empty_strided((size, count), (self.rows, 1), device=self.pool.device, dtype=torch.long)
                    owners = torch.empty((count,), device=self.pool.device, dtype=torch.long)
                self.programs[size, count] = prepare_head_read((hidden, self.pool, slots, rows, owners), self.budget)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.nn.Parameter | None]:
        """Preserve the projection's actual bias and fused slices."""
        base = self.forward_base(hidden)
        assert isinstance(base, tuple)
        output = base[0]
        pinned_head_projection(hidden, self.pool, self.table, output, self.captured)
        return base

    def emit(
        self,
        hidden: torch.Tensor,
        pool: torch.Tensor,
        table: torch.Tensor,
        output: torch.Tensor,
        captured: torch.Tensor,
    ) -> None:
        """Run only tensor instructions; this method is bypassed by CUDA graph replay."""
        size, count = _request_capacity(), hidden.shape[0]
        slots = table[:size]
        rows = table[self.requests : self.requests + self.requests * self.rows].view(self.requests, self.rows)[
            :size, :count
        ]
        owners = table[-self.rows :][:count]
        values = (hidden, pool, slots, rows, owners)
        key = size, count
        result = (
            self.programs[key]
            .run(dict(zip(("hidden", "pool", "slots", "rows", "owners"), values, strict=True)))
            .values[0]
        )
        assert isinstance(result, torch.Tensor)
        output[:, self.offset : self.offset + self.width].add_((result * self.binding.scale).to(output.dtype))
        captured[:count].copy_(hidden)

    def prepare(
        self,
        rows: PackedPhaseRowBatch,
        context: StateTaskContext[DataT],
        directory: RequestStateDirectory,
        transition: ProjectionTransition | None = None,
    ) -> None:
        """Upload pin-authorized indices; request identity never becomes a compiled constant."""
        assert transition is None
        self.stream = context.pin.stream
        assert self.stream is not None
        if self.readiness is not None:
            self.readiness.order_on(self.stream)
        for value in (self.pool, self.table, self.captured):
            value.record_stream(self.stream)
        host = self.staging.take(self.table.numel()).numpy()
        sources = host[: self.requests]
        positions = host[self.requests : self.requests + self.requests * self.rows].reshape(self.requests, self.rows)
        owners = host[-self.rows :]
        sources.fill(-1)
        owners.fill(-1)
        identities = tuple(dict.fromkeys(row.identity for row in rows.rows))
        count = self._token_capacity(rows.total_rows)
        for index, identity in enumerate(identities):
            selected = tuple(row for row in rows.rows if row.identity == identity)
            ref = context.selection.ref_for(identity)
            _, (slot,) = context.arena.field_slots((context.pin,), (ref,), self.binding.fields[0])
            if index == 0:
                # Padded request lanes may load only a field held by this invocation's pin.
                sources.fill(slot)
            sources[index] = slot
            packed = tuple(row.packed_row_index for row in selected)
            positions[index, : len(packed)] = packed
            # The capacity class is the upstream model's padded token count, not the valid row count.
            owners[list(packed)] = index * count + np.arange(len(packed))
            self.updated_apply_count += len(packed) if ref.version_id else 0
        self.staging.copy_to(self.table)
        self.apply_count += len(rows.rows)
        self.sources = rows

    def _token_capacity(self, rows: int) -> int:
        return next(size for size in capacities(self.rows) if size >= rows)

    def take_captures(self) -> tuple[CapturedParameterOperand, ...]:
        """Read graph-written operands for the ordinary post-forward transition."""
        assert self.sources is not None
        rows = self.sources.rows
        result: list[CapturedParameterOperand] = []
        for identity in dict.fromkeys(row.identity for row in rows):
            selected = tuple(row for row in rows if row.identity == identity)
            first, last = selected[0].packed_row_index, selected[-1].packed_row_index + 1
            assert len(selected) == last - first
            hidden = self.captured[first:last].clone()
            result.append(
                CapturedParameterOperand(
                    identity,
                    self.binding.target,
                    selected[0].phase,
                    hidden,
                    hidden,
                    self.module.weight[self.offset : self.offset + self.width].detach(),
                )
            )
        return tuple(result)

    def clear(self) -> None:
        """Order the next overwrite after all consumers without making the host wait."""
        if self.sources is not None:
            assert self.stream is not None
            self.readiness = CudaStreamReadiness(self.stream)
        self.sources = None

    def close(self) -> None:
        """Drop the binding after the normal session has drained its consumers."""
        self.module.forward = self.forward_base
        _READERS.pop(self.pool.data_ptr())
