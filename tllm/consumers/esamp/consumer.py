#!/usr/bin/env python3
"""BaseConsumer wrapper around the ESamp training engine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence, cast

import torch

from tllm.common.path_resolution import candidate_capture_paths
from tllm.common.state import resolve_object_by_path
from tllm.consumers.base import BaseConsumer
from tllm.consumers.esamp.config import ESampConsumerConfig
from tllm.consumers.esamp.engine import ESampTrainEngine, ESampStats
from tllm.consumers.esamp.initializers.svd import build_model_bank_initializer
from tllm.consumers.esamp.sampler_provider import ESampSamplerModifierProvider
from tllm.consumers._common.bundle_inputs import residual_pair_input
from tllm.contracts.port_bundle import PortBundle
from tllm.contracts.runtime_context import RuntimeContext
from tllm.ports.address import port_address_desc
from tllm.ports.base import (
    BufferOwnership,
    ConsumerFlow,
    FlowFusion,
    TimingContract,
    TimingDeadline,
    TimingMobility,
    RowSelection,
    VersionObservation,
)
from tllm.ports.request_meta import RequestMeta
from tllm.ports.residual_stream import ResidualStream
from tllm.workgraph.frontend import (
    Capability,
    CapturedFlow,
    Device,
    Effect,
    ExternContract,
    PortBindingDesc,
    PortBindingDirection,
    StateScope,
    extern,
    flow,
)
from tllm.workgraph.hierarchy import CaptureAttachment
from tllm.workgraph.ir import ExecutorKey, NodeKey
from tllm.workgraph.physical import (
    FenceKey,
    PhysicalCaptureRegion,
    PhysicalRegionExecutorDesc,
    StreamClass,
)
from tllm.runtime.scheduling.plan_store import (
    PlanResources,
    PlanSnapshotStore,
    RuntimeStream,
)
from tllm.runtime.scheduling.physical import Fence
from tllm.runtime.scheduling.workgraph_executor import (
    PhysicalExecutionContext,
    PhysicalExecutionStats,
    PhysicalExecutor,
    WorkGraphRuntimeStats,
    runtime_stats_from_executor,
)
from tllm.runtime.specialization import RuntimeSpecializationSpec


_ESAMP_SOURCE_READ = ResidualStream.read(
    layer=0,
    site="block_output",
    phase="decode",
    role="source",
)
_ESAMP_TARGET_READ = ResidualStream.read(
    layer=-1,
    site="block_output",
    phase="decode",
    role="target",
)
_ESAMP_RESIDUAL_BINDINGS = (
    PortBindingDesc(
        port=port_address_desc(_ESAMP_SOURCE_READ),
        direction=PortBindingDirection.INPUT,
        value_path=(0,),
    ),
    PortBindingDesc(
        port=port_address_desc(_ESAMP_TARGET_READ),
        direction=PortBindingDirection.INPUT,
        value_path=(1,),
    ),
)
_ESAMP_EXECUTOR_BINDINGS = (
    PortBindingDesc(
        port=port_address_desc(_ESAMP_SOURCE_READ),
        direction=PortBindingDirection.INPUT,
        value_path=(1,),
    ),
    PortBindingDesc(
        port=port_address_desc(_ESAMP_TARGET_READ),
        direction=PortBindingDirection.INPUT,
        value_path=(2,),
    ),
)


@flow(bindings=_ESAMP_RESIDUAL_BINDINGS)
def _esamp_residual_side_work(
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Describe the capture-safe residual handoff inside legacy ESamp work."""

    return source, target


@extern(
    effect=Effect.WRITE_STATE,
    state=StateScope.REPLICA,
    device=Device.CUDA,
    capability=Capability.EAGER_ONLY,
    bindings=_ESAMP_EXECUTOR_BINDINGS,
)
def _stage_esamp_residuals(
    engine: ESampTrainEngine,
    source: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """Typed external implementation for the migrated residual side work."""

    engine.launch_step(source, target)


_ESAMP_STAGE_EXECUTOR = ExecutorKey(
    "extern",
    "tllm.consumers.esamp.consumer._stage_esamp_residuals",
)


@dataclass(slots=True)
class _TrainingRows:
    source_hidden: torch.Tensor
    target_hidden: torch.Tensor
    prompt_idxs: list[int]
    request_meta_is_first_per_prompt: bool


class _ESampRegionStream:
    """Lazily resolve the CUDA stream used by the ESamp staging launch."""

    __slots__ = ("_engine", "_device", "_resolved_stream")

    def __init__(self, engine: ESampTrainEngine) -> None:
        self._engine = engine
        self._device: torch.device | None = None
        self._resolved_stream: torch.cuda.Stream | None = None

    def bind_device(self, device: torch.device) -> None:
        self._device = device
        self._resolved_stream = self._resolve_engine_stream()

    def resolve(self) -> torch.cuda.Stream | None:
        return self._resolved_stream

    def _resolve_engine_stream(self) -> torch.cuda.Stream | None:
        device = self._device
        if device is None or device.type != "cuda":
            return None
        streams = self._engine.state.streams
        if streams.serial:
            return torch.cuda.current_stream(device=device)
        stream = streams.forward_stream
        if stream is None:
            raise RuntimeError("ESamp WorkGraph stream is unavailable before launch")
        return stream

    def wait_event(self, event: object) -> None:
        stream = self.resolve()
        if stream is None:
            return
        stream.wait_event(cast(torch.cuda.Event, event))


class _ESampRegionRetireEvent:
    """CUDA retirement event recorded on the real ESamp staging stream."""

    __slots__ = ("_event", "_completed_without_cuda")

    def __init__(self) -> None:
        self._event: torch.cuda.Event | None = None
        self._completed_without_cuda = False

    def record(self, stream: object) -> None:
        resolved = cast(_ESampRegionStream, stream).resolve()
        if resolved is None:
            self._completed_without_cuda = True
            return
        event = self._event
        if event is None:
            event = torch.cuda.Event(blocking=False)
            self._event = event
        event.record(resolved)

    def query(self) -> bool:
        event = self._event
        return self._completed_without_cuda if event is None else bool(event.query())


class _ESampFusionGroupHandle:
    def __init__(
        self,
        consumer: "ESampConsumer",
        flow_ids: tuple[str, ...],
    ) -> None:
        self._consumer = consumer
        self._flow_ids = flow_ids

    def execute(
        self,
        bundles: Mapping[str, PortBundle],
        ctx: RuntimeContext,
    ) -> None:
        if tuple(bundles) != self._flow_ids:
            raise RuntimeError(
                f"ESamp fusion handle expected flows {self._flow_ids!r}, "
                f"got {tuple(bundles)!r}"
            )
        self._consumer._execute_fusion_group(
            bundles[self._flow_ids[0]],
            ctx,
        )

    def replay_count(self) -> int:
        return self._consumer.fusion_replay_count()


class ESampConsumer(BaseConsumer[ESampConsumerConfig]):
    """Dispatch-plan friendly ESamp wrapper around the formal engine."""

    @staticmethod
    def _build_engine(config: ESampConsumerConfig) -> ESampTrainEngine:
        return ESampTrainEngine(
            hidden_dim=int(config.distiller_hidden_dim),
            lr=float(config.distiller_lr),
            enabled=bool(config.enable_esamp_training),
            per_request_models=bool(config.per_request_models),
            per_request_model_bank=bool(config.per_request_model_bank),
            model_bank_slots=int(config.model_bank_slots),
            model_bank_flush_interval=int(config.model_bank_flush_interval),
            model_bank_rank=int(config.model_bank_rank),
            model_bank_use_output_layernorm=bool(config.model_bank_use_output_layernorm),
            model_bank_initializer=build_model_bank_initializer(config.model_bank_initializer),
            model_bank_train_cudagraph=bool(config.model_bank_train_cudagraph),
            model_bank_forward_backend=str(config.model_bank_forward_backend),
            adaptation_pipeline_slots=int(config.adaptation_pipeline_slots),
            adaptation_stream_mode=str(config.adaptation_stream_mode),
            adaptation_stream_priority=int(config.adaptation_stream_priority),
            trace_per_request_losses=bool(config.trace_per_request_losses),
            trace_interval=int(config.trace_interval),
            trace_max_points=int(config.trace_max_points),
        )

    def __init__(
        self,
        config: ESampConsumerConfig,
        *,
        engine: Optional[ESampTrainEngine] = None,
    ) -> None:
        self.config = config
        self._engine = engine or self._build_engine(config)
        self._target_resolved_path = str(config.target_layer_path)
        self._last_active_rows = 0
        self._last_prompt_idxs: tuple[int, ...] = ()
        self._runtime_layers_cache_key: tuple[int, str, str] | None = None
        self._runtime_source_layer: Optional[torch.nn.Module] = None
        self._runtime_target_layer: Optional[torch.nn.Module] = None
        self._runtime_initializer_key: tuple[int, str, int, str, torch.dtype] | None = None
        self._runtime_resource_key: tuple[object, ...] | None = None
        self._sampler_provider = ESampSamplerModifierProvider(config=self.config, engine=self._engine)
        self._fusion_handle: _ESampFusionGroupHandle | None = None
        self._workgraph_capture: CapturedFlow | None = None
        self._workgraph_attachment: CaptureAttachment | None = None
        self._workgraph_plan_store: PlanSnapshotStore | None = None
        self._workgraph_executor: PhysicalExecutor | None = None
        self._workgraph_prepared_step_id = -1
        self._workgraph_deactivate_pending = False

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    def flows(self) -> Sequence[ConsumerFlow]:
        if not bool(self.config.enable_esamp_training):
            return ()
        compact_first_per_prompt = self._compact_first_per_prompt()
        static_rows = self._static_training_rows()
        fusion = (
            FlowFusion.SIDE_STREAM_CUDAGRAPH
            if bool(self.config.model_bank_train_cudagraph) and static_rows > 0
            else FlowFusion.NONE
        )
        return [
            ConsumerFlow(
                reads=(
                    _ESAMP_SOURCE_READ,
                    _ESAMP_TARGET_READ,
                    RequestMeta.read(),
                ),
                writes=(),
                window="out_of_band",
                delivery="device_lease",
                ownership=BufferOwnership.RUNTIME_LEASE,
                lease_lifetime="step",
                row_compaction=(
                    RowSelection.FIRST_PER_PROMPT
                    if compact_first_per_prompt
                    else RowSelection.ALL
                ),
                max_bundle_rows=static_rows,
                fusion=fusion,
                bundle_key=("engine_step_id", "phase"),
                timing=TimingContract(
                    mobility=TimingMobility.DEFERRABLE,
                    deadline=TimingDeadline.NEXT_PREPARE,
                    max_defer_steps=1,
                    observes_version=VersionObservation.AT_DECLARED_WINDOW,
                ),
            )
        ]

    def _compact_first_per_prompt(self) -> bool:
        return bool(self.config.per_request_model_bank) or (
            self.config.training_row_selection == RowSelection.FIRST_PER_PROMPT
        )

    def _static_training_rows(self) -> int:
        if self._compact_first_per_prompt():
            return max(
                0,
                int(self.config.model_bank_slots or self.config.graph_scratch_rows),
            )
        return max(0, int(self.config.graph_scratch_rows))

    def update_config(self, config: ESampConsumerConfig) -> None:
        preserve_workgraph_capture = (
            self._workgraph_capture is not None
            and self._workgraph_attachment is not None
            and str(config.consumer_id) == str(self.config.consumer_id)
            and bool(config.enable_workgraph_execution)
        )
        self.config = config
        self._sampler_provider.config = config
        self._engine.configure(config)
        self._runtime_layers_cache_key = None
        self._runtime_source_layer = None
        self._runtime_target_layer = None
        self._runtime_initializer_key = None
        self._runtime_resource_key = None
        if not preserve_workgraph_capture:
            self._workgraph_capture = None
            self._workgraph_attachment = None
            self.activate_workgraph_regions(())

    def specialize_workgraph(
        self,
        source_example: torch.Tensor,
        target_example: torch.Tensor,
    ) -> tuple[CapturedFlow, CaptureAttachment]:
        """Capture residual handoff metadata on an explicit control-plane path."""

        if not bool(self.config.enable_esamp_training):
            raise RuntimeError("disabled ESamp cannot publish a WorkGraph specialization")
        if source_example.ndim < 2 or source_example.ndim != target_example.ndim:
            raise ValueError("ESamp residual examples must have matching rank of at least two")
        if source_example.layout != target_example.layout:
            raise ValueError("ESamp residual examples must have matching tensor layouts")
        if source_example.device != target_example.device:
            raise ValueError("ESamp residual examples must reside on the same device")

        captured = _esamp_residual_side_work.capture(source_example, target_example)
        attachment = CaptureAttachment(
            node_key=NodeKey.consumer_flow(self.consumer_id, 0),
            artifact_key=captured.artifact_key,
            bindings=_ESAMP_RESIDUAL_BINDINGS,
        )
        self._workgraph_capture = captured
        self._workgraph_attachment = attachment
        return captured, attachment

    def cold_specialize_workgraph(self, spec: RuntimeSpecializationSpec) -> None:
        if not bool(self.config.enable_esamp_training) or not bool(
            self.config.enable_workgraph_execution
        ):
            return
        source_example = torch.empty(
            (spec.rows, spec.hidden_size),
            dtype=spec.dtype,
            device=spec.device,
        )
        target_example = torch.empty_like(source_example)
        self.specialize_workgraph(source_example, target_example)

    def workgraph_captures(self) -> tuple[CapturedFlow, ...]:
        if not bool(self.config.enable_esamp_training) or self._workgraph_capture is None:
            return ()
        return (self._workgraph_capture,)

    def workgraph_attachments(self) -> tuple[CaptureAttachment, ...]:
        if not bool(self.config.enable_esamp_training) or self._workgraph_attachment is None:
            return ()
        return (self._workgraph_attachment,)

    def workgraph_region_executors(self) -> tuple[PhysicalRegionExecutorDesc, ...]:
        capture = self._workgraph_capture
        if (
            not bool(self.config.enable_esamp_training)
            or not bool(self.config.enable_workgraph_execution)
            or capture is None
        ):
            return ()
        return (
            PhysicalRegionExecutorDesc(
                artifact_key=capture.artifact_key,
                executor=_ESAMP_STAGE_EXECUTOR,
                contract=_stage_esamp_residuals.extern_contract,
            ),
        )

    def activate_workgraph_regions(
        self,
        regions: tuple[PhysicalCaptureRegion, ...],
    ) -> None:
        capture = self._workgraph_capture
        attachment = self._workgraph_attachment
        if not regions:
            self._workgraph_deactivate_pending = True
            self._finalize_workgraph_deactivation()
            return
        if capture is None or attachment is None:
            raise RuntimeError("ESamp physical region requires a specialized capture")
        matching = tuple(
            region
            for region in regions
            if region.attachment.node_key == attachment.node_key
            and region.attachment.artifact_key == attachment.artifact_key
        )
        if len(matching) != 1:
            raise ValueError("ESamp requires exactly one explicitly attached region")
        region = matching[0]
        plan = region.plan
        if len(plan.nodes) != 1:
            raise ValueError("ESamp physical handoff requires exactly one capture task")
        node = plan.nodes[0]
        if node.artifact_key != capture.artifact_key:
            raise ValueError("ESamp physical region artifact differs from its capture")
        if attachment.artifact_key != capture.artifact_key:
            raise ValueError("ESamp capture attachment artifact is inconsistent")
        if region.executor_contract != _stage_esamp_residuals.extern_contract:
            raise ValueError("ESamp physical region requires its typed staging extern")
        if node.executor != _ESAMP_STAGE_EXECUTOR:
            raise ValueError("ESamp physical region selected an unexpected executor")
        region_stream = _ESampRegionStream(self._engine)
        resources = PlanResources(
            plan=plan,
            streams=(
                RuntimeStream(
                    StreamClass.SIDE_FORWARD,
                    region_stream,
                    retire_fence=Fence(
                        FenceKey("esamp-region", f"retire-epoch-{plan.plan_epoch}"),
                        _ESampRegionRetireEvent(),
                    ),
                ),
            ),
        )
        self._workgraph_deactivate_pending = False
        store = self._workgraph_plan_store
        if store is None:
            store = PlanSnapshotStore(resources)
            self._workgraph_plan_store = store
            self._workgraph_executor = PhysicalExecutor(
                snapshots=store,
                handlers={node.executor: self._execute_workgraph_staging},
            )
            return
        if plan.plan_epoch > store.current.plan.plan_epoch:
            store.stage(resources)

    def _execute_workgraph_staging(self, context: PhysicalExecutionContext) -> None:
        if len(context.tensors) != 2:
            raise RuntimeError("ESamp training region requires source and target tensors")
        stream = cast(_ESampRegionStream, context.stream)
        stream.bind_device(context.tensors[0].device)
        _stage_esamp_residuals(
            self._engine,
            context.tensors[0],
            context.tensors[1],
        )

    def prepare_workgraph_step(self, step_id: int) -> None:
        if self._workgraph_deactivate_pending:
            self._finalize_workgraph_deactivation()
            return
        executor = self._workgraph_executor
        if executor is None:
            return
        previous = self._workgraph_prepared_step_id
        if previous >= 0 and previous != int(step_id):
            executor.finish_step(previous)
        executor.prepare_step(int(step_id))
        self._workgraph_prepared_step_id = int(step_id)

    def finish_workgraph_step(self, step_id: int) -> None:
        executor = self._workgraph_executor
        if executor is None or self._workgraph_prepared_step_id != int(step_id):
            return
        executor.finish_step(int(step_id))
        self._workgraph_prepared_step_id = -1
        self._finalize_workgraph_deactivation()

    def drain_workgraph_specializations(self) -> int:
        executor = self._workgraph_executor
        if executor is None or self._workgraph_deactivate_pending:
            return 0
        return executor.drain_specialization_requests()

    def _finalize_workgraph_deactivation(self) -> None:
        if not self._workgraph_deactivate_pending:
            return
        store = self._workgraph_plan_store
        if self._workgraph_prepared_step_id >= 0:
            return
        if store is not None:
            store.collect_retired()
            if not store.closeable:
                return
        self._workgraph_plan_store = None
        self._workgraph_executor = None
        self._workgraph_deactivate_pending = False

    def physical_execution_stats(self) -> PhysicalExecutionStats:
        executor = self._workgraph_executor
        return (
            PhysicalExecutionStats()
            if executor is None
            else replace(executor.stats)
        )

    def workgraph_execution_stats(self) -> PhysicalExecutionStats | None:
        executor = self._workgraph_executor
        return None if executor is None else replace(executor.stats)

    def read_workgraph_runtime_stats(self) -> WorkGraphRuntimeStats:
        executor = self._workgraph_executor
        if executor is None:
            return WorkGraphRuntimeStats()
        store = self._workgraph_plan_store
        leases = 0 if store is None else len(store.current.leases)
        return runtime_stats_from_executor(executor.stats, active_leases=leases)

    def sampler_modifier_provider(self) -> ESampSamplerModifierProvider:
        return self._sampler_provider

    @staticmethod
    def _infer_hidden_dtype(layer: torch.nn.Module) -> torch.dtype:
        for p in layer.parameters():
            return p.dtype
        for b in layer.buffers():
            return b.dtype
        return torch.float32

    def _resolve_layer_with_fallback(self, model: torch.nn.Module, path: str) -> tuple[torch.nn.Module, str]:
        errors: list[str] = []
        for candidate in candidate_capture_paths(path):
            try:
                obj = resolve_object_by_path(model, candidate)
            except RuntimeError as exc:
                errors.append(f"{candidate}: {exc}")
                continue
            if not isinstance(obj, torch.nn.Module):
                errors.append(f"{candidate}: resolved to non-module {type(obj).__name__}")
                continue
            return obj, candidate
        details = "; ".join(errors) if errors else "no candidate"
        raise RuntimeError(f"Cannot resolve ESamp layer path `{path}`. Tried: {details}")

    def _maybe_prepare_initializer(self, *, model: torch.nn.Module, target_layer: torch.nn.Module, target_resolved: str) -> None:
        hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
        if hidden_size <= 0:
            return
        self._engine.prepare_model_bank_initializer(
            model=model,
            target_layer=target_layer,
            target_resolved=target_resolved,
            hidden_size=hidden_size,
        )

    def _resource_config_key(self) -> tuple[object, ...]:
        return (
            int(self.config.distiller_hidden_dim),
            float(self.config.distiller_lr),
            bool(self.config.per_request_models),
            bool(self.config.per_request_model_bank),
            int(self.config.model_bank_slots),
            int(self.config.model_bank_flush_interval),
            int(self.config.model_bank_rank),
            bool(self.config.model_bank_use_output_layernorm),
            self.config.model_bank_initializer,
            bool(self.config.model_bank_train_cudagraph),
            str(self.config.model_bank_forward_backend),
            int(self.config.adaptation_pipeline_slots),
            str(self.config.adaptation_stream_mode),
            int(self.config.adaptation_stream_priority),
        )

    def _resolve_runtime_layers(
        self, model: torch.nn.Module
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        source_layer, source_resolved = self._resolve_layer_with_fallback(model, self.config.source_layer_path)
        target_layer, target_resolved = self._resolve_layer_with_fallback(model, self.config.target_layer_path)
        self._target_resolved_path = target_resolved
        return source_layer, target_layer

    def _runtime_layers_for_model(self, model: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module]:
        cache_key = (id(model), str(self.config.source_layer_path), str(self.config.target_layer_path))
        if (
            self._runtime_layers_cache_key == cache_key
            and self._runtime_source_layer is not None
            and self._runtime_target_layer is not None
        ):
            return self._runtime_source_layer, self._runtime_target_layer
        source_layer, target_layer = self._resolve_runtime_layers(model)
        self._runtime_layers_cache_key = cache_key
        self._runtime_source_layer = source_layer
        self._runtime_target_layer = target_layer
        self._runtime_initializer_key = None
        return source_layer, target_layer

    def _ensure_runtime_resources(
        self,
        ctx: RuntimeContext,
        *,
        rows_hidden: Optional[torch.Tensor] = None,
    ) -> None:
        model = ctx.model
        source_layer = None
        target_layer = None
        if model is not None:
            source_layer, target_layer = self._runtime_layers_for_model(model)

        hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
        hidden_dtype = self._infer_hidden_dtype(source_layer) if source_layer is not None else None
        if rows_hidden is not None and rows_hidden.ndim >= 2:
            hidden_size = max(hidden_size, int(rows_hidden.shape[1]))
            hidden_dtype = hidden_dtype or rows_hidden.dtype
        if hidden_size <= 0 or hidden_dtype is None:
            raise RuntimeError("ESamp consumer cannot infer hidden_size/hidden_dtype for runtime resources")
        rows = self._static_training_rows()
        if rows <= 0:
            runner_rows = int(getattr(ctx.runner, "max_num_reqs", 0) or 0)
            rows = runner_rows
        if rows_hidden is not None:
            rows = max(rows, int(rows_hidden.shape[0]))
        if rows <= 0:
            raise RuntimeError("ESamp consumer requires positive rows via graph_scratch_rows, runner.max_num_reqs, or rows_hidden")
        device = rows_hidden.device if rows_hidden is not None else ctx.device
        resource_key = (
            id(model),
            str(device),
            hidden_dtype,
            int(hidden_size),
            int(rows),
            self._resource_config_key(),
        )
        if self._runtime_resource_key == resource_key:
            return
        self._engine.ensure_resources(
            device=device,
            rows=rows,
            hidden_size=hidden_size,
            hidden_dtype=hidden_dtype,
        )
        initializer_key = (
            id(model),
            str(self._target_resolved_path),
            int(hidden_size),
            str(device),
            hidden_dtype,
        ) if model is not None else None
        if model is not None and target_layer is not None and self._runtime_initializer_key != initializer_key:
            self._maybe_prepare_initializer(model=model, target_layer=target_layer, target_resolved=self._target_resolved_path)
            self._runtime_initializer_key = initializer_key
        self._runtime_resource_key = resource_key

    @staticmethod
    def _first_row_per_prompt(prompt_idxs: Sequence[int]) -> list[int]:
        seen: set[int] = set()
        row_ids: list[int] = []
        for row_i, prompt_idx in enumerate(prompt_idxs):
            prompt_key = int(prompt_idx)
            if prompt_key < 0 or prompt_key in seen:
                continue
            seen.add(prompt_key)
            row_ids.append(int(row_i))
        return row_ids

    def _extract_training_rows(
        self,
        bundle: PortBundle,
    ) -> _TrainingRows | None:
        inputs = residual_pair_input(bundle)
        if inputs is None:
            return None
        source_hidden = inputs.source
        target_hidden = inputs.target
        if int(source_hidden.shape[0]) != int(target_hidden.shape[0]):
            raise RuntimeError("ESamp consumer requires source/target hidden batches to have the same row count")

        request_meta = inputs.request_meta
        if request_meta is None:
            raw_meta = bundle.entries.get("request_meta")
            raise RuntimeError(
                "ESamp consumer requires request_meta to be RowBatchMeta; "
                f"got {type(raw_meta).__module__}.{type(raw_meta).__qualname__} "
                f"with bundle entries {tuple(bundle.entries)} and "
                f"uses_device_lease={inputs.uses_device_lease}"
            )
        if inputs.uses_device_lease and len(request_meta.request_ids) != int(source_hidden.shape[0]):
            raise RuntimeError("ESamp consumer requires row metadata to match active lease rows")
        prompt_idxs = [int(prompt_idx) for prompt_idx in request_meta.prompt_idxs]
        request_meta_is_first_per_prompt = str(request_meta.row_compaction) == "first_per_prompt"
        while len(prompt_idxs) < int(source_hidden.shape[0]):
            prompt_idxs.append(-1)

        return _TrainingRows(
            source_hidden=source_hidden,
            target_hidden=target_hidden,
            prompt_idxs=prompt_idxs,
            request_meta_is_first_per_prompt=request_meta_is_first_per_prompt,
        )

    def _stage_training_bundle(self, bundle: PortBundle, ctx: RuntimeContext) -> None:
        if not bool(self.config.enable_esamp_training):
            return
        rows = self._extract_training_rows(bundle)
        if rows is None:
            return
        source_hidden = rows.source_hidden
        target_hidden = rows.target_hidden
        prompt_idxs = rows.prompt_idxs

        if bool(self.config.per_request_model_bank) and not rows.request_meta_is_first_per_prompt:
            row_ids = self._first_row_per_prompt(prompt_idxs[: int(source_hidden.shape[0])])
            if row_ids:
                row_tensor = torch.as_tensor(row_ids, device=source_hidden.device, dtype=torch.long)
                source_hidden = source_hidden.index_select(0, row_tensor)
                target_hidden = target_hidden.index_select(0, row_tensor)
                prompt_idxs = [prompt_idxs[row_i] for row_i in row_ids]
            else:
                source_hidden = source_hidden[:0]
                target_hidden = target_hidden[:0]
                prompt_idxs = []

        self._ensure_runtime_resources(ctx, rows_hidden=source_hidden)
        executor = self._workgraph_executor
        step_id = int(bundle.key.engine_step_id)
        if executor is not None and self._workgraph_prepared_step_id == step_id:
            executor.execute_step(
                step_id,
                tensors=(source_hidden, target_hidden),
            )
        else:
            _esamp_residual_side_work(source_hidden, target_hidden)
            self._engine.launch_step(source_hidden, target_hidden)
        self._last_active_rows = int(source_hidden.shape[0])
        self._last_prompt_idxs = tuple(prompt_idxs[: int(source_hidden.shape[0])])

    def _launch_staged_training(self, ctx: RuntimeContext) -> None:
        self._engine.launch_delayed_backward(
            int(self._last_active_rows),
            prompt_idxs=self._last_prompt_idxs,
            wait_events=ctx.predecessor_events,
        )
        self._last_active_rows = 0
        self._last_prompt_idxs = ()

    def _execute_fusion_group(
        self,
        bundle: PortBundle,
        ctx: RuntimeContext,
    ) -> None:
        """Stage and launch the compiler-owned ESamp training node."""
        self._stage_training_bundle(bundle, ctx)
        self._launch_staged_training(ctx)
        self.finish_workgraph_step(int(bundle.key.engine_step_id))

    def consume_bundle(self, bundle: PortBundle, ctx: RuntimeContext) -> None:
        self._stage_training_bundle(bundle, ctx)

    def on_step_end(self, ctx: RuntimeContext) -> None:
        self._launch_staged_training(ctx)
        step_id = self._workgraph_prepared_step_id
        if step_id >= 0:
            self.finish_workgraph_step(step_id)

    def retire_device_lease(self, ctx: RuntimeContext) -> None:
        del ctx
        self._engine.retire_step_inputs()

    def fusion_replay_count(self) -> int:
        return self._engine.fusion_replay_count()

    def fusion_group_handle(
        self,
        flow_ids: tuple[str, ...],
    ) -> _ESampFusionGroupHandle | None:
        expected = (f"{self.consumer_id}#0",)
        if flow_ids != expected:
            return None
        handle = self._fusion_handle
        if handle is None or handle._flow_ids != expected:
            handle = _ESampFusionGroupHandle(self, expected)
            self._fusion_handle = handle
        return handle

    def set_enabled(self, enabled: bool) -> None:
        self.config.enable_esamp_training = bool(enabled)
        self._engine.set_enabled(bool(enabled))

    def synchronize(self) -> None:
        self._engine.synchronize()

    def read_and_reset_stats(self, sync: bool = True) -> ESampStats:
        return self._engine.read_and_reset_stats(sync=sync)

    def read_and_reset_per_request_stats(self, sync: bool = True):
        return self._engine.read_and_reset_per_request_stats(sync=sync)
