"""Causal TLM objective and multifactor AdamW over the shared signed state owner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import nullcontext
from enum import StrEnum
from hashlib import sha256
import json
from math import isfinite
from typing import Protocol

import torch

from tllm.canonical.backend_program import ParameterTargetShape
from tllm.canonical.kernels.adamw import AdamWConfig, AdamWFields, apply_adamw
from tllm.canonical.kernels.tlm_objective import tlm_weighted_selection_objective
from tllm.canonical.ptts.page_bundle import PageBundleArena, PagePublication, tensor_dtype
from tllm.canonical.ptts.page_bundle_planning import PageBundleProviderFacts, compile_page_bundle_plan
from tllm.canonical.ptts.request_state import RequestStateDirectory
from tllm.canonical.state_execution import PublishStateTask, StateExecution, StateTaskContext
from tllm.contracts.request_state import IndexedRequestIdentityBatch
from tllm.contracts.parameter_overlay import ParameterModuleKind, ParameterRole
from tllm.contracts.semantic_graph import (
    AnchorId, AnchorSymbol, ExecutionKind, HostPayloadSchemaId, HostServiceId,
    HostServiceInputSource, HostServicePort, ImplementationId, SchedulerTimeline,
    SemanticEffect, SemanticGraph, SemanticInput, SemanticOutput, SemanticOutputKind,
    SemanticState, SemanticTask, StateEffectKind, StateResource, StateScope,
    TaskImplementationSpec, TaskResultInputSource, TaskSymbol, TimingConstraint, WorkloadId,
)
from tllm.contracts.tensor_layout import (
    LayoutFieldId, MutationGranularity, StateTensorLayout, TensorDType,
    TensorFieldSpec, TensorLayoutRequest, TensorLayoutSpec,
)


class TLMMode(StrEnum):
    ONLINE = "online"
    OFFLINE = "offline"


@dataclass(frozen=True, slots=True)
class TLMConfig:
    mode: TLMMode = TLMMode.ONLINE
    optimizer: AdamWConfig = AdamWConfig(5e-5, (0.9, 0.999), 1e-8, 0.01)
    threshold: float = 3.0
    lamb: float = 0.1
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if (type(self.mode) is not TLMMode or not isfinite(self.threshold)
            or not isfinite(self.lamb) or self.lamb < 0):
            raise ValueError("TLM objective configuration is not finite or closed")


@dataclass(frozen=True, slots=True)
class TLMTrainBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor | None = None


class TLMModel(Protocol):
    """Frozen causal mathematics, with caller-owned q/v factors in target order."""
    @property
    def targets(self) -> tuple[ParameterTargetShape, ...]: ...
    @property
    def initial_factors(self) -> tuple[torch.Tensor, ...]: ...
    @property
    def tensor_dtype(self) -> TensorDType: ...
    @property
    def device(self) -> torch.device: ...
    @property
    def fingerprint(self) -> str: ...
    def logits(self, batch: TLMTrainBatch, factors: tuple[torch.Tensor, ...] | None) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class TLMResult:
    loss: torch.Tensor
    selected_count: int
    publication: PagePublication | None


@dataclass(slots=True)
class TLMData:
    batch: TLMTrainBatch
    loss: torch.Tensor | None = None
    selected_count: int = 0


_STATE = StateResource("qv_factors")
_UPDATE = TaskSymbol("update")
_PUBLISH = TaskSymbol("publish")
_FIELDS = AdamWFields(*(LayoutFieldId(index) for index in range(5)))


@dataclass(frozen=True, slots=True)
class TLMWorkload:
    config: TLMConfig
    model: TLMModel

    def initialize(self, roots: tuple[tuple[torch.Tensor, ...], ...]) -> None:
        (root,) = roots
        for index, factor in enumerate(self.model.initial_factors):
            root[5 * index].copy_(factor)

    def rows(self, value: TLMTrainBatch) -> IndexedRequestIdentityBatch:
        return IndexedRequestIdentityBatch(())

    def prepare(self, value: TLMTrainBatch, context: StateTaskContext[TLMData]) -> TLMData:
        if (value.input_ids.ndim != 2 or value.input_ids.shape != value.labels.shape
            or value.input_ids.shape[0] < 1 or value.input_ids.shape[1] < 2
            or value.input_ids.dtype != torch.int64 or value.labels.dtype != torch.int64
            or value.input_ids.device != self.model.device or value.labels.device != self.model.device
            or value.attention_mask is not None and (value.attention_mask.shape != value.input_ids.shape
                or value.attention_mask.device != self.model.device)):
            raise ValueError("TLM batch must contain aligned causal token/label tensors")
        return TLMData(value)

    def output(self, context: StateTaskContext[TLMData]) -> TLMResult:
        data = context.data()
        if data.loss is None or context.write is not None:
            raise RuntimeError("TLM signed objective/publication has not completed")
        return TLMResult(data.loss, data.selected_count, context.publication)


    def compile(self, timeline: SchedulerTimeline, providers: tuple[PageBundleProviderFacts, ...],
        *, state_memory_budget_bytes: int, plan_epoch: int = 1) -> StateExecution[TLMTrainBatch, TLMData, TLMResult]:
        config, model = self.config, self.model
        if not model.targets or len(model.initial_factors) != 2 * len(model.targets):
            raise ValueError("TLM requires one A/B factor pair per q/v target")
        targets = tuple(item.target for item in model.targets)
        if targets != tuple(sorted(set(targets))) or any(target.module is not ParameterModuleKind.ATTENTION
            or target.role not in (ParameterRole.QUERY_PROJECTION_WEIGHT, ParameterRole.VALUE_PROJECTION_WEIGHT)
            for target in targets):
            raise ValueError("TLM factor topology must name sorted unique q/v targets")
        for index, target in enumerate(model.targets):
            a, b = model.initial_factors[2 * index:2 * index + 2]
            if (a.ndim != 2 or b.ndim != 2 or a.shape[0] < 1 or a.shape[0] != b.shape[1]
                or a.shape[1] != target.input_size or b.shape[0] != target.output_size
                or a.device != model.device or b.device != model.device
                or a.dtype != tensor_dtype(model.tensor_dtype) or b.dtype != a.dtype):
                raise ValueError("TLM factors differ from the original q/v shape facts")
        start, end = (("request.prepare", "model.execute") if config.mode is TLMMode.OFFLINE else
            ("token.commit", "next.prepare"))
        capabilities = (f"compute/{model.device.type}@1", "optimizer/adamw@1", "transition/full_root@1")
        update_id = ImplementationId(f"canonical.tlm:{model.device.type}_adamw@1")
        publish_id = ImplementationId("canonical.state:publish@1")
        digest = sha256(json.dumps((asdict(config), model.fingerprint), sort_keys=True).encode()).hexdigest()
        graph = SemanticGraph(WorkloadId("tlm.qv_lora"), (
            SemanticTask(_PUBLISH, TimingConstraint(AnchorSymbol(start), AnchorSymbol(end)),
                inputs=(SemanticInput("updated", TaskResultInputSource(_UPDATE, "updated")),),
                outputs=(SemanticOutput("publication", SemanticOutputKind.HOST, HostPayloadSchemaId("canonical.tlm.result@1")),),
                implementations=(TaskImplementationSpec(publish_id, ExecutionKind.NATIVE_TASK, ("transition/full_root@1",)),),
                effects=(SemanticEffect(_STATE, StateEffectKind.PUBLISH),)),
            SemanticTask(_UPDATE, TimingConstraint(AnchorSymbol(start), AnchorSymbol(end)),
                inputs=(SemanticInput("batch", HostServiceInputSource(HostServiceId("training"),
                    HostServicePort("batch"), HostPayloadSchemaId("canonical.causal_tokens@1"))),),
                outputs=(SemanticOutput("updated", SemanticOutputKind.TEMPORARY),),
                implementations=(TaskImplementationSpec(update_id, ExecutionKind.NATIVE_TASK, capabilities),),
                effects=(SemanticEffect(_STATE, StateEffectKind.TRANSITION_MAYBE),)),
        ), digest, states=(SemanticState(_STATE, StateScope.INSTANCE, ()),))
        fields = tuple(TensorFieldSpec(LayoutFieldId(5 * index + offset), () if offset == 2 else tuple(factor.shape),
            TensorDType.FLOAT32 if offset == 2 else model.tensor_dtype, 16, MutationGranularity.WHOLE_TENSOR)
            for index, factor in enumerate(model.initial_factors) for offset in range(5))
        plan = compile_page_bundle_plan(graph, timeline,
            TensorLayoutRequest((StateTensorLayout(str(_STATE), TensorLayoutSpec("canonical.qv_adamw@1", fields)),)),
            providers, 1, state_memory_budget_bytes, model.device, plan_epoch=plan_epoch)
        arena = PageBundleArena(plan)
        sampler_anchor, = (AnchorId(anchor.ordinal) for anchor in timeline.anchors if anchor.symbol == "sampler.complete")
        return StateExecution(arena, RequestStateDirectory(arena, graph.states[0]), self,
            (PublishStateTask[TLMData](publish_id), TLMUpdateTask(self, update_id)), sampler_anchor)


@dataclass(frozen=True, slots=True)
class TLMUpdateTask:
    math: TLMWorkload
    implementation_id: ImplementationId

    def execute(self, context: StateTaskContext[TLMData]) -> None:
        model, config, data = self.math.model, self.math.config, context.data()
        (ref,) = context.pin.refs
        with (nullcontext() if context.pin.stream is None else torch.cuda.stream(context.pin.stream)), torch.inference_mode(False), torch.enable_grad():
            factors = tuple(context.arena.borrow_field(context.pin, ref, LayoutFieldId(5 * index))
                .requires_grad_(True) for index in range(len(model.initial_factors)))
            selection = None
            if config.mode is TLMMode.OFFLINE:
                with torch.no_grad():
                    selection = model.logits(data.batch, None)
            objective = tlm_weighted_selection_objective(model.logits(data.batch, factors), data.batch.labels,
                selection_logits=selection, threshold=config.threshold, lamb=config.lamb, ignore_index=config.ignore_index)
            data.loss, data.selected_count = objective.loss.detach(), int(objective.selected.sum().item())
            if not data.selected_count or config.lamb == 0:
                return
            gradients = torch.autograd.grad(objective.loss, factors)
        context.write = context.arena.prepare_transition(context.pin.refs, context.pin.stream)
        (root,) = context.arena.writable_roots(context.write)
        groups = tuple(root[start:start + 5] for start in range(0, len(root), 5))
        with nullcontext() if context.pin.stream is None else torch.cuda.stream(context.pin.stream):
            apply_adamw(groups, _FIELDS, config.optimizer, gradients)
        context.loss_count = context.update_count = 1
        context.trained_rows = data.selected_count
