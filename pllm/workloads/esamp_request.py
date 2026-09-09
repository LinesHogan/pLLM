"""Request distillation expressed as prediction, loss and an AdamW update."""

from dataclasses import dataclass, replace
from math import isfinite
from typing import Annotated

import torch
import pllm

from pllm import authoring as a
from pllm.config import ExecutionOptions
from pllm.execution.ptts.page_capture import CapturedPageFields, KernelCapture, PagedKernel
from pllm.execution.tensor_binding import TensorBinding, TensorFrame, TensorOutput, Tensors, select_rows, training_observations
from pllm.kernels.low_rank_gated import LowRankGatedLayout, low_rank_gated_forward, low_rank_gated_loss_and_gradients
from pllm.kernels.paged_linear import PagedRows, TensorRows, paged_workspace
from pllm.kernels.adamw import AdamWFields, apply_adamw
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.backend_attachment import BackendAttachmentSpec
from pllm.contracts.request_identity import RequestKeyField, canonical_request_key_fields
from pllm.contracts.sampler import PackedSamplerGuidanceBatch
from pllm.contracts.semantic_graph import (
    SchedulerTimeline, SemanticGraph, TaskImplementationSpec, TaskResourceFact, TaskSymbol, ValueResourceFact,
)
from pllm.contracts.tensor_layout import LayoutFieldId, MutationGranularity, TensorLayoutRequest

ADAM_FIELDS = AdamWFields(LayoutFieldId(0), LayoutFieldId(1), LayoutFieldId(2), LayoutFieldId(3), LayoutFieldId(4))


@dataclass(frozen=True, slots=True)
class ESampRequestConfig:
    rank: int = 128
    source_layer: int = 0
    target_layer: int = -1
    output_layernorm: bool = True
    intervention: bool = False
    intervention_beta: float = 0.0
    learning_rate: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    key_fields: tuple[RequestKeyField, ...] = (RequestKeyField.REQUEST_ID,)
    paged: bool = False

    def __post_init__(self) -> None:
        a.AdamW(self.learning_rate, self.betas, self.eps, self.weight_decay)
        canonical_request_key_fields(self.key_fields)
        if self.rank < 1 or not isfinite(self.intervention_beta) or self.intervention_beta < 0.0:
            raise ValueError("ESamp request configuration is invalid")


def initialize(parameters: Tensors, generator: torch.Generator) -> None:
    for name in ("up", "gate", "down"):
        parameters[name].normal_(0.0, 0.02, generator=generator)
    parameters["bias"].fill_(1.0)
    parameters["scale"].fill_(1.0)
    parameters["shift"].zero_()


def _resources(config: ESampRequestConfig, facts: a.BackendCompileFacts, states: int, rows: int,
               prediction_rows: int, resident: int) -> SchedulerTimeline:
    layout = LowRankGatedLayout(facts.probe.hidden_size, config.rank, config.output_layernorm)
    hidden, size = layout.hidden_size, layout.parameter_count
    def workspace(rows: int) -> int:
        # Known FP32 packing, saved intermediates and gradients, excluding library scratch.
        return (paged_workspace(rows, facts.max_admitted_lineages,
            max(provider.page_capacity for provider in facts.providers), hidden + config.rank, size) if config.paged
            else 4 * (rows * (2 * size + 16 * (hidden + config.rank)) + states * 4 * size))
    # Direct AdamW: two five-field slot tables (80 B), offsets (4 B), corrections (8 B), rounded per state.
    optimizer_workspace = states * (96 if config.paged else 24 * size)
    return replace(facts.timeline, task_resources=(TaskResourceFact(TaskSymbol("__predict_ready"), prediction_rows * 16),
        TaskResourceFact(TaskSymbol("loss"), workspace(rows), resident,
            ((rows, hidden), (rows, hidden), (states, size), (rows,)) if resident else ((rows, hidden),)),
        TaskResourceFact(TaskSymbol("predict"), workspace(prediction_rows), 0, ((prediction_rows, hidden),)),
        TaskResourceFact(TaskSymbol("train"), optimizer_workspace, resident, ((states, size),))),
        value_resources=(ValueResourceFact(TaskSymbol("__predict_ready"), "value", prediction_rows),
            ValueResourceFact(TaskSymbol("loss"), "value", 4 * states * (size + 1)),
            ValueResourceFact(TaskSymbol("predict"), "value", 8 * prediction_rows * hidden),
            ValueResourceFact(TaskSymbol("train"), "publication", 0)))


class ESampRequestWorkload(pllm.Workload):
    name = "esamp.request"
    state_name = "request_distiller"

    def __init__(self, config: ESampRequestConfig = ESampRequestConfig()) -> None:
        self.config = config

    @property
    def request_distiller(self) -> pllm.RequestState:
        config = self.config
        rank, norm = config.rank, config.output_layernorm
        return pllm.RequestState({"up": lambda m: (m.hidden_size, rank), "gate": lambda m: (m.hidden_size, rank),
            "down": lambda m: (rank, m.hidden_size), "bias": (rank,),
            "scale": lambda m: (m.hidden_size if norm else 0,), "shift": lambda m: (m.hidden_size if norm else 0,)},
            initialize=initialize, optimizer=self.optimizer, alignment=4, mutation=MutationGranularity.PAGE,
            namespace="canonical.low_rank_gated_adamw@1", keys=config.key_fields)

    def ready(self, versions: Annotated[torch.Tensor, pllm.StateVersion("request_distiller")]) -> torch.Tensor:
        return (versions > 0) & self.config.intervention

    def forward(self, source: torch.Tensor, up: torch.Tensor, gate: torch.Tensor, down: torch.Tensor,
                bias: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        x = source.unsqueeze(1)
        product = torch.bmm(x, up).squeeze(1) * torch.nn.functional.silu(torch.bmm(x, gate).squeeze(1) + bias)
        output = source + torch.bmm(product.unsqueeze(1), down).squeeze(1)
        return torch.nn.functional.layer_norm(output, (source.shape[1],)) * scale + shift if self.config.output_layernorm else output

    @pllm.on(pllm.Phase.BEFORE_SAMPLE, rows="ready")
    @pllm.guide_sampling(beta=pllm.Config("config.intervention_beta"))
    def predict(self, source: Annotated[torch.Tensor, pllm.Hidden(pllm.Config("config.source_layer"))],
                up: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "up")],
                gate: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "gate")],
                down: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "down")],
                bias: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "bias")],
                scale: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "scale")],
                shift: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "shift")],
                versions: Annotated[torch.Tensor, pllm.StateVersion("request_distiller")]) -> torch.Tensor:
        return self.forward(source, up, gate, down, bias, scale, shift)

    @pllm.on(pllm.Phase.AFTER_SAMPLE)
    @pllm.train(state="request_distiller", reduction="state_mean")
    def loss(self, source: Annotated[torch.Tensor, pllm.Hidden(pllm.Config("config.source_layer"))],
             target: Annotated[torch.Tensor, pllm.Hidden(pllm.Config("config.target_layer"))],
             up: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "up")],
             gate: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "gate")],
             down: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "down")],
             bias: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "bias")],
             scale: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "scale")],
             shift: Annotated[torch.Tensor, pllm.Parameter("request_distiller", "shift")],
             groups: Annotated[torch.Tensor, pllm.StateIndex("request_distiller")]) -> torch.Tensor:
        losses = (self.forward(source, up, gate, down, bias, scale, shift) - target).square().mean(1)
        counts = torch.bincount(groups).to(losses)
        return losses.new_zeros(counts.shape).index_add(0, groups, losses) / counts

    @property
    def optimizer(self) -> a.AdamW:
        config = self.config
        return a.AdamW(config.learning_rate, config.betas, config.eps, config.weight_decay)

    def implementations(self, operation: str, resources: tuple[TaskResourceFact, ...], *,
                        explicit_math: bool = False) -> tuple[TaskImplementationSpec, ...]:
        if operation not in ("predict", "loss", "train") or not (
            explicit_math or self.config.paged or any(fact.resident_bytes for fact in resources)):
            return ()
        transition = operation == "train"
        if self.config.paged and not transition:
            eager = a.native(f"canonical.esamp_request:paged_{operation}@2", "compute/cuda@1")
            return (eager,) if operation == "predict" else (
                a.native("canonical.esamp_request:paged_cuda_graph_loss@1", "compute/cuda_graph@1"), eager)
        cpu = a.native(f"canonical.esamp_request:cpu_{operation}@1", "compute/cpu@1",
            *(("state/dense_page_cpu@1",) if transition else ()))
        cuda = a.native(f"canonical.esamp_request:cuda_{operation}@1", "compute/cuda@1",
            *(("state/dense_page_cuda@1",) if transition else ()))
        return (cpu, cuda) if operation == "predict" else (cpu,
            a.native(f"canonical.esamp_request:cuda_graph_{operation}@1", "compute/cuda_graph@1", "state/dense_page_cuda@1"), cuda)

    def resource_facts(self, facts: a.BackendCompileFacts, options: ExecutionOptions) -> SchedulerTimeline:
        self.require_native_methods(_NATIVE_MATH, resource_declaration=_RESOURCE_DECLARATION)
        count = facts.max_admitted_lineages
        return _resources(self.config, facts, count, count, count, options.capture_memory_bytes if options.capture else 0)

    def declare(self, facts: a.BackendCompileFacts) -> tuple[SemanticGraph, TensorLayoutRequest, BackendAttachmentSpec]:
        if self.config.key_fields != (RequestKeyField.REQUEST_ID,):
            raise ValueError("attached ESamp requires request-id state keys")
        return super().declare(facts)

    def operations(self, facts: a.BackendCompileFacts, plan: PlanBundle,
                   b: TensorBinding) -> a.MathBinding[a.TensorStepInput, TensorFrame, TensorOutput]:
        if plan.timeline.task_resources:
            self.require_native_methods(_NATIVE_MATH, resource_declaration=_RESOURCE_DECLARATION)
        config, device = self.config, facts.device
        layout = LowRankGatedLayout(facts.probe.hidden_size, config.rank, config.output_layernorm)
        bindings = {binding.locator_key: binding for binding in (resource.binding for resource in facts.probe.hidden_resources)}
        requested = tuple((str(layer), "index", "block_output") for layer in (config.source_layer, config.target_layer))
        if set(bindings) != set(requested) or len(bindings) != 2:
            raise ValueError("ESamp hidden bindings differ from the signed declaration")
        explicit_math = "compute/explicit_math@1" in plan.timeline.capabilities
        if not config.paged and not explicit_math and not plan.timeline.task_resources:
            return super().operations(facts, plan, b)
        if config.paged and device.type != "cuda":
            raise ValueError("paged ESamp requires CUDA")
        limits = {fact.task: fact for fact in plan.timeline.task_resources}
        if not {"__predict_ready", "predict", "loss", "train"} <= limits.keys():
            raise ValueError("ESamp requires signed predicate, prediction, loss and optimizer workspace")
        loss_limit, train_limit, prediction_limit = (limits[TaskSymbol(name)] for name in ("loss", "train", "predict"))
        if (not loss_limit.input_shapes or len(loss_limit.input_shapes[0]) != 2
            or len(train_limit.input_shapes) != 1 or len(train_limit.input_shapes[0]) != 2
            or len(prediction_limit.input_shapes) != 1 or len(prediction_limit.input_shapes[0]) != 2):
            raise ValueError("ESamp requires signed [rows, hidden] and [states, parameters] shapes")
        row_count, hidden = loss_limit.input_shapes[0]
        states, parameters = train_limit.input_shapes[0]
        prediction_rows = prediction_limit.input_shapes[0][0]
        if (min(row_count, prediction_rows, states) < 1 or states > min(row_count, facts.max_admitted_lineages)
            or hidden != layout.hidden_size or parameters != layout.parameter_count):
            raise ValueError("ESamp shapes disagree with the mathematical layout or state capacity")
        self.require_resources(plan, _resources(config, facts, states, row_count, prediction_rows, loss_limit.resident_bytes))
        prepare_frame = b.bind_prepare()
        def prepare(value: a.TensorStepInput, batch: a.TensorBatch) -> TensorFrame:
            data = prepare_frame(value, batch)
            if len(data.rows) > min(row_count, prediction_rows) or len(batch.roots) > states:
                raise ValueError("ESamp rows or states exceed the signed workspace envelope")
            return data
        math = replace(self._functions(b, device, facts.initialization_seed), prepare_data=prepare)
        if not config.paged and not explicit_math and not any(fact.resident_bytes for fact in plan.timeline.task_resources):
            return a.MathBinding(math, b.bind_all(plan))
        self.require_native_methods(_NATIVE_MATH)
        def loss_inputs(data: TensorFrame, batch: a.TensorBatch) -> tuple[torch.Tensor, ...]:
            hidden = data.step.hidden_batch(batch.stream)
            indices = tuple(row.packed_row_index for row in data.rows)
            return tuple(select_rows(hidden.hidden_for(bindings[locator]), indices) for locator in requested)
        def gradient_inputs(data: TensorFrame, batch: a.TensorBatch) -> tuple[torch.Tensor, ...]:
            output = data.pending_update
            if output is None:
                raise RuntimeError("optimizer requires its planned loss result")
            return output.gradients
        def evaluate_loss(data: TensorFrame, batch: a.TensorBatch) -> a.TaskOutput:
            if batch.captured is None:
                inputs = loss_inputs(data, batch)
                fields, = (batch.paged if config.paged else batch.tensor_rows)((0,), batch.selection.state_indices)
                losses, gradients = low_rank_gated_loss_and_gradients(inputs[0], inputs[1], fields, layout)
            else:
                losses, gradients = batch.captured
                inputs = loss_inputs(data, batch) if batch.observing else ()
            data.pending_update = a.TaskOutput(gradients=(gradients,))
            data.parameter_gradient_fields.update((ref, (value,)) for ref, value in zip(batch.selection.refs, gradients.unbind(), strict=True))
            return a.TaskOutput(loss=losses, loss_count=len(losses), observations=training_observations(data, batch,
                losses, dict(zip(("source", "target"), inputs[:2])), tuple(range(len(data.rows)))))
        def prediction(data: TensorFrame, batch: a.TensorBatch) -> a.TaskOutput:
            local = {row.packed_row_index: index for index, row in enumerate(data.rows)}
            links = tuple(link for link in data.step.sampler_rows.links if config.intervention
                and link.packed_row_index in local
                and batch.selection.refs[batch.selection.state_indices[local[link.packed_row_index]]].version_id > 0)
            source = data.step.hidden_batch(batch.stream).hidden_for(bindings[requested[0]])
            hidden = source.new_empty((0, layout.hidden_size))
            if links:
                indices = tuple(batch.selection.state_indices[local[link.packed_row_index]] for link in links)
                fields, = (batch.paged if config.paged else batch.tensor_rows)((0,), indices)
                hidden = low_rank_gated_forward(select_rows(source, tuple(link.packed_row_index for link in links)), fields, layout)
            return a.TaskOutput(guidance=PackedSamplerGuidanceBatch(data.step.sampler_rows, hidden,
                tuple(link.sampler_row_index for link in links), config.intervention_beta))
        def optimize(data: TensorFrame, batch: a.TensorBatch) -> a.TaskOutput:
            if batch.captured is None:
                gradients, = gradient_inputs(data, batch)
                if config.paged:
                    PagedRows.adamw(batch.paged((0, 1, 2, 3, 4), ()), gradients, self.optimizer)
                else:
                    apply_adamw(batch.roots, ADAM_FIELDS, self.optimizer, tuple(gradients.unbind()))
            data.pending_update = None
            return a.TaskOutput(update_count=len(batch.roots), trained_rows=len(data.rows))
        operations = dict(zip((task.symbol for task in b.graph.tasks), b.bind(plan), strict=True))
        loss = replace(operations[TaskSymbol("loss")], function=evaluate_loss)
        predict = replace(operations[TaskSymbol("predict")], function=prediction)
        train = replace(operations[TaskSymbol("train")], function=optimize, mode="transition", optimizer=None, publish=True)
        captured_loss = loss.implementation_id in ("canonical.esamp_request:cuda_graph_loss@1",
            "canonical.esamp_request:paged_cuda_graph_loss@1")
        if captured_loss != (train.implementation_id == "canonical.esamp_request:cuda_graph_train@1"):
            raise ValueError("ESamp captured training requires both signed kernels")
        if captured_loss:
            if device.type != "cuda" or not loss_limit.resident_bytes or not train_limit.resident_bytes:
                raise ValueError("captured loss/update requires signed CUDA input shapes and resident entitlements")
            def loss_buffers(device: torch.device) -> tuple[torch.Tensor, ...]:
                return (torch.zeros((row_count, hidden), device=device, dtype=torch.float32),
                    torch.zeros((row_count, hidden), device=device, dtype=torch.float32),
                    torch.arange(row_count, device=device) % states)

            def gradient_buffers(device: torch.device) -> tuple[torch.Tensor, ...]:
                return (torch.zeros((states, parameters), device=device, dtype=torch.float32),)

            def compute(values: tuple[torch.Tensor | PagedRows, ...], inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
                fields = values[0]
                return low_rank_gated_loss_and_gradients(inputs[0], inputs[1],
                    fields if isinstance(fields, PagedRows) else TensorRows(fields, inputs[2]), layout)

            def update(view: CapturedPageFields, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
                view.arena.adamw_captured(view, inputs[0], self.optimizer)
                return ()

            captured = KernelCapture(self.state_name, (ADAM_FIELDS.parameter,), states, loss_buffers,
                PagedKernel.rows(2, compute) if self.config.paged else compute, loss_limit, writable=False)
            def capture_inputs(data: TensorFrame, batch: a.TensorBatch) -> tuple[torch.Tensor, ...]:
                return (*loss_inputs(data, batch), torch.tensor(batch.selection.state_indices, dtype=torch.long, device=device))
            loss = replace(loss, capture=captured, capture_inputs=capture_inputs, capture_padding=(0, 0, -1))
            fields = (ADAM_FIELDS.parameter, ADAM_FIELDS.gradient, ADAM_FIELDS.step, ADAM_FIELDS.exp_avg, ADAM_FIELDS.exp_avg_sq)
            train = replace(train, capture=KernelCapture(self.state_name, fields, states, gradient_buffers, PagedKernel(update), train_limit),
                capture_inputs=gradient_inputs)
        operations.update({TaskSymbol("loss"): loss, TaskSymbol("predict"): predict, TaskSymbol("train"): train})
        return a.MathBinding(math, tuple(operations.values()),
            tuple(operation.capture for operation in operations.values() if operation.capture is not None))


_NATIVE_MATH = pllm.Workload.native_methods({name: function for name, function in (
    ("predict", ESampRequestWorkload.predict), ("loss", ESampRequestWorkload.loss),
    ("forward", ESampRequestWorkload.forward), ("ready", ESampRequestWorkload.ready))})
_RESOURCE_DECLARATION = ESampRequestWorkload.resource_facts


__all__ = ["ESampRequestConfig", "ESampRequestWorkload"]
