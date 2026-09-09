"""Shared and independent output-head CE/SGD on the original sampler and state path."""

from dataclasses import dataclass, replace
from collections.abc import Callable
from math import isfinite
from typing import Annotated

import torch
import pllm
from torch.nn import functional as F

from pllm import authoring as a
from pllm.config import ExecutionOptions
from pllm.kernels.paged_linear import PagedRows, TensorRows, paged_workspace
from pllm.execution.ptts.page_bundle import KernelCapture, PagedKernel
from pllm.execution.tensor_binding import TensorBinding, TensorFrame, TensorOutput, Tensors, select_rows, weighted_training_output
from pllm.contracts.request_identity import RequestKeyField, canonical_request_key_fields
from pllm.contracts.semantic_graph import ImplementationId, SchedulerTimeline, StateScope, TaskImplementationSpec, TaskSymbol, TaskResourceFact, ValueResourceFact
from pllm.contracts.tensor_layout import LayoutFieldId

_TRAIN = TaskSymbol("train")
_PREPARE = TaskSymbol("__train_prepare")
_TRAIN_ID = ImplementationId("canonical.lora:weighted_ce_sgd@1")
_CAPTURED_TRAIN_ID = ImplementationId("canonical.lora:cuda_graph_weighted_ce_sgd@1")
_PAGED_CAPTURED_TRAIN_ID = ImplementationId("canonical.lora:paged_cuda_graph_weighted_ce_sgd@1")


def _tensor_workspace(states: int, rows: int, hidden: int, vocab: int, rank: int) -> int:
    """FP32 tensor envelope: packed inputs/scores and parameter/gradient working copies."""
    return 4 * states * (rows * (hidden + 5 * vocab + rank) + 5 * rank * (hidden + vocab))


@dataclass(frozen=True, slots=True)
class OutputLoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    learning_rate: float = 1e-3
    steps: int = 1
    seed: int = 0
    scope: StateScope = StateScope.INSTANCE
    key_fields: tuple[RequestKeyField, ...] = (RequestKeyField.PROMPT_IDX, RequestKeyField.SAMPLE_IDX)
    paged: bool = False

    def __post_init__(self) -> None:
        if (min(self.rank, self.steps) < 1 or not isfinite(self.alpha) or self.alpha <= 0
            or not isfinite(self.learning_rate) or self.learning_rate <= 0 or not 0 <= self.seed < 2**63):
            raise ValueError("output LoRA requires positive rank/steps/scale/rate and a nonnegative seed")
        if self.scope not in (StateScope.INSTANCE, StateScope.REQUEST):
            raise ValueError("output LoRA supports instance or independent request state")
        if self.scope is StateScope.REQUEST:
            canonical_request_key_fields(self.key_fields)


def _resources(config: OutputLoRAConfig, facts: a.BackendCompileFacts, count: int, rows: int,
               prediction_rows: int, resident: int) -> SchedulerTimeline:
    hidden, vocab, rank = facts.probe.hidden_size, facts.vocab_size, config.rank
    workspace = _tensor_workspace(count, rows, hidden, vocab, rank)
    prediction_workspace = _tensor_workspace(prediction_rows if config.scope is StateScope.REQUEST else 1,
        1 if config.scope is StateScope.REQUEST else prediction_rows, hidden, vocab, rank)
    if config.paged:
        pages = max(provider.page_capacity for provider in facts.providers)
        workspace = paged_workspace(count * rows, facts.max_admitted_lineages, pages,
            hidden + vocab + rank, rank * (hidden + vocab))
        prediction_workspace = paged_workspace(prediction_rows, facts.max_admitted_lineages, pages,
            hidden + vocab + rank, rank * (hidden + vocab))
    parameters = count * rank * (hidden + vocab) * 4
    prepared = count * rows * (4 * (hidden + vocab + 1) + 8) + parameters
    return replace(facts.timeline,
        task_resources=(TaskResourceFact(_PREPARE, _tensor_workspace(count, rows, hidden, vocab, rank),
                input_shapes=((count, rows, hidden), (count, rows, vocab))),
            TaskResourceFact(TaskSymbol("predict"), prediction_workspace, 0, ((prediction_rows, hidden),)),
            TaskResourceFact(_TRAIN, workspace, resident, ((count, rows, hidden),))),
        value_resources=(ValueResourceFact(_PREPARE, "batch", prepared),
            ValueResourceFact(TaskSymbol("predict"), "value", prediction_rows * vocab * 4),
            ValueResourceFact(_TRAIN, "publication", parameters + count * 4)))


@dataclass(frozen=True)
class OutputLoRAWorkload(pllm.Workload):
    name = "lora.output_head"
    config: OutputLoRAConfig = OutputLoRAConfig()

    def resource_facts(self, facts: a.BackendCompileFacts, options: ExecutionOptions) -> SchedulerTimeline:
        self.require_native_methods(_NATIVE_MATH, resource_declaration=_RESOURCE_DECLARATION)
        count = facts.max_admitted_lineages if self.config.scope is StateScope.REQUEST else 1
        return _resources(self.config, facts, count, options.training_rows, facts.max_admitted_lineages,
            options.capture_memory_bytes if options.capture else 0)

    @property
    def output_factors(self) -> pllm.InstanceState:
        rank, seed = self.config.rank, self.config.seed
        def initialize(parameters: Tensors, generator: torch.Generator) -> None:
            parameters["a"].normal_(std=0.01, generator=generator.manual_seed(seed))
            parameters["b"].zero_()
        fields: dict[str, Callable[[a.Dimensions], tuple[int, ...]]] = {
            "a": lambda m: (m.hidden_size, rank), "b": lambda m: (rank, m.vocab_size)}
        if self.config.scope is StateScope.REQUEST:
            return pllm.RequestState(fields, optimizer=pllm.SGD(self.config.learning_rate), initialize=initialize, namespace="canonical.output_lora@1",
                keys=self.config.key_fields)
        return pllm.InstanceState(fields, optimizer=pllm.SGD(self.config.learning_rate), initialize=initialize, namespace="canonical.output_lora@1")

    def reset(self, parameters: Tensors, generator: torch.Generator) -> None:
        down = parameters["a"]
        down.copy_(torch.randn(down.shape, device=down.device, dtype=down.dtype,
            generator=generator.manual_seed(self.config.seed)) * 0.01)
        parameters["b"].zero_()

    @pllm.on(pllm.Phase.BEFORE_SAMPLE)
    @pllm.add_logits
    def predict(self, hidden: Annotated[torch.Tensor, pllm.Hidden(-1, pllm.HiddenSite.MODEL_OUTPUT)],
                down: Annotated[torch.Tensor, pllm.Parameter("output_factors", "a")],
                up: Annotated[torch.Tensor, pllm.Parameter("output_factors", "b")]) -> torch.Tensor:
        return torch.matmul(torch.matmul(hidden.unsqueeze(1), down), up).squeeze(1) * (self.config.alpha / self.config.rank)

    @pllm.on(pllm.Phase.AFTER_SAMPLE)
    @pllm.train(state="output_factors", steps=pllm.Config("config.steps"), reset="reset", reduction="weighted_state_mean_then_mean")
    def train(self, hidden: Annotated[torch.Tensor, pllm.Batch("hidden")],
              logits: Annotated[torch.Tensor, pllm.Batch("base_logits")],
              labels: Annotated[torch.Tensor, pllm.Batch("labels")],
              weights: Annotated[torch.Tensor, pllm.Batch("weights")],
              down: Annotated[torch.Tensor, pllm.Parameter("output_factors", "a")],
              up: Annotated[torch.Tensor, pllm.Parameter("output_factors", "b")]) -> torch.Tensor:
        scores = logits + hidden @ down @ up * (self.config.alpha / self.config.rank)
        row_losses = F.cross_entropy(scores.flatten(0, 1), labels.flatten(), reduction="none").view_as(weights)
        return (row_losses * weights).sum(1) / weights.sum(1).clamp_min(1e-8)

    def implementations(self, operation: str, resources: tuple[TaskResourceFact, ...], *,
                        explicit_math: bool = False) -> tuple[TaskImplementationSpec, ...]:
        if operation not in ("predict", "train"):
            return ()
        if not (explicit_math or self.config.paged or any(item.resident_bytes for item in resources)):
            return ()
        if operation == "predict":
            return (a.native("canonical.lora:paged_output@2", "compute/cuda@1") if self.config.paged else a.native("canonical.lora:output@1"),)
        resource = next((item for item in resources if item.task == _TRAIN), None)
        implementations = (a.native("canonical.lora:paged_weighted_ce_sgd@2", "compute/cuda@1"),) if self.config.paged else (a.native(_TRAIN_ID),)
        if resource is not None and resource.resident_bytes:
            implementations = (a.native(_PAGED_CAPTURED_TRAIN_ID if self.config.paged else _CAPTURED_TRAIN_ID,
                "compute/cuda_graph@1"), *implementations)
        return implementations

    def operations(self, facts: a.BackendCompileFacts, plan: a.PlanBundle,
                   b: TensorBinding) -> a.MathBinding[a.TensorStepInput, TensorFrame, TensorOutput]:
        self.require_native_methods(_NATIVE_MATH, resource_declaration=_RESOURCE_DECLARATION)
        if self.config.paged and facts.device.type != "cuda":
            raise ValueError("paged LoRA requires CUDA")
        limits = {item.task: item for item in plan.timeline.task_resources}
        if not {_PREPARE, TaskSymbol("predict"), _TRAIN} <= limits.keys():
            raise ValueError("LoRA requires signed preparation, prediction and training workspace")
        resource, prediction_limit = limits[_TRAIN], limits[TaskSymbol("predict")]
        if (len(resource.input_shapes) != 1 or len(resource.input_shapes[0]) != 3
            or len(prediction_limit.input_shapes) != 1 or len(prediction_limit.input_shapes[0]) != 2):
            raise ValueError("LoRA requires signed [states, rows, hidden] training and [rows, hidden] prediction shapes")
        count, rows, hidden = resource.input_shapes[0]
        prediction_rows = prediction_limit.input_shapes[0][0]
        if (min(count, rows, prediction_rows) < 1 or count > facts.max_admitted_lineages
            or hidden != facts.probe.hidden_size or self.config.scope is StateScope.INSTANCE and count != 1):
            raise ValueError("LoRA training shape differs from signed state/input capacity")
        self.require_resources(plan, _resources(self.config, facts, count, rows, prediction_rows, resource.resident_bytes))
        prepare_frame = b.bind_prepare()
        def prepare(value: a.TensorStepInput, batch: a.TensorBatch) -> TensorFrame:
            active_rows = sum(value.rows.rows[link.packed_row_index].phase is a.HiddenPhase.DECODE for link in value.sampler_rows.links)
            if active_rows > prediction_rows:
                raise ValueError("LoRA rows exceed the signed prediction envelope")
            return prepare_frame(value, batch)
        math = replace(self._functions(b, facts.device, facts.initialization_seed), prepare_data=prepare)
        if not self.config.paged and not any(task.implementation_id in (_TRAIN_ID, _CAPTURED_TRAIN_ID)
            for task in plan.schedule_physical.tasks):
            return a.MathBinding(math, b.bind_all(plan))
        self.require_native_methods(_NATIVE_MATH)
        config = self.config
        def training_inputs(data: TensorFrame) -> tuple[torch.Tensor, ...]:
            return (*data.batch_inputs, data.parameters[0], data.parameters[0].new_tensor(float(len(data.selected))))
        def train_tensors(fields: tuple[torch.Tensor, ...], inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            selected = torch.arange(fields[0].shape[0], device=fields[0].device)
            return training_loop(config, tuple(TensorRows(value.flatten(1), inputs[0].shape[1], writable=True)
                for value in fields), inputs, selected)
        def prediction(data: TensorFrame, batch: a.TensorBatch) -> a.TaskOutput:
            down, up = batch.roots[0]
            hidden = data.step.hidden_batch(batch.stream).snapshots[0].hidden
            eligible = {row.packed_row_index for row in data.rows}
            links = tuple(link for link in data.step.sampler_rows.links if link.packed_row_index in eligible)
            selected = select_rows(hidden, tuple(link.packed_row_index for link in links))
            indices = tuple(batch.selection.row_states[data.step.rows.rows[link.packed_row_index].identity] for link in links)
            with torch.no_grad():
                a_rows, b_rows = (batch.paged if config.paged else batch.tensor_rows)((0, 1), indices)
                delta = b_rows.linear(a_rows.linear(selected, 0,
                    (down.shape[0], config.rank)), 0, (config.rank, up.shape[1]))
                delta.mul_(config.alpha / config.rank)
            observations = tuple(a.Observation("guidance", delta[index], data.step.rows.rows[link.packed_row_index],
                batch.selection.refs[indices[index]], data.step.rows.step_id) for index, link in enumerate(links)) if batch.observing else ()
            return a.TaskOutput(guidance=a.Guidance(data.step.sampler_rows, delta,
                tuple(link.sampler_row_index for link in links), stage=a.SamplerStage.PACKED_PRE_FILTER_ADD), observations=observations)
        def optimize(data: TensorFrame, batch: a.TensorBatch) -> a.TaskOutput:
            if not data.selected:
                return a.TaskOutput()
            parameters: tuple[torch.Tensor, ...] = ()
            if batch.captured is not None:
                results = batch.captured
            else:
                inputs = training_inputs(data)
                if config.paged:
                    rows = tuple(state for state in data.selected for _ in range(inputs[0].shape[1]))
                    fields = batch.paged((0, 1), rows)
                    if data.trained_rows < len(rows):
                        fields = PagedRows.select_rows(fields, inputs[3].flatten() > 0)
                    results = training_loop(config, fields, inputs,
                        torch.tensor(data.selected, device=facts.device, dtype=torch.long))
                else:
                    parameters = data.parameters
                    results = training_loop(config, tuple(TensorRows(value.flatten(1), inputs[0].shape[1], writable=True)
                        for value in parameters), inputs, torch.arange(len(data.selected), device=facts.device), reset=False)
            return weighted_training_output(data, batch, results[0], results[1:], ("a", "b"), config.steps, parameters)
        operations = dict(zip((task.symbol for task in b.graph.tasks), b.bind(plan), strict=True))
        predict, train = operations[TaskSymbol("predict")], operations[_TRAIN]
        predict = replace(predict, function=prediction)
        train = replace(train, function=optimize)
        if train.implementation_id in (_CAPTURED_TRAIN_ID, _PAGED_CAPTURED_TRAIN_ID):
            def buffers(device: torch.device) -> tuple[torch.Tensor, ...]:
                return (torch.zeros((count, rows, hidden), device=device, dtype=torch.float32),
                    torch.zeros((count, rows, facts.vocab_size), device=device, dtype=torch.float32),
                    torch.zeros((count, rows), device=device, dtype=torch.long),
                    torch.ones((count, rows), device=device, dtype=torch.float32),
                    torch.zeros((count, hidden, config.rank), device=device, dtype=torch.float32),
                    torch.ones((), device=device, dtype=torch.float32),
                    *((torch.arange(count, device=device).repeat_interleave(rows),) if config.paged else ()))
            capture = KernelCapture("output_factors", (LayoutFieldId(0), LayoutFieldId(1)), count, buffers,
                PagedKernel.rows(6, lambda fields, tensors: training_loop(config, fields, tensors[:6],
                    torch.arange(count, device=tensors[0].device))) if config.paged else train_tensors, resource)

            def capture_inputs(data: TensorFrame, batch: a.TensorBatch) -> tuple[torch.Tensor, ...]:
                if not config.paged:
                    return training_inputs(data)
                actual_count, actual_rows = data.batch_inputs[0].shape[:2]
                state_rows = torch.arange(actual_count, device=facts.device).unsqueeze(1).expand(-1, rows)
                state_rows = state_rows.masked_fill(F.pad(data.batch_inputs[3], (0, rows - actual_rows)) <= 0, -1)
                return (*training_inputs(data), state_rows.flatten())
            train = replace(train, capture=capture, capture_inputs=capture_inputs,
                capture_padding=(0, 0, 0, 0, 0, 0, -1) if config.paged else ())
        operations.update({TaskSymbol("predict"): predict, _TRAIN: train})
        return a.MathBinding(math, tuple(operations.values()), () if train.capture is None else (train.capture,))

@torch.no_grad()
def training_loop(config: OutputLoRAConfig, fields: tuple[PagedRows | TensorRows, ...], inputs: tuple[torch.Tensor, ...],
                  selected: torch.Tensor, *, reset: bool = True) -> tuple[torch.Tensor, ...]:
    hidden, logits, labels, weights, initial, divisor = inputs
    down, up = fields
    a_shape, b_shape = (hidden.shape[2], config.rank), (config.rank, logits.shape[2])
    if reset:
        down.write(initial, selected)
        up.write(hidden.new_zeros((selected.numel(), *b_shape)), selected)
    source = hidden.flatten(0, 1)
    scale = config.alpha / config.rank
    totals = weights.sum(1).clamp_min(1e-8)
    row_weights = (weights / totals.unsqueeze(1) / divisor).flatten().unsqueeze(1) * scale
    label_rows = labels.flatten().unsqueeze(1)
    subtract = weights.new_full((weights.numel(), 1), -1)
    gradient_states = selected if isinstance(down, PagedRows) else None
    def step() -> tuple[torch.Tensor, ...]:
        projected = down.linear(source, 0, a_shape)
        log_probs = F.log_softmax(up.linear(projected, 0, b_shape).mul_(scale).add_(logits.flatten(0, 1)), dim=1)
        row_losses = F.nll_loss(log_probs, labels.flatten(), reduction="none").view_as(weights)
        losses = (row_losses * weights).sum(1) / totals
        derivative = log_probs.exp_().scatter_add_(1, label_rows, subtract).mul_(row_weights)
        da = down.parameter_gradient(source, up.input_gradient(derivative, 0, b_shape), gradient_states)
        db = up.parameter_gradient(projected, derivative, gradient_states)
        down.write(da, selected, add=True, scale=-config.learning_rate)
        up.write(db, selected, add=True, scale=-config.learning_rate)
        return losses, da, db

    result = step()
    for _ in range(config.steps - 1):
        result = step()
    return result


_NATIVE_MATH = pllm.Workload.native_methods({"predict": OutputLoRAWorkload.predict,
    "train": OutputLoRAWorkload.train, "reset": OutputLoRAWorkload.reset})
_RESOURCE_DECLARATION = OutputLoRAWorkload.resource_facts
