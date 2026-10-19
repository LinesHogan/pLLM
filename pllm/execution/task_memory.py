"""Maximum-shape task operands and derived resource facts for tensor authoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import torch

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import (
    HiddenInputSource,
    SchedulerTimeline,
    StateScope,
    TaskResourceFact,
    TaskResultInputSource,
    TaskSymbol,
    TokenwiseLayerReplay,
    ValueResourceFact,
)
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.memory import MemoryUsage, measure, optimizer_scratch
from pllm.execution.ptts.request_state import RequestStateSelection
from pllm.execution.shared_tensor import tensor_dtype
from pllm.execution.step_capture import TableLayout, reads_rows
from pllm.execution.tensor_frame import FieldRoots, RowIndex, TensorFrame, TensorMath
from pllm.execution.tensor_program import PoolAdamW, TensorBatch
from pllm.execution.tensor_task import TensorTask
from pllm.kernels.adamw import AdamWFields, apply_adamw

if TYPE_CHECKING:
    from pllm.authoring import BackendCompileFacts, TensorStepInput
    from pllm.config import ExecutionOptions
    from pllm.execution.tensor_binding import OverlayTransition, TensorBinding


class TaskInputs:
    """Meta operands matching the framework's gathered row and state views."""

    def __init__(
        self, binding: TensorBinding, facts: BackendCompileFacts, rows: int, task: TensorTask | None = None
    ) -> None:
        self.binding, self.facts, self.rows = binding, facts, rows
        selected: dict[TaskSymbol, TensorTask] = {}

        def collect(task: TensorTask) -> None:
            selected[task.symbol] = task
            for source in task.sources.values():
                if isinstance(source, TaskResultInputSource):
                    collect(binding.tensor_tasks[source.task])

        if task is not None:
            collect(task)
        self.read_fields = {
            port.field for item in selected.values() for port in item.ports.values() if port.field is not None
        }
        if any(item.policy.parameter_gradients for item in selected.values()):
            self.read_fields.add(0)
        self.feedback_names = {
            port.feedback for item in selected.values() for port in item.ports.values() if port.feedback
        }
        self.states = (
            facts.max_admitted_lineages if any(s.scope is StateScope.REQUEST for s in binding.graph.states) else 1
        )
        conditional = any(
            item.policy.update_states in item.domains
            or (binding.operations[symbol].optimizer is not None and item.policy.forward_rows)
            for symbol, item in binding.tensor_tasks.items()
            if reads_rows(item, binding.tensor_tasks)
        )
        self.gather = task is not None and reads_rows(task, binding.tensor_tasks) and not conditional
        self.state_rows = self.states + int(self.gather)
        self.fields = tuple(
            torch.empty((self.state_rows, *field.shape), device="meta", dtype=tensor_dtype(field.dtype))
            for entry in binding.layouts.layouts[:1]
            for field in entry.layout.fields
        )
        self.steps = tuple(torch.zeros(()) for _ in range(self.states))
        self.hidden = {
            (str(source.layer), "index", source.site.value): torch.empty(
                (rows, facts.probe.hidden_size),
                device="meta",
                dtype=torch.float64 if tensor_dtype(facts.probe.tensor_dtype) is torch.float64 else torch.float32,
            )
            for task in binding.tensor_tasks.values()
            for source in task.sources.values()
            if isinstance(source, HiddenInputSource)
        }
        self.feedback = {
            "base_logits": torch.empty((rows, facts.vocab_size or facts.probe.hidden_size), device="meta"),
            "sampled_tokens": torch.empty(rows, device="meta", dtype=torch.long),
        }
        self.feedback = {key: value for key, value in self.feedback.items() if key in self.feedback_names}

    def frame(self) -> tuple[TensorFrame, TensorBatch]:
        rows = tuple(
            PackedPhaseRow(i, i, 0, HiddenPhase.DECODE, RequestIdentity(str(i), i, 0)) for i in range(self.rows)
        )
        layout = TableLayout(self.rows, self.states, len(self.fields), len(self.feedback_names))
        table = torch.empty(layout.size, dtype=torch.long, device="meta")
        columns, counts, _, sources, _, _ = layout.split(table)
        # Trace the same table and page gathers as StepCapture, including all field slots.
        fields = tuple(
            value.index_select(0, slots.clamp_min(0)).masked_fill(
                (slots < 0).reshape(-1, *(1 for _ in value.shape[1:])), 0
            )
            if self.gather and i in self.read_fields
            else value
            for i, (value, slots) in enumerate(zip(self.fields, sources, strict=True))
        )
        frame = TensorFrame(
            cast("TensorStepInput", None),
            rows,
            fields=fields,
            index=RowIndex((), (), (), (), (), columns[:5], fixed=True),
            hidden_inputs=self.hidden,
            feedback_inputs={key: value.clone() for key, value in self.feedback.items()},
            counts=counts,
            weights=columns[5],
        )
        batch = TensorBatch(RequestStateSelection(rows, ()), FieldRoots(fields), (), None)
        return frame, batch

    def evaluate(self, task: TensorTask, frame: TensorFrame, batch: TensorBatch) -> tuple[torch.Tensor, ...]:
        for source in task.sources.values():
            if isinstance(source, TaskResultInputSource) and (source.task, source.output) not in batch.results:
                self.evaluate(self.binding.tensor_tasks[source.task], frame, batch)
        operation = self.binding.operations[task.symbol]
        training = operation.optimizer is not None
        if training or task.policy.parameter_gradients:
            frame.leaf = frame.fields[0].detach().requires_grad_(True)
        with torch.set_grad_enabled(training or task.differentiable):
            inputs, rows = task.read(frame, batch, compact=False)
            if training:
                # Scalar instance losses use the eager reduction; row losses use the padded one.
                if task.policy.reduction != "state_mean":
                    frame.weights = None
                loss, gradients = task.differentiate(inputs, rows, frame, batch)
                optimizer = operation.optimizer
                assert isinstance(optimizer, PoolAdamW)
                # Foreach/non-FP32 implementations can use more scratch than paged Adam.
                apply_adamw(
                    tuple(
                        tuple(self.steps[i] if j == 2 else value[i] for j, value in enumerate(self.fields[:5]))
                        for i in range(self.states)
                    ),
                    AdamWFields(*(LayoutFieldId(i) for i in range(5))),
                    optimizer.config,
                    tuple(gradients[0][i] for i in range(self.states)),
                )
                result = loss.detach()
                if task.policy.reduction == "state_mean":
                    result = result[: self.states]
                parameter_gradient = gradients[0].detach()[: self.states].contiguous()
                retained = (
                    result.clone(),
                    parameter_gradient.clone() if self.gather else parameter_gradient,
                    *(gradient.detach().clone() if self.gather else gradient.detach() for gradient in gradients[1:]),
                )
            else:
                result = task.function(inputs)
                if task.policy.forward_rows:
                    result = task.expand_selected(result, inputs, rows, frame)
                retained = (result.clone(),)
        batch.results[task.symbol, task.output] = result
        return retained


def derive_resources(
    binding: TensorBinding, facts: BackendCompileFacts, options: ExecutionOptions
) -> SchedulerTimeline:
    """Trace each declared task at its admitted row and state capacities."""
    from pllm.execution.tensor_binding import OverlayTransition

    tasks: list[TaskResourceFact] = []
    values: list[ValueResourceFact] = []
    for task in binding.graph.tasks:
        symbol = task.symbol
        if isinstance(task.replay, TokenwiseLayerReplay):
            continue
        rows = facts.max_admitted_lineages
        export = binding.exports.get(symbol)
        if export is not None:
            rows = min(export[0], facts.probe.max_packed_rows)
        inputs = TaskInputs(binding, facts, rows, binding.tensor_tasks.get(symbol))
        shapes = ((rows, facts.probe.hidden_size),)
        if symbol in binding.tensor_tasks:

            def run() -> object:
                if export is not None and not export[2]:
                    return ()
                frame, batch = inputs.frame()
                if export is not None:
                    frame.columns = export[1]
                return inputs.evaluate(binding.tensor_tasks[symbol], frame, batch)

            usage = measure(run)
            if binding.operations[symbol].optimizer is not None:
                optimizer = binding.operations[symbol].optimizer
                assert isinstance(optimizer, PoolAdamW)
                scratch = optimizer_scratch(tuple(inputs.fields[0][i] for i in range(inputs.states)), optimizer.config)
                usage = replace(usage, workspace=usage.workspace + scratch.workspace)
        elif symbol in binding.memory_probes:
            usage, shapes = binding.memory_probes[symbol](facts, options)
        elif symbol == "update" and (
            binding.overlay_import
            or any("canonical.activation:import" in str(i.implementation_id) for i in task.implementations)
        ):
            fields = tuple(value for value in inputs.fields)
            usage = MemoryUsage(
                max((measure(lambda: torch.isfinite(value).all()).workspace for value in fields), default=0), 0
            )
        elif symbol in binding.operations and isinstance(binding.operations[symbol].function, OverlayTransition):
            usage = overlay_memory(binding, cast(OverlayTransition, binding.operations[symbol].function), facts)
        else:
            usage = MemoryUsage(0, 0)
        tasks.append(
            TaskResourceFact(
                symbol,
                usage.workspace,
                options.capture_memory_bytes if options.capture and symbol in binding.tensor_tasks else 0,
                shapes,
            )
        )
        values.extend(ValueResourceFact(symbol, output.name, usage.output) for output in task.outputs)
    return replace(facts.timeline, task_resources=tuple(tasks), value_resources=tuple(values))


def weighted_memory(
    binding: TensorBinding,
    function: TensorMath,
    arguments: Mapping[str, tuple[str, str]],
    names: tuple[str, ...],
    steps: int,
    rate: float,
    facts: BackendCompileFacts,
    options: ExecutionOptions,
    *,
    prepare: bool,
) -> tuple[MemoryUsage, tuple[tuple[int, ...], ...]]:
    """Trace batched weighted losses and SGD, including the masked-batch preparation."""
    states = facts.max_admitted_lineages if binding.graph.states[0].scope is StateScope.REQUEST else 1
    shape = (states, options.training_rows)
    fields = binding.layouts.layouts[0].layout.fields
    parameters = {
        name: torch.empty((states, *field.shape), device="meta", dtype=tensor_dtype(field.dtype))
        for name, field in zip(names, fields, strict=True)
    }
    external = {
        "hidden": torch.empty((*shape, facts.probe.hidden_size), device="meta"),
        "base_logits": torch.empty((*shape, facts.vocab_size), device="meta"),
        "labels": torch.empty(shape, device="meta", dtype=torch.long),
        "weights": torch.empty(shape, device="meta"),
    }

    def run() -> object:
        if prepare:
            valid = (external["labels"] >= 0) & (external["labels"] < facts.vocab_size) & (external["weights"] > 0)
            packed = tuple(
                value.masked_fill(~valid.unsqueeze(-1) if value.ndim == 3 else ~valid, 0).clone()
                for value in external.values()
            )
            for value in packed:
                torch.isfinite(value).all()
            return packed
        leaves = {key: value.clone().requires_grad_(True) for key, value in parameters.items()}
        inputs = {key: (leaves if kind == "parameter" else external)[field] for key, (kind, field) in arguments.items()}
        losses = external["weights"]
        gradients: tuple[torch.Tensor, ...] = ()
        for _ in range(steps):
            losses = function(inputs)
            gradients = (
                torch.autograd.grad(losses.mean(), tuple(leaves.values()), materialize_grads=True)
                if losses.requires_grad
                else tuple(torch.zeros_like(parameter) for parameter in leaves.values())
            )
            with torch.no_grad():
                for parameter, gradient in zip(leaves.values(), gradients, strict=True):
                    parameter.add_(gradient, alpha=-rate)
        return (losses, *gradients)

    shapes = ((*shape, facts.probe.hidden_size),)
    if prepare:
        shapes = (*shapes, (*shape, facts.vocab_size))
    return measure(run), shapes


def overlay_memory(binding: TensorBinding, transition: OverlayTransition, facts: BackendCompileFacts) -> MemoryUsage:
    """Trace a request overlay's tensor function with maximum packed operands."""
    fields = binding.layouts.layouts[0].layout.fields
    width = len(transition.buffers)
    dtype = tensor_dtype(facts.probe.tensor_dtype)
    examples = tuple(
        {
            **{
                name: torch.empty(field.shape, dtype=tensor_dtype(field.dtype), device="meta")
                for name, field in zip(transition.buffers, fields[i * width : (i + 1) * width], strict=True)
            },
            "input": torch.empty((facts.probe.max_packed_rows, target.input_size), dtype=dtype, device="meta"),
            "target": torch.empty((facts.probe.max_packed_rows, target.output_size), dtype=dtype, device="meta"),
            "weight": torch.empty((target.output_size, target.input_size), dtype=dtype, device="meta"),
        }
        for i, target in enumerate(facts.targets)
    )
    with torch.no_grad():
        return measure(lambda: tuple(transition.function(values).output for values in examples))
