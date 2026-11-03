"""Deterministic CPU schedule checks used by pllm.testing.check and workload authors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from math import lcm
from traceback import format_exception_only
from types import MappingProxyType
from typing import Literal

import numpy as np
import torch

from pllm.backends.vllm.compile_facts import compile_facts
from pllm.backends.vllm.model_shape import ModelShape
from pllm.config import ModelConfig
from pllm.contracts.backend_attachment import AttachmentHiddenResource
from pllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from pllm.contracts.packed_hidden import HiddenRowBatch, HiddenRows
from pllm.contracts.packed_rows import PackedPhaseRow, label_rows
from pllm.contracts.parameter_overlay import ProjectionOperands
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import SamplerFeedback, SamplerStage
from pllm.contracts.semantic_graph import (
    StateResource,
    StateScope,
    StateSelectionInputSource,
    TaskResultInputSource,
)
from pllm.contracts.state_ref import StateRef
from pllm.contracts.state_representation import StateDeviceKind
from pllm.execution.checkpoint import checkpoint_directory
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.preemption import preemption_policy, retained_rows
from pllm.execution.runtime_facts import model_timeline
from pllm.execution.state_execution import StateExecution, StateTaskContext
from pllm.execution.step_input import StepInput
from pllm.execution.tensor_binding import TensorBinding, TensorOutput
from pllm.execution.tensor_executor import reads_rows
from pllm.execution.tensor_frame import TensorFrame
from pllm.execution.tensor_task import TensorTask
from pllm.execution.trace import TraceEventKind
from pllm.testing_timing import VersionChecks
from pllm.usage import Workload


@dataclass(frozen=True, slots=True)
class CheckSkip:
    """A contract outside the declared workload's CPU-checkable behavior."""

    property: str
    reason: str


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Verified properties and explicit skips; dataclasses.asdict produces JSON-ready data."""

    passed: tuple[str, ...]
    skipped: tuple[CheckSkip, ...]

    def __str__(self) -> str:
        return "\n".join(
            (
                "status  contract / reason",
                *(f"PASS    {name}" for name in self.passed),
                *(f"SKIP    {item.property}: {item.reason}" for item in self.skipped),
            )
        )


class ContractError(AssertionError):
    """All observed contract failures, with the responsible method or framework routing named."""

    def __init__(self, failures: tuple[str, ...]) -> None:
        self.failures = failures
        super().__init__(f"{len(failures)} contract failure(s)\n" + "\n".join(failures))


@dataclass(frozen=True, slots=True)
class OperationResult:
    """Owned per-request method inputs and outputs; mappings cannot change during comparison."""

    inputs: Mapping[str, torch.Tensor]
    outputs: Mapping[str, torch.Tensor]


# Keys use token positions, not engine-step numbers: mixed schedules advance independently.
Point = tuple[str, int]
Values = dict[tuple[str, int, str], torch.Tensor]


@dataclass
class _Run:
    timing: VersionChecks = field(default_factory=VersionChecks)
    preemption_skip: str | None = None
    values: Values = field(default_factory=lambda: {})
    operations: dict[tuple[str, int, str], OperationResult] = field(default_factory=lambda: {})

    failures: list[str] = field(default_factory=lambda: [])
    shared_resources: frozenset[str] = frozenset()
    shared_values: set[tuple[str, int, str]] = field(default_factory=lambda: set())

    def record(
        self, task: TensorTask, data: TensorFrame, inputs: Mapping[str, torch.Tensor], outputs: tuple[torch.Tensor, ...]
    ) -> None:
        rows = (
            data.rows
            if data.fixed is None
            else tuple(
                data.step.rows.rows[data.step.sampler_rows.links[int(index)].packed_row_index]
                for index in data.fixed.index.sampler
            )
        )
        self.timing.record(task, data, rows)
        for index, row in enumerate(rows):
            if task.phase is not None and row not in data.step.rows.producer_rows(task.phase):
                self.failures.append(
                    f"pllm misrouted request {row.identity.request_id!r} at token {row.token_offset} "
                    f"to `{task.symbol}` with tokens={task.phase.value!r}; report a framework phase-selection bug."
                )
            point = row.identity.request_id, row.token_offset, str(task.symbol)
            arguments = {
                name: (value[index] if task.ports[name].rowwise else value).detach().clone()
                for name, value in inputs.items()
            }
            results = {
                name: (value[index] if value.ndim and len(value) == len(rows) else value).detach().clone()
                for name, value in zip(task.outputs, outputs, strict=True)
            }
            self.operations[point] = OperationResult(MappingProxyType(arguments), MappingProxyType(results))
            if self.reads_shared(task, data, index, point[:2]):
                self.shared_values.update((*point[:2], f"method {task.symbol}.{name}") for name in results)
                if task.sampler_stage is not None:
                    self.shared_values.update((*point[:2], name) for name in ("sampler", "sampler weight"))
                if task.export is not None:
                    self.shared_values.add((*point[:2], f"export {task.export.name}"))

    def outputs(self) -> Values:
        """Combine stored engine outputs with operation results only while comparing runs."""
        return self.values | {
            (request, token, f"method {method}.{name}"): value
            for (request, token, method), operation in self.operations.items()
            for name, value in operation.outputs.items()
        }

    def reads_shared(self, task: TensorTask, data: TensorFrame, index: int, point: Point) -> bool:
        for source in task.sources.values():
            if isinstance(source, StateSelectionInputSource) and str(source.resource) in self.shared_resources:
                if data.fixed is not None and int(data.fixed.for_resource(source.resource).index.versions[index]) > 0:
                    return True
            if (
                isinstance(source, TaskResultInputSource)
                and (*point, f"method {source.task}.{source.output}") in self.shared_values
            ):
                return True
        return False


_IDS = tuple(RequestIdentity(name, index, index) for index, name in enumerate(("r2", "r0", "r1")))
_PROMPTS = (5, 1, 9)


def _schedule(
    order: tuple[int, ...], mixed: bool, preempted: str = "none"
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    if mixed:
        return (
            ((0, 4, 0), (1, 1, 0)),
            ((2, 4, 0), (1, 1, 1), (0, 1, 4)),
            ((0, 1, 5), (1, 1, 2), (2, 4, 4)),
            ((2, 1, 8), (0, 1, 6), (1, 1, 3)),
            ((1, 1, 4), (2, 1, 9), (0, 1, 7)),
            ((0, 1, 8), (2, 1, 10)),
            ((2, 1, 11),),
            ((2, 1, 12),),
        )
    steps = (
        tuple((i, _PROMPTS[i], 0) for i in order),
        *(tuple((i, 1, _PROMPTS[i] + offset) for i in order) for offset in range(4)),
    )

    if preempted == "none":
        return steps
    replay = ((1, 1, 0),), ((1, 1, 1),)
    resume = ((1, 2, 2),) if preempted == "chunked" else ((1, _PROMPTS[1] + 3, 0),)
    resumed = tuple(item for item in steps[3] if item[0] != 1) + resume
    return (*steps[:3], *(replay if preempted == "chunked" else ()), resumed, steps[4])


def _random(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _token_values(index: int, offsets: range, width: int, channel: int, perturbed: int | None) -> torch.Tensor:
    salt = 1_000_000 if perturbed is not None and index != perturbed else 0
    return torch.stack(tuple(_random((width,), salt + 10000 * index + 100 * offset + channel) for offset in offsets))


def _hidden_values(
    shape: ModelShape,
    facts: BackendCompileFacts,
    resource: AttachmentHiddenResource,
    counts: tuple[tuple[int, int, int], ...],
    channel: int,
    perturbed: int | None,
) -> torch.Tensor:
    if resource.binding.site is HiddenSite.LAYER_LOGITS:
        return torch.cat(
            tuple(
                torch.stack(
                    tuple(
                        _token_values(
                            i,
                            range(done, done + n),
                            shape.vocab_size,
                            90 if layer == shape.layer_count - 1 else 120 + layer,
                            perturbed,
                        )
                        for layer in facts.probe.layer_logits
                    ),
                    dim=1,
                )
                for i, n, done in counts
            )
        )
    return torch.cat(
        tuple(_token_values(i, range(done, done + n), shape.hidden_size, channel, perturbed) for i, n, done in counts)
    )


def scheduled_step(
    identities: tuple[RequestIdentity, ...],
    scheduled: list[int],
    prompt: list[int],
    computed: list[int],
    hidden: Mapping[AttachmentHiddenResource, torch.Tensor],
    *,
    step_id: int = 0,
    operands: tuple[ProjectionOperands, ...] = (),
) -> StepInput:
    """Build a CPU step from scheduler counts, selecting only the runners' tapped rows.

    Hidden inputs contain one value per packed token. Only completed prompt endings
    and latest decode rows are retained, just as in HiddenTaps.prepare.
    """
    sampler = label_rows(identities, scheduled, prompt, computed, step_id)
    rows = sampler.row_batch
    taps: list[HiddenRows] = []
    for binding in dict.fromkeys(resource.binding for resource in hidden):
        sources = tuple((resource, value) for resource, value in hidden.items() if resource.binding == binding)
        selected = tuple(row for resource, _ in sources for row in rows.producer_rows(resource.phase))
        storage = torch.cat(
            tuple(
                value[[row.packed_row_index for row in rows.producer_rows(resource.phase)]]
                for resource, value in sources
            )
        )
        mapping = np.zeros(rows.total_rows, dtype=np.int64)
        for index, row in enumerate(selected):
            mapping[row.packed_row_index] = index
        taps.append(HiddenRows(binding, storage, mapping))
    return StepInput(rows, sampler, HiddenRowBatch(rows, tuple(taps)) if taps else None, operands=operands)


def _step(
    shape: ModelShape,
    facts: BackendCompileFacts,
    counts: tuple[tuple[int, int, int], ...],
    step: int,
    perturbed: int | None,
    identities: tuple[RequestIdentity, ...],
) -> StepInput:
    dtype = facts.probe.tensor_dtype.torch
    hidden = {
        resource: _hidden_values(shape, facts, resource, counts, channel, perturbed).to(
            torch.float32 if resource.binding.site is HiddenSite.LAYER_LOGITS else dtype
        )
        for channel, resource in enumerate(facts.probe.hidden_resources)
    }
    operands = tuple(
        ProjectionOperands(
            identities[i],
            target.target,
            phase,
            _token_values(i, offsets, target.input_size, 30 + channel, perturbed).to(dtype),
            _token_values(i, offsets, target.output_size, 50 + channel, perturbed).to(dtype),
            _random((target.output_size, target.input_size), 80 + channel).to(dtype),
            torch.tensor(offsets.stop == _PROMPTS[i]),
        )
        for i, n, done in counts
        for channel, target in enumerate(shape.targets)
        for phase, start, end in (
            (HiddenPhase.PREFILL, done, min(done + n, _PROMPTS[i])),
            (HiddenPhase.DECODE, max(done, _PROMPTS[i]), done + n),
        )
        if start < end
        for offsets in (range(start, end),)
    )
    return scheduled_step(
        tuple(identities[i] for i, _, _ in counts),
        [n for _, n, _ in counts],
        [_PROMPTS[i] for i, _, _ in counts],
        [done for _, _, done in counts],
        hidden,
        step_id=step,
        operands=operands,
    )


def _feedback(value: StepInput, shape: ModelShape, perturbed: int | None) -> SamplerFeedback:
    rows = tuple(value.rows.rows[link.packed_row_index] for link in value.sampler_rows.links)
    logits = torch.cat(
        tuple(
            _token_values(
                row.identity.prompt_index,
                range(row.token_offset, row.token_offset + 1),
                shape.vocab_size,
                90,
                perturbed,
            )
            for row in rows
        )
    )
    return SamplerFeedback(value.sampler_rows, logits, logits.argmax(-1))


def _field_name(declaration: TensorBinding, resource: str, index: int) -> str:
    names = tuple(
        name for (state, name), view in declaration.field_views.items() if state == resource and view.field == index
    )
    return ",".join(names) if names else f"field_{index}"


def _state_values(
    execution: StateExecution[StepInput, TensorFrame, TensorOutput], row: PackedPhaseRow, declaration: TensorBinding
) -> dict[str, torch.Tensor]:
    selection = execution.directory.select((row,))
    return {
        f"state {group.resource}.{_field_name(declaration, str(group.resource), index)}": value.detach().clone()
        for group in selection.groups
        for ref in group.refs
        for index, value in enumerate(execution.snapshot(ref))
    }


def _shared_resources(declaration: TensorBinding) -> frozenset[str]:
    return frozenset(
        str(state.resource)
        for state in declaration.graph.states
        if state.scope is StateScope.INSTANCE
        and any(
            task.resource == state.resource and task.training is not None for task in declaration.tensor_tasks.values()
        )
    )


def _check_idle_state(
    execution: StateExecution[StepInput, TensorFrame, TensorOutput],
    versions: tuple[tuple[StateResource, StateRef], ...],
    run: _Run,
    observed: int,
) -> None:
    """A tensor step without a method invocation cannot publish request state."""
    if execution.math.parameter_phases or len(run.operations) != observed:
        return
    run.failures.extend(
        f"pllm published request state {resource!r} without running a tensor method; "
        "report a framework tokens= or @pllm.on dispatch bug."
        for resource, ref in versions
        if execution.arena.latest(ref.lineage_id) != ref
    )


def _projection_step(
    execution: StateExecution[StepInput, TensorFrame, TensorOutput],
    context: StateTaskContext[TensorFrame],
    value: StepInput,
    run: _Run,
) -> None:
    """Exercise the model driver's prompt-before-decode projection segments without a loaded model."""
    task = execution.plan.tasks[context.next_task]
    phases = tuple(phase for phase in HiddenPhase if any(o.phase is phase for o in value.operands))
    for phase in phases:
        operands = tuple(o for o in value.operands if o.phase is phase)
        outputs = execution.execute_part(
            context, replace(value, operands=operands), task.task_id, complete=phase is phases[-1]
        )
        for operand in operands:
            rows = tuple(row for row in value.rows.rows if row.identity == operand.identity and row.phase is phase)
            for row, tensor in zip(rows, outputs[operand.identity, operand.target], strict=True):
                run.values[row.identity.request_id, row.token_offset, f"projection {operand.target}"] = (
                    tensor.detach().clone()
                )


def _execute_step(
    execution: StateExecution[StepInput, TensorFrame, TensorOutput],
    value: StepInput,
    feedback: SamplerFeedback,
    run: _Run,
) -> None:
    run.timing.begin(execution, value)
    overlap = execution.post_sample
    context = execution.start_step(value, history=() if overlap is None else overlap.history)
    versions = tuple(
        (group.resource, ref) for group in context.selection.groups if group.keys != (None,) for ref in group.refs
    )
    observed = len(run.operations)
    for anchor in execution.arena.plan.timeline.anchors:
        if anchor.symbol == "model.execute" and execution.arena.plan.semantic_graph.updates_in_forward:
            _projection_step(execution, context, value, run)
        execution.advance_to(context, anchor.symbol)
        if anchor.ordinal == execution.sampler_anchor_id:
            guidance = context.guidance
            if guidance is not None:
                for index, sampler_index in enumerate(guidance.affected_row_ids):
                    row = value.rows.rows[value.sampler_rows.links[sampler_index].packed_row_index]
                    point = row.identity.request_id, row.token_offset
                    run.values[*point, "sampler"] = guidance.values[index].detach().clone()
                    if guidance.row_weights is not None:
                        run.values[*point, "sampler weight"] = guidance.row_weights[index].detach().clone()
            execution.complete_sampler(context, feedback=feedback)
    for entry in () if execution.arena.plan.semantic_graph.updates_in_forward else context.data().entries:
        rows = tuple(row for row in value.rows.rows if row.identity == entry.identity)
        for target in entry.targets:
            for row, tensor in zip(rows, target.update.output, strict=True):
                run.values[entry.identity.request_id, row.token_offset, f"projection {target.target}"] = (
                    tensor.detach().clone()
                )
    execution.finish_step(context)
    _check_idle_state(execution, versions, run, observed)
    for event in execution.trace.snapshot().events:
        if event.kind is TraceEventKind.TENSOR and event.step_id == value.rows.step_id:
            for identity, tensor in zip(event.identities, event.tensor(), strict=True):
                row = next(row for row in value.rows.request_rows if row.identity == identity)
                run.values[identity.request_id, row.token_offset, f"export {event.tensor_name}"] = tensor


def _run(
    workload: Workload,
    facts: BackendCompileFacts,
    declaration: TensorBinding,
    shape: ModelShape,
    order: tuple[int, ...],
    *,
    mixed: bool = False,
    perturbed: int | None = None,
    preempted: Literal["none", "full", "chunked"] = "none",
) -> _Run:
    run = _Run(shared_resources=_shared_resources(declaration))
    tasks = declaration.tensor_tasks
    # Global results have no request axis; their state and export effects are still compared.
    declaration.tensor_tasks = {
        symbol: replace(task, observe=partial(run.record, task))
        if task.writer is None and reads_rows(task, tasks)
        else task
        for symbol, task in tasks.items()
    }
    try:
        deployment = workload.bind_backend(facts, workload.plan_backend(facts, declaration), declaration)
    finally:
        declaration.tensor_tasks = tasks
    execution = deployment.execution
    policy = preemption_policy(deployment.attachment, execution.math.model_sites)
    if (
        preempted != "none"
        and policy == "reset"
        and deployment.attachment.plan.schedule.recompute_rejection is not None
    ):
        run.preemption_skip = deployment.attachment.plan.schedule.recompute_rejection
        execution.close()
        return run
    execution.admit_instances()
    limit = workload.backend_requirements().sample_limit or 3
    script = tuple(replace(identity, sample_index=identity.sample_index % limit) for identity in _IDS)
    identities = tuple(script[i] for i in order)
    execution.admit(identities)
    frontiers: dict[RequestIdentity, int] = {}
    try:
        stride = lcm(*(task.export.every for task in tasks.values() if task.export is not None))
        for step, counts in enumerate(_schedule(order, mixed, preempted)):
            if preempted != "none" and step == 3 and policy == "reset":
                execution.retire((script[1],), wait=True)
                execution.admit((script[1],))
            value = _step(shape, facts, counts, step * stride, perturbed, script)
            if policy == "keep":
                sampler = retained_rows(value.sampler_rows, frontiers)
                value = replace(
                    value,
                    rows=sampler.row_batch,
                    sampler_rows=sampler,
                    hidden=None if value.hidden is None else replace(value.hidden, row_batch=sampler.row_batch),
                )
            _execute_step(execution, value, _feedback(value, shape, perturbed), run)
            for row in value.rows.producer_rows(None):
                for group in execution.directory.select((row,)).groups:
                    if str(group.resource) in run.shared_resources and any(ref.version_id > 0 for ref in group.refs):
                        run.shared_values.update(
                            (row.identity.request_id, row.token_offset, name)
                            for name in _state_values(execution, row, declaration)
                            if name.startswith(f"state {group.resource}.")
                        )
                run.values.update(
                    ((row.identity.request_id, row.token_offset, name), tensor)
                    for name, tensor in _state_values(execution, row, declaration).items()
                )
    finally:
        execution.release_history()
        execution.retire(identities, wait=True)
        execution.retire((), wait=True)
        execution.close()
    run.failures.extend(run.timing.failures)
    return run


def _difference(left: torch.Tensor, right: torch.Tensor) -> str:
    try:
        torch.testing.assert_close(left.float(), right.float())
        return ""
    except AssertionError:
        if left.shape != right.shape:
            return f"shape {tuple(left.shape)} versus {tuple(right.shape)}"
        difference = torch.where(left == right, 0, (left.float() - right.float()).abs())
        return f"max |diff| {difference.max().item():.6g}"


def _fault(reference: _Run, candidate: _Run, point: Point) -> str:
    for key, operation in candidate.operations.items():
        if key[:2] != point or key not in reference.operations:
            continue
        before = reference.operations[key]
        changed = any(_difference(value, before.outputs[name]) for name, value in operation.outputs.items())
        equal_inputs = operation.inputs.keys() == before.inputs.keys() and all(
            not _difference(value, before.inputs[name]) for name, value in operation.inputs.items()
        )
        if changed and equal_inputs:
            return (
                f"method `{key[2]}` mixes rows: equal inputs for this request produced different outputs. "
                "Dimension 0 holds request rows; reduce features (dim=-1), not requests (dim=0)."
            )
        if changed:
            return (
                f"pllm misrouted rows before `{key[2]}`: its per-request inputs already differ. "
                "Report this message as a framework bug."
            )
    return (
        "pllm misrouted rows: per-request method outputs agree, but the state, sampler or export destination differs. "
        "Report this message as a framework bug."
    )


def _compare(
    reference: _Run,
    candidate: _Run,
    variant: str,
    *,
    request: str | None = None,
    shared: bool = False,
    exports: frozenset[str] = frozenset(),
) -> list[str]:
    failures = [f"[{variant}] {failure}" for failure in candidate.failures]
    expected, actual = reference.outputs(), candidate.outputs()
    for key in dict.fromkeys((*expected, *actual)):
        name, token, output = key
        if (key not in expected and output.startswith("state ")) or request is not None and name != request:
            continue
        if shared and key in reference.shared_values | candidate.shared_values:
            continue
        if (
            variant in ("mixed", "preempted")
            and key in reference.timing.values
            and key in candidate.timing.values
            and reference.timing.values[key] != candidate.timing.values[key]
        ):
            continue
        left, right = expected.get(key), actual.get(key)
        if (left is None or right is None) and any(output.startswith(prefix) for prefix in exports):
            continue
        difference = (
            "missing or unexpectedly executed output" if left is None or right is None else _difference(left, right)
        )
        if difference:
            timing = " Check the method's tokens= selection and @pllm.on phase." if variant == "mixed" else ""
            failures.append(
                f"[{variant}] request {name!r}, {output} at token {token}: {difference}. "
                + _fault(reference, candidate, (name, token))
                + timing
            )
    return failures


def _facts(workload: Workload, config: ModelConfig) -> tuple[ModelShape, BackendCompileFacts]:
    requirements = workload.backend_requirements()
    shape = ModelShape.of_config(config, requirements.parameter_targets)
    facts = compile_facts(
        shape,
        requirements,
        replace(config, max_num_seqs=3, max_num_batched_tokens=15),
        model_identity=shape.architecture,
        hidden_layouts=(),
        overlay_graph_bytes=0,
        memory_bytes=1 << 40,
        checkpoint=checkpoint_directory(config.model, config.revision) if requirements.checkpoint_names else "",
    )
    return shape, replace(
        facts,
        probe=replace(facts.probe, device_kind=StateDeviceKind.CPU),
        timeline=model_timeline(
            torch.device("cpu"),
            ("transition/full_root@1", "optimizer/adamw@1", "model/custom_forward@1", "execution/token_capture@1"),
        ),
        providers=tuple(
            replace(provider, provider_id="testing.cpu@1", device_kind=StateDeviceKind.CPU)
            for provider in facts.providers
        ),
    )


def _variants(
    workload: Workload,
    facts: BackendCompileFacts,
    declaration: TensorBinding,
    shape: ModelShape,
    exports: frozenset[str],
    skipped: list[CheckSkip],
) -> list[str]:
    failures: list[str] = []
    shared = bool(_shared_resources(declaration))
    together = _run(workload, facts, declaration, shape, (0, 1, 2))
    alone = _Run()
    for index in range(3):
        single = _run(workload, facts, declaration, shape, (index,))
        failures.extend(f"[alone] {failure}" for failure in single.failures)
        alone.values.update(single.values)
        alone.operations.update(single.operations)
        alone.shared_values.update(single.shared_values)
        alone.timing.values.update(single.timing.values)
    failures.extend(_compare(alone, together, "row permutation", shared=bool(shared), exports=exports))
    failures.extend(
        _compare(together, _run(workload, facts, declaration, shape, (2, 1, 0)), "reversed", exports=exports)
    )
    for index, identity in enumerate(_IDS):
        failures.extend(
            _compare(
                together,
                _run(workload, facts, declaration, shape, (0, 1, 2), perturbed=index),
                "isolation",
                request=identity.request_id,
                shared=bool(shared),
                exports=exports,
            )
        )
    failures.extend(
        _compare(
            alone,
            _run(workload, facts, declaration, shape, (0, 1, 2), mixed=True),
            "mixed",
            shared=bool(shared),
            exports=exports,
        )
    )
    for replay in ("full", "chunked"):
        preempted = _run(workload, facts, declaration, shape, (0, 1, 2), preempted=replay)
        if preempted.preemption_skip is not None:
            skipped.append(CheckSkip("preempted", preempted.preemption_skip))
            break
        failures.extend(_compare(alone, preempted, "preempted", shared=shared, exports=exports))
    return failures


def _neutral_failures(neutral: Workload, config: ModelConfig) -> list[str]:
    failures: list[str] = []
    neutral_shape, neutral_facts = _facts(neutral, config)
    neutral_facts, neutral_declaration = neutral.specialize(neutral_facts, config.execution)
    sampler = neutral.plan_backend(neutral_facts, neutral_declaration).spec.sampler
    stage = None if sampler is None else sampler.stage
    if stage is SamplerStage.PACKED_POST_FILTER_EXACT:
        failures.append("[neutral] guide_sampling still declares a sampler stage; set its beta to zero.")
    if neutral.backend_requirements().custom_forward or neutral.backend_requirements().custom_residual:
        neutral.plan_backend(neutral_facts, neutral_declaration)
        return failures
    result = _run(neutral, neutral_facts, neutral_declaration, neutral_shape, (0, 1, 2))
    failures.extend(result.failures)
    failures.extend(
        f"[neutral] {key}: add_logits must return exactly zero"
        for key, value in result.values.items()
        if key[2] == "sampler" and torch.count_nonzero(value)
    )
    return failures


def _check(workload: Workload, config: ModelConfig) -> CheckResult:
    """Check request schedules, replay and state visibility on CPU, without loading a model.

    Uses fixed per-token inputs, real planner/executor paths and the runners' row labeller.
    ContractError lists every discrepancy; unsupported model and host effects have explicit skips.
    """
    passed: list[str] = []
    skipped: list[CheckSkip] = []
    failures: list[str] = []
    requirements = workload.backend_requirements()
    if requirements.causal_training or requirements.feeds:
        reason = "requires a causal model" if requirements.causal_training else "requires a host feed"
        return CheckResult((), (CheckSkip("CPU schedules", reason),))
    try:
        shape, facts = _facts(workload, config)
        facts, declaration = workload.specialize(facts, config.execution)
        attachment = workload.plan_backend(facts, declaration)
    except (ValueError, RuntimeError, TypeError) as error:
        raise ContractError((f"[compile] {''.join(format_exception_only(error)).strip()}",)) from error
    passed.append("compile")
    if requirements.custom_forward and declaration.custom_forward_programs is not None:
        from pllm.testing_depth import check_depth, check_never_exit

        for program in declaration.custom_forward_programs.values():
            if program.adaptive:
                if program.entries or declaration.overlay_targets or declaration.sites.rewrites:
                    skipped.append(CheckSkip("adaptive decoder", "PositionState and model-site composition need GPU"))
                    continue
                try:
                    passed.extend(
                        check_depth(program, shape.layer_count, shape.hidden_size, facts.probe.tensor_dtype.torch)
                    )
                    check_never_exit(program, shape.layer_count, shape.hidden_size, facts.probe.tensor_dtype.torch)
                    passed.append("never exit bitwise")
                except AssertionError as error:
                    raise ContractError((f"[adaptive depth] {error}",)) from error
        skipped.append(CheckSkip("model arithmetic", "CPU fake decoder; native kernels need GPU"))
        return CheckResult(tuple(passed), tuple(skipped))
    if requirements.custom_residual:
        return CheckResult(
            tuple(passed), (CheckSkip("model-site execution", "custom_forward/custom_residual need a loaded model"),)
        )
    if requirements.parameter_targets:
        skipped.append(
            CheckSkip(
                "adapter application",
                "operand math and state are checked; applying the adapter inside the model needs GPU",
            )
        )
    if requirements.activation_layers:
        skipped.append(CheckSkip("activation application", "needs a loaded model; state producers are checked"))
    if declaration.graph.trains_on_batch:
        skipped.append(CheckSkip("host training", "no host batch in the request script; inference is checked"))
        if not requirements.hidden_inputs and requirements.sampler_stage is None:
            return CheckResult(
                tuple(passed), (CheckSkip("CPU schedules", "host-batch-only workload; no request computation"),)
            )
    shared = _shared_resources(declaration)
    if shared:
        skipped.append(
            CheckSkip(
                "isolation after shared training", f"InstanceState {', '.join(sorted(shared))} learns from all requests"
            )
        )
    exports = frozenset(
        prefix
        for task in declaration.tensor_tasks.values()
        if task.export is not None
        for prefix in (f"method {task.symbol}.", f"export {task.export.name}")
    )
    if exports:
        skipped.append(
            CheckSkip(
                "export eligibility",
                "every= is exercised at aligned step ids; max_rows omissions are allowed, delivered values are compared",
            )
        )
    if any(task.read_distance for task in attachment.plan.schedule.tasks):
        skipped.append(
            CheckSkip(
                "timing-dependent values",
                "publish_lag counts engine steps: mixed/preempted values reading different permitted versions are not compared; every version read is checked",
            )
        )
    failures.extend(_variants(workload, facts, declaration, shape, exports, skipped))
    passed.extend(("row permutation", "request isolation", "mixed prefill/decode", "chunking / tokens="))
    if any(
        isinstance(source, StateSelectionInputSource)
        for task in declaration.tensor_tasks.values()
        for source in task.sources.values()
    ):
        passed.append("publish_lag")
    else:
        skipped.append(CheckSkip("publish_lag", "no row method reads versioned state"))
    if declaration.graph.updates_in_forward:
        passed.append("visibility=exact (chunking / replay)")
    if any(
        task.visibility is not None and task.visibility.policy == "forward_pinned" for task in declaration.graph.tasks
    ):
        skipped.append(
            CheckSkip(
                "visibility=forward_pinned",
                "model application needs GPU; operand math and step-entry state are checked",
            )
        )
    if not any(skip.property == "preempted" for skip in skipped):
        passed.append("preempted")
    if failures:
        raise ContractError(tuple(failures))
    return CheckResult(tuple(passed), tuple(skipped))


def check(workload: Workload, config: ModelConfig, *, neutral: Workload | None = None) -> CheckResult:
    """Check fixed request schedules on CPU using config shapes and the real planner/executor.

    Permutes rows, perturbs neighbors, mixes chunked prompts, and replays preempted histories.
    Checks actual state versions against publish_lag; exact projection visibility uses chunk/replay equivalence.
    ContractError lists discrepancies and identifies workload row mixing or framework routing.
    Pass neutral= with disabled author settings to check zero additive logits or absent guidance.
    Model execution, host inputs and shared training exceptions are reported as explicit skips.
    """
    failures: list[str] = []
    result = CheckResult((), ())
    try:
        result = _check(workload, config)
    except ContractError as error:
        failures.extend(error.failures)
    except (RuntimeError, ValueError, TypeError) as error:
        failures.append(f"[CPU execution] {''.join(format_exception_only(error)).strip()}")
    if neutral is None:
        result = replace(
            result, skipped=(*result.skipped, CheckSkip("neutral", "pass neutral= with the author's disabled settings"))
        )
    else:
        try:
            failures.extend(_neutral_failures(neutral, config))
            result = replace(result, passed=(*result.passed, "neutral"))
        except (RuntimeError, ValueError, TypeError) as error:
            failures.append(f"[neutral compile/execution] {''.join(format_exception_only(error)).strip()}")
    if failures:
        raise ContractError(tuple(failures))
    return result
