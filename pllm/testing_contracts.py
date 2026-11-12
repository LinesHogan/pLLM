"""Deterministic CPU schedule checks used by pllm.testing.check and workload authors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import dataclass, field, replace
from functools import partial
from math import lcm, prod
from traceback import format_exception_only
from types import MappingProxyType
from typing import Literal

import numpy as np
import torch
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from pllm.backends.vllm.compat import admit_parallel
from pllm.backends.vllm.compile_facts import compile_facts
from pllm.backends.vllm.model_shape import ModelShape
from pllm.config import ModelConfig
from pllm.contracts.backend_attachment import AttachmentGraphMode, AttachmentHiddenResource, BackendAttachment
from pllm.contracts.codec import boundary_digest
from pllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from pllm.contracts.host_batch import StateImport
from pllm.contracts.packed_hidden import HiddenRowBatch, HiddenRows
from pllm.contracts.packed_rows import PackedRows, RowLabeler, label_rows
from pllm.contracts.parameter_overlay import ProjectionOperands
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import SamplerFeedback, SamplerStage
from pllm.contracts.semantic_graph import (
    AnchorId,
    StateResource,
    StateScope,
    StateSelectionInputSource,
    TaskResultInputSource,
    TaskSymbol,
)
from pllm.contracts.state_ref import StateRef
from pllm.contracts.state_representation import StateDeviceKind
from pllm.execution.checkpoint import checkpoint_directory
from pllm.execution.compile_facts import BackendCompileFacts, BackendRequirements
from pllm.execution.model_sites import StateInput
from pllm.execution.runtime_facts import model_timeline
from pllm.execution.state_execution import StateExecution, StateTaskContext
from pllm.execution.step_driver import StepDriver
from pllm.execution.step_input import StepInput
from pllm.execution.tensor_declaration import TensorDeclaration
from pllm.execution.tensor_executor import reads_rows
from pllm.execution.tensor_frame import TensorFrame
from pllm.execution.tensor_task import TensorTask
from pllm.execution.token_program import TokenProgram
from pllm.execution.trace import TraceEventKind
from pllm.testing_forward import CheckAttention, CheckForward
from pllm.testing_kernels import check_sites, compare_method, replay_cells
from pllm.testing_timing import VersionChecks
from pllm.workload import Workload, component_workloads


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
    """Owned per-request observations; immutable tables are compared through their row outputs."""

    inputs: Mapping[str, torch.Tensor]
    outputs: Mapping[str, torch.Tensor]


# Keys use token positions, not engine-step numbers: mixed schedules advance independently.
Point = tuple[str, int]
Values = dict[tuple[str, int, str], torch.Tensor]


@dataclass
class _Run:
    device_equivalence: bool = False
    # Whether residual methods must reproduce the base model's residual (testing.check(neutral=)).
    neutral: bool = False
    timing: VersionChecks = field(default_factory=VersionChecks)
    preemption_skip: str | None = None
    values: Values = field(default_factory=lambda: {})
    operations: dict[tuple[str, int, str], OperationResult] = field(default_factory=lambda: {})

    failures: list[str] = field(default_factory=lambda: [])
    shared_resources: frozenset[str] = frozenset()
    immutable_resources: frozenset[str] = frozenset()
    shared_values: set[tuple[str, int, str]] = field(default_factory=lambda: set())

    def record(
        self,
        task: TensorTask,
        data: TensorFrame,
        inputs: Mapping[str, torch.Tensor],
        outputs: tuple[torch.Tensor, ...],
        *,
        rowwise: bool,
    ) -> None:
        if self.device_equivalence:
            expected = outputs if task.tuple_output else outputs[0]
            compare_method(task, inputs, expected)
        # Global results have no request axis; their state and export effects are still compared.
        if not rowwise:
            return
        rows = (
            data.rows
            if data.fixed is None
            else tuple(data.step.rows.rows[int(index)] for index in data.fixed.index.packed)
        )
        self.timing.record(task, data, rows)
        for index, row in enumerate(rows):
            if task.phase is not None and row not in data.step.rows.producer_rows(task.phase):
                self.failures.append(
                    f"pLLM misrouted request {row.identity.request_id!r} at token {row.token_offset} "
                    f"to `{task.symbol}` with tokens={task.phase.value!r}; report a framework phase-selection bug."
                )
            point = row.identity.request_id, row.token_offset, str(task.symbol)
            arguments = {
                name: (value[index] if task.ports[name].rowwise else value).detach().clone()
                for name, value in inputs.items()
                if not (
                    isinstance(source := task.sources[name], StateSelectionInputSource)
                    and str(source.resource) in self.immutable_resources
                )
            }
            results = {
                name: (value[index] if value.ndim and len(value) == len(rows) else value).detach().clone()
                for name, value in zip(task.outputs, outputs, strict=True)
            }
            self.operations[point] = OperationResult(MappingProxyType(arguments), MappingProxyType(results))
            if self.reads_shared(task, data, index, point[:2]):
                self.shared_values.update((*point[:2], f"method {task.symbol}.{name}") for name in results)
                if task.sampler_stage is not None:
                    self.shared_values.update(
                        (*point[:2], name) for name in (f"sampler {task.symbol}", f"sampler weight {task.symbol}")
                    )
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


# The backend features a CPU simulation of the vLLM runner executes.
SIMULATED_CAPABILITIES = (
    "execution/token_capture@1",
    "model/custom_forward@1",
    "optimizer/adamw@1",
    "state/fenced_reclaim@1",
    "transition/full_root@1",
)
_IDS = tuple(RequestIdentity(name, index, index) for index, name in enumerate(("r2", "r0", "r1")))
_PROMPTS = (5, 1, 9)
_LOGITS_CHANNEL = 90


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
                            _LOGITS_CHANNEL if layer == shape.layer_count - 1 else 120 + layer,
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


def contract_step_input(
    shape: ModelShape,
    facts: BackendCompileFacts,
    counts: tuple[tuple[int, int, int], ...],
    step: int,
    perturbed: int | None,
    identities: tuple[RequestIdentity, ...],
    history: int = -1,
) -> StepInput:
    """Build deterministic model inputs for one step of a contract-check schedule."""
    dtype = facts.probe.tensor_dtype.torch
    hidden = {
        resource: _hidden_values(shape, facts, resource, counts, channel, perturbed).to(
            torch.float32 if resource.binding.site is HiddenSite.LAYER_LOGITS else dtype
        )
        for resource in facts.probe.hidden_resources
        for channel in (int(boundary_digest(resource.binding.locator_key)[:8], 16),)
    }
    operands = tuple(
        ProjectionOperands(
            identities[i],
            target.target,
            phase,
            _token_values(i, offsets, target.input_size, 30 + channel, perturbed).to(dtype),
            _token_values(
                i,
                offsets,
                target.target_size,
                50 + channel,
                perturbed,
            ).to(dtype),
            _random((target.output_size, target.input_size), 80 + channel).to(dtype),
            torch.tensor(offsets.stop == _PROMPTS[i]),
            _token_values(i, offsets, shape.hidden_size, 70, perturbed).to(dtype),
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
    value = scheduled_step(
        tuple(identities[i] for i, _, _ in counts),
        [n for _, n, _ in counts],
        [_PROMPTS[i] for i, _, _ in counts],
        [done for _, _, done in counts],
        hidden,
        step_id=step,
        operands=operands,
    )

    return replace(value, input_tokens=_input_token_rows(shape, counts, history, perturbed))


def _input_token_rows(
    shape: ModelShape, counts: tuple[tuple[int, int, int], ...], history: int, perturbed: int | None
) -> torch.Tensor | None:
    """Seed complete request histories independently of scheduler chunk boundaries."""
    if history < 0:
        return None
    rows: list[torch.Tensor] = []
    for index, count, done in counts:
        seed = index + (1_000_000 if perturbed is not None and index != perturbed else 0)
        prompt = _PROMPTS[index]
        ids = torch.randint(shape.vocab_size, (prompt,), generator=torch.Generator().manual_seed(seed))
        if done + count > prompt:
            generated = _token_values(
                index, range(prompt - 1, done + count - 1), shape.vocab_size, _LOGITS_CHANNEL, perturbed
            ).argmax(-1)
            ids = torch.cat((ids, generated))
        positions = torch.arange(done, done + count)[:, None] - torch.arange(history + 1)[None, :]
        rows.append(ids[positions.clamp_min(0)].masked_fill(positions < 0, -1))
    return torch.cat(rows)


def _feedback(value: StepInput, shape: ModelShape, perturbed: int | None) -> SamplerFeedback:
    rows = value.rows.inspect(value.sampler_rows.packed_indices)
    logits = torch.cat(
        tuple(
            _token_values(
                row.identity.prompt_index,
                range(row.token_offset, row.token_offset + 1),
                shape.vocab_size,
                _LOGITS_CHANNEL,
                perturbed,
            )
            for row in rows
        )
    )
    return SamplerFeedback(value.sampler_rows, logits, logits.argmax(-1))


def _field_name(declaration: TensorDeclaration, resource: str, index: int) -> str:
    names = tuple(
        name for (state, name), view in declaration.field_views.items() if state == resource and view.field == index
    )
    return ",".join(names) if names else f"field_{index}"


def _state_values(
    execution: StateExecution, rows: PackedRows, declaration: TensorDeclaration
) -> dict[str, torch.Tensor]:
    selection = execution.directory.select(rows)
    immutable = {state.resource for state in declaration.graph.states if state.immutable}
    return {
        f"state {group.resource}.{_field_name(declaration, str(group.resource), index)}": value.detach().clone()
        for group in selection.groups
        if group.resource not in immutable
        for ref in group.refs
        for index, value in enumerate(execution.snapshot(ref))
    }


def _shared_resources(declaration: TensorDeclaration) -> frozenset[str]:
    return frozenset(
        str(state.resource)
        for state in declaration.graph.states
        if state.scope is StateScope.INSTANCE
        and any(
            task.resource == state.resource and task.training is not None for task in declaration.tensor_tasks.values()
        )
    )


def _check_idle_state(
    execution: StateExecution,
    versions: tuple[tuple[StateResource, StateRef], ...],
    run: _Run,
    observed: int,
    value: StepInput,
) -> None:
    """A tensor step without a method invocation cannot publish request state."""
    if execution.math.parameter_phases or len(run.operations) != observed or isinstance(value.batch, StateImport):
        return
    run.failures.extend(
        f"pLLM published request state {resource!r} without running a tensor method; "
        "report a framework tokens= or @pllm.on dispatch bug."
        for resource, ref in versions
        if execution.arena.latest(ref.lineage_id) != ref
    )


def _projection_step(driver: StepDriver, context: StateTaskContext, value: StepInput, run: _Run) -> None:
    """Exercise the model driver's prompt-before-decode projection segments without a loaded model."""
    task = driver.execution.plan.tasks[context.next_task]
    phases = tuple(phase for phase in HiddenPhase if any(o.phase is phase for o in value.operands))
    for phase in phases:
        operands = tuple(o for o in value.operands if o.phase is phase)
        outputs = driver.part(context, replace(value, operands=operands), task.task_id, complete=phase is phases[-1])
        for operand in operands:
            rows = tuple(row for row in value.rows.rows if row.identity == operand.identity and row.phase is phase)
            for row, tensor in zip(rows, outputs[operand.identity, operand.target], strict=True):
                run.values[row.identity.request_id, row.token_offset, f"projection {operand.target}"] = (
                    tensor.detach().clone()
                )


def _drive_step(
    driver: StepDriver,
    value: StepInput,
    feedback: SamplerFeedback,
    run: _Run,
    forward_anchor: AnchorId | None,
    sites: Callable[[Mapping[tuple[TaskSymbol, str], torch.Tensor]], None] | None,
) -> bool:
    """One step in VllmSession's order: begin, the CPU model's forward (a custom forward's anchor, residual methods),
    after_forward, sample, commit. False when the driver skipped it because the workload reads none of its rows."""
    execution = driver.execution
    run.timing.begin(execution, value)
    observed = len(run.operations)
    context = driver.begin(replace(value, hidden=None, hidden_readiness=None, operands=()), None)
    if context is not None and execution.arena.plan.semantic_graph.updates_in_forward:
        _projection_step(driver, context, value, run)
    if context is not None and forward_anchor is not None:
        driver.anchor(context, forward_anchor)
    if sites is not None:
        sites({} if context is None else context.results)
    driver.after_forward(value, None)
    if driver.step is None:
        run.timing.skip()
        return False
    context = driver.context()
    for stage in context.guidance:
        for index, sampler_index in enumerate(stage.affected_row_ids):
            row = value.rows.inspect_row(int(value.sampler_rows.packed_indices[sampler_index]))
            point = row.identity.request_id, row.token_offset
            run.values[*point, f"sampler {stage.member}"] = stage.values[index].detach().clone()
            if stage.row_weights is not None:
                run.values[*point, f"sampler weight {stage.member}"] = stage.row_weights[index].detach().clone()
    driver.sample(feedback, None)
    for entry in () if execution.arena.plan.semantic_graph.updates_in_forward else context.data().entries:
        rows = tuple(row for row in value.rows.rows if row.identity == entry.identity)
        for target in entry.targets:
            for row, tensor in zip(rows, target.update.output, strict=True):
                run.values[entry.identity.request_id, row.token_offset, f"projection {target.target}"] = (
                    tensor.detach().clone()
                )
    versions = tuple(
        (group.resource, ref) for group in context.selection.groups if group.keys != (None,) for ref in group.refs
    )
    driver.commit()
    _check_idle_state(execution, versions, run, observed, value)
    for event in execution.trace.snapshot().events:
        if event.kind is TraceEventKind.TENSOR and event.step_id == value.rows.step_id:
            for identity, position, tensor in zip(event.identities, event.positions, event.tensor(), strict=True):
                run.values[identity.request_id, position, f"export {event.tensor_name}"] = tensor
    return True


def _imported_batch(
    declaration: TensorDeclaration, identities: tuple[RequestIdentity, ...], perturbed: int | None
) -> StateImport[torch.Tensor] | None:
    """Seed distinct per-request rows, independent of admission and packed order."""
    if not declaration.imports:
        return None
    (resource,) = declaration.imports
    state = next(state for state in declaration.graph.states if state.resource == resource)
    if state.scope is not StateScope.REQUEST:
        return None
    layout = next(entry.layout for entry in declaration.layouts.layouts if entry.resource == resource)
    values = tuple(
        torch.stack(
            tuple(
                _token_values(identity.prompt_index, range(1), prod(view.shape), channel + 700, perturbed).reshape(
                    view.shape
                )
                for identity in identities
            )
        ).to(layout.fields[view.field].dtype.torch)
        for channel, view in enumerate(declaration.imports[resource])
    )
    return StateImport(values)


def _embeddings(value: StepInput, width: int, dtype: torch.dtype, perturbed: int | None) -> torch.Tensor:
    return torch.cat(
        tuple(
            _token_values(row.identity.prompt_index, range(row.token_offset, row.token_offset + 1), width, 0, perturbed)
            for row in value.rows.rows
        )
    ).to(dtype)


def _site_step(
    binding: TensorDeclaration,
    facts: BackendCompileFacts,
    value: StepInput,
    tokens: torch.Tensor,
    run: _Run,
    results: Mapping[tuple[TaskSymbol, str], torch.Tensor],
) -> None:
    outputs = check_sites(
        binding,
        replace(facts, max_admitted_lineages=len(tokens)),
        tokens,
        tokens * 0.5,
        seed=facts.initialization_seed,
        device_equivalence=run.device_equivalence,
        neutral=run.neutral,
        results=results,
    )
    for name, output in outputs.items():
        for row, tensor in zip(value.rows.rows, output, strict=True):
            run.values[row.identity.request_id, row.token_offset, f"custom_residual {name}"] = tensor.detach().clone()


def _run(
    workload: Workload,
    facts: BackendCompileFacts,
    declaration: TensorDeclaration,
    shape: ModelShape,
    order: tuple[int, ...],
    *,
    mixed: bool = False,
    perturbed: int | None = None,
    preempted: Literal["none", "full", "chunked"] = "none",
    device_equivalence: bool = False,
    neutral: bool = False,
) -> _Run:
    run = _Run(
        device_equivalence=device_equivalence,
        neutral=neutral,
        shared_resources=_shared_resources(declaration),
        immutable_resources=frozenset(str(state.resource) for state in declaration.graph.states if state.immutable),
    )
    tasks = declaration.tensor_tasks
    observed = copy(declaration)
    observed.tensor_tasks = {
        symbol: replace(task, observe=partial(run.record, task, rowwise=reads_rows(task, tasks)))
        if task.writer is None
        else task
        for symbol, task in tasks.items()
    }
    deployment = workload.bind_backend(
        facts, workload.plan_backend(facts, observed), observed, map_host=get_accelerator_view_from_cpu_tensor
    )
    execution = deployment.execution
    forward = None if declaration.custom_forward_programs is None else CheckForward(execution)
    attention = (
        CheckAttention(declaration, facts, device_equivalence=device_equivalence)
        if declaration.attention.methods
        else None
    )
    if forward is not None:
        forward.attention = attention
    driver = StepDriver(execution, deployment.attachment)
    if preempted != "none" and driver.preemption == "reset" and deployment.attachment.plan.schedule.recompute_rejection:
        run.preemption_skip = deployment.attachment.plan.schedule.recompute_rejection
        execution.close()
        return run
    forward_task = deployment.attachment.plan.semantic_graph.custom_forward_task
    forward_anchor = (
        None if forward_task is None else deployment.attachment.plan.scheduled(forward_task.symbol).anchor_id
    )
    execution.admit_instances()
    limit = workload.backend_requirements().sample_limit or 3
    script = tuple(replace(identity, sample_index=identity.sample_index % limit) for identity in _IDS)
    identities = tuple(script[i] for i in order)
    driver.stage(identities, _imported_batch(declaration, identities, perturbed))
    try:
        stride = lcm(*(task.export.every for task in tasks.values() if task.export is not None))
        for step, counts in enumerate(_schedule(order, mixed, preempted)):
            preempt = (script[1].request_id,) if preempted != "none" and step == 3 else ()
            driver.schedule((), preempt, [script[i].request_id for i, _, _ in counts])
            value = contract_step_input(
                shape, facts, counts, step * stride, perturbed, script, declaration.graph.input_token_history
            )
            if driver.preemption == "keep":
                value = retained_rows(driver.labeler, value, counts, script)
            sites = None
            if forward is not None or declaration.sites.rewrites or attention is not None:
                tokens = _embeddings(value, shape.hidden_size, facts.probe.tensor_dtype.torch, perturbed)
                if attention is not None:
                    if forward is None:
                        run.values.update(attention.step(value, tokens))
                    else:
                        attention.prepare(value)
                if forward is not None:
                    forward.prepare(value, tokens)
                if declaration.sites.rewrites:
                    sites = partial(_site_step, declaration, facts, value, tokens, run)
            if not _drive_step(driver, value, _feedback(value, shape, perturbed), run, forward_anchor, sites):
                continue
            if forward is not None:
                run.values.update(
                    ((row.identity.request_id, row.token_offset, "custom_forward"), output.detach().clone())
                    for row, output in zip(value.rows.rows, forward.output, strict=True)
                )
            producers = value.rows.producer_rows(None)
            for index, row in enumerate(producers):
                selected = producers[index : index + 1]
                for group in execution.directory.select(selected).groups:
                    if str(group.resource) in run.shared_resources and any(ref.version_id > 0 for ref in group.refs):
                        run.shared_values.update(
                            (row.identity.request_id, row.token_offset, name)
                            for name in _state_values(execution, selected, declaration)
                            if name.startswith(f"state {group.resource}.")
                        )
                run.values.update(
                    ((row.identity.request_id, row.token_offset, name), tensor)
                    for name, tensor in _state_values(execution, selected, declaration).items()
                )
    finally:
        driver.finish_call(wait=True)
        driver.release()
        driver.close()
    run.failures.extend(run.timing.failures)
    return run


def _difference(left: torch.Tensor, right: torch.Tensor) -> str:
    try:
        torch.testing.assert_close(left, right)
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
                f"pLLM misrouted rows before `{key[2]}`: its per-request inputs already differ. "
                "Report this message as a framework bug."
            )
    return (
        "pLLM misrouted rows: per-request method outputs agree, but the state, sampler or export destination differs. "
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
                + (
                    "custom attention mixes requests or reads future positions; keep selections and reductions request-local and causal."
                    if output.startswith("attention ")
                    else _fault(reference, candidate, (name, token))
                )
                + timing
            )
    return failures


def contract_facts(workload: Workload, config: ModelConfig) -> tuple[ModelShape, BackendCompileFacts]:
    """Build CPU contract-check facts and model dimensions without loading model weights."""
    requirements = workload.backend_requirements()
    shape = ModelShape.of_config(config, requirements.parameter_targets)
    facts = compile_facts(
        shape,
        requirements,
        replace(config, max_num_seqs=3, max_num_batched_tokens=15),
        model_identity=shape.architecture,
        hidden_layouts=(),
        overlay_graph_mode=AttachmentGraphMode.EAGER,
        memory_bytes=1 << 40,
        checkpoint=checkpoint_directory(config.model, config.revision) if requirements.checkpoint_names else "",
    )
    return shape, replace(
        facts,
        probe=replace(facts.probe, device_kind=StateDeviceKind.CPU),
        timeline=model_timeline(torch.device("cpu"), SIMULATED_CAPABILITIES),
        providers=tuple(
            replace(provider, provider_id="testing.cpu@1", device_kind=StateDeviceKind.CPU)
            for provider in facts.providers
        ),
    )


def _variants(
    workload: Workload,
    facts: BackendCompileFacts,
    declaration: TensorDeclaration,
    shape: ModelShape,
    exports: frozenset[str],
    skipped: list[CheckSkip],
    *,
    device_equivalence: bool = False,
) -> list[str]:
    _run_checked = partial(_run, device_equivalence=device_equivalence)
    failures: list[str] = []
    shared = bool(_shared_resources(declaration))
    together = _run_checked(workload, facts, declaration, shape, (0, 1, 2))
    alone = _Run()
    for index in range(3):
        single = _run_checked(workload, facts, declaration, shape, (index,))
        failures.extend(f"[alone] {failure}" for failure in single.failures)
        alone.values.update(single.values)
        alone.operations.update(single.operations)
        alone.shared_values.update(single.shared_values)
        alone.timing.values.update(single.timing.values)
    failures.extend(_compare(alone, together, "row permutation", shared=bool(shared), exports=exports))
    failures.extend(
        _compare(together, _run_checked(workload, facts, declaration, shape, (2, 1, 0)), "reversed", exports=exports)
    )
    for index, identity in enumerate(_IDS):
        failures.extend(
            _compare(
                together,
                _run_checked(workload, facts, declaration, shape, (0, 1, 2), perturbed=index),
                "isolation",
                request=identity.request_id,
                shared=bool(shared),
                exports=exports,
            )
        )
    failures.extend(
        _compare(
            alone,
            _run_checked(workload, facts, declaration, shape, (0, 1, 2), mixed=True),
            "mixed",
            shared=bool(shared),
            exports=exports,
        )
    )
    for replay in ("full", "chunked"):
        preempted = _run_checked(workload, facts, declaration, shape, (0, 1, 2), preempted=replay)
        if preempted.preemption_skip is not None:
            skipped.append(CheckSkip("preempted", preempted.preemption_skip))
            break
        failures.extend(_compare(alone, preempted, "preempted", shared=shared, exports=exports))
    return failures


def _component_independence(workload: Workload, config: ModelConfig) -> list[str]:
    """Compare each field's state, methods and sampler stages with its standalone execution."""
    failures: list[str] = []
    shape, facts = contract_facts(workload, config)
    facts, declaration = workload.specialize(facts, config.execution)
    composite = _run(workload, facts, declaration, shape, (0, 1, 2))
    outputs = composite.outputs()
    for path, component in component_workloads(workload)[1:]:
        shape, facts = contract_facts(component, config)
        facts, declaration = component.specialize(facts, config.execution)
        alone = _run(component, facts, declaration, shape, (0, 1, 2))
        for (request, token, output), expected in alone.outputs().items():
            kind = next(
                (
                    kind
                    for kind in ("sampler weight", "sampler", "method", "state", "export")
                    if output.startswith(kind + " ")
                ),
                None,
            )
            if kind is None:
                continue
            qualified = f"{kind} {path}.{output.removeprefix(kind + ' ')}"
            actual = outputs.get((request, token, qualified))
            try:
                assert actual is not None, "missing component output"
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            except AssertionError as error:
                failures.append(f"[component independence] {qualified}, request {request!r} at token {token}: {error}")
    return failures


def affects_model(workload: Workload, config: ModelConfig) -> bool:
    """Whether the workload changes anything; with every effect disabled (e.g. guidance with beta=0) it is the base
    model, which is exactly neutral."""
    return workload.build(contract_facts(workload, config)[1]).affects_model


def _check_neutral(neutral: Workload, config: ModelConfig, failures: list[str]) -> CheckSkip | None:
    """Run the neutral workload on the CPU schedules and append how it differs from the base model.

    Returns a skip when it declares a model effect the CPU harness does not execute; check_on_gpu compares those
    with vanilla generation."""
    if not affects_model(neutral, config):
        return None
    neutral_shape, neutral_facts = contract_facts(neutral, config)
    neutral_facts, neutral_declaration = neutral.specialize(neutral_facts, config.execution)
    sampler = neutral.plan_backend(neutral_facts, neutral_declaration).spec.sampler
    if any(stage.stage is SamplerStage.PACKED_POST_FILTER_EXACT for stage in sampler):
        failures.append("[neutral] guide_sampling still declares a sampler stage; set its beta to zero.")
    requirements = neutral.backend_requirements()
    unexecuted = {
        "a custom forward": requirements.custom_forward,
        "stateful or multi-stream residual methods": any(
            rewrite.memory is not None or rewrite.streams > 1 for rewrite in neutral_declaration.sites.rewrites
        ),
        "adapters": bool(requirements.parameter_targets),
        "activation patches": bool(requirements.activation_layers),
    }
    for effect, declared in unexecuted.items():
        if declared:
            return CheckSkip("neutral", f"the CPU harness does not execute {effect}; run check_on_gpu(neutral=)")
    try:
        result = _run(neutral, neutral_facts, neutral_declaration, neutral_shape, (0, 1, 2), neutral=True)
    except AssertionError as error:
        failures.append(str(error))
        return None
    failures.extend(result.failures)
    additive_outputs = {
        f"sampler {stage.task}" for stage in sampler if stage.stage is SamplerStage.PACKED_PRE_FILTER_ADD
    }
    failures.extend(
        f"[neutral] {key}: add_logits must return exactly zero"
        for key, value in result.values.items()
        if key[2] in additive_outputs and torch.count_nonzero(value)
    )
    return None


def _initial_parameters(
    declaration: TensorDeclaration, programs: tuple[TokenProgram, ...], seed: int
) -> dict[StateInput, torch.Tensor]:
    """Give the adaptive oracle the same declared initial fields as the state executor."""
    sources = tuple(dict.fromkeys(source for program in programs for _, source in program.parameters))
    layouts = {entry.resource: entry.layout for entry in declaration.layouts.layouts}
    roots = {
        resource: tuple(torch.zeros(field.shape, dtype=field.dtype.torch) for field in layouts[resource].fields)
        for resource in dict.fromkeys(source.resource for source in sources)
    }
    for resource, root in roots.items():
        declaration.initializer(resource, torch.device("cpu"), seed)((root,), (None,))
    return {source: source.view.read(roots[source.resource][source.view.field]) for source in sources}


def check_contracts(workload: Workload, config: ModelConfig, *, device_equivalence: bool = False) -> CheckResult:
    """Check request schedules, replay and state visibility on CPU, without loading a model.

    Uses fixed per-token inputs, real planner/executor paths and the runners' row labeller.
    ContractError lists every discrepancy; unsupported model and host effects have explicit skips.
    """
    passed: list[str] = []
    skipped: list[CheckSkip] = []
    requirements = workload.backend_requirements()
    admit_parallel(config, requirements)
    passed.append(f"tensor_parallel_size={config.tensor_parallel_size}: admission")
    if config.tensor_parallel_size > 1:
        skipped.append(CheckSkip("TP rank equality", "CPU schedules do not run TP workers; verify on multiple GPUs"))
    if requirements.causal_training or requirements.feeds:
        reason = "requires a causal model" if requirements.causal_training else "requires a host feed"
        return CheckResult(tuple(passed), (*skipped, CheckSkip("CPU schedules", reason)))
    try:
        shape, facts = contract_facts(workload, config)
        facts, declaration = workload.specialize(facts, config.execution)
        attachment = workload.plan_backend(facts, declaration)
    except (ValueError, RuntimeError, TypeError) as error:
        raise ContractError((f"[compile] {''.join(format_exception_only(error)).strip()}",)) from error
    passed.append("compile")
    from pllm.cost_estimates import estimate_tasks

    _, ops, _ = estimate_tasks(declaration, facts, config.execution, padded_vocab_size=shape.padded_vocab_size)
    missing = tuple(op.name for op in ops if not op.cpu_reference)
    if missing:
        return CheckResult(
            tuple(passed),
            tuple(
                CheckSkip(
                    "CPU schedules", f'no CPU kernel for {name}: add {name.split("::")[-1]}.register_kernel("cpu", ...)'
                )
                for name in missing
            ),
        )
    if device_equivalence and declaration.custom_forward_programs:
        _, gpu = workload.specialize(
            replace(facts, probe=replace(facts.probe, device_kind=StateDeviceKind.CUDA)), config.execution
        )
        assert gpu.custom_forward_programs is not None
        declaration.custom_forward_programs = replay_cells(
            declaration.custom_forward_programs, gpu.custom_forward_programs
        )
    programs = tuple((declaration.custom_forward_programs or {}).values())
    if any(program.adaptive for program in programs):
        _check_adaptive(declaration, programs, shape, facts, config.seed, passed, skipped)
    if (
        ops
        and requirements.custom_residual
        and not programs
        and all(rewrite.memory is None for rewrite in declaration.sites.rewrites)
    ):
        failures = _variants(
            workload, facts, declaration, shape, frozenset(), skipped, device_equivalence=device_equivalence
        )
        if failures:
            raise ContractError(tuple(failures))
        passed.append("custom-residual method math / CPU schedules")
    if (
        requirements.custom_residual
        and not declaration.model_site_tasks
        or programs
        and (
            requirements.parameter_targets
            or requirements.activation_layers
            or any(program.over == "span" or program.entries for program in programs)
        )
    ):
        return CheckResult(
            tuple(passed),
            (
                *skipped,
                CheckSkip(
                    "model-site execution",
                    "CPU decoder does not execute residual hooks, overlays, spans or PositionState",
                ),
            ),
        )
    if programs:
        passed.append("custom-forward instance reads / CPU decoder")
    if declaration.graph.trains_on_batch and not requirements.hidden_inputs and not requirements.sampler_stages:
        return CheckResult(
            tuple(passed), (CheckSkip("CPU schedules", "host-batch-only workload; no request computation"),)
        )
    exports = frozenset(
        prefix
        for task in declaration.tensor_tasks.values()
        if task.export is not None
        for prefix in (f"method {task.symbol}.", f"export {task.export.name}")
    )
    skipped.extend(_unchecked_effects(requirements, declaration, attachment, programs, exports))
    failures = _variants(workload, facts, declaration, shape, exports, skipped, device_equivalence=device_equivalence)
    passed.extend(_schedule_properties(declaration, skipped))
    if failures:
        raise ContractError(tuple(failures))
    return CheckResult(tuple(passed), tuple(skipped))


def _check_adaptive(
    declaration: TensorDeclaration,
    programs: tuple[TokenProgram, ...],
    shape: ModelShape,
    facts: BackendCompileFacts,
    seed: int,
    passed: list[str],
    skipped: list[CheckSkip],
) -> None:
    """Check adaptive-depth programs against the CPU decoder oracle, bitwise when no request exits."""
    from pllm.testing_depth import check_depth, check_never_exit

    parameters = _initial_parameters(declaration, programs, seed)
    dtype = facts.probe.tensor_dtype.torch
    for program in programs:
        if not program.adaptive:
            continue
        if program.entries or declaration.overlay_targets or declaration.sites.rewrites:
            skipped.append(CheckSkip("adaptive decoder", "PositionState and model-site composition need GPU"))
            continue
        try:
            passed.extend(check_depth(program, shape.layer_count, shape.hidden_size, dtype, parameters))
            check_never_exit(program, shape.layer_count, shape.hidden_size, dtype, parameters)
            passed.append("never exit bitwise")
        except AssertionError as error:
            raise ContractError((f"[adaptive depth] {error}",)) from error
    if parameters:
        passed.append("custom-forward instance reads / CPU decoder")
    skipped.append(CheckSkip("model arithmetic", "CPU fake decoder; native kernels need GPU"))


def _unchecked_effects(
    requirements: BackendRequirements,
    declaration: TensorDeclaration,
    attachment: BackendAttachment,
    programs: tuple[TokenProgram, ...],
    exports: frozenset[str],
) -> list[CheckSkip]:
    """What CPU schedules leave to the GPU checks, or compare only partly."""
    skipped: list[CheckSkip] = []
    if programs:
        skipped.append(
            CheckSkip("model attention / KV", "CPU decoder checks token math and state reads; real attention needs GPU")
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
    if requirements.custom_residual:
        skipped.append(
            CheckSkip("residual arithmetic", "CPU schedules check pre-forward results; residual hooks need GPU")
        )
    shared = _shared_resources(declaration)
    if shared:
        skipped.append(
            CheckSkip(
                "isolation after shared training", f"InstanceState {', '.join(sorted(shared))} learns from all requests"
            )
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
                "publish_lag counts the steps the workload runs: mixed/preempted values reading different permitted versions are not compared; every version read is checked",
            )
        )
    return skipped


def _schedule_properties(declaration: TensorDeclaration, skipped: list[CheckSkip]) -> list[str]:
    """The properties the schedule variants checked; adds a skip for each that does not apply."""
    passed: list[str] = []
    if any(
        state.resource in declaration.imports and state.scope is StateScope.REQUEST
        for state in declaration.graph.states
    ):
        passed.append("imported rows / request isolation")
    if declaration.attention.methods:
        passed.append("custom-attention method math / CPU paged KV schedules")
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
    return passed


def check(workload: Workload, config: ModelConfig, *, neutral: Workload | None = None) -> CheckResult:
    """Check fixed request schedules on CPU using config shapes and the real planner/executor.

    Permutes rows, perturbs neighbors, mixes chunked prompts, and replays preempted histories.
    Checks actual state versions against publish_lag; exact projection visibility uses chunk/replay equivalence.
    ContractError lists discrepancies and identifies workload row mixing or framework routing.
    Composites also compare each field component with its standalone outputs, bitwise.
    Pass neutral= with disabled author settings to check zero additive logits, absent guidance and residual methods
    that return the base residual; neutral model effects the CPU harness does not execute are reported as skips.
    Model execution, host inputs and shared training exceptions are reported as explicit skips.
    """
    failures: list[str] = []
    result = CheckResult((), ())
    try:
        result = check_contracts(workload, config)
        if len(component_workloads(workload)) > 1 and not any(
            skip.property in ("CPU schedules", "model-site execution", "adaptive decoder") for skip in result.skipped
        ):
            failures.extend(_component_independence(workload, config))
            result = replace(result, passed=(*result.passed, "component independence"))
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
            skip = _check_neutral(neutral, config, failures)
            result = (
                replace(result, passed=(*result.passed, "neutral"))
                if skip is None
                else replace(result, skipped=(*result.skipped, skip))
            )
        except (RuntimeError, ValueError, TypeError) as error:
            failures.append(f"[neutral compile/execution] {''.join(format_exception_only(error)).strip()}")
    if failures:
        raise ContractError(tuple(failures))
    return result


def retained_rows(
    labeler: RowLabeler,
    value: StepInput,
    counts: tuple[tuple[int, int, int], ...],
    script: tuple[RequestIdentity, ...],
) -> StepInput:
    """Relabel a step as the runner's labeler does when retained state survives preemption: tokens a request
    already consumed, including partial replay chunks, are marked recomputed.

    Reset-and-replay workloads do not call this: their model-state transitions must run again.
    """
    scheduled, prompt, computed = (
        np.array(column, dtype=np.int32) for column in zip(*((n, _PROMPTS[i], done) for i, n, done in counts))
    )
    ids = np.array([labeler.ids[script[i].request_id] for i, _, _ in counts], dtype=np.int64)
    sampler = labeler.prepare(scheduled, prompt, computed, ids, value.rows.step_id, True)
    hidden = None if value.hidden is None else replace(value.hidden, row_batch=sampler.row_batch)
    return replace(value, rows=sampler.row_batch, sampler_rows=sampler, hidden=hidden)
