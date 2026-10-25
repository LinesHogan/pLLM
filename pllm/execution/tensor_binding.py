"""Tensor declarations bound once to the plan: model effects, row tasks and their execution binding.

TensorBinding combines ModelEffectBinding and RowTaskBinding and binds the declared tasks to the
specialized plan (native implementations, scans, fixed-shape steps, input preparation).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch

from pllm.contracts.auxiliary import StateTensorUpdate
from pllm.contracts.backend_attachment import (
    AttachmentKind,
)
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.request_identity import RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerStage
from pllm.contracts.semantic_graph import (
    HiddenInputSource,
    StateScope,
)
from pllm.contracts.state_ref import StateRef
from pllm.execution.ptts.page_bundle import PagePublication
from pllm.execution.scan import ScanContext
from pllm.execution.shared_tensor import tensor_dtype
from pllm.execution.step_capture import StepCapture
from pllm.execution.tensor_executor import TensorExecutor, reads_rows
from pllm.execution.tensor_frame import (
    OverlayEntry,
    TensorFrame,
    flat_gradient,
)
from pllm.execution.tensor_program import (
    InvocationOutput,
    PageUpdate,
    TensorBatch,
)

if TYPE_CHECKING:
    from pllm.execution.compile_facts import BackendCompileFacts
    from pllm.execution.step_input import TensorStepInput


from pllm.execution.model_effect_binding import ModelEffectBinding
from pllm.execution.row_task_binding import RowTaskBinding


@dataclass(frozen=True, slots=True)
class TensorResult:
    value: torch.Tensor | None
    invocation: InvocationOutput
    entries: tuple[OverlayEntry, ...] = ()
    input_gradients: Mapping[tuple[str, str], torch.Tensor] = field(
        default_factory=lambda: dict[tuple[str, str], torch.Tensor]()
    )
    parameter_gradient_fields: Mapping[StateRef, tuple[torch.Tensor, ...]] = field(
        default_factory=lambda: dict[StateRef, tuple[torch.Tensor, ...]]()
    )

    @property
    def parameter_gradients(self) -> Mapping[StateRef, torch.Tensor]:
        """Explicit flat inspection; execution retains the original gradient fields."""
        return {ref: flat_gradient(values) for ref, values in self.parameter_gradient_fields.items()}

    @property
    def loss(self) -> torch.Tensor | None:
        return self.invocation.loss

    @property
    def guidance(self) -> PackedSamplerGuidanceBatch | None:
        return self.invocation.guidance

    @property
    def publication(self) -> PagePublication | None:
        return self.invocation.publication


TensorOutput = TensorResult | torch.Tensor | None


class TensorBinding(ModelEffectBinding, RowTaskBinding):
    """Authoring projections only; tensor results remain invocation-local."""

    def configure_scans(self, facts: BackendCompileFacts) -> None:
        """Bind scan specialization to the plan's policy and row capacity."""
        if self.overlay_math is not None and any(task.replay is not None for task in self.graph.tasks):
            self.overlay_math = replace(
                self.overlay_math,
                function=ScanContext(
                    facts.timeline.region_policy,
                    self.route_rows,
                    facts.device,
                ).bind(self.overlay_math.function),
            )
        for symbol, operation in self.operations.items():
            budget = facts.timeline.memory_budget_bytes
            scans = ScanContext(
                facts.timeline.region_policy,
                facts.probe.max_packed_rows,
                facts.device,
                None if budget is None else budget // max(1, facts.max_admitted_lineages),
            )
            self.operations[symbol] = replace(operation, body=scans.bind(operation.body))

    def configure_steps(self, facts: BackendCompileFacts, capture: bool) -> None:
        """Select the same tensor step for planning and execution."""
        captured = {
            symbol: task
            for symbol, task in self.tensor_tasks.items()
            if reads_rows(task, self.tensor_tasks) and symbol not in self.exports
        }
        conditional = any(
            task.policy.update_states in task.domains
            or (self.operations[symbol].optimizer and task.policy.forward_rows)
            for symbol, task in captured.items()
        )
        if conditional:
            if capture:
                raise ValueError("step capture needs row-wise independent training without dynamic update selection")
            return
        limits = {fact.task: fact for fact in self.resources}
        states = (
            facts.max_admitted_lineages if any(state.scope is StateScope.REQUEST for state in self.graph.states) else 1
        )
        for symbol, task in captured.items():
            operation = self.operations[symbol]
            optimizer = operation.optimizer
            assert optimizer is None or isinstance(optimizer, PageUpdate)
            executor = TensorExecutor(
                task,
                self.tensor_tasks,
                optimizer,
                compact=False,
                capture=StepCapture(limits[symbol], states) if capture else None,
                device=facts.device,
            )
            self.operations[symbol] = replace(operation, body=executor, capture_resident=executor.resident)

    def bind_prepare(self) -> Callable[[TensorStepInput, TensorBatch], TensorFrame]:
        dimensions = self.dimensions
        if dimensions is None:
            raise ValueError("tensor execution requires specialized model dimensions")
        if (
            self.attachment is not None
            and (
                self.attachment.kind is AttachmentKind.MODEL_REPLAY
                or any(task.replay is not None for task in self.graph.tasks)
            )
            and not self.attachment.activation_bindings
        ):

            def prepare_replay(value: TensorStepInput, batch: TensorBatch) -> TensorFrame:
                if (
                    value.auxiliary is not None
                    or value.operands
                    or value.hidden is not None
                    or value.rows.plan_epoch != batch.epoch
                ):
                    raise ValueError("tokenwise replay requires only original model rows at its signed epoch")
                return TensorFrame(value, value.rows.rows)

            return prepare_replay
        if self.attachment is not None and (self.attachment.activation_bindings or self.overlay_import):

            def prepare_replacement(value: TensorStepInput, batch: TensorBatch) -> TensorFrame:
                if value.rows.plan_epoch != batch.epoch or value.hidden is not None or value.operands:
                    raise ValueError("import input differs from its signed sources or epoch")
                update = value.auxiliary
                data = TensorFrame(value, value.rows.rows)
                if update is None:
                    return data
                if not isinstance(update, StateTensorUpdate):
                    raise ValueError("state imports require original state-field tensors")
                if not update.values:
                    return data
                selection = batch.selection
                if selection.keys == (None,):
                    if update.keys:
                        raise ValueError("instance replacement does not accept request keys")
                    data.replacements = ((0, update.values),)
                else:
                    if (
                        not update.keys
                        or len(set(update.keys)) != len(update.keys)
                        or any(
                            key.fields != (RequestKeyField.PROMPT_IDX,)
                            or key.request_id
                            or key.sample_idx != -1
                            or key.prompt_idx < 0
                            for key in update.keys
                        )
                        or any(tensor.ndim < 1 or tensor.shape[0] != len(update.keys) for tensor in update.values)
                    ):
                        raise ValueError(
                            "request replacement requires distinct original prompt keys and aligned field rows"
                        )
                    indices = {key: index for index, key in enumerate(update.keys)}
                    selected: dict[int, int] = {}
                    for row, index in zip(selection.rows, selection.state_indices, strict=True):
                        if (key := RequestStateKey.project(row.identity, (RequestKeyField.PROMPT_IDX,))) in indices:
                            if selection.refs[index].version_id != 0:
                                continue
                            # Several packed rows of one request share its state; two
                            # different import values for one root would be ambiguous.
                            if index in selected:
                                if selected[index] != indices[key]:
                                    raise ValueError(
                                        "one import batch cannot replace a shared request state with two values"
                                    )
                                continue
                            selected[index] = indices[key]
                    data.replacements = tuple(
                        (index, tuple(tensor[row] for tensor in update.values))
                        for index, row in sorted(selected.items())
                    )
                for index, values in data.replacements:
                    root = batch.roots[index]
                    if len(values) != len(root) or any(
                        source.shape != target.shape or source.dtype != target.dtype or source.device != target.device
                        for source, target in zip(values, root, strict=True)
                    ):
                        raise ValueError("replacement tensors differ from the signed layout")
                return data

            return prepare_replacement
        if self.overlay_targets:
            shapes = dict(zip(self.overlay_targets, dimensions.target_shapes, strict=True))
            in_forward = self.graph.updates_in_forward

            def prepare_overlay(value: TensorStepInput, batch: TensorBatch) -> TensorFrame:
                expected = {
                    (row.identity, target)
                    for row in batch.selection.rows
                    if in_forward or row.phase in self.parameter_phases
                    for target in self.overlay_targets
                }
                actual = tuple((operand.identity, operand.target) for operand in value.operands)
                if in_forward:
                    targets = {operand.target for operand in value.operands}
                    expected = {(identity, target) for identity, target in expected if target in targets}
                if len(actual) != len(expected) or set(actual) != expected:
                    raise ValueError("captured operands must cover each selected request and target exactly once")
                if value.rows.plan_epoch != batch.epoch or value.auxiliary is not None or value.hidden is not None:
                    raise ValueError("overlay input differs from its signed sources or epoch")
                phases = {row.identity: row.phase for row in batch.selection.rows}
                for operand in value.operands:
                    inputs, outputs = shapes[operand.target]
                    if (
                        operand.phase != phases[operand.identity]
                        or operand.hidden.ndim != 2
                        or operand.hidden.shape[0] < 1
                        or operand.hidden.shape[1] != inputs
                        or operand.desired.shape != (operand.hidden.shape[0], outputs)
                        or operand.base_weight.shape != (outputs, inputs)
                        or any(
                            tensor.dtype != tensor_dtype(dimensions.dtype) or tensor.device != batch.roots[0][0].device
                            for tensor in (operand.hidden, operand.desired, operand.base_weight)
                        )
                    ):
                        raise ValueError("operand tensors differ from specialization")
                if self.probe is not None and any(
                    sum(operand.hidden.shape[0] for operand in value.operands if operand.target == target)
                    > self.probe.max_packed_rows
                    for target in self.overlay_targets
                ):
                    raise ValueError("overlay operands exceed the signed packed-row capacity")
                return TensorFrame(value, value.rows.request_rows)

            return prepare_overlay
        expected = {
            (str(source.layer), "index", source.site.value)
            for task in self.graph.tasks
            for port in task.inputs
            if isinstance(source := port.source, HiddenInputSource)
        }

        def rows(value: TensorStepInput) -> tuple[PackedPhaseRow, ...]:
            return value.rows.decode_rows

        if self.attachment is not None and self.attachment.sampler_stage is SamplerStage.PACKED_PRE_FILTER_ADD:

            def rows(value: TensorStepInput) -> tuple[PackedPhaseRow, ...]:
                return tuple(
                    value.rows.rows[link.packed_row_index]
                    for link in value.sampler_rows.links
                    if value.rows.rows[link.packed_row_index].phase is HiddenPhase.DECODE
                )

        row_selection = rows
        if self.sample_limit is not None:
            limit = self.sample_limit
            select_rows = rows

            def row_selection(value: TensorStepInput) -> tuple[PackedPhaseRow, ...]:
                selected = select_rows(value)
                if any(row.identity.sample_idx >= limit for row in selected):
                    raise ValueError("sample index is outside configured iterations")
                return selected

        def prepare(value: TensorStepInput, batch: TensorBatch) -> TensorFrame:
            if value.auxiliary is not None and not self.weighted_training or value.operands:
                raise ValueError("ordinary hidden functions do not consume external state updates")
            if value.rows.plan_epoch != batch.epoch:
                raise ValueError("tensor input differs from the signed epoch")
            snapshots = value.hidden_batch(batch.stream).snapshots
            if self.probe is not None and tuple(snapshot.resource for snapshot in snapshots) != tuple(
                resource.binding for resource in self.probe.hidden_resources
            ):
                raise ValueError("sampler hidden input differs from its signed bindings")
            if {snapshot.resource.locator_key for snapshot in snapshots} != expected:
                raise ValueError("hidden input differs from its declared sources")
            if any(
                (self.probe is not None or batch.roots)
                and snapshot.hidden.shape[1] != dimensions.hidden_size
                or batch.roots
                and (
                    snapshot.hidden.device != batch.roots[0][0].device
                    or self.weighted_training
                    and snapshot.hidden.dtype != batch.roots[0][0].dtype
                )
                for snapshot in snapshots
            ):
                raise ValueError("sampler hidden input differs from parameter specialization")
            return TensorFrame(value, row_selection(value))

        return prepare

    @staticmethod
    def result(data: TensorFrame, invocation: InvocationOutput) -> TensorResult:
        return TensorResult(
            invocation.value, invocation, tuple(data.entries), data.input_gradients, data.parameter_gradient_fields
        )
