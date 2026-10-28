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
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch
from pllm.contracts.request_identity import RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerStage
from pllm.contracts.semantic_graph import (
    HiddenInputSource,
    StateEffectKind,
    StateResource,
    StateScope,
    TaskResourceFact,
    TaskSymbol,
)
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.ptts.page_bundle import PagePublication
from pllm.execution.scan import ScanContext
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
    TensorOperation,
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

    @property
    def has_exports(self) -> bool:
        """Whether tensor tasks need a host observation callback."""
        return any(task.export is not None for task in self.tensor_tasks.values())

    def configure_scans(self, facts: BackendCompileFacts) -> None:
        """Scope model-route scans; ordinary task scopes are built in executables."""
        if self.overlay_math is not None and any(task.replay is not None for task in self.graph.tasks):
            self.overlay_math = replace(
                self.overlay_math,
                function=ScanContext(
                    facts.timeline.tensor_policy,
                    self.route_rows,
                    facts.device,
                    replay="compute/cuda_graph@1" in facts.timeline.capabilities,
                ).bind(self.overlay_math.function),
            )

    def executables(
        self, facts: BackendCompileFacts | None, resources: tuple[TaskResourceFact, ...]
    ) -> dict[TaskSymbol, TensorOperation[TensorFrame]]:
        """Build each tensor operation once, with the same scan scope for eager and captured math."""
        capture = facts is not None and "compute/cuda_graph@1" in facts.timeline.capabilities
        fixed = {
            symbol: task
            for symbol, task in self.tensor_tasks.items()
            if (task.writer is not None or reads_rows(task, self.tensor_tasks)) and task.export is None
        }
        if not resources or any(
            task.policy.update_states in task.domains or (task.training is not None and task.policy.forward_rows)
            for task in fixed.values()
        ):
            fixed.clear()
        limits = {fact.task: fact for fact in resources}
        operations: dict[TaskSymbol, TensorOperation[TensorFrame]] = {}
        declarations = {task.symbol: task for task in self.graph.tasks}
        for symbol in (*self.operations, *self.tensor_tasks):
            declaration = declarations[symbol]
            scans = (
                None
                if facts is None
                else ScanContext(
                    facts.timeline.tensor_policy,
                    facts.probe.max_packed_rows,
                    facts.device,
                    None
                    if facts.timeline.memory_budget_bytes is None
                    else facts.timeline.memory_budget_bytes // max(1, facts.max_admitted_lineages),
                    replay=capture,
                )
            )
            if symbol not in self.tensor_tasks:
                operation = self.operations[symbol]
                operations[symbol] = operation if scans is None else replace(operation, body=scans.bind(operation.body))
                continue
            task = self.tensor_tasks[symbol]
            state = task.training
            optimizer = None
            if state is not None:
                assert state.optimizer is not None
                optimizer = PageUpdate(state.optimizer, state.optimizer_fields // 5)
            executor = TensorExecutor(
                task,
                self.tensor_tasks,
                optimizer,
                compact=symbol not in fixed and task.writer is None,
                capture=StepCapture(
                    limits[symbol],
                    facts.max_admitted_lineages
                    if any(
                        state.resource == task.resource and state.scope is StateScope.REQUEST
                        for state in self.graph.states
                    )
                    else 1,
                )
                if capture and symbol in fixed and facts is not None
                else None,
                device=torch.device("cpu") if facts is None else facts.device,
            )
            operations[symbol] = TensorOperation(
                declaration.implementations[0].implementation_id,
                executor if scans is None else scans.bind(executor),
                "read"
                if state is None
                else "gradient"
                if any(
                    effect.kind in (StateEffectKind.TRANSITION, StateEffectKind.TRANSITION_MAYBE)
                    for effect in declaration.effects
                )
                else "update",
                optimizer=optimizer,
                copy_fields=None
                if state is None
                else tuple(
                    LayoutFieldId(i) for i in range(len(state.fields)) if i >= state.optimizer_fields or i % 5 != 1
                ),
                capture_resident=executor.resident,
            )
        return operations

    def prepare_import(self, update: StateTensorUpdate[torch.Tensor], batch: TensorBatch, data: TensorFrame) -> None:
        """Validate imported fields and map them onto resource-specific private roots."""
        if not update.values:
            return
        (resource,) = self.imports
        selection = batch.selection
        group = selection.for_resource(StateResource(resource))
        if group.keys == (None,):
            if update.keys:
                raise ValueError("instance replacement does not accept request keys")
            data.replacements = ((selection.refs.index(group.refs[0]), update.values),)
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
                raise ValueError("request replacement requires distinct original prompt keys and aligned field rows")
            indices = {key: index for index, key in enumerate(update.keys)}
            selected: dict[int, int] = {}
            for row, index in zip(selection.rows, group.state_indices, strict=True):
                if (key := RequestStateKey.project(row.identity, (RequestKeyField.PROMPT_IDX,))) in indices:
                    if selection.refs[index].version_id != 0:
                        continue
                    # Several packed rows of one request share its state; two
                    # different import values for one root would be ambiguous.
                    if index in selected:
                        if selected[index] != indices[key]:
                            raise ValueError("one import batch cannot replace a shared request state with two values")
                        continue
                    selected[index] = indices[key]
            data.replacements = tuple(
                (index, tuple(tensor[row] for tensor in update.values)) for index, row in sorted(selected.items())
            )
        for index, values in data.replacements:
            root = tuple(view.read(batch.roots[index][view.field]) for view in self.imports[resource])
            if len(values) != len(root) or any(
                source.shape != target.shape or source.dtype != target.dtype or source.device != target.device
                for source, target in zip(values, root, strict=True)
            ):
                raise ValueError("replacement tensors must match the declared layout shape, dtype and device")

    def read_rows(self, rows: PackedPhaseRowBatch) -> tuple[PackedPhaseRow, ...]:
        """Rows the tensor tasks read: every request's final row for overlays, activation patches and prefill
        inputs, otherwise each request's latest decode row."""
        return (
            rows.request_rows
            if self.overlay_targets
            or (self.attachment is not None and self.attachment.activation_bindings)
            or any(
                isinstance(port.source, HiddenInputSource) and port.source.phase is HiddenPhase.PREFILL
                for task in self.graph.tasks
                for port in task.inputs
            )
            else rows.latest_decode_rows
        )

    def bind_prepare(self) -> Callable[[TensorStepInput, TensorBatch], TensorFrame]:
        """Prepare independent input channels into the same invocation frame."""

        def prepare(value: TensorStepInput, batch: TensorBatch) -> TensorFrame:
            rows = self.read_rows(value.rows)
            if (
                self.attachment is not None
                and self.attachment.guides_sampler
                and self.attachment.sampler_stage is SamplerStage.PACKED_PRE_FILTER_ADD
            ):
                rows = tuple(
                    value.rows.rows[link.packed_row_index]
                    for link in value.sampler_rows.links
                    if value.rows.rows[link.packed_row_index].phase is HiddenPhase.DECODE
                )
            data = TensorFrame(value, rows)
            match value.auxiliary:
                case StateTensorUpdate() as update:
                    if not self.imports:
                        raise ValueError("declare imported=True to consume StateTensorUpdate")
                    self.prepare_import(update, batch, data)
                case _:
                    pass
            return data

        return prepare

    @staticmethod
    def result(data: TensorFrame, invocation: InvocationOutput) -> TensorResult:
        return TensorResult(
            invocation.value, invocation, tuple(data.entries), data.input_gradients, data.parameter_gradient_fields
        )
