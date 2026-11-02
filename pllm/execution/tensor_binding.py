"""Tensor declarations bound once to the plan: model effects, row tasks and their execution binding.

TensorBinding holds declaration state and binds the declared tasks to the
specialized plan (native implementations, scans, fixed-shape steps, input preparation).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from math import prod
from typing import TYPE_CHECKING

import torch

from pllm.contracts.backend_attachment import ModelProbe
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.host_batch import StateImport
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch
from pllm.contracts.parameter_overlay import FieldView, Projection
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
from pllm.contracts.state_feed import FeedBinding
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import TensorFieldSpec
from pllm.execution.causal_batch import CausalBinder
from pllm.execution.declaration import Builder, Dimensions, native, operation_mode
from pllm.execution.module_state import copy_initial
from pllm.execution.overlay_batch import OverlayBatch
from pllm.execution.ptts.page_bundle import PagePublication
from pllm.execution.scan import ScanContext
from pllm.execution.step_capture import StepCapture
from pllm.execution.tensor_executor import FieldWriteExecutor, TensorExecutor, reads_rows
from pllm.execution.tensor_frame import (
    OverlayEntry,
    TensorFrame,
    flat_gradient,
)
from pllm.execution.tensor_program import (
    FunctionMath,
    InvocationOutput,
    Roots,
    TensorBatch,
    TensorOperation,
    native_body,
)
from pllm.execution.tensor_task import TensorTask

if TYPE_CHECKING:
    from pllm.execution.compile_facts import BackendCompileFacts
    from pllm.execution.step_input import StepInput


@dataclass(frozen=True, slots=True)
class StepResult:
    value: torch.Tensor | None
    invocation: InvocationOutput
    entries: tuple[OverlayEntry, ...] = ()
    input_gradients: Mapping[tuple[str, str], torch.Tensor] = field(
        default_factory=lambda: dict[tuple[str, str], torch.Tensor]()
    )
    parameter_gradient_fields: Mapping[StateRef, tuple[torch.Tensor, ...]] = field(
        default_factory=lambda: dict[StateRef, tuple[torch.Tensor, ...]]()
    )

    metrics: Mapping[str, torch.Tensor] = field(default_factory=lambda: dict[str, torch.Tensor]())
    # A host training step's state after its update, by field name: owned copies on the step's stream. Empty for
    # engine steps.
    parameters: Mapping[str, torch.Tensor] = field(default_factory=lambda: dict[str, torch.Tensor]())

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


TensorOutput = StepResult | torch.Tensor | None


class TensorBinding(Builder[TensorFrame]):
    """Own tensor declaration state and bind tasks; results remain invocation-local."""

    def __init__(
        self,
        name: str,
        digest: str,
        dimensions: Dimensions,
        *,
        host: bool = False,
    ) -> None:
        super().__init__(name, digest, dimensions)
        self.dimensions = dimensions
        self.host = host
        self.checkpoint = ""
        self.feeds: tuple[FeedBinding, ...] = ()
        self.tensor_tasks: dict[TaskSymbol, TensorTask] = {}
        self.initializers: dict[str, Callable[[tuple[torch.Tensor, ...], torch.Generator], None]] = {}
        # Module states: their flat initial values, built once at bind from the model dimensions and seed.
        self.initial_values: dict[str, Callable[[Dimensions, int], torch.Tensor]] = {}
        self.causal: CausalBinder | None = None
        self.probe: ModelProbe | None = None
        self.overlay_targets: tuple[Projection, ...] = ()
        self.parameter_phases: tuple[HiddenPhase, ...] = ()
        self.field_views: dict[tuple[str, str], FieldView] = {}
        self.imports: dict[str, tuple[FieldView, ...]] = {}
        self.position_states: tuple[tuple[str, float], ...] = ()

    def retained_fields(self, resource: str) -> dict[int, TensorFieldSpec]:
        """Layout fields whose initial values admission copies; zero-start fields are left alone."""
        layout = next(item.layout for item in self.layouts.layouts if item.resource == resource)
        zero = self.zero_start_fields.get(resource, frozenset())
        return {i: field for i, field in enumerate(layout.fields) if i not in zero}

    @property
    def template_bytes(self) -> int:
        """Device memory held by the retained initial values of every initialized state."""
        return sum(
            prod(field.shape) * field.dtype.itemsize
            for resource in {*self.initializers, *self.initial_values}
            for field in self.retained_fields(resource).values()
        )

    def initializer(
        self, resource: str, device: torch.device, seed: int
    ) -> Callable[[Roots, tuple[RequestStateKey | None, ...]], None]:
        """Copy one resource's shared initial tensors into each newly admitted root."""
        init = self.initializers.get(resource)
        if resource in self.initial_values:
            init = partial(copy_initial, self.initial_values[resource](self.dimensions, seed).to(device))
        if init is None:
            return lambda roots, keys: None
        layout = next(item.layout for item in self.layouts.layouts if item.resource == resource)
        tensors = tuple(torch.zeros(field.shape, dtype=field.dtype.torch, device=device) for field in layout.fields)
        # A resource's initial tensors are shared values, independent of admission history.
        init(tensors, torch.Generator(device=device).manual_seed(seed))
        values = {i: tensors[i] for i in self.retained_fields(resource)}

        def initialize(roots: Roots, keys: tuple[RequestStateKey | None, ...]) -> None:
            for root in roots:
                for index, value in values.items():
                    root[index].copy_(value)

        return initialize

    def select_capture(self, resources: tuple[TaskResourceFact, ...]) -> None:
        """Run each tensor task that was funded resident memory as a captured CUDA graph."""
        captured = {fact.task for fact in resources if fact.resident_bytes}
        self.graph = replace(
            self.graph,
            tasks=tuple(
                replace(
                    task,
                    implementations=(
                        native(f"python.cuda_graph:{self.graph.workload_id}:{task.symbol}@1", "compute/cuda_graph@1"),
                    ),
                )
                if task.symbol in self.tensor_tasks and task.symbol in captured
                else task
                for task in self.graph.tasks
            ),
        )

    def require_resident_side_tasks(
        self, plan: PlanBundle, operations: Mapping[TaskSymbol, TensorOperation[TensorFrame]]
    ) -> None:
        """Side-lane tensor tasks run concurrently with the model, so their inputs must be fixed-size."""
        if not plan.timeline.task_resources:
            return
        for task in plan.schedule.tasks:
            symbol = plan.semantic_graph.tasks[task.task_id].symbol
            if task.lane == "side" and symbol in self.tensor_tasks and not operations[symbol].resident(1):
                raise ValueError(
                    "side tensor tasks need fixed row inputs; declare row inputs or move the task before the model"
                )

    def step_row_reader(self) -> Callable[[StepInput], tuple[PackedPhaseRow, ...]]:
        """The rows that open a step's state context, chosen once at bind: a model effect reading request state
        selects every packed row's request; a custom forward over no request state selects none."""
        effects, request = self.effects, any(state.scope is StateScope.REQUEST for state in self.graph.states)
        if (
            effects.activation_bindings
            or self.sites.rewrites
            or (effects.custom_forward is not None and effects.overlay_bindings)
        ):
            return (lambda value: value.rows.rows) if request else (lambda value: ())
        if effects.custom_forward is not None and not request:
            return lambda value: ()
        read = self.row_reader()
        return lambda value: read(value.rows)

    def function_math(self, device: torch.device, seed: int) -> FunctionMath[StepInput, TensorFrame, TensorOutput]:
        """Bind state initialization, step rows and frame preparation for execution."""
        return FunctionMath(
            {str(state.resource): self.initializer(str(state.resource), device, seed) for state in self.graph.states},
            self.step_row_reader(),
            self.bind_prepare(),
            lambda data, invocation: self.result(data, invocation) if self.graph.states else invocation.value,
            lambda previous, incoming: replace(
                previous, step=incoming.step, rows=incoming.rows, plan=None, training=None
            ),
            parameter_phases=self.parameter_phases,
            model_sites=tuple(self.sites.rewrites),
        )

    @property
    def has_exports(self) -> bool:
        """Whether tensor tasks need a host observation callback."""
        return any(task.export is not None for task in self.tensor_tasks.values())

    def configure_scans(self, facts: BackendCompileFacts) -> None:
        """Scope model-route scans and give a chunked route overlay its eager batched executor; ordinary task
        scopes and executors are built in executables."""
        if self.overlay_math is not None and any(task.custom_forward is not None for task in self.graph.tasks):
            math = replace(
                self.overlay_math,
                function=ScanContext(
                    facts.timeline.tensor_policy,
                    self.custom_forward_rows,
                    facts.device,
                    replay="compute/cuda_graph@1" in facts.timeline.capabilities,
                ).bind(self.overlay_math.function),
            )
            self.overlay_math = math if math.chunk is None else replace(math, executor=OverlayBatch(math))

    def executables(
        self, facts: BackendCompileFacts | None, resources: tuple[TaskResourceFact, ...]
    ) -> dict[TaskSymbol, TensorOperation[TensorFrame]]:
        """Build each tensor operation once, with the same scan scope for eager and captured math."""
        capture = facts is not None and "compute/cuda_graph@1" in facts.timeline.capabilities
        limits = {fact.task: fact for fact in resources}
        operations = {
            symbol: self._operation(symbol, capture, limits)
            for symbol in self.operations
            if symbol not in self.tensor_tasks
        }
        operations.update(
            (symbol, self._tensor_operation(task, facts, limits if capture else {}))
            for symbol, task in self.tensor_tasks.items()
        )
        if facts is None:
            return operations
        scans = ScanContext(
            facts.timeline.tensor_policy,
            facts.probe.max_packed_rows,
            facts.device,
            None
            if facts.timeline.memory_budget_bytes is None
            else facts.timeline.memory_budget_bytes // max(1, facts.max_admitted_lineages),
            replay=capture,
        )
        return {symbol: replace(operation, body=scans.bind(operation.body)) for symbol, operation in operations.items()}

    def _operation(
        self, symbol: TaskSymbol, capture: bool, limits: Mapping[TaskSymbol, TaskResourceFact]
    ) -> TensorOperation[TensorFrame]:
        """A declared operation; a chunked overlay transition runs through its OverlayBatch."""
        operation, math = self.operations[symbol], self.overlay_math
        if math is None or math.chunk is None or symbol != math.task:
            return operation
        overlay = OverlayBatch(math, limits.get(symbol) if capture else None)
        return replace(
            operation, body=partial(native_body, replace(math, executor=overlay)), capture_resident=overlay.resident
        )

    def _tensor_operation(
        self, task: TensorTask, facts: BackendCompileFacts | None, limits: Mapping[TaskSymbol, TaskResourceFact]
    ) -> TensorOperation[TensorFrame]:
        """A tensor task's executor: a field writer or a body over masked tables, captured when `limits` funds it."""
        declaration = next(item for item in self.graph.tasks if item.symbol == task.symbol)
        state = task.training
        optimizer = None if state is None else state.page_update()
        step_capture = self._step_capture(task, facts, limits)
        executor = (
            TensorExecutor(
                task,
                self.tensor_tasks,
                optimizer,
                capture=step_capture,
                device=torch.device("cpu") if facts is None else facts.device,
                filter_rows=task.phase is not None and task.phase is not self.row_phase(),
            )
            if task.writer is None
            else FieldWriteExecutor(task.writer, step_capture)
        )
        return TensorOperation(
            declaration.implementations[0].implementation_id,
            executor,
            operation_mode(
                state,
                transition=any(
                    effect.kind in (StateEffectKind.TRANSITION, StateEffectKind.TRANSITION_MAYBE)
                    for effect in declaration.effects
                ),
            ),
            optimizer=optimizer,
            copy_fields=None if state is None else state.copy_fields,
            capture_resident=executor.resident,
            resource=StateResource("") if state is None else state.declaration.resource,
        )

    def _step_capture(
        self, task: TensorTask, facts: BackendCompileFacts | None, limits: Mapping[TaskSymbol, TaskResourceFact]
    ) -> StepCapture | None:
        """Capture a funded writer or row-reading task; exports run eagerly."""
        if facts is None or task.symbol not in limits or task.export is not None:
            return None
        if task.writer is None and not reads_rows(task, self.tensor_tasks):
            return None
        # A task without its own resource reads every state of the invocation.
        request = any(
            task.resource in ("", state.resource) and state.scope is StateScope.REQUEST for state in self.graph.states
        )
        return StepCapture(limits[task.symbol], facts.max_admitted_lineages if request else 1)

    def prepare_import(self, update: StateImport[torch.Tensor], batch: TensorBatch, data: TensorFrame) -> None:
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
                    key.fields != (RequestKeyField.PROMPT_INDEX,)
                    or key.request_id
                    or key.sample_index != -1
                    or key.prompt_index < 0
                    for key in update.keys
                )
                or any(tensor.ndim < 1 or tensor.shape[0] != len(update.keys) for tensor in update.values)
            ):
                raise ValueError("request replacement requires distinct original prompt keys and aligned field rows")
            indices = {key: index for index, key in enumerate(update.keys)}
            selected: dict[int, int] = {}
            for row, index in zip(selection.rows, group.state_indices, strict=True):
                if (key := RequestStateKey.project(row.identity, (RequestKeyField.PROMPT_INDEX,))) in indices:
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

    def row_phase(self) -> HiddenPhase | None:
        """The frame's preselected phase; None keeps every request's final row for model effects or prefill."""
        if (
            self.overlay_targets
            or self.effects.activation_bindings
            or any(
                isinstance(port.source, HiddenInputSource) and port.source.phase is HiddenPhase.PREFILL
                for task in self.graph.tasks
                for port in task.inputs
            )
        ):
            return None
        return HiddenPhase.DECODE

    def row_reader(self) -> Callable[[PackedPhaseRowBatch], tuple[PackedPhaseRow, ...]]:
        """Bind the frame's row selection once; tasks filter only when they need a narrower phase."""
        if self.row_phase() is None:
            return lambda rows: rows.request_rows
        return lambda rows: rows.latest_decode_rows

    def bind_prepare(self) -> Callable[[StepInput, TensorBatch], TensorFrame]:
        """Prepare independent input channels into the same invocation frame."""
        read = self.row_reader()
        # Additive logits read the rows the sampler draws from that complete a prompt or decode, in sampler order.
        sampled = self.effects.sampler is not None and self.effects.sampler.stage is SamplerStage.PACKED_PRE_FILTER_ADD

        def prepare(value: StepInput, batch: TensorBatch) -> TensorFrame:
            rows = read(value.rows)
            if sampled:
                rows = tuple(
                    row
                    for link in value.sampler_rows.links
                    for row in (value.rows.rows[link.packed_row_index],)
                    if row.phase is HiddenPhase.DECODE or row.token_offset == row.prompt_length - 1
                )
            data = TensorFrame(value, rows)
            match value.batch:
                case StateImport() as update:
                    if not self.imports:
                        raise ValueError("declare imported=True to consume StateImport")
                    self.prepare_import(update, batch, data)
                case _:
                    pass
            return data

        return prepare

    @staticmethod
    def result(data: TensorFrame, invocation: InvocationOutput) -> StepResult:
        return StepResult(
            invocation.value if data.training is None else data.training.output.detach(),
            invocation,
            tuple(data.entries),
            data.input_gradients,
            data.parameter_gradient_fields,
            {} if data.training is None else data.training.metrics,
            data.parameters,
        )
