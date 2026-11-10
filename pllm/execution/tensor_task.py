"""One declared tensor method, executed over the rows of one task invocation.

A TensorTask reads the method's inputs for the current rows, calls it, and stores the result. A
training task also differentiates its loss with respect to the selected states' parameters and
hands the per-state gradients to the optimizer. Export and sampler settings adapt the stored result
after execution. TensorDeclaration builds one per declared method.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

import torch

from pllm import _core
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import FieldView
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerStage
from pllm.contracts.semantic_graph import (
    MathContract,
    SemanticInputSource,
    StateResource,
    StateSelectionInputSource,
    TaskResultInputSource,
    TaskSymbol,
)
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.declaration import Call, State
from pllm.execution.module_graph import ModuleCall
from pllm.execution.ptts.request_state import RequestStateSelection
from pllm.execution.tensor_frame import (
    LeafRoots,
    TensorFrame,
    TensorMath,
    TensorPort,
    field_values,
    gradient_rows,
    take_rows,
)
from pllm.execution.tensor_program import Observation, TaskOutput, TensorBatch
from pllm.execution.trace import MAX_SNAPSHOT_BYTES


def _row_ref(selection: RequestStateSelection, identity: RequestIdentity) -> StateRef | None:
    """The state version a row's guidance came from; a prediction that reads no state has none."""
    if not selection.groups:
        return None
    return selection.refs[0] if selection.keys == (None,) else selection.ref_for(identity)


@dataclass(frozen=True)
class FieldWriter:
    """Write a task result into its declared field; used by the write lowering."""

    source: Call | None
    fields: tuple[FieldView, ...]
    resource: StateResource
    phase: HiddenPhase | None

    def values(
        self, data: TensorFrame, batch: TensorBatch, selected: tuple[int, ...], *, validate: bool
    ) -> tuple[torch.Tensor, ...]:
        """Stage normalized imports or convert a method's results to their declared storage precision."""
        if self.source is None:
            return tuple(
                torch.stack(tuple(fields[i] for _, fields in data.replacements)) for i in range(len(self.fields))
            )
        values = tuple(batch.results[self.source.symbol, name] for name in self.source.outputs)
        if batch.selection.for_resource(self.resource).instance:
            values = tuple(value.unsqueeze(0) for value in values)
        targets = tuple(
            view.read(batch.roots[selected[0]][view.field]).unsqueeze(0).expand(len(selected), *view.shape)
            for view in self.fields
        )
        return field_values(values, targets, resource=self.resource, validate=validate)

    def select(self, data: TensorFrame, selection: RequestStateSelection) -> tuple[int, ...]:
        """Select only this producer's eligible resource rows."""
        if self.source is None:
            return tuple(index for index, _ in data.replacements)
        group = selection.for_resource(self.resource)
        rows = data.producer_rows(self.phase)
        if self.phase is not None and not rows:
            return ()
        if group.instance:
            return tuple(selection.refs.positions(group.refs).tolist())
        indices = _core.select_states(
            selection.rows.identity_indices,
            group.state_indices,
            rows.identity_indices,
        )
        return tuple(indices.tolist())


@dataclass(frozen=True, slots=True)
class TensorExport:
    """Bounded observation settings carried by an exported TensorTask.

    A step delivers at most max_rows rows, split into events of at most MAX_SNAPSHOT_BYTES so the trace's
    bounded host buffers hold a step of every-token rows too.
    """

    name: str
    every: int
    max_rows: int
    max_cols: int
    enabled: bool
    overflow: Literal["drop", "block"]

    def __post_init__(self) -> None:
        if min(self.every, self.max_rows, self.max_cols) < 1 or self.max_cols * 8 > MAX_SNAPSHOT_BYTES:
            raise ValueError(
                f"tensor export needs positive every, max_rows and max_cols, with "
                f"max_cols * 8 <= {MAX_SNAPSHOT_BYTES} bytes (32 KiB), so one row fits an export event"
            )

    def prepare(self, data: TensorFrame) -> TensorFrame | None:
        """Skip disabled steps and bound the rows before evaluating tensor math."""
        if not self.enabled or data.step.rows.step_id % self.every or not data.rows:
            return None
        return replace(data, rows=data.rows[: self.max_rows], plan=None)

    def output(self, value: torch.Tensor, data: TensorFrame, source_rows: int) -> TaskOutput:
        """Publish the bounded result with each row's request and token position, in events of whole rows."""
        value = value[: self.max_rows, : self.max_cols]
        identities, positions = data.rows.identities, tuple(data.rows.token_offsets.tolist())
        rows = MAX_SNAPSHOT_BYTES // value[0].nbytes
        return TaskOutput(
            observations=tuple(
                Observation(
                    self.name,
                    value[start : start + rows],
                    step_id=data.step.rows.step_id,
                    identities=identities[start : start + rows],
                    positions=positions[start : start + rows],
                    export=self.overflow,
                    dropped_rows=source_rows - len(data.rows) if start == 0 else 0,
                )
                for start in range(0, len(value), rows)
            )
        )


@dataclass(frozen=True, slots=True)
class TensorTask:
    """One row-wise method with its input, gradient and output policies."""

    symbol: TaskSymbol
    function: TensorMath
    ports: Mapping[str, TensorPort]
    sources: Mapping[str, SemanticInputSource]
    policy: MathContract
    outputs: tuple[str, ...]
    # Memory fields written from inputs at publication; only training tasks have them.
    carries: Mapping[str, LayoutFieldId]
    phase: HiddenPhase | None = None
    resource: StateResource = StateResource("")
    writer: FieldWriter | None = None
    # A write returns a tuple even when it declares only one field.
    tuple_output: bool = False
    export: TensorExport | None = None
    sampler_stage: SamplerStage | None = None
    sampler_beta: float = 0.0
    sampler_order: int = 0
    training: State | None = None
    # CPU contract diagnostics attach after shape tracing; ordinary execution keeps this unset.
    observe: Callable[[TensorFrame, Mapping[str, torch.Tensor], tuple[torch.Tensor, ...]], None] | None = None

    def sampler_output(self, data: TensorFrame, batch: TensorBatch, weights: torch.Tensor | None) -> TaskOutput:
        """Adapt a computed prediction to sampler rows without executing its mathematics."""
        result = batch.results[self.symbol, self.outputs[0]]
        stage = self.sampler_stage
        assert stage is not None
        local = {row.packed_row_index: index for index, row in enumerate(data.rows)}
        links = tuple(
            (sampler, packed)
            for sampler, packed in enumerate(data.step.sampler_rows.packed_indices.tolist())
            if packed in local
        )
        if stage is SamplerStage.PACKED_POST_FILTER_EXACT:
            positions = tuple(local[packed] for _, packed in links)
            result = take_rows(result, positions)
            # Unselected rows keep weight 0, so the sampler leaves their logits unchanged;
            # selecting rows on the host would need a device-to-host read.
            if weights is not None:
                weights = take_rows(weights, positions).to(result.dtype)
        observations = (
            tuple(
                Observation(
                    "guidance",
                    result[index],
                    row,
                    _row_ref(batch.selection, row.identity),
                    data.step.rows.step_id,
                )
                for index, (_, packed) in enumerate(links)
                for row in (data.step.rows.inspect_row(packed),)
            )
            if batch.observing
            else ()
        )
        return TaskOutput(
            guidance=PackedSamplerGuidanceBatch(
                data.step.sampler_rows,
                result.detach(),
                tuple(sampler for sampler, _ in links),
                self.sampler_beta,
                stage=stage,
                row_weights=weights,
                member=str(self.symbol),
                order=self.sampler_order,
            ),
            observations=observations,
        )

    def dependencies(self, tasks: Mapping[TaskSymbol, TensorTask]) -> tuple[TensorTask, ...]:
        """This task's tensor DAG in dependency order, ending with the task itself."""
        ordered: dict[TaskSymbol, TensorTask] = {}
        for source in self.sources.values():
            match source:
                case TaskResultInputSource(task=upstream):
                    ordered.update((task.symbol, task) for task in tasks[upstream].dependencies(tasks))
                case _:
                    pass
        return (*ordered.values(), self)

    @property
    def domains(self) -> tuple[str, ...]:
        """Inputs that are boolean row masks (row selector, gradient rows, update states)."""
        policy = self.policy
        keys = (policy.forward_rows, policy.input_gradient_rows, policy.parameter_gradient_rows, policy.update_rows)
        return tuple(
            key
            for key in dict.fromkeys(keys)
            if key in self.sources and type(self.sources[key]) is not StateSelectionInputSource
        )

    @property
    def differentiable(self) -> bool:
        return bool(self.policy.input_gradients or self.policy.parameter_gradients)

    def read(
        self, data: TensorFrame, batch: TensorBatch
    ) -> tuple[dict[str, torch.Tensor], dict[tuple[str, str], torch.Tensor]]:
        """The function's inputs over every row of the fixed table, and the inputs it differentiates.

        Row selectors stay boolean masks on the device: the caller masks results and losses, so no
        selection is read back to the host.
        """
        policy = self.policy
        if policy.parameter_gradients:
            leaf = data.table.leaf
            assert leaf is not None
            batch = replace(batch, roots=LeafRoots(leaf, batch.roots))
        domains = self.domains
        inputs = {key: self.ports[key].read(data, batch) for key in domains}
        if any(inputs[key].dtype != torch.bool or inputs[key].shape != (len(data.rows),) for key in domains):
            raise ValueError("row domain must be a boolean mask over the operation rows")
        sources: dict[tuple[str, str], torch.Tensor] = {}
        for key in self.ports:
            inputs[key] = self._input(key, inputs, data, batch, sources)
        return inputs, sources

    def _input(
        self,
        key: str,
        inputs: dict[str, torch.Tensor],
        data: TensorFrame,
        batch: TensorBatch,
        sources: dict[tuple[str, str], torch.Tensor],
    ) -> torch.Tensor:
        """Read one input and attach the gradient behaviour its declaration asks for."""
        policy, port, source = self.policy, self.ports[key], self.sources[key]
        match source:
            case StateSelectionInputSource(resource=state_resource):
                differentiable = key not in policy.detached_inputs and state_resource in policy.parameter_gradients
                leaf = False
                domain = policy.parameter_gradient_rows
            case TaskResultInputSource():
                differentiable = key not in policy.detached_inputs and key in policy.input_gradients
                leaf = False
                domain = policy.input_gradient_rows
            case _:
                differentiable = key not in policy.detached_inputs and key in policy.input_gradients
                leaf = True
                domain = policy.input_gradient_rows
        try:
            value = inputs[key] if key in self.domains else port.read(data, batch)
        except (RuntimeError, ValueError, LookupError) as error:
            if self.observe is not None:
                error.add_note(
                    f"pLLM input routing failed for `{self.symbol}.{key}` before the method ran; "
                    "report this as a framework bug."
                )
            raise
        if not differentiable:
            value = value.detach()
        elif leaf:
            assert value.is_floating_point()
            value = value.detach().requires_grad_(True)
            sources[str(self.symbol), key] = value
        if differentiable and domain and value.requires_grad:
            assert port.rowwise and value.ndim > 0 and value.shape[0] == len(data.rows)
            value = gradient_rows(value, inputs[domain].reshape(len(data.rows), *([1] * (value.ndim - 1))))
        return value

    def arguments(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor | ModuleCall]:
        """Adapt method arguments after tensor selection and gradient policy, omitting row selectors."""
        domains = self.domains
        return {key: port.adapt(inputs[key]) for key, port in self.ports.items() if key not in domains}

    def observation_inputs(self, inputs: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """Diagnostics retain inputs before authored in-place math can change them."""
        return inputs if self.observe is None else {name: value.detach().clone() for name, value in inputs.items()}

    def call(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run authored math, checking borrowed input versions during CPU contract checks."""
        if self.observe is None:
            return self.function(**self.arguments(inputs))
        versions = {name: value._version for name, value in inputs.items()}
        try:
            return self.function(**self.arguments(inputs))
        finally:
            for name, value in inputs.items():
                if value._version != versions[name]:
                    raise ValueError(
                        f"method `{self.symbol}` mutates borrowed input `{name}` in place; inputs and parameters "
                        "are read-only views. Use out-of-place tensor math or clone() before writing; "
                        "publish state changes through @pllm.write or @pllm.train."
                    )

    def evaluate(self, data: TensorFrame, batch: TensorBatch) -> dict[tuple[str, str], torch.Tensor]:
        """Run a forward method on every row and store its result for later tasks and the invocation output.

        Rows outside `rows=` get a zero result. Returns the inputs a training task differentiates through it.
        """
        with torch.set_grad_enabled(self.differentiable):
            inputs, sources = self.read(data, batch)
            observed_inputs = self.observation_inputs(inputs)
            results = self.call(inputs)
            if not self.tuple_output:
                if not isinstance(results, torch.Tensor):
                    raise ValueError("tensor methods must return one tensor")
                results = (results,)
            if type(results) is not tuple or len(results) != len(self.outputs):
                raise ValueError("write must return a tuple matching its declared fields")
            for name, result in zip(self.outputs, results, strict=True):
                for borrowed in batch.borrowed_roots:
                    borrowed.check_output(result)
                if self.policy.forward_rows:
                    result = self.selected_rows(result, inputs[self.policy.forward_rows])
                batch.results[self.symbol, name] = result
            if self.observe is not None:
                self.observe(data, observed_inputs, tuple(batch.results[self.symbol, name] for name in self.outputs))
        return sources

    @staticmethod
    def selected_rows(result: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Give unselected rows a zero result, on the device."""
        if result.ndim == 0 or result.shape[0] != len(mask):
            raise ValueError("selected forward functions must preserve their row axis")
        zero = torch.zeros((), dtype=result.dtype, device=result.device)
        return torch.where(mask.reshape(-1, *(1,) * (result.ndim - 1)), result, zero)

    def differentiate(
        self,
        inputs: dict[str, torch.Tensor],
        sources: Mapping[tuple[str, str], torch.Tensor],
        data: TensorFrame,
        rows: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Forward, loss reduction over the selected `rows` and one autograd step to the parameters and `sources`."""
        observed_inputs = self.observation_inputs(inputs)
        loss = self.call(inputs)
        if not isinstance(loss, torch.Tensor) or loss.shape != (len(data.rows),):
            raise ValueError("training methods return one loss per row; pLLM reduces the selected rows per state")
        if self.observe is not None:
            self.observe(data, observed_inputs, (loss,))
        loss, objective = self._loss(loss, data, rows)
        leaf = data.table.leaf
        assert leaf is not None
        targets = (leaf, *sources.values())
        derivatives = (
            torch.autograd.grad(objective, targets, allow_unused=True, materialize_grads=True)
            if objective.requires_grad
            else tuple(torch.zeros_like(target) for target in targets)
        )
        return loss, derivatives

    def _loss(self, loss: torch.Tensor, data: TensorFrame, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The reported loss and the scalar to differentiate.

        `torch.where`, not a product, drops unselected rows, so a NaN or Inf there cannot reach the sums.
        """
        loss = torch.where(rows, loss.to(torch.promote_types(loss.dtype, torch.float32)), 0)
        states = data.table.index.states
        size = data.table.states + 1
        totals = loss.new_zeros(size).index_add(0, states, loss)
        if self.policy.reduction == "mean":
            counts = loss.new_zeros(size).index_add(0, states, rows.to(loss.dtype))
            totals = totals / counts.clamp_min(1)
        return totals, totals.sum()


__all__ = ["TensorTask"]
