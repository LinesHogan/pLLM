"""Version-read observations for the CPU schedule checker, independent of retained-pin selection."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from pllm.contracts.host_batch import StateImport
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.semantic_graph import (
    StateEffectKind,
    StateSelectionInputSource,
    TaskResultInputSource,
    TaskSymbol,
)
from pllm.contracts.state_ref import LineageId, StateRef
from pllm.execution.state_execution import StateExecution
from pllm.execution.step_input import StepInput
from pllm.execution.tensor_binding import TensorOutput
from pllm.execution.tensor_frame import TensorFrame
from pllm.execution.tensor_task import TensorTask


@dataclass
class VersionChecks:
    """Compare actual method reads with an independent history of step-entry publications."""

    history: list[dict[LineageId, StateRef]] = field(default_factory=lambda: [])
    expected: dict[tuple[TaskSymbol, str, str], int] = field(default_factory=lambda: {})
    lagged: frozenset[TaskSymbol] = frozenset()
    failures: list[str] = field(default_factory=lambda: [])
    values: dict[tuple[str, int, str], frozenset[tuple[str, int]]] = field(default_factory=lambda: {})

    def begin(self, execution: StateExecution[StepInput, TensorFrame, TensorOutput], value: StepInput) -> None:
        """Derive the permitted version before the executor selects its retained reader pages."""
        selection = execution.directory.select(value.rows.request_rows)
        entry = {ref.lineage_id: ref for ref in selection.refs}
        self.expected = {}
        visibility = execution.arena.plan.timeline.visibility
        lag = 0 if visibility is None else visibility.lag
        self.lagged = frozenset(
            execution.arena.plan.semantic_graph.tasks[task.task_id].symbol
            for task in execution.plan.tasks
            if task.read_origin == "boundary" and lag
        )
        graph = execution.arena.plan.semantic_graph
        imported_state = graph.host_state(StateImport.schema) if isinstance(value.batch, StateImport) else None
        for task in execution.plan.tasks:
            symbol = execution.arena.plan.semantic_graph.tasks[task.task_id].symbol
            for group in selection.groups:
                for row, index in zip(selection.rows, group.state_indices, strict=True):
                    ref = selection.refs[int(index)]
                    if task.read_origin == "boundary" and lag:
                        # The oldest input of the last L engine steps bounds visibility.
                        for before in self.history[-lag:]:
                            if ref.lineage_id in before:
                                ref = before[ref.lineage_id]
                                break
                    produced = any(
                        effect.resource == group.resource
                        and effect.kind is StateEffectKind.READ
                        and effect.version is not None
                        for effect in graph.tasks[task.task_id].effects
                    )
                    imported = (
                        produced
                        and imported_state is not None
                        and group.resource == imported_state.resource
                        and ref.version_id == 0
                    )
                    self.expected[symbol, str(group.resource), row.identity.request_id] = ref.version_id + int(imported)
        self.history.append(entry)

    def record(self, task: TensorTask, data: TensorFrame, rows: Sequence[PackedPhaseRow]) -> None:
        """Inspect the version column used by the method, rather than trusting the planned distance."""
        if data.fixed is None:
            return
        for index, row in enumerate(rows):
            versions: set[tuple[str, int]] = set()
            point = row.identity.request_id, row.token_offset
            for source in task.sources.values():
                if isinstance(source, StateSelectionInputSource):
                    resource = str(source.resource)
                    version = int(data.fixed.for_resource(source.resource).index.versions[index])
                    if task.symbol in self.lagged:
                        versions.add((resource, version))
                    expected = self.expected.get((task.symbol, resource, row.identity.request_id))
                    if expected is not None and version != expected:
                        self.failures.append(
                            f"[publish_lag / visibility] `{task.symbol}` request {row.identity.request_id!r}, "
                            f"state {resource!r} at token {row.token_offset} read version {version}; "
                            f"the declared timing permits version {expected}. "
                            "Report this message as a framework state-visibility bug."
                        )
                if isinstance(source, TaskResultInputSource):
                    versions.update(self.values.get((*point, f"method {source.task}.{source.output}"), ()))
            names = [f"method {task.symbol}.{name}" for name in task.outputs]
            if task.sampler_stage is not None:
                names.extend((f"sampler {task.symbol}", f"sampler weight {task.symbol}"))
            if task.export is not None:
                names.append(f"export {task.export.name}")
            for name in names:
                self.values[*point, name] = frozenset(versions)
