"""Causal chunk mathematics over request-owned, per-target buffers."""

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Annotated, TypeAlias

import torch
import pllm
from pllm import TensorUpdate, ParameterTarget, ConditionalLayerReplay
from pllm import authoring as a
from pllm.contracts.semantic_graph import SchedulerTimeline, TaskResourceFact, TaskSymbol, ValueResourceFact
from pllm.contracts.tensor_layout import TensorDType
from pllm.contracts.region_program import Block, Call, Program, Region, SizeParameter, Value
from pllm.contracts.region_selection import RegionPayloadLimit, RegionSelectionPolicy
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.shared_tensor import tensor_dtype
from pllm.execution.tensor_binding import TensorBinding, TensorFrame, TensorOutput, Tensors
from pllm.execution.tensor_scan import TensorScan

Input = Annotated[torch.Tensor, pllm.Operand.INPUT]
Target = Annotated[torch.Tensor, pllm.Operand.TARGET]
Weight = Annotated[torch.Tensor, pllm.Operand.WEIGHT]
Delta = Annotated[torch.Tensor, pllm.Parameter("dense_fast_weights", "delta")]
TailInput = Annotated[torch.Tensor, pllm.Parameter("dense_fast_weights", "tail_input")]
TailTarget = Annotated[torch.Tensor, pllm.Parameter("dense_fast_weights", "tail_target")]
TailLength = Annotated[torch.Tensor, pllm.Parameter("dense_fast_weights", "tail_length")]


@dataclass(frozen=True, slots=True)
class InPlaceTTTConfig:
    targets: tuple[ParameterTarget, ...]
    chunk_size: int = 4096
    learning_rate: float = 1e-3
    update_prefill: bool = True
    update_decode: bool = False
    replay: ConditionalLayerReplay | None = None

    def __post_init__(self) -> None:
        if (not self.targets or tuple(sorted(set(self.targets))) != self.targets
            or self.chunk_size < 1 or self.learning_rate <= 0.0):
            raise ValueError("In-Place-TTT mathematical configuration is invalid")


@dataclass(frozen=True)
class InPlaceTTTWorkload(pllm.Workload):
    name = "inplace_ttt.dense_fast_weight"
    config: InPlaceTTTConfig

    def resource_facts(self, facts: a.BackendCompileFacts, options: pllm.ExecutionOptions) -> SchedulerTimeline:
        timeline = super().resource_facts(facts, options)
        if not timeline.task_resources and timeline.memory_budget_bytes is None and options.memory_budget_bytes is None:
            return timeline
        self.require_native_methods(_NATIVE, resource_declaration=_RESOURCES)
        if self.config.replay is not None:
            raise ValueError("finite in-place costs require a separate declaration for model replay scratch")
        rows = facts.probe.max_packed_rows + facts.max_admitted_lineages * (self.config.chunk_size - 1)
        width = 8 if facts.probe.tensor_dtype is TensorDType.FLOAT64 else 4
        # Boundary concatenation, immutable scan inputs/results, final cat and
        # functional cell temporaries (including a layout-preserving base copy).
        workspace = max(width * (rows * (2 * target.input_size + 4 * target.output_size)
            + 8 * target.input_size * target.output_size) for target in facts.targets)
        output = width * rows * sum(target.output_size for target in facts.targets)
        return replace(timeline,
            task_resources=(TaskResourceFact(TaskSymbol("publish"), 0), TaskResourceFact(TaskSymbol("transition"), workspace)),
            value_resources=(ValueResourceFact(TaskSymbol("publish"), "publication", 0),
                ValueResourceFact(TaskSymbol("transition"), "transition", output)))

    def operations(self, facts: a.BackendCompileFacts, plan: a.PlanBundle,
                   b: TensorBinding) -> a.MathBinding[a.TensorStepInput, TensorFrame, TensorOutput]:
        self.require_native_methods(_NATIVE)
        if plan.timeline.task_resources:
            self.require_resources(plan, self.minimum_resources(facts, plan))
        # Export is cold authoring work, before any request owns private pages.
        regions = tuple(_chunk_region(self.config.learning_rate, self.config.chunk_size,
            target.input_size, target.output_size, tensor_dtype(facts.probe.tensor_dtype), facts.device, transposed)
            for target in facts.targets for transposed in (False, True))
        transition = _ChunkTransition(self.config.chunk_size, regions, facts.probe.max_packed_rows)
        rows = facts.probe.max_packed_rows + facts.max_admitted_lineages * (self.config.chunk_size - 1)
        width = 8 if facts.probe.tensor_dtype is TensorDType.FLOAT64 else 4
        # Without a declared task budget, retain the same local envelope as
        # resource_facts. With one, candidates must fit the signed reservation.
        workspace = next((fact.workspace_bytes for fact in plan.timeline.task_resources if fact.task == "transition"),
            max(width * (rows * (2 * target.input_size + 4 * target.output_size)
                + 8 * target.input_size * target.output_size) for target in facts.targets))
        sources = {(cells[0].input_signatures[0].shape, cells[0].input_signatures[1].stride): cells
                   for _, cells in regions}
        transition.scans = {layout: _tune_chunk_scan(scan, sources[layout], rows, workspace, plan.timeline.region_policy)
                            for layout, scan in transition.scans.items()}
        b.lower_overlay("transition", transition)
        return super().operations(facts, plan, b)

    @property
    def dense_fast_weights(self) -> pllm.RequestState:
        tail = max(1, self.config.chunk_size - 1)
        return pllm.RequestState({
            "delta": pllm.Buffer(lambda inputs, outputs: (outputs, inputs)),
            "tail_input": pllm.Buffer(lambda inputs, outputs: (tail, inputs)),
            "tail_target": pllm.Buffer(lambda inputs, outputs: (tail, outputs)),
            "tail_length": pllm.Buffer((), dtype=torch.int64),
        }, namespace="canonical.dense_fast_weight.state@1", keys=(pllm.RequestKeyField.REQUEST_ID,
            pllm.RequestKeyField.PROMPT_IDX, pllm.RequestKeyField.SAMPLE_IDX))

    @pllm.on(pllm.Phase.MODEL_UPDATE)
    @pllm.overlay(state="dense_fast_weights", targets=pllm.Config("config.targets"), weight="delta",
        prefill=pllm.Config("config.update_prefill"), decode=pllm.Config("config.update_decode"),
        replay=pllm.Config("config.replay"))
    def transition(self, hidden: Input, target: Target, base: Weight, delta: Delta,
                   tail_input: TailInput, tail_target: TailTarget, tail_length: TailLength) -> pllm.TensorUpdate:
        prefix = int(tail_length.item())
        inputs = torch.cat((tail_input[:prefix], hidden), dim=0)
        targets = torch.cat((tail_target[:prefix], target), dim=0)
        chunk = self.config.chunk_size
        trained = (inputs.shape[0] // chunk) * chunk
        outputs: list[torch.Tensor] = []
        # Each chunk emits against its input version before updating the next chunk's weights.
        for start in range(0, trained, chunk):
            x, y = inputs[start:start + chunk], targets[start:start + chunk]
            outputs.append(x @ (base + delta).T)
            delta.add_((self.config.learning_rate * (y.T @ x)).to(delta))
        remaining = inputs.shape[0] - trained
        tail_input.zero_()
        tail_target.zero_()
        tail_input[:remaining].copy_(inputs[trained:])
        tail_target[:remaining].copy_(targets[trained:])
        tail_length.fill_(remaining)
        if remaining:
            outputs.append(inputs[trained:] @ (base + delta).T)
        output = torch.cat(outputs, dim=0)[-hidden.shape[0]:]
        return TensorUpdate(output, changed=trained > 0, update_count=trained // chunk,
            trained_rows=trained, metrics={"tail_rows": remaining})


ChunkRegion: TypeAlias = tuple[Program, tuple[ExportedCell, ...]]


@lru_cache(maxsize=128)
def _chunk_region(rate: float, chunk: int, inputs: int, outputs: int,
                  dtype: torch.dtype, device: torch.device, transposed: bool) -> ChunkRegion:
    # Preserve each ATen evaluation path and expose its actual data flow:
    # forward reads the old delta; the increment depends only on this chunk's
    # data; addition creates the version used by the next iteration.
    def forward(delta: torch.Tensor, base: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (x @ (base + delta).T,)
    def increment(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (rate * (y.T @ x),)
    def update(delta: torch.Tensor, increment: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (delta + increment,)
    base = torch.zeros((inputs, outputs) if transposed else (outputs, inputs), dtype=dtype, device=device)
    examples = (torch.zeros((outputs, inputs), dtype=dtype, device=device), base.T if transposed else base,
                torch.zeros((chunk, inputs), dtype=dtype, device=device),
                torch.zeros((chunk, outputs), dtype=dtype, device=device))
    delta, base, x, y = examples
    prediction = ExportedCell.capture(forward, ("delta", "base", "input"), ("output",), (delta, base, x))
    change = ExportedCell.capture(increment, ("input", "target"), ("increment",), (x, y))
    addition = ExportedCell.capture(update, ("delta", "increment"), ("next_delta",), (delta, torch.zeros_like(delta)))
    arguments = tuple(Value(name, type_) for name, type_ in zip(("delta", "base", "input", "target"),
        (*prediction.declaration.inputs, change.declaration.inputs[1]), strict=True))
    program = Program(Region(Block(arguments, (
        Call(("delta", "base", "input"), (Value("output", prediction.declaration.outputs[0]),), prediction.digest),
        Call(("input", "target"), (Value("increment", change.declaration.outputs[0]),), change.digest),
        Call(("delta", "increment"), (Value("next_delta", addition.declaration.outputs[0]),), addition.digest),
    ), ("next_delta", "output"))), (prediction.declaration, change.declaration, addition.declaration))
    return program, (prediction, change, addition)


def _tune_chunk_scan(scan: TensorScan, cells: tuple[ExportedCell, ...], rows: int,
                     workspace: int, policy: RegionSelectionPolicy) -> TensorScan:
    """Spend only the signed transition workspace left after its outer buffers.

    Inputs/targets are concatenated outside the scan. Reserve those buffers,
    remainder output, private base and the remainder's effective weight before
    admitting a candidate. The scan's retained carry/columns remain in its
    payload envelope. Final concatenation uses the separately declared output
    value reservation; pages retain their original owner reservations.
    """
    if not isinstance(scan.count, SizeParameter):
        raise ValueError("chunk planning requires the original bounded iteration domain")
    if scan.count.maximum == 0:
        return scan
    specs = (*cells[0].input_signatures, cells[1].input_signatures[1])
    outputs, inputs = specs[0].shape
    width = 8 if specs[0].dtype is torch.float64 else 4
    outside = width * (rows * (inputs + 2 * outputs) + 2 * inputs * outputs)
    if workspace <= outside:
        raise ValueError("transition workspace leaves no capacity for its tensor region")
    # Zero-valued cold examples use the actual guarded layouts. No request
    # state, initialization RNG or request-owned page is touched by tuning.
    examples = tuple(torch.empty_strided(spec.shape, spec.stride, dtype=spec.dtype, device=spec.device).zero_()
                     for spec in specs)
    delta, base, x, y = examples
    count = min(3, scan.count.maximum)
    return scan.optimized((delta,), (base,), ((x,) * count, (y,) * count),
        payload_limits=(RegionPayloadLimit(str(specs[0].device), workspace - outside),), options=policy,
        max_candidates=16)


class _ChunkTransition:
    """Import the recurrence; adapt only tails and private-buffer writeback here."""

    def __init__(self, chunk: int, regions: tuple[ChunkRegion, ...], max_rows: int) -> None:
        self.chunk = chunk
        # The bank length and recurrence share one invocation shape parameter.
        # Only tensor layout affects cold specialization, never the chunk count.
        unique = {(cells[0].input_signatures[0].shape, cells[0].input_signatures[1].stride): (program, cells)
                  for program, cells in regions}
        count = SizeParameter("chunks", 0, (max_rows + chunk - 1) // chunk)
        self.scans = {layout: TensorScan(program, count, carry=1, invariants=1,
            functions={cell.digest: cell for cell in cells}) for layout, (program, cells) in unique.items()}

    def __call__(self, values: Tensors) -> TensorUpdate:
        hidden, target, base = values["input"], values["target"], values["weight"]
        delta = values["delta"]
        tail_input, tail_target, tail_length = (values[key] for key in ("tail_input", "tail_target", "tail_length"))
        prefix = int(tail_length.item())
        inputs = torch.cat((tail_input[:prefix], hidden), dim=0)
        targets = torch.cat((tail_target[:prefix], target), dim=0)
        chunk = self.chunk
        trained = (inputs.shape[0] // chunk) * chunk
        outputs: list[torch.Tensor] = []
        if trained:
            # Preserve the layout used by base + delta: forcing row-major can
            # change GEMM rounding even without changing any math operators.
            private_base = base.detach().clone()
            key = (tuple(delta.shape), tuple(private_base.stride()))
            result = self.scans[key]((delta,), (private_base,), (
                tuple(inputs[start:start + chunk] for start in range(0, trained, chunk)),
                tuple(targets[start:start + chunk] for start in range(0, trained, chunk))))
            next_delta, = result.carry
            column, = result.columns
            outputs.extend(column)
            delta.copy_(next_delta)
        remaining = inputs.shape[0] - trained
        tail_input.zero_()
        tail_target.zero_()
        tail_input[:remaining].copy_(inputs[trained:])
        tail_target[:remaining].copy_(targets[trained:])
        tail_length.fill_(remaining)
        if remaining:
            outputs.append(inputs[trained:] @ (base + delta).T)
        output = torch.cat(outputs, dim=0)[-hidden.shape[0]:]
        return TensorUpdate(output, changed=trained > 0, update_count=trained // chunk,
            trained_rows=trained, metrics={"tail_rows": remaining})


__all__ = ["InPlaceTTTConfig", "InPlaceTTTWorkload"]

_NATIVE = pllm.Workload.native_methods({"transition": InPlaceTTTWorkload.transition})
_RESOURCES = InPlaceTTTWorkload.resource_facts
