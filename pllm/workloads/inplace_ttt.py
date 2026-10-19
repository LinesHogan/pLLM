"""Causal chunk mathematics over request-owned, per-target buffers."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, TypeAlias

import torch

import pllm
from pllm import ConditionalLayerReplay, ParameterTarget, TensorUpdate
from pllm import authoring as a
from pllm.contracts.region_program import Block, Call, Program, Region, SizeParameter, Value
from pllm.contracts.region_selection import RegionSelectionPolicy
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_streams import RegionStreamBindings
from pllm.execution.shared_tensor import tensor_dtype
from pllm.execution.tensor_binding import TensorBinding, TensorOutput
from pllm.execution.tensor_frame import TensorFrame, Tensors
from pllm.execution.tensor_scan import TensorScan
from pllm.kernels.page_copy import prepend_rows

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
        if (
            not self.targets
            or tuple(sorted(set(self.targets))) != self.targets
            or self.chunk_size < 1
            or self.learning_rate <= 0.0
        ):
            raise ValueError("In-Place-TTT mathematical configuration is invalid")


@dataclass(frozen=True)
class InPlaceTTTWorkload(pllm.Workload):
    name = "inplace_ttt.dense_fast_weight"
    config: InPlaceTTTConfig

    def declaration(self, facts: a.BackendCompileFacts | None = None) -> TensorBinding:
        binding = super().declaration(facts)
        if facts is not None and type(self).transition is _TRANSITION:
            binding.lower_overlay("transition", chunk_transition(self.config, facts))
        return binding

    def operations(
        self, facts: a.BackendCompileFacts, plan: a.PlanBundle, b: TensorBinding
    ) -> a.MathBinding[a.TensorStepInput, TensorFrame, TensorOutput]:
        if type(self).transition is not _TRANSITION:
            return super().operations(facts, plan, b)
        transition = chunk_transition(self.config, facts)
        regions = transition.regions
        sources = {
            (cells[0].input_signatures[0].shape, cells[0].input_signatures[1].stride): cells for _, cells in regions
        }
        transition.scans = {
            layout: tune_chunk_scan(scan, sources[layout], plan.timeline.region_policy)
            for layout, scan in transition.scans.items()
        }
        if any(scan.parallel for scan in transition.scans.values()):
            transition.streams = (torch.cuda.Stream(device=facts.device),)
        b.lower_overlay("transition", transition)
        return super().operations(facts, plan, b)

    @property
    def dense_fast_weights(self) -> pllm.RequestState:
        tail = max(1, self.config.chunk_size - 1)
        return pllm.RequestState(
            {
                "delta": pllm.Buffer(lambda inputs, outputs: (outputs, inputs)),
                "tail_input": pllm.Buffer(lambda inputs, outputs: (tail, inputs)),
                "tail_target": pllm.Buffer(lambda inputs, outputs: (tail, outputs)),
                "tail_length": pllm.Buffer((), dtype=torch.int64),
            },
            namespace="canonical.dense_fast_weight.state@1",
            keys=(pllm.RequestKeyField.REQUEST_ID, pllm.RequestKeyField.PROMPT_IDX, pllm.RequestKeyField.SAMPLE_IDX),
        )

    @pllm.on(pllm.Phase.MODEL_UPDATE)
    @pllm.overlay(
        state="dense_fast_weights",
        targets=pllm.Config("config.targets"),
        weight="delta",
        prefill=pllm.Config("config.update_prefill"),
        decode=pllm.Config("config.update_decode"),
        replay=pllm.Config("config.replay"),
    )
    def transition(
        self,
        hidden: Input,
        target: Target,
        base: Weight,
        delta: Delta,
        tail_input: TailInput,
        tail_target: TailTarget,
        tail_length: TailLength,
    ) -> pllm.TensorUpdate:
        chunk, rate = self.config.chunk_size, self.config.learning_rate

        def serial(
            x: torch.Tensor, y: torch.Tensor, base: torch.Tensor, delta: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            outputs: list[torch.Tensor] = []
            for start in range(0, len(x), chunk):
                inputs, targets = x[start : start + chunk], y[start : start + chunk]
                outputs.append(inputs @ (base + delta).T)
                delta.add_((rate * (targets.T @ inputs)).to(delta))
            return tuple(outputs)

        return chunk_update(hidden, target, base, delta, tail_input, tail_target, tail_length, chunk, serial)


__all__ = ["InPlaceTTTConfig", "InPlaceTTTWorkload"]

_TRANSITION = InPlaceTTTWorkload.transition


ChunkRegion: TypeAlias = tuple[Program, tuple[ExportedCell, ...]]


@lru_cache(maxsize=128)
def chunk_region(
    rate: float, chunk: int, inputs: int, outputs: int, dtype: torch.dtype, device: torch.device, transposed: bool
) -> ChunkRegion:
    """Export the chunk recurrence with its original evaluation order and layouts."""

    # Preserve each ATen evaluation path and expose its actual data flow:
    # forward reads the old delta; the increment depends only on this chunk's
    # data; addition creates the version used by the next iteration.
    def forward(delta: torch.Tensor, base: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (torch.mm(x, (base + delta).permute(1, 0)),)

    def increment(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (rate * torch.mm(y.permute(1, 0), x),)

    def update(delta: torch.Tensor, increment: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (delta + increment,)

    base = torch.zeros((inputs, outputs) if transposed else (outputs, inputs), dtype=dtype, device=device)
    examples = (
        torch.zeros((outputs, inputs), dtype=dtype, device=device),
        base.T if transposed else base,
        torch.zeros((chunk, inputs), dtype=dtype, device=device),
        torch.zeros((chunk, outputs), dtype=dtype, device=device),
    )
    delta, base, x, y = examples
    prediction = ExportedCell.capture(forward, ("delta", "base", "input"), ("output",), (delta, base, x))
    change = ExportedCell.capture(increment, ("input", "target"), ("increment",), (x, y))
    addition = ExportedCell.capture(update, ("delta", "increment"), ("next_delta",), (delta, torch.zeros_like(delta)))
    input_types = (*prediction.declaration.inputs, change.declaration.inputs[1])
    arguments = tuple(
        Value(name, type_) for name, type_ in zip(("delta", "base", "input", "target"), input_types, strict=True)
    )

    def call(cell: ExportedCell, args: tuple[str, ...], output: str) -> Call:
        return Call(args, (Value(output, cell.declaration.outputs[0]),), cell.digest)

    program = Program(
        Region(
            Block(
                arguments,
                (
                    call(prediction, ("delta", "base", "input"), "output"),
                    call(change, ("input", "target"), "increment"),
                    call(addition, ("delta", "increment"), "next_delta"),
                ),
                ("next_delta", "output"),
            )
        ),
        (prediction.declaration, change.declaration, addition.declaration),
    )
    return program, (prediction, change, addition)


def tune_chunk_scan(scan: TensorScan, cells: tuple[ExportedCell, ...], policy: RegionSelectionPolicy) -> TensorScan:
    """Tune the workload's scan within its source region envelope."""
    if scan.program.parameters[0].maximum == 0:
        return scan
    specs = (*cells[0].input_signatures, cells[1].input_signatures[1])
    if policy.tuning == "heuristic":
        return scan.prepared(options=policy, parallel=specs[0].device.type == "cuda")
    # Zero-valued cold examples use the actual guarded layouts. No request
    # state, initialization RNG or request-owned page is touched by tuning.
    examples = tuple(
        torch.empty_strided(spec.shape, spec.stride, dtype=spec.dtype, device=spec.device).zero_() for spec in specs
    )
    delta, base, x, y = examples
    count = min(3, scan.program.parameters[0].maximum)
    return scan.optimized((delta,), (base,), ((x,) * count, (y,) * count), options=policy, max_candidates=8)[0]


class ChunkTransition:
    """Import the recurrence; adapt only tails and private-buffer writeback here."""

    def __init__(self, chunk: int, regions: tuple[ChunkRegion, ...], max_rows: int) -> None:
        self.chunk, self.regions = chunk, regions
        self.streams: tuple[torch.cuda.Stream, ...] = ()
        # The bank length and recurrence share one invocation shape parameter.
        # Only tensor layout affects cold specialization, never the chunk count.
        unique = {
            (cells[0].input_signatures[0].shape, cells[0].input_signatures[1].stride): (program, cells)
            for program, cells in regions
        }
        count = SizeParameter("chunks", 0, (max_rows + chunk - 1) // chunk)
        self.scans = {
            layout: TensorScan(program, count, carry=1, invariants=1, functions={cell.digest: cell for cell in cells})
            for layout, (program, cells) in unique.items()
        }

    def __call__(self, values: Tensors) -> TensorUpdate:
        return chunk_update(
            values["input"],
            values["target"],
            values["weight"],
            values["delta"],
            values["tail_input"],
            values["tail_target"],
            values["tail_length"],
            self.chunk,
            self.run,
        )

    def run(
        self, inputs: torch.Tensor, targets: torch.Tensor, base: torch.Tensor, delta: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        private_base = base.detach().clone()
        scan = self.scans[tuple(delta.shape), tuple(private_base.stride())]
        streams = RegionStreamBindings(torch.cuda.current_stream(delta.device), self.streams) if scan.parallel else None
        columns = tuple(
            tuple(value[start : start + self.chunk] for start in range(0, len(inputs), self.chunk))
            for value in (inputs, targets)
        )
        result = scan((delta,), (private_base,), columns, streams=streams)
        delta.copy_(result.carry[0])
        return result.columns[0]


def chunk_transition(config: InPlaceTTTConfig, facts: a.BackendCompileFacts) -> ChunkTransition:
    """Specialize the recurrence for each guarded target layout before execution."""
    regions = tuple(
        chunk_region(
            config.learning_rate,
            config.chunk_size,
            target.input_size,
            target.output_size,
            tensor_dtype(facts.probe.tensor_dtype),
            facts.device,
            transposed,
        )
        for target in facts.targets
        for transposed in (False, True)
    )
    return ChunkTransition(config.chunk_size, regions, facts.probe.max_packed_rows)


def chunk_update(
    hidden: torch.Tensor,
    target: torch.Tensor,
    base: torch.Tensor,
    delta: torch.Tensor,
    tail_input: torch.Tensor,
    tail_target: torch.Tensor,
    tail_length: torch.Tensor,
    chunk: int,
    run: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, ...]],
) -> TensorUpdate:
    """The shared tail transition around either serial math or the imported scan."""
    inputs, targets = prepend_rows([tail_input, tail_target], [hidden, target], tail_length, chunk - 1)
    trained = len(inputs) // chunk * chunk
    outputs = list(run(inputs[:trained], targets[:trained], base, delta)) if trained else []
    remaining = len(inputs) - trained
    tail_input.zero_()
    tail_target.zero_()
    tail_input[:remaining].copy_(inputs[trained:])
    tail_target[:remaining].copy_(targets[trained:])
    tail_length.fill_(remaining)
    if remaining:
        outputs.append(inputs[trained:] @ (base + delta).T)
    return TensorUpdate(
        torch.cat(outputs)[-len(hidden) :], trained > 0, trained // chunk, trained, {"tail_rows": remaining}
    )
