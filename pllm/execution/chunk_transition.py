"""Native chunk scan lowering and memory tracing used by the in-place tensor workload."""

from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache
from typing import TypeAlias

import torch

from pllm import authoring as a
from pllm.contracts.region_program import Block, Call, Program, Region, SizeParameter, Value
from pllm.contracts.region_selection import RegionPayloadLimit, RegionSelectionPolicy
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.memory import MemoryUsage, measure
from pllm.execution.region_compilation import analyze_region_program
from pllm.execution.region_storage import analyze_region_payload
from pllm.execution.region_streams import RegionStreamBindings
from pllm.execution.shared_tensor import tensor_dtype
from pllm.execution.tensor_frame import Tensors, TensorUpdate
from pllm.execution.tensor_scan import TensorScan

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
    arguments = tuple(
        Value(name, type_)
        for name, type_ in zip(
            ("delta", "base", "input", "target"),
            (*prediction.declaration.inputs, change.declaration.inputs[1]),
            strict=True,
        )
    )
    program = Program(
        Region(
            Block(
                arguments,
                (
                    Call(
                        ("delta", "base", "input"),
                        (Value("output", prediction.declaration.outputs[0]),),
                        prediction.digest,
                    ),
                    Call(("input", "target"), (Value("increment", change.declaration.outputs[0]),), change.digest),
                    Call(
                        ("delta", "increment"), (Value("next_delta", addition.declaration.outputs[0]),), addition.digest
                    ),
                ),
                ("next_delta", "output"),
            )
        ),
        (prediction.declaration, change.declaration, addition.declaration),
    )
    return program, (prediction, change, addition)


def tune_chunk_scan(
    scan: TensorScan, cells: tuple[ExportedCell, ...], rows: int, workspace: int, policy: RegionSelectionPolicy
) -> TensorScan:
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
    outside = chunk_buffers(cells, rows)
    if workspace <= outside:
        raise ValueError("transition workspace leaves no capacity for its tensor region")
    limits = (RegionPayloadLimit(str(specs[0].device), workspace - outside),)
    if policy.tuning == "heuristic":
        return scan.prepared(payload_limits=limits, options=policy, parallel=specs[0].device.type == "cuda")
    # Zero-valued cold examples use the actual guarded layouts. No request
    # state, initialization RNG or request-owned page is touched by tuning.
    examples = tuple(
        torch.empty_strided(spec.shape, spec.stride, dtype=spec.dtype, device=spec.device).zero_() for spec in specs
    )
    delta, base, x, y = examples
    count = min(3, scan.count.maximum)
    return scan.optimized(
        (delta,), (base,), ((x,) * count, (y,) * count), payload_limits=limits, options=policy, max_candidates=8
    )[0]


class ChunkTransition:
    """Import the recurrence; adapt only tails and private-buffer writeback here."""

    def __init__(self, chunk: int, regions: tuple[ChunkRegion, ...], max_rows: int) -> None:
        self.chunk = chunk
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
            scan = self.scans[key]
            streams = None
            if scan.parallel:
                streams = RegionStreamBindings(torch.cuda.current_stream(delta.device), self.streams)
            result = scan(
                (delta,),
                (private_base,),
                (
                    tuple(inputs[start : start + chunk] for start in range(0, trained, chunk)),
                    tuple(targets[start : start + chunk] for start in range(0, trained, chunk)),
                ),
                streams=streams,
            )
            (next_delta,) = result.carry
            (column,) = result.columns
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
        output = torch.cat(outputs, dim=0)[-hidden.shape[0] :]
        return TensorUpdate(
            output,
            changed=trained > 0,
            update_count=trained // chunk,
            trained_rows=trained,
            metrics={"tail_rows": remaining},
        )


def chunk_memory(
    rate: float,
    chunk: int,
    source: Callable[[Tensors], TensorUpdate],
    lowered: bool,
    facts: a.BackendCompileFacts,
) -> tuple[MemoryUsage, tuple[tuple[int, ...], ...]]:
    """Trace the selected scan, including retained chunk outputs and maximum tails."""
    rows = facts.probe.max_packed_rows + facts.max_admitted_lineages * (chunk - 1)
    dtype = tensor_dtype(facts.probe.tensor_dtype)
    regions = (
        tuple(
            chunk_region(
                rate,
                chunk,
                target.input_size,
                target.output_size,
                dtype,
                torch.device("meta"),
                transposed,
            )
            for target in facts.targets
            for transposed in (False, True)
        )
        if lowered
        else ()
    )
    function = ChunkTransition(chunk, regions, rows) if lowered else source
    examples: list[dict[str, torch.Tensor]] = []
    for target in facts.targets:
        inputs, outputs = target.input_size, target.output_size
        for transposed in (False, True):
            values = {
                "input": torch.empty((rows, inputs), device="meta", dtype=dtype),
                "target": torch.empty((rows, outputs), device="meta", dtype=dtype),
                "weight": torch.empty(
                    (inputs, outputs) if transposed else (outputs, inputs), device="meta", dtype=dtype
                ),
                "delta": torch.empty((outputs, inputs), device="meta", dtype=dtype),
                "tail_input": torch.empty((max(1, chunk - 1), inputs), device="meta", dtype=dtype),
                "tail_target": torch.empty((max(1, chunk - 1), outputs), device="meta", dtype=dtype),
                "tail_length": torch.tensor(0),
            }
            if transposed:
                values["weight"] = values["weight"].T
            examples.append(values)
    # Source targets run serially, retaining their outputs until the invocation returns.
    usage = measure(lambda: tuple(function(values).output for values in examples))
    if isinstance(function, ChunkTransition):
        # Tuning admits against the region's proof, which includes recovery lifetimes.
        # Fund that proof as well as the measured eager execution and outer buffers.
        bounds: list[int] = []
        for _, cells in regions:
            scan = function.scans[cells[0].input_signatures[0].shape, cells[0].input_signatures[1].stride]
            schedule = analyze_region_program(scan.program, optimize=False).reference_schedule
            payload = analyze_region_payload(scan.program, schedule, {cell.digest: cell for cell in cells})
            bounds.append(chunk_buffers(cells, rows) + max(device.peak_bytes_upper_bound for device in payload.devices))
        usage = replace(usage, workspace=max(usage.workspace, max(bounds) + usage.output))
    return usage, ((rows, facts.probe.hidden_size),)


def chunk_buffers(cells: tuple[ExportedCell, ...], rows: int) -> int:
    """Trace concatenation, private base and the remainder outside the scanned region."""
    outputs, inputs = cells[0].input_signatures[0].shape
    dtype = cells[0].input_signatures[0].dtype
    x = torch.empty((rows, inputs), device="meta", dtype=dtype)
    y = torch.empty((rows, outputs), device="meta", dtype=dtype)
    base = torch.empty((outputs, inputs), device="meta", dtype=dtype)
    delta = torch.empty_like(base)

    def run() -> object:
        packed_x, packed_y = torch.cat((x[:0], x)), torch.cat((y[:0], y))
        private = base.clone()
        effective = base + delta
        remainder = packed_x @ effective.T
        return packed_x, packed_y, private, effective, remainder

    return measure(run).workspace
