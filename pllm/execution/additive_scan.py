"""Bind Rust-proven additive scan candidates inside the shared region executor."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from pllm import _core
from pllm.contracts.additive_scan import AdditiveScanEvidence, AdditiveScanPlan
from pllm.contracts.region_program import Program
from pllm.contracts.region_selection import RegionCandidate, RegionPayloadLimit
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.region_compilation import analyze_region_program
from pllm.execution.region_native import OrderedScanPrimitive, RegionFunction
from pllm.execution.region_storage import analyze_region_payload, import_region_functions
from pllm.execution.region_values import Bank, Data
from pllm.kernels.ordered_scan import ordered_weights

if TYPE_CHECKING:
    from pllm.execution.region_optimization import PreparedRegionCandidate


def _tensors(bank: Bank) -> tuple[torch.Tensor, ...]:
    tensors: list[torch.Tensor] = []
    for value in bank.elements:
        assert isinstance(value, torch.Tensor)
        tensors.append(value)
    return tuple(tensors)


def _pack(rows: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Borrow consecutive scan slices; pack separately allocated bank elements."""
    first = rows[0]
    stride = first.shape[0] * first.stride(0)
    if all(
        row.untyped_storage() == first.untyped_storage()
        and row.stride() == first.stride()
        and row.storage_offset() == first.storage_offset() + i * stride
        for i, row in enumerate(rows)
    ):
        return first.as_strided((len(rows), *first.shape), (stride, *first.stride()))
    return torch.stack(rows)


@dataclass(frozen=True, slots=True)
class OrderedUpdate:
    """Run the verified window size without retaining invocation state."""

    scale: float
    window: int

    def __call__(self, inputs: tuple[Data, ...]) -> tuple[Data, ...]:
        carry, base, xs, ys = inputs
        assert isinstance(carry, torch.Tensor) and isinstance(base, torch.Tensor)
        assert isinstance(xs, Bank) and isinstance(ys, Bank)
        x, y = _tensors(xs), _tensors(ys)
        state = carry
        outputs: list[torch.Tensor] = []
        for start in range(0, len(x), self.window):
            xx = _pack(x[start : start + self.window])
            yy = _pack(y[start : start + self.window])
            increments = torch.bmm(yy.transpose(1, 2), xx)
            weights, state = ordered_weights(increments, state, base, self.scale)
            result = torch.bmm(xx, weights.transpose(1, 2))
            outputs.extend(row.clone() for row in result.unbind())
            # Release the window before allocating its successor. The Rust
            # scratch proof deliberately funds one window, not two.
            del xx, yy, increments, weights, result
        return state.clone(), Bank(tuple(outputs))


def candidates(
    program: Program, cells: Mapping[str, RegionFunction], limits: tuple[RegionPayloadLimit, ...]
) -> Iterator["PreparedRegionCandidate"]:
    """Offer budget-fitting windows; the normal selector chooses their execution."""
    from pllm.execution.region_optimization import PreparedRegionCandidate

    # Enumerate only the port frontier this kernel can implement. The Rust
    # derivation remains the authority for every source operation and shape.
    if program.resources or len(program.entry.block.arguments) != 4 or len(program.entry.block.operations) != 1:
        return
    imports = import_region_functions(program, cells)
    # These are backend workspace facts, not a claim about currently free VRAM.
    workspace = torch.backends.cuda.cublas_workspace_size() + torch.backends.cuda.cublaslt_workspace_size()
    for window in (2, 4, 8, 16, 32, 64):
        evidence = AdditiveScanEvidence(window, workspace)
        encoded = _core.derive_additive_scan(
            encode_boundary(program), encode_boundary(imports), encode_boundary(evidence)
        )
        if encoded == "null":
            return
        plan = decode_boundary(encoded, AdditiveScanPlan)
        primitive = OrderedScanPrimitive(plan.implementation, OrderedUpdate(plan.scale, plan.window))
        functions: dict[str, RegionFunction] = {**cells, primitive.declaration.name: primitive}
        schedule = analyze_region_program(plan.program, optimize=False).reference_schedule
        payload = analyze_region_payload(plan.program, schedule, functions)
        if any(
            not any(limit.device == d.device and limit.bytes >= d.peak_bytes_upper_bound for limit in limits)
            for d in payload.devices
        ):
            # This rule's window allocations grow monotonically; larger
            # windows cannot restore feasibility under the same envelope.
            return
        candidate = RegionCandidate(
            schedule, import_region_functions(plan.program, functions), (), additive_scan=evidence
        )
        yield PreparedRegionCandidate(plan.program, candidate, (), functions)


def payload_budget(
    program: Program,
    cells: Mapping[str, RegionFunction],
    reference: tuple[RegionPayloadLimit, ...],
    ceiling: int | None,
) -> tuple[RegionPayloadLimit, ...]:
    """Derive a funded window envelope before task memory tracing and admission."""
    available = tuple(
        RegionPayloadLimit(limit.device, ceiling if ceiling is not None else 1 << 60) for limit in reference
    )
    budget = reference
    for candidate in candidates(program, cells, available):
        evidence = candidate.candidate.additive_scan
        assert evidence is not None
        if ceiling is None and evidence.window > 8:
            break
        payload = analyze_region_payload(candidate.program, candidate.candidate.schedule_map, candidate.cells)
        budget = tuple(
            RegionPayloadLimit(
                d.device, max(d.peak_bytes_upper_bound, next(r.bytes for r in reference if r.device == d.device))
            )
            for d in payload.devices
        )
    return budget
