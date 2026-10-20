"""Bind Rust-proven additive scan candidates inside the shared region executor."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from pllm import _core
from pllm.contracts.additive_scan import AdditiveScanEvidence, AdditiveScanPlan
from pllm.contracts.region_imports import ScanForm
from pllm.contracts.region_program import Program
from pllm.contracts.region_selection import RegionCandidate, RegionPayloadLimit
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.region_compilation import analyze_region_program
from pllm.execution.region_native import AdditiveScanPrimitive, RegionFunction
from pllm.execution.region_storage import analyze_region_payload, import_region_functions
from pllm.execution.region_values import Bank, Data
from pllm.kernels.additive_scan import AdditiveBuffers, launch_window

if TYPE_CHECKING:
    from pllm.execution.region_optimization import PreparedRegionCandidate


def _tensors(bank: Bank) -> tuple[torch.Tensor, ...]:
    tensors: list[torch.Tensor] = []
    for value in bank.elements:
        assert isinstance(value, torch.Tensor)
        tensors.append(value)
    return tuple(tensors)


def _pack(rows: tuple[torch.Tensor, ...], window: int, chunk: int, width: int, like: torch.Tensor) -> torch.Tensor:
    """Keep a fixed window shape; inactive slices are additive identities."""
    if len(rows) == window:
        first = rows[0]
        stride = first.shape[0] * first.stride(0)
        if all(
            row.untyped_storage() == first.untyped_storage()
            and row.stride() == first.stride()
            and row.storage_offset() == first.storage_offset() + i * stride
            for i, row in enumerate(rows)
        ):
            return first.as_strided((window, *first.shape), (stride, *first.stride()))
        return torch.stack(rows)
    packed = like.new_zeros((window, chunk, width))
    if rows:
        torch.stack(rows, out=packed[: len(rows)])
    return packed


@dataclass(frozen=True, slots=True)
class AdditiveUpdate:
    """Run the verified window size without retaining invocation state."""

    plan: AdditiveScanPlan

    def __call__(self, inputs: tuple[Data, ...]) -> tuple[Data, ...]:
        carry, base, xs, ys = inputs
        assert isinstance(carry, torch.Tensor) and isinstance(base, torch.Tensor)
        assert isinstance(xs, Bank) and isinstance(ys, Bank)
        x, y = _tensors(xs), _tensors(ys)
        state, plan = carry, self.plan
        outputs: list[torch.Tensor] = []
        buffers = AdditiveBuffers(carry, plan.window, plan.chunk)
        for start in range(0, plan.capacity, plan.window):
            size = min(max(0, len(x) - start), plan.window)
            xx = _pack(x[start : start + plan.window], plan.window, plan.chunk, carry.shape[1], carry)
            yy = _pack(y[start : start + plan.window], plan.window, plan.chunk, carry.shape[0], carry)
            active = torch.full((), size, dtype=torch.int32, device=carry.device)
            launch_window(xx, yy, state, base, active, buffers, plan.scale, plan.implementation.form, plan.offsets)
            state = buffers.final
            outputs.extend(row.clone() for row in buffers.output.unbind()[:size])
            # Outputs are independent leaves; the next window may reuse every
            # scratch buffer on the same stream without retaining an old bank.
            del xx, yy, active

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
    for window in (1, 2, 4, 8, 16, 32, 64):
        for form in (ScanForm.ORDERED, ScanForm.HILLIS_STEELE):
            offsets = (
                tuple(1 << level for level in range(window.bit_length() - 1)) if form is ScanForm.HILLIS_STEELE else ()
            )
            evidence = AdditiveScanEvidence(window, workspace, form, offsets)
            encoded = _core.derive_additive_scan(
                encode_boundary(program), encode_boundary(imports), encode_boundary(evidence)
            )
            if encoded == "null":
                return
            plan = decode_boundary(encoded, AdditiveScanPlan)
            primitive = AdditiveScanPrimitive(plan.implementation, AdditiveUpdate(plan))
            functions: dict[str, RegionFunction] = {**cells, primitive.declaration.name: primitive}
            schedule = analyze_region_program(plan.program, optimize=False).reference_schedule
            payload = analyze_region_payload(plan.program, schedule, functions)
            if any(
                not any(limit.device == d.device and limit.bytes >= d.peak_bytes_upper_bound for limit in limits)
                for d in payload.devices
            ):
                return  # Both forms have the same bound; larger windows only add scratch.
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
    budget = (
        reference if ceiling is None else tuple(RegionPayloadLimit(r.device, min(r.bytes, ceiling)) for r in reference)
    )
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


def preference(candidate: "PreparedRegionCandidate") -> tuple[int, int]:
    """Amortize GEMM launches; use tree depth only for wide prefix windows.

    Measurements put small-window ordered and tree kernels within noise;
    retain the smaller rounding change there. Measured selection overrides
    this cold ordering with completed execution costs at the real C.
    """
    evidence = candidate.candidate.additive_scan
    assert evidence is not None
    return evidence.window, int((evidence.form is ScanForm.HILLIS_STEELE) == (evidence.window >= 16))
