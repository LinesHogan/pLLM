"""Source-derived capacity, measured selection, and selected-only rewrite binding."""

from dataclasses import replace
import warnings

import pytest
import torch

from pllm import _core
from pllm.config import ExecutionOptions
from pllm.contracts.region_selection import (
    RegionCandidate, RegionPayloadLimit, RegionSelection, RegionSelectionRequest, heuristic_candidate_cost,
)
from pllm.execution.codec import boundary_digest, decode_boundary, encode_boundary
from pllm.execution.exported_cell import ExportedCell, NumericalTransformWarning
from pllm.execution.region_compilation import analyze_region_program
from pllm.execution.region_optimization import (
    assess_region_candidates, bind_selected_region, optimize_region, select_region_candidate,
)
from pllm.execution.region_storage import import_region_functions
from pllm.execution.tensor_rewrites import matmul_reassociation_candidates
from test.boundary.test_region_native_capacity import single_call
from test.boundary.test_region_tensor import tensor_grid, inputs_for


def example(device="cpu", shapes=((4, 1), (1, 4), (4, 32))):
    tensors = tuple(torch.ones(shape, dtype=torch.float64, device=device) for shape in shapes)
    cell = ExportedCell.capture(lambda a, b, c: ((a @ b) @ c,), ("a", "b", "c"), ("out",), tensors)
    program = single_call(cell)
    variant, = matmul_reassociation_candidates(cell)
    schedule = analyze_region_program(program).reference_schedule
    original = import_region_functions(program, {cell.digest: cell})
    rewritten = import_region_functions(program, {cell.digest: variant.cell})
    request = RegionSelectionRequest(original, (RegionCandidate(schedule, original, (100, 101, 900)),
        RegionCandidate(schedule, rewritten, (50, 51, 600))), (RegionPayloadLimit(device, 1 << 20),), True,
        "deterministic timing oracle")
    inputs = dict(zip(("in0", "in1", "in2"), tensors, strict=True))
    return cell, program, variant, request, inputs


def test_latency_ranks_candidates_even_when_lower_flops_costs_more_storage_and_traffic():
    cell, program, variant, request, _ = example()
    selected = select_region_candidate(program, request)
    original, changed = selected.candidates
    assert variant.rewrite.after.flops < variant.rewrite.before.flops
    assert changed.capacity.devices[0].peak_tensor_bytes_upper_bound > original.capacity.devices[0].peak_tensor_bytes_upper_bound
    assert changed.capacity.devices[0].logical_read_bytes_upper_bound > original.capacity.devices[0].logical_read_bytes_upper_bound
    assert selected.selected == 1  # Runtime beats both bytes and FLOPs as objective.
    slower = replace(request, candidates=(request.candidates[0], replace(request.candidates[1], elapsed_ns=(200, 201, 202))))
    assert select_region_candidate(program, slower).selected == 0
    tied = replace(request, candidates=(request.candidates[0], replace(request.candidates[1], elapsed_ns=(101, 101, 101))))
    assert select_region_candidate(program, tied).selected == 0
    assert changed.changed_functions == (cell.digest,)


def test_payload_limits_and_numerical_optout_are_hard_constraints():
    _, program, _, request, _ = example()
    source_bytes = assess_region_candidates(program, request)[0].capacity.devices[0].peak_tensor_bytes_upper_bound
    limited = replace(request, payload_limits=(RegionPayloadLimit("cpu", source_bytes),))
    selected = select_region_candidate(program, limited)
    assert selected.selected == 0
    assert not selected.candidates[1].feasible
    strict = replace(request, allow_numerical_transforms=False)
    assert select_region_candidate(program, strict).selected == 0
    with pytest.raises(ValueError, match="no region candidate"):
        select_region_candidate(program, replace(request, payload_limits=(RegionPayloadLimit("cpu", 0),)))
    with pytest.raises(ValueError, match="unique device"):
        select_region_candidate(program, replace(request, payload_limits=request.payload_limits * 2))


def test_an_infeasible_reference_does_not_prevent_a_smaller_variant_from_being_selected():
    _, program, _, request, _ = example(shapes=((4, 3), (3, 16), (16, 2)))
    original, changed = assess_region_candidates(program, request)
    assert changed.capacity.devices[0].peak_tensor_bytes_upper_bound < original.capacity.devices[0].peak_tensor_bytes_upper_bound
    request = replace(request, payload_limits=(RegionPayloadLimit("cpu", changed.capacity.devices[0].peak_tensor_bytes_upper_bound),),
        candidates=(replace(request.candidates[0], elapsed_ns=()), request.candidates[1]))
    selected = select_region_candidate(program, request)
    assert selected.selected == 1
    assert selected.candidates[0].median_ns is None


def test_selection_requires_the_reference_as_candidate_zero():
    _, program, _, request, _ = example()
    with pytest.raises(ValueError, match="reference execution"):
        select_region_candidate(program, replace(request, candidates=request.candidates[::-1]))
    with pytest.raises(ValueError, match="execution measurements"):
        select_region_candidate(program, replace(request, candidates=tuple(replace(c, elapsed_ns=()) for c in request.candidates)))
    with pytest.raises(ValueError, match="positive measured"):
        select_region_candidate(program, replace(request, candidates=(replace(request.candidates[0], elapsed_ns=(0,)),)))


def test_heuristic_candidate_cost_uses_existing_facts_without_measurement():
    assert heuristic_candidate_cost(1000, 0, 1) == 1000.0
    assert heuristic_candidate_cost(8448 + 7936, 0, 2) == 16384.0
    assert heuristic_candidate_cost(1000, 200, 1, flop_byte_ratio=10) == 1020.0
    assert heuristic_candidate_cost(1000, 200, 4, flop_byte_ratio=10) == 1005.0
    assert heuristic_candidate_cost(0, 0, 1) == 0.0
    for args in ((-1, 0, 1), (0, -1, 1), (0, 0, 0)):
        with pytest.raises(ValueError, match="nonnegative facts"):
            heuristic_candidate_cost(*args)
    with pytest.raises(ValueError, match="nonnegative facts"):
        heuristic_candidate_cost(0, 0, 1, flop_byte_ratio=0)


def test_fast_measurement_cannot_authorize_an_illegal_loop_schedule():
    cell = ExportedCell.capture(lambda h, w: (h + w, w + 1), ("h", "w"), ("nh", "nw"),
        (torch.ones(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64)))
    program = tensor_grid(3, 2, cell)
    source = analyze_region_program(program)
    imports = import_region_functions(program, {cell.digest: cell})
    request = RegionSelectionRequest(imports, (
        RegionCandidate(source.reference_schedule, imports, (100,)),
        RegionCandidate("{ S0[c,l] -> [-c,-l] : 0 <= c < 3 and 0 <= l < 2 }", imports, (1,)),
    ), (RegionPayloadLimit("cpu", 100000),), True, "illegal schedule oracle")
    with pytest.raises(ValueError):
        select_region_candidate(program, request)


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_selected_rewrite_is_replayed_once_without_request_time_planning(monkeypatch, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cell, program, variant, request, inputs = example(device)
    request = replace(request, measurement_context=boundary_digest(device, "selected replay"))
    selected = select_region_candidate(program, request)
    with pytest.warns(NumericalTransformWarning, match="pLLM selected"):
        execution = bind_selected_region(program, {cell.digest: cell}, request, selected, (variant.rewrite,))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bind_selected_region(program, {cell.digest: cell}, request, selected, (variant.rewrite,))
    assert not any(isinstance(w.message, NumericalTransformWarning) for w in caught)
    def forbidden(*args):
        raise AssertionError("selected execution must not plan or select")
    for name in ("select_region_candidate", "lower_region_schedule", "lower_region_bindings",
                 "analyze_region_program", "analyze_region_capacity", "analyze_region_storage"):
        monkeypatch.setattr(_core, name, forbidden)
    for _ in range(2):
        result, = execution.run(inputs).values
        torch.testing.assert_close(result, inputs["in0"] @ (inputs["in1"] @ inputs["in2"]), rtol=0, atol=0)
        for value in inputs.values():
            assert torch.equal(value, torch.ones_like(value))


def test_binding_rejects_missing_or_forged_rewrite_evidence_and_local_optout():
    cell, program, variant, request, _ = example()
    selected = select_region_candidate(program, request)
    with pytest.raises(ValueError, match="replayed mathematical evidence"):
        bind_selected_region(program, {cell.digest: cell}, request, selected, ())
    with pytest.raises(ValueError, match="prerequisites"):
        bind_selected_region(program, {cell.digest: cell}, request, selected,
            (replace(variant.rewrite, dimensions=(1, 1, 1, 1)),))
    with pytest.raises(ValueError, match="local numerical policy"):
        bind_selected_region(program, {cell.digest: cell}, request, selected, (variant.rewrite,),
                             options=ExecutionOptions(allow_numerical_transforms=False))


@pytest.mark.parametrize("allow", (False, True))
def test_tuner_measures_interleaved_candidates_but_only_warns_for_a_selected_rewrite(monkeypatch, allow):
    import pllm.execution.region_optimization as tuning
    cell, program, _, _, inputs = example()
    calls = []
    def time_candidate(execution, inputs, devices, sizes, streams=None, pages=None):
        assert pages is None
        calls.append(execution)
        return 100  # Equal runtimes retain source; candidates must not warn.
    monkeypatch.setattr(tuning, "_measure", time_candidate)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = optimize_region(program, {cell.digest: cell}, inputs, payload_limits=(RegionPayloadLimit("cpu", 10000),),
            options=ExecutionOptions(allow_numerical_transforms=allow, schedule_policy="serial"),
            warmup_rounds=1, measurement_rounds=3)
    assert result.selection.selected == 0
    assert not result.rewrites
    assert not any(isinstance(w.message, NumericalTransformWarning) for w in caught)
    count = 2 if allow else 1
    assert len(calls) == 4 * count
    assert all(calls[i] is calls[i % count] for i in range(len(calls)))
    assert len(result.request.candidates) == count


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_real_cold_tuning_executes_source_isl_candidates_on_private_inputs(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cell = ExportedCell.capture(lambda h, w: (h + w, w + 1), ("h", "w"), ("nh", "nw"),
        (torch.ones(4, dtype=torch.float64, device=device), torch.ones(4, dtype=torch.float64, device=device)))
    program = tensor_grid(3, 2, cell)
    inputs = inputs_for(3, 2, device)
    saved = tuple(t.clone() for bank in inputs.values() for t in bank.elements)
    cuda_rng = torch.cuda.get_rng_state(device) if device.startswith("cuda") else None
    cpu_rng = torch.get_rng_state()
    result = optimize_region(program, {cell.digest: cell}, inputs, payload_limits=(RegionPayloadLimit(device, 1 << 20),),
        measurement_rounds=2)
    assert all(len(c.elapsed_ns) == 2 and all(t > 0 for t in c.elapsed_ns) for c in result.request.candidates)
    actual = result.run(inputs)
    assert actual.invocations == 6
    weights, hidden = actual.values
    # Independent closed-form oracle: layer l starts at l+1 and advances once
    # per chunk; chunk c reads both weights at exactly their c-th versions.
    for l, value in enumerate(weights.elements):
        torch.testing.assert_close(value, torch.full_like(value, l + 4), rtol=0, atol=0)
    for c, value in enumerate(hidden.elements):
        torch.testing.assert_close(value, torch.full_like(value, 3 * c + 5), rtol=0, atol=0)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    for before, after in zip(saved, (t for bank in inputs.values() for t in bank.elements), strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
