"""Region tuning and selection: measured latency inside hard legality and memory limits.

``select_region_candidate`` ranks legal candidates by measured latency, but only
among those whose symbolic capacity (tensors plus retained recovery storage)
fits the payload limits and whose numerical changes are permitted.
``optimize_region`` and ``TensorScan.optimized`` measure on private inputs and
return an execution that serves every declared size without replanning.
"""

import warnings
from dataclasses import replace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from pllm.config import ExecutionOptions
from pllm.contracts.backend_attachment import CausalModelDomain
from pllm.contracts.region_imports import ScanForm
from pllm.contracts.region_program import (
    Call,
    Program,
    SizeParameter,
    TensorType,
    Value,
)
from pllm.contracts.region_selection import (
    RegionCandidate,
    RegionPayloadLimit,
    RegionSelectionPolicy,
    RegionSelectionRequest,
    RegionTuningSize,
)
from pllm.execution.affine_scan import candidates as affine_candidates
from pllm.execution.exported_cell import ExportedCell, NumericalTransformWarning
from pllm.execution.region_compilation import (
    analyze_region_program,
    analyze_region_retention,
    explore_region_schedules,
    lower_region_schedule,
)
from pllm.execution.region_optimization import (
    assess_region_candidates,
    bind_region_candidate,
    optimize_region,
    select_region_candidate,
)
from pllm.execution.region_schedule import RegionScheduleExecution
from pllm.execution.region_storage import analyze_region_payload, import_region_functions
from pllm.execution.region_values import Bank
from pllm.execution.replay import Recurrence
from pllm.execution.tensor_scan import TensorScan
from pllm.execution.tokenwise_region import TokenwiseRegion
from pllm.kernels.additive_scan import AdditiveBuffers, launch_window
from test.region_fixtures import (
    StorageObserver,
    assert_grid_closed_form,
    grid_cell,
    inputs_for,
    reassociation_candidates,
    region,
    single_call,
    tensor_grid,
)


def measured(**changes):
    return replace(ExecutionOptions(**changes), tuning="measured")


def chain_request(shapes=((4, 1), (1, 4), (4, 32))):
    """A source call and its reassociated variant with fixed timing evidence."""
    tensors = tuple(torch.ones(shape, dtype=torch.float64) for shape in shapes)
    cell = ExportedCell.capture(lambda a, b, c: ((a @ b) @ c,), ("a", "b", "c"), ("out",), tensors)
    program = single_call(cell)
    (variant,) = reassociation_candidates(cell)
    schedule = analyze_region_program(program).reference_schedule
    source = import_region_functions(program, {cell.digest: cell})
    rewritten = import_region_functions(program, {cell.digest: variant.cell})
    request = RegionSelectionRequest(
        source,
        (RegionCandidate(schedule, source, (100, 101, 900)), RegionCandidate(schedule, rewritten, (50, 51, 600))),
        (RegionPayloadLimit("cpu", 1 << 20),),
        True,
        "deterministic timing oracle",
    )
    return cell, program, variant, request


def with_times(request, *times):
    return replace(
        request, candidates=tuple(replace(c, elapsed_ns=t) for c, t in zip(request.candidates, times, strict=True))
    )


def test_latency_ranks_only_feasible_permitted_candidates():
    cell, program, variant, request = chain_request()
    selected = select_region_candidate(program, request)
    assert selected.selected == 1  # The faster rewrite wins although it needs more storage.
    assert select_region_candidate(program, with_times(request, (100,), (200, 201, 202))).selected == 0
    source_bytes = selected.candidates[0].capacity.devices[0].peak_tensor_bytes_upper_bound
    limited = select_region_candidate(
        program, replace(request, payload_limits=(RegionPayloadLimit("cpu", source_bytes),))
    )
    assert limited.selected == 0
    assert select_region_candidate(program, replace(request, allow_numerical_transforms=False)).selected == 0
    with pytest.raises(ValueError):
        select_region_candidate(program, replace(request, payload_limits=(RegionPayloadLimit("cpu", 0),)))


def test_fast_measurement_cannot_authorize_an_illegal_schedule():
    cell = grid_cell()
    program = tensor_grid(3, 2, cell)
    imports = import_region_functions(program, {cell.digest: cell})
    request = RegionSelectionRequest(
        imports,
        (
            RegionCandidate(analyze_region_program(program).reference_schedule, imports, (100,)),
            RegionCandidate("{ S0[c,l] -> [-c,-l] : 0 <= c < 3 and 0 <= l < 2 }", imports, (1,)),
        ),
        (RegionPayloadLimit("cpu", 100000),),
        True,
        "illegal schedule oracle",
    )
    with pytest.raises(ValueError):
        select_region_candidate(program, request)


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_cold_tuning_measures_candidates_on_private_inputs(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cell = grid_cell(device)
    inputs = inputs_for(3, 2, device)
    saved = tuple(t.clone() for bank in inputs.values() for t in bank.elements)
    cuda_rng = torch.cuda.get_rng_state(device) if device.startswith("cuda") else None
    cpu_rng = torch.get_rng_state()
    result = optimize_region(
        tensor_grid(3, 2, cell),
        {cell.digest: cell},
        inputs,
        payload_limits=(RegionPayloadLimit(device, 1 << 20),),
        measurement_rounds=2,
        options=RegionSelectionPolicy(tuning="measured"),
    )
    assert all(len(c.elapsed_ns) == 2 and all(t > 0 for t in c.elapsed_ns) for c in result.request.candidates)
    actual = result.execution.run(inputs)
    assert actual.invocations == 6
    assert_grid_closed_form(actual.values, 3, 2)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    for before, after in zip(saved, (t for bank in inputs.values() for t in bank.elements), strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_tuning_one_example_size_serves_every_declared_size():
    cell = grid_cell()
    program = replace(
        tensor_grid("chunks", "layers", cell), parameters=(SizeParameter("layers", 0, 5), SizeParameter("chunks", 0, 8))
    )
    selected = optimize_region(
        program,
        {cell.digest: cell},
        inputs_for(2, 3),
        sizes={"layers": 3, "chunks": 2},
        payload_limits=(RegionPayloadLimit("cpu", 10000),),
        warmup_rounds=0,
        measurement_rounds=1,
        max_candidates=3,
        options=RegionSelectionPolicy(tuning="measured"),
    )
    for chunks, layers in ((0, 0), (3, 0), (8, 5), (2, 3), (8, 5), (0, 0)):
        result = selected.execution.run(inputs_for(chunks, layers), sizes={"chunks": chunks, "layers": layers})
        assert result.invocations == chunks * layers
        assert_grid_closed_form(result.values, chunks, layers)
    with pytest.raises(ValueError):
        selected.execution.run(inputs_for(2, 3), sizes={"layers": 3, "chunks": 9})


def test_capacity_is_checked_for_the_full_declared_range_not_the_timing_example():
    cell = ExportedCell.capture(lambda x: (x + 1,), ("x",), ("y",), (torch.ones(4),))
    scan = TensorScan(cell, SizeParameter("steps", 0, 1000), carry=0, invariants=1)
    imports = import_region_functions(scan.program, {cell.digest: cell})
    request = RegionSelectionRequest(
        imports,
        (RegionCandidate(analyze_region_program(scan.program).reference_schedule, imports, (100,)),),
        (RegionPayloadLimit("cpu", 1 << 30),),
        True,
        "parameter timing oracle",
        (RegionTuningSize("steps", 2),),
    )
    (assessment,) = assess_region_candidates(scan.program, request)
    other = replace(request, tuning_sizes=(RegionTuningSize("steps", 999),))
    assert assess_region_candidates(scan.program, other)[0].capacity == assessment.capacity
    with pytest.raises(ValueError):  # 100 bytes fit two steps, not the declared thousand.
        select_region_candidate(scan.program, replace(request, payload_limits=(RegionPayloadLimit("cpu", 100),)))


def test_tuned_tensor_scan_keeps_one_symbolic_plan_and_its_guards():
    device = "cpu"
    cell = ExportedCell.capture(
        lambda w, x: (w + x, w),
        ("w", "x"),
        ("next", "old"),
        (torch.ones(3, device=device), torch.zeros(3, device=device)),
    )
    scan = TensorScan(cell, SizeParameter("n", 0, 9), carry=1, invariants=0)
    w = torch.ones(3, device=device)
    rows = tuple(torch.full((3,), i + 1.0, device=device) for i in range(9))
    tuned, report = scan.optimized(
        (w,),
        (),
        (rows[:2],),
        payload_limits=(RegionPayloadLimit(device, 10000),),
        options=measured(allow_numerical_transforms=False),
        measurement_rounds=1,
    )
    assert report.selection is not None and tuned.count == scan.count
    for count in (9, 0, 2):
        for run in (scan, tuned):
            result = run((w,), (), (rows[:count],))
            torch.testing.assert_close(result.carry[0], w + sum(range(1, count + 1)), rtol=0, atol=0)
            assert len(result.columns[0]) == count
            for i, before in enumerate(result.columns[0]):
                torch.testing.assert_close(before, w + sum(range(1, i + 1)), rtol=0, atol=0)
    with pytest.raises(ValueError):  # Beyond the declared maximum trip count.
        tuned((w,), (), (rows + rows[:1],))
    assert torch.equal(w, torch.ones_like(w))

    # A claimed loop rewrite must replay against its source, even when the
    # supplied timing makes it attractive. These checks need no GPU execution.
    with FakeTensorMode():
        operands = tuple(torch.empty(shape, device="cuda:0") for shape in ((3, 4), (2, 4), (2, 3)))
        cell = ExportedCell.capture(
            lambda w, x, y: (w + 0.013 * (y.T @ x), x @ w.T),
            ("w", "x", "y"),
            ("next", "out"),
            operands,
        )
        scan = TensorScan(cell, SizeParameter("n", 0, 9), carry=1, invariants=0)
    imports = import_region_functions(scan.program, {cell.digest: cell})
    limits = (RegionPayloadLimit("cuda:0", 32 << 20),)
    alternatives = tuple(affine_candidates(scan.program, {cell.digest: cell}, limits))
    assert alternatives
    candidate = replace(alternatives[0].candidate, elapsed_ns=(50,))
    request = RegionSelectionRequest(
        imports,
        (RegionCandidate(analyze_region_program(scan.program).reference_schedule, imports, (100,)), candidate),
        limits,
        True,
        "rewrite verification",
        (RegionTuningSize("n", 5),),
    )
    assert select_region_candidate(scan.program, request).selected == 1
    assert select_region_candidate(scan.program, replace(request, allow_numerical_transforms=False)).selected == 0
    source_peak = assess_region_candidates(scan.program, request)[0].peak_bytes
    assert select_region_candidate(scan.program, replace(request, payload_limits=source_peak)).selected == 0
    assert candidate.affine_scan is not None
    forged = replace(candidate, affine_scan=replace(candidate.affine_scan, window=4))
    with pytest.raises(ValueError, match="affine scan imports differ"):
        select_region_candidate(scan.program, replace(request, candidates=(request.candidates[0], forged)))
    tree = next(
        p
        for p in alternatives
        if p.candidate.affine_scan.form is ScanForm.HILLIS_STEELE and p.candidate.affine_scan.window == 8
    )
    evidence = tree.candidate.affine_scan
    assert evidence is not None
    forged = replace(tree.candidate, affine_scan=replace(evidence, form=ScanForm.ORDERED), elapsed_ns=(50,))
    with pytest.raises(ValueError, match="affine scan imports differ"):
        select_region_candidate(scan.program, replace(request, candidates=(request.candidates[0], forged)))
    if torch.cuda.is_available():
        # Separate allocations exercise the bank-packing path as well as a
        # partial final window. Outputs must never expose caller-owned inputs.
        w = torch.ones(3, 4, device="cuda:0")
        xs = tuple(torch.full((2, 4), i / 10, device=w.device) for i in range(5))
        ys = tuple(torch.full((2, 3), i / 20, device=w.device) for i in range(5))
        expected, outputs = w, []
        for x, y in zip(xs, ys, strict=True):
            outputs.append(x @ expected.T)
            expected = expected + 0.013 * (y.T @ x)
        execution = bind_region_candidate(alternatives[0], limits, assess_region_candidates(scan.program, request)[1])
        arguments = dict(
            zip((v.name for v in scan.program.entry.block.arguments), (w, Bank(xs), Bank(ys)), strict=True)
        )
        final, bank = execution.run(arguments, sizes={"n": 5}).values
        assert isinstance(bank, Bank)
        torch.testing.assert_close(final, expected)
        torch.testing.assert_close(bank.elements, tuple(outputs))
        torch.testing.assert_close(w, torch.ones_like(w))
        for actual, value in zip(xs, range(5), strict=True):
            torch.testing.assert_close(actual, torch.full_like(actual, value / 10))
        request = replace(request, candidates=(request.candidates[0], replace(tree.candidate, elapsed_ns=(50,))))
        execution = bind_region_candidate(tree, limits, assess_region_candidates(scan.program, request)[1])
        for count in (0, 1, 3, 5):
            expected, outputs = w, []
            for x, y in zip(xs[:count], ys[:count], strict=True):
                outputs.append(x @ expected.T)
                expected = expected + 0.013 * (y.T @ x)
            arguments = dict(
                zip(
                    (v.name for v in scan.program.entry.block.arguments),
                    (w, Bank(xs[:count]), Bank(ys[:count])),
                    strict=True,
                )
            )
            final, bank = execution.run(arguments, sizes={"n": count}).values
            assert isinstance(bank, Bank)
            torch.testing.assert_close(final, expected)
            torch.testing.assert_close(bank.elements, tuple(outputs))
        # Replay changes only a device count. No recapture, storage replacement
        # or C-specific kernel compilation is needed for the same window.
        full_x = torch.arange(8 * 2 * 4, device=w.device, dtype=w.dtype).view(8, 2, 4) / 100
        full_y = torch.arange(8 * 2 * 3, device=w.device, dtype=w.dtype).view(8, 2, 3) / 100
        staged_x, staged_y = torch.zeros_like(full_x), torch.zeros_like(full_y)
        active = torch.zeros((), device=w.device, dtype=torch.int32)
        buffers = AdditiveBuffers(w, 8, 2)
        launch_window(staged_x, staged_y, w, active, buffers, 0.013, ScanForm.HILLIS_STEELE)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch_window(staged_x, staged_y, w, active, buffers, 0.013, ScanForm.HILLIS_STEELE)
        storage = (buffers.increments, buffers.weights, buffers.output, buffers.final, active)
        pointers = tuple(t.data_ptr() for t in storage)
        for count in (8, 0, 1, 5, 3, 8):
            staged_x.zero_()
            staged_y.zero_()
            staged_x[:count].copy_(full_x[:count])
            staged_y[:count].copy_(full_y[:count])
            active.fill_(count)
            graph.replay()
            increments = 0.013 * torch.bmm(staged_y.transpose(1, 2), staged_x)
            prefixes = torch.cat((w[None], w[None] + increments.double().cumsum(0).float()))
            torch.testing.assert_close(buffers.final, prefixes[count])
            torch.testing.assert_close(buffers.weights[:count], prefixes[:count])
            torch.testing.assert_close(
                buffers.output[:count], torch.bmm(staged_x[:count], (prefixes[:count]).transpose(1, 2))
            )
            assert pointers == tuple(t.data_ptr() for t in storage)
        _check_right_matrix_rewrite()


def _right_oracle(initial, values, keys, alpha, beta):
    """Independent row-wise FP64 SGD, not a coefficient or tree implementation."""
    state, outputs = initial.double(), []
    for x, k in zip(values.double(), keys.double(), strict=True):
        outputs.append(torch.einsum("hij,thj->thi", state, x))
        decayed = state * alpha
        gradient = torch.zeros_like(state)
        for row, key in zip(x, k, strict=True):
            error = torch.einsum("hij,hj->hi", decayed, key) - row
            gradient += torch.einsum("hi,hj->hij", error, key)
        state = decayed - beta * gradient
    return state, tuple(outputs)


def _check_right_matrix_rewrite():
    """Prove the head-batched source, reject bf16, and replay both numerical forms."""
    from pllm.contracts.parameter_overlay import ParameterModuleKind, ParameterRole, ParameterTarget
    from pllm.contracts.region_imports import ScanAction
    from pllm.workloads.gated_delta import GatedDeltaConfig, GatedDeltaWorkload

    target = ParameterTarget(0, ParameterModuleKind.ATTENTION, ParameterRole.QUERY_PROJECTION_WEIGHT)
    workload = GatedDeltaWorkload(GatedDeltaConfig((target,), head_width=3, chunk_size=4, alpha=0.83, beta=0.17))
    limits = (RegionPayloadLimit("cuda:0", 32 << 20),)
    for dtype in (torch.bfloat16, torch.float32):
        with FakeTensorMode():
            samples = tuple(
                torch.empty(shape, dtype=dtype, device="cuda:0") for shape in ((2, 3, 3), (4, 2, 3), (4, 2, 3))
            )
            cell = ExportedCell.capture(workload.step, ("state", "values", "keys"), ("next", "output"), samples)
            scan = TensorScan(cell, SizeParameter("c", 0, 5), carry=1, invariants=0)
        choices = tuple(affine_candidates(scan.program, {cell.digest: cell}, limits))
        if dtype is torch.bfloat16:
            assert not choices
            continue
        assert choices and all(
            c.cells[c.program.functions[-1].name].implementation.action is ScanAction.RIGHT_MATRIX for c in choices
        )
        imports = import_region_functions(scan.program, {cell.digest: cell})
        reference = RegionCandidate(
            analyze_region_program(scan.program, optimize=False).reference_schedule, imports, (100,)
        )
        generator = torch.Generator(device="cuda:0").manual_seed(92)
        initial = torch.randn(2, 3, 3, device="cuda:0", generator=generator) * 0.2
        values = torch.randn(5, 4, 2, 3, device="cuda:0", generator=generator)
        keys = torch.nn.functional.normalize(values, dim=-1)
        serial, serial_outputs = initial, []
        for x, k in zip(values, keys, strict=True):
            serial, output = workload.step(serial, x, k)
            serial_outputs.append(output)
        strict = scan((initial,), (), (tuple(values), tuple(keys)))
        assert torch.equal(strict.carry[0], serial)
        assert all(torch.equal(a, b) for a, b in zip(strict.columns[0], serial_outputs, strict=True))
        for choice in (p for p in choices if p.candidate.affine_scan.window == 4):
            request = RegionSelectionRequest(
                imports,
                (reference, replace(choice.candidate, elapsed_ns=(50,))),
                limits,
                True,
                "right action",
                (RegionTuningSize("c", 5),),
            )
            assert (
                select_region_candidate(scan.program, replace(request, allow_numerical_transforms=False)).selected == 0
            )
            execution = bind_region_candidate(choice, limits, assess_region_candidates(scan.program, request)[1])
            for count in (0, 1, 3, 5):
                args = dict(
                    zip(
                        (v.name for v in scan.program.entry.block.arguments),
                        (initial, Bank(tuple(values[:count])), Bank(tuple(keys[:count]))),
                        strict=True,
                    )
                )
                final, outputs = execution.run(args, sizes={"c": count}).values
                expected, expected_outputs = _right_oracle(initial, values[:count], keys[:count], 0.83, 0.17 / 4)
                torch.testing.assert_close(final.double(), expected, rtol=2e-6, atol=2e-7)
                torch.testing.assert_close(
                    tuple(v.double() for v in outputs.elements), expected_outputs, rtol=2e-6, atol=2e-7
                )
        _check_right_matrix_graph(initial, values, keys)
    # Cost follows the source chunk work, not an absolute head-width threshold.
    for width, chunk, competitive in ((64, 64, True), (896, 64, False), (896, 896, True)):
        with FakeTensorMode():
            samples = tuple(
                torch.empty(shape, device="cuda:0")
                for shape in ((1, width, width), (chunk, 1, width), (chunk, 1, width))
            )
            cell = ExportedCell.capture(workload.step, ("state", "values", "keys"), ("next", "output"), samples)
            scan = TensorScan(cell, 8, carry=1, invariants=0)
        limits = (RegionPayloadLimit("cuda:0", 1 << 30),)
        offered = tuple(affine_candidates(scan.program, {cell.digest: cell}, limits))
        assert bool(offered) == competitive
        if competitive:
            forms = {(p.candidate.affine_scan.form, p.candidate.affine_scan.window) for p in offered}
            assert (ScanForm.HILLIS_STEELE, 8) in forms
            assert (ScanForm.ORDERED, 2) in forms and (ScanForm.HILLIS_STEELE, 2) not in forms


def _check_right_matrix_graph(initial, values, keys):
    """Changing C reuses one captured window; reversed right actions fail the oracle."""
    from pllm.kernels.right_matrix_scan import RightMatrixBuffers, launch_window as launch_right_window

    x = initial.new_zeros((8, 2, 4, 3))
    k = torch.zeros_like(x)
    active = torch.zeros((), dtype=torch.int32, device=initial.device)
    buffers = RightMatrixBuffers(initial, 8, 4)
    launch_right_window(x, k, initial, active, buffers, 0.17, 0.83, ScanForm.HILLIS_STEELE)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_right_window(x, k, initial, active, buffers, 0.17, 0.83, ScanForm.HILLIS_STEELE)
    storage = (x, k, active, buffers.a, buffers.b, buffers.states, buffers.final, buffers.output)
    addresses = tuple(t.data_ptr() for t in storage)
    for count in (5, 0, 1, 3, 5):
        x.zero_()
        k.zero_()
        x[:count].copy_(values[:count].transpose(1, 2))
        k[:count].copy_(keys[:count].transpose(1, 2))
        active.fill_(count)
        graph.replay()
        expected, outputs = _right_oracle(initial, values[:count], keys[:count], 0.83, 0.17)
        torch.testing.assert_close(buffers.final.double(), expected, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(
            tuple(v.transpose(0, 1).double() for v in buffers.output[:count]), outputs, rtol=2e-6, atol=2e-7
        )
        assert addresses == tuple(t.data_ptr() for t in storage)
    # Noncommuting symmetric per-chunk A matrices expose reversed composition;
    # independent B and a nonsymmetric initial state also expose left action.
    vv, kk = values.double().transpose(1, 2), keys.double().transpose(1, 2)
    a = 0.83 * (torch.eye(3, device=initial.device) - 0.17 * (kk.transpose(-2, -1) @ kk))
    b = 0.17 * (vv.transpose(-2, -1) @ kk)
    wrong = initial.double()
    for aa, bb in zip(a, b, strict=True):
        wrong = aa @ wrong + bb
    assert not torch.allclose(wrong, expected, rtol=1e-3, atol=1e-3)
    prefix_a, prefix_b = a[0], b[0]
    for aa, bb in zip(a[1:], b[1:], strict=True):
        prefix_a, prefix_b = aa @ prefix_a, bb @ prefix_a + prefix_b
    assert not torch.allclose(initial.double() @ prefix_a + prefix_b, expected, rtol=1e-3, atol=1e-3)


def test_tuner_without_numerical_permission_never_warns_or_offers_rewrites():
    cell, program, _, request = chain_request()
    inputs = dict(
        zip(
            ("in0", "in1", "in2"),
            (
                torch.ones(4, 1, dtype=torch.float64),
                torch.ones(1, 4, dtype=torch.float64),
                torch.ones(4, 32, dtype=torch.float64),
            ),
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = optimize_region(
            program,
            {cell.digest: cell},
            inputs,
            payload_limits=(RegionPayloadLimit("cpu", 10000),),
            options=measured(allow_numerical_transforms=False, schedule_policy="serial"),
            warmup_rounds=1,
            measurement_rounds=3,
        )
    assert result.selection.selected == 0 and not result.rewrites and len(result.request.candidates) == 1
    assert not any(isinstance(w.message, NumericalTransformWarning) for w in caught)


def two_chains(device="cpu"):
    tensors = tuple(torch.ones(shape, dtype=torch.float64, device=device) for shape in ((4, 3), (3, 128), (128, 2)))
    cells = tuple(
        ExportedCell.capture(f, ("a", "b", "c"), ("out",), tensors)
        for f in (lambda a, b, c: ((a @ b) @ c,), lambda a, b, c: (torch.matmul(torch.matmul(a, b), c),))
    )
    arguments = tuple(Value(name, t) for name, t in zip(("a", "b", "c"), cells[0].declaration.inputs))
    calls = tuple(
        Call(("a", "b", "c"), (Value(f"out{i}", cell.declaration.outputs[0]),), cell.digest)
        for i, cell in enumerate(cells)
    )
    program = Program(region(arguments, calls, ("out0", "out1")), tuple(c.declaration for c in cells))
    return program, {c.digest: c for c in cells}, dict(zip(("a", "b", "c"), tensors))


def test_only_a_joint_rewrite_fits_and_budgets_never_authorize_an_over_budget_one():
    # Each source call needs a 4*128*8-byte product; only rewriting both calls fits 5392 bytes.
    program, cells, inputs = two_chains()
    with pytest.warns(NumericalTransformWarning):
        selected = optimize_region(
            program,
            cells,
            inputs,
            payload_limits=(RegionPayloadLimit("cpu", 5392),),
            options=measured(schedule_policy="serial"),
            warmup_rounds=0,
            measurement_rounds=2,
        )
    with StorageObserver(tuple(inputs.values())) as observed:
        result = selected.execution.run(inputs)
    assert observed.peak - sum(t.numel() * t.element_size() for t in inputs.values()) <= 5392
    for value in result.values:
        torch.testing.assert_close(value, torch.full_like(value, 384), rtol=0, atol=0)
    for limit in ({"max_rewrite_depth": 1}, {"options": measured(allow_numerical_transforms=False)}):
        with pytest.raises(ValueError):
            optimize_region(
                program,
                cells,
                inputs,
                payload_limits=(RegionPayloadLimit("cpu", 5392),),
                warmup_rounds=0,
                measurement_rounds=1,
                **{"options": measured(schedule_policy="serial"), **limit},
            )


def recovery_region():
    return TokenwiseRegion(
        TensorType((1, 4), "torch.float32"),
        8,
        CausalModelDomain(1, 3, 8),
        Recurrence().effect(3),
    )


def test_bounded_window_is_selected_when_the_faster_candidate_exceeds_the_budget():
    tokenwise = recovery_region()
    analysis = analyze_region_program(tokenwise.program)
    window = next(c for c in explore_region_schedules(tokenwise.program).candidates if "-window-2-" in c.source)
    bound = analyze_region_retention(tokenwise.program, window.schedule_map, tokenwise.retained)
    assert bound.allocations[0].peak_instances_upper_bound == 2
    windowed = RegionScheduleExecution(tokenwise.program, lower_region_schedule(tokenwise.program, window.schedule_map))
    serial = RegionScheduleExecution(
        tokenwise.program, lower_region_schedule(tokenwise.program, analysis.reference_schedule)
    )
    for count in (0, 1, 3, 8):
        sizes = {"tokens0": count, "prefix0": 0}
        actual, expected = tuple(windowed.invocations(sizes)), tuple(serial.invocations(sizes))
        assert set(actual) == set(expected) and len(actual) == len(expected)
        windows_seen = tuple(point.coordinates[0] // 2 for point in actual)
        assert windows_seen == tuple(sorted(windows_seen))
    imports = import_region_functions(tokenwise.program, tokenwise.functions)
    payload = analyze_region_payload(tokenwise.program, window.schedule_map, tokenwise.functions, tokenwise.retained)
    limit = payload.devices[0].peak_bytes_upper_bound
    # Fixed scores isolate the capacity constraint: the fastest candidate retains too much.
    request = RegionSelectionRequest(
        imports,
        (
            RegionCandidate(analysis.reference_schedule, imports, (100,)),
            RegionCandidate(window.schedule_map, imports, (10,)),
            RegionCandidate(analysis.candidates[-1].schedule_map, imports, (1,)),
        ),
        (RegionPayloadLimit("cpu", limit),),
        True,
        "window capacity oracle",
        (RegionTuningSize("prefix0", 0), RegionTuningSize("tokens0", 3)),
        tokenwise.retained,
    )
    selection = select_region_candidate(tokenwise.program, request)
    assert selection.selected == 1
    assert [c.feasible for c in selection.candidates] == [True, True, False]
    assert selection.candidates[1].payload.retention == bound
