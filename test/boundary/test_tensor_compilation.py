"""Real fused backend code, read-only borrows, and graph-derived capacity facts."""

from dataclasses import replace
import math
import warnings

import pytest
import torch

from pllm.config import ExecutionOptions
from pllm.contracts.region_imports import RegionTensorImport
from pllm.contracts.region_selection import RegionCandidate, RegionPayloadLimit, RegionSelectionRequest
from pllm.contracts.region_program import Block, Call, Program, Region, ScalarType, Scan, SizeParameter, Value
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.contracts.tensor_compilation import UnsupportedTensorKernel
from pllm.backends.torch_inductor import COMPILER_SLACK_FRACTION, InductorTensorCompiler, compile_inductor_tensor_graph
from pllm.execution.exported_cell import ExportedCell, NumericalTransformWarning
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionExecution, RegionInstructionExecution
from pllm.execution.region_native import CompiledPrimitive
from pllm.execution.region_optimization import bind_selected_region, optimize_region, select_region_candidate
from pllm.execution.region_storage import analyze_region_capacity, import_region_functions
from pllm.execution.tensor_compilation import compile_tensor_cell, supports_tensor_compilation
from pllm.execution.tensor_scan import TensorScan
from test.boundary.test_region_native_capacity import single_call


def example(device="cpu", dtype=torch.float32, *, duplicate=False):
    x, y = torch.full((128,), 2, device=device, dtype=dtype), torch.ones(128, device=device, dtype=dtype)
    def arithmetic(x, y):
        return ((x + y) * (x - y) + x,)
    def update(x, y):
        x.add_(1)
        return x, x, y
    cell = ExportedCell.capture(update if duplicate else arithmetic, ("x", "y"),
                                ("new", "duplicate", "old") if duplicate else ("out",), (x, y))
    return cell, (x, y)


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
@pytest.mark.parametrize("dtype", (torch.float64, torch.float32, torch.bfloat16))
def test_codegen_exports_real_allocations_and_preserves_values_inputs_and_rng(device, dtype):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    source, tensors = example(device, dtype)
    original = str(source.graph.graph)
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.startswith("cuda") else None
    with warnings.catch_warnings(record=True) as caught:
        compiled = compile_tensor_cell(source)
    assert not any(isinstance(w.message, NumericalTransformWarning) for w in caught)
    assert len([n for n in source.graph.graph.nodes if n.op == "call_function"]) == 4
    # Capacity is a graph-derived account: the layout span of every tensor
    # value in the normalized graph plus the named compiler slack — never a
    # code-generation allocation read. Fusion only makes the true peak lower.
    subtotal = 4 * tensors[0].numel() * tensors[0].element_size()
    allocations = compiled.implementation.allocations
    assert sum(a.bytes for a in allocations) == subtotal + math.ceil(subtotal * COMPILER_SLACK_FRACTION)
    assert allocations[-1].name == f"compiler-slack@{COMPILER_SLACK_FRACTION}"
    actual, = compiled.invoke(tensors, None)
    torch.testing.assert_close(actual, torch.full_like(actual, 5), rtol=0, atol=0)
    torch.testing.assert_close(tensors[0], torch.full_like(tensors[0], 2), rtol=0, atol=0)
    torch.testing.assert_close(tensors[1], torch.ones_like(tensors[1]), rtol=0, atol=0)
    assert str(source.graph.graph) == original
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    assert decode_boundary(encode_boundary(compiled.implementation), RegionTensorImport) == compiled.implementation


@pytest.mark.parametrize("duplicate", (False, True))
def test_capacity_includes_entry_inputs_raw_kernel_results_and_every_output_copy(duplicate):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA allocation accounting")
    source, tensors = example("cuda:0", duplicate=duplicate)
    compiled = compile_tensor_cell(source)
    program = single_call(source)
    schedule = analyze_region_program(program).reference_schedule
    capacity = analyze_region_capacity(program, schedule, {source.digest: compiled})
    facts, = capacity.devices
    scratch = sum(a.bytes for a in compiled.implementation.allocations)
    assert facts.entry_copy_bytes_upper_bound == 1024
    assert facts.call_scratch_bytes_upper_bound == scratch  # Graph account; inputs borrow entry roots.
    assert facts.peak_tensor_bytes_upper_bound == 1024 + scratch + (1536 if duplicate else 512)
    # Access traffic is deliberately unknown: wrapper copies do not describe
    # the fused kernel's global/cache transactions.
    assert capacity.unknown_functions == (source.digest,)
    run = RegionInstructionExecution(program, lower_region_schedule(program, schedule),
        lower_region_bindings(program, schedule), {source.digest: compiled})
    inputs = dict(zip(("in0", "in1"), tensors))
    warmup = run.run(inputs)
    del warmup
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    actual = run.run(inputs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    # The graph-derived account charges every value as if it materialized, so
    # the fused kernel's measured peak stays strictly under the bound.
    assert peak <= facts.peak_tensor_bytes_upper_bound
    if duplicate:
        new, repeated, old = actual.values
        torch.testing.assert_close(new, torch.full_like(new, 3), rtol=0, atol=0)
        torch.testing.assert_close(repeated, new, rtol=0, atol=0)
        torch.testing.assert_close(old, tensors[1], rtol=0, atol=0)
        assert len({t.untyped_storage().data_ptr() for t in actual.values}) == 3
    with pytest.raises(ValueError, match="selection decision"):
        RegionExecution(program, lower_region_schedule(program, schedule), lower_region_bindings(program, schedule),
                        {source.digest: compiled})


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_automatic_selection_replays_compiler_and_run_does_not_compile(device, monkeypatch):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    import pllm.execution.region_optimization as tuning
    source, tensors = example(device)
    monkeypatch.setattr(tuning, "_measure", lambda execution, *args:
        1 if isinstance(execution._cells[source.digest], CompiledPrimitive) else 100)
    calls = []
    class RecordingCompiler:
        def supports(self, graph):
            calls.append("supports")
            return InductorTensorCompiler().supports(graph)

        def compile(self, graph):
            calls.append("compile")
            code = compile_inductor_tensor_graph(graph)
            # A distinct provider identity exposes accidental default-backend
            # reconstruction while retaining real generated code and budgets.
            return replace(code, backend="recording-provider:" + code.backend)
    options = ExecutionOptions(schedule_policy="serial")
    with pytest.warns(NumericalTransformWarning, match="kernel compilation"):
        selected = optimize_region(single_call(source), {source.digest: source}, dict(zip(("in0", "in1"), tensors)),
            payload_limits=(RegionPayloadLimit(device, 10000),), options=options, warmup_rounds=0, measurement_rounds=1,
            compiler=RecordingCompiler())
    assert selected.selection.selected == 1 and not selected.rewrites
    assert "supports" in calls and calls.count("compile") == 1  # Candidate only; finish installs the measured executor.
    calls.clear()
    assert selected.selection.candidates[1].changed_functions == (source.digest,)
    def forbidden(*args, **kwargs):
        raise AssertionError("selected execution must not compile or select")
    monkeypatch.setattr(tuning, "compile_tensor_cell", forbidden)
    import torch._inductor.compile_fx as inductor
    monkeypatch.setattr(inductor, "compile_fx_inner", forbidden)
    if device.startswith("cuda"):
        import triton
        from triton.runtime.jit import JITFunction
        monkeypatch.setattr(triton, "compile", forbidden)
        monkeypatch.setattr(JITFunction, "_do_compile", forbidden)
    actual, = selected.run(dict(zip(("in0", "in1"), tensors))).values
    torch.testing.assert_close(actual, torch.full_like(actual, 5), rtol=0, atol=0)
    assert not calls


def test_backend_declining_compilation_preserves_original_schedule_execution():
    class DecliningCompiler:
        def supports(self, graph):
            return False

        def compile(self, graph):
            raise AssertionError("a declined implementation must not be compiled")
    source, tensors = example()
    inputs = dict(zip(("in0", "in1"), tensors))
    selected = optimize_region(single_call(source), {source.digest: source}, inputs,
        payload_limits=(RegionPayloadLimit("cpu", 10000),), compiler=DecliningCompiler(),
        options=ExecutionOptions(schedule_policy="serial"), warmup_rounds=0, measurement_rounds=1)
    assert len(selected.request.candidates) == 1
    torch.testing.assert_close(selected.run(inputs).values, (torch.full_like(tensors[0], 5),), rtol=0, atol=0)


def test_closed_backend_facts_are_replayed_and_cannot_be_replaced_by_zero_cost_claims():
    source, _ = example()
    compiled = compile_tensor_cell(source)
    again = compile_tensor_cell(source)
    assert again.implementation == compiled.implementation
    program = single_call(source)
    schedule = analyze_region_program(program).reference_schedule
    original = import_region_functions(program, {source.digest: source})
    request = RegionSelectionRequest(original, (RegionCandidate(schedule, original, (100,)),
        RegionCandidate(schedule, (compiled.implementation,), (1,))), (RegionPayloadLimit("cpu", 10000),), True, "backend replay oracle")
    # Old selections must be rebuilt from source; do not silently strip a
    # field from signed imports and reuse the previous capacity/selection.
    options = ExecutionOptions(schedule_policy="serial")
    with pytest.warns(NumericalTransformWarning):
        bind_selected_region(program, {source.digest: source}, request, select_region_candidate(program, request), (), options=options)
    for fake in (replace(compiled.implementation, allocations=()),
                 replace(compiled.implementation, code_digest="0" * 64),
                 replace(compiled.implementation, backend="another compiler")):
        forged = replace(request, candidates=(request.candidates[0], replace(request.candidates[1], imports=(fake,))))
        decision = select_region_candidate(program, forged)
        with pytest.raises(ValueError, match="backend code or allocation facts"):
            bind_selected_region(program, {source.digest: source}, forged, decision, (), options=options)
    for fake in (replace(compiled.implementation, code_digest="not a digest"),
                 replace(compiled.implementation, allocations=compiled.implementation.allocations * 2)):
        with pytest.raises(ValueError):
            select_region_candidate(program, replace(request, candidates=(request.candidates[0],
                replace(request.candidates[1], imports=(fake,)))))
    assert select_region_candidate(program, replace(request, allow_numerical_transforms=False)).selected == 0


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_compiler_eliminated_view_keeps_output_copy_and_capacity_contract(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    x = torch.arange(128, dtype=torch.float32, device=device)
    source = ExportedCell.capture(lambda x: (x.view(128),), ("x",), ("out",), (x,))
    compiled = compile_tensor_cell(source)
    # The compiler eliminates the view; the graph account still charges its
    # span plus slack — a deliberate over-count, never an omission.
    scratch = x.numel() * x.element_size()
    scratch += math.ceil(scratch * COMPILER_SLACK_FRACTION)
    assert sum(a.bytes for a in compiled.implementation.allocations) == scratch
    actual, = compiled.invoke((x,), None)
    torch.testing.assert_close(actual, x, rtol=0, atol=0)
    assert actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    program = single_call(source)
    schedule = analyze_region_program(program, optimize=False).reference_schedule
    capacity = analyze_region_capacity(program, schedule, {source.digest: compiled})
    facts, = capacity.devices
    assert facts.call_scratch_bytes_upper_bound == scratch
    assert facts.peak_tensor_bytes_upper_bound == 2 * x.numel() * x.element_size() + scratch


def test_optout_avoids_compilation_and_capacity_can_reject_a_faster_compiled_plan(monkeypatch):
    import pllm.execution.region_optimization as tuning
    source, tensors = example()
    source = ExportedCell.capture(lambda x, y: (x + y,), ("x", "y"), ("out",), tensors)
    program, cells, inputs = single_call(source), {source.digest: source}, dict(zip(("in0", "in1"), tensors))
    def forbidden(*args, **kwargs):
        raise AssertionError("numerical optout must not compile a candidate")
    with monkeypatch.context() as patch:
        patch.setattr(tuning, "compile_tensor_cell", forbidden)
        selected = optimize_region(program, cells, inputs, payload_limits=(RegionPayloadLimit("cpu", 10000),),
            options=ExecutionOptions(allow_numerical_transforms=False, schedule_policy="serial"),
            warmup_rounds=0, measurement_rounds=1)
    assert len(selected.request.candidates) == 1
    original = analyze_region_capacity(program, analyze_region_program(program).reference_schedule, cells)
    def measure(execution, *args):
        assert not isinstance(execution._cells[source.digest], CompiledPrimitive)
        return 100
    monkeypatch.setattr(tuning, "_measure", measure)
    selected = optimize_region(program, cells, inputs,
        payload_limits=(RegionPayloadLimit("cpu", original.devices[0].peak_tensor_bytes_upper_bound),),
        options=ExecutionOptions(schedule_policy="serial"), warmup_rounds=0, measurement_rounds=1)
    assert selected.selection.selected == 0 and not selected.selection.candidates[1].feasible
    assert selected.request.candidates[1].elapsed_ns == ()


def test_unsupported_external_workspace_and_layouts_are_not_assigned_zero_cost():
    x = torch.ones(4, 4)
    matrix = ExportedCell.capture(lambda x, y: (torch.mm(x, y),), ("x", "y"), ("out",), (x, x.clone()))
    view = ExportedCell.capture(lambda x: ((x + 1).T,), ("x",), ("out",), (x,))
    for source in (matrix, view):
        assert not supports_tensor_compilation(source)
        with pytest.raises(UnsupportedTensorKernel):
            compile_tensor_cell(source)
    # Bypassing the frontend cannot assign CPU library GEMM the allocation
    # contract of a generated CUDA template.
    with pytest.raises(UnsupportedTensorKernel, match="CUDA"):
        compile_inductor_tensor_graph(matrix.graph)


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_reduction_normalization_enters_selection_with_owned_scratch_and_readonly_inputs(device, monkeypatch):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    import pllm.execution.region_optimization as tuning
    # Non-power-of-two rows/columns, two reduction directions, scalar output,
    # and widened accumulation exercise more than a pointwise epilogue.
    base = torch.linspace(-2, 3, 17 * 129 + 1, device=device)
    x = base[1:].view(17, 129)
    before = base.clone()
    def normalize(x):
        mean_square = (x * x).mean(dim=1, keepdim=True)
        return x * torch.rsqrt(mean_square + 1e-5), x.sum(dim=0), x.sum(dtype=torch.float64)
    source = ExportedCell.capture(normalize, ("x",), ("normalized", "columns", "total"), (x,))
    monkeypatch.setattr(tuning, "_measure", lambda execution, *args:
        1 if isinstance(execution._cells[source.digest], CompiledPrimitive) else 100)
    with pytest.warns(NumericalTransformWarning, match="kernel compilation"):
        selected = optimize_region(single_call(source), {source.digest: source}, {"in0": x},
            payload_limits=(RegionPayloadLimit(device, 1_000_000),),
            options=ExecutionOptions(schedule_policy="serial"), max_candidates=2,
            warmup_rounds=0, measurement_rounds=1)
    assert selected.selection.selected == 1
    compiled = selected._execution._cells[source.digest]
    assert isinstance(compiled, CompiledPrimitive)
    allocations = compiled.implementation.allocations
    assert allocations and sum(a.bytes for a in allocations) >= source.output_bytes
    capacity = analyze_region_capacity(single_call(source), selected.request.candidates[1].schedule_map,
                                       {source.digest: compiled})
    assert capacity.devices[0].call_scratch_bytes_upper_bound == sum(a.bytes for a in allocations)
    assert capacity.unknown_functions == (source.digest,)  # Allocation facts are not measured traffic.
    def forbidden(*args, **kwargs):
        raise AssertionError("selected reduction must run its already-compiled kernels")
    monkeypatch.setattr(tuning, "compile_tensor_cell", forbidden)
    import torch._inductor.compile_fx as inductor
    monkeypatch.setattr(inductor, "compile_fx_inner", forbidden)
    actual = selected.run({"in0": x}).values
    reference = x.double()
    expected = ((reference / torch.sqrt((reference * reference).sum(dim=1, keepdim=True) / 129 + 1e-5)).float(),
                reference.sum(dim=0).float(), reference.sum())
    for result, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, oracle)
    torch.testing.assert_close(base, before, rtol=0, atol=0)
    assert len({value.untyped_storage().data_ptr() for value in (*actual, base)}) == 4


def test_selection_combines_compilers_across_loop_boundary_without_singleton_wins(monkeypatch):
    import pllm.execution.region_optimization as tuning
    x = torch.arange(1, 130, dtype=torch.float64)
    reduce = ExportedCell.capture(lambda x: ((x * x).mean(),), ("x",), ("mean",), (x,))
    advance = ExportedCell.capture(lambda x: (x * 0.5 + 1,), ("x",), ("next",), (x.mean(),))
    t, = reduce.declaration.outputs
    loop = Region(Block((Value("i", ScalarType.INDEX), Value("carry", t)), (
        Call(("carry",), (Value("next", t),), advance.digest),
    ), ("next",)))
    program = Program(Region(Block((Value("x", reduce.declaration.inputs[0]),), (
        Call(("x",), (Value("mean", t),), reduce.digest),
        Scan(("mean",), (Value("result", t),), 3, loop),
    ), ("result",))), (reduce.declaration, advance.declaration))
    def measure(execution, *args):
        count = sum(isinstance(cell, CompiledPrimitive) for cell in execution._cells.values())
        return (100, 200, 1)[count]  # Either individual change loses; their combination wins.
    monkeypatch.setattr(tuning, "_measure", measure)
    with pytest.warns(NumericalTransformWarning, match="kernel compilation"):
        selected = optimize_region(program, {reduce.digest: reduce, advance.digest: advance}, {"x": x},
            payload_limits=(RegionPayloadLimit("cpu", 10000),),
            options=ExecutionOptions(schedule_policy="serial"), max_candidates=4,
            warmup_rounds=0, measurement_rounds=1)
    assert selected.selection.selected == 3
    assert len(selected.selection.candidates[3].changed_functions) == 2
    assert all(not c.outlines for c in selected.request.candidates)
    actual, = selected.run({"x": x}).values
    expected = sum(i * i for i in range(1, 130)) / 129
    for _ in range(3):
        expected = expected * 0.5 + 1
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=x.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
@pytest.mark.parametrize("duplicate", (False, True))
def test_readonly_kernel_borrows_overlapping_unaligned_inputs_without_hidden_copies(device, duplicate):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    source, _ = example(device, duplicate=duplicate)
    compiled = compile_tensor_cell(source)
    base = torch.arange(130, dtype=torch.float32, device=device)
    x, y = base[1:129], base[2:130]
    assert x.data_ptr() % 16 == 4 and y.data_ptr() % 16 == 8
    before = base.clone()
    expected = (x + 1, x + 1, y.clone()) if duplicate else ((x + y) * (x - y) + x,)
    observed = []
    def record(*inputs):
        observed.append(tuple(t.data_ptr() for t in inputs))
        return compiled.function(*inputs)
    run = replace(compiled, function=record)
    # Compile and warm before observing allocation. The next call may neither
    # copy its borrows nor hide an alignment copy inside the generated wrapper.
    warmup = run.invoke((x, y), None)
    del warmup
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    actual = run.invoke((x, y), None)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        # One 512-byte raw result plus each independent 512-byte output.
        assert torch.cuda.max_memory_allocated() - allocated == (2048 if duplicate else 1024)
    assert observed == [(x.data_ptr(), y.data_ptr())] * 2
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    torch.testing.assert_close(base, before, rtol=0, atol=0)
    assert len({value.untyped_storage().data_ptr() for value in (*actual, base)}) == len(actual) + 1


@pytest.mark.parametrize("violation", ("mutation", "alignment"))
def test_readonly_import_rejects_backend_writes_and_hidden_alignment_copies(violation, monkeypatch):
    import torch._inductor.compile_fx as inductor
    source, _ = example()
    original = inductor.compile_fx_inner
    def with_violation(*args, **kwargs):
        artifact = original(*args, **kwargs)
        if violation == "mutation":
            artifact.mutated_inputs.add("arg0_1")
        else:
            artifact.inputs_to_check = (0,)
        return artifact
    monkeypatch.setattr(inductor, "compile_fx_inner", with_violation)
    with pytest.raises(UnsupportedTensorKernel, match="mutation.*alignment copies"):
        compile_tensor_cell(source)


def test_readonly_codegen_overrides_and_restores_global_alignment_policy():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA alignment checks")
    from torch._inductor import config
    source, _ = example("cuda:0")
    with config.patch(assume_aligned_inputs=True):
        compiled = compile_tensor_cell(source)
        assert config.assume_aligned_inputs
    base = torch.ones(129, device="cuda:0")
    actual, = compiled.invoke((base[1:], base[1:]), None)
    torch.testing.assert_close(actual, torch.ones_like(actual), rtol=0, atol=0)


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_compiled_borrow_keeps_region_view_origin_and_old_versions(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    base = torch.arange(129, dtype=torch.float32, device=device)
    view = ExportedCell.capture(lambda base: (base[1:],), ("base",), ("view",), (base,))
    update, _ = example(device, duplicate=True)
    compiled = compile_tensor_cell(update)
    observed = []
    def record(*inputs):
        observed.append(tuple((t.storage_offset(), t.data_ptr()) for t in inputs))
        return compiled.function(*inputs)
    t, = view.declaration.outputs
    program = Program(Region(Block((Value("base", view.declaration.inputs[0]),), (
        Call(("base",), (Value("view", t),), view.digest),
        Call(("view", "view"), (Value("new", t), Value("again", t), Value("old", t)), update.digest),
    ), ("base", "view", "new", "again", "old"))), (view.declaration, update.declaration))
    cells = {view.digest: view, update.digest: replace(compiled, function=record)}
    schedule = analyze_region_program(program).reference_schedule
    capacity = analyze_region_capacity(program, schedule, cells)
    assert capacity.devices[0].call_scratch_bytes_upper_bound == sum(
        a.bytes for a in compiled.implementation.allocations)
    run = RegionInstructionExecution(program, lower_region_schedule(program, schedule),
        lower_region_bindings(program, schedule), cells)
    old_base, old_view, new, again, old = run.run({"base": base}).values
    assert old_view.untyped_storage().data_ptr() == old_base.untyped_storage().data_ptr()
    assert observed == [((1, old_view.data_ptr()), (1, old_view.data_ptr()))]
    for value in (old_view, old):
        torch.testing.assert_close(value, base[1:], rtol=0, atol=0)
    for value in (new, again):
        torch.testing.assert_close(value, base[1:] + 1, rtol=0, atol=0)
    torch.testing.assert_close(old_base, base, rtol=0, atol=0)


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_outlined_calls_can_be_selected_as_one_real_backend_kernel(device, monkeypatch):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    import pllm.execution.region_optimization as tuning
    x, y = torch.full((128,), 2.0, device=device), torch.ones(128, device=device)
    first = ExportedCell.capture(lambda x, y: (x + y, x - y), ("x", "y"), ("sum", "diff"), (x, y))
    second = ExportedCell.capture(lambda s, d, x: (s * d + x,), ("sum", "diff", "x"), ("out",), (x + y, x - y, x))
    t, = second.declaration.outputs
    program = Program(Region(Block((Value("x", t), Value("y", t)), (
        Call(("x", "y"), (Value("sum", t), Value("diff", t)), first.digest),
        Call(("sum", "diff", "x"), (Value("out", t),), second.digest),
    ), ("out",))), (first.declaration, second.declaration))
    sources = {first.digest: first, second.digest: second}
    monkeypatch.setattr(tuning, "_measure", lambda execution, *args:
        1 if any(isinstance(cell, CompiledPrimitive) and name not in sources for name, cell in execution._cells.items()) else 100)
    with pytest.warns(NumericalTransformWarning, match="kernel compilation"):
        selected = optimize_region(program, sources, {"x": x, "y": y},
            payload_limits=(RegionPayloadLimit(device, 10000),), options=ExecutionOptions(schedule_policy="serial"),
            warmup_rounds=0, measurement_rounds=1)
    candidate = selected.request.candidates[selected.selection.selected]
    assert len(candidate.outlines) == 1
    compiled = [c for c in selected._execution._cells.values() if isinstance(c, CompiledPrimitive)]
    assert len(compiled) == 1
    result = selected.run({"x": x, "y": y})
    assert result.invocations == 1
    torch.testing.assert_close(result.values[0], torch.full_like(x, 5), rtol=0, atol=0)


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_zero_length_tuning_compiles_once_for_other_legal_scan_lengths(device, monkeypatch):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    import pllm.execution.region_optimization as tuning
    x = torch.ones(128, device=device)
    cell = ExportedCell.capture(lambda x: (x * 2 + 1,), ("x",), ("out",), (x,))
    scan = TensorScan(cell, SizeParameter("steps", 0, 1_000_000), carry=1, invariants=0)
    monkeypatch.setattr(tuning, "_measure", lambda execution, *args:
        1 if isinstance(execution._cells[cell.digest], CompiledPrimitive) else 100)
    with pytest.warns(NumericalTransformWarning, match="kernel compilation"):
        selected = scan.optimized((x,), (), (), count=0, payload_limits=(RegionPayloadLimit(device, 10000),),
            options=ExecutionOptions(schedule_policy="serial"), warmup_rounds=0, measurement_rounds=1)
    assert selected.selection.candidates[selected.selection.selected].changed_functions == (cell.digest,)
    def forbidden(*args, **kwargs):
        raise AssertionError("another loop count must not trigger compilation")
    monkeypatch.setattr(tuning, "compile_tensor_cell", forbidden)
    for count in (0, 1, 3):
        result = selected((x,), (), (), count=count)
        torch.testing.assert_close(result.carry[0], torch.full_like(x, 2 ** (count + 1) - 1), rtol=0, atol=0)
