"""Kernel compilation: generated backend code behaves like the source graph.

``compile_tensor_cell`` produces Inductor/Triton kernels that must match the
source, borrow read-only inputs without hidden copies, return independently
owned outputs and stay inside the graph-derived capacity bound. Selection
compiles at most once; selected executions never compile again, and the
numerical opt-out or a declining backend keeps the original path.
"""

from dataclasses import replace

import pytest
import torch

from pllm.backends.torch_inductor import InductorTensorCompiler, compile_inductor_tensor_graph
from pllm.config import ExecutionOptions
from pllm.contracts.region_program import Call, Program, SizeParameter, Value
from pllm.contracts.region_selection import RegionPayloadLimit
from pllm.contracts.tensor_compilation import UnsupportedTensorKernel
from pllm.execution.exported_cell import ExportedCell, NumericalTransformWarning
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionInstructionExecution
from pllm.execution.region_native import CompiledPrimitive
from pllm.execution.region_optimization import optimize_region
from pllm.execution.region_storage import analyze_region_capacity
from pllm.execution.tensor_compilation import compile_tensor_cell, supports_tensor_compilation
from pllm.execution.tensor_scan import TensorScan
from test.region_fixtures import region, single_call

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
DEVICES = ("cpu", pytest.param("cuda:0", marks=requires_cuda))


def arithmetic_cell(device="cpu", dtype=torch.float32, *, duplicate=False):
    x, y = torch.full((128,), 2, device=device, dtype=dtype), torch.ones(128, device=device, dtype=dtype)

    def arithmetic(x, y):
        return ((x + y) * (x - y) + x,)

    def update(x, y):
        x.add_(1)
        return x, x, y

    if duplicate:
        return ExportedCell.capture(update, ("x", "y"), ("new", "duplicate", "old"), (x, y)), (x, y)
    return ExportedCell.capture(arithmetic, ("x", "y"), ("out",), (x, y)), (x, y)


def prefer_compiled(monkeypatch, digest=None):
    """Deterministic timing: any candidate with a compiled kernel wins."""
    import pllm.execution.region_optimization as tuning

    def measure(execution, *args):
        cells = execution.cells.values() if digest is None else (execution.cells[digest],)
        return 1 if any(isinstance(cell, CompiledPrimitive) for cell in cells) else 100

    monkeypatch.setattr(tuning, "_measure", measure)


def forbid_compilation(monkeypatch, device) -> None:
    import torch._inductor.compile_fx as inductor

    import pllm.execution.region_optimization as tuning

    def forbidden(*args, **kwargs):
        raise AssertionError("selected execution must not compile")

    monkeypatch.setattr(tuning, "compile_tensor_cell", forbidden)
    monkeypatch.setattr(inductor, "compile_fx_inner", forbidden)
    if device.startswith("cuda"):
        import triton
        from triton.runtime.jit import JITFunction

        monkeypatch.setattr(triton, "compile", forbidden)
        monkeypatch.setattr(JITFunction, "_do_compile", forbidden)


def bind(program, cells, schedule):
    """Bind already-transformed cells directly, as selection installs them."""
    return RegionInstructionExecution(
        program, lower_region_schedule(program, schedule), lower_region_bindings(program, schedule), cells
    )


SERIAL = replace(ExecutionOptions(schedule_policy="serial"), tuning="measured")


@pytest.mark.parametrize(
    "device,dtype", (("cpu", torch.float32), pytest.param("cuda:0", torch.bfloat16, marks=requires_cuda))
)
def test_generated_kernel_matches_source_and_preserves_inputs_and_rng(device, dtype):
    source, tensors = arithmetic_cell(device, dtype)
    graph = str(source.graph.graph)
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.startswith("cuda") else None
    compiled = compile_tensor_cell(source)
    (actual,) = compiled.invoke(tensors, None)
    torch.testing.assert_close(actual, torch.full_like(actual, 5), rtol=0, atol=0)
    torch.testing.assert_close(tensors[0], torch.full_like(tensors[0], 2), rtol=0, atol=0)
    torch.testing.assert_close(tensors[1], torch.ones_like(tensors[1]), rtol=0, atol=0)
    assert str(source.graph.graph) == graph
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    # A view the compiler eliminates is still returned as independent storage.
    x = torch.arange(128, dtype=torch.float32, device=device)
    (view,) = compile_tensor_cell(ExportedCell.capture(lambda x: (x.view(128),), ("x",), ("out",), (x,))).invoke(
        (x,), None
    )
    torch.testing.assert_close(view, x, rtol=0, atol=0)
    assert view.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()


@pytest.mark.parametrize("device", DEVICES)
def test_selection_compiles_once_and_selected_execution_never_compiles(device, monkeypatch):
    source, tensors = arithmetic_cell(device)
    prefer_compiled(monkeypatch)
    calls = []

    class RecordingCompiler:
        def supports(self, graph):
            calls.append("supports")
            return InductorTensorCompiler().supports(graph)

        def compile(self, graph):
            calls.append("compile")
            code = compile_inductor_tensor_graph(graph)
            # A distinct provider identity exposes accidental default-backend reconstruction.
            return replace(code, backend="recording-provider:" + code.backend)

    inputs = dict(zip(("in0", "in1"), tensors))
    with pytest.warns(NumericalTransformWarning, match="kernel compilation"):
        selected = optimize_region(
            single_call(source),
            {source.digest: source},
            inputs,
            payload_limits=(RegionPayloadLimit(device, 10000),),
            options=SERIAL,
            warmup_rounds=0,
            measurement_rounds=1,
            compiler=RecordingCompiler(),
        )
    assert selected.selection.selected == 1 and calls.count("compile") == 1
    # A scan tuned at length zero still selects (and compiles) its kernel for later lengths.
    x = torch.ones(128, device=device)
    cell = ExportedCell.capture(lambda x: (x * 2 + 1,), ("x",), ("out",), (x,))
    scan = TensorScan(cell, SizeParameter("steps", 0, 1_000_000), carry=1, invariants=0)
    with pytest.warns(NumericalTransformWarning, match="kernel compilation"):
        tuned, report = scan.optimized(
            (x,),
            (),
            (),
            count=0,
            payload_limits=(RegionPayloadLimit(device, 10000),),
            options=SERIAL,
            warmup_rounds=0,
            measurement_rounds=1,
        )
    assert report.selection.candidates[report.selection.selected].changed_functions == (cell.digest,)
    calls.clear()
    forbid_compilation(monkeypatch, device)
    (actual,) = selected.execution.run(inputs).values
    torch.testing.assert_close(actual, torch.full_like(actual, 5), rtol=0, atol=0)
    assert not calls
    for count in (0, 1, 3):
        result = tuned((x,), (), (), count=count)
        torch.testing.assert_close(result.carry[0], torch.full_like(x, 2 ** (count + 1) - 1), rtol=0, atol=0)


def test_unsupported_declined_and_opted_out_kernels_keep_the_source_path(monkeypatch):
    import pllm.execution.region_optimization as tuning

    x = torch.ones(4, 4)
    matrix = ExportedCell.capture(lambda x, y: (torch.mm(x, y),), ("x", "y"), ("out",), (x, x.clone()))
    transposed = ExportedCell.capture(lambda x: ((x + 1).T,), ("x",), ("out",), (x,))
    for source in (matrix, transposed):  # External workspaces and layouts are not assigned zero cost.
        assert not supports_tensor_compilation(source)
        with pytest.raises(UnsupportedTensorKernel):
            compile_tensor_cell(source)
    with pytest.raises(UnsupportedTensorKernel, match="CUDA"):
        compile_inductor_tensor_graph(matrix.graph)

    class DecliningCompiler:
        def supports(self, graph):
            return False

        def compile(self, graph):
            raise AssertionError("a declined implementation must not be compiled")

    source, tensors = arithmetic_cell()
    program, cells, inputs = single_call(source), {source.digest: source}, dict(zip(("in0", "in1"), tensors))
    common = dict(payload_limits=(RegionPayloadLimit("cpu", 10000),), warmup_rounds=0, measurement_rounds=1)
    declined = optimize_region(program, cells, inputs, compiler=DecliningCompiler(), options=SERIAL, **common)
    assert len(declined.request.candidates) == 1
    torch.testing.assert_close(declined.execution.run(inputs).values, (torch.full_like(tensors[0], 5),), rtol=0, atol=0)
    with monkeypatch.context() as patch:
        patch.setattr(tuning, "compile_tensor_cell", lambda *a, **k: pytest.fail("numerical opt-out compiled"))
        opted_out = optimize_region(
            program, cells, inputs, options=replace(SERIAL, allow_numerical_transforms=False), **common
        )
    assert len(opted_out.request.candidates) == 1
    # A faster compiled kernel that needs more than the source budget is never measured.
    budget = analyze_region_capacity(program, analyze_region_program(program).reference_schedule, cells)
    monkeypatch.setattr(tuning, "_measure", lambda execution, *args: 100)
    limited = optimize_region(
        program,
        cells,
        inputs,
        options=SERIAL,
        payload_limits=(RegionPayloadLimit("cpu", budget.devices[0].peak_tensor_bytes_upper_bound),),
        warmup_rounds=0,
        measurement_rounds=1,
    )
    assert limited.selection.selected == 0 and not limited.selection.candidates[1].feasible
    assert limited.request.candidates[1].elapsed_ns == ()


@pytest.mark.parametrize("device", DEVICES)
def test_reduction_kernel_matches_a_float64_oracle_with_readonly_inputs(device, monkeypatch):
    # Non-power-of-two shapes, two reduction directions, a scalar output and widened accumulation.
    base = torch.linspace(-2, 3, 17 * 129 + 1, device=device)
    x = base[1:].view(17, 129)
    before = base.clone()

    def normalize(x):
        mean_square = (x * x).mean(dim=1, keepdim=True)
        return x * torch.rsqrt(mean_square + 1e-5), x.sum(dim=0), x.sum(dtype=torch.float64)

    source = ExportedCell.capture(normalize, ("x",), ("normalized", "columns", "total"), (x,))
    prefer_compiled(monkeypatch, source.digest)
    with pytest.warns(NumericalTransformWarning, match="kernel compilation"):
        selected = optimize_region(
            single_call(source),
            {source.digest: source},
            {"in0": x},
            payload_limits=(RegionPayloadLimit(device, 1_000_000),),
            options=SERIAL,
            max_candidates=2,
            warmup_rounds=0,
            measurement_rounds=1,
        )
    assert isinstance(selected.execution.cells[source.digest], CompiledPrimitive)
    forbid_compilation(monkeypatch, device)
    actual = selected.execution.run({"in0": x}).values
    reference = x.double()
    expected = (
        (reference / torch.sqrt((reference * reference).sum(dim=1, keepdim=True) / 129 + 1e-5)).float(),
        reference.sum(dim=0).float(),
        reference.sum(),
    )
    for result, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, oracle)
    torch.testing.assert_close(base, before, rtol=0, atol=0)
    assert len({value.untyped_storage().data_ptr() for value in (*actual, base)}) == 4


@pytest.mark.parametrize("device", DEVICES)
def test_kernels_borrow_unaligned_overlapping_inputs_without_hidden_copies(device):
    source, _ = arithmetic_cell(device, duplicate=True)
    compiled = compile_tensor_cell(source)
    base = torch.arange(130, dtype=torch.float32, device=device)
    x, y = base[1:129], base[2:130]
    assert x.data_ptr() % 16 == 4 and y.data_ptr() % 16 == 8
    before = base.clone()
    observed = []

    def record(*inputs):
        observed.append(tuple(t.data_ptr() for t in inputs))
        return compiled.function(*inputs)

    actual = replace(compiled, function=record).invoke((x, y), None)
    assert observed == [(x.data_ptr(), y.data_ptr())]  # The wrapper receives the views themselves.
    for value, reference in zip(actual, (x + 1, x + 1, y.clone()), strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    torch.testing.assert_close(base, before, rtol=0, atol=0)
    assert len({value.untyped_storage().data_ptr() for value in (*actual, base)}) == 4
    if device.startswith("cuda"):
        from torch._inductor import config

        # Read-only codegen overrides, then restores, a global alignment assumption.
        with config.patch(assume_aligned_inputs=True):
            aligned = compile_tensor_cell(arithmetic_cell(device)[0])
            assert config.assume_aligned_inputs
        ones = torch.ones(129, device=device)
        (result,) = aligned.invoke((ones[1:], ones[1:]), None)
        torch.testing.assert_close(result, torch.ones_like(result), rtol=0, atol=0)


def test_compiled_kernel_keeps_region_view_origins_and_old_versions():
    base = torch.arange(129, dtype=torch.float32)
    view = ExportedCell.capture(lambda base: (base[1:],), ("base",), ("view",), (base,))
    update, _ = arithmetic_cell(duplicate=True)
    compiled = compile_tensor_cell(update)
    observed = []

    def record(*inputs):
        observed.append(tuple((t.storage_offset(), t.data_ptr()) for t in inputs))
        return compiled.function(*inputs)

    (t,) = view.declaration.outputs
    program = Program(
        region(
            (Value("base", view.declaration.inputs[0]),),
            (
                Call(("base",), (Value("view", t),), view.digest),
                Call(("view", "view"), (Value("new", t), Value("again", t), Value("old", t)), update.digest),
            ),
            ("base", "view", "new", "again", "old"),
        ),
        (view.declaration, update.declaration),
    )
    cells = {view.digest: view, update.digest: replace(compiled, function=record)}
    old_base, old_view, new, again, old = (
        bind(program, cells, analyze_region_program(program).reference_schedule).run({"base": base}).values
    )
    assert old_view.untyped_storage().data_ptr() == old_base.untyped_storage().data_ptr()
    assert observed == [((1, old_view.data_ptr()), (1, old_view.data_ptr()))]
    for value in (old_view, old):
        torch.testing.assert_close(value, base[1:], rtol=0, atol=0)
    for value in (new, again):
        torch.testing.assert_close(value, base[1:] + 1, rtol=0, atol=0)


@requires_cuda
def test_capacity_bound_covers_the_measured_cuda_peak_including_output_copies():
    source, tensors = arithmetic_cell("cuda:0", duplicate=True)
    compiled = compile_tensor_cell(source)
    program = single_call(source)
    schedule = analyze_region_program(program).reference_schedule
    capacity = analyze_region_capacity(program, schedule, {source.digest: compiled})
    # Kernel traffic is unknown: wrapper copies do not describe global/cache transactions.
    assert capacity.unknown_functions == (source.digest,)
    run = bind(program, {source.digest: compiled}, schedule)
    inputs = dict(zip(("in0", "in1"), tensors))
    warm = run.run(inputs)  # Warm allocation and module loading.
    del warm
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    new, repeated, old = run.run(inputs).values
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - before <= capacity.devices[0].peak_tensor_bytes_upper_bound
    torch.testing.assert_close(new, torch.full_like(new, 3), rtol=0, atol=0)
    torch.testing.assert_close(repeated, new, rtol=0, atol=0)
    torch.testing.assert_close(old, tensors[1], rtol=0, atol=0)
    assert len({t.untyped_storage().data_ptr() for t in (new, repeated, old)}) == 3


def test_outlined_calls_can_be_selected_as_one_generated_kernel(monkeypatch):
    import pllm.execution.region_optimization as tuning

    x, y = torch.full((128,), 2.0), torch.ones(128)
    first = ExportedCell.capture(lambda x, y: (x + y, x - y), ("x", "y"), ("sum", "diff"), (x, y))
    second = ExportedCell.capture(lambda s, d, x: (s * d + x,), ("sum", "diff", "x"), ("out",), (x + y, x - y, x))
    (t,) = second.declaration.outputs
    program = Program(
        region(
            (Value("x", t), Value("y", t)),
            (
                Call(("x", "y"), (Value("sum", t), Value("diff", t)), first.digest),
                Call(("sum", "diff", "x"), (Value("out", t),), second.digest),
            ),
            ("out",),
        ),
        (first.declaration, second.declaration),
    )
    sources = {first.digest: first, second.digest: second}
    monkeypatch.setattr(
        tuning,
        "_measure",
        lambda execution, *args: (
            1
            if any(isinstance(c, CompiledPrimitive) and name not in sources for name, c in execution.cells.items())
            else 100
        ),
    )
    with pytest.warns(NumericalTransformWarning, match="kernel compilation"):
        selected = optimize_region(
            program,
            sources,
            {"x": x, "y": y},
            payload_limits=(RegionPayloadLimit("cpu", 10000),),
            options=SERIAL,
            warmup_rounds=0,
            measurement_rounds=1,
        )
    assert len(selected.request.candidates[selected.selection.selected].outlines) == 1
    result = selected.execution.run({"x": x, "y": y})
    assert result.invocations == 1
    torch.testing.assert_close(result.values[0], torch.full_like(x, 5), rtol=0, atol=0)


@requires_cuda
def test_generated_gemm_kernels_match_eager_with_readonly_strided_operands():
    def unaligned(shape, modulus):
        count = 1 + torch.Size(shape).numel()
        base = torch.arange(count, device="cuda", dtype=torch.float32).remainder(modulus).to(torch.bfloat16)
        return base, base[1:].reshape(shape)

    for leading in ((), (2,)):  # Epilogue fused into one kernel, plain and batched.
        (xb, x), (wb, w) = unaligned((*leading, 32, 64), 7), unaligned((*leading, 64, 48), 7)
        saved = xb.clone(), wb.clone()
        source = ExportedCell.capture(lambda x, w: ((x @ w) + 1,), ("x", "w"), ("out",), (x, w))
        compiled = compile_tensor_cell(source)
        (result,) = compiled.invoke((x, w), None)
        torch.testing.assert_close(result, (x.float() @ w.float()).to(torch.bfloat16) + 1)
        torch.testing.assert_close(xb, saved[0], rtol=0, atol=0)
        torch.testing.assert_close(wb, saved[1], rtol=0, atol=0)
        assert compile_tensor_cell(source).implementation == compiled.implementation  # Replays without timings.
    x = torch.arange(32 * 64, device="cuda", dtype=torch.float32).remainder(7).reshape(32, 64).to(torch.bfloat16)
    base = torch.arange(1 + 48 * 64, device="cuda", dtype=torch.float32).remainder(5).to(torch.bfloat16)
    weight = base[1:].view(48, 64).T  # A transposed, offset weight keeps its strides.
    source = ExportedCell.capture(lambda x, w: (x @ w,), ("x", "weight"), ("out",), (x, weight))
    (result,) = compile_tensor_cell(source).invoke((x, weight), None)
    torch.testing.assert_close(result, (x.float() @ weight.float()).to(torch.bfloat16))


@requires_cuda
def test_fused_bfloat16_recurrence_keeps_intermediate_rounding():
    generator = torch.Generator(device="cuda").manual_seed(127)
    base = torch.randn((7, 5), dtype=torch.bfloat16, device="cuda", generator=generator).T
    delta = torch.zeros((5, 7), dtype=base.dtype, device=base.device)
    x = torch.randn((8, 7), dtype=base.dtype, device=base.device, generator=generator)
    y = torch.randn((8, 5), dtype=base.dtype, device=base.device, generator=generator)

    def step(delta, base, x, y):
        return delta + 0.017 * (y.T @ x), x @ (base + delta).T

    source = ExportedCell.capture(step, ("delta", "base", "x", "y"), ("next", "out"), (delta, base, x[:3], y[:3]))
    compiled = compile_tensor_cell(source)
    actual, expected = delta, delta
    # The second update exposes rounding lost by fusing the bf16 multiply into the carry addition.
    for start in (0, 3):
        actual, output = compiled.invoke((actual, base, x[start : start + 3], y[start : start + 3]), None)
        expected, reference = step(expected, base, x[start : start + 3], y[start : start + 3])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
