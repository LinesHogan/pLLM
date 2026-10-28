"""Operator-based scan classification, independent recurrences, funding and replay."""

from unittest.mock import patch

import pytest
import torch

from pllm import scan
from pllm.contracts.tensor_policy import ScanForm, TensorPolicy
from pllm.execution.scan import ScanContext
from pllm.execution.scan_analysis import analyze
from pllm.execution.scan_execution import competitive, diagonal_prefix


def specifications(values):
    return tuple((t.shape, t.stride(), t.dtype, t.device) for t in values)


def additive(state, x, y):
    features = torch.nn.functional.silu(x)
    increment = y.T @ features
    return state + increment * 0.01, torch.tanh(state) @ features.T


def diagonal(state, x, y):
    gate = torch.sigmoid(x.sum(0)) * 1.2 - 0.2
    bias = y.sum(0).tanh()
    return gate * state + bias, torch.tanh(state) + x


def left(state, x, y):
    coefficient = x.T @ x * 0.01
    return coefficient @ state + y.T, torch.tanh(state)


@pytest.mark.parametrize(
    "step,shapes,kind",
    (
        (additive, ((3, 5), (4, 5), (4, 3)), "additive"),
        (diagonal, ((5,), (4, 5), (4, 5)), "diagonal"),
        (left, ((4, 4), (4, 4), (4, 4)), "left"),
    ),
)
def test_unseen_affine_coefficients_and_nonlinear_readout(step, shapes, kind):
    torch.manual_seed(501)
    values = tuple(torch.randn(shape, dtype=torch.float64) for shape in shapes)
    result = analyze(step, specifications(values))
    assert result is not None and result.kind == kind
    coefficient, bias, *cuts = result.coefficients(*values)
    expected, output = step(*values)
    state = values[0]
    actual = (
        state + result.scale * bias
        if kind == "additive"
        else coefficient @ state + result.scale * bias
        if kind == "left"
        else coefficient * state + result.scale * bias
    )
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(result.readout(state, *cuts), output, rtol=0, atol=0)


@pytest.mark.parametrize(
    "step",
    (
        lambda s, x: (s.tanh() + x, s),
        lambda s, x: (s * s + x, s),
        lambda s, x: (x / s, s),
        lambda s, x: (torch.where(s > 0, s + x, s - x), s),
        lambda s, x: (s.to(torch.bfloat16).float() + x, s),
        lambda s, x: (x @ s + s @ x, s),
        lambda s, x: ((s + x).sin() + s, s),
    ),
)
def test_dependent_refusals_propagate_to_returned_carry(step):
    values = (torch.randn(4, 4), torch.randn(4, 4))
    assert analyze(step, specifications(values)) is None


def test_equivalent_operator_order_and_matrix_guards():
    x, y = torch.randn(4, 4), torch.randn(4, 4)
    for step in (lambda s, x, y: (s + (x.T @ y) * 0.02, x @ s), lambda s, x, y: ((x.T @ y) * 0.02 + s, x @ s)):
        result = analyze(step, specifications((x.clone(), x, y)))
        assert result is not None and result.kind == "additive"
    for form in ScanForm:
        assert competitive(64, 64, 16, form)
        assert not competitive(896, 64, 64, form)
        assert competitive(896, 896, 16, form)
    assert not competitive(64, 64, 2, ScanForm.HILLIS_STEELE)
    assert competitive(64, 64, 8, ScanForm.HILLIS_STEELE)


@pytest.mark.parametrize("step,shape,chunk", ((additive, (3, 5), 4), (diagonal, (5,), 4), (left, (4, 4), 4)))
@pytest.mark.parametrize("count", (0, 1, 5, 8, 17))
def test_public_unseen_scan_against_fp64(step, shape, chunk, count):
    torch.manual_seed(13)
    state = torch.randn(shape, device="cuda") * 0.1
    width = shape[-1]
    x = torch.randn(count * chunk, width, device="cuda") * 0.1
    ywidth = shape[0] if step is additive else width
    y = torch.randn(count * chunk, ywidth, device="cuda") * 0.1
    expected, outputs = state.double(), []
    for xx, yy in zip(x.double().split(chunk), y.double().split(chunk), strict=True):
        if len(xx):
            expected, output = step(expected, xx, yy)
            outputs.append(output)
    context = ScanContext(TensorPolicy(), max(1, count * chunk), torch.device("cuda"), 128 << 20)
    actual, reads = context.bind(scan)(step, carry=state, over=(x, y), chunk=chunk)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(tuple(t.double() for t in reads), tuple(outputs), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("window", (1, 5, 64))
def test_diagonal_prefix_matches_independent_order(dtype, window):
    torch.manual_seed(81)
    state = torch.randn(7, dtype=dtype) * 0.1
    gates = (torch.rand(window, 7) * 1.5 - 0.5).to(dtype)
    biases = (torch.randn(window, 7) * 0.1).to(dtype)
    actual = diagonal_prefix(gates, biases, state)
    expected = []
    value = state.double()
    for gate, bias in zip(gates.double(), biases.double(), strict=True):
        value = gate * value + bias
        expected.append(value)
    tolerance = dict(rtol=0.03, atol=0.003) if dtype == torch.bfloat16 else dict(rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(actual.double(), torch.stack(expected), **tolerance)


def test_counted_diagonal_graph_masks_nonfinite_coefficients_and_keeps_readouts():
    def step(s, x, y):
        return x[0] * s + y[0], s.tanh() + x

    state = torch.ones(4, device="cuda") * 0.25
    x = torch.full((8, 4), 0.5, device="cuda")
    y = torch.full_like(x, 0.1)
    count = torch.tensor(8, device="cuda")
    context = ScanContext(TensorPolicy(), 8, state.device, 128 << 20)
    run = context.bind(scan)
    run(step, carry=state, over=(x, y), chunk=1, valid_rows=count)
    with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
        actual, outputs = run(step, carry=state, over=(x, y), chunk=1, valid_rows=count)
    for valid in (0, 1, 5, 8, 3):
        x.fill_(0.5)
        y.fill_(0.1)
        x[valid:].fill_(float("nan"))
        y[valid:].fill_(float("inf"))
        count.fill_(valid)
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("device read")):
            graph.replay()
        expected = state.clone()
        reads = []
        for index in range(8):
            updated, out = step(expected, x[index : index + 1], y[index : index + 1])
            if index < valid:
                expected = updated
            reads.append(out)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(outputs, tuple(reads), rtol=1e-5, atol=1e-6, equal_nan=True)


def test_budget_boundary_and_two_graph_slots():
    state = torch.zeros(3, 5, device="cuda")
    x = torch.ones(32, 5, device="cuda")
    y = torch.ones(32, 3, device="cuda")
    context = ScanContext(TensorPolicy(), 32, state.device)
    specs = specifications((state, x[:4], y[:4]))
    plan = context.specialize(additive, specs, 0, 4, 8)
    assert plan.analysis is not None
    args = (state, (x, y), (), None)
    plan(*args)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = plan(*args)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - baseline <= plan.workspace
    graphs = []
    for _ in range(2):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            plan(*args)
        with torch.cuda.graph(graph := torch.cuda.CUDAGraph(), stream=stream):
            result = plan(*args)
        graphs.append((graph, result, stream))
    assert graphs[0][1][0].data_ptr() != graphs[1][1][0].data_ptr()
    completions = []
    for graph, _, stream in graphs:
        with torch.cuda.stream(stream):
            graph.replay()
            completions.append(stream.record_event())
    for (_, result, _), completion in zip(graphs, completions, strict=True):
        completion.synchronize()
        torch.testing.assert_close(result, output)
    strict = ScanContext(TensorPolicy(False), 32, state.device)
    bound = strict.specialize(additive, specs, 0, 4, 8).workspace
    with pytest.raises(ValueError, match="budget"):
        ScanContext(TensorPolicy(False), 32, state.device, bound - 1).specialize(additive, specs, 0, 4, 8)
    assert (
        ScanContext(TensorPolicy(False), 32, state.device, bound).specialize(additive, specs, 0, 4, 8).workspace
        == bound
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("window", (1, 5, 64))
def test_pointwise_higher_order_and_fallback_share_the_fp64_oracle(dtype, window):
    from pllm.execution.scan_execution import associative_prefix

    def prefix(a, b, initial, live):
        mask = live[:, None]
        return associative_prefix(torch.where(mask, a, 1), torch.where(mask, b, 0), initial)

    torch.manual_seed(707)
    scale = torch.tensor([1.0, 1e20, 1e-30, 1e-40], device="cuda")
    a = (torch.rand(window, 4, device="cuda") * 1.5 - 0.5).to(dtype)
    a[0, 0] = 0
    b = (torch.randn(window, 4, device="cuda") * scale).to(dtype)
    initial = (torch.ones(4, device="cuda") * scale * 0.1).to(dtype)
    live = torch.ones(window, device="cuda", dtype=torch.bool)
    compiled = torch.compile(prefix, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    compiled(a, b, initial, live)
    with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
        actual = compiled(a, b, initial, live)
    normalizer = scale.double().clamp_min(torch.finfo(dtype).tiny)
    for count in dict.fromkeys((0, 1, min(3, window), window)):
        live.copy_(torch.arange(window, device="cuda") < count)
        graph.replay()
        reference = initial.double()
        expected = []
        for index in range(window):
            if index < count:
                reference = reference * a[index].double() + b[index].double()
            expected.append(reference)
        expected = torch.stack(expected)
        fallback = diagonal_prefix(torch.where(live[:, None], a, 1), torch.where(live[:, None], b, 0), initial)
        tolerance = dict(rtol=0.03, atol=0.015) if dtype == torch.bfloat16 else dict(rtol=1e-5, atol=1e-6)
        for result in (actual, fallback):
            assert torch.equal(torch.isfinite(result), torch.isfinite(expected))
            torch.testing.assert_close(result.double() / normalizer, expected / normalizer, **tolerance)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_source_aliases_autograd_and_tensor_invariants(device):
    def step(s, x, w):
        return s + x.sum(0) @ w, x + s

    state = torch.ones(3, device=device, requires_grad=True)
    x = torch.ones(7, 3, device=device, requires_grad=True)
    weight = torch.eye(3, device=device, requires_grad=True)
    with patch("pllm.execution.scan.analyze", side_effect=AssertionError("autograd entered optimization")):
        result, outputs = scan(step, carry=state, over=(x,), invariants=(weight,), chunk=2)
    loss = result.sum() + sum(output.sum() for output in outputs)
    gradients = torch.autograd.grad(loss, (state, x, weight))
    reference, reads = state, []
    for rows in x.split(2):
        reference, output = step(reference, rows, weight)
        reads.append(output)
    expected = torch.autograd.grad(reference.sum() + sum(output.sum() for output in reads), (state, x, weight))
    torch.testing.assert_close(gradients, expected, rtol=0, atol=0)


def test_transpose_permute_and_identity_views_classify_equally():
    values = (torch.randn(2, 4, 4), torch.randn(2, 4, 4))
    for step in (
        lambda s, x: ((x @ s.transpose(1, 2)).transpose(1, 2).contiguous(), s),
        lambda renamed, data: (
            (data @ renamed.view_as(renamed).permute(0, 2, 1)).permute(0, 2, 1).contiguous(),
            renamed,
        ),
    ):
        result = analyze(step, specifications(values))
        assert result is not None and result.kind == "right"


def test_left_action_restores_rectangular_readout_layout():
    def step(s, x, y):
        return (x.T @ x * 0.1) @ s + y, s.view(-1).tanh()

    state = torch.randn(2, 3, device="cuda") * 0.1
    x = torch.randn(10, 2, device="cuda") * 0.1
    y = torch.randn(10, 3, device="cuda") * 0.1
    expected, reads = state.double(), []
    for xx, yy in zip(x.double().split(2), y.double().split(2), strict=True):
        expected, out = step(expected, xx, yy)
        reads.append(out)
    actual, outputs = scan(step, carry=state, over=(x, y), chunk=2)
    assert actual.stride() == state.stride()
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(tuple(t.double() for t in outputs), tuple(reads), rtol=2e-5, atol=2e-6)


def test_shape_tracing_retains_owned_prefix_outputs_when_source_returns_aliases():
    from pllm.execution.torch_private import measure

    def step(s, x):
        return s + x[0], x

    state = torch.empty(4, device="meta")
    x = torch.empty(8, 4, device="meta")
    run = ScanContext(TensorPolicy(), 8, torch.device("cuda")).bind(scan)
    usage = measure(lambda: run(step, carry=state, over=(x,), chunk=1))
    assert usage.output >= 9 * 512


@pytest.mark.parametrize("kind", ("additive", "right"))
def test_counted_prefix_masks_nan_inf_and_keeps_exclusive_readouts(kind):
    def step(s, x, y):
        return (s + y if kind == "additive" else s @ x + y), s

    state = torch.full((2, 2), 0.25, device="cuda")
    x = torch.eye(2, device="cuda").repeat(8, 1)
    y = torch.full_like(x, 0.1)
    valid = torch.tensor(16, device="cuda")
    run = ScanContext(TensorPolicy(), 16, state.device, 128 << 20).bind(scan)
    run(step, carry=state, over=(x, y), chunk=2, valid_rows=valid)
    with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
        actual, outputs = run(step, carry=state, over=(x, y), chunk=2, valid_rows=valid)
    for rows in (0, 2, 10, 16, 6):
        x.copy_(torch.eye(2, device="cuda").repeat(8, 1))
        y.fill_(0.1)
        x[rows:].fill_(float("nan"))
        y[rows:].fill_(float("inf"))
        valid.fill_(rows)
        graph.replay()
        expected, reads = state, []
        for index, (xx, yy) in enumerate(zip(x.split(2), y.split(2), strict=True)):
            updated, out = step(expected, xx, yy)
            if (index + 1) * 2 <= rows:
                expected = updated
            reads.append(out)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(outputs, tuple(reads), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_source_preserves_gradients_of_closed_parameters(device):
    weight = torch.nn.Parameter(torch.eye(3, device=device))
    state = torch.zeros(3, 3, device=device)
    rows = torch.ones(9, 3, device=device)

    def step(s, x):
        return s + x @ weight, s

    actual, outputs = scan(step, carry=state, over=(rows,), chunk=3)
    gradient = torch.autograd.grad(actual.sum() + sum(t.sum() for t in outputs), weight)
    expected, reads = state, []
    for x in rows.split(3):
        expected, output = step(expected, x)
        reads.append(output)
    wanted = torch.autograd.grad(expected.sum() + sum(t.sum() for t in reads), weight)
    torch.testing.assert_close(gradient, wanted, rtol=0, atol=0)
    with torch.no_grad():
        assert analyze(step, specifications((state, rows[:3]))) is None
        assert analyze(lambda s, x, w: (s + x @ w, s), specifications((state, rows[:3], weight))) is not None


def test_prefix_outputs_own_disjoint_values_even_when_readout_returns_an_input():
    def step(s, x):
        return s + x[0], x

    initial = torch.zeros(4, device="cuda")
    rows = torch.ones(20, 4, device="cuda")
    final, outputs = scan(step, carry=initial, over=(rows,), chunk=1)
    outputs[0].fill_(17)
    assert torch.equal(rows, torch.ones_like(rows))
    assert all(torch.equal(output, torch.ones_like(output)) for output in outputs[1:])
    assert (
        sum(storage.nbytes() for storage in {output.untyped_storage() for output in outputs})
        == rows.numel() * rows.element_size()
    )
    final.fill_(31)
    assert not initial.any()


def test_scan_funding_accounts_for_live_caller_temporaries():
    from pllm.execution.torch_private import measure

    def step(s, x):
        return s + x[0], x + s

    specs = specifications((torch.empty(4, device="meta"), torch.empty(1, 4, device="meta")))
    policy = TensorPolicy(False)
    device = torch.device("cuda")
    bound = ScanContext(policy, 8, device).specialize(step, specs, 0, 1, 8).workspace

    def prepare(budget):
        live = torch.empty(16, device="meta")
        plan = ScanContext(policy, 8, device, budget).specialize(step, specs, 0, 1, 8)
        return live, plan

    with pytest.raises(ValueError, match="budget"):
        measure(lambda: prepare(bound + 511))
    measure(lambda: prepare(bound + 512))
