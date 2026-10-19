"""Tensor graph rewrites, their search, and the memory profiles that rank them.

Rewrites (matmul reassociation, independent-product batching) are generated
cold from the graph, never by running user math, and only where they are
legal. Each rewrite replays from its evidence record, keeps the functional ABI
and matches the source numerically. ``profile_tensor_memory`` must describe
the storage the unfused graph really uses.
"""

import warnings
from dataclasses import replace

import pytest
import torch

from pllm.config import ExecutionOptions
from pllm.contracts.region_program import Call, Program, Value
from pllm.contracts.region_selection import RegionPayloadLimit
from pllm.contracts.tensor_rewrite import MatmulAssociation, MatmulBatching, MatmulReassociation
from pllm.execution.codec import decode_boundary, encode_boundary, tensor_graph_expression_digest
from pllm.execution.exported_cell import ExportedCell, NumericalTransformWarning
from pllm.execution.region_optimization import optimize_region, select_region_candidate
from pllm.execution.tensor_access import profile_tensor_memory, validate_tensor_memory
from pllm.execution.tensor_batching import batching_steps
from pllm.execution.tensor_rewrites import TensorRewritePath, explore_tensor_combinations, replay_tensor_rewrites
from test.region_fixtures import StorageObserver, reassociation_candidates, region, single_call


def left_chain(a, b, c):
    return ((a @ b) @ c,)


def right_chain(a, b, c):
    return (a @ (b @ c),)


def operands(dimensions=(12, 2, 16, 3), *, device="cpu", dtype=torch.float64):
    m, k, n, p = dimensions
    generator = torch.Generator().manual_seed(183)
    return tuple(
        (torch.rand(shape, generator=generator, dtype=torch.float64) * 0.5 - 0.25).to(device, dtype)
        for shape in ((m, k), (k, n), (n, p))
    )


def capture(function=left_chain, inputs=None, outputs=("result",)):
    return ExportedCell.capture(function, ("a", "b", "c"), outputs, inputs or operands())


def reassociation_paths(source, **kwargs):
    return tuple(
        TensorRewritePath(replay_tensor_rewrites(source, rs), rs)
        for rs in explore_tensor_combinations((source,), **kwargs)
        if all(isinstance(r, MatmulReassociation) for r in rs)
    )


def four_chain(device="cpu", balanced=False):
    tensors = tuple(torch.ones(shape, dtype=torch.float64, device=device) for shape in ((4, 2), (2, 8), (8, 4), (4, 2)))

    def chain(a, b, c, d):
        return (((a @ b) @ c) @ d,)

    def tree(a, b, c, d):
        return ((a @ b) @ (c @ d),)

    return ExportedCell.capture(tree if balanced else chain, ("a", "b", "c", "d"), ("out",), tensors), tensors


EXACT, CLOSE64, CLOSE32 = (0, 0), (1e-12, 1e-12), (1e-5, 1e-6)


CPU_CASES = (
    (left_chain, lambda: operands(), CLOSE64, MatmulAssociation.LEFT_TO_RIGHT),
    (right_chain, lambda: operands(), CLOSE64, MatmulAssociation.RIGHT_TO_LEFT),
    (left_chain, lambda: tuple(t.T.contiguous().T for t in operands()), CLOSE64, None),  # Transposed operands.
    (left_chain, lambda: operands((3, 0, 4, 2)), EXACT, None),  # Empty reduction keeps output shape.
    (
        left_chain,
        lambda: tuple(torch.full(s, 0.125, dtype=torch.bfloat16) for s in ((4, 2), (2, 8), (8, 2))),
        EXACT,
        None,
    ),
)
CUDA_CASES = (
    (left_chain, lambda: operands(device="cuda", dtype=torch.float32), CLOSE32, MatmulAssociation.LEFT_TO_RIGHT),
)


@pytest.mark.parametrize(
    "cases",
    (
        pytest.param(CPU_CASES, id="cpu"),
        pytest.param(
            CUDA_CASES, id="cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
        ),
    ),
)
def test_reassociated_graph_matches_the_source_and_replays_from_evidence(cases):
    for function, make_inputs, tolerance, association in cases:
        inputs = make_inputs()
        saved = tuple(t.clone() for t in inputs)
        source = capture(function, inputs)
        graph = str(source.graph.graph)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            (candidate,) = reassociation_candidates(source)
        assert not caught  # Generating a numerical candidate does not select it.
        if association is not None:
            assert candidate.rewrite.association is association
        assert candidate.cell.declaration == source.declaration
        assert source.transformations == () and str(source.graph.graph) == graph
        expected = function(*saved)[0]
        actual = candidate.cell(*inputs)[0]
        torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])
        for current, original in zip(inputs, saved, strict=True):
            assert torch.equal(current, original)
        assert torch.equal(source(*inputs)[0], expected)  # The source stays the numerical reference.
        record = decode_boundary(encode_boundary(candidate.rewrite), MatmulReassociation)
        assert torch.equal(replay_tensor_rewrites(source, (record,))(*inputs)[0], actual)


def test_illegal_or_unsupported_sites_produce_no_candidates():
    def returned(a, b, c):
        inner = a @ b
        return inner @ c, inner

    def fanout(a, b, c):
        inner = a @ b
        return inner @ c, inner.sum()

    for function in (returned, fanout):  # Shared or returned intermediates are never erased.
        assert reassociation_candidates(capture(function, outputs=("final", "observed"))) == ()
    for dtype in (torch.int64, torch.complex64):  # Only real floating domains.
        assert reassociation_candidates(capture(inputs=tuple(t.to(dtype) for t in operands()))) == ()
    batched = tuple(t.unsqueeze(0).repeat(2, 1, 1) for t in operands())
    assert reassociation_candidates(capture(inputs=batched)) == ()

    def square(a, b):
        x = a @ b
        return (x @ x,)  # Two uses by one node are not a private intermediate.

    assert (
        reassociation_candidates(
            ExportedCell.capture(square, ("a", "b"), ("out",), (torch.ones(3, 2), torch.ones(2, 3)))
        )
        == ()
    )


def test_rewrites_keep_mutations_output_positions_guards_and_returned_views():
    def update(a, b, c, state):
        state.copy_(torch.mm(torch.mm(a, b), c))
        return state * 2, state, a + 1

    inputs = (*operands(), torch.zeros(12, 3, dtype=torch.float64))
    saved = tuple(t.clone() for t in inputs)
    source = ExportedCell.capture(update, ("a", "b", "c", "state"), ("twice", "new_state", "other"), inputs)
    (candidate,) = reassociation_candidates(source)
    assert candidate.cell.output_positions == source.output_positions == (1, 2, 3)
    assert candidate.cell.mutations == source.mutations == (("state", "new_state"),)
    for actual, expected in zip(candidate.cell(*inputs), update(*saved), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(inputs[-1]) == 0
    transposed = tuple(t.T.contiguous().T for t in operands())
    (candidate,) = reassociation_candidates(capture(inputs=transposed))
    with pytest.raises(ValueError, match="guard"):
        candidate.cell(*(t.contiguous() for t in transposed))
    with pytest.raises(ValueError, match="guard"):
        candidate.cell(*operands((12, 2, 16, 4)))
    # A two-step path keeps a returned view aliasing its updated state.
    tensors = (*four_chain()[1], torch.zeros(4, 2, dtype=torch.float64))

    def chain_update(a, b, c, d, state):
        state.copy_(((a @ b) @ c) @ d)
        return state, state[:1]

    chained = ExportedCell.capture(chain_update, ("a", "b", "c", "d", "state"), ("new", "view"), tensors)
    path = next(p for p in reassociation_paths(chained) if len(p.rewrites) == 2)
    result, view = replay_tensor_rewrites(chained, path.rewrites)(*tensors)
    torch.testing.assert_close(result, tensors[0] @ tensors[1] @ tensors[2] @ tensors[3], rtol=0, atol=0)
    assert result.untyped_storage().data_ptr() == view.untyped_storage().data_ptr()
    assert torch.count_nonzero(tensors[-1]) == 0


@pytest.mark.parametrize("balanced", (False, True))
def test_search_reaches_every_parenthesization_without_running_user_math(balanced):
    source, tensors = four_chain(balanced=balanced)
    graph = str(source.graph.graph)

    def forbidden(*args):
        raise AssertionError("search must not execute user mathematics")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        paths = reassociation_paths(replace(source, function=forbidden), max_depth=6, max_candidates=32)
    assert not any(isinstance(w.message, NumericalTransformWarning) for w in caught)
    # Five parenthesizations of four factors, one of which is the source.
    expressions = {tensor_graph_expression_digest(p.cell.graph) for p in paths}
    assert len(expressions) == 4 and tensor_graph_expression_digest(source.graph) not in expressions
    assert source.transformations == () and str(source.graph.graph) == graph
    for path in paths:
        replayed = replay_tensor_rewrites(source, path.rewrites)
        assert replayed.declaration == source.declaration
        torch.testing.assert_close(
            replayed(*tensors)[0], tensors[0] @ tensors[1] @ tensors[2] @ tensors[3], rtol=0, atol=0
        )
        assert profile_tensor_memory(replayed) == profile_tensor_memory(path.cell)
    assert len(reassociation_paths(source, max_depth=6, max_candidates=1)) == 1
    assert reassociation_paths(source, max_depth=0) == () == reassociation_paths(source, max_candidates=0)
    for kwargs in ({"max_depth": -1}, {"max_candidates": -1}, {"max_depth": True}):
        with pytest.raises(ValueError, match="nonnegative"):
            reassociation_paths(source, **kwargs)


def test_search_crosses_a_local_minimum_instead_of_pruning_greedily():
    tensors = tuple(torch.ones(shape, dtype=torch.float64) for shape in ((1, 1), (1, 3), (3, 4), (4, 2)))
    source = ExportedCell.capture(lambda a, b, c, d: (((a @ b) @ c) @ d,), ("a", "b", "c", "d"), ("out",), tensors)
    before = profile_tensor_memory(source).devices[0].peak_new_tensor_bytes
    shallow = reassociation_paths(source, max_depth=1)
    assert shallow and all(profile_tensor_memory(p.cell).devices[0].peak_new_tensor_bytes > before for p in shallow)
    better = [
        p
        for p in reassociation_paths(source, max_depth=2)
        if profile_tensor_memory(p.cell).devices[0].peak_new_tensor_bytes < before
    ]
    assert better and all(len(p.rewrites) == 2 for p in better)


def test_joint_search_across_cells_is_canonical_and_shares_one_depth_budget():
    tensors = tuple(torch.ones(shape, dtype=torch.float64) for shape in ((4, 3), (3, 128), (128, 2)))
    cells = tuple(
        ExportedCell.capture(f, ("a", "b", "c"), ("out",), tensors)
        for f in (left_chain, lambda a, b, c: (torch.matmul(torch.matmul(a, b), c),))
    )
    paths = explore_tensor_combinations(cells, max_depth=6)
    assert [len(p) for p in paths] == [1, 1, 2]
    assert {r.source_digest for r in paths[-1]} == {c.digest for c in cells}
    assert explore_tensor_combinations(cells[::-1], max_depth=6) == paths  # Commuting steps deduplicate.
    with pytest.raises(ValueError, match="unique source"):
        explore_tensor_combinations((cells[0], cells[0]))
    deep, deep_tensors = four_chain()
    paths = explore_tensor_combinations((deep, cells[0]), max_depth=3, max_candidates=100)
    assert all(1 <= len(path) <= 3 for path in paths)
    both = [p for p in paths if sum(r.source_digest == deep.digest for r in p) == 2 and len(p) == 3]
    assert both
    for path in both:
        for source, args, expected in ((deep, deep_tensors, 64), (cells[0], tensors, 384)):
            (result,) = replay_tensor_rewrites(source, tuple(r for r in path if r.source_digest == source.digest))(
                *args
            )
            torch.testing.assert_close(result, torch.full_like(result, expected), rtol=0, atol=0)


def test_selection_binds_the_whole_selected_chain_within_its_depth_limit(monkeypatch):
    import pllm.execution.region_optimization as tuning

    source, tensors = four_chain()
    inputs = dict(zip(("in0", "in1", "in2", "in3"), tensors))
    monkeypatch.setattr(
        tuning,
        "_measure",
        lambda execution, *args: 1 if len(execution.cells[source.digest].transformations) == 2 else 100,
    )
    options = dict(
        payload_limits=(RegionPayloadLimit("cpu", 10000),),
        options=replace(ExecutionOptions(schedule_policy="serial"), tuning="measured"),
        warmup_rounds=0,
        measurement_rounds=1,
    )
    with pytest.warns(NumericalTransformWarning):
        selected = optimize_region(single_call(source), {source.digest: source}, inputs, **options)
    assert len(selected.rewrites) == 2
    (actual,) = selected.execution.run(inputs).values
    torch.testing.assert_close(actual, tensors[0] @ tensors[1] @ tensors[2] @ tensors[3], rtol=0, atol=0)
    for depth, candidates in ((0, 1), (1, 3)):
        limited = optimize_region(
            single_call(source), {source.digest: source}, inputs, max_rewrite_depth=depth, **options
        )
        assert len(limited.request.candidates) == candidates and limited.rewrites == ()


def test_independent_products_run_as_one_bmm_but_dependent_ones_do_not(monkeypatch):
    import pllm.execution.region_optimization as tuning

    tensors = tuple(
        torch.arange(1, 1 + r * c, dtype=torch.float64).reshape(r, c) for r, c in ((8, 4), (4, 6), (8, 4), (4, 6))
    )
    cell = ExportedCell.capture(lambda x, w: (x @ w,), ("x", "w"), ("out",), tensors[:2])
    left, right = cell.declaration.inputs
    (output,) = cell.declaration.outputs
    program = Program(
        region(
            tuple(Value(n, t) for n, t in zip("abcd", (left, right, left, right), strict=True)),
            (
                Call(("a", "b"), (Value("first", output),), cell.digest),
                Call(("c", "d"), (Value("second", output),), cell.digest),
            ),
            ("first", "second"),
        ),
        (cell.declaration,),
    )
    batched = lambda execution: any(  # noqa: E731
        "independent matmul batching" in t
        for source in execution.cells.values()
        if isinstance(source, ExportedCell)
        for t in source.transformations
    )
    monkeypatch.setattr(tuning, "_measure", lambda execution, *args: 1 if batched(execution) else 100)
    inputs = dict(zip("abcd", tensors))
    with pytest.warns(NumericalTransformWarning, match="batching"):
        selected = optimize_region(
            program,
            {cell.digest: cell},
            inputs,
            payload_limits=(RegionPayloadLimit("cpu", 10000),),
            options=replace(ExecutionOptions(schedule_policy="serial"), tuning="measured"),
            warmup_rounds=0,
            measurement_rounds=1,
        )
    (record,) = selected.rewrites
    assert isinstance(record, MatmulBatching)
    assert select_region_candidate(program, replace(selected.request, allow_numerical_transforms=False)).selected == 0

    class Products(StorageObserver):
        def __init__(self, inputs):
            super().__init__(inputs)
            self.products = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default):
                self.products.append(func)
            return super().__torch_dispatch__(func, types, args, kwargs)

    with Products(tensors) as observed:
        actual = selected.execution.run(inputs).values
    assert observed.products == [torch.ops.aten.bmm.default]
    for value, x, w in zip(actual, tensors[::2], tensors[1::2], strict=True):
        torch.testing.assert_close(value, x @ w, rtol=0, atol=0)
    bound = selected.selection.candidates[selected.selection.selected].capacity.devices[0].peak_tensor_bytes_upper_bound
    assert observed.peak - sum(t.untyped_storage().nbytes() for t in tensors) <= bound

    def chain(a, b):
        first = a @ b
        return first, first @ b

    square = torch.arange(1, 17, dtype=torch.float64).reshape(4, 4)
    dependent = ExportedCell.capture(chain, ("a", "b"), ("first", "second"), (square, square.clone()))
    assert tuple(batching_steps(dependent)) == ()


def transposed_copy(x):
    a = x + 1
    b = a.T[:2].clone()
    c = b * 2
    return c, c.T  # Two outputs share one allocation.


def swiglu(hidden, gate_up_weight, down_weight):
    gate, up = torch.nn.functional.linear(hidden, gate_up_weight).chunk(2, dim=-1)
    return (torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down_weight),)


def test_memory_profile_matches_the_storage_the_unfused_graph_really_uses():
    for function, shapes in (
        (transposed_copy, ((3, 4),)),
        (swiglu, ((3, 4), (12, 4), (4, 6))),
        (lambda x: ((x + 1)[:1, :1],), ((64, 64),)),  # A tiny returned slice retains its base.
    ):
        inputs = tuple(torch.randn(shape) for shape in shapes)
        saved = tuple(t.clone() for t in inputs)
        names = tuple(f"in{i}" for i in range(len(inputs)))
        outputs = tuple(f"out{i}" for i in range(len(function(*inputs))))
        source = ExportedCell.capture(function, names, outputs, inputs)
        profile = profile_tensor_memory(source)
        assert not profile.unknown_accesses
        with StorageObserver(inputs) as observed:
            actual = source(*inputs)
        assert observed.peak == profile.devices[0].peak_live_tensor_bytes
        for value, reference in zip(actual, function(*saved), strict=True):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        for value, original in zip(inputs, saved, strict=True):
            torch.testing.assert_close(value, original, rtol=0, atol=0)
        validate_tensor_memory(source, profile)


def test_memory_profile_is_cold_and_explicit_about_unknown_traffic():
    def forbidden(*args):
        raise AssertionError("real tensor math was evaluated")

    source = ExportedCell.capture(
        left_chain, ("a", "b", "c"), ("out",), (torch.ones(4, 2), torch.ones(2, 8), torch.ones(8, 3))
    )
    assert profile_tensor_memory(replace(source, function=forbidden)) == profile_tensor_memory(source)
    with pytest.raises(ValueError, match="backend access model"):
        profile_tensor_memory(replace(source, transformations=("torch.compile",)))
    sine = ExportedCell.capture(lambda x: (x.sin(),), ("x",), ("out",), (torch.ones(8),))
    profile = profile_tensor_memory(sine)
    assert profile.unknown_accesses == ("sin",)  # Not a zero-traffic estimate.
    # Fewer FLOPs can mean more traffic and peak storage: profiles, not FLOPs, rank candidates.
    inputs = tuple(torch.ones(shape, dtype=torch.float64) for shape in ((4, 1), (1, 4), (4, 32)))
    chain = capture(inputs=inputs)
    (candidate,) = reassociation_candidates(chain)
    (before,) = profile_tensor_memory(chain).devices
    (after,) = profile_tensor_memory(candidate.cell).devices
    assert candidate.rewrite.after.flops < candidate.rewrite.before.flops
    assert before.logical_read_bytes + before.logical_write_bytes < after.logical_read_bytes + after.logical_write_bytes
    assert before.peak_new_tensor_bytes < after.peak_new_tensor_bytes
