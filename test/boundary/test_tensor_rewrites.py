"""Tensor graph rewrites, their search, and the memory profiles that rank them.

Rewrites (matmul reassociation, independent-product batching) are generated
cold from the graph, never by running user math, and only where they are
legal. Each rewrite replays from its evidence record, keeps the functional ABI
and matches the source numerically. ``profile_tensor_memory`` must describe
the storage the unfused graph really uses.
"""

from dataclasses import replace

import pytest
import torch

from pllm.config import ExecutionOptions
from pllm.contracts.region_program import Call, Program, Value
from pllm.contracts.region_selection import RegionPayloadLimit
from pllm.contracts.tensor_rewrite import MatmulAssociation, MatmulReassociation
from pllm.execution.codec import decode_boundary, encode_boundary, tensor_graph_expression_digest
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_optimization import optimize_region
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
    (left_chain, lambda: operands((3, 0, 4, 2)), EXACT, None),  # Empty reduction keeps output shape.
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
        (candidate,) = reassociation_candidates(source)
        if association is not None:
            assert candidate.rewrite.association is association
        assert candidate.cell.declaration == source.declaration
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
    # Only real floating domains: integer products must stay exact.
    assert reassociation_candidates(capture(inputs=tuple(t.to(torch.int64) for t in operands()))) == ()


def test_rewrites_keep_mutations_and_output_positions():
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


@pytest.mark.parametrize("balanced", (False, True))
def test_search_reaches_every_parenthesization_without_running_user_math(balanced):
    source, tensors = four_chain(balanced=balanced)

    def forbidden(*args):
        raise AssertionError("search must not execute user mathematics")

    paths = reassociation_paths(replace(source, function=forbidden), max_depth=6, max_candidates=32)
    # Five parenthesizations of four factors, one of which is the source.
    expressions = {tensor_graph_expression_digest(p.cell.graph) for p in paths}
    assert len(expressions) == 4 and tensor_graph_expression_digest(source.graph) not in expressions
    for path in paths:
        replayed = replay_tensor_rewrites(source, path.rewrites)
        assert replayed.declaration == source.declaration
        torch.testing.assert_close(
            replayed(*tensors)[0], tensors[0] @ tensors[1] @ tensors[2] @ tensors[3], rtol=0, atol=0
        )
    assert reassociation_paths(source, max_depth=0) == ()


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
    for cell in cells:
        (result,) = replay_tensor_rewrites(cell, tuple(r for r in paths[-1] if r.source_digest == cell.digest))(
            *tensors
        )
        torch.testing.assert_close(result, torch.full_like(result, 384), rtol=0, atol=0)


def test_selection_offers_rewrite_chains_within_its_depth_limit():
    source, tensors = four_chain()
    inputs = dict(zip(("in0", "in1", "in2", "in3"), tensors))
    options = dict(
        payload_limits=(RegionPayloadLimit("cpu", 10000),),
        options=replace(ExecutionOptions(schedule_policy="serial"), tuning="measured"),
        warmup_rounds=0,
        measurement_rounds=1,
    )
    selected = optimize_region(single_call(source), {source.digest: source}, inputs, **options)
    (actual,) = selected.execution.run(inputs).values
    torch.testing.assert_close(actual, tensors[0] @ tensors[1] @ tensors[2] @ tensors[3], rtol=0, atol=0)
    for depth, candidates in ((0, 1), (1, 3)):
        limited = optimize_region(
            single_call(source), {source.digest: source}, inputs, max_rewrite_depth=depth, **options
        )
        assert len(limited.request.candidates) == candidates


def test_independent_products_are_offered_as_one_bmm_but_dependent_ones_are_not():
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
    inputs = dict(zip("abcd", tensors))
    common = dict(payload_limits=(RegionPayloadLimit("cpu", 10000),), warmup_rounds=0, measurement_rounds=1)
    options = replace(ExecutionOptions(schedule_policy="serial"), tuning="measured")
    selected = optimize_region(program, {cell.digest: cell}, inputs, options=options, **common)
    for value, x, w in zip(selected.execution.run(inputs).values, tensors[::2], tensors[1::2], strict=True):
        torch.testing.assert_close(value, x @ w, rtol=0, atol=0)
    exact = replace(options, allow_numerical_transforms=False)
    assert len(selected.request.candidates) > len(
        optimize_region(program, {cell.digest: cell}, inputs, options=exact, **common).request.candidates
    )

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
    sine = ExportedCell.capture(lambda x: (x.sin(),), ("x",), ("out",), (torch.ones(8),))
    profile = profile_tensor_memory(sine)
    assert profile.unknown_accesses == ("sin",)  # Not a zero-traffic estimate.
