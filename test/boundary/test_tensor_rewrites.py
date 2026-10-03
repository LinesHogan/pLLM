"""Real ATen graph rewrites, their prerequisites, and the functional export ABI."""

from dataclasses import replace
import warnings

import pytest
import torch

from pllm.contracts.tensor_rewrite import MatmulAssociation, MatmulReassociation
from pllm.execution.codec import decode_boundary, encode_boundary, tensor_node_metadata
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.tensor_rewrites import TensorRewriteCandidate, explore_tensor_combinations, replay_tensor_rewrites


def reassociation_candidates(source: ExportedCell) -> tuple[TensorRewriteCandidate[MatmulReassociation], ...]:
    """Single-step reassociation candidates through the general explore/replay door."""
    return tuple(TensorRewriteCandidate(replay_tensor_rewrites(source, p), p[0])
                 for p in explore_tensor_combinations((source,), max_depth=1) if isinstance(p[0], MatmulReassociation))


def left_chain(a, b, c):
    return ((a @ b) @ c,)


def right_chain(a, b, c):
    return (a @ (b @ c),)


def examples(dimensions=(12, 2, 16, 3), *, device="cpu", dtype=torch.float64):
    m, k, n, p = dimensions
    generator = torch.Generator().manual_seed(183)
    return tuple((torch.rand(shape, generator=generator, dtype=torch.float64) * 0.5 - 0.25).to(device, dtype)
                 for shape in ((m, k), (k, n), (n, p)))


def capture(function=left_chain, inputs=None, outputs=("result",)):
    return ExportedCell.capture(function, ("a", "b", "c"), outputs, inputs or examples())


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("function,association", [
    (left_chain, MatmulAssociation.LEFT_TO_RIGHT),
    (right_chain, MatmulAssociation.RIGHT_TO_LEFT),
])
def test_reassociation_rewrites_real_graph_and_preserves_source(device, dtype, function, association):
    inputs = examples(device=device, dtype=dtype)
    saved = tuple(t.clone() for t in inputs)
    source = capture(function, inputs)
    original_graph = str(source.graph.graph)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        candidates = reassociation_candidates(source)
    assert not caught  # Generating a numerical candidate does not select it.
    assert len(candidates) == 1
    candidate, = candidates
    evidence = candidate.rewrite
    assert evidence.association is association
    assert evidence.dimensions == (12, 2, 16, 3)
    assert candidate.cell.declaration == source.declaration
    assert source.transformations == ()
    assert str(source.graph.graph) == original_graph
    assert candidate.cell.graph is not source.graph
    assert evidence.input_graph_digest != evidence.output_graph_digest
    assert set(evidence.source_nodes).isdisjoint(n.name for n in candidate.cell.graph.graph.nodes)
    nodes = {n.name: n for n in candidate.cell.graph.graph.nodes}
    assert tuple(n.name for n in nodes[evidence.replacement_nodes[0]].args) == (
        evidence.operands[1:] if function is left_chain else evidence.operands[:2])
    inner = tensor_node_metadata(nodes[evidence.replacement_nodes[0]])
    assert inner.shape == ((2, 3) if function is left_chain else (12, 16))
    assert inner.device == inputs[0].device  # Fake metadata retains the actual target.
    actual = candidate.cell(*inputs)[0]
    expected = function(*saved)[0]
    rtol, atol = (1e-12, 1e-12) if dtype == torch.float64 else (1e-5, 1e-6)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    for current, original in zip(inputs, saved, strict=True):
        assert torch.equal(current, original)
    # The original remains the numerical reference after candidate execution.
    assert torch.equal(source(*inputs)[0], expected)
    replayed = replay_tensor_rewrites(source, (evidence,))
    assert torch.equal(replayed(*inputs)[0], actual)


def test_cost_facts_are_subgraph_counts_not_a_placement_or_latency_score():
    left, = reassociation_candidates(capture())
    assert left.rewrite.before.flops == 2 * (12 * 2 * 16 + 12 * 16 * 3)
    assert left.rewrite.after.flops == 2 * (2 * 16 * 3 + 12 * 2 * 3)
    assert left.rewrite.before.intermediate_bytes == 12 * 16 * 8
    assert left.rewrite.after.intermediate_bytes == 2 * 3 * 8
    # Legal but more expensive alternatives are still exposed to the planner.
    right, = reassociation_candidates(capture(right_chain))
    assert right.rewrite.before == left.rewrite.after
    assert right.rewrite.after == left.rewrite.before


def test_shared_or_returned_intermediates_are_not_erased():
    def returned(a, b, c):
        inner = a @ b
        return inner @ c, inner

    def fanout(a, b, c):
        inner = a @ b
        return inner @ c, inner.sum()

    for function in (returned, fanout):
        source = capture(function, outputs=("final", "observed"))
        assert reassociation_candidates(source) == ()


def test_functional_mutation_output_positions_and_unrelated_users_are_preserved():
    def update(a, b, c, state):
        state.copy_(torch.mm(torch.mm(a, b), c))
        return state * 2, state, a + 1

    inputs = (*examples(), torch.zeros(12, 3, dtype=torch.float64))
    saved = tuple(t.clone() for t in inputs)
    source = ExportedCell.capture(update, ("a", "b", "c", "state"), ("twice", "new_state", "other"), inputs)
    assert source.output_positions == (1, 2, 3)
    assert source.mutations == (("state", "new_state"),)
    candidate, = reassociation_candidates(source)
    assert candidate.cell.output_positions == source.output_positions
    assert candidate.cell.mutations == source.mutations
    for actual, expected in zip(candidate.cell(*inputs), update(*saved), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(inputs[-1]) == 0
    original_others = [(n.name, n.target) for n in source.graph.graph.nodes
                       if n.name not in candidate.rewrite.source_nodes]
    candidate_others = [(n.name, n.target) for n in candidate.cell.graph.graph.nodes
                        if n.name not in candidate.rewrite.replacement_nodes]
    assert candidate_others == original_others


@pytest.mark.parametrize("dtype", [torch.int64, torch.complex64])
def test_rule_rejects_non_real_floating_domains(dtype):
    inputs = tuple(t.to(dtype) for t in examples())
    assert reassociation_candidates(capture(inputs=inputs)) == ()


def test_batched_matmul_is_not_mistaken_for_a_rank_two_equation():
    inputs = tuple(t.unsqueeze(0).repeat(2, 1, 1) for t in examples())
    assert reassociation_candidates(capture(inputs=inputs)) == ()


def test_two_uses_by_the_same_node_do_not_count_as_a_private_intermediate():
    def square(a, b):
        x = a @ b
        return (x @ x,)

    source = ExportedCell.capture(square, ("a", "b"), ("out",),
                                  (torch.ones(3, 2), torch.ones(2, 3)))
    assert reassociation_candidates(source) == ()


def test_multiple_sites_produce_individual_alternatives_with_local_provenance():
    def diamond(a, b, c, d):
        return ((a @ b) @ (c @ d),)

    inputs = (*examples(), torch.ones(3, 4, dtype=torch.float64))
    source = ExportedCell.capture(diamond, ("a", "b", "c", "d"), ("out",), inputs)
    candidates = reassociation_candidates(source)
    assert len(candidates) == 2
    assert {c.rewrite.association for c in candidates} == set(MatmulAssociation)
    for candidate in candidates:
        torch.testing.assert_close(candidate.cell(*inputs)[0], diamond(*inputs)[0], rtol=1e-12, atol=1e-12)
        assert len([n for n in candidate.cell.graph.graph.nodes if n.op == "call_function"]) == 3


@pytest.mark.parametrize("dimensions", [(0, 2, 4, 3), (3, 0, 4, 2), (3, 2, 0, 4), (3, 2, 4, 0)])
def test_zero_dimensions_keep_output_shape_and_empty_reductions(dimensions):
    inputs = examples(dimensions)
    candidate, = reassociation_candidates(capture(inputs=inputs))
    assert torch.equal(candidate.cell(*inputs)[0], left_chain(*inputs)[0])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_dtype_support_uses_exact_dyadic_fixture(dtype):
    inputs = tuple(torch.full(shape, 0.125, dtype=dtype) for shape in ((4, 2), (2, 8), (8, 2)))
    candidate, = reassociation_candidates(capture(inputs=inputs))
    assert candidate.rewrite.dtype == str(dtype)
    assert torch.equal(candidate.cell(*inputs)[0], left_chain(*inputs)[0])


def test_transposed_operands_and_runtime_signature_guards():
    inputs = tuple(t.T.contiguous().T for t in examples())
    source = capture(inputs=inputs)
    candidate, = reassociation_candidates(source)
    torch.testing.assert_close(candidate.cell(*inputs)[0], left_chain(*inputs)[0], rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="guard"):
        candidate.cell(*(t.contiguous() for t in inputs))
    with pytest.raises(ValueError, match="guard"):
        candidate.cell(*examples((12, 2, 16, 4)))


def test_evidence_is_serializable_and_rechecked_not_just_hashed():
    source = capture()
    candidate, = reassociation_candidates(source)
    record = decode_boundary(encode_boundary(candidate.rewrite), MatmulReassociation)
    assert record == candidate.rewrite
    forged = (
        replace(record, source_digest="other"),
        replace(record, input_graph_digest="other"),
        replace(record, output_graph_digest="other"),
        replace(record, dimensions=(12, 2, 16, 4)),
        replace(record, operands=tuple(reversed(record.operands))),
        replace(record, before=replace(record.before, flops=1)),
        replace(record, after=replace(record.after, intermediate_bytes=0)),
        replace(record, association=MatmulAssociation.RIGHT_TO_LEFT),
    )
    for evidence in forged:
        with pytest.raises(ValueError, match="source"):
            replay_tensor_rewrites(source, (evidence,))


def test_candidate_generation_is_cold_and_does_not_execute_the_source_callable():
    def forbidden(*args):
        raise AssertionError("candidate generation executed user math")

    source = replace(capture(), function=forbidden)
    candidate, = reassociation_candidates(source)
    torch.testing.assert_close(candidate.cell(*examples())[0], left_chain(*examples())[0], rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="original exported cell"):
        reassociation_candidates(candidate.cell)
