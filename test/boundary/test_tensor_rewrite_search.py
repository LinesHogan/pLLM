"""Bounded multi-step mathematics retains evidence and real storage costs."""

import warnings
from dataclasses import replace

import pytest
import torch

from pllm.config import ExecutionOptions
from pllm.contracts.region_selection import RegionPayloadLimit
from pllm.contracts.tensor_rewrite import MatmulReassociation
from pllm.execution.codec import tensor_graph_expression_digest, tensor_graph_structure_digest
from pllm.execution.exported_cell import ExportedCell, NumericalTransformWarning
from pllm.execution.region_optimization import (
    PreparedRegionCandidate,
    bind_region_candidate,
    optimize_region,
)
from pllm.execution.tensor_access import profile_tensor_memory
from pllm.execution.tensor_rewrites import TensorRewritePath, explore_tensor_combinations, replay_tensor_rewrites
from test.boundary.test_region_native_capacity import single_call


def reassociation_paths(source: ExportedCell, **kwargs) -> tuple[TensorRewritePath[MatmulReassociation], ...]:
    """Multi-step reassociation search through the general explore/replay door."""
    return tuple(
        TensorRewritePath(replay_tensor_rewrites(source, rs), rs)
        for rs in explore_tensor_combinations((source,), **kwargs)
        if all(isinstance(r, MatmulReassociation) for r in rs)
    )


def example(device="cpu", balanced=False):
    tensors = tuple(torch.ones(shape, dtype=torch.float64, device=device) for shape in ((4, 2), (2, 8), (8, 4), (4, 2)))

    def chain(a, b, c, d):
        return (((a @ b) @ c) @ d,)

    def tree(a, b, c, d):
        return ((a @ b) @ (c @ d),)

    cell = ExportedCell.capture(tree if balanced else chain, ("a", "b", "c", "d"), ("out",), tensors)
    return cell, tensors


@pytest.mark.parametrize("balanced", (False, True))
def test_search_deduplicates_inverse_cycles_and_preserves_the_original_graph(balanced):
    source, _ = example(balanced=balanced)
    original = str(source.graph.graph)

    def forbidden(*args):
        raise AssertionError("search must not execute user mathematics")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        paths = reassociation_paths(replace(source, function=forbidden), max_depth=6, max_candidates=32)
    assert not any(isinstance(w.message, NumericalTransformWarning) for w in caught)
    # Five mathematical parenthesizations total, including the original.
    # Different orders for independent products can remain physical candidates.
    assert len({tensor_graph_expression_digest(p.cell.graph) for p in paths}) == 4
    assert len({tensor_graph_structure_digest(p.cell.graph) for p in paths}) == len(paths)
    assert all(
        tensor_graph_expression_digest(p.cell.graph) != tensor_graph_expression_digest(source.graph) for p in paths
    )
    assert len(paths) < 32
    assert source.transformations == () and str(source.graph.graph) == original
    assert any(len(p.rewrites) == 2 for p in paths)


def test_depth_and_candidate_budgets_do_not_greedily_prune_by_local_cost():
    source, _ = example()
    assert len(reassociation_paths(source, max_depth=6, max_candidates=1)) == 1
    assert reassociation_paths(source, max_depth=0) == ()
    assert reassociation_paths(source, max_candidates=0) == ()
    for kwargs in ({"max_depth": -1}, {"max_candidates": -1}, {"max_depth": True}):
        with pytest.raises(ValueError, match="nonnegative"):
            reassociation_paths(source, **kwargs)
    # Reverse the useful chain: expensive first steps remain in the search.
    tensors = tuple(torch.ones(s, dtype=torch.float64) for s in ((4, 2), (2, 8), (8, 4), (4, 2)))
    right = ExportedCell.capture(lambda a, b, c, d: (a @ (b @ (c @ d)),), ("a", "b", "c", "d"), ("out",), tensors)
    assert any(p.rewrites[0].after.flops > p.rewrites[0].before.flops for p in reassociation_paths(right))


def test_two_step_search_crosses_a_local_compute_and_memory_minimum():
    tensors = tuple(torch.ones(shape, dtype=torch.float64) for shape in ((1, 1), (1, 3), (3, 4), (4, 2)))
    source = ExportedCell.capture(lambda a, b, c, d: (((a @ b) @ c) @ d,), ("a", "b", "c", "d"), ("out",), tensors)
    before = profile_tensor_memory(source).devices[0].peak_new_tensor_bytes
    shallow = reassociation_paths(source, max_depth=1)
    assert len(shallow) == 2
    assert all(p.rewrites[0].after.flops > p.rewrites[0].before.flops for p in shallow)
    assert all(profile_tensor_memory(p.cell).devices[0].peak_new_tensor_bytes > before for p in shallow)
    paths = reassociation_paths(source, max_depth=2)
    better = [p for p in paths if profile_tensor_memory(p.cell).devices[0].peak_new_tensor_bytes < before]
    assert better and all(len(p.rewrites) == 2 for p in better)
    # Initial graph: 46 FLOPs. Both immediate neighbors cost more, but one
    # two-step path reaches 44 FLOPs and 48 rather than 56 new tensor bytes.
    assert before == 56
    assert any(46 + sum(r.after.flops - r.before.flops for r in p.rewrites) == 44 for p in better)
    assert min(profile_tensor_memory(p.cell).devices[0].peak_new_tensor_bytes for p in better) == 48


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_every_step_replays_and_final_access_graph_is_recomputed(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    source, tensors = example(device)
    paths = reassociation_paths(source)
    original = profile_tensor_memory(source)
    assert any(len(p.rewrites) > 1 for p in paths)
    for path in paths:
        assert len(path.cell.transformations) == len(path.rewrites)
        for before, after in zip(path.rewrites, path.rewrites[1:]):
            assert before.output_graph_digest == after.input_graph_digest
        replayed = replay_tensor_rewrites(source, path.rewrites)
        assert str(replayed.graph.graph) == str(path.cell.graph.graph)
        assert replayed.declaration == source.declaration
        torch.testing.assert_close(
            replayed(*tensors)[0], tensors[0] @ tensors[1] @ tensors[2] @ tensors[3], rtol=0, atol=0
        )
        assert profile_tensor_memory(replayed) == profile_tensor_memory(path.cell)
    assert (
        min(profile_tensor_memory(p.cell).devices[0].peak_new_tensor_bytes for p in paths)
        < original.devices[0].peak_new_tensor_bytes
    )


def test_missing_reordered_duplicated_or_forged_steps_cannot_replay():
    source, _ = example()
    path = next(p for p in reassociation_paths(source) if len(p.rewrites) == 2)
    first, second = path.rewrites
    for records in (
        (second,),
        (second, first),
        (first, first),
        (first, replace(second, input_graph_digest=first.input_graph_digest)),
        (first, replace(second, after=replace(second.after, flops=0))),
    ):
        with pytest.raises(ValueError, match="source"):
            replay_tensor_rewrites(source, records)
    with pytest.raises(ValueError, match="original exported"):
        reassociation_paths(path.cell)


def test_multistep_search_preserves_functional_mutation_and_returned_views():
    tensors = (*example()[1], torch.zeros(4, 2, dtype=torch.float64))

    def update(a, b, c, d, state):
        state.copy_(((a @ b) @ c) @ d)
        return state, state[:1]

    source = ExportedCell.capture(update, ("a", "b", "c", "d", "state"), ("new", "view"), tensors)
    path = next(p for p in reassociation_paths(source) if len(p.rewrites) == 2)
    replayed = replay_tensor_rewrites(source, path.rewrites)
    assert replayed.output_positions == source.output_positions
    assert replayed.mutations == source.mutations
    result, view = replayed(*tensors)
    torch.testing.assert_close(result, tensors[0] @ tensors[1] @ tensors[2] @ tensors[3], rtol=0, atol=0)
    assert result.untyped_storage().data_ptr() == view.untyped_storage().data_ptr()
    assert torch.count_nonzero(tensors[-1]) == 0
    profile_tensor_memory(replayed)


def test_native_selection_binds_the_whole_selected_chain_and_rejects_a_prefix(monkeypatch):
    import pllm.execution.region_optimization as tuning

    source, tensors = example()
    inputs = dict(zip(("in0", "in1", "in2", "in3"), tensors))

    def measure(execution, *args):
        cell = execution._cells[source.digest]
        return 1 if len(cell.transformations) == 2 else 100

    monkeypatch.setattr(tuning, "_measure", measure)
    with pytest.warns(NumericalTransformWarning):
        selected = optimize_region(
            single_call(source),
            {source.digest: source},
            inputs,
            payload_limits=(RegionPayloadLimit("cpu", 10000),),
            options=replace(ExecutionOptions(schedule_policy="serial"), tuning="measured"),
            warmup_rounds=0,
            measurement_rounds=1,
        )
    assert len(selected.rewrites) == 2
    partial = replay_tensor_rewrites(source, selected.rewrites[:1])
    with pytest.raises(ValueError, match="admitted imports"):
        bind_region_candidate(
            PreparedRegionCandidate(
                single_call(source),
                selected.request.candidates[selected.selection.selected],
                selected.rewrites[:1],
                {source.digest: partial},
            ),
            selected.request.payload_limits,
            selected.selection.candidates[selected.selection.selected],
        )
    (actual,) = selected.execution.run(inputs).values
    torch.testing.assert_close(actual, tensors[0] @ tensors[1] @ tensors[2] @ tensors[3], rtol=0, atol=0)
    for depth, expected_candidates in ((0, 1), (1, 3)):
        limited = optimize_region(
            single_call(source),
            {source.digest: source},
            inputs,
            payload_limits=(RegionPayloadLimit("cpu", 10000),),
            options=replace(ExecutionOptions(schedule_policy="serial"), tuning="measured"),
            warmup_rounds=0,
            measurement_rounds=1,
            max_rewrite_depth=depth,
        )
        assert len(limited.request.candidates) == expected_candidates
        assert limited.rewrites == ()  # All measured times tie without depth two.
