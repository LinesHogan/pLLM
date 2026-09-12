"""Joint mathematical choices are assessed with the region's real lifetimes."""

from dataclasses import replace
import warnings

import pytest
import torch

from pllm.config import ExecutionOptions
from pllm.contracts.region_program import Block, Call, Program, Region, Value
from pllm.contracts.region_selection import RegionPayloadLimit
from pllm.execution.exported_cell import ExportedCell, NumericalTransformWarning
from pllm.execution.region_optimization import bind_selected_region, optimize_region
from pllm.execution.tensor_rewrites import explore_matmul_combinations, replay_matmul_reassociations
from test.boundary.test_tensor_access import StorageObserver
from test.boundary.test_tensor_rewrite_search import example as deep_example


def example(device="cpu"):
    tensors = tuple(torch.ones(shape, dtype=torch.float64, device=device)
                    for shape in ((4, 3), (3, 128), (128, 2)))
    def first(a, b, c):
        return ((a @ b) @ c,)
    def second(a, b, c):
        return (torch.matmul(torch.matmul(a, b), c),)
    cells = tuple(ExportedCell.capture(f, ("a", "b", "c"), ("out",), tensors) for f in (first, second))
    arguments = tuple(Value(name, t) for name, t in zip(("a", "b", "c"), cells[0].declaration.inputs))
    calls = tuple(Call(("a", "b", "c"), (Value(f"out{i}", cell.declaration.outputs[0]),), cell.digest)
                  for i, cell in enumerate(cells))
    program = Program(Region(Block(arguments, calls, ("out0", "out1"))), tuple(c.declaration for c in cells))
    return program, {c.digest: c for c in cells}, dict(zip(("a", "b", "c"), tensors))


def test_joint_search_is_bounded_canonical_and_deduplicates_commuting_steps():
    _, cells, _ = example()
    original_graphs = tuple(str(c.graph.graph) for c in cells.values())
    def forbidden(*args):
        raise AssertionError("search must not execute user mathematics")
    sources = tuple(replace(c, function=forbidden) for c in cells.values())
    with warnings.catch_warnings(record=True) as caught:
        paths = explore_matmul_combinations(sources, max_depth=6)
    assert not any(isinstance(w.message, NumericalTransformWarning) for w in caught)
    assert [len(p) for p in paths] == [1, 1, 2]
    assert {r.source_digest for r in paths[-1]} == set(cells)
    assert explore_matmul_combinations(sources[::-1], max_depth=6) == paths
    assert explore_matmul_combinations(sources, max_depth=1) == paths[:2]
    assert explore_matmul_combinations(sources, max_candidates=1) == paths[:1]
    assert explore_matmul_combinations(sources, max_depth=0) == ()
    assert explore_matmul_combinations(sources, max_candidates=0) == ()
    assert explore_matmul_combinations(()) == ()
    assert tuple(str(c.graph.graph) for c in cells.values()) == original_graphs
    for kwargs in ({"max_depth": -1}, {"max_depth": True}, {"max_candidates": -1}):
        with pytest.raises(ValueError, match="nonnegative"):
            explore_matmul_combinations(sources, **kwargs)
    with pytest.raises(ValueError, match="unique source"):
        explore_matmul_combinations((sources[0], sources[0]))
    with pytest.raises(ValueError, match="original exported"):
        explore_matmul_combinations((replace(sources[0], transformations=("untrusted",)),))


def test_depth_is_shared_across_functions_and_each_chain_replays_in_order():
    deep, tensors = deep_example()
    _, cells, inputs = example()
    simple = next(iter(cells.values()))
    sources = (deep, simple)
    paths = explore_matmul_combinations(sources, max_depth=3, max_candidates=100)
    assert all(1 <= len(path) <= 3 for path in paths)
    both = [path for path in paths if sum(r.source_digest == deep.digest for r in path) == 2
            and sum(r.source_digest == simple.digest for r in path) == 1]
    assert both
    shallow = explore_matmul_combinations(sources, max_depth=2, max_candidates=100)
    assert all(len(path) <= 2 for path in shallow)
    for path in both:
        for source, args, expected in ((deep, tensors, 64), (simple, tuple(inputs.values()), 384)):
            records = tuple(r for r in path if r.source_digest == source.digest)
            replayed = replay_matmul_reassociations(source, records)
            result, = replayed(*args)
            torch.testing.assert_close(result, torch.full_like(result, expected), rtol=0, atol=0)


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_only_joint_rewrite_fits_and_only_feasible_candidates_are_measured(device, monkeypatch):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    import pllm.execution.region_optimization as tuning
    program, cells, inputs = example(device)
    # Isolate combinations that retain both call boundaries. Outlining has
    # its own end-to-end selection tests and adds another candidate family.
    monkeypatch.setattr(tuning, "_outline_candidates", lambda *args: iter(()))
    saved = tuple(t.clone() for t in inputs.values())
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.startswith("cuda") else None
    measure = tuning._measure
    measured = []
    def checked_measure(execution, *args):
        # Each original call needs a 4*128*8-byte internal product. Changing
        # just one call leaves that maximum scratch requirement in the region.
        assert all(len(c.transformations) == 1 for c in execution._cells.values())
        measured.append(execution)
        return measure(execution, *args)
    monkeypatch.setattr(tuning, "_measure", checked_measure)
    options = ExecutionOptions(schedule_policy="serial")
    with pytest.warns(NumericalTransformWarning):
        selected = optimize_region(program, cells, inputs,
            payload_limits=(RegionPayloadLimit(device, 5392),), options=options,
            warmup_rounds=0, measurement_rounds=2)
    assert len(measured) == 2
    assert selected.selection.selected == 3
    assert [c.feasible for c in selected.selection.candidates] == [False, False, False, True]
    assert [c.capacity.devices[0].peak_tensor_bytes_upper_bound for c in selected.selection.candidates] == [9440] * 3 + [5392]
    assert all(c.elapsed_ns == () for c in selected.request.candidates[:3])
    assert len(selected.request.candidates[3].elapsed_ns) == 2
    assert selected.selection.candidates[3].changed_functions == tuple(sorted(cells))
    with StorageObserver(tuple(inputs.values())) as observed:
        result = selected.run(inputs)
    assert result.invocations == 2
    assert observed.peak - sum(t.numel() * t.element_size() for t in inputs.values()) <= 5392
    for value in result.values:
        torch.testing.assert_close(value, torch.full_like(value, 384), rtol=0, atol=0)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    for before, after in zip(saved, inputs.values(), strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    with pytest.raises(ValueError, match="replayed mathematical evidence"):
        bind_selected_region(program, cells, selected.request, selected.selection,
                             selected.rewrites[:1], options=options)
    first, second = selected.rewrites
    with pytest.raises(ValueError, match="source graph"):
        bind_selected_region(program, cells, selected.request, selected.selection,
            (first, replace(second, after=replace(second.after, flops=0))), options=options)
    # Steps in different functions commute; only each function's internal
    # proof chain has an execution order.
    rebound = bind_selected_region(program, cells, selected.request, selected.selection,
                                   selected.rewrites[::-1], options=options)
    assert rebound.selection == selected.selection


def test_optout_and_search_budgets_do_not_authorize_an_over_budget_candidate(monkeypatch):
    import pllm.execution.region_optimization as tuning
    program, cells, inputs = example()
    monkeypatch.setattr(tuning, "_outline_candidates", lambda *args: iter(()))
    def forbidden(*args):
        raise AssertionError("an infeasible candidate must not be measured")
    monkeypatch.setattr(tuning, "_measure", forbidden)
    with warnings.catch_warnings(record=True) as caught:
        for kwargs in ({"max_rewrite_depth": 1}, {"max_candidates": 3},
                       {"options": ExecutionOptions(allow_numerical_transforms=False, schedule_policy="serial")}):
            settings = {"options": ExecutionOptions(schedule_policy="serial"), **kwargs}
            with pytest.raises(ValueError, match="no region candidate"):
                optimize_region(program, cells, inputs, payload_limits=(RegionPayloadLimit("cpu", 5392),),
                                warmup_rounds=0, measurement_rounds=1, **settings)
    assert not any(isinstance(w.message, NumericalTransformWarning) for w in caught)
