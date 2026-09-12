"""Original AER selects and replays structural and tensor transformations together."""

from dataclasses import replace
import warnings

import pytest
import torch

from pllm.config import ExecutionOptions
from pllm.contracts.region_outlining import CallOutline, CallOutlineEvidence
from pllm.contracts.region_program import Block, Call, Program, Region, ScalarType, Scan, SizeParameter, Value
from pllm.contracts.region_selection import RegionCandidate, RegionPayloadLimit, RegionSelectionRequest, RegionTuningSize
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.exported_cell import ExportedCell, NumericalTransformWarning
from pllm.execution.region_compilation import analyze_region_program
from pllm.execution.region_optimization import bind_region_candidate, bind_selected_region, optimize_region, select_region_candidate
from pllm.execution.region_outlining import outline_region_calls
from pllm.execution.region_storage import import_region_functions
from pllm.execution.tensor_access import import_tensor_access_graph
from test.boundary.test_region_outlining import chain
from test.boundary.test_tensor_access import StorageObserver


def test_selected_tensor_rewrite_preserves_surrounding_invocation_effects():
    from pllm.contracts.region_program import ExternalResource, Function
    from pllm.execution.region_native import external_primitive
    from pllm.execution.region_resources import ExternalFunction, RegionResources
    from pllm.execution.tensor_rewrites import explore_matmul_reassociations
    from test.boundary.test_region_pages import owner_fixture

    pure, sources, inputs = chain()

    begin = Function("state.begin", (), (), ("state",))
    publish = Function("state.publish", (pure.functions[-1].outputs[0],), (), ("state",))
    block = pure.entry.block
    program = Program(Region(Block((*block.arguments,), (
        Call((), (), begin.name), *block.operations,
        Call((block.yields[0],), (), publish.name),
    ), (*block.yields,))), (*pure.functions, begin, publish), (ExternalResource("state"),))
    sources = {**sources, begin.name: external_primitive(begin), publish.name: external_primitive(publish)}
    original_imports = import_region_functions(program, sources)
    outlined = outline_region_calls(program, sources, CallOutline((), 1, 2))
    path = explore_matmul_reassociations(outlined.cell)[0]
    derived = outlined.outlined.program
    functions = {**sources, outlined.cell.digest: path.cell}
    proof = CallOutlineEvidence(outlined.site, import_tensor_access_graph(outlined.cell))
    candidates = (
        RegionCandidate(analyze_region_program(program, optimize=False).reference_schedule, original_imports, (100,)),
        RegionCandidate(analyze_region_program(derived, optimize=False).reference_schedule,
            import_region_functions(derived, functions), (1,), (proof,)),
    )
    # Fixed costs exercise installation, not a performance claim. The stateful
    # engine, rather than optimize_region, is responsible for real trials.
    request = RegionSelectionRequest(original_imports, candidates, (RegionPayloadLimit("cpu", 100000),),
        True, "stateful installation fixture")
    selection = select_region_candidate(program, request)
    assert selection.selected == 1
    with pytest.warns(NumericalTransformWarning):
        selected = bind_selected_region(program, sources, request, selection, path.rewrites)
    trials = replace(request, candidates=tuple(replace(candidate, elapsed_ns=()) for candidate in candidates))
    with warnings.catch_warnings(record=True) as caught:
        trial = bind_region_candidate(program, sources, trials, 1, path.rewrites)
    assert not any(isinstance(w.message, NumericalTransformWarning) for w in caught)
    for invalid in (replace(trials, payload_limits=(RegionPayloadLimit("cpu", 1),)),
                    replace(trials, allow_numerical_transforms=False)):
        with pytest.raises(ValueError, match="payload or numerical policy"):
            bind_region_candidate(program, sources, invalid, 1, path.rewrites)
    with pytest.raises(ValueError, match="pure program"):
        optimize_region(program, sources, inputs, payload_limits=request.payload_limits)
    # A source-bound native effect cannot become a mathematical replacement.
    with pytest.raises(ValueError, match="resource accessors or opaque kernels"):
        bind_selected_region(program, sources, request, selection,
            (replace(path.rewrites[0], source_digest=publish.name),))
    with pytest.raises(ValueError):
        outline_region_calls(program, sources, CallOutline((), 0, 2))

    execution, _, rows, _ = owner_fixture(rounds=1)
    invocation = execution.start_step(rows)
    events = []
    def start(args):
        events.append("begin")
        return ()
    def commit(args):
        assert events == ["begin"]
        events.append(args[0].clone())
        return ()
    resources = RegionResources(program, invocation.context, (),
        (ExternalFunction(begin, start), ExternalFunction(publish, commit)))
    try:
        assert events == []
        for runnable in (trial, selected):
            events.clear()  # The caller explicitly isolates each invocation.
            actual, = runnable.run(inputs, pages=resources).values
            assert len(events) == 2
            torch.testing.assert_close(actual, torch.full_like(actual, 384), rtol=0, atol=0)
            torch.testing.assert_close(events[1], actual, rtol=0, atol=0)
    finally:
        execution.cancel_step(invocation)
        execution.retire(tuple(row.identity for row in rows))


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_automatic_outlining_exposes_the_only_math_candidate_that_fits(device, monkeypatch):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    import pllm.execution.region_optimization as tuning
    program, sources, inputs = chain(device)
    before = tuple(t.clone() for t in inputs.values())
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.startswith("cuda") else None
    measure = tuning._measure
    seen = []
    def checked_measure(execution, *args):
        changed = [cell for cell in execution._cells.values() if cell.transformations]
        assert len(changed) == 1 and changed[0].digest not in sources
        seen.append(execution)
        return measure(execution, *args)
    monkeypatch.setattr(tuning, "_measure", checked_measure)
    options = ExecutionOptions(schedule_policy="serial")
    with pytest.warns(NumericalTransformWarning):
        selected = optimize_region(program, sources, inputs, payload_limits=(RegionPayloadLimit(device, 5328),),
            options=options, warmup_rounds=0, measurement_rounds=2)
    assert len(seen) == 2
    assert [c.feasible for c in selected.selection.candidates] == [False, False, True]
    assert [c.capacity.devices[0].peak_tensor_bytes_upper_bound for c in selected.selection.candidates] == [9376, 9376, 5328]
    candidate = selected.request.candidates[selected.selection.selected]
    assert len(candidate.outlines) == len(selected.rewrites) == 1
    assert candidate.outlines[0].access.source_digest == selected.rewrites[0].source_digest
    assert selected.selection.candidates[2].changed_functions == (selected.rewrites[0].source_digest,)
    assert decode_boundary(encode_boundary(selected.request), RegionSelectionRequest) == selected.request
    with StorageObserver(tuple(inputs.values())) as observed:
        actual = selected.run(inputs)
    assert actual.invocations == 1
    assert observed.peak - sum(t.numel() * t.element_size() for t in inputs.values()) <= 5328
    torch.testing.assert_close(actual.values[0], torch.full_like(actual.values[0], 384), rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    for saved, tensor in zip(before, inputs.values(), strict=True):
        torch.testing.assert_close(saved, tensor, rtol=0, atol=0)
    with pytest.raises(ValueError, match="mathematical evidence"):
        bind_selected_region(program, sources, selected.request, selected.selection, (), options=options)
    def forbidden(*args, **kwargs):
        raise AssertionError("runtime must not compile, select or replay")
    for name in ("outline_region_calls", "select_region_candidate", "_measure", "_replay", "_executor"):
        monkeypatch.setattr(tuning, name, forbidden)
    torch.testing.assert_close(selected.run(inputs).values[0], actual.values[0], rtol=0, atol=0)


def test_exact_outlining_is_available_with_numerical_transforms_disabled(monkeypatch):
    import pllm.execution.region_optimization as tuning
    program, sources, inputs = chain()
    monkeypatch.setattr(tuning, "_measure", lambda execution, *args: 1 if len(execution._program.functions) == 3 else 100)
    options = ExecutionOptions(allow_numerical_transforms=False, schedule_policy="serial")
    with warnings.catch_warnings(record=True) as caught:
        selected = optimize_region(program, sources, inputs, payload_limits=(RegionPayloadLimit("cpu", 10000),),
            options=options, warmup_rounds=0, measurement_rounds=1)
    assert not any(isinstance(w.message, NumericalTransformWarning) for w in caught)
    assert selected.selection.selected == 1 and selected.rewrites == ()
    assert len(selected.request.candidates) == 2
    assert selected.selection.candidates[1].changed_functions == ()
    actual, = selected.run(inputs).values
    torch.testing.assert_close(actual, torch.full_like(actual, 384), rtol=0, atol=0)
    limited = optimize_region(program, sources, inputs, payload_limits=(RegionPayloadLimit("cpu", 10000),),
        options=options, max_candidates=1, warmup_rounds=0, measurement_rounds=1)
    assert len(limited.request.candidates) == 1 and not limited.request.candidates[0].outlines


def test_forged_outlining_baselines_cannot_authorize_different_math(monkeypatch):
    import pllm.execution.region_optimization as tuning
    program, sources, inputs = chain()
    monkeypatch.setattr(tuning, "_measure", lambda execution, *args: 1 if len(execution._program.functions) == 3 else 100)
    options = ExecutionOptions(allow_numerical_transforms=False, schedule_policy="serial")
    selected = optimize_region(program, sources, inputs, payload_limits=(RegionPayloadLimit("cpu", 10000),),
        options=options, warmup_rounds=0, measurement_rounds=1)
    candidate = selected.request.candidates[1]
    proof, = candidate.outlines
    # Native selection can validate the closed graph's storage, but only local
    # tensor replay can prove that a claimed baseline contains the original ATen.
    forged_access = replace(proof.access, graph_digest="0" * 64)
    forged_candidate = replace(candidate,
        outlines=(replace(proof, access=forged_access),),
        imports=tuple(replace(i, access=forged_access) if i.access.source_digest == proof.access.source_digest else i
                      for i in candidate.imports))
    request = replace(selected.request, candidates=(selected.request.candidates[0], forged_candidate))
    decision = select_region_candidate(program, request)
    assert decision.selected == 1 and not decision.candidates[1].changed_functions
    with pytest.raises(ValueError, match="baseline differs"):
        bind_selected_region(program, sources, request, decision, (), options=options)
    for altered in (replace(proof, site=replace(proof.site, first=1)),
                    replace(proof, access=replace(proof.access, source_digest="forged"))):
        bad = replace(selected.request, candidates=(selected.request.candidates[0], replace(candidate, outlines=(altered,))))
        with pytest.raises(ValueError):
            select_region_candidate(program, bad)
    with pytest.raises(ValueError, match="original reference"):
        select_region_candidate(program, replace(selected.request, candidates=(candidate,)))


def test_outlining_remains_symbolic_and_zero_sample_does_not_bypass_numerical_policy(monkeypatch):
    import pllm.execution.region_optimization as tuning
    a = torch.ones(2, 2, dtype=torch.float64)
    b = torch.ones(2, 16, dtype=torch.float64)
    c = torch.ones(16, 2, dtype=torch.float64)
    first = ExportedCell.capture(lambda a, b: (a @ b,), ("a", "b"), ("ab",), (a, b))
    second = ExportedCell.capture(lambda ab, c: (ab @ c,), ("ab", "c"), ("out",), (a @ b, c))
    ta, tb = first.declaration.inputs
    tab, tc = second.declaration.inputs
    body = Region(Block((Value("i", ScalarType.INDEX), Value("a", ta), Value("b", tb), Value("c", tc)), (
        Call(("a", "b"), (Value("ab", tab),), first.digest),
        Call(("ab", "c"), (Value("next_a", ta),), second.digest),
    ), ("next_a", "b", "c")))
    program = Program(Region(Block((Value("a", ta), Value("b", tb), Value("c", tc)), (
        Scan(("a", "b", "c"), (Value("final", ta), Value("final_b", tb), Value("final_c", tc)), "steps", body),
    ), ("final",))), (first.declaration, second.declaration), parameters=(SizeParameter("steps", 0, 1_000_000),))
    sources = {first.digest: first, second.digest: second}
    monkeypatch.setattr(tuning, "_measure", lambda execution, *args: 1 if any(c.transformations for c in execution._cells.values()) else 100)
    with pytest.warns(NumericalTransformWarning):
        selected = optimize_region(program, sources, {"a": a, "b": b, "c": c},
            payload_limits=(RegionPayloadLimit("cpu", 10000),), options=ExecutionOptions(schedule_policy="serial"),
            sizes={"steps": 0}, warmup_rounds=0, measurement_rounds=1)
    candidate = selected.request.candidates[selected.selection.selected]
    assert candidate.outlines[0].site == CallOutline((0, 0), 0, 2)
    assert selected.request.tuning_sizes == (RegionTuningSize("steps", 0),)
    assert selected.selection.candidates[selected.selection.selected].changed_functions
    strict = select_region_candidate(program, replace(selected.request, allow_numerical_transforms=False))
    assert strict.selected == 0
    for count in (0, 1, 3):
        result = selected.run({"a": a, "b": b, "c": c}, sizes={"steps": count})
        assert result.invocations == count
        torch.testing.assert_close(result.values[0], torch.full_like(a, 32**count), rtol=0, atol=0)


def test_chained_structural_evidence_replays_against_each_intermediate_program():
    program, sources, inputs = chain()
    last_type = program.functions[-1].outputs[0]
    last = ExportedCell.capture(lambda x: (x + 1,), ("x",), ("next",), (torch.ones(4, 2, dtype=torch.float64),))
    original = replace(program, entry=Region(replace(program.entry.block,
        operations=(*program.entry.block.operations, Call(("out",), (Value("next", last_type),), last.digest)), yields=("next",))),
        functions=(*program.functions, last.declaration))
    sources = {**sources, last.digest: last}
    first = outline_region_calls(original, sources, CallOutline((), 0, 2))
    middle = {**sources, first.cell.digest: first.cell}
    second = outline_region_calls(first.outlined.program, middle, CallOutline((), 0, 2))
    final = {**middle, second.cell.digest: second.cell}
    records = tuple(CallOutlineEvidence(o.site, import_tensor_access_graph(o.cell)) for o in (first, second))
    source_imports = import_region_functions(original, sources)
    request = RegionSelectionRequest(source_imports, (
        RegionCandidate(analyze_region_program(original).reference_schedule, source_imports, (100,)),
        RegionCandidate(analyze_region_program(second.outlined.program).reference_schedule,
                        import_region_functions(second.outlined.program, final), (1,), records),
    ), (RegionPayloadLimit("cpu", 10000),), False, "deterministic structural selection")
    decision = select_region_candidate(original, request)
    selected = bind_selected_region(original, sources, request, decision, (),
                                    options=ExecutionOptions(allow_numerical_transforms=False, schedule_policy="serial"))
    result = selected.run(inputs)
    assert result.invocations == 1 and not decision.candidates[1].changed_functions
    torch.testing.assert_close(result.values[0], torch.full_like(result.values[0], 385), rtol=0, atol=0)
    with pytest.raises(ValueError):
        select_region_candidate(original, replace(request, candidates=(request.candidates[0],
            replace(request.candidates[1], outlines=records[::-1]))))
