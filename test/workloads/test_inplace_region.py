"""The production overlay binds a real region scan with original page publication."""

import pytest
import torch

from pllm import _core
from pllm.authoring import BackendCompileFacts, ParameterTargetShape
from pllm.contracts.backend_attachment import AttachmentGraphMode, AttachmentTopology, BackendPlacementProbe
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import CapturedParameterOperand, ParameterModuleKind, ParameterRole, ParameterTarget
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.tensor_scan import TensorScan
from pllm.workloads.inplace_ttt import InPlaceTTTConfig, InPlaceTTTWorkload
from test.workloads.test_inplace_ttt import _input


def facts_for(device, dtype, target):
    kind = StateDeviceKind.CUDA if device.type == "cuda" else StateDeviceKind.CPU
    return BackendCompileFacts(BackendPlacementProbe("test.region.overlay@1", "a" * 64, "b" * 64, "c" * 64,
        AttachmentTopology.IN_PROCESS, AttachmentGraphMode.EAGER, kind, 0, 1, 1,
        TensorDType(str(dtype).removeprefix("torch.")), 7, 8, (), ()),
        RuntimeFacts(device).timeline(("transition/full_root@1", "state/fenced_reclaim@1")),
        (PageBundleProviderFacts("test.region.full_root@1", 4, device_kind=kind),), 2, 65536, 0,
        (ParameterTargetShape(target, 7, 5),))


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
@pytest.mark.parametrize("dtype", (torch.float64, torch.float32, torch.bfloat16))
@pytest.mark.parametrize("transposed_base", (False, True))
def test_bound_scan_matches_original_chunk_math_bitwise_and_compiles_only_cold(
    monkeypatch, device, dtype, transposed_base,
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device(device)
    target = ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT)
    workload = InPlaceTTTWorkload(InPlaceTTTConfig((target,), chunk_size=3, learning_rate=0.017,
        update_decode=True))
    facts = facts_for(device, dtype, target)
    execution = workload.compile_backend(facts).execution
    stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
    identity = RequestIdentity("region", 0, 0)
    ref, = execution.admit((identity,), stream)
    reference = tuple(t.clone() for t in execution.snapshot(ref, stream))
    generator = torch.Generator(device=device).manual_seed(127)
    base = torch.randn((7, 5) if transposed_base else (5, 7), dtype=dtype, device=device, generator=generator)
    if transposed_base:
        base = base.T
    calls = []
    original = TensorScan.__call__
    def record(self, *args):
        assert self.program.parameters[0].name == "chunks"
        calls.append(len(args[2][0]))
        return original(self, *args)
    monkeypatch.setattr(TensorScan, "__call__", record)
    def cold_only(*args, **kwargs):
        raise AssertionError("compilation entered the request execution path")
    monkeypatch.setattr(ExportedCell, "capture", cold_only)
    for name in ("analyze_region_program", "lower_region_schedule", "lower_region_bindings"):
        monkeypatch.setattr(_core, name, cold_only)
    for rows, phase in ((1, HiddenPhase.PREFILL), (7, HiddenPhase.PREFILL),
                        (2, HiddenPhase.DECODE), (1, HiddenPhase.DECODE)):
        hidden = torch.randn(rows, 7, dtype=dtype, device=device, generator=generator)
        desired = torch.randn(rows, 5, dtype=dtype, device=device, generator=generator)
        expected = workload.transition(hidden, desired, base, *reference)
        result, = execution.step(_input((CapturedParameterOperand(identity, target, phase,
            hidden, desired, base),)), stream=stream).entries
        got = result.targets[0].update
        torch.testing.assert_close(got.output, expected.output, rtol=0, atol=0)
        assert (got.changed, got.update_count, got.trained_rows, got.metrics) == (
            expected.changed, expected.update_count, expected.trained_rows, expected.metrics)
        assert result.before_ref == ref
        assert result.after_ref.version_id == ref.version_id + 1
        ref = result.after_ref
        for a, b in zip(execution.snapshot(ref, stream), reference, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert calls == [2, 1]
    execution.retire((identity,))
    execution.arena.collect_completed(wait=True)
    assert execution.arena.active_lineages == 0


def test_customized_source_cannot_inherit_the_builtin_region_lowering():
    # A changed ordinary method may declare new costs, but that does not prove
    # the built-in region still implements its mathematics.
    class Customized(InPlaceTTTWorkload):
        def transition(self, hidden, target, base, delta, tail_input, tail_target, tail_length):
            return hidden
    original = InPlaceTTTWorkload.transition
    Customized.transition.__annotations__ = original.__annotations__
    for name in ("__pllm_phase__", "__pllm_effect__"):
        setattr(Customized.transition, name, getattr(original, name))
    target = ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT)
    workload = Customized(InPlaceTTTConfig((target,), chunk_size=3))
    facts = facts_for(torch.device("cpu"), torch.float32, target)
    with pytest.raises(ValueError, match="explicit matching native implementation"):
        workload.compile_backend(facts)


def test_chunk_capacity_does_not_create_a_length_specialization_family():
    from pllm.workloads.inplace_ttt import _ChunkTransition, _chunk_region
    from pllm.contracts.region_program import SizeParameter
    regions = tuple(_chunk_region(.017, 3, 7, 5, torch.float32, torch.device("cpu"), transposed)
        for transposed in (False, True))
    for capacity in (8, 1_000_000):
        binding = _ChunkTransition(3, regions, capacity)
        assert len(binding.scans) == 2
        for scan in binding.scans.values():
            assert scan.count == SizeParameter("chunks", 0, (capacity + 2) // 3)
            assert len(scan._execution._lowered.statements) == 3


def test_chunk_region_isl_moves_independent_updates_without_changing_forward_versions():
    from pllm.workloads.inplace_ttt import _chunk_region
    from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule, preview_region_schedule
    from pllm.execution.region_execution import RegionExecution
    from pllm.execution.region_values import Bank
    program, cells = _chunk_region(.017, 3, 7, 5, torch.float64, torch.device("cpu"), False)
    functions = {cell.digest: cell for cell in cells}
    scan = TensorScan(program, 3, carry=1, invariants=1, functions=functions)
    generator = torch.Generator().manual_seed(21)
    delta = torch.randn(5, 7, dtype=torch.float64, generator=generator)
    base = torch.randn(5, 7, dtype=torch.float64, generator=generator)
    xs = tuple(torch.randn(3, 7, dtype=torch.float64, generator=generator) for _ in range(3))
    ys = tuple(torch.randn(3, 5, dtype=torch.float64, generator=generator) for _ in range(3))
    expected, current = [], delta.clone()
    for x, y in zip(xs, ys, strict=True):
        expected.append(x @ (base + current).T)
        current.add_(.017 * (y.T @ x))
    inputs = {"arg0": delta, "arg1": base, "arg2": Bank(xs), "arg3": Bank(ys)}
    analysis = analyze_region_program(scan.program)
    orders = set()
    for candidate in analysis.candidates:
        orders.add(tuple((item.statement_id, item.coordinates)
            for item in preview_region_schedule(scan.program, candidate.schedule_map)))
        run = RegionExecution(scan.program, lower_region_schedule(scan.program, candidate.schedule_map),
            lower_region_bindings(scan.program, candidate.schedule_map), functions)
        final, column = run.run(inputs).values
        torch.testing.assert_close(final, current, rtol=0, atol=0)
        for actual, want in zip(column.elements, expected, strict=True):
            torch.testing.assert_close(actual, want, rtol=0, atol=0)
    # This order comes from ISL: compute independent increments, then carry
    # updates, then forwards. Each forward must still read its own old delta.
    assert tuple((statement, (i,)) for statement in (1, 2, 0) for i in range(3)) in orders
    assert len(orders) > 1


def test_production_tuning_uses_signed_policy_and_remaining_transition_budget(monkeypatch):
    from dataclasses import replace
    import pllm
    import pllm.execution.region_optimization as tuning
    from pllm.contracts.region_selection import RegionSelectionPolicy
    from pllm.execution.codec import decode_boundary, encode_boundary
    target = ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT)
    workload = InPlaceTTTWorkload(InPlaceTTTConfig((target,), chunk_size=3))
    facts = workload.specialize(facts_for(torch.device("cpu"), torch.float32, target),
        pllm.ExecutionOptions(allow_numerical_transforms=False, schedule_policy="serial", memory_budget_bytes=1 << 20))
    policy = RegionSelectionPolicy(False, "serial")
    assert facts.timeline.region_policy == policy
    facts = decode_boundary(encode_boundary(facts), BackendCompileFacts)
    attachment = workload.plan_backend(facts)
    assert next(f.workspace_bytes for f in attachment.inputs.plan.timeline.task_resources if f.task == "transition") == 2752
    selected = []
    original = TensorScan.optimized
    def optimize(self, *args, **kwargs):
        assert kwargs["options"] == policy
        # Eight packed rows plus two request tails, with the external buffers
        # deducted from the transition's original declared workspace.
        assert kwargs["payload_limits"][0].bytes == 4 * (12 * (7 + 2 * 5) + 6 * 7 * 5)
        result = original(self, *args, **kwargs)
        selected.append(result.selection)
        return result
    def forbidden(*args, **kwargs):
        raise AssertionError("numerical optout reached backend compilation")
    monkeypatch.setattr(TensorScan, "optimized", optimize)
    monkeypatch.setattr(tuning, "compile_tensor_cell", forbidden)
    monkeypatch.setattr(tuning, "_measure", lambda execution, *args: len(execution._lowered.statements))
    rng = torch.get_rng_state().clone()
    workload.bind_backend(facts, attachment)
    assert torch.equal(rng, torch.get_rng_state())
    assert len(selected) == 2 and all(s.selected != 0 for s in selected)
    assert attachment.inputs.plan.timeline.region_policy == policy
    with pytest.raises(ValueError, match="worker backend facts"):
        workload.bind_backend(replace(facts, timeline=replace(facts.timeline,
            region_policy=RegionSelectionPolicy())), attachment)


def test_chunk_candidates_exceeding_the_remaining_budget_are_not_measured(monkeypatch):
    import pllm.execution.region_optimization as tuning
    from pllm.contracts.region_selection import RegionSelectionPolicy
    from pllm.workloads.inplace_ttt import _chunk_region, _ChunkTransition, _tune_chunk_scan
    program, cells = _chunk_region(.017, 3, 7, 5, torch.float32, torch.device("cpu"), False)
    transition = _ChunkTransition(3, ((program, cells),), 8)
    scan, = transition.scans.values()
    outside = 4 * (12 * (7 + 2 * 5) + 2 * 7 * 5)
    def forbidden(*args, **kwargs):
        raise AssertionError("an infeasible production candidate was measured")
    monkeypatch.setattr(tuning, "_measure", forbidden)
    with pytest.raises(ValueError, match="no region candidate satisfies"):
        _tune_chunk_scan(scan, cells, 12, outside + 1, RegionSelectionPolicy(False, "serial"))
