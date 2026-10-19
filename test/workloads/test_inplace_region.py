"""The production overlay binds a real region scan with original page publication."""

import pytest
import torch

from pllm import _core
from pllm.authoring import BackendCompileFacts, ParameterTargetShape
from pllm.contracts.backend_attachment import AttachmentGraphMode, AttachmentTopology, BackendPlacementProbe
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import (
    CapturedParameterOperand,
    ParameterModuleKind,
    ParameterRole,
    ParameterTarget,
)
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
    return BackendCompileFacts(
        BackendPlacementProbe(
            "test.region.overlay@1",
            "a" * 64,
            "b" * 64,
            "c" * 64,
            AttachmentTopology.IN_PROCESS,
            AttachmentGraphMode.EAGER,
            kind,
            0,
            1,
            1,
            TensorDType(str(dtype).removeprefix("torch.")),
            7,
            8,
            (),
            (),
        ),
        RuntimeFacts(device).timeline(("transition/full_root@1", "state/fenced_reclaim@1")),
        (PageBundleProviderFacts("test.region.full_root@1", 4, device_kind=kind),),
        2,
        65536,
        0,
        (ParameterTargetShape(target, 7, 5),),
    )


@pytest.mark.parametrize(
    "device,dtype,transposed_base",
    (
        ("cpu", torch.float32, True),
        ("cuda:0", torch.bfloat16, True),
    ),
)
def test_bound_scan_matches_original_chunk_math_bitwise_and_compiles_only_cold(
    monkeypatch,
    device,
    dtype,
    transposed_base,
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device(device)
    target = ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT)
    workload = InPlaceTTTWorkload(InPlaceTTTConfig((target,), chunk_size=3, learning_rate=0.017, update_decode=True))
    facts = facts_for(device, dtype, target)
    execution = workload.compile_backend(facts).execution
    stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
    identity = RequestIdentity("region", 0, 0)
    (ref,) = execution.admit((identity,), stream)
    reference = tuple(t.clone() for t in execution.snapshot(ref, stream))
    generator = torch.Generator(device=device).manual_seed(127)
    base = torch.randn((7, 5) if transposed_base else (5, 7), dtype=dtype, device=device, generator=generator)
    if transposed_base:
        base = base.T

    def cold_only(*args, **kwargs):
        raise AssertionError("compilation entered the request execution path")

    monkeypatch.setattr(ExportedCell, "capture", cold_only)
    for name in ("analyze_region_program", "lower_region_schedule", "lower_region_bindings"):
        monkeypatch.setattr(_core, name, cold_only)
    for rows, phase in (
        (1, HiddenPhase.PREFILL),
        (7, HiddenPhase.PREFILL),
        (2, HiddenPhase.DECODE),
        (1, HiddenPhase.DECODE),
    ):
        hidden = torch.randn(rows, 7, dtype=dtype, device=device, generator=generator)
        desired = torch.randn(rows, 5, dtype=dtype, device=device, generator=generator)
        expected = workload.transition(hidden, desired, base, *reference)
        (result,) = execution.step(
            _input((CapturedParameterOperand(identity, target, phase, hidden, desired, base),)), stream=stream
        ).entries
        got = result.targets[0].update
        torch.testing.assert_close(got.output, expected.output, rtol=0, atol=0)
        assert (got.changed, got.update_count, got.trained_rows, got.metrics) == (
            expected.changed,
            expected.update_count,
            expected.trained_rows,
            expected.metrics,
        )
        assert result.before_ref == ref
        assert result.after_ref.version_id == ref.version_id + 1
        ref = result.after_ref
        for a, b in zip(execution.snapshot(ref, stream), reference, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
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


def test_production_tuning_uses_signed_policy_and_remaining_transition_budget(monkeypatch):
    from dataclasses import replace

    import pllm
    import pllm.execution.region_optimization as tuning
    from pllm.contracts.region_selection import RegionSelectionPolicy
    from pllm.execution.codec import decode_boundary, encode_boundary

    target = ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT)
    workload = InPlaceTTTWorkload(InPlaceTTTConfig((target,), chunk_size=3))
    facts = workload.specialize(
        facts_for(torch.device("cpu"), torch.float32, target),
        pllm.ExecutionOptions(
            allow_numerical_transforms=False, schedule_policy="serial", tuning="measured", memory_budget_bytes=1 << 20
        ),
    )
    policy = RegionSelectionPolicy(False, "serial", "measured")
    assert facts.timeline.region_policy == policy
    facts = decode_boundary(encode_boundary(facts), BackendCompileFacts)
    attachment = workload.plan_backend(facts)
    selected = []
    original = TensorScan.optimized

    def optimize(self, *args, **kwargs):
        assert kwargs["options"] == policy
        result = original(self, *args, **kwargs)
        selected.append(result[1].selection)
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
        workload.bind_backend(
            replace(facts, timeline=replace(facts.timeline, region_policy=RegionSelectionPolicy())), attachment
        )
