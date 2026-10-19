"""The production In-Place TTT overlay runs a bound region scan with page publication.

Its outputs match the workload's own chunk transition bitwise on CPU and CUDA, and
the measured region-tuning policy is signed into the backend facts.
"""

from dataclasses import replace

import pytest
import torch

import pllm
from pllm.authoring import BackendCompileFacts, ParameterTargetShape
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import (
    CapturedParameterOperand,
    ParameterModuleKind,
    ParameterRole,
    ParameterTarget,
)
from pllm.contracts.region_selection import RegionSelectionPolicy
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.workloads.inplace_ttt import InPlaceTTTConfig, InPlaceTTTWorkload
from test.backend_facts import backend_facts
from test.workloads.test_inplace_ttt import _input

TARGET = ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT)


def facts_for(device, dtype):
    timeline = RuntimeFacts(device).timeline(("transition/full_root@1", "state/fenced_reclaim@1"))
    provider = PageBundleProviderFacts("test.region.full_root@1", 4, StateDeviceKind(device.type))
    return backend_facts(
        7,
        (),
        timeline,
        provider,
        memory_bytes=65536,
        max_rows=8,
        dtype=TensorDType(str(dtype).removeprefix("torch.")),
        targets=(ParameterTargetShape(TARGET, 7, 5),),
    )


@pytest.mark.parametrize("device,dtype", (("cpu", torch.float32), ("cuda:0", torch.bfloat16)))
def test_bound_scan_matches_original_chunk_math_bitwise(device, dtype):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device(device)
    workload = InPlaceTTTWorkload(InPlaceTTTConfig((TARGET,), chunk_size=3, learning_rate=0.017, update_decode=True))
    execution = workload.compile_backend(facts_for(device, dtype)).execution
    stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
    identity = RequestIdentity("region", 0, 0)
    (ref,) = execution.admit((identity,), stream)
    reference = tuple(t.clone() for t in execution.snapshot(ref, stream))
    generator = torch.Generator(device=device).manual_seed(127)
    # A transposed (non-contiguous) base weight, as vLLM linear layers store it.
    base = torch.randn((7, 5), dtype=dtype, device=device, generator=generator).T
    # Partial, full and multi-chunk prefill, then decode rows.
    for rows, phase in (
        (1, HiddenPhase.PREFILL),
        (7, HiddenPhase.PREFILL),
        (2, HiddenPhase.DECODE),
        (1, HiddenPhase.DECODE),
    ):
        hidden = torch.randn(rows, 7, dtype=dtype, device=device, generator=generator)
        desired = torch.randn(rows, 5, dtype=dtype, device=device, generator=generator)
        expected = workload.transition(hidden, desired, base, *reference)
        operand = CapturedParameterOperand(identity, TARGET, phase, hidden, desired, base)
        (result,) = execution.step(_input((operand,)), stream=stream).entries
        got = result.targets[0].update
        torch.testing.assert_close(got.output, expected.output, rtol=0, atol=0)
        assert (got.changed, got.update_count, got.trained_rows, got.metrics) == (
            expected.changed,
            expected.update_count,
            expected.trained_rows,
            expected.metrics,
        )
        assert result.before_ref == ref and result.after_ref.version_id == ref.version_id + 1
        ref = result.after_ref
        for a, b in zip(execution.snapshot(ref, stream), reference, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    execution.retire((identity,))
    assert execution.arena.active_lineages == 0


def test_production_tuning_policy_is_signed_into_worker_facts():
    workload = InPlaceTTTWorkload(InPlaceTTTConfig((TARGET,), chunk_size=3))
    options = pllm.ExecutionOptions(
        allow_numerical_transforms=False, schedule_policy="serial", tuning="measured", memory_budget_bytes=1 << 20
    )
    facts = workload.specialize(facts_for(torch.device("cpu"), torch.float32), options)
    policy = RegionSelectionPolicy(False, "serial", "measured")
    assert facts.timeline.region_policy == policy
    # The policy crosses the process boundary inside the facts the worker binds with.
    facts = decode_boundary(encode_boundary(facts), BackendCompileFacts)
    attachment = workload.plan_backend(facts)
    rng = torch.get_rng_state().clone()
    workload.bind_backend(facts, attachment)
    assert torch.equal(rng, torch.get_rng_state())  # measurement does not consume the global RNG
    assert attachment.inputs.plan.timeline.region_policy == policy
    with pytest.raises(ValueError):  # worker facts must carry the signed policy
        workload.bind_backend(
            replace(facts, timeline=replace(facts.timeline, region_policy=RegionSelectionPolicy())), attachment
        )
