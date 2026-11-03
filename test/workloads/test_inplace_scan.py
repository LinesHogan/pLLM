"""The production In-Place TTT overlay runs a bound region scan with page publication.

Its outputs match the workload's own chunk transition bitwise on CPU and CUDA, and
the measured region-tuning policy is signed into the backend facts.
"""

import warnings
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import pllm
from pllm.contracts.codec import decode_boundary, encode_boundary
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import (
    Projection,
    ProjectionOperands,
    ProjectionShape,
)
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.contracts.tensor_policy import TensorPolicy
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.overlay_task import ChunkUpdate
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import model_timeline
from pllm.execution.scan import ScanContext, ScanTemplate
from pllm.execution.scan_analysis import analyze
from pllm.testing import backend_facts
from test.dense_update import DenseUpdate, DenseUpdateConfig
from test.workloads.test_inplace_ttt import _input

TARGET = Projection("down_proj", 0)


def facts_for(device, dtype):
    timeline = model_timeline(device, ("transition/full_root@1", "state/fenced_reclaim@1"))
    provider = PageBundleProviderFacts("test.region.full_root@1", 4, StateDeviceKind(device.type))
    return backend_facts(
        7,
        (),
        timeline,
        provider,
        memory_bytes=65536,
        max_rows=8,
        dtype=TensorDType(str(dtype).removeprefix("torch.")),
        targets=(ProjectionShape(TARGET, 7, 5),),
    )


@pytest.mark.parametrize(
    "device,dtype", (("cpu", torch.float32), pytest.param("cuda:0", torch.bfloat16, marks=pytest.mark.gpu))
)
@pytest.mark.parametrize("numerics", (False, True))
def test_bound_scan_matches_original_chunk_math_bitwise(device, dtype, numerics):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device(device)
    workload = DenseUpdate(DenseUpdateConfig((TARGET,), chunk_size=3, learning_rate=0.017, tokens="all"))
    facts = facts_for(device, dtype)
    facts = replace(facts, timeline=replace(facts.timeline, tensor_policy=TensorPolicy(numerics)))
    execution = workload.compile_backend(facts).execution
    reference_transition = ScanContext(TensorPolicy(False), 8, device).bind(workload.build(facts).overlay_math.function)
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
        expected = reference_transition(
            delta=reference[0],
            inputs=hidden,
            targets=desired,
            _pending_rows_inputs=reference[1],
            _pending_rows_targets=reference[2],
            _pending_length=reference[3],
        )
        operand = ProjectionOperands(
            identity, TARGET, phase, hidden, desired, base, hidden.new_full((), False, dtype=torch.bool)
        )
        (result,) = execution.step(_input((operand,)), stream=stream).entries
        got = result.targets[0].update
        tolerance = {"rtol": 0.02, "atol": 0.001} if numerics else {"rtol": 0, "atol": 0}
        torch.testing.assert_close(got.output, expected.output, **tolerance)
        assert (got.changed, got.update_count, got.trained_rows, got.metrics) == (
            expected.changed,
            expected.update_count,
            expected.trained_rows,
            expected.metrics,
        )
        assert result.before_ref == ref and result.after_ref.version_id == ref.version_id + 1
        ref = result.after_ref
        for a, b in zip(execution.snapshot(ref, stream), reference, strict=True):
            torch.testing.assert_close(a, b, **tolerance)
    execution.retire((identity,))
    assert execution.arena.active_states == 0


def test_production_tuning_policy_is_signed_into_worker_facts():
    workload = DenseUpdate(DenseUpdateConfig((TARGET,), chunk_size=3))
    options = pllm.ExecutionOptions(allow_numerical_transforms=False, tuning="measured", memory_budget_bytes=1 << 20)
    facts, _ = workload.specialize(facts_for(torch.device("cpu"), torch.float32), options)
    policy = TensorPolicy(False, "measured")
    assert facts.timeline.tensor_policy == policy
    # The policy crosses the process boundary inside the facts the worker binds with.
    facts = decode_boundary(encode_boundary(facts), BackendCompileFacts)
    declaration = workload.build(facts)
    attachment = workload.plan_backend(facts, declaration)
    rng = torch.get_rng_state().clone()
    workload.bind_backend(facts, attachment, declaration)
    assert torch.equal(rng, torch.get_rng_state())  # measurement does not consume the global RNG
    assert attachment.plan.timeline.tensor_policy == policy


@pytest.mark.parametrize("device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)))
def test_scan_matches_the_serial_loop_including_carry_views(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required")

    def step(carry, x):
        # Views of the carry are rematerialized inside the cell that consumes them.
        view = carry.view_as(carry).reshape(1, -1).permute(1, 0).transpose(0, 1).expand(2, -1)[1]
        return carry + view + x.sum(0), x + carry

    for rows in (0, 1, 6):
        carry, over = torch.zeros(7, device=device), torch.ones(rows, 7, device=device)
        expected, outputs = carry, []
        for start in range(0, rows, 3):
            expected, output = step(expected, over[start : start + 3])
            outputs.append(output)
        actual = pllm.scan(step, carry=carry, over=(over,), chunk=3)
        torch.testing.assert_close(actual, (expected, tuple(outputs)), rtol=0, atol=0)

    if device == "cpu":
        return
    generator = torch.Generator(device=device).manual_seed(315)
    carry = torch.randn(11, 13, device=device, generator=generator) * 0.01
    x = torch.randn(20, 13, device=device, generator=generator)
    y = torch.randn(20, 11, device=device, generator=generator)
    workload = DenseUpdate(DenseUpdateConfig((TARGET,), chunk_size=4, learning_rate=0.013))

    for kind in ("additive", "nonlinear", "bilinear", "downcast", "random", "alpha"):

        def transition(state, inputs, targets):
            result, output = workload.step(state, inputs, targets)
            if kind == "nonlinear":
                result = result.tanh()
            elif kind == "bilinear":
                result = result * state
            elif kind == "downcast":
                result = result.to(torch.bfloat16).float()
            elif kind == "random":
                result = result + torch.rand_like(result)
            elif kind == "alpha":
                result = torch.add(state, result - state, alpha=2)
            return result, output

        with FakeTensorMode():
            samples = tuple(torch.empty(shape, device=device) for shape in ((11, 13), (4, 13), (4, 11)))
            classification = analyze(transition, tuple((t.shape, t.stride(), t.dtype, t.device) for t in samples))
        if kind in ("nonlinear", "bilinear", "downcast", "random"):
            assert classification is None
        else:
            assert classification is not None and classification.kind == "additive"
        reference, outputs = carry, []
        rng = torch.cuda.get_rng_state()
        for xx, yy in zip(x.split(4), y.split(4), strict=True):
            reference, output = transition(reference, xx, yy)
            outputs.append(output)
        for numerics in (False, True) if kind == "additive" else (True,):
            torch.cuda.set_rng_state(rng)
            context = ScanContext(TensorPolicy(numerics), 20, torch.device(device))
            with warnings.catch_warnings(record=True) as reported:
                actual = context.bind(pllm.scan)(transition, carry=carry, over=(x, y), chunk=4)
            tolerance = {"rtol": 1e-5, "atol": 1e-6} if numerics else {"rtol": 0, "atol": 0}
            torch.testing.assert_close(actual, (reference, tuple(outputs)), **tolerance)
            if kind not in ("additive", "alpha") or not numerics:
                assert not any("additive scan" in str(item.message) for item in reported)
    # Alias guards are checked on each invocation, not inferred from fake samples.
    alias_y = x[:, :11]
    reference, outputs = carry, []
    for xx, yy in zip(x.split(4), alias_y.split(4), strict=True):
        reference, output = workload.step(reference, xx, yy)
        outputs.append(output)
    actual = pllm.scan(workload.step, carry=carry, over=(x, alias_y), chunk=4)
    torch.testing.assert_close(actual, (reference, tuple(outputs)), rtol=0, atol=0)
    # Prime bounded buckets with real C, then vary C within them without any
    # warm export. The measured policy must not specialize on every count.
    x, y = torch.cat((x, x[:12])), torch.cat((y, y[:12]))
    context = ScanContext(TensorPolicy(True, tuning="measured"), 32, torch.device(device))
    run = context.bind(pllm.scan)
    run(workload.step, carry=carry, over=(x[:20], y[:20]), chunk=4)
    with (
        patch("pllm.execution.scan.analyze", side_effect=AssertionError("warm C caused a trace")),
        patch.object(ScanTemplate, "_prepare", side_effect=AssertionError("warm C caused plan preparation")),
    ):
        for count in (0, 6, 5, 8, 7):
            expected, outputs = carry, []
            for xx, yy in zip(x[: 4 * count].split(4), y[: 4 * count].split(4), strict=True):
                if not len(xx):
                    continue
                expected, output = workload.step(expected, xx, yy)
                outputs.append(output)
            actual = run(workload.step, carry=carry, over=(x[: 4 * count], y[: 4 * count]), chunk=4)
            torch.testing.assert_close(actual, (expected, tuple(outputs)), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("numerics", (False, True))
@pytest.mark.gpu
def test_counted_gated_tail_keeps_inactive_chunks_and_never_reads_device_counts(numerics):
    """An independent chunk oracle covers C=0, carried tails and inactive capacity."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from pllm.kernels.head_overlay import head_correction
    from pllm.workloads.gated_delta import GatedDeltaConfig, GatedDeltaWorkload

    target = Projection("q_proj", 0)
    workload = GatedDeltaWorkload(GatedDeltaConfig((target,), head_width=4, chunk_size=4))
    context = ScanContext(TensorPolicy(numerics), 32, torch.device("cuda:0"), 64 << 20)
    run = context.bind(ChunkUpdate(workload.step, 4, ("state", "hidden")))
    generator = torch.Generator(device="cuda").manual_seed(512)
    weights = torch.randn(2, 4, 4, device="cuda", generator=generator) * 0.1
    tail = torch.zeros(3, 8, device="cuda")
    length = torch.zeros((), dtype=torch.int64, device="cuda")
    state, retained = weights.clone(), tail[:0].clone()
    for count in (1, 1, 9, 2, 17, 3):
        hidden = torch.randn(count, 8, device="cuda", generator=generator)
        joined = torch.cat((retained, hidden))
        complete = len(joined) // 4 * 4
        expected = []
        for start in range(0, complete, 4):
            x = joined[start : start + 4].view(4, 2, 4).transpose(0, 1)
            k = torch.nn.functional.normalize(x, dim=-1)
            expected.append((x @ state.transpose(1, 2)).transpose(0, 1).reshape(4, 8))
            decayed = state * workload.config.alpha
            error = k @ decayed.transpose(1, 2) - x
            state = decayed - workload.config.beta / 4 * (error.transpose(1, 2) @ k)
        retained = joined[complete:]
        expected.append(head_correction(retained, state))
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("device count read")):
            result = run(state=weights, hidden=hidden, _pending_rows_hidden=tail, _pending_length=length)
        torch.testing.assert_close(result.output, torch.cat(expected)[-count:], rtol=2e-5, atol=1e-6)
        torch.testing.assert_close(weights, state, rtol=2e-5, atol=1e-6)
        torch.testing.assert_close(tail[: len(retained)], retained, rtol=0, atol=0)
        assert int(length) == len(retained) and int(result.trained_rows) == complete
        assert bool(result.changed) == (complete > 0)
    # Warm the same source/layout, then forbid every host CUDA synchronization.
    torch.cuda.synchronize()
    with (
        patch.object(torch.Tensor, "item", side_effect=AssertionError("scalar read")),
        patch.object(torch.cuda.Event, "synchronize", side_effect=AssertionError("host event wait")),
        patch.object(torch.cuda.Stream, "synchronize", side_effect=AssertionError("host stream wait")),
    ):
        run(state=weights, hidden=hidden, _pending_rows_hidden=tail, _pending_length=length)
