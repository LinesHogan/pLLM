"""The production In-Place TTT overlay runs a bound region scan with page publication.

Its outputs match the workload's own chunk transition bitwise on CPU and CUDA, and
the measured region-tuning policy is signed into the backend facts.
"""

import warnings
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import pllm
from pllm.contracts.codec import decode_boundary, encode_boundary
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import (
    Projection,
    ProjectionOperands,
)
from pllm.contracts.tensor_policy import TensorPolicy
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.overlay_task import ChunkUpdate
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.scan import ScanContext, ScanTemplate
from pllm.execution.scan_analysis import analyze
from pllm.testing import RequestIdentity, backend_facts
from test.dense_update import DenseUpdate, DenseUpdateConfig
from test.workloads.test_inplace_ttt import projection_input

TARGET = Projection("down_proj", 0)


def facts_for(device, dtype):
    return backend_facts(
        7,
        device=device.type,
        pages=4,
        memory_bytes=65536,
        max_rows=8,
        dtype=dtype,
        targets={TARGET: (7, 5)},
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
            identity,
            TARGET,
            phase,
            hidden,
            desired,
            base,
            hidden.new_full((), False, dtype=torch.bool),
            embedding=hidden.new_empty((0, 0)),
        )
        arena = execution.arena
        pin = arena.pin_exact(StateRefs.of((ref,)), stream)
        before = arena.snapshot(pin, ref)
        with patch.object(arena, "fill_versions", wraps=arena.fill_versions) as fill:
            (result,) = execution.step(projection_input((operand,)), stream=stream).entries
        assert not any(copies for (copies,), _ in fill.call_args_list)
        torch.testing.assert_close(arena.snapshot(pin, ref), before, rtol=0, atol=0)
        arena.wait_release(arena.release_pin(pin))
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
    options = pllm.ExecutionOptions(
        allow_numerical_transforms=False, tuning="measured", execution_peak_budget_bytes=1 << 20
    )
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
        patch.object(ScanTemplate, "prepare_plan", side_effect=AssertionError("warm C caused plan preparation")),
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


@pytest.mark.parametrize("read", ("local", "coupled", "broadcast"))
def test_decode_readout_narrows_only_independent_rows(read):
    """A mean over rows and a row-shaped constant must retain full-chunk execution."""
    from pllm.execution.scan_analysis import row_readout

    def step(carry, inputs):
        output = inputs @ carry
        if read == "coupled":
            output = output + inputs.mean(0)
        elif read == "broadcast":
            output = output + carry
        return carry, output

    carry, inputs = torch.randn(4, 4), torch.randn(4, 4)
    specs = tuple((value.shape, value.stride(), value.dtype, value.device) for value in (carry, inputs))
    selected = row_readout(step, specs, 1)
    assert (selected is not None) == (read == "local")
    if selected is not None:
        torch.testing.assert_close(selected(carry, inputs[-1:]), step(carry, inputs)[1][-1:])


@pytest.mark.parametrize("dense", (False, True))
def test_decode_graph_only_scans_completing_lanes(device_name, dense):
    """A fresh carry can inherit on one replay and be overwritten on the next."""
    from pllm.execution.cuda_stream import PinnedStaging
    from pllm.execution.overlay_batch import DecodeOverlay
    from pllm.execution.scan_analysis import DenseReadout
    from pllm.execution.scan_execution import PreparedScan
    from pllm.execution.step_capture import capture_graph

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA conditional graphs")
    math = (
        DenseUpdate(DenseUpdateConfig((TARGET,), chunk_size=3, tokens="all", visibility="exact"))
        .build(backend_facts(1, targets={TARGET: (1, 1)}))
        .overlay_math
    )
    pools = (
        torch.zeros(4, 1, 1, device=device),
        torch.zeros(4, 3, 1, device=device),
        torch.zeros(4, 3, 1, device=device),
        torch.zeros(4, dtype=torch.long, device=device),
    )
    carry, inputs, targets, lengths = pools
    calls = torch.zeros((), device=device)

    def step(state, x, y):
        calls.add_(1)
        return state + y.T @ x, x @ state.T

    # This test supplies the device slot table directly, including a padded third lane.
    with patch("pllm.execution.overlay_batch.PinnedStaging", return_value=PinnedStaging()):
        decode = DecodeOverlay(math, 0, pools, {}, 3, TensorPolicy(), math.keys)
    decode.plan = PreparedScan(step, 3)
    decode.read_full = lambda state, x, y: x @ state.T
    if dense:
        decode.dense_readout = DenseReadout(0, False)
    decode.table[:4] = torch.tensor([0, 1, 0], device=device)
    decode.table[4:8] = torch.tensor([2, 3, -1], device=device)
    decode.table[-1] = torch.tensor([1, 1, 0], device=device)
    carry[:2, 0, 0] = torch.tensor([2.0, 5.0], device=device)
    x = torch.tensor([[3.0], [4.0], [0.0]], device=device)
    y = x + 1
    empty = x[:0]

    def body():
        return decode.run(x, y, empty, empty)

    if device.type == "cuda":
        graph, output, _ = capture_graph(device, body)
    for i, pending in enumerate(((2, 0), (1, 2), (0, 0))):
        decode.table[8:10].fill_(10 + i)
        if i == 0:
            for pool in pools:
                pool[2:].fill_(99)
        lengths[:2] = torch.tensor(pending, device=device)
        inputs[:2].zero_()
        targets[:2].zero_()
        for lane, n in enumerate(pending):
            inputs[lane, :n].fill_(2)
            targets[lane, :n].fill_(3)
        incoming = tuple(pool[:2].clone() for pool in pools)
        calls.zero_()
        if device.type == "cuda":
            graph.replay()
            assert int(calls) == pending.count(2)
        else:
            output = body()
        torch.testing.assert_close(output[:2], x[:2] * incoming[0][:, 0])
        for lane, n in enumerate(pending):
            complete = n + 1 == 3
            dst = lane + 2
            expected = incoming[0][lane] + (6 * n + y[lane] * x[lane] if complete else 0)
            torch.testing.assert_close(carry[dst], expected)
            assert int(lengths[dst]) == (n + 1) % 3
            for old, pool, row in ((incoming[1], inputs, x), (incoming[2], targets, y)):
                tail = torch.zeros_like(pool[dst])
                if not complete:
                    tail[:n].copy_(old[lane, :n])
                    tail[n].copy_(row[lane])
                torch.testing.assert_close(pool[dst], tail)
        torch.testing.assert_close(tuple(pool[:2] for pool in pools), incoming)

    # Abort a second visit to this allocation after the row write but before its bound is recorded.
    from pllm.kernels import page_copy

    original_write = page_copy._write_tail_cpu

    def interrupted_write(*args):
        original_write(*args)
        raise RuntimeError("interrupted append")

    lane = slice(0, 1)
    with (
        patch("pllm.kernels.page_copy._write_tail_cpu", interrupted_write),
        pytest.raises(RuntimeError, match="interrupted"),
    ):
        decode.append(1, lane, x[:1], torch.ones_like(lengths[:1]))
    torch.testing.assert_close(inputs[2, 1], x[0])
    decode.table[8, 0].add_(1)
    decode.append(1, lane, x[:1], torch.zeros_like(lengths[:1]))
    torch.testing.assert_close(inputs[2, 1:], torch.zeros_like(inputs[2, 1:]))


def test_decode_append_touches_only_prefix_and_stale_rows(device_name):
    from pllm.kernels.page_copy import append_tail

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    # A guard row between pages also catches a flattened copy that ignores the page stride.
    storage = torch.full((4, 1025, 17), -9.0, device=device)
    pages = storage[:, :-1]
    pages[0, :2].fill_(3)
    original = storage.clone()
    rows = torch.arange(17 * 3, device=device, dtype=pages.dtype).reshape(17, 3).T
    bounds = torch.tensor([[-1, 0], [9, 5], [10, 3], [-1, 0]], dtype=torch.long, device=device)
    src, dst, lengths, gens = torch.tensor([[0, 2, 0], [1, 2, -1], [2, 1, 0], [10, 10, 10]], device=device)
    append_tail(pages, rows, bounds, src, dst, lengths, gens)
    expected = original.clone()
    expected[1, :2] = 3
    expected[1, 2] = rows[0]
    expected[1, 3:5] = 0
    expected[2, 1] = rows[1]
    expected[2, 2] = 0
    torch.testing.assert_close(storage, expected, rtol=0, atol=0)
    assert bounds.tolist() == [[-1, 0], [10, 3], [10, 2], [-1, 0]]


def test_overlay_tile_clears_padded_counts_across_staging_rotation():
    """Shrinking active request counts across PinnedStaging rotation must not leak stale row counts into padding."""
    from pllm.execution.overlay_batch import OverlayTile
    from pllm.workloads.gated_delta import GatedDeltaConfig, GatedDeltaWorkload

    q_target = Projection("q_proj", 0)
    facts = backend_facts(6, device="cpu", pages=6, memory_bytes=65536, max_rows=8, targets={q_target: (6, 6)})
    math = GatedDeltaWorkload(GatedDeltaConfig((q_target,), head_width=3, chunk_size=3)).build(facts).overlay_math
    assert math is not None
    pools = (
        torch.zeros(6, 2, 3, 3),
        torch.zeros(6, 2, 6),
        torch.zeros(6, dtype=torch.long),
    )
    pools[2][:2] = torch.tensor([2, 2])
    base = torch.zeros(6, 6)

    def item(name: str, index: int) -> ProjectionOperands:
        return ProjectionOperands(
            RequestIdentity(name, index, 0),
            q_target,
            HiddenPhase.PREFILL,
            torch.ones(1, 6),
            torch.ones(1, 6),
            base,
            torch.tensor(False),
            torch.empty(0, 0),
        )

    tile = OverlayTile(math, pools, item("a", 0), states=4, rows=4, train=True, invariants={})
    assert tile.batchable
    for _ in range(2):
        reads = np.array([[0, 1], [0, 1], [0, 1]], dtype=np.int64)
        writes = np.array([[2, 3], [2, 3], [2, 3]], dtype=np.int64)
        tile.run((item("a", 0), item("b", 1)), reads, writes, pools, None)
    # Third call reuses staging slot 0 with only 1 active request; lane 1 must have count 0.
    pools[2][1] = 2
    out = tile.run((item("a", 0),), np.zeros((3, 1), dtype=np.int64), np.full((3, 1), 4, dtype=np.int64), pools, None)
    assert len(out) == 1
    assert int(tile.table[-1, 1]) == 0
