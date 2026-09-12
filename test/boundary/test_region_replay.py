"""The lowered replay loop preserves first-step and floating convergence semantics."""

import math

import pytest
import torch

from pllm import _core
from pllm.contracts.region_program import TensorType
from pllm.contracts.semantic_graph import ConditionalLayerReplay
from pllm.execution.region_replay import ConditionalReplayRegion
from test.boundary.test_region_pages import owner_fixture


@pytest.mark.parametrize("cuda", (False, True))
@pytest.mark.parametrize("limit,mode", ((0, "same"), (1, "same"), (6, "same"),
    (6, "converge"), (4, "nan_distance"), (4, "nonfinite")))
def test_replay_control_matches_original_loop_without_hot_compilation(monkeypatch, cuda, limit, mode):
    if cuda and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    stream = torch.cuda.Stream() if cuda else None
    execution, _, rows, _ = owner_fixture(rounds=1, stream=stream)
    invocation = execution.start_step(rows, stream)
    device = execution.arena.device
    initial = torch.full((1, 2), 1e30 if mode == "nan_distance" else 1., device=device)
    effect = ConditionalLayerReplay((0,), max_iterations=limit, relative_l2_ppm=100_000)
    compiled = ConditionalReplayRegion(TensorType((1, 2), "torch.float32", str(device)), effect)
    def cold_only(*args, **kwargs):
        raise AssertionError("replay compiled inside a model invocation")
    for name in ("analyze_region_program", "lower_region_schedule", "lower_region_bindings",
                 "validate_region_bindings", "validate_lowered_region_schedule"):
        monkeypatch.setattr(_core, name, cold_only)
    calls, observed, recorded = [], [], []
    def step(carry):
        if stream is not None:
            assert torch.cuda.current_stream(device) == stream
        calls.append(carry.clone())
        if mode == "same":
            return carry
        if mode == "nan_distance":
            return -carry
        if mode == "nonfinite":
            return carry * float("inf")
        return carry * 0.5 + 0.25
    def observe(value, iteration):
        observed.append((iteration, value.clone()))
    def record(iteration, relative):
        recorded.append((iteration, relative))
    context = torch.cuda.stream(stream) if stream is not None else torch.no_grad()
    with context:
        if mode == "nonfinite":
            with pytest.raises(RuntimeError, match="nonfinite carry"):
                compiled.run(invocation.context, initial, step=step, observe=observe, record=record)
        else:
            compiled.run(invocation.context, initial, step=step, observe=observe, record=record)
    if stream is not None:
        torch.cuda.current_stream().wait_stream(stream)
    if mode == "nonfinite":
        assert len(calls) == len(observed) == 1 and not recorded
    else:
        previous, expected = None, []
        carry = initial
        for iteration in range(limit):
            result = carry if mode == "same" else -carry if mode == "nan_distance" else carry * 0.5 + 0.25
            relative = None if previous is None else float(((result.float() - previous.float()).square().sum().sqrt()
                / previous.float().square().sum().sqrt().clamp_min(1e-12)).item())
            expected.append((iteration, result.clone(), relative))
            carry = previous = result
            if relative is not None and relative <= 0.1:
                break
        assert len(calls) == len(observed) == len(recorded) == len(expected)
        for (i, actual), (j, relative), (k, wanted, distance) in zip(observed, recorded, expected, strict=True):
            assert i == j == k
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            assert relative == distance or math.isnan(relative) and math.isnan(distance)
        if mode == "nan_distance":
            assert len(calls) == limit
    execution.cancel_step(invocation)
    execution.retire(tuple(row.identity for row in rows))
    execution.arena.collect_completed(wait=True)
