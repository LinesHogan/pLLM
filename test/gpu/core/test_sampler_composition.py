"""Qwen composition oracle: independent additions, stock filtering, then one ESamp guide."""

from dataclasses import dataclass

import pytest
import torch
from vllm import SamplingParams

import pllm
from pllm.backends.vllm import compat, sampler
from pllm.backends.vllm.sampler import GuidedSampler, SignedSamplerStages, reweight
from pllm.backends.vllm.sampler_epilogue import copy_additions
from pllm.workloads import DoLaWorkload, ESampConfig, ESampWorkload

pytestmark = pytest.mark.gpu


class Steering(pllm.Workload):
    """An author's independent vocabulary bias beside the built-in effects."""

    @pllm.add_logits()
    def bias(self, logits=pllm.layer_logits((-1,))):
        return logits[:, -1].tanh() * 0.1


@dataclass(frozen=True)
class Stack(pllm.Workload):
    steer: Steering = Steering()
    dola: DoLaWorkload = DoLaWorkload(layers=(0, 4, 8))
    esamp: ESampWorkload = ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5))


@pytest.mark.parametrize("capacity,passthrough", ((None, False), (512, False), (None, True)))
def test_additions_then_filtered_guidance(monkeypatch, capacity, passthrough):
    workload = Stack()
    add, reweight = SignedSamplerStages.add_logits, GuidedSampler.reweight_finite_survivors
    checks = {"add": 0, "guide": 0, "trained": 0}
    fused = sampler.guided_sample
    raw_logprobs = []
    call = GuidedSampler.__call__

    def inspect_call(instance, logits, *args, **kwargs):
        output = call(instance, logits, *args, **kwargs)
        if raw_logprobs:
            expected = raw_logprobs.pop()
            scores = output.logprobs_tensors
            torch.testing.assert_close(
                scores.logprobs, expected.gather(-1, scores.logprob_token_ids.long()), rtol=1e-5, atol=1e-5
            )
        return output

    def inspect_add(stages, logits, *, dtype=torch.float32):
        assert [stage.guidance.member for stage in stages.stages[:2]] == ["steer.bias", "dola.contrast"]
        assert [stage.guidance.member for stage in stages.stages[2:]] in ([], ["esamp.predict"])
        expected = logits.clone()
        for stage in stages.stages[:2]:
            rows = torch.tensor(stage.guidance.affected_row_ids, device=logits.device)
            expected[rows] = expected[rows] + stage.pred_dense_logits.to(logits)
        if passthrough:
            assert dtype == logits.dtype
        actual = add(stages, logits, dtype=dtype)
        torch.testing.assert_close(actual, expected.to(dtype), rtol=0, atol=0)
        raw_logprobs.append(expected.float().log_softmax(-1))
        checks["add"] += 1
        return actual

    def inspect_guide(logits, stage):
        expected = logits.clone()
        weights = stage.guidance.row_weights
        assert weights is not None
        for index, row in enumerate(stage.guidance.affected_row_ids):
            support = logits[row].isfinite()
            beta = stage.guidance.beta * weights[index]
            expected[row, support] = (1 + beta) * logits[row, support] - beta * stage.pred_dense_logits[index, support]
        actual = reweight(logits, stage)
        torch.testing.assert_close(actual, expected)
        if not passthrough:
            assert (logits.isfinite().sum(-1) <= 7).all()
        checks["guide"] += 1
        checks["trained"] += int(weights.count_nonzero())
        return actual

    def inspect_fused(
        logits, hidden, weight, rows, weights, beta, limit, mapping, temperature, seeds, positions, p, *, fp64
    ):
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

        expected = logits.clone()
        predicted = torch.nn.functional.linear(hidden.to(weight.dtype), weight).float()
        active = rows >= 0
        selected = rows[active]
        strengths = beta * weights[selected, None]
        expected[active] = torch.where(
            expected[active].isfinite(),
            (1 + strengths) * expected[active] - strengths * predicted[selected],
            expected[active],
        )
        actual = fused(
            logits, hidden, weight, rows, weights, beta, limit, mapping, temperature, seeds, positions, p, fp64=fp64
        )
        torch.testing.assert_close(logits, expected, atol=0.02, rtol=0.004)
        draw_args = {} if compat.GUMBEL_MODE == 0 else {"is_drafting": False}
        token = gumbel_sample(
            expected, mapping, temperature, seeds, positions, apply_temperature=False, use_fp64=fp64, **draw_args
        )
        assert torch.equal(actual, token)
        if not passthrough:
            assert (logits.isfinite().sum(-1) <= 7).all()
        checks["guide"] += 1
        checks["trained"] += int(weights.count_nonzero())
        return actual

    config = pllm.ModelConfig(
        "Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=64,
        max_num_seqs=2,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.25,
        execution=pllm.ExecutionOptions(guidance_capacity=capacity),
    )
    with pllm.make_llm(config, workload=workload) as handle:
        monkeypatch.setattr(GuidedSampler, "__call__", inspect_call)
        monkeypatch.setattr(SignedSamplerStages, "add_logits", inspect_add)
        monkeypatch.setattr(GuidedSampler, "reweight_finite_survivors", staticmethod(inspect_guide))
        monkeypatch.setattr(sampler, "guided_sample", inspect_fused)
        outputs = handle.generate(
            ["The sky is", "Water freezes because"],
            SamplingParams(
                max_tokens=5,
                ignore_eos=True,
                temperature=0.0 if passthrough else 0.8,
                top_k=-1 if passthrough else 7,
                top_p=1.0 if passthrough else 0.9,
                seed=7,
                logprobs=3,
            ),
        )
        assert len(outputs) == 2 and all(len(output.outputs[0].token_ids) == 5 for output in outputs)
        assert all(checks.values())
        assert handle.diagnostics().update_count == 8


@pytest.mark.parametrize("prediction_dtype", (torch.float32, torch.bfloat16))
def test_reweight_matches_unfused_arithmetic(prediction_dtype):
    generator = torch.Generator(device="cuda").manual_seed(42)
    logits = torch.randn(3, 259, device="cuda", generator=generator)
    logits[:, ::7] = -torch.inf
    prediction = torch.randn(3, 259, device="cuda", dtype=prediction_dtype, generator=generator)
    prediction[:, ::7] = torch.nan
    beta = logits.new_tensor([0, 0.25, 1.5])[:, None]
    expected = torch.where(torch.isfinite(logits), (1 + beta) * logits - beta * prediction.float(), logits)
    actual = reweight(logits, prediction, beta)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
def test_ordered_add_copy_matches_materialized_stages(dtype):
    """Subsets, empty stages and strided predictions retain each declaration's rounding."""
    generator = torch.Generator(device="cuda").manual_seed(91)
    logits = torch.randn(4, 2062, device="cuda", dtype=dtype, generator=generator)[:, ::2]
    first = torch.randn(4, 1031, device="cuda", generator=generator) * 128
    subset = -first[[3, 1]]
    broadcast = torch.randn(1, 1031, device="cuda", dtype=torch.bfloat16, generator=generator).expand(4, -1)
    additions = (first, subset, broadcast, first[:0])
    rows = ((0, 1, 2, 3), (3, 1), (0, 1, 2, 3), ())
    expected = logits.clone()
    for prediction, selected in zip(additions, rows, strict=True):
        ids = torch.tensor(selected, device="cuda", dtype=torch.long)
        expected[ids] = expected[ids] + prediction.to(dtype)
    for output_dtype in dict.fromkeys((dtype, torch.float32)):
        torch.cuda.set_sync_debug_mode("error")
        try:
            actual = copy_additions(logits, additions, rows, dtype=output_dtype)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.testing.assert_close(actual, expected.to(output_dtype), rtol=0, atol=0)
