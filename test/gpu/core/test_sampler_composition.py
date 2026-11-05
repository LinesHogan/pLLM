"""Qwen composition oracle: independent additions, stock filtering, then one ESamp guide on both runners."""

from dataclasses import dataclass

import pytest
import torch
from vllm import SamplingParams

import pllm
from pllm.backends.vllm import compat, sampler
from pllm.backends.vllm.sampler import SamplerGuidance, SignedSamplerStages
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


@pytest.mark.parametrize("runner,capacity", (("mrv1", None), ("mrv2", None), ("mrv2", 4096)))
def test_additions_then_filtered_guidance(monkeypatch, runner, capacity):
    workload = Stack()
    add, reweight = SignedSamplerStages.add_logits, SamplerGuidance._reweight_finite_survivors
    checks = {"add": 0, "guide": 0, "trained": 0}
    fused = sampler.guided_sample

    def inspect_add(stages, logits):
        assert [stage.guidance.member for stage in stages.stages[:2]] == ["steer.bias", "dola.contrast"]
        assert [stage.guidance.member for stage in stages.stages[2:]] in ([], ["esamp.predict"])
        expected = logits.clone()
        for stage in stages.stages[:2]:
            rows = torch.tensor(stage.guidance.affected_row_ids, device=logits.device)
            expected[rows] = expected[rows] + stage.pred_dense_logits.to(logits)
        actual = add(stages, logits)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
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
        assert torch.equal(actual.isfinite(), logits.isfinite())
        assert (logits.isfinite().sum(-1) <= 7).all()
        checks["guide"] += 1
        checks["trained"] += int(weights.count_nonzero())
        return actual

    def inspect_fused(
        logits, hidden, weight, rows, weights, beta, limit, mapping, temperature, seeds, positions, p, *, fp64
    ):
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

        expected = logits.clone()
        predicted = torch.nn.functional.linear(hidden, weight).float()
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
        draw_args = {} if compat.VLLM_027 else {"is_drafting": False}
        token = gumbel_sample(
            expected, mapping, temperature, seeds, positions, apply_temperature=False, use_fp64=fp64, **draw_args
        )
        assert torch.equal(actual, token)
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
        runner=runner,
        execution=pllm.ExecutionOptions(guidance_capacity=capacity),
    )
    with pllm.make_llm(config, workload=workload) as handle:
        monkeypatch.setattr(SignedSamplerStages, "add_logits", inspect_add)
        monkeypatch.setattr(SamplerGuidance, "_reweight_finite_survivors", staticmethod(inspect_guide))
        monkeypatch.setattr(sampler, "guided_sample", inspect_fused)
        outputs = handle.generate(
            ["The sky is", "Water freezes because"],
            SamplingParams(max_tokens=5, ignore_eos=True, temperature=0.8, top_k=7, top_p=0.9, seed=7),
        )
        assert len(outputs) == 2 and all(len(output.outputs[0].token_ids) == 5 for output in outputs)
        assert all(checks.values())
        assert handle.diagnostics().update_count == 8
