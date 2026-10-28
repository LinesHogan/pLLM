"""ESamp on vLLM's MRv1 runner: packed rows match the scheduler and guidance keeps finite support.

Rows encoded from MRv1's persistent batch must list every scheduled token with its
phase, and the sampler reweighting must change only finite logits by the beta formula.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from vllm import SamplingParams

from pllm import make_llm
from pllm.backends.vllm import mrv1
from pllm.backends.vllm.sampler import SamplerGuidance
from pllm.config import ModelConfig, RunnerKind, VllmDType
from pllm.workloads.esamp_request import ESampRequestConfig, ESampRequestWorkload


def main() -> None:
    handle = make_llm(
        workload=ESampRequestWorkload(ESampRequestConfig(rank=2, intervention_beta=0.2)),
        config=ModelConfig(
            "Qwen/Qwen2.5-0.5B-Instruct", VllmDType.BFLOAT16, 64, 2, 128, 0.25, seed=7, runner=RunnerKind.MRV1
        ),
    )
    native_rows, native_reweight = mrv1.encode_mrv1_rows, SamplerGuidance._reweight_finite_survivors
    widths, phases, identities_seen = set(), set(), set()
    row_checks = sampler_checks = affected_checks = 0

    def inspect_rows(batch, scheduler, logits_indices, identities, **kwargs):
        nonlocal row_checks
        result = native_rows(batch, scheduler, logits_indices, identities, **kwargs)
        expected, links = [], []
        for request, key in enumerate(batch.req_ids):
            start, prompt = int(batch.num_computed_tokens_cpu[request]), int(batch.num_prompt_tokens[request])
            for offset in range(start, start + scheduler.num_scheduled_tokens[key]):
                expected.append((key, offset, "prefill" if offset < prompt else "decode"))
            links.append(len(expected) - 1)
        assert [
            (row.identity.request_id, row.token_offset, row.phase.value) for row in result.row_batch.rows
        ] == expected
        assert [link.packed_row_index for link in result.links] == links == logits_indices.tolist()
        widths.add(batch.num_reqs)
        phases.update(row.phase.value for row in result.row_batch.rows)
        identities_seen.update(identity.prompt_idx for identity in identities)
        row_checks += len(expected)
        return result

    def inspect_reweight(logits, guidance):
        nonlocal sampler_checks, affected_checks
        expected = logits.clone()
        weights = guidance.guidance.row_weights
        weights = [1.0] * len(guidance.guidance.affected_row_ids) if weights is None else weights.tolist()
        for index, row in enumerate(guidance.guidance.affected_row_ids):
            finite = torch.isfinite(logits[row])
            beta = guidance.guidance.beta * weights[index]
            expected[row, finite] = (1 + beta) * logits[row, finite] - (
                beta * guidance.pred_dense_logits[index, finite].to(logits)
            )
        actual = native_reweight(logits, guidance)
        assert torch.equal(torch.isfinite(actual), torch.isfinite(logits))
        torch.testing.assert_close(actual, expected)
        sampler_checks += 1
        affected_checks += sum(1 for weight in weights if weight)
        return actual

    try:
        assert isinstance(handle.runner, mrv1.Mrv1Runner)
        with (
            patch.object(mrv1, "encode_mrv1_rows", inspect_rows),
            patch.object(SamplerGuidance, "_reweight_finite_survivors", staticmethod(inspect_reweight)),
        ):
            batch = handle.generate(
                ["The sky is", "Explain why water freezes at", "The sea is"],
                SamplingParams(max_tokens=4, temperature=0.8, top_k=7, top_p=0.9, ignore_eos=True, logprobs=2),
                prompt_indices=(7, 9, 11),
                feedback_group="mrv1",
            )
        assert widths == {1, 2} and phases == {"prefill", "decode"} and identities_seen == {7, 9, 11}
        assert row_checks > 9 and sampler_checks > 0 and affected_checks > 0
        assert batch.hidden.shape == (3, 896) and torch.isfinite(batch.hidden).all()
        assert tuple(entry.identity.prompt_idx for entry in batch.completions.entries) == (7, 9, 11)
        diagnostics = handle.worker_state().diagnostics
        assert diagnostics.loss_count == diagnostics.update_count == 9
        assert diagnostics.active_lineages == diagnostics.pending_state_releases == 0
        print(f"MRv1: rows={row_checks} sampler_checks={sampler_checks} updates={diagnostics.update_count}")
    finally:
        handle.close()


if __name__ == "__main__":
    main()
