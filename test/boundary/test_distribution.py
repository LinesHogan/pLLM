"""FP32 distribution diagnostics compared with independent FP64 probability arithmetic."""

import math

import torch

from pllm.testing_distribution import _distances, _failures, _summaries


def test_distribution_arithmetic():
    # A rare large change must remain visible in P99 even when most positions are neutral.
    p = torch.tensor([[0.7, 0.2, 0.1]] * 98 + [[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]], dtype=torch.float64)
    q = p.clone()
    q[-2:] = torch.tensor([[0.01, 0.01, 0.98], [0.98, 0.01, 0.01]], dtype=torch.float64)
    expected = torch.stack(
        ((p * (p / q).log()).sum(-1), (q * (q / p).log()).sum(-1), (p - q).abs().sum(-1) / 2), dim=-1
    )
    # Exercise the same FP32 normalization from logits as the real sampler observer.
    actual = _distances(p.log().float().log_softmax(-1), q.log().float().log_softmax(-1))
    torch.testing.assert_close(actual.double(), expected, atol=2e-7, rtol=2e-6)
    summaries = _summaries(actual, [(i // 10, i % 10) for i in range(100)], 2)
    for index, summary in enumerate(summaries):
        assert math.isclose(summary.mean, float(expected[:, index].mean()), abs_tol=2e-7)
        assert math.isclose(summary.p99, float(expected[:, index].sort().values[98]), abs_tol=2e-6)
        assert math.isclose(summary.max, float(expected[:, index].max()), abs_tol=2e-6)
        assert (summary.worst[0].prompt_index, summary.worst[0].token_offset) == (9, 9)
    assert not _failures(summaries, 0.1, None)
    assert len(_failures(summaries, 0.1, 0.1)) == 2
    support = _distances(torch.tensor([[0.0, -torch.inf]]), torch.tensor([[-torch.inf, 0.0]]))
    assert torch.equal(support, torch.tensor([[torch.inf, torch.inf, 1.0]]))
    assert torch.equal(
        _distances(torch.tensor([[0.0, -torch.inf]]), torch.tensor([[0.0, -torch.inf]])), torch.zeros(1, 3)
    )
    tail = _summaries(torch.cat((actual, support)), [(0, i) for i in range(101)], 1)
    assert math.isinf(tail[0].max) and math.isfinite(tail[0].p99)
    assert _failures(tail, 0.1, None)
    assert not _failures(tail, None, None)
    invalid = _summaries(torch.full((1, 3), torch.nan), [(0, 0)], 1)
    assert len(_failures(invalid, None, None)) == 3
