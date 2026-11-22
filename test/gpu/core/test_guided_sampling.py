"""Bounded guidance against FP64 arithmetic and vLLM's draw, including capture and overflow."""

import pytest
import torch
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

from pllm.backends.vllm import compat
from pllm.backends.vllm.guided_sampling import guided_sample
from pllm.execution.step_capture import capture_graph

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("fp64,capacity,dtype", ((False, 32, torch.float32), (True, 4096, torch.bfloat16)))
def test_bounded_guidance_draw_and_capture(fp64, capacity, dtype):
    torch.manual_seed(41)
    rows, vocab, width = 10, 10003, 37
    weight = torch.randn(vocab, width, device="cuda") * 0.1
    hidden = torch.randn(width, 8, device="cuda").T
    logits = torch.randn(rows, vocab, device="cuda").to(dtype)
    counts = (1, 3, capacity, capacity + 1, vocab, vocab, 7, 9, 0, 5)
    for row, count in enumerate(counts):
        logits[row, count:] = -torch.inf
    logits[3, : capacity + 1] = 0.5  # Capacity ties select the lower token IDs.
    logits[4, 9001] = 100
    row_map = torch.tensor((0, -1, 1, 2, 3, 4, 5, 6, 7, -1), device="cuda")
    weights = torch.tensor((1, 0.25, 1, 1, 1, 0, 1, 1), device="cuda")
    mapping = torch.randperm(rows, device="cuda")
    positions = (torch.arange(2 * rows, device="cuda", dtype=torch.int64) + 19)[::2]
    seeds = torch.arange(rows, device="cuda", dtype=torch.int64) + 2147483648
    temperature = torch.ones(rows, device="cuda")
    temperature[mapping[2]] = 0
    top_p = torch.full((rows,), 0.95, device="cuda")
    expected = logits.float().clone()
    # The reference rounds the FP64 head to its native output dtype, as the upstream head does.
    prediction = (hidden.double() @ weight.double().T).float()
    for row, count in enumerate(counts):
        index = int(row_map[row])
        if index < 0 or not weights[index]:
            continue
        if row == 4:
            support = torch.zeros(vocab, dtype=torch.bool, device="cuda")
            support[9001] = True
            expected[row, ~support] = -torch.inf
        else:
            ids = torch.argsort(logits[row], descending=True, stable=True)[:capacity]
            support = torch.zeros(vocab, dtype=torch.bool, device="cuda")
            support[ids] = logits[row, ids].isfinite()
        beta = 0.5 * weights[index]
        expected[row, support] = (1 + beta) * logits[row, support].float() - beta * prediction[index, support]
    expected = expected.to(dtype)
    draw_args = {} if compat.GUMBEL_MODE == 0 else {"is_drafting": False}
    reference = gumbel_sample(
        expected, mapping, temperature, seeds, positions, apply_temperature=False, use_fp64=fp64, **draw_args
    )
    workspace = logits.clone()

    def body():
        workspace.copy_(logits)
        return guided_sample(
            workspace,
            hidden,
            weight,
            row_map,
            weights,
            0.5,
            capacity,
            mapping,
            temperature,
            seeds,
            positions,
            top_p,
            fp64=fp64,
        )

    graph, output, _ = capture_graph(logits.device, body)
    for _ in range(2):
        graph.replay()
        torch.testing.assert_close(workspace, expected, atol=2e-6, rtol=2e-6)
        assert torch.equal(output, reference)
    for row in (1, 6, 9):
        assert torch.equal(workspace[row], logits[row])
