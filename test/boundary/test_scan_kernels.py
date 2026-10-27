"""Independent FP64 recurrence and fixed-address replay oracles for scan kernels."""

import pytest
import torch

from pllm.contracts.tensor_policy import ScanForm


def _right_oracle(initial, values, keys, alpha, beta):
    """Independent row-wise FP64 SGD, not a coefficient or tree implementation."""
    state, outputs = initial.double(), []
    for x, k in zip(values.double(), keys.double(), strict=True):
        outputs.append(torch.einsum("hij,thj->thi", state, x))
        decayed = state * alpha
        gradient = torch.zeros_like(state)
        for row, key in zip(x, k, strict=True):
            error = torch.einsum("hij,hj->hi", decayed, key) - row
            gradient += torch.einsum("hi,hj->hij", error, key)
        state = decayed - beta * gradient
    return state, tuple(outputs)


def _check_right_matrix_graph(initial, values, keys):
    """Changing C reuses one captured window; reversed right actions fail the oracle."""
    from pllm.kernels.right_matrix_scan import RightMatrixBuffers, launch_window as launch_right_window

    x = initial.new_zeros((8, 2, 4, 3))
    k = torch.zeros_like(x)
    active = torch.zeros(8, dtype=torch.bool, device=initial.device)
    buffers = RightMatrixBuffers(initial, 8, 4)
    launch_right_window(x, k, initial, active, buffers, 0.17, 0.83, ScanForm.HILLIS_STEELE)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_right_window(x, k, initial, active, buffers, 0.17, 0.83, ScanForm.HILLIS_STEELE)
    storage = (x, k, active, buffers.a, buffers.b, buffers.states, buffers.final, buffers.output)
    addresses = tuple(t.data_ptr() for t in storage)
    for count in (5, 0, 1, 3, 5):
        x.zero_()
        k.zero_()
        x[:count].copy_(values[:count].transpose(1, 2))
        k[:count].copy_(keys[:count].transpose(1, 2))
        active.copy_(torch.arange(8, device=active.device) < count)
        graph.replay()
        expected, outputs = _right_oracle(initial, values[:count], keys[:count], 0.83, 0.17)
        torch.testing.assert_close(buffers.final.double(), expected, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(
            tuple(v.transpose(0, 1).double() for v in buffers.output[:count]), outputs, rtol=2e-6, atol=2e-7
        )
        assert addresses == tuple(t.data_ptr() for t in storage)
    # The counted source's mask bank need not be a prefix: inactive actions are identities.
    active.copy_(torch.tensor([False, True, False, True, True, False, False, False], device=initial.device))
    graph.replay()
    expected = initial.double()
    for index in (1, 3, 4):
        expected, _ = _right_oracle(expected, values[index : index + 1], keys[index : index + 1], 0.83, 0.17)
    torch.testing.assert_close(buffers.final.double(), expected, rtol=2e-6, atol=2e-7)
    expected, _ = _right_oracle(initial, values[:5], keys[:5], 0.83, 0.17)
    # Noncommuting symmetric per-chunk A matrices expose reversed composition;
    # independent B and a nonsymmetric initial state also expose left action.
    vv, kk = values.double().transpose(1, 2), keys.double().transpose(1, 2)
    a = 0.83 * (torch.eye(3, device=initial.device) - 0.17 * (kk.transpose(-2, -1) @ kk))
    b = 0.17 * (vv.transpose(-2, -1) @ kk)
    wrong = initial.double()
    for aa, bb in zip(a, b, strict=True):
        wrong = aa @ wrong + bb
    assert not torch.allclose(wrong, expected, rtol=1e-3, atol=1e-3)
    prefix_a, prefix_b = a[0], b[0]
    for aa, bb in zip(a[1:], b[1:], strict=True):
        prefix_a, prefix_b = aa @ prefix_a, bb @ prefix_a + prefix_b
    assert not torch.allclose(initial.double() @ prefix_a + prefix_b, expected, rtol=1e-3, atol=1e-3)


def test_right_matrix_graph_matches_rowwise_oracle():
    """The kernel oracle remains runnable without any Region execution fixtures."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    generator = torch.Generator(device="cuda:0").manual_seed(92)
    initial = torch.randn(2, 3, 3, device="cuda:0", generator=generator) * 0.2
    values = torch.randn(5, 4, 2, 3, device="cuda:0", generator=generator)
    keys = torch.nn.functional.normalize(values, dim=-1)
    _check_right_matrix_graph(initial, values, keys)
