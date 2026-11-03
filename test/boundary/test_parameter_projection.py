"""Pinned projection math, field-pool borrowing and fixed-address graph replay."""

import pytest
import torch

from pllm.contracts.parameter_overlay import ParameterOverlayOperator
from pllm.execution.torch_private import blas_workspace_bytes


@pytest.mark.gpu
def test_pinned_projection_borrows():
    """Index tables vary under replay without materializing the field pool."""
    from pllm.contracts.parameter_overlay import ParameterOverlayOperator
    from pllm.execution.parameter_projection import prepare_projection_read, projection_read

    generator = torch.Generator(device="cuda").manual_seed(109)
    hidden = torch.randn(8, 8, device="cuda", generator=generator)
    pool = torch.randn(32, 2, 4, 4, device="cuda", generator=generator)
    slots = torch.tensor([[1], [3]], device="cuda")
    rows = torch.tensor([[0, 1, 2, 0], [3, 4, 5, 6]], device="cuda")
    owners = torch.tensor([0, 1, 2, 4, 5, 6, 7, -1], device="cuda")
    values = (hidden, slots, rows, owners, pool)
    execution = prepare_projection_read(
        values, ParameterOverlayOperator.HEAD_ADD, 128 << 20, blas_workspace_bytes=blas_workspace_bytes()
    )
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profile:
        result = execution(*values)[0]
    assert not any(
        event.name == "aten::clone" and event.input_shapes[0] == list(pool.shape) for event in profile.events()
    )
    torch.testing.assert_close(result, projection_read(*values, operator=ParameterOverlayOperator.HEAD_ADD)[0])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        execution(*values)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = execution(*values)[0]
    pointers = tuple(value.data_ptr() for value in values)
    for state_slots in ((3, 1), (7, 2), (1, 3)):
        slots.copy_(torch.tensor(state_slots, device="cuda")[:, None])
        graph.replay()
        torch.testing.assert_close(captured, projection_read(*values, operator=ParameterOverlayOperator.HEAD_ADD)[0])
        assert pointers == tuple(value.data_ptr() for value in values)
    assert not captured[-1].any()


@pytest.mark.parametrize("operator", tuple(ParameterOverlayOperator))
@pytest.mark.gpu
def test_projection_math_budget_and_allocator_peak(operator):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from pllm.execution.parameter_projection import prepare_projection_read, projection_read
    from pllm.execution.torch_private import measure

    torch.manual_seed(29)
    hidden = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16)
    shapes = ((32, 2, 4, 4),) if operator is ParameterOverlayOperator.HEAD_ADD else ((32, 6, 8),)
    if operator is ParameterOverlayOperator.LOW_RANK_ADD:
        shapes = ((32, 3, 8), (32, 6, 3))
    dtype = torch.float32 if operator is ParameterOverlayOperator.HEAD_ADD else hidden.dtype
    pools = tuple(torch.randn(shape, device="cuda", dtype=dtype) for shape in shapes)
    slots = torch.tensor([[1] * len(pools), [3] * len(pools)], device="cuda")
    rows = torch.arange(16, device="cuda").view(2, 8)
    owners = torch.arange(16, device="cuda")
    owners[-1] = -1
    values = (hidden, slots, rows, owners, *pools)
    with FakeTensorMode():
        samples = tuple(
            torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=value.device) for value in values
        )
        usage = measure(lambda: projection_read(*samples, operator=operator))
    scratch = blas_workspace_bytes()
    budget = usage.workspace + scratch
    with pytest.raises(ValueError, match="budget"):
        prepare_projection_read(values, operator, budget - 1, blas_workspace_bytes=scratch)
    execute = prepare_projection_read(values, operator, budget, blas_workspace_bytes=scratch)
    execute(*values)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    (actual,) = execute(*values)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - baseline <= budget
    expected = []
    for row in range(16):
        slot = 1 if row < 8 else 3
        if operator is ParameterOverlayOperator.HEAD_ADD:
            output = torch.stack(
                [pools[0][slot, head] @ hidden[row].float().view(2, 4)[head] for head in range(2)]
            ).flatten()
        else:
            output = pools[0][slot] @ hidden[row]
            if operator is ParameterOverlayOperator.LOW_RANK_ADD:
                output = pools[1][slot] @ output
        expected.append(output if row < 15 else torch.zeros_like(output))
    torch.testing.assert_close(actual, torch.stack(expected))
