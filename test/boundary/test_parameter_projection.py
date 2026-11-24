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


@pytest.mark.parametrize(
    "operator",
    (
        ParameterOverlayOperator.DENSE_ADD,
        ParameterOverlayOperator.LOW_RANK_ADD,
        ParameterOverlayOperator.HEAD_ADD,
    ),
)
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


def test_row_weight_operand_bypasses_overlay():
    """An identity dequantization borrows base weights and restores the installed projection hook."""
    from vllm.model_executor.layers.linear import LinearMethodBase, RowParallelLinear

    from pllm.backends.vllm.parameter_overlay import ProjectionHook, base_weight

    class IdentityMethod(LinearMethodBase):
        def create_weights(self, *args, **kwargs):
            pass

        def apply(self, layer, x, bias=None):
            return x @ layer.weight.T

    module = RowParallelLinear.__new__(RowParallelLinear)
    torch.nn.Module.__init__(module)
    module.weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    module.params_dtype = torch.float32
    module.input_size_per_partition = module.output_size_per_partition = module.tp_size = 2
    module.quant_method = IdentityMethod()
    hook = ProjectionHook(module)

    def overlay(hidden):
        return hidden @ module.weight.T + 1, None

    hook.install(overlay)
    try:
        torch.testing.assert_close(base_weight(module, True, torch.device("cpu")), module.weight)
        torch.testing.assert_close(module.quant_method.apply(module, torch.eye(2)), module.weight.T + 1)
    finally:
        hook.close()


def test_custom_projection_operator_cpu_and_static_budget():
    """Custom projection operators work unbatched, batched via vmap, and enforce static FakeTensorMode budgets."""
    from pllm.contracts.tensor_layout import TensorDType
    from pllm.execution.parameter_projection import (
        parameter_correction,
        prepare_static_projection_read,
        projection_read,
        validate_custom_projection_operator,
    )

    def scaled_low_rank(hidden: torch.Tensor, a: torch.Tensor, b: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        mid = torch.matmul(hidden, a.to(hidden.dtype).transpose(-1, -2))
        return torch.matmul(mid * torch.sigmoid(gate.to(hidden.dtype)), b.to(hidden.dtype).transpose(-1, -2))

    validate_custom_projection_operator(
        scaled_low_rank,
        inputs=8,
        outputs=6,
        hidden_dtype=TensorDType.FLOAT32,
        weight_specs=(((3, 8), TensorDType.FLOAT32), ((6, 3), TensorDType.FLOAT32), ((3,), TensorDType.FLOAT32)),
        projection="q_proj of layer 0",
    )

    with pytest.raises(ValueError, match="must return shape"):
        validate_custom_projection_operator(
            lambda h, a: h @ a.T,
            inputs=8,
            outputs=6,
            hidden_dtype=TensorDType.FLOAT32,
            weight_specs=(((4, 8), TensorDType.FLOAT32),),
            projection="q_proj of layer 0",
        )

    torch.manual_seed(17)
    hidden = torch.randn(6, 8)
    a = torch.randn(4, 3, 8)
    b = torch.randn(4, 6, 3)
    gate = torch.randn(4, 3)
    direct = parameter_correction(
        hidden, (a[1], b[1], gate[1]), ParameterOverlayOperator.CUSTOM, batched=False, custom_operator=scaled_low_rank
    )
    torch.testing.assert_close(direct, scaled_low_rank(hidden, a[1], b[1], gate[1]))

    slots = torch.tensor([[1, 1, 1], [2, 2, 2]])
    rows = torch.tensor([[0, 1, 2], [3, 4, 5]])
    owners = torch.tensor([0, 1, 2, 3, 4, -1])
    (batched_out,) = projection_read(
        hidden,
        slots,
        rows,
        owners,
        a,
        b,
        gate,
        operator=ParameterOverlayOperator.CUSTOM,
        custom_operator=scaled_low_rank,
    )
    expected = torch.cat(
        (
            scaled_low_rank(hidden[:3], a[1], b[1], gate[1]),
            scaled_low_rank(hidden[3:5], a[2], b[2], gate[2]),
            torch.zeros(1, 6),
        )
    )
    torch.testing.assert_close(batched_out, expected)

    usage = prepare_static_projection_read(
        hidden,
        (a[1], b[1], gate[1]),
        ParameterOverlayOperator.CUSTOM,
        budget=1 << 20,
        blas_workspace_bytes=0,
        custom_operator=scaled_low_rank,
    )
    assert usage > 0
    with pytest.raises(ValueError, match="budget"):
        prepare_static_projection_read(
            hidden,
            (a[1], b[1], gate[1]),
            ParameterOverlayOperator.CUSTOM,
            budget=usage - 1,
            blas_workspace_bytes=0,
            custom_operator=scaled_low_rank,
        )
