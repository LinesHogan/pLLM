"""Independent products become a measured batch with replayed source evidence."""

from dataclasses import replace

import pytest
import torch

from pllm.config import ExecutionOptions
from pllm.contracts.region_program import Block, Call, Program, Region, Value
from pllm.contracts.region_selection import RegionPayloadLimit
from pllm.contracts.tensor_rewrite import MatmulBatching
from pllm.execution.exported_cell import ExportedCell, NumericalTransformWarning
from pllm.execution.region_optimization import optimize_region, select_region_candidate
from pllm.execution.tensor_batching import batching_steps
from pllm.execution.tensor_rewrites import replay_tensor_rewrites
from test.boundary.test_tensor_access import StorageObserver


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_outlined_independent_calls_run_one_bmm_and_charge_packing(device, monkeypatch):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    import pllm.execution.region_optimization as tuning

    tensors = tuple(
        torch.arange(1, 1 + rows * cols, dtype=torch.float64, device=device).reshape(rows, cols)
        for rows, cols in ((8, 4), (4, 6), (8, 4), (4, 6))
    )
    cell = ExportedCell.capture(lambda x, w: (x @ w,), ("x", "w"), ("out",), tensors[:2])
    left, right = cell.declaration.inputs
    (output,) = cell.declaration.outputs
    program = Program(
        Region(
            Block(
                tuple(
                    Value(name, type_)
                    for name, type_ in zip(("a", "b", "c", "d"), (left, right, left, right), strict=True)
                ),
                (
                    Call(("a", "b"), (Value("first", output),), cell.digest),
                    Call(("c", "d"), (Value("second", output),), cell.digest),
                ),
                ("first", "second"),
            )
        ),
        (cell.declaration,),
    )
    monkeypatch.setattr(
        tuning,
        "_measure",
        lambda execution, *args: (
            1
            if any(
                "independent matmul batching" in transformation
                for source in execution._cells.values()
                if isinstance(source, ExportedCell)
                for transformation in source.transformations
            )
            else 100
        ),
    )
    with pytest.warns(NumericalTransformWarning, match="batching"):
        selected = optimize_region(
            program,
            {cell.digest: cell},
            dict(zip(("a", "b", "c", "d"), tensors)),
            payload_limits=(RegionPayloadLimit(device, 10000),),
            options=replace(ExecutionOptions(schedule_policy="serial"), tuning="measured"),
            warmup_rounds=0,
            measurement_rounds=1,
        )
    (record,) = selected.rewrites
    assert isinstance(record, MatmulBatching) and record.dimensions == (2, 8, 4, 6)
    assert len(selected.request.candidates[selected.selection.selected].outlines) == 1
    assert select_region_candidate(program, replace(selected.request, allow_numerical_transforms=False)).selected == 0
    original_bytes = selected.selection.candidates[0].capacity.devices[0].peak_tensor_bytes_upper_bound
    tighter = select_region_candidate(
        program, replace(selected.request, payload_limits=(RegionPayloadLimit(device, original_bytes),))
    )
    assert not tighter.candidates[selected.selection.selected].feasible

    class Observe(StorageObserver):
        def __init__(self, inputs):
            super().__init__(inputs)
            self.products = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default):
                self.products.append(func)
            return super().__torch_dispatch__(func, types, args, kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("bound batching must not search or rewrite at runtime")

    monkeypatch.setattr(tuning, "explore_tensor_combinations", forbidden)
    with Observe(tensors) as observed:
        actual = selected.execution.run(dict(zip(("a", "b", "c", "d"), tensors))).values
    assert observed.products == [torch.ops.aten.bmm.default]
    for value, x, w in zip(actual, tensors[::2], tensors[1::2], strict=True):
        torch.testing.assert_close(value, x @ w, rtol=0, atol=0)
    for tensor in tensors:
        torch.testing.assert_close(
            tensor,
            torch.arange(1, 1 + tensor.numel(), dtype=tensor.dtype, device=device).reshape(tensor.shape),
            rtol=0,
            atol=0,
        )
    assert observed.peak - sum(t.untyped_storage().nbytes() for t in tensors) <= (
        selected.selection.candidates[selected.selection.selected].capacity.devices[0].peak_tensor_bytes_upper_bound
    )


def test_batching_rejects_dependencies_and_replays_shared_users():
    tensors = tuple(torch.arange(1, 17, dtype=torch.float64).reshape(4, 4) for _ in range(4))

    def chain(a, b):
        first = a @ b
        return first, first @ b

    dependent = ExportedCell.capture(chain, ("a", "b"), ("first", "second"), tensors[:2])
    # The final product reads a prior product, so it cannot join that batch.
    assert tuple(batching_steps(dependent)) == ()

    def shared(a, b, c, d):
        first, second = a @ b, c @ d
        return first, second, first + second

    source = ExportedCell.capture(shared, ("a", "b", "c", "d"), ("first", "second", "sum"), tensors)
    transformed, record = next(batching_steps(source))
    replayed = replay_tensor_rewrites(source, (record,))
    for actual, expected in zip(replayed(*tensors), source(*tensors), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert str(transformed.graph.graph) == str(replayed.graph.graph)
