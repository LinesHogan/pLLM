"""Model mathematics reads original weights and returns independently owned values."""

from dataclasses import replace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from pllm.contracts.region_imports import BorrowedTensorImport
from pllm.contracts.region_program import (
    AccessMode,
    AffineIndex,
    Block,
    BorrowedResource,
    Call,
    Coordinate,
    Function,
    If,
    Program,
    Region,
    ResourceAccess,
    ScalarType,
    TensorType,
    Value,
)
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.decoder_region import index_cell
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_native import BorrowedTensorPrimitive
from pllm.execution.region_resources import ReadonlyResourceBinding, RegionResources
from pllm.execution.region_storage import analyze_region_capacity, analyze_region_storage, validate_region_capacity
from pllm.execution.region_streams import RegionStreamBindings, lower_region_streams
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView
from test.boundary.test_region_pages import owner_fixture
from test.boundary.test_region_streams import limits, stream_execution


def reader(name, tensor, field):
    declaration = Function(
        name,
        (ScalarType.INDEX,),
        (tensor,),
        ("weights",),
        (ResourceAccess("weights", AccessMode.READ, (index_cell(0),)),),
    )
    return BorrowedTensorPrimitive(BorrowedTensorImport(declaration, field))


class Copies(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.clone.default:
            self.shapes.append(tuple(args[0].shape))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))],
)
def test_swiglu_uses_original_weights_without_matrix_copies_and_owns_returned_views(device):
    dtype = torch.float64 if device == "cpu" else torch.bfloat16
    hidden, gate_up, down = (torch.randn(shape, device=device, dtype=dtype) for shape in ((2, 64), (128, 64), (64, 64)))
    originals = gate_up.clone(), down.clone()

    def mlp(h, gw, dw):
        gate, up = torch.nn.functional.linear(h, gw).chunk(2, -1)
        return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, dw), gw[:1]

    cell = ExportedCell.capture(mlp, ("hidden", "gate_up", "down"), ("output", "weight_view"), (hidden, gate_up, down))
    h, g, d = cell.declaration.inputs
    out, view_type = cell.declaration.outputs
    load_gate, load_down = reader("model.gate_up", g, 0), reader("model.down", d, 1)
    index = ScalarType.INDEX
    resource = BorrowedResource("weights", (1,))
    program = Program(
        Region(
            Block(
                (Value("hidden", h),),
                (
                    Coordinate((), (Value("zero", index),), AffineIndex(0)),
                    Call(("zero",), (Value("gw", g),), load_gate.declaration.name),
                    Call(("zero",), (Value("dw", d),), load_down.declaration.name),
                    Call(("hidden", "gw", "dw"), (Value("result", out), Value("slice", view_type)), cell.digest),
                ),
                ("result", "slice"),
            )
        ),
        (load_gate.declaration, load_down.declaration, cell.declaration),
        (resource,),
    )
    functions = {value.declaration.name: value for value in (load_gate, load_down, cell)}
    loans = (load_gate.implementation, load_down.implementation)
    assert decode_boundary(encode_boundary(loans[0]), BorrowedTensorImport) == loans[0]
    schedule = "{ S0[] -> [0]; S1[] -> [1]; S2[] -> [3] }"
    storage = analyze_region_storage(program, schedule, functions)
    assert "V" in storage.borrowed_storages
    capacity = analyze_region_capacity(program, schedule, functions)
    (facts,) = capacity.devices
    assert not capacity.unknown_functions
    assert facts.result_copy_bytes_upper_bound == (2 * 64 + 64) * hidden.element_size()
    assert facts.peak_tensor_bytes_upper_bound < gate_up.numel() * gate_up.element_size()
    validate_region_capacity(program, functions, capacity)
    streams = None
    if device == "cpu":
        runner = RegionExecution(
            program, lower_region_schedule(program, schedule), lower_region_bindings(program, schedule), functions
        )
    else:
        streams = RegionStreamBindings(torch.cuda.Stream(device=device), (torch.cuda.Stream(device=device),))
        streams.owner.wait_stream(torch.cuda.current_stream(device))
        plan = lower_region_streams(program, schedule, functions, axis=0, lanes=2, pipeline_depth=2)
        assert not plan.host_fences
        runner = stream_execution(program, functions, plan, payload_limits=limits(plan))
    owner_stream = None if streams is None else streams.owner
    execution, _, rows, _ = owner_fixture(rounds=1, stream=owner_stream)
    invocation = execution.start_step(rows, owner_stream)
    proof = ReadonlyResourceBinding(
        ReadonlyTensorView(resource, (ReadonlyTensorCell.bind((0,), (gate_up, down)),)), invocation.context
    )
    pages = RegionResources(program, invocation.context, (), borrows=(proof,), tensor_borrows=loans)
    try:
        assert load_gate.invoke((0,), pages)[0].data_ptr() == gate_up.data_ptr()
        with Copies() as copies:
            result = runner.run({"hidden": hidden}, pages=pages, streams=streams).values
        assert tuple(gate_up.shape) not in copies.shapes and tuple(down.shape) not in copies.shapes
        for actual, expected in zip(result, mlp(hidden, gate_up, down), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        result[1].zero_()
        torch.testing.assert_close(gate_up, originals[0], rtol=0, atol=0)
        torch.testing.assert_close(down, originals[1], rtol=0, atol=0)
        wrong = RegionResources(
            program, invocation.context, (), borrows=(proof,), tensor_borrows=(replace(loans[0], field=1), loans[1])
        )
        with pytest.raises(ValueError, match="original source bindings"):
            runner.run({"hidden": hidden}, pages=wrong, streams=streams)
    finally:
        execution.cancel_step(invocation)
        execution.retire(tuple(row.identity for row in rows))


def test_borrow_through_atomic_control_cannot_escape_as_a_mutable_weight_alias():
    weight = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    before = weight.clone()
    tensor, index = TensorType((8, 8), "torch.float32"), ScalarType.INDEX
    load = reader("model.weight", tensor, 0)
    branch = Region(
        Block(
            (),
            (
                Coordinate((), (Value("zero", index),), AffineIndex(0)),
                Call(("zero",), (Value("w", tensor),), load.declaration.name),
            ),
            ("w",),
        )
    )
    resource = BorrowedResource("weights", (1,))
    program = Program(
        Region(
            Block(
                (Value("choose", ScalarType.BOOL),),
                (If(("choose",), (Value("out", tensor),), branch, branch),),
                ("out",),
            )
        ),
        (load.declaration,),
        (resource,),
    )
    functions = {load.declaration.name: load}
    schedule = analyze_region_program(program, optimize=False).reference_schedule
    capacity = analyze_region_capacity(program, schedule, functions)
    assert capacity.devices[0].result_copy_bytes_upper_bound == weight.numel() * weight.element_size()
    runner = RegionExecution(
        program, lower_region_schedule(program, schedule), lower_region_bindings(program, schedule), functions
    )
    execution, _, rows, _ = owner_fixture(rounds=1)
    invocation = execution.start_step(rows)
    proof = ReadonlyResourceBinding(
        ReadonlyTensorView(resource, (ReadonlyTensorCell.bind((0,), (weight,)),)), invocation.context
    )
    pages = RegionResources(program, invocation.context, (), borrows=(proof,), tensor_borrows=(load.implementation,))
    try:
        for predicate in (False, True):
            (output,) = runner.run({"choose": predicate}, pages=pages).values
            torch.testing.assert_close(output, before, rtol=0, atol=0)
            output.fill_(-1)
            torch.testing.assert_close(weight, before, rtol=0, atol=0)
    finally:
        execution.cancel_step(invocation)
        execution.retire(tuple(row.identity for row in rows))
