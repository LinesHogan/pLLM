"""An original model graph reads the owner's parameter and buffer versions."""

from pllm.contracts.region_selection import RegionSelectionPolicy

import pytest
import torch

from pllm.contracts.region_composition import inline_function_bodies
from pllm.contracts.region_imports import BorrowedTensorImport, CompiledTensorImport
from pllm.contracts.region_selection import RegionPayloadLimit
from pllm.contracts.region_program import AffineIndex, Block, BorrowedResource, Call, Coordinate, Program, Region, ScalarType, Value
from pllm.execution.exported_region import import_exported_region
from pllm.execution.exported_resources import borrow_exported_inputs
from pllm.execution.module_graph import exported_parameter_fields
from pllm.execution.region_compilation import analyze_region_program
from pllm.execution.region_optimization import optimize_region
from pllm.execution.region_native import BorrowedTensorPrimitive
from pllm.execution.region_resources import ReadonlyResourceBinding, RegionResources
from pllm.execution.region_storage import analyze_region_capacity
from pllm.execution.region_streams import RegionStreamBindings
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView
from test.boundary.test_region_pages import owner_fixture
from test.boundary.test_region_tensor_borrow import Copies


@pytest.mark.parametrize("device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"))))
def test_export_parameters_and_buffers_borrow_original_storage_through_the_region(device):
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((64, 64), 1 / 64, device=device))
            self.register_buffer("offset", torch.full((64,), 0.25, device=device))

        def forward(self, x):
            return torch.nn.functional.silu(x @ self.weight + self.offset), self.weight[:1]

    model, x = Projection(), torch.full((2, 64), 2.0, device=device)
    resource = BorrowedResource("parameters", (1,))
    def unexpected_kernel(partition):
        pytest.fail(f"ordinary Torch graph unexpectedly requires a kernel adapter: {partition}")
    exported = torch.export.export(model, (x,))
    source = import_exported_region(exported, bind_kernel=unexpected_kernel)
    body, functions = borrow_exported_inputs(source, resource,
        exported_parameter_fields(exported, (model.weight, model.offset)))
    declaration = body.declaration
    outputs = tuple(Value(f"out{i}", type_) for i, type_ in enumerate(declaration.outputs))
    wrapper = Program(Region(Block((Value("x", declaration.inputs[0]),), (
        Coordinate((), (Value("layer", ScalarType.INDEX),), AffineIndex(0)),
        Call(("x", "layer"), outputs, declaration.name),
    ), tuple(value.name for value in outputs))), (declaration,), (resource,))
    program = inline_function_bodies(wrapper, (body,))
    assert declaration not in program.functions  # There is no wrapper executor.
    schedule = analyze_region_program(program, optimize=False).reference_schedule
    loans: tuple[BorrowedTensorImport, ...] = tuple(function.implementation for function in functions.values()
        if isinstance(function, BorrowedTensorPrimitive))
    streams = None
    if device.startswith("cuda"):
        streams = RegionStreamBindings(torch.cuda.Stream(device=device), (torch.cuda.Stream(device=device),))
        streams.owner.wait_stream(torch.cuda.current_stream(device))
    capacity = analyze_region_capacity(program, schedule, functions)
    assert capacity.devices[0].peak_tensor_bytes_upper_bound < model.weight.numel() * model.weight.element_size()
    execution, _, rows, _ = owner_fixture(rounds=1, stream=None if streams is None else streams.owner)
    invocation = execution.start_step(rows, None if streams is None else streams.owner)
    original = model.weight.clone(), model.offset.clone()
    proof = ReadonlyResourceBinding(ReadonlyTensorView(resource,
        (ReadonlyTensorCell.bind((0,), (model.weight, model.offset)),)), invocation.context)
    pages = RegionResources(program, invocation.context, (), borrows=(proof,), tensor_borrows=loans)
    try:
        with pytest.raises(ValueError, match="original readonly resource bindings"):
            optimize_region(program, functions, {"x": x}, payload_limits=(RegionPayloadLimit(device, 1 << 20),), max_candidates=1, options=RegionSelectionPolicy(tuning="measured"))
        if streams is not None:
            streams.owner.wait_stream(torch.cuda.current_stream(device))
        with pages.scope():
            runner = optimize_region(program, functions, {"x": x}, pages=pages, streams=streams,
                payload_limits=(RegionPayloadLimit(device, 1 << 20),), warmup_rounds=0, measurement_rounds=1, max_candidates=8, options=RegionSelectionPolicy(tuning="measured"))
            if streams is not None:
                assert any(candidate.elapsed_ns and any(isinstance(item, CompiledTensorImport) for item in candidate.imports)
                    for candidate in runner.request.candidates)
            with Copies() as copies:
                actual = runner.execution.run({"x": x}, pages=pages, streams=streams).values
            assert tuple(model.weight.shape) not in copies.shapes
            for value, reference in zip(actual, model(x), strict=True):
                torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-6)
            actual[1].zero_()
            torch.testing.assert_close(model.weight, original[0], rtol=0, atol=0)
            torch.testing.assert_close(model.offset, original[1], rtol=0, atol=0)
    finally:
        execution.cancel_step(invocation)
        execution.retire(tuple(row.identity for row in rows))
