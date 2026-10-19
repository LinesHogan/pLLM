"""Native stream placement needs locally bound completion and workspace facts."""

from dataclasses import replace

import pytest
import torch

from pllm.contracts.region_imports import KernelAllocation, NativeKernelImport, RegionTensorImport
from pllm.contracts.region_program import (
    AccessMode,
    AffineIndex,
    BorrowedResource,
    Call,
    Coordinate,
    Function,
    Program,
    ResourceAccess,
    ScalarType,
    TensorType,
    Value,
)
from pllm.contracts.region_selection import RegionPayloadLimit
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.decoder_region import index_cell
from pllm.execution.region_compilation import analyze_region_program
from pllm.execution.region_native import NativePrimitive, external_primitive
from pllm.execution.region_resources import ExternalFunction, ReadonlyResourceBinding, RegionResources
from pllm.execution.region_storage import analyze_region_capacity
from pllm.execution.region_streams import RegionStreamBindings, explore_region_streams, lower_region_streams
from pllm.execution.region_values import Bank
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView
from pllm.execution.tensor_scan import TensorScan
from test.boundary.test_region_pages import owner_fixture
from test.boundary.test_region_program import region
from test.boundary.test_region_streams import stream_execution


def native_scan(device):
    tensor = TensorType((4,), "torch.float32", device)
    declaration = Function("test.native.sliced_buffer", (tensor,), (tensor,))
    implementation = NativeKernelImport(
        declaration, "test.sliced_buffer.current_stream@1", device, (KernelAllocation("backing", device, 64),)
    )
    streams = []

    def kernel(inputs, resources):
        (value,) = inputs
        if device.startswith("cuda"):
            streams.append(torch.cuda.current_stream().cuda_stream)
        backing = torch.empty(16, device=device)
        backing[:4].copy_(value).add_(2)
        return (backing[:4],)

    cell = NativePrimitive(declaration, kernel, implementation)
    return TensorScan(cell, 3, carry=0, invariants=0), cell, streams


def test_native_kernel_import_includes_backing_storage_and_needs_an_explicit_abi():
    scan, cell, _ = native_scan("cpu")
    program, source = scan.program, analyze_region_program(scan.program)
    assert decode_boundary(encode_boundary(cell.implementation), RegionTensorImport) == cell.implementation
    cells = {cell.declaration.name: cell}
    capacity = analyze_region_capacity(program, source.reference_schedule, cells)
    assert capacity.devices[0].call_scratch_bytes_upper_bound >= 64  # The declared kernel backing.
    assert explore_region_streams(
        program, cells, schedule=analyze_region_program(program, parallel_only=True).candidates[-1].schedule_map
    ).candidates
    # A generic callback is still opaque, even when its Python code happens
    # to be the same as a kernel carrying an explicit backend ABI.
    assert not explore_region_streams(
        program,
        {cell.declaration.name: replace(cell, implementation=None)},
        schedule=analyze_region_program(program, parallel_only=True).candidates[-1].schedule_map,
    ).candidates


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_kernels_execute_on_auxiliary_stream_with_independent_output_copies():
    scan, cell, launches = native_scan("cuda:0")
    cells = {cell.declaration.name: cell}
    (plan,) = explore_region_streams(
        scan.program,
        cells,
        schedule=analyze_region_program(scan.program, parallel_only=True).candidates[-1].schedule_map,
    ).candidates
    runner = stream_execution(
        scan.program,
        cells,
        plan,
        payload_limits=(RegionPayloadLimit("cuda:0", plan.devices[0].peak_bytes_upper_bound),),
    )
    bindings = RegionStreamBindings(torch.cuda.Stream(), (torch.cuda.Stream(),))
    values = tuple(torch.full((4,), float(i), device="cuda:0") for i in range(3))
    bindings.owner.wait_stream(torch.cuda.current_stream())
    result = runner.run({"arg0": Bank(values)}, streams=bindings)
    assert set(launches) == {bindings.owner.cuda_stream, bindings.auxiliary[0].cuda_stream}
    (outputs,) = result.values
    with torch.cuda.stream(bindings.owner):
        for i, (value, original) in enumerate(zip(outputs.elements, values, strict=True)):
            assert value.untyped_storage().nbytes() == 16  # The 64-byte kernel backing is private scratch.
            torch.testing.assert_close(value, torch.full_like(value, i + 2), rtol=0, atol=0)
            torch.testing.assert_close(original, torch.full_like(original, i), rtol=0, atol=0)
    bindings.owner.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_external_kernel_borrows_original_weights_and_requires_matching_local_abi():
    from pllm.backends.vllm.replay import VllmTokenwiseReplay

    owner = torch.cuda.Stream()
    execution, _, rows, _ = owner_fixture(rounds=1, stream=owner)
    invocation = execution.start_step(rows, owner)
    # Exercise the actual adapter's loan method without loading model weights.
    backend = VllmTokenwiseReplay.__new__(VllmTokenwiseReplay)
    backend._invocation = invocation
    backend._auxiliary_streams = ()
    streams = backend.borrow_streams(invocation.context, 2)
    assert backend.borrow_streams(invocation.context, 2) == streams
    assert streams.owner is owner
    (auxiliary,) = streams.auxiliary
    weights = BorrowedResource("weights", (2,))
    index = ScalarType.INDEX
    tensor = TensorType((2,), "torch.float32", "cuda:0")
    function = Function(
        "test.readonly.kernel",
        (index,),
        (tensor,),
        (weights.name,),
        (ResourceAccess(weights.name, AccessMode.READ, (index_cell(0),)),),
    )
    program = Program(
        region(
            (),
            (
                Coordinate((), (Value("zero", index),), AffineIndex(0)),
                Coordinate((), (Value("one", index),), AffineIndex(1)),
                Call(("zero",), (Value("a", tensor),), function.name),
                Call(("one",), (Value("b", tensor),), function.name),
            ),
            ("a", "b"),
        ),
        (function,),
        (weights,),
    )
    abi = NativeKernelImport(function, "test.borrowed_weights.current_stream@1", "cuda:0", ())
    cells = {function.name: external_primitive(function, abi)}
    plan = lower_region_streams(program, "{ S0[] -> [0]; S1[] -> [1] }", cells, axis=0, lanes=2)
    runner = stream_execution(program, cells, plan, payload_limits=(RegionPayloadLimit("cuda:0", 1024),))
    value = torch.tensor([3.0, 7.0], device="cuda:0")
    view = ReadonlyTensorView(weights, tuple(ReadonlyTensorCell.bind((i,), (value,)) for i in range(2)))
    launches = []

    def endpoint(inputs):
        launches.append(torch.cuda.current_stream().cuda_stream)
        return (value,)

    def resources(local_abi):
        return RegionResources(
            program,
            invocation.context,
            (),
            (ExternalFunction(function, endpoint, local_abi),),
            borrows=(ReadonlyResourceBinding(view, invocation.context),),
        )

    owner.wait_stream(torch.cuda.current_stream())
    try:
        for different in (None, replace(abi, implementation="another.kernel@1")):
            with pytest.raises(ValueError, match="execution contract"):
                runner.run({}, pages=resources(different), streams=streams)
        assert not launches
        result = runner.run({}, pages=resources(abi), streams=streams)
        assert set(launches) == {owner.cuda_stream, auxiliary.cuda_stream}
        with torch.cuda.stream(owner):
            for output in result.values:
                torch.testing.assert_close(output, value, rtol=0, atol=0)
                assert output.data_ptr() != value.data_ptr()
        assert not invocation.context.page_publications
    finally:
        execution.cancel_step(invocation)
        execution.retire(tuple(row.identity for row in rows))
