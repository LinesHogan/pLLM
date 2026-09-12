"""Native wrapper copies compose with source storage without modeling kernels."""

from dataclasses import replace

import pytest
import torch

from pllm import _core
from pllm.contracts.region_imports import ExportedTensorImport, NativeTensorImport, RegionTensorImport
from pllm.contracts.region_program import BankType, Block, Call, Function, If, Program, Region, ScalarType, Scan, SizeParameter, TensorType, Value
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_adamw import AdamWRegion
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_native import NativePrimitive
from pllm.execution.region_values import Bank
from pllm.execution.region_storage import analyze_region_capacity, analyze_region_storage, validate_region_capacity
from pllm.execution.tensor_access import import_tensor_access_graph
from pllm.execution.tensor_scan import TensorScan
from pllm.kernels.adamw import AdamWConfig
from test.boundary.test_tensor_access import StorageObserver


def single_call(cell):
    declaration = cell.declaration
    inputs = tuple(Value(f"in{i}", t) for i, t in enumerate(declaration.inputs))
    outputs = tuple(Value(f"out{i}", t) for i, t in enumerate(declaration.outputs))
    return Program(Region(Block(inputs, (Call(tuple(v.name for v in inputs), outputs, declaration.name),),
                                tuple(v.name for v in outputs))), (declaration,))


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_native_duplicate_views_become_independent_roots_and_preserve_input(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    full = TensorType((4, 4), "torch.float32", device)
    small = TensorType((1, 4), "torch.float32", device)
    def update(inputs, resources):
        x, = inputs
        x.add_(1)
        return x[:1], x[:1]
    cell = NativePrimitive(Function("native.views", (full,), (small, small)), update)
    program = single_call(cell)
    source = analyze_region_program(program)
    cells = {cell.declaration.name: cell}
    storage = analyze_region_storage(program, source.reference_schedule, cells)
    bound = analyze_region_capacity(program, source.reference_schedule, cells)
    assert [r.bytes for r in storage.roots] == [64, 16, 16]
    facts, = bound.devices
    assert facts.entry_copy_bytes_upper_bound == 64
    assert facts.live_tensor_bytes_upper_bound == 96
    assert facts.call_scratch_bytes_upper_bound == 64
    assert facts.peak_tensor_bytes_upper_bound == 160
    assert facts.logical_read_bytes_upper_bound == facts.logical_write_bytes_upper_bound == 160
    assert bound.unknown_functions == (cell.declaration.name,)
    execution = RegionExecution(program, lower_region_schedule(program, source.reference_schedule),
        lower_region_bindings(program, source.reference_schedule), cells)
    x = torch.ones((4, 4), device=device)
    with StorageObserver((x,)) as observed:
        result = execution.run({"in0": x})
    # This test kernel allocates no internal storage, so wrapper accounting
    # alone covers it. General native kernels still need their own bounds.
    assert observed.peak - 64 == facts.peak_tensor_bytes_upper_bound
    a, b = result.values
    assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
    torch.testing.assert_close(a, x[:1] + 1, rtol=0, atol=0)
    torch.testing.assert_close(x, torch.ones_like(x), rtol=0, atol=0)
    validate_region_capacity(program, cells, bound)


def test_export_aliases_and_native_copies_have_distinct_storage_rules():
    exported = ExportedCell.capture(lambda x: (x.view(x.shape),), ("x",), ("view",), (torch.ones(4),))
    t, = exported.declaration.inputs
    native = NativePrimitive(Function("native.identity", (t,), (t,)), lambda args, resources: args)
    program = Program(Region(Block((Value("x", t),), (
        Call(("x",), (Value("view", t),), exported.digest),
        Call(("view",), (Value("copy", t),), native.declaration.name),
    ), ("view", "copy"))), (exported.declaration, native.declaration))
    schedule = analyze_region_program(program).reference_schedule
    cells = {exported.digest: exported, native.declaration.name: native}
    storage = analyze_region_storage(program, schedule, cells)
    bound = analyze_region_capacity(program, schedule, cells)
    assert len(storage.roots) == 2
    assert bound.devices[0].peak_tensor_bytes_upper_bound == 48
    assert bound.unknown_functions == (native.declaration.name,)
    for imported in (ExportedTensorImport(import_tensor_access_graph(exported)), NativeTensorImport(native.declaration)):
        assert decode_boundary(encode_boundary(imported), RegionTensorImport) == imported


def test_native_parametric_carry_and_zero_trip_keep_wrapper_costs_separate():
    t = TensorType((4,), "torch.float32", "cpu")
    cell = NativePrimitive(Function("native.identity", (t,), (t,)), lambda args, resources: args)
    for count in (0, SizeParameter("steps", 0, 1_000_000)):
        scan = TensorScan(cell, count, carry=1, invariants=0)
        schedule = analyze_region_program(scan.program).reference_schedule
        bound = analyze_region_capacity(scan.program, schedule, {cell.declaration.name: cell})
        facts, = bound.devices
        if count == 0:
            assert facts.peak_tensor_bytes_upper_bound == 16
            assert facts.call_scratch_bytes_upper_bound == 0
            assert not bound.unknown_functions
        else:
            assert facts.peak_tensor_bytes_upper_bound == 64
            assert facts.call_scratch_bytes_upper_bound == 16
            assert facts.logical_read_bytes_upper_bound == facts.logical_write_bytes_upper_bound == 16 + 32 * 1_000_000
            assert bound.unknown_functions == (cell.declaration.name,)


def test_native_nested_bank_copies_use_symbolic_leaf_lifetimes_and_original_versions():
    tensor = TensorType((4,), "torch.float32", "cpu")
    bank = BankType(BankType(tensor, "n"), 2)
    def update(inputs, resources):
        values, = inputs
        for row in values.elements:
            for value in row.elements:
                value.add_(1)
        return values, values
    cell = NativePrimitive(Function("native.bank_update", (bank,), (bank, bank)), update)
    program = replace(single_call(cell), parameters=(SizeParameter("n", 0, 5),))
    cells = {cell.declaration.name: cell}
    schedule = analyze_region_program(program, optimize=False).reference_schedule
    storage = analyze_region_storage(program, schedule, cells)
    capacity = analyze_region_capacity(program, schedule, cells)
    assert len(storage.roots) == 3 and all(root.bytes == 16 for root in storage.roots)
    facts, = capacity.devices
    assert facts.entry_copy_bytes_upper_bound == facts.call_scratch_bytes_upper_bound == 160
    assert facts.live_tensor_bytes_upper_bound == 480
    assert facts.peak_tensor_bytes_upper_bound == 640
    assert facts.logical_read_bytes_upper_bound == facts.logical_write_bytes_upper_bound == 640
    assert capacity.unknown_functions == (cell.declaration.name,)
    execution = RegionExecution(program, lower_region_schedule(program, schedule),
        lower_region_bindings(program, schedule), cells)
    original = torch.arange(4, dtype=torch.float32)
    for count in (0, 2, 5):
        # Every input leaf aliases the same original storage. Native copies
        # separate them before mutation; each returned occurrence is copied too.
        inputs = Bank(tuple(Bank((original,) * count) for _ in range(2)))
        with StorageObserver((original,)) as observed:
            actual = execution.run({"in0": inputs}, sizes={"n": count}).values
        assert observed.peak - 16 <= facts.peak_tensor_bytes_upper_bound
        if count == 5:
            assert observed.peak - 16 == facts.peak_tensor_bytes_upper_bound
        leaves = [leaf for values in actual for row in values.elements for leaf in row.elements]
        assert len({value.untyped_storage().data_ptr() for value in leaves}) == 4 * count
        for value in leaves:
            torch.testing.assert_close(value, original + 1, rtol=0, atol=0)
        torch.testing.assert_close(original, torch.arange(4, dtype=torch.float32), rtol=0, atol=0)
    validate_region_capacity(program, cells, capacity)
    with pytest.raises(ValueError, match="extent"):
        execution.run({"in0": Bank((Bank(()), Bank(())))}, sizes={"n": 2})


def test_capacity_cannot_be_reused_across_different_execution_boundaries():
    exported = ExportedCell.capture(lambda x: (x.view(x.shape),), ("x",), ("out",), (torch.ones(4),))
    # Identical source semantics; different physical alias/copy contracts.
    native = NativePrimitive(exported.declaration, lambda args, resources: args)
    program = single_call(exported)
    schedule = analyze_region_program(program).reference_schedule
    bound = analyze_region_capacity(program, schedule, {exported.digest: exported})
    copied = analyze_region_capacity(program, schedule, {exported.digest: native})
    assert bound.devices[0].peak_tensor_bytes_upper_bound == 16
    assert copied.devices[0].peak_tensor_bytes_upper_bound == 48
    assert bound.imports_digest != copied.imports_digest
    with pytest.raises(ValueError, match="region capacity"):
        validate_region_capacity(program, {exported.digest: native}, bound)


def test_native_import_boundary_rejects_changed_declarations_and_unsupported_ports():
    t = TensorType((4,), "torch.float32", "cpu")
    cell = NativePrimitive(Function("native.identity", (t,), (t,)), lambda args, resources: args)
    program = single_call(cell)
    schedule = analyze_region_program(program).reference_schedule
    forged = NativeTensorImport(replace(cell.declaration, inputs=(replace(t, shape=(1,)),)))
    for analyze in (_core.analyze_region_capacity, _core.analyze_region_storage):
        with pytest.raises(ValueError, match="differs from its source"):
            analyze(encode_boundary(program), schedule, encode_boundary((forged,)))
        with pytest.raises(ValueError, match="complete and sorted"):
            analyze(encode_boundary(program), schedule, encode_boundary(()))
    scalar = NativePrimitive(Function("native.scalar", (ScalarType.INT,), (ScalarType.INT,)), lambda args, resources: args)
    program = single_call(scalar)
    bound = analyze_region_capacity(program, analyze_region_program(program).reference_schedule, {scalar.declaration.name: scalar})
    assert not bound.devices and bound.unknown_functions == (scalar.declaration.name,)


def test_nested_parameterized_scan_bounds_follow_slot_release_instead_of_trip_count():
    tensor = TensorType((16,), "torch.float32")
    cell = NativePrimitive(Function("copy", (ScalarType.INDEX, tensor), (tensor,)), lambda inputs, resources: (inputs[1],))
    body = Region(Block((Value("i", ScalarType.INDEX), Value("x", tensor)), (
        Call(("i", "x"), (Value("next", tensor),), cell.declaration.name),
    ), ("next",)))
    yes = Region(Block((Value("x", tensor),), (Scan(("x",), (Value("final", tensor),), "n", body),), ("final",)))
    no = Region(Block((Value("x", tensor),), (), ("x",)))
    program = Program(Region(Block((Value("enabled", ScalarType.BOOL), Value("x", tensor)), (
        If(("enabled", "x"), (Value("result", tensor),), yes, no),
    ), ("result",))), (cell.declaration,), parameters=(SizeParameter("n", 0, 1_000_000),))
    cells = {cell.declaration.name: cell}
    schedule = analyze_region_program(program, optimize=False).reference_schedule
    bound = analyze_region_capacity(program, schedule, cells)
    assert bound.devices[0].peak_tensor_bytes_upper_bound < 1024
    assert bound.unknown_functions == (cell.declaration.name,)
    runner = RegionExecution(program, lower_region_schedule(program, schedule), lower_region_bindings(program, schedule), cells)
    x = torch.arange(16, dtype=torch.float32)
    for enabled, n in ((False, 37), (True, 0), (True, 1), (True, 19)):
        with StorageObserver((x,)) as observed:
            result = runner.run({"enabled": enabled, "x": x}, sizes={"n": n})
        assert observed.peak - x.untyped_storage().nbytes() <= bound.devices[0].peak_tensor_bytes_upper_bound
        torch.testing.assert_close(result.values[0], x, rtol=0, atol=0)
    validate_region_capacity(program, cells, bound)


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_adamw_bridge_capacity_covers_real_wrapper_overlap(monkeypatch, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    import pllm.execution.region_adamw as imported
    # Remove kernel internals from this storage measurement. The production
    # test separately checks the unchanged real kernel and original reservation.
    monkeypatch.setattr(imported, "apply_adamw", lambda *args: None)
    shapes = ((2, 3), (3,))
    optimizer = AdamWRegion(tuple(TensorType(s, "torch.float32", device) for s in shapes), AdamWConfig(0.01))
    roots = tuple((torch.ones(s, device=device), torch.zeros(s, device=device), torch.tensor(0., device=device),
                   torch.zeros(s, device=device), torch.zeros(s, device=device)) for s in shapes)
    gradients = tuple(torch.ones(s, device=device) for s in shapes)
    external = (*tuple(t for group in roots for t in group), *gradients)
    external_bytes = sum(t.untyped_storage().nbytes() for t in external)
    with StorageObserver(external) as observed:
        result = optimizer(roots, gradients)
    assert len(result) == 2
    assert observed.peak - external_bytes == optimizer.copy_workspace_bytes == 13 * 36 + 2 * 12
    assert optimizer.capacity.unknown_functions == (optimizer.program.functions[0].name,)
