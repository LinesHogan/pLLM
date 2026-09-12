"""Actual Triton GEMM code, readonly ports and source-selected batch lowering."""

import pytest
import torch

from pllm.config import ExecutionOptions
from pllm.contracts.region_program import Block, Call, Program, Region, Value
from pllm.contracts.region_selection import RegionPayloadLimit
from pllm.execution.exported_cell import ExportedCell, NumericalTransformWarning
from pllm.execution.region_native import CompiledPrimitive
from pllm.execution.region_optimization import optimize_region
from pllm.execution.tensor_compilation import compile_tensor_cell


@pytest.mark.parametrize("batch", (False, True))
def test_gemm_epilogue_is_one_generated_kernel_with_unaligned_readonly_borrows(batch):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA Triton templates")
    leading = (2,) if batch else ()
    shapes = ((*leading, 32, 64), (*leading, 64, 48))
    bases = tuple(torch.arange(1 + torch.Size(shape).numel(), device="cuda", dtype=torch.float32)
                  .remainder(7).to(torch.bfloat16) for shape in shapes)
    tensors = tuple(base[1:].reshape(shape) for base, shape in zip(bases, shapes, strict=True))
    before = tuple(base.clone() for base in bases)
    source = ExportedCell.capture(lambda x, w: ((x @ w) + 1,), ("x", "w"), ("out",), tensors)
    compiled = compile_tensor_cell(source)
    assert sum(buffer.bytes for buffer in compiled.implementation.allocations) == source.output_bytes
    expected = (tensors[0].float() @ tensors[1].float()).to(torch.bfloat16) + 1
    result, = compiled.invoke(tensors, None)
    torch.testing.assert_close(result, expected)
    for actual, saved in zip(bases, before, strict=True):
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
    # A separate local compiler replay must recover the same selected template,
    # code identity and allocation facts, without depending on trial timings.
    assert compile_tensor_cell(source).implementation == compiled.implementation


def test_automatic_batching_can_select_compiled_gemm_with_packing_and_output_copies(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA Triton templates")
    import pllm.execution.region_optimization as tuning
    shapes = ((32, 64), (64, 48), (32, 64), (64, 48))
    tensors = tuple(torch.arange(torch.Size(shape).numel(), device="cuda", dtype=torch.float32)
                    .remainder(7).reshape(shape).to(torch.bfloat16) for shape in shapes)
    source = ExportedCell.capture(lambda x, w: (x @ w,), ("x", "w"), ("out",), tensors[:2])
    left, right = source.declaration.inputs
    output, = source.declaration.outputs
    names = ("a", "b", "c", "d")
    program = Program(Region(Block(tuple(Value(name, type_) for name, type_ in zip(
        names, (left, right, left, right), strict=True)), (
        Call(("a", "b"), (Value("first", output),), source.digest),
        Call(("c", "d"), (Value("second", output),), source.digest),
    ), ("first", "second"))), (source.declaration,))
    monkeypatch.setattr(tuning, "_measure", lambda execution, *args:
        1 if any(isinstance(cell, CompiledPrimitive) and any("batching" in t for t in cell.transformations)
                 for cell in execution._cells.values()) else 100)
    with pytest.warns(NumericalTransformWarning, match="batching.*kernel compilation"):
        selected = optimize_region(program, {source.digest: source}, dict(zip(names, tensors)),
            payload_limits=(RegionPayloadLimit("cuda:0", 100000),),
            options=ExecutionOptions(schedule_policy="serial"), warmup_rounds=0, measurement_rounds=1)
    record, = selected.rewrites
    compiled = selected._execution._cells[record.source_digest]
    assert isinstance(compiled, CompiledPrimitive)
    assert sum(a.bytes for a in compiled.implementation.allocations) >= record.packed_operand_bytes + 2 * source.output_bytes
    # The second returned source slice has a nonzero offset in the batched
    # allocation. The selected compiled ABI still returns independent roots.
    warm = selected.run(dict(zip(names, tensors)))
    del warm
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    def forbidden(*args, **kwargs):
        raise AssertionError("selected GEMM must not compile during execution")
    monkeypatch.setattr(tuning, "compile_tensor_cell", forbidden)
    from triton.runtime.jit import JITFunction
    monkeypatch.setattr(JITFunction, "_do_compile", forbidden)
    actual = selected.run(dict(zip(names, tensors))).values
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - allocated <= (
        selected.selection.candidates[selected.selection.selected].capacity.devices[0].peak_tensor_bytes_upper_bound)
    for result, x, w in zip(actual, tensors[::2], tensors[1::2], strict=True):
        torch.testing.assert_close(result, (x.float() @ w.float()).to(torch.bfloat16))
    assert actual[0].untyped_storage().data_ptr() != actual[1].untyped_storage().data_ptr()


def test_fused_bfloat16_recurrence_preserves_intermediate_rounding():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA Triton templates")
    generator = torch.Generator(device="cuda").manual_seed(127)
    base = torch.randn((7, 5), dtype=torch.bfloat16, device="cuda", generator=generator).T
    delta = torch.zeros((5, 7), dtype=base.dtype, device=base.device)
    inputs, targets = [], []
    for rows in (1, 7):
        inputs.append(torch.randn((rows, 7), dtype=base.dtype, device=base.device, generator=generator))
        targets.append(torch.randn((rows, 5), dtype=base.dtype, device=base.device, generator=generator))
    x, y = torch.cat(inputs), torch.cat(targets)

    def step(delta, base, x, y):
        return delta + .017 * (y.T @ x), x @ (base + delta).T

    source = ExportedCell.capture(step, ("delta", "base", "x", "y"), ("next", "out"),
                                  (delta, base, x[:3], y[:3]))
    compiled = compile_tensor_cell(source)
    actual, expected = delta, delta
    # The second update exposes rounding lost by fusing the bf16 multiply
    # into the carry addition; it also changes the subsequent remainder.
    for start in (0, 3):
        actual, output = compiled.invoke((actual, base, x[start:start + 3], y[start:start + 3]), None)
        expected, reference = step(expected, base, x[start:start + 3], y[start:start + 3])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
    torch.testing.assert_close(x[6:] @ (base + actual).T, x[6:] @ (base + expected).T, rtol=0, atol=0)


def test_transposed_weight_borrow_keeps_stride_offset_and_has_no_materialization():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA Triton templates")
    x = torch.arange(32 * 64, device="cuda", dtype=torch.float32).remainder(7).reshape(32, 64).to(torch.bfloat16)
    base = torch.arange(1 + 48 * 64, device="cuda", dtype=torch.float32).remainder(5).to(torch.bfloat16)
    weight = base[1:].view(48, 64).T
    before = base.clone()
    source = ExportedCell.capture(lambda x, w: (x @ w,), ("x", "weight"), ("out",), (x, weight))
    compiled = compile_tensor_cell(source)
    assert sum(buffer.bytes for buffer in compiled.implementation.allocations) == source.output_bytes
    result, = compiled.invoke((x, weight), None)
    torch.testing.assert_close(result, (x.float() @ weight.float()).to(torch.bfloat16))
    torch.testing.assert_close(base, before, rtol=0, atol=0)
    # Compiler facts contain only the raw GEMM result; the borrow has neither
    # an input copy nor a transpose materialization inside the wrapper.
    del result
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result, = compiled.invoke((x, weight), None)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - allocated == 2 * source.output_bytes
