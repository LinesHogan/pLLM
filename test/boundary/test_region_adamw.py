"""An imported AdamW recurrence keeps every original optimizer field bitwise."""

import pytest
import torch

from pllm.contracts.region_program import TensorType
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.region_adamw import adamw_cell
from pllm.execution.tensor_scan import TensorScan
from pllm.kernels.adamw import AdamWConfig, AdamWFields, apply_adamw


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
@pytest.mark.parametrize("count", (0, 5))
def test_imported_adamw_scan_preserves_the_existing_kernel_path(device, dtype, count):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    config = AdamWConfig(0.013, betas=(0.81, 0.97), eps=1e-4, weight_decay=0.07)
    cell = adamw_cell(TensorType((3, 7), str(dtype), device), config)
    scan = TensorScan(cell, count, carry=5, invariants=0)
    generator = torch.Generator(device=device).manual_seed(37)
    initial = (torch.randn((3, 7), dtype=dtype, device=device, generator=generator),
        torch.zeros((3, 7), dtype=dtype, device=device), torch.tensor(11., device=device),
        torch.randn((3, 7), dtype=dtype, device=device, generator=generator),
        torch.rand((3, 7), dtype=dtype, device=device, generator=generator))
    gradients = tuple(torch.randn((3, 7), dtype=dtype, device=device, generator=generator) for _ in range(count))
    expected = tuple(t.clone() for t in initial)
    saved = tuple(t.clone() for t in initial)
    fields = AdamWFields(*(LayoutFieldId(i) for i in range(5)))
    for gradient in gradients:
        apply_adamw((expected,), fields, config, (gradient,))
    result = scan(initial, (), (gradients,))
    assert result.columns == ()
    for actual, want, original, before in zip(result.carry, expected, initial, saved, strict=True):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)
        torch.testing.assert_close(original, before, rtol=0, atol=0)
    assert result.carry[2].item() == 11 + count


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
def test_heterogeneous_adamw_region_preserves_one_original_foreach_batch(device, dtype, monkeypatch):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from pllm import _core
    from pllm.execution.region_adamw import AdamWRegion
    import pllm.execution.region_adamw as imported
    config = AdamWConfig(0.023, betas=(0.79, 0.93), eps=1e-4, weight_decay=0.13)
    parameters = (TensorType((2, 7), str(dtype), device), TensorType((3,), "torch.float32", device))
    program = AdamWRegion(parameters, config)
    generator = torch.Generator(device=device).manual_seed(417)
    roots = tuple((torch.randn(shape, dtype=t, device=device, generator=generator),
        torch.full(shape, float("nan"), dtype=t, device=device), torch.tensor(7., device=device),
        torch.randn(shape, dtype=t, device=device, generator=generator),
        torch.rand(shape, dtype=t, device=device, generator=generator))
        for shape, t in (((2, 7), dtype), ((3,), torch.float32)))
    expected = tuple(tuple(value.clone() for value in root) for root in roots)
    fields = AdamWFields(*(LayoutFieldId(i) for i in range(5)))
    calls = []
    original = imported.apply_adamw
    def batch_update(roots, fields, config, gradients):
        calls.append(tuple(root[0].shape for root in roots))
        return original(roots, fields, config, gradients)
    monkeypatch.setattr(imported, "apply_adamw", batch_update)
    def forbidden(*args):
        raise AssertionError("optimizer execution must not compile")
    for name in ("analyze_region_program", "lower_region_schedule", "lower_region_bindings"):
        monkeypatch.setattr(_core, name, forbidden)
    for _ in range(4):
        gradients = tuple(torch.randn(root[0].shape, dtype=root[0].dtype, device=device, generator=generator) for root in roots)
        saved = tuple(tuple(value.clone() for value in root) for root in roots)
        apply_adamw(expected, fields, config, gradients)
        updated = program(roots, gradients)
        for result, want, old, before in zip(updated, expected, roots, saved, strict=True):
            for value, reference, input_, snapshot in zip(result, want, old, before, strict=True):
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
                torch.testing.assert_close(input_, snapshot, rtol=0, atol=0, equal_nan=True)
        roots = updated
    assert calls == [((2, 7), (3,))] * 4
