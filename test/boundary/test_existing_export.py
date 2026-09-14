"""Original export graphs enter the middle-end with explicit lifted state."""

import math
from unittest.mock import patch

import pytest
import torch

from pllm.backends.torch_inductor import COMPILER_SLACK_FRACTION
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.tensor_compilation import compile_tensor_cell
from pllm.execution.tensor_rewrites import matmul_reassociation_candidates


def test_existing_model_graph_import_does_not_recapture_execute_or_bind_weights():
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Parameter(torch.randn(2, 16, dtype=torch.float64))
            self.right = torch.nn.Parameter(torch.randn(16, 3, dtype=torch.float64))

        def forward(self, x):
            return (x @ self.left) @ self.right

    model, x = Projection(), torch.randn(12, 2, dtype=torch.float64)
    program = torch.export.export(model, (x,))
    source_text = str(program.graph)
    saved = tuple(parameter.clone() for parameter in model.parameters())
    with patch.object(torch.export, "export", side_effect=AssertionError("must not recapture")), \
         patch.object(program.graph_module, "forward", side_effect=AssertionError("must not execute graph")):
        cell = ExportedCell.from_export(program)
    assert cell.inputs == ("left", "right", "x")
    # Functionalization may derive a normalized graph; the supplied source is
    # unchanged and its lifted weights do not become owned graph parameters.
    assert not tuple(cell.graph.parameters()) and not tuple(cell.graph.buffers())
    assert str(program.graph) == source_text
    expected = model(x)
    actual, = cell(model.left, model.right, x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    candidate, = matmul_reassociation_candidates(cell)
    rewritten, = candidate.cell(model.left, model.right, x)
    torch.testing.assert_close(rewritten, expected, rtol=1e-12, atol=1e-12)
    # Another version is an explicit input, never an implicit read from model.
    next_left = model.left.detach() * 2
    changed, = cell(next_left, model.right, x)
    torch.testing.assert_close(changed, (x @ next_left) @ model.right, rtol=0, atol=0)
    for parameter, original in zip(model.parameters(), saved, strict=True):
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)


def test_lifted_buffer_update_is_a_returned_version_without_module_writeback():
    class Accumulator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("total", torch.zeros(4, dtype=torch.float64))

        def forward(self, x):
            self.total.add_(x)
            return self.total * 2

    model, x = Accumulator(), torch.arange(4, dtype=torch.float64)
    program = torch.export.export(model, (x,))
    original = model.total.clone()
    cell = ExportedCell.from_export(program, outputs=("next_total", "prediction"))
    assert cell.inputs == ("total", "x")
    assert cell.mutations == (("total", "next_total"),)
    next_total, prediction = cell(model.total, x)
    torch.testing.assert_close(next_total, original + x, rtol=0, atol=0)
    torch.testing.assert_close(prediction, (original + x) * 2, rtol=0, atol=0)
    final_total, final_prediction = cell(next_total, x)
    torch.testing.assert_close(final_total, original + 2 * x, rtol=0, atol=0)
    torch.testing.assert_close(final_prediction, (original + 2 * x) * 2, rtol=0, atol=0)
    torch.testing.assert_close(model.total, original, rtol=0, atol=0)
    torch.testing.assert_close(next_total, original + x, rtol=0, atol=0)
    with pytest.raises(ValueError, match="complete graph signature"):
        ExportedCell.from_export(program, outputs=("prediction",))


def test_tensor_import_cannot_silently_specialize_dynamic_constraints_or_scalar_ports():
    class Scale(torch.nn.Module):
        def forward(self, x, scale=2):
            return x * scale

    dynamic = torch.export.export(Scale(), (torch.ones(3, 4),),
        dynamic_shapes=({0: torch.export.Dim("rows", min=1, max=8)},))
    with pytest.raises(ValueError, match="symbolic constraints"):
        ExportedCell.from_export(dynamic)
    specialized = torch.export.export(Scale(), (torch.ones(3, 4), 2))
    with pytest.raises(ValueError, match="explicit tensor ports"):
        ExportedCell.from_export(specialized)


@pytest.mark.parametrize("device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"))))
def test_original_activation_graph_uses_backend_support_without_an_operator_whitelist(device):
    class Activation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.full((128,), 0.5, device=device))

        def forward(self, x):
            return torch.nn.functional.silu(x * self.scale)

    model = Activation()
    x = torch.linspace(-2, 2, 128, device=device)
    source = ExportedCell.from_export(torch.export.export(model, (x,)))
    assert source.inputs == ("scale", "x")
    saved = model.scale.clone(), x.clone()
    compiled = compile_tensor_cell(source)
    actual, = compiled.invoke((model.scale, x), None)
    torch.testing.assert_close(actual, model(x), rtol=1e-5, atol=1e-6)
    # Graph-derived account: silu decomposes to five pointwise values at the
    # pinned Torch version; each is charged one span, plus the named slack.
    subtotal = 5 * x.numel() * x.element_size()
    assert sum(allocation.bytes for allocation in compiled.implementation.allocations) == (
        subtotal + math.ceil(subtotal * COMPILER_SLACK_FRACTION))
    for value, original in zip((model.scale, x), saved, strict=True):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
