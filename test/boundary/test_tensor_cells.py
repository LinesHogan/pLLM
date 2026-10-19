"""Tensor cells: capturing user tensor math and existing exports as functional cells.

``ExportedCell.capture`` / ``from_export`` and module capture turn PyTorch code
into explicit-port functional graphs. They must preserve every input version,
keep mutation results explicit, guard shapes/dtypes/layouts, never bind model
weights implicitly, and reject untraceable user code with an actionable error.
"""

import warnings

import pytest
import torch

from pllm.execution.exported_cell import ExportedCell, NumericalTransformWarning, warn_numerical_transforms
from pllm.execution.module_graph import capture_module_export


def update_cell(hidden, weights):
    output = hidden @ weights
    weights.add_(0.01 * (hidden.T @ output))
    return output, weights


def test_functionalized_export_repeats_the_update_and_preserves_all_input_versions():
    torch.manual_seed(17)
    hidden, weights = torch.randn(2, 2, dtype=torch.float64), torch.randn(2, 2, dtype=torch.float64)
    saved_h, saved_w = hidden.clone(), weights.clone()
    cell = ExportedCell.capture(update_cell, ("hidden", "weights"), ("output", "next_weights"), (hidden, weights))
    assert cell.mutations == (("weights", "next_weights"),)
    actual, expected = (hidden, weights), (hidden.clone(), weights.clone())
    for _ in range(4):
        actual, expected = cell(*actual), update_cell(*expected)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=1e-12, atol=1e-12)
    assert torch.equal(hidden, saved_h) and torch.equal(weights, saved_w)


def test_split_views_preserve_mutation_aliases_empty_parts_and_noncontiguous_inputs():
    def update(x):
        first, empty, tail = x.split((2, 0, 3), dim=-1)
        first.add_(2)
        return x, first, empty, tail

    original = torch.arange(15, dtype=torch.float64).reshape(5, 3).T
    saved = original.clone()
    cell = ExportedCell.capture(update, ("x",), ("updated", "first", "empty", "tail"), (original,))
    expected = update(original.clone())
    assert cell.mutations == (("x", "updated"),)
    for actual, reference in zip(cell(original), expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(original, saved, rtol=0, atol=0)


def test_export_shape_and_dtype_guards():
    cell = ExportedCell.capture(update_cell, ("h", "w"), ("y", "nw"), (torch.ones(2, 2), torch.ones(2, 2)))
    for wrong in (torch.ones(3, 2), torch.ones(2, 2, dtype=torch.float64), torch.ones(2, 2).T):
        with pytest.raises(ValueError):
            cell(wrong, torch.ones(2, 2))


def test_numerical_warning_requires_selected_changes_and_describes_opt_out():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_numerical_transforms("test-no-change", ())
        assert not caught
        warn_numerical_transforms("test-selected-fusion", ("reduction fusion",))
        warn_numerical_transforms("test-selected-fusion", ("reduction fusion",))
    assert len(caught) == 1
    assert caught[0].category is NumericalTransformWarning
    assert "reduction fusion" in str(caught[0].message)
    assert "allow_numerical_transforms=False" in str(caught[0].message)


def test_existing_model_graph_import_takes_weights_as_explicit_inputs():
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Parameter(torch.randn(2, 16, dtype=torch.float64))
            self.right = torch.nn.Parameter(torch.randn(16, 3, dtype=torch.float64))

        def forward(self, x):
            return (x @ self.left) @ self.right

    model, x = Projection(), torch.randn(12, 2, dtype=torch.float64)
    saved = tuple(parameter.clone() for parameter in model.parameters())
    cell = ExportedCell.from_export(torch.export.export(model, (x,)))
    assert cell.inputs == ("left", "right", "x")
    (actual,) = cell(model.left, model.right, x)
    torch.testing.assert_close(actual, model(x), rtol=0, atol=0)
    # Another version is an explicit input, never an implicit read from model.
    next_left = model.left.detach() * 2
    (changed,) = cell(next_left, model.right, x)
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
    with pytest.raises(ValueError):  # The buffer update must be named as an output.
        ExportedCell.from_export(program, outputs=("prediction",))


def test_capture_guards_follow_original_attributes_and_allow_weight_value_updates():
    class Source(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
            self.settings = {"scale": 0.5}
            self.enabled = True
            self.register_buffer("offset", torch.ones(4))

        def forward(self, x):
            value = x + self.weight.data + self.offset
            result = value * self.settings["scale"] if self.enabled else value
            return {"value": result, "inputs": (x, x)}

    module, x = Source(), torch.ones(4)
    captured = capture_module_export(module, (x,), parameters={"weight": module.weight.detach()})
    captured.validate()
    torch.testing.assert_close(captured.program.module()(x), module(x), rtol=0, atol=0)
    returned = captured.program.module()(x)
    assert returned["inputs"][0] is x and returned["inputs"][1] is x
    with torch.no_grad():
        module.weight.add_(2)
    captured.validate()
    torch.testing.assert_close(captured.program.module()(x), module(x), rtol=0, atol=0)
    module.settings["scale"] = 0.25
    with pytest.raises(ValueError):
        captured.validate()
    module.settings["scale"] = 0.5
    captured.validate()
    module.forward = lambda x: x
    with pytest.raises(ValueError):
        captured.validate()


def test_capture_rejects_user_mistakes_with_actionable_errors():
    def unreturned(x, w):
        w.add_(x)
        return (x * 2,)

    hidden = torch.ones(2)

    def closed(x, w):
        return (x + hidden,)

    def random(x, w):
        return (x + torch.rand_like(x),)

    for function in (unreturned, closed, random):
        with pytest.raises(ValueError):
            ExportedCell.capture(function, ("x", "w"), ("y",), (torch.ones(2), torch.zeros(2)))
    data = torch.ones(2, 2)
    cell = ExportedCell.capture(update_cell, ("h", "w"), ("y", "nw"), (data, data.clone()))
    with pytest.raises(ValueError):  # Aliased inputs would make the mutation order ambiguous.
        cell(data, data)

    class Effects(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.events = []

        def forward(self, x):
            self.events.append("executed")
            return x + 1

    module = Effects()
    with pytest.raises(RuntimeError):
        capture_module_export(module, (torch.ones(4),))
    assert module.events == []
