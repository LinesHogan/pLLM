"""Real Torch export imported into structured loops with immutable state values."""

import warnings

import pytest
import torch

from pllm.contracts.region_program import Block, Call, Program, Region, ScalarType, Scan, Value
from pllm.execution.exported_cell import ExportedCell, NumericalTransformWarning, warn_numerical_transforms
from test.region_reference import interpret


def update_cell(hidden, weights):
    output = hidden @ weights
    weights.add_(0.01 * (hidden.T @ output))
    return output, weights


def test_functionalized_export_runs_in_scan_and_preserves_all_input_versions():
    torch.manual_seed(17)
    hidden, weights = torch.randn(2, 2, dtype=torch.float64), torch.randn(2, 2, dtype=torch.float64)
    saved_h, saved_w = hidden.clone(), weights.clone()
    cell = ExportedCell.capture(update_cell, ("hidden", "weights"), ("output", "next_weights"), (hidden, weights))
    assert torch.equal(weights, saved_w)
    assert cell.mutations == (("weights", "next_weights"),)
    declaration = cell.declaration
    h, w = declaration.inputs
    body = Region(
        Block(
            (Value("i", ScalarType.INDEX), Value("h", h), Value("w", w)),
            (Call(("h", "w"), (Value("next_h", h), Value("next_w", w)), declaration.name),),
            ("next_h", "next_w"),
        )
    )
    program = Program(
        Region(
            Block(
                (Value("h0", h), Value("w0", w)),
                (Scan(("h0", "w0"), (Value("hf", h), Value("wf", w)), 4, body),),
                ("hf", "wf"),
            )
        ),
        (declaration,),
    )
    result = interpret(program, {"h0": hidden, "w0": weights}, functions={declaration.name: lambda args: cell(*args)})
    expected_h, expected_w = hidden.clone(), weights.clone()
    for _ in range(4):
        expected_h, expected_w = update_cell(expected_h, expected_w)
    torch.testing.assert_close(result.values[0], expected_h, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(result.values[1], expected_w, rtol=1e-12, atol=1e-12)
    assert torch.equal(hidden, saved_h)
    assert torch.equal(weights, saved_w)
    assert len(result.calls) == 4
    cell(hidden, weights)
    assert torch.equal(weights, saved_w)  # Do not invoke ExportedProgram.module() writeback.


def test_mutation_without_explicit_return_is_rejected():
    def bad(x, w):
        w.add_(x)
        return (x * 2,)

    with pytest.raises(ValueError, match="also be returned"):
        ExportedCell.capture(bad, ("x", "w"), ("result",), (torch.ones(2), torch.zeros(2)))


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


def test_hidden_tensor_state_and_rng_are_rejected():
    hidden = torch.ones(2)

    def closed(x):
        return (x + hidden,)

    with pytest.raises(ValueError, match="explicit input ports"):
        ExportedCell.capture(closed, ("x",), ("y",), (torch.ones(2),))

    def random(x):
        return (x + torch.rand_like(x),)

    with pytest.raises(ValueError, match="RNG"):
        ExportedCell.capture(random, ("x",), ("y",), (torch.ones(2),))


def test_aliasing_cannot_be_erased_by_cloning_examples():
    data = torch.ones(2, 2)
    with pytest.raises(ValueError, match="distinct storage"):
        ExportedCell.capture(update_cell, ("h", "w"), ("y", "nw"), (data, data.view(2, 2)))
    cell = ExportedCell.capture(update_cell, ("h", "w"), ("y", "nw"), (data, data.clone()))
    with pytest.raises(ValueError, match="distinct storage"):
        cell(data, data)


def test_fake_source_import_preserves_alias_guards_and_runs_on_real_inputs():
    from torch._subclasses.fake_tensor import FakeTensorMode

    def add(x, y):
        return (x + y,)

    with FakeTensorMode():
        x = torch.empty((2, 4))
        with pytest.raises(ValueError, match="distinct storage"):
            ExportedCell.capture(add, ("x", "y"), ("out",), (x, x.view(2, 4)))
        cell = ExportedCell.capture(add, ("x", "y"), ("out",), (x, torch.empty_like(x)))
    (actual,) = cell(torch.ones((2, 4)), torch.full((2, 4), 2.0))
    torch.testing.assert_close(actual, torch.full((2, 4), 3.0), rtol=0, atol=0)


def test_export_shape_and_dtype_guards():
    cell = ExportedCell.capture(update_cell, ("h", "w"), ("y", "nw"), (torch.ones(2, 2), torch.ones(2, 2)))
    with pytest.raises(ValueError, match="guard"):
        cell(torch.ones(3, 2), torch.ones(2, 2))
    with pytest.raises(ValueError, match="guard"):
        cell(torch.ones(2, 2, dtype=torch.float64), torch.ones(2, 2))
    with pytest.raises(ValueError, match="guard"):
        cell(torch.ones(2, 2).T, torch.ones(2, 2))


def test_tensor_block_compilation_is_separate_from_capture():
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weights = torch.eye(2)
    cell = ExportedCell.capture(update_cell, ("h", "w"), ("y", "nw"), (hidden, weights))
    assert cell.transformations == ()
    candidate = cell.compiled((hidden, weights))
    assert candidate.transformations == ("torch.compile",)
    assert candidate.declaration == cell.declaration
    actual, expected = candidate(hidden, weights), cell(hidden, weights)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-5, atol=1e-6)
    assert torch.equal(weights, torch.eye(2))


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
