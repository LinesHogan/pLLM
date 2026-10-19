"""Tensor cells: capturing user tensor math and existing exports as functional cells.

``ExportedCell.capture`` / ``from_export`` and module capture turn PyTorch code
into explicit-port functional graphs. They must preserve every input version,
keep mutation results explicit, guard shapes/dtypes/layouts, never bind model
weights implicitly, and reject untraceable user code with an actionable error.
"""

import warnings
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer

from pllm.contracts.region_program import Block, Call, Program, Region, ScalarType, Scan, Value
from pllm.execution.exported_cell import ExportedCell, NumericalTransformWarning, warn_numerical_transforms
from pllm.execution.module_graph import capture_module_export, capture_module_partition, exported_parameter_fields
from test.region_fixtures import reassociation_candidates
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
    with (
        patch.object(torch.export, "export", side_effect=AssertionError("must not recapture")),
        patch.object(program.graph_module, "forward", side_effect=AssertionError("must not execute graph")),
    ):
        cell = ExportedCell.from_export(program)
    assert cell.inputs == ("left", "right", "x")
    # Functionalization may derive a normalized graph; the supplied source is
    # unchanged and its lifted weights do not become owned graph parameters.
    assert not tuple(cell.graph.parameters()) and not tuple(cell.graph.buffers())
    assert str(program.graph) == source_text
    expected = model(x)
    (actual,) = cell(model.left, model.right, x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (candidate,) = reassociation_candidates(cell)
    (rewritten,) = candidate.cell(model.left, model.right, x)
    torch.testing.assert_close(rewritten, expected, rtol=1e-12, atol=1e-12)
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
    torch.testing.assert_close(next_total, original + x, rtol=0, atol=0)
    with pytest.raises(ValueError, match="complete graph signature"):
        ExportedCell.from_export(program, outputs=("prediction",))


class Norm(torch.nn.Module):
    def __init__(self, calls, name):
        super().__init__()
        self.calls, self.name = calls, name

    def forward(self, hidden, residual=None):
        self.calls.append(self.name)
        if residual is None:
            return hidden / 2
        residual.add_(hidden)
        hidden.copy_(residual / 2)
        return hidden, residual


class Attention(torch.nn.Module):
    def __init__(self, calls, kv):
        super().__init__()
        self.calls, self.kv = calls, kv

    def forward(self, positions, hidden_states):
        self.calls.append("kv-write")
        self.kv.copy_(hidden_states)
        return hidden_states + positions[:, None]


class Dense(torch.nn.Module):
    def __init__(self, calls):
        super().__init__()
        self.calls = calls

    def forward(self, hidden):
        self.calls.append("dense")
        return hidden * 3


def test_export_binds_actual_parameter_aliases_and_rejects_same_shape_copies():
    class Source(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))

        def forward(self, x):
            x.add_(self.weight.data)
            return x * 2

    module = Source()
    x = torch.ones(4)
    original = module.weight
    captured = capture_module_export(module, (x,), parameters={"weight": module.weight.detach()})
    exported = captured.program
    captured.validate()
    fields = exported_parameter_fields(exported, (original,))
    assert len(fields) == 1 and set(fields.values()) == {0}
    assert module.weight is original
    torch.testing.assert_close(x, torch.ones(4), rtol=0, atol=0)
    torch.testing.assert_close(module.weight, torch.arange(4, dtype=torch.float32), rtol=0, atol=0)
    with pytest.raises(ValueError, match="not a field of the original owner"):
        exported_parameter_fields(exported, (original.detach().clone(),))


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
    with pytest.raises(ValueError, match="Torch capture guards"):
        captured.validate()
    module.settings["scale"] = 0.5
    captured.validate()
    module.enabled = False
    with pytest.raises(ValueError, match="Torch capture guards"):
        captured.validate()
    module.enabled = True
    captured.validate()
    offset_view = module.offset.detach()
    module.offset.set_(torch.zeros_like(module.offset))
    with pytest.raises(ValueError, match="tensor field changed"):
        captured.validate()
    module.offset.set_(offset_view)
    captured.validate()
    weight = module.weight
    module.weight = torch.nn.Parameter(weight.detach().clone())
    with pytest.raises(ValueError, match="parameter fields changed"):
        captured.validate()
    module.weight = weight
    captured.validate()
    module.forward = lambda x: x
    with pytest.raises(ValueError, match="model call changed"):
        captured.validate()


@pytest.mark.parametrize("has_residual", (False, True))
def test_original_qwen_forward_is_partitioned_with_mutations_and_none_guard(has_residual):
    # Use the real source method, with small CPU leaves. Capturing must never
    # invoke those leaves or construct/load a model's parameter storage.
    layer = Qwen2DecoderLayer.__new__(Qwen2DecoderLayer)
    torch.nn.Module.__init__(layer)
    calls, kv = [], torch.empty((1, 4), dtype=torch.float64)
    layer.input_layernorm = Norm(calls, "input-norm")
    layer.self_attn = Attention(calls, kv)
    layer.post_attention_layernorm = Norm(calls, "post-norm")
    layer.mlp = Dense(calls)
    piece = capture_module_partition(
        layer,
        stateful=("self_attn",),
        readonly=("input_layernorm", "post_attention_layernorm", "mlp"),
        none_inputs=() if has_residual else ("residual",),
    )
    assert not calls
    positions, hidden = torch.tensor([2]), torch.randn((1, 4), dtype=torch.float64)
    residual = torch.randn_like(hidden) if has_residual else None
    expected = layer(positions, hidden.clone(), None if residual is None else residual.clone())
    expected_kv = kv.clone()
    calls.clear()
    frontier = piece.run_prefix((positions, hidden.clone(), None if residual is None else residual.clone()))
    assert calls == ["input-norm", "kv-write"]
    torch.testing.assert_close(kv, expected_kv, rtol=0, atol=0)
    result = piece.run_suffix(frontier)
    assert calls == ["input-norm", "kv-write", "post-norm", "dense"]
    for actual, reference in zip(result, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert piece.prefix.self_attn is layer.self_attn
    assert piece.suffix.mlp is layer.mlp
    if not has_residual:
        with pytest.raises(AssertionError, match="specialized"):
            piece.run_prefix((positions, hidden, torch.ones_like(hidden)))


def test_partition_uses_source_dataflow_for_a_different_module_structure():
    class Different(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []
            self.prepare = Dense(self.calls)
            self.publish = Attention(self.calls, torch.empty((1, 4)))
            self.finish = Norm(self.calls, "finish")

        def forward(self, position, value):
            value = self.publish(position, self.prepare(value))
            return (self.prepare(self.finish(value)),)

    module = Different()
    piece = capture_module_partition(module, stateful=("publish",), readonly=("prepare", "finish"))
    assert not module.calls
    position, value = torch.tensor([1]), torch.randn(1, 4)
    expected = module(position, value)
    module.calls.clear()
    result = piece.run_suffix(piece.run_prefix((position, value)))
    assert module.calls == ["dense", "kv-write", "finish", "dense"]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    module.calls.clear()
    with pytest.raises(ValueError, match="no resource contract"):
        capture_module_partition(module, stateful=("publish",), readonly=("finish",))
    assert not module.calls


def test_capture_rejects_user_mistakes_with_actionable_errors():
    def unreturned(x, w):
        w.add_(x)
        return (x * 2,)

    hidden = torch.ones(2)

    def closed(x, w):
        return (x + hidden,)

    def random(x, w):
        return (x + torch.rand_like(x),)

    for function, message in ((unreturned, "also be returned"), (closed, "explicit input ports"), (random, "RNG")):
        with pytest.raises(ValueError, match=message):
            ExportedCell.capture(function, ("x", "w"), ("y",), (torch.ones(2), torch.zeros(2)))
    data = torch.ones(2, 2)
    with pytest.raises(ValueError, match="distinct storage"):
        ExportedCell.capture(update_cell, ("h", "w"), ("y", "nw"), (data, data.view(2, 2)))
    cell = ExportedCell.capture(update_cell, ("h", "w"), ("y", "nw"), (data, data.clone()))
    with pytest.raises(ValueError, match="distinct storage"):
        cell(data, data)

    class Effects(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.events = []

        def forward(self, x):
            self.events.append("executed")
            return x + 1

    module = Effects()
    with pytest.raises(RuntimeError, match="side effects"):
        capture_module_export(module, (torch.ones(4),))
    assert module.events == []

    class Scale(torch.nn.Module):
        def forward(self, x, scale=2):
            return x * scale

    dynamic = torch.export.export(
        Scale(), (torch.ones(3, 4),), dynamic_shapes=({0: torch.export.Dim("rows", min=1, max=8)},)
    )
    with pytest.raises(ValueError, match="symbolic constraints"):
        ExportedCell.from_export(dynamic)
    with pytest.raises(ValueError, match="explicit tensor ports"):
        ExportedCell.from_export(torch.export.export(Scale(), (torch.ones(3, 4), 2)))
