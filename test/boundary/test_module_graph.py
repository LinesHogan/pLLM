"""Partition the actual source method without executing or rebuilding its leaves."""

import pytest
import torch
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer

from pllm.execution.module_graph import capture_module_export, capture_module_partition, exported_parameter_fields


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


def test_capture_rejects_python_effects_that_cannot_enter_the_tensor_graph():
    class Source(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.events = []

        def forward(self, x):
            self.events.append("executed")
            return x + 1

    module = Source()
    with pytest.raises(RuntimeError, match="side effects"):
        capture_module_export(module, (torch.ones(4),))
    assert module.events == []


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
