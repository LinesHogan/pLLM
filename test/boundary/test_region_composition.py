"""Backend bodies expose internal dataflow before the original ISL analysis."""

from dataclasses import replace

import pytest
import torch

from pllm.backends.vllm.region_layers import VllmDecoderBodies, decoder_partition_bodies
from pllm.contracts.backend_attachment import CausalModelDomain
from pllm.contracts.region_composition import RegionFunctionBody, inline_function_bodies, inline_region
from pllm.contracts.region_program import BankType, Block, BorrowedResource, Call, Constant, Function, Map, Program, Region, ScalarType, SizeParameter, TensorType, Value
from pllm.execution.module_graph import ModuleCallBinding
from pllm.execution.region_compilation import analyze_region_program, lower_region_schedule


@pytest.mark.parametrize("invalid", ("output", "resource", "borrow_access", "parameter", "capture", "self_call", "self_import"))
def test_outer_function_signature_is_checked_without_becoming_a_callable(invalid):
    integer = ScalarType.INT
    program = Program(Region(Block((Value("x", integer),), (), ("x",))))
    declaration = Function("outer", (integer,), (integer,))
    RegionFunctionBody(declaration, program).validate()
    if invalid == "output":
        declaration = replace(declaration, outputs=(ScalarType.BOOL,))
    elif invalid == "resource":
        declaration = replace(declaration, resources=("missing",))
    elif invalid == "borrow_access":
        program = replace(program, resources=(BorrowedResource("weights", (1,)),))
        declaration = replace(declaration, resources=("weights",))
    elif invalid == "parameter":
        program = replace(program, parameters=(SizeParameter("n", 0, 1),))
    elif invalid == "capture":
        program = replace(program, entry=Region(replace(program.entry.block, yields=("hidden",))))
    elif invalid == "self_call":
        program = replace(program, entry=Region(replace(program.entry.block,
            operations=(Call(("x",), (Value("y", integer),), declaration.name),), yields=("y",))))
    else:
        program = replace(program, functions=(declaration,))
    with pytest.raises(ValueError):
        RegionFunctionBody(declaration, program).validate()


def test_inlining_preserves_argument_aliases_duplicate_returns_and_fresh_names():
    integer = ScalarType.INT
    leaf = Program(Region(Block((Value("x", integer),), (
        Constant((), (Value("one", integer),), 1),
    ), ("x", "one", "one"))))
    declaration = Function("leaf", (integer,), (integer, integer, integer))
    body = RegionFunctionBody(declaration, leaf)
    caller = Program(Region(Block((Value("seed", integer), Value("body0_one", integer)), (
        Call(("seed",), (Value("a", integer), Value("b", integer), Value("c", integer)), "leaf"),
        Call(("a",), (Value("d", integer), Value("e", integer), Value("f", integer)), "leaf"),
    ), ("a", "b", "c", "d", "e", "f", "body0_one"))), (declaration,))
    expanded = inline_function_bodies(caller, (body,))
    from pllm.execution.region_reference import interpret
    assert interpret(expanded, {"seed": 7, "body0_one": 99}).values == (7, 1, 1, 7, 1, 1, 99)
    invalid = replace(leaf, entry=Region(replace(leaf.entry.block, yields=("hidden",))))
    with pytest.raises(ValueError, match="captures must be explicit"):
        inline_region(invalid, (Value("seed", integer),), "body")


def test_data_ports_preserve_effects_and_reject_effectful_maps():
    from pllm.execution.region_compilation import validate_region_program
    from pllm.execution.region_reference import interpret
    from test.boundary.test_region_external import external_program

    original = external_program()
    assert original.entry.block.arguments == ()
    validate_region_program(original)
    events = []
    def call(inputs):
        events.append(len(events) + 1)
        return (events[-1],)
    result = interpret(original, {}, functions={f.name: call for f in original.functions})
    assert result.values == (1, 2) and events == [1, 2]
    assert [event.resource for event in result.effects] == ["first", "second"]

    body = Region(Block((Value("i", ScalarType.INDEX),),
        (Call((), (Value("x", ScalarType.INT),), original.functions[0].name),), ("x",)))
    forbidden = Program(Region(Block((),
        (Map((), (Value("out", BankType(ScalarType.INT, 2)),), 2, body),), ("out",))),
        original.functions, original.resources)
    with pytest.raises(ValueError, match="pure region"):
        validate_region_program(forbidden)


def test_decoder_imports_custom_readonly_modules_without_suffix_class_rules():
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationMode
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.models.qwen2 import Qwen2Attention, Qwen2DecoderLayer

    class ResidualSum(torch.nn.Module):
        def forward(self, hidden, residual):
            return hidden + residual, residual

    def make_layer(width):
        layer = Qwen2DecoderLayer.__new__(Qwen2DecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.self_attn = Qwen2Attention.__new__(Qwen2Attention)
        torch.nn.Module.__init__(layer.self_attn)
        layer.input_layernorm = RMSNorm(4)
        layer.post_attention_layernorm = ResidualSum()
        layer.mlp = torch.nn.Sequential(torch.nn.Linear(4, width, bias=False), torch.nn.SiLU(),
                                        torch.nn.Linear(width, 4, bias=False))
        return layer

    config = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE))
    with set_current_vllm_config(config), torch.inference_mode():
        # This checks capture and the real suffix, not attention execution.
        # Keep the original layer method and its unexecuted resource endpoint.
        layer = make_layer(8)
        tensor = TensorType((1, 4), "torch.float32")
        bodies = VllmDecoderBodies.inspect((layer,), tensor, CausalModelDomain(1, 1, 2), 2)
        assert bodies is not None
        bodies.validate((layer,))
        inputs = torch.randn(1, 4), torch.randn(1, 4)
        expected = bodies.graphs[0][0].run_suffix(inputs)
        actual = bodies.exports[0].program.module()(*inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert bodies.imports and bodies.suffix.program.functions

        # Two different intermediate shapes use their own original parameter
        # fields through the existing region executor and readonly owner.
        from pllm.contracts.region_imports import BorrowedTensorImport
        from pllm.execution.region_compilation import lower_region_bindings
        from pllm.execution.region_execution import RegionExecution
        from pllm.execution.region_resources import ReadonlyResourceBinding, RegionResources
        from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView
        from test.boundary.test_region_pages import owner_fixture

        class OutOfOrderMlp(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Registration order deliberately differs from computation.
                self.down = torch.nn.Linear(12, 4, bias=False)
                self.up = torch.nn.Linear(4, 12, bias=False)

            def forward(self, x):
                return self.down(torch.nn.functional.silu(self.up(x)))

        other = make_layer(12)
        other.mlp = OutOfOrderMlp()
        layers = layer, other
        varied = VllmDecoderBodies.inspect(layers, tensor, CausalModelDomain(1, 2, 2), 2)
        assert varied is not None
        varied.validate(layers)
        program = varied.suffix.program
        schedule = analyze_region_program(program, optimize=False).reference_schedule
        runner = RegionExecution(program, lower_region_schedule(program, schedule),
                                 lower_region_bindings(program, schedule), varied.functions)
        execution, _, rows, _ = owner_fixture(rounds=1)
        invocation = execution.start_step(rows)
        resources = {resource.name: resource for resource in program.resources}
        weight_cells = (*(ReadonlyTensorCell.bind((i,), tuple(item.parameters())) for i, item in enumerate(layers)),
                        ReadonlyTensorCell.bind((2,), ()))
        context_cells = tuple(ReadonlyTensorCell.bind((i,), ()) for i in range(2))
        loans = tuple(item for item in varied.imports if isinstance(item, BorrowedTensorImport))
        pages = RegionResources(program, invocation.context, (), borrows=(
            ReadonlyResourceBinding(ReadonlyTensorView(resources["weights"], weight_cells), invocation.context),
            ReadonlyResourceBinding(ReadonlyTensorView(resources["context"], context_cells), invocation.context)),
            tensor_borrows=loans)
        try:
            for i in range(len(layers)):
                expected = varied.graphs[i][0].run_suffix(inputs)
                actual = runner.run({"row": 0, "layer": i, "hidden": inputs[0], "residual": inputs[1]}, pages=pages).values
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            execution.cancel_step(invocation)
            execution.retire(tuple(row.identity for row in rows))

        # A newly introduced state update cannot inherit the readonly contract.
        class StatefulResidual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("counter", torch.zeros(1))

            def forward(self, hidden, residual):
                self.counter.add_(1)
                return hidden + residual + self.counter, residual

        layer.post_attention_layernorm = StatefulResidual()
        with pytest.raises(ValueError, match="original owner|update original buffers"):
            VllmDecoderBodies.inspect((layer,), tensor, CausalModelDomain(1, 1, 2), 2)


def test_model_source_exposes_kv_before_feed_forward_without_weakening_causality():
    tensor = TensorType((1, 4), "torch.float32")
    bodies = decoder_partition_bodies(tensor, CausalModelDomain(1, 1, 2), 2, "test-source-graph")
    body, _ = bodies
    function = body.declaration
    index = ScalarType.INDEX
    position = TensorType((1,), "torch.int64")
    source = Program(Region(Block((Value("x", tensor), Value("position", position)), (
        Constant((), (Value("zero", index),), 0), Constant((), (Value("one", index),), 1),
        Call(("zero", "zero", "zero", "zero", "position", "x"),
             (Value("h0", tensor), Value("r0", tensor)), function.name),
        Call(("one", "zero", "one", "zero", "position", "x"),
             (Value("h1", tensor), Value("r1", tensor)), function.name),
    ), ("h0", "r0", "h1", "r1"))), (function,), body.program.resources)
    expanded = inline_function_bodies(source, bodies)
    analysis = analyze_region_program(expanded, optimize=False)
    assert len(analysis.statements) == 4
    assert function.name not in {f.name for f in expanded.functions}
    # Attention of token 1 may use token 0's published KV before token 0's
    # feed-forward suffix. Reversing the attention pair remains illegal.
    lower_region_schedule(expanded, "{ S0[] -> [0]; S1[] -> [2]; S2[] -> [1]; S3[] -> [3] }")
    with pytest.raises(ValueError):
        lower_region_schedule(expanded, "{ S0[] -> [1]; S1[] -> [2]; S2[] -> [0]; S3[] -> [3] }")
    with pytest.raises(ValueError, match="original resource family"):
        bad = replace(body, program=replace(body.program, resources=decoder_partition_bodies(tensor, CausalModelDomain(1, 1, 3), 2, "test-source-graph")[0].program.resources))
        inline_function_bodies(source, (bad,))


def test_model_method_binding_rejects_hooks_and_same_shape_module_replacement():
    module = torch.nn.Linear(4, 4)
    binding = ModuleCallBinding.bind(module)
    binding.validate(module)
    # Resource field ordinals follow named_parameters order. Reordering two
    # original objects must not silently change the meaning of those ordinals.
    weight = module._parameters.pop("weight")
    module._parameters["weight"] = weight
    with pytest.raises(ValueError, match="parameter fields changed"):
        binding.validate(module)
    module._parameters = {"weight": weight, "bias": module.bias}
    binding.validate(module)
    with pytest.raises(ValueError, match="changed after"):
        binding.validate(torch.nn.Linear(4, 4))
    hook = module.register_forward_hook(lambda _module, _inputs, output: output + 1)
    try:
        with pytest.raises(ValueError, match="changed after"):
            binding.validate(module)
    finally:
        hook.remove()
    module.forward = lambda value: value
    with pytest.raises(ValueError, match="changed after"):
        binding.validate(module)
