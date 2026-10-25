"""Model regions: decoder bodies, replay control and vLLM attention metadata as region imports.

The backend exposes decoder-layer dataflow, KV and weights as region functions
and resources. These tests check that composition keeps causality and readonly
ownership, that replay control matches the original loop, and that attention
metadata binds to the original resources. Real vLLM kernels inside a model are
covered by test/gpu/vllm_recirculation_smoke.py.
"""

from dataclasses import replace

import pytest
import torch
from vllm.config import AttentionConfig, CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.forward_context import ForwardContext
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.qwen2 import Qwen2Attention, Qwen2DecoderLayer
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionMetadata

from pllm.backends.vllm.region_attention import (
    attention_paged_access,
    select_attention_metadata,
    validate_attention_resources,
)
from pllm.backends.vllm.region_layers import VllmDecoderBodies, decoder_partition_bodies
from pllm.contracts.backend_attachment import (
    CausalModelDomain,
)
from pllm.contracts.region_composition import RegionFunctionBody, inline_function_bodies, inline_region
from pllm.contracts.region_imports import BorrowedTensorImport
from pllm.contracts.region_program import (
    AffineIndex,
    Block,
    Call,
    Constant,
    Coordinate,
    Function,
    Program,
    Region,
    ScalarType,
    TensorType,
    Value,
)
from pllm.execution.decoder_region import DecoderRegion
from pllm.execution.paged_access import PagedLayer, PagedResourceView, bind_paged_access
from pllm.execution.region_compilation import (
    analyze_region_program,
    lower_region_bindings,
    lower_region_schedule,
)
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_resources import (
    PagedResourceBinding,
    ReadonlyResourceBinding,
    RegionResources,
)
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView
from test.region_fixtures import owner_fixture, paged_rows
from test.region_reference import interpret


def test_inlining_preserves_argument_aliases_duplicate_returns_and_fresh_names():
    integer = ScalarType.INT
    leaf = Program(
        Region(Block((Value("x", integer),), (Constant((), (Value("one", integer),), 1),), ("x", "one", "one")))
    )
    declaration = Function("leaf", (integer,), (integer, integer, integer))
    body = RegionFunctionBody(declaration, leaf)
    caller = Program(
        Region(
            Block(
                (Value("seed", integer), Value("body0_one", integer)),
                (
                    Call(("seed",), (Value("a", integer), Value("b", integer), Value("c", integer)), "leaf"),
                    Call(("a",), (Value("d", integer), Value("e", integer), Value("f", integer)), "leaf"),
                ),
                ("a", "b", "c", "d", "e", "f", "body0_one"),
            )
        ),
        (declaration,),
    )
    expanded = inline_function_bodies(caller, (body,))
    assert interpret(expanded, {"seed": 7, "body0_one": 99}).values == (7, 1, 1, 7, 1, 1, 99)
    # A body must match its declaration and may not capture undeclared values.
    with pytest.raises(ValueError):
        RegionFunctionBody(replace(declaration, outputs=(ScalarType.BOOL,)), leaf).validate()
    invalid = replace(leaf, entry=Region(replace(leaf.entry.block, yields=("hidden",))))
    with pytest.raises(ValueError):
        inline_region(invalid, (Value("seed", integer),), "body")


def test_decoder_imports_custom_readonly_modules_and_runs_them_on_borrowed_weights():
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
        layer.mlp = torch.nn.Sequential(
            torch.nn.Linear(4, width, bias=False), torch.nn.SiLU(), torch.nn.Linear(width, 4, bias=False)
        )
        return layer

    config = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE))
    with set_current_vllm_config(config), torch.inference_mode():
        # Capture and the feed-forward suffix only, not attention execution. Two different
        # intermediate shapes use their own parameter fields through the readonly owner.
        layer = make_layer(8)
        tensor = TensorType((1, 4), "torch.float32")
        inputs = torch.randn(1, 4), torch.randn(1, 4)

        layers = layer, make_layer(12)
        varied = VllmDecoderBodies.inspect(layers, tensor, CausalModelDomain(1, 2, 2), 2)
        assert varied is not None
        varied.validate(layers)
        program = varied.suffix.program
        schedule = analyze_region_program(program, optimize=False).reference_schedule
        runner = RegionExecution(
            program,
            lower_region_schedule(program, schedule),
            lower_region_bindings(program, schedule),
            varied.functions,
        )
        execution, _, rows, _ = owner_fixture(rounds=1)
        invocation = execution.start_step(rows)
        resources = {resource.name: resource for resource in program.resources}
        weight_cells = (
            *(ReadonlyTensorCell.bind((i,), tuple(item.parameters())) for i, item in enumerate(layers)),
            ReadonlyTensorCell.bind((2,), ()),
        )
        context_cells = tuple(ReadonlyTensorCell.bind((i,), ()) for i in range(2))
        loans = tuple(item for item in varied.imports if isinstance(item, BorrowedTensorImport))
        pages = RegionResources(
            program,
            invocation.context,
            (),
            borrows=(
                ReadonlyResourceBinding(ReadonlyTensorView(resources["weights"], weight_cells), invocation.context),
                ReadonlyResourceBinding(ReadonlyTensorView(resources["context"], context_cells), invocation.context),
            ),
            tensor_borrows=loans,
        )
        try:
            for i, item in enumerate(layers):
                actual = runner.run(
                    {"row": 0, "layer": i, "hidden": inputs[0], "residual": inputs[1]}, pages=pages
                ).values
                expected = item.mlp(inputs[0] + inputs[1]), inputs[1]
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
        with pytest.raises(ValueError):
            VllmDecoderBodies.inspect((layer,), tensor, CausalModelDomain(1, 1, 2), 2)


def test_model_source_exposes_kv_before_feed_forward_without_weakening_causality():
    tensor = TensorType((1, 4), "torch.float32")
    bodies = decoder_partition_bodies(tensor, CausalModelDomain(1, 1, 2), 2, "test-source-graph")
    body, _ = bodies
    function = body.declaration
    index = ScalarType.INDEX
    position = TensorType((1,), "torch.int64")
    source = Program(
        Region(
            Block(
                (Value("x", tensor), Value("position", position)),
                (
                    Coordinate((), (Value("zero", index),), AffineIndex(0)),
                    Coordinate((), (Value("one", index),), AffineIndex(1)),
                    Call(
                        ("zero", "zero", "zero", "zero", "position", "x"),
                        (Value("h0", tensor), Value("r0", tensor)),
                        function.name,
                    ),
                    Call(
                        ("one", "zero", "one", "zero", "position", "x"),
                        (Value("h1", tensor), Value("r1", tensor)),
                        function.name,
                    ),
                ),
                ("h0", "r0", "h1", "r1"),
            )
        ),
        (function,),
        body.program.resources,
    )
    expanded = inline_function_bodies(source, bodies)
    # Attention of token 1 may use token 0's published KV before token 0's
    # feed-forward suffix. Reversing the attention pair remains illegal.
    lower_region_schedule(expanded, "{ S0[] -> [0]; S1[] -> [2]; S2[] -> [1]; S3[] -> [3] }")
    with pytest.raises(ValueError):
        lower_region_schedule(expanded, "{ S0[] -> [1]; S1[] -> [2]; S2[] -> [0]; S3[] -> [3] }")


def test_decoder_binds_request_and_token_for_symbolic_kv_access():
    layers, taps = (1, 2), (2, 1)
    execution, _, rows, refs = owner_fixture(rounds=1)
    invocation = execution.start_step(rows)
    decoder = DecoderRegion(TensorType((1, 4), "torch.float32"), layers, CausalModelDomain(2, 3, 8), taps=taps)

    def layer(ordinal, positions, hidden, residual):
        return hidden + ordinal + positions, hidden.clone()

    hidden = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    for request, token in ((0, 0), (1, 7), (0, 3)):
        row = replace(rows[0], request_ordinal=request, token_offset=token)
        caches = tuple(
            PagedLayer(str(i), torch.zeros(2, 1, 8, 4), torch.tensor([[0], [1]]), torch.tensor([request * 8 + token]))
            for i in layers
        )
        view = PagedResourceView(
            decoder.source.kv, bind_paged_access((row,), caches), tuple((i, str(i)) for i in layers)
        )
        kv = PagedResourceBinding(view, invocation.context)
        result, residual, *observed = decoder.run(invocation.context, row, torch.tensor([token]), hidden, layer, kv=kv)
        torch.testing.assert_close(result, hidden + sum(layers) + len(layers) * token, rtol=0, atol=0)
        torch.testing.assert_close(residual, hidden + sum(layers[:-1]) + (len(layers) - 1) * token, rtol=0, atol=0)
        for tap, actual in zip(taps, observed, strict=True):
            preceding = layers[: layers.index(tap)]
            torch.testing.assert_close(
                actual, 2 * (hidden + sum(preceding) + len(preceding) * token) + tap + token, rtol=0, atol=0
            )
    with pytest.raises(ValueError):  # Token offset outside the causal domain.
        decoder.run(invocation.context, replace(rows[0], token_offset=8), torch.tensor([0]), hidden, layer, kv=kv)
    assert tuple(execution.arena.latest(ref.lineage_id) for ref in refs) == refs
    execution.cancel_step(invocation)
    execution.retire(tuple(row.identity for row in rows))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="vLLM FlashAttention backend requires CUDA")
def test_original_attention_resource_binds_without_a_model():
    config = VllmConfig(attention_config=AttentionConfig(flash_attn_version=2))
    with set_current_vllm_config(config):
        attention = Attention(
            2, 64, 0.125, num_kv_heads=1, prefix="independent.attention", attn_backend=FlashAttentionBackend
        )
    # Resource-only check: real backend object, CPU storage, no attention forward.
    attention.kv_cache = torch.zeros(4, 1, 16, 128)
    validate_attention_resources((attention,), hidden_size=128)
    slots = torch.tensor([16, 17, 32, 33])
    metadata = FlashAttentionMetadata(
        4,
        2,
        torch.tensor([0, 2, 4], dtype=torch.int32),
        18,
        torch.tensor([18, 18], dtype=torch.int32),
        torch.tensor([[0, 1], [0, 2]]),
        slots,
        False,
        0,
        None,
        None,
        None,
    )
    name = attention.layer_name
    context = ForwardContext({name: attention}, {name: metadata}, {name: slots})
    batch = paged_rows((16, 17, 16, 17))
    access = attention_paged_access(context, (attention,), batch)
    selected, cache, block, offset = select_attention_metadata(
        context, attention, batch[2], 17, torch.tensor([0, 1, 17], dtype=torch.int32), access
    )
    assert cache is attention.kv_cache and (block, offset) == (2, 0)
    torch.testing.assert_close(selected.slot_mapping, slots[2:3])
    torch.testing.assert_close(selected.block_table, metadata.block_table[1:])
    assert selected.seq_lens.tolist() == [17] and selected.num_actual_tokens == 1
    assert context.attn_metadata[name] is metadata and metadata.num_actual_tokens == 4
    with pytest.raises(ValueError):  # Non-causal attention cannot expose ordinary paged KV.
        attention_paged_access(
            replace(context, attn_metadata={name: replace(metadata, causal=False)}), (attention,), batch
        )
