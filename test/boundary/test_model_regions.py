"""Model regions: decoder bodies, replay control and real vLLM kernels as region imports.

The backend exposes decoder-layer dataflow, KV and weights as region functions
and resources. These tests check that composition keeps causality and readonly
ownership, that replay control matches the original loop, and that actual vLLM
kernels and attention metadata bind to the original resources.
"""

from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from vllm import ir
from vllm.config import AttentionConfig, CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.forward_context import ForwardContext
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.qwen2 import Qwen2Attention, Qwen2DecoderLayer, Qwen2MLP
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionMetadata

from pllm.backends.vllm.region_attention import (
    attention_paged_access,
    select_attention_metadata,
    validate_attention_resources,
)
from pllm.backends.vllm.region_kernels import VllmRmsNormKernel, bind_vllm_tensor_kernel
from pllm.backends.vllm.region_layers import VllmDecoderBodies, decoder_partition_bodies
from pllm.contracts.backend_attachment import (
    CausalModelDomain,
)
from pllm.contracts.region_composition import RegionFunctionBody, inline_function_bodies, inline_region
from pllm.contracts.region_imports import BorrowedTensorImport
from pllm.contracts.region_program import (
    AffineIndex,
    Block,
    BorrowedResource,
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
from pllm.contracts.region_proposal import RegionScheduleProposal
from pllm.contracts.region_selection import RegionPayloadLimit, RegionSelectionPolicy
from pllm.contracts.semantic_graph import ConditionalLayerReplay
from pllm.execution.decoder_region import DecoderRegion
from pllm.execution.exported_region import import_exported_region
from pllm.execution.module_graph import ModuleCallBinding
from pllm.execution.paged_access import PagedLayer, PagedResourceView, bind_paged_access
from pllm.execution.region_compilation import (
    analyze_region_program,
    lower_region_bindings,
    lower_region_schedule,
)
from pllm.execution.region_execution import RegionExecution, RegionExecutionPlan
from pllm.execution.region_native import external_primitive
from pllm.execution.region_optimization import optimize_region
from pllm.execution.region_replay import ConditionalReplayRegion
from pllm.execution.region_resources import (
    ExternalFunction,
    HostResourceBinding,
    PagedResourceBinding,
    ReadonlyResourceBinding,
    RegionResources,
)
from pllm.execution.region_storage import analyze_region_payload, import_region_functions
from pllm.execution.region_streams import RegionStreamBindings, lower_region_streams
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView
from test.region_fixtures import limits, owner_fixture, paged_rows, region, stream_execution
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


def test_model_method_binding_rejects_hooks_and_same_shape_module_replacement():
    module = torch.nn.Linear(4, 4)
    binding = ModuleCallBinding.bind(module)
    binding.validate(module)
    with pytest.raises(ValueError):
        binding.validate(torch.nn.Linear(4, 4))
    hook = module.register_forward_hook(lambda _module, _inputs, output: output + 1)
    with pytest.raises(ValueError):
        binding.validate(module)
    hook.remove()
    binding.validate(module)


@pytest.mark.parametrize(
    "limit,mode,cuda", ((0, "same", False), (6, "converge", True), (4, "converge", False), (4, "nonfinite", True))
)
def test_replay_control_matches_the_plain_loop(cuda, limit, mode):
    if cuda and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    stream = torch.cuda.Stream() if cuda else None
    execution, _, rows, _ = owner_fixture(rounds=1, stream=stream)
    invocation = execution.start_step(rows, stream)
    initial = torch.ones((1, 2), device=execution.arena.device)
    effect = ConditionalLayerReplay((0,), max_iterations=limit, relative_l2_ppm=100_000)
    replay = ConditionalReplayRegion(TensorType((1, 2), "torch.float32", str(initial.device)), effect)
    observed = []

    def step(carry):
        if stream is not None:
            assert torch.cuda.current_stream(initial.device) == stream
        return carry * float("inf") if mode == "nonfinite" else carry * 0.5 + 0.25

    with torch.cuda.stream(stream) if stream is not None else torch.no_grad():
        run = lambda: replay.run(  # noqa: E731
            invocation.context,
            initial,
            step=step,
            observe=lambda value, i: observed.append(value.clone()),
            record=lambda i, relative: None,
        )
        if mode == "nonfinite":
            with pytest.raises(RuntimeError):
                run()
        else:
            run()
    if stream is not None:
        torch.cuda.current_stream().wait_stream(stream)
    # The plain loop: stop at the limit or once the relative change is at most 10%.
    expected, carry = [], initial
    for _ in range(0 if mode == "nonfinite" else limit):
        result = carry * 0.5 + 0.25
        expected.append(result)
        if len(expected) > 1 and float((result - carry).norm() / carry.norm()) <= 0.1:
            break
        carry = result
    if mode != "nonfinite":
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    execution.cancel_step(invocation)
    execution.retire(tuple(row.identity for row in rows))


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


def test_import_retains_torch_state_update_signature_without_implicit_publication():
    class Stateful(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("counter", torch.arange(4, dtype=torch.float32))

        def forward(self, x):
            self.counter.add_(1)
            x.mul_(2)
            return x + self.counter

    module = Stateful()
    x = torch.arange(4, dtype=torch.float32)
    old = module.counter.clone(), x.clone()
    exported = torch.export.export(module, (x,))

    imported = import_exported_region(exported, bind_kernel=lambda partition: None)
    program, sources = imported.program, imported.functions
    schedule = analyze_region_program(program, optimize=False).reference_schedule
    imports = import_region_functions(program, sources)
    proposal = RegionScheduleProposal(analyze_region_payload(program, schedule, imports))
    execution = RegionExecutionPlan(program, imports, proposal, proposal.peak_bytes).bind(sources)
    updated_buffer, updated_input, result = execution.run({"counter": module.counter, "x": x}).values
    torch.testing.assert_close(updated_buffer, old[0] + 1, rtol=0, atol=0)
    torch.testing.assert_close(updated_input, old[1] * 2, rtol=0, atol=0)
    torch.testing.assert_close(result, old[1] * 2 + old[0] + 1, rtol=0, atol=0)
    torch.testing.assert_close(module.counter, old[0], rtol=0, atol=0)
    torch.testing.assert_close(x, old[1], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_original_qwen_export_retains_swiglu_and_enters_measured_selection():
    dtype = torch.bfloat16
    config = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE, custom_ops=["all"]))
    with set_current_vllm_config(config), ExitStack() as scope, torch.no_grad():
        # Only construction-time TP metadata is supplied locally. Forward,
        # export, custom CUDA dispatch, scheduling and measurement are real.
        for module in ("vllm.model_executor.layers.linear", "vllm.model_executor.parameter"):
            for field, value in (("rank", 0), ("world_size", 1)):
                scope.enter_context(patch(module + ".get_tensor_model_parallel_" + field, return_value=value))
        model = Qwen2MLP(16, 32, "silu").to(device="cuda:0", dtype=dtype)
        for weight in model.parameters():
            weight.normal_(std=0.05)
        x = torch.randn((1, 16), device="cuda:0", dtype=dtype)
        exported = torch.export.export(model, (x,))
        imported = import_exported_region(exported, bind_kernel=bind_vllm_tensor_kernel)
        program, sources = imported.program, imported.functions
        # Two original calls share immutable weights and have independent input
        # versions. This supplies real scheduling freedom around the kernels.
        arguments = program.entry.block.arguments
        y_port = Value("y", next(value.type for value in arguments if value.name == "x"))
        first = inline_region(program, arguments, "first")
        second = inline_region(program, tuple(y_port if value.name == "x" else value for value in arguments), "second")
        program = replace(
            program,
            entry=Region(
                Block(
                    (*arguments, y_port),
                    (*first.operations, *second.operations),
                    tuple(value.name for value in (*first.results, *second.results)),
                )
            ),
        )
        imports = import_region_functions(program, sources)
        schedule = analyze_region_program(program, optimize=False).reference_schedule
        payload = analyze_region_payload(program, schedule, imports)
        proposal = RegionScheduleProposal(payload)
        original = RegionExecutionPlan(program, imports, proposal, proposal.peak_bytes).bind(sources)
        y = x * 2
        inputs = {**dict(model.named_parameters()), "x": x, "y": y}
        before = {name: value.clone() for name, value in inputs.items()}
        expected = model(x), model(y)
        for actual, reference in zip(original.run(inputs).values, expected, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        streams = RegionStreamBindings(torch.cuda.Stream(), (torch.cuda.Stream(),))
        streams.owner.wait_stream(torch.cuda.current_stream())
        # The test assesses source scheduling. Numerical rewrites have their
        # own fixed-tolerance tests; no tolerance changes hide a source mismatch.
        selected = optimize_region(
            program,
            sources,
            inputs,
            payload_limits=(RegionPayloadLimit("cuda:0", 1 << 20),),
            options=replace(RegionSelectionPolicy(allow_numerical_transforms=False), tuning="measured"),
            streams=streams,
            warmup_rounds=1,
            measurement_rounds=1,
            max_candidates=2,
        )
        assert all(candidate.elapsed_ns for candidate in selected.request.candidates)
        result = selected.execution.run(inputs, streams=streams).values
        streams.owner.synchronize()
        for actual, reference in zip(result, expected, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        for name, value in inputs.items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_model_norm_contract_binds_actual_kernel_on_both_original_streams():
    dtype = torch.bfloat16
    config = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE, custom_ops=["all"]))
    with set_current_vllm_config(config), ir.ops.fused_add_rms_norm.set_priority(["vllm_c", "native"]):
        norm = RMSNorm(16, dtype=dtype).to("cuda:0")
        tensor = TensorType((1, 16), str(dtype), "cuda:0")
        kernel = VllmRmsNormKernel.inspect(norm, tensor, 3)
        assert kernel is not None
        function = kernel.implementation.declaration
        context, weights = BorrowedResource("context", (2,)), BorrowedResource("weights", (4,))
        index = ScalarType.INDEX
        program = Program(
            region(
                (Value("h", tensor), Value("r", tensor)),
                (
                    Coordinate((), (Value("zero", index),), AffineIndex(0)),
                    Coordinate((), (Value("one", index),), AffineIndex(1)),
                    Call(("zero", "h", "r"), (Value("a", tensor),), function.name),
                    Call(("one", "h", "r"), (Value("b", tensor),), function.name),
                ),
                ("a", "b"),
            ),
            (function,),
            (context, weights),
        )
        cells = {function.name: external_primitive(function, kernel.implementation)}
        plan = lower_region_streams(program, "{ S0[] -> [0]; S1[] -> [1] }", cells, axis=0, lanes=2, pipeline_depth=2)
        runner = stream_execution(program, cells, plan, payload_limits=limits(plan))
        streams = RegionStreamBindings(torch.cuda.Stream(), (torch.cuda.Stream(),))
        execution, _, rows, _ = owner_fixture(rounds=1, stream=streams.owner)
        invocation = execution.start_step(rows, streams.owner)
        view = ReadonlyTensorView(
            weights, tuple(ReadonlyTensorCell.bind((i,), (norm.weight,) if i == 3 else ()) for i in range(4))
        )

        class Context:
            resource = context
            namespace = id(invocation)

            def validate(self):
                invocation.context.check_page_execution()

            def check(self, mode, ranges):
                self.validate()
                assert ranges in (((0, 1),), ((1, 2),))

        launches = []

        def normalize(inputs):
            _, h, r = inputs
            launches.append(torch.cuda.current_stream().cuda_stream)
            return (kernel.run(h, r),)

        def resources(abi):
            return RegionResources(
                program,
                invocation.context,
                (),
                (ExternalFunction(function, normalize, abi),),
                borrows=(
                    ReadonlyResourceBinding(view, invocation.context),
                    HostResourceBinding(Context(), invocation.context),
                ),
            )

        h, r = torch.randn((1, 16), dtype=dtype, device="cuda:0"), torch.randn((1, 16), dtype=dtype, device="cuda:0")
        expected, _ = norm.forward(h.clone(), r.clone())
        original = h.clone(), r.clone()
        streams.owner.wait_stream(torch.cuda.current_stream())
        try:
            result = runner.run({"h": h, "r": r}, pages=resources(kernel.implementation), streams=streams)
            assert set(launches) == {streams.owner.cuda_stream, streams.auxiliary[0].cuda_stream}
            for value in result.values:
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
                assert value.data_ptr() not in (h.data_ptr(), r.data_ptr())
            torch.testing.assert_close(h, original[0], rtol=0, atol=0)
            torch.testing.assert_close(r, original[1], rtol=0, atol=0)
        finally:
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
