"""Original Qwen Torch graphs retain their actual custom CUDA operator."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch.export.graph_signature import OutputKind
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.model_executor.models.qwen2 import Qwen2MLP

from pllm.backends.vllm.region_kernels import bind_vllm_tensor_kernel
from pllm.contracts.region_proposal import RegionScheduleProposal
from pllm.contracts.region_composition import inline_region
from pllm.contracts.region_program import Block, Region, Value
from pllm.contracts.region_selection import RegionPayloadLimit, RegionSelectionPolicy
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.exported_region import import_exported_region
from pllm.execution.region_compilation import analyze_region_program
from pllm.execution.region_execution import RegionExecutionPlan
from pllm.execution.region_native import FunctionalKernelPrimitive, NativePrimitive, source_region_functions
from pllm.execution.region_optimization import optimize_region
from pllm.execution.region_storage import analyze_region_payload, import_region_functions
from pllm.execution.region_streams import RegionStreamBindings


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
    def forbidden(partition):
        raise AssertionError("ordinary functional ATen must not need a native backend")
    imported = import_exported_region(exported, bind_kernel=forbidden)
    program, sources = imported.program, imported.functions
    assert tuple(spec.kind for spec in imported.signature.output_specs) == (
        OutputKind.BUFFER_MUTATION, OutputKind.USER_INPUT_MUTATION, OutputKind.USER_OUTPUT)
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
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_original_qwen_export_retains_swiglu_and_enters_measured_selection(dtype):
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
        assert tuple(imported.signature.inputs_to_parameters.values()) == tuple(dict(model.named_parameters()))
        # Two original calls share immutable weights and have independent input
        # versions. This supplies real scheduling freedom around the kernels.
        arguments = program.entry.block.arguments
        y_port = Value("y", next(value.type for value in arguments if value.name == "x"))
        first = inline_region(program, arguments, "first")
        second = inline_region(program, tuple(y_port if value.name == "x" else value for value in arguments), "second")
        program = replace(program, entry=Region(Block((*arguments, y_port), (*first.operations, *second.operations),
            tuple(value.name for value in (*first.results, *second.results)))))
        native, = (cell for cell in sources.values() if isinstance(cell, FunctionalKernelPrimitive))
        assert sum(isinstance(cell, ExportedCell) for cell in sources.values()) == 2
        assert native.implementation is not None and not native.declaration.resources
        # auto_functionalized_v2 allocates the result base. The original graph
        # keeps that allocation; no additional wrapper copies are required.
        assert sum(buffer.bytes for buffer in native.implementation.allocations) == 32 * x.element_size()
        imports = import_region_functions(program, sources)
        assert source_region_functions(program.functions, {}, imports) == {}
        schedule = analyze_region_program(program, optimize=False).reference_schedule
        payload = analyze_region_payload(program, schedule, imports)
        proposal = RegionScheduleProposal(payload)
        plan = RegionExecutionPlan(program, imports, proposal, proposal.peak_bytes)
        with pytest.raises(ValueError, match="complete original tensor imports"):
            plan.bind({})
        original = plan.bind(sources)
        y = x * 2
        inputs = {**dict(model.named_parameters()), "x": x, "y": y}
        before = {name: value.clone() for name, value in inputs.items()}
        expected = model(x), model(y)
        with patch("pllm.execution.region_native.copy_data", side_effect=AssertionError("redundant kernel wrapper copy")):
            for actual, reference in zip(original.run(inputs).values, expected, strict=True):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert all(device.call_scratch_bytes_upper_bound == 0 for device in payload.capacity.devices)
        alias = deepcopy(native.graph)
        output = next(node for node in alias.graph.nodes if node.op == "output")
        argument = next(node for node, type_ in zip(
            (node for node in alias.graph.nodes if node.op == "placeholder"), native.declaration.inputs, strict=True)
            if type_ == native.declaration.outputs[0])
        output.args = ((argument,),)
        alias.recompile()
        with pytest.raises(ValueError, match="own distinct complete backings"):
            FunctionalKernelPrimitive.bind(alias, native.implementation)
        uncloned = deepcopy(native.graph)
        wrapped = next(node for node in uncloned.graph.nodes if "auto_functionalized_v2" in str(node.target))
        wrapped.kwargs = {**wrapped.kwargs, "_only_clone_these_bases": ()}
        uncloned.recompile()
        with pytest.raises(ValueError, match="original clone-all policy"):
            FunctionalKernelPrimitive.bind(uncloned, native.implementation)
        with pytest.raises(ValueError, match="copying kernel ABI"):
            NativePrimitive(native.declaration, lambda values, resources: values, native.implementation)
        with pytest.raises(ValueError, match="include every owned result"):
            analyze_region_payload(program, schedule, tuple(
                replace(item, allocations=()) if item == native.implementation else item for item in imports))
        streams = RegionStreamBindings(torch.cuda.Stream(), (torch.cuda.Stream(),))
        streams.owner.wait_stream(torch.cuda.current_stream())
        # The test assesses source scheduling. Numerical rewrites have their
        # own fixed-tolerance tests; no tolerance changes hide a source mismatch.
        selected = optimize_region(program, sources, inputs, payload_limits=(RegionPayloadLimit("cuda:0", 1 << 20),),
            options=replace(RegionSelectionPolicy(allow_numerical_transforms=False), tuning="measured"), streams=streams,
            warmup_rounds=1, measurement_rounds=1, max_candidates=2)
        assert len(selected.request.candidates) == 2
        assert all(candidate.elapsed_ns for candidate in selected.request.candidates)
        result = selected.execution.run(inputs, streams=streams).values
        streams.owner.synchronize()
        for actual, reference in zip(result, expected, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        for name, value in inputs.items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        # The same graph accepts new explicit parameter values; it never
        # captures the model's initial weights in an executable closure.
        changed = {name: value * 0.5 for name, value in model.named_parameters()}
        expected_changed = tuple(torch.func.functional_call(model, changed, (value,)) for value in (x, y))
        for actual, reference in zip(original.run({**changed, "x": x, "y": y}).values, expected_changed, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        with pytest.raises(ValueError, match="audited pure kernels"):
            optimize_region(program, {**sources, native.declaration.name: NativePrimitive(native.declaration, lambda values, resources: values)}, inputs,
                payload_limits=proposal.peak_bytes, max_candidates=1, options=RegionSelectionPolicy(tuning="measured"))
