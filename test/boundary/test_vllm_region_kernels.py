"""The actual vLLM normalization kernel uses the original resource bindings."""

from dataclasses import replace

import pytest
import torch
from vllm import ir
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.model_executor.layers.layernorm import RMSNorm

from pllm.backends.vllm.region_kernels import VllmRmsNormKernel
from pllm.contracts.backend_attachment import AttachmentGraphMode, AttachmentTopology, BackendPlacementProbe, CausalModelDomain
from pllm.contracts.region_imports import NativeKernelImport
from pllm.contracts.region_program import BorrowedResource, Call, Constant, Program, ScalarType, TensorType, Value
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.region_execution import RegionStreamExecution
from pllm.execution.region_native import external_primitive
from pllm.execution.region_resources import ExternalFunction, HostResourceBinding, ReadonlyResourceBinding, RegionResources
from pllm.execution.region_streams import RegionStreamBindings, lower_region_streams
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView
from test.boundary.test_region_pages import owner_fixture
from test.boundary.test_region_program import region
from test.boundary.test_region_streams import limits


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_model_norm_contract_binds_actual_kernel_on_both_original_streams(dtype):
    config = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE, custom_ops=["all"]))
    with set_current_vllm_config(config), ir.ops.fused_add_rms_norm.set_priority(["vllm_c", "native"]):
        norm = RMSNorm(16, dtype=dtype).to("cuda:0")
        tensor = TensorType((1, 16), str(dtype), "cuda:0")
        kernel = VllmRmsNormKernel.inspect(norm, tensor, 3)
        assert kernel is not None and kernel.implementation.allocations == ()
        assert decode_boundary(encode_boundary(kernel.implementation), NativeKernelImport) == kernel.implementation
        probe = BackendPlacementProbe("test.norm@1", "a" * 64, "b" * 64, "c" * 64,
            AttachmentTopology.IN_PROCESS, AttachmentGraphMode.EAGER, StateDeviceKind.CUDA, 0, 1, 1,
            TensorDType.FLOAT32 if dtype is torch.float32 else TensorDType.BFLOAT16, 16, 2, (), (),
            causal_domain=CausalModelDomain(1, 3, 2), region_imports=(kernel.implementation,))
        assert decode_boundary(encode_boundary(probe), BackendPlacementProbe) == probe
        with ir.ops.fused_add_rms_norm.set_priority(["native"]):
            assert VllmRmsNormKernel.inspect(norm, tensor, 3) is None
        function = kernel.implementation.declaration
        context, weights = BorrowedResource("context", (2,)), BorrowedResource("weights", (4,))
        index = ScalarType.INDEX
        program = Program(region((Value("h", tensor), Value("r", tensor)), (
            Constant((), (Value("zero", index),), 0), Constant((), (Value("one", index),), 1),
            Call(("zero", "h", "r"), (Value("a", tensor),), function.name),
            Call(("one", "h", "r"), (Value("b", tensor),), function.name),
        ), ("a", "b")), (function,), (context, weights))
        cells = {function.name: external_primitive(function, kernel.implementation)}
        plan = lower_region_streams(program, "{ S0[] -> [0]; S1[] -> [1] }", cells, axis=0, lanes=2, pipeline_depth=2)
        assert not plan.host_fences
        # Only the two mutable wrapper inputs need private call scratch.
        assert plan.payload.capacity.devices[0].call_scratch_bytes_upper_bound == 2 * 16 * norm.weight.element_size()
        runner = RegionStreamExecution(program, cells, plan, payload_limits=limits(plan))
        streams = RegionStreamBindings(torch.cuda.Stream(), (torch.cuda.Stream(),))
        execution, _, rows, _ = owner_fixture(rounds=1, stream=streams.owner)
        invocation = execution.start_step(rows, streams.owner)
        view = ReadonlyTensorView(weights, tuple(ReadonlyTensorCell.bind((i,), (norm.weight,) if i == 3 else ()) for i in range(4)))
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
            return RegionResources(program, invocation.context, (), (ExternalFunction(function, normalize, abi),), borrows=(
                ReadonlyResourceBinding(view, invocation.context), HostResourceBinding(Context(), invocation.context)))
        h, r = torch.randn((1, 16), dtype=dtype, device="cuda:0"), torch.randn((1, 16), dtype=dtype, device="cuda:0")
        expected, _ = norm.forward(h.clone(), r.clone())
        original = h.clone(), r.clone()
        streams.owner.wait_stream(torch.cuda.current_stream())
        try:
            with pytest.raises(ValueError, match="execution contract"):
                runner.run({"h": h, "r": r}, pages=resources(replace(kernel.implementation, implementation="different")), streams=streams)
            assert not launches
            result = runner.run({"h": h, "r": r}, pages=resources(kernel.implementation), streams=streams)
            assert set(launches) == {streams.owner.cuda_stream, streams.auxiliary[0].cuda_stream}
            for value in result.values:
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
                assert value.data_ptr() not in (h.data_ptr(), r.data_ptr())
            torch.testing.assert_close(h, original[0], rtol=0, atol=0)
            torch.testing.assert_close(r, original[1], rtol=0, atol=0)
            # An equal-layout replacement is still outside the borrowed weight
            # binding, whose original tensor may remain entirely unchanged.
            weight = norm.weight
            norm.weight = torch.nn.Parameter(weight.detach().clone())
            with pytest.raises(ValueError, match="changed after"):
                kernel.run(h, r)
            norm.weight = weight
            norm.variance_epsilon *= 2
            with pytest.raises(ValueError, match="changed after"):
                kernel.run(h, r)
        finally:
            execution.cancel_step(invocation)
            execution.retire(tuple(row.identity for row in rows))
