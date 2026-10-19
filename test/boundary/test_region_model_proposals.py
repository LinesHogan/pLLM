"""Model schedule proposals survive driver-only planning, tuning and installation on the original owner."""

import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from pllm.authoring import BackendCompileFacts, TensorStepInput
from pllm.config import ExecutionOptions
from pllm.contracts.backend_attachment import (
    AttachmentGraphMode,
    AttachmentTopology,
    BackendPlacementProbe,
    CausalModelDomain,
    SignedBackendAttachment,
)
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.region_composition import RegionFunctionBody, inline_region
from pllm.contracts.region_program import (
    AffineIndex,
    Block,
    BorrowedResource,
    Coordinate,
    Program,
    Region,
    ScalarType,
    TensorType,
    Value,
)
from pllm.contracts.region_proposal import REGION_STREAMS_CAPABILITY, RegionScheduleProposal
from pllm.contracts.region_selection import RegionSelectionPolicy
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import AnchorId
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.decoder_region import decoder_normalize_function
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.exported_resources import borrow_exported_inputs
from pllm.execution.paged_access import PagedLayer, PagedResourceView, bind_paged_access
from pllm.execution.region_resources import HostResourceBinding, PagedResourceBinding, ReadonlyResourceBinding
from pllm.execution.region_storage import import_region_functions
from pllm.execution.region_streams import RegionStreamBindings
from pllm.execution.replay import TOKENWISE_REPLAY_IMPLEMENTATION, TokenwiseReplayPrimitive, TokenwiseReplayTask
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView
from pllm.execution.tokenwise_region import compile_tokenwise_region
from pllm.execution.tokenwise_tuning import TokenwiseTuning
from pllm.workloads.recirculation import RecirculationWorkload


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tuning_installs_a_parallel_model_proposal_with_the_original_owner():
    device = torch.device("cuda:0")
    tensor = TensorType((1, 4), "torch.float32", "cuda:0")

    class OriginalNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.arange(4, device=device, dtype=torch.float32) + 1)

        def forward(self, hidden, residual):
            return (hidden + residual) * self.scale

    norm = OriginalNorm()
    original_scale = norm.scale.detach().clone()
    examples = (torch.zeros((1, 4), device=device), torch.zeros((1, 4), device=device))
    graph = ExportedCell.from_export(torch.export.export(norm, examples))
    weights, context = BorrowedResource("weights", (4,)), BorrowedResource("context", (4,))
    borrowed, local_sources = borrow_exported_inputs(graph, weights, {"scale": 0})
    h, r = Value("hidden", tensor), Value("residual", tensor)
    layer = Value("layer", ScalarType.INDEX)
    inlined = inline_region(borrowed.program, (h, r, layer), "norm")
    body = RegionFunctionBody(
        decoder_normalize_function(tensor, 3),
        Program(
            Region(
                Block(
                    (Value("row", ScalarType.INDEX), h, r),
                    (Coordinate((), (layer,), AffineIndex(3)), *inlined.operations),
                    (inlined.results[0].name,),
                )
            ),
            borrowed.program.functions,
            (context, weights),
        ),
    )
    body.validate()
    imports = import_region_functions(borrowed.program, local_sources)
    capabilities = (REGION_STREAMS_CAPABILITY, "model/tokenwise_replay@1")
    probe = BackendPlacementProbe(
        "test.model.proposals@1",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        AttachmentTopology.IN_PROCESS,
        AttachmentGraphMode.EAGER,
        StateDeviceKind.CUDA,
        0,
        1,
        1,
        TensorDType.FLOAT32,
        4,
        4,
        (),
        capabilities,
        causal_domain=CausalModelDomain(1, 3, 4),
        region_imports=imports,
        region_bodies=(body,),
    )
    facts = BackendCompileFacts(probe, RuntimeFacts(device).timeline(capabilities), (), 1, 0, 0, layer_count=3)
    workload = RecirculationWorkload()
    # The measured tuning protocol below is the explicit opt-in: specialize so
    # the arena reservation funds every measured candidate, not the reference only.
    facts = workload.specialize(facts, ExecutionOptions(tuning="measured"))
    # The driver receives only the signed-source descriptors. It has neither
    # OriginalNorm's class nor its graph/parameter objects, and never owns CUDA.
    script = """
import sys
import torch
from pllm.authoring import BackendCompileFacts
from pllm.config import ExecutionOptions
from pllm.contracts.backend_attachment import BackendPlacementProbe
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.workloads.recirculation import RecirculationWorkload
probe = decode_boundary(sys.stdin.read(), BackendPlacementProbe)
facts = BackendCompileFacts(probe, RuntimeFacts(torch.device('cuda:0')).timeline(probe.capabilities), (), 1, 0, 0, layer_count=3)
workload = RecirculationWorkload()
facts = workload.specialize(facts, ExecutionOptions(tuning="measured"))
planned = workload.plan_backend(facts)
assert not torch.cuda.is_initialized()
print(encode_boundary(planned))
"""
    planned = subprocess.run(
        [sys.executable, "-c", script],
        input=encode_boundary(probe),
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    deployment = workload.bind_backend(facts, decode_boundary(planned.stdout, SignedBackendAttachment))
    execution = deployment.execution
    task = TokenwiseReplayTask.from_implementation(execution.implementation(TOKENWISE_REPLAY_IMPLEMENTATION))
    execution.rebind_implementation(replace(task, region=task.region.bind_functions({graph.digest: graph})))
    source = compile_tokenwise_region(
        tensor,
        4,
        probe.causal_domain,
        workload.config.effect(3),
        RegionSelectionPolicy(tuning="measured"),
        stream_lanes=2,
        source_imports=imports,
        function_bodies=(body,),
    )
    parallel = next(
        proposal
        for proposal in source.candidate_proposals
        if proposal.stream_plan is not None and proposal.stream_plan.pipeline_depth == 2
    )
    proposals = (RegionScheduleProposal(source.payload), parallel)
    rng = torch.random.get_rng_state().clone()
    owner = torch.cuda.Stream(device=device)
    streams = RegionStreamBindings(owner, (torch.cuda.Stream(device=device),))
    identity = RequestIdentity("private", 0, 0)
    rows = PackedPhaseRowBatch(1, 1, 3, tuple(PackedPhaseRow(i, 0, i, HiddenPhase.PREFILL, identity) for i in range(3)))
    sample = TensorStepInput(rows, PackedSamplerRowBatch(rows, (PackedSamplerRowLink(0, 2),)))
    inputs = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4) + 1
    owner.wait_stream(torch.cuda.current_stream(device))
    execution.admit((identity,), owner)

    class Decoder(TokenwiseReplayPrimitive):
        def __init__(self):
            self.cache = torch.zeros((1, 3, 4, 4), device=device)
            self.saved = {}
            self.output = None
            self.calls = []
            self.expected = {}
            owner.wait_stream(torch.cuda.current_stream(device))

        @property
        def rows(self):
            return rows.rows

        @property
        def function_bodies(self):
            return (body,)

        def borrow_streams(self, context, lanes):
            return streams

        def bind_kv(self, resource, context):
            table = torch.zeros((1, 1), device=device, dtype=torch.int32)
            slots = torch.arange(3, device=device, dtype=torch.int64)
            layers = tuple(PagedLayer(str(i), self.cache[:, i].unsqueeze(1), table, slots) for i in range(3))
            plan = bind_paged_access(self.rows, layers)
            return PagedResourceBinding(
                PagedResourceView(resource, plan, tuple((i, str(i)) for i in range(3))), context
            )

        def bind_weights(self, resource, context):
            return ReadonlyResourceBinding(
                ReadonlyTensorView(
                    resource, tuple(ReadonlyTensorCell.bind((i,), (norm.scale,) if i == 3 else ()) for i in range(4))
                ),
                context,
            )

        def bind_context(self, resource, context):
            decoder = self

            class View:
                namespace = id(decoder)

                def validate(self):
                    assert len(decoder.rows) <= resource.domain[0]

                def check(self, mode, ranges):
                    self.validate()

            view = View()
            view.resource = resource
            return HostResourceBinding(view, context)

        def begin(self, row):
            self.saved[row] = self.cache[0, :, row.token_offset].clone()
            return torch.tensor([row.token_offset], device=device), inputs[
                row.packed_row_index : row.packed_row_index + 1
            ]

        def layer(self, row, ordinal, positions, hidden, residual):
            self.calls.append((row.token_offset, ordinal, torch.cuda.current_stream(device).cuda_stream))
            slot = self.cache[0, ordinal]
            previous = slot[row.token_offset - 1] if row.token_offset else torch.zeros_like(hidden)
            slot[row.token_offset].copy_(hidden[0])
            output = hidden + previous + ordinal + 1
            if ordinal == 2 and row.token_offset not in self.expected:
                with torch.no_grad():
                    self.expected[row.token_offset] = norm(output, torch.zeros_like(output))
            return output, torch.zeros_like(output)

        def normalize(self, row, hidden, residual):
            raise AssertionError("normalization must execute the original exported graph")

        def commit(self, row, record):
            record()
            del self.saved[row]

        def restore(self):
            for row, saved in reversed(tuple(self.saved.items())):
                self.cache[0, :, row.token_offset].copy_(saved)
            self.saved.clear()

        def finish(self, outputs):
            self.output = torch.cat(outputs)

    def run():
        decoder = Decoder()
        invocation = execution.start_step(sample, owner)
        invocation.context.replay = decoder
        try:
            for anchor in range(8):
                execution.execute_anchor(invocation, AnchorId(anchor))
                if anchor == 3:
                    execution.complete_sampler(invocation, owner)
            execution.finish_step(invocation)
        except BaseException:
            execution.cancel_step(invocation)
            raise
        owner.synchronize()
        assert not decoder.saved
        # The unaudited model callback must stay on the original stream even
        # while eligible pure statements use a parallel physical plan.
        assert {stream for _, _, stream in decoder.calls} == {owner.cuda_stream}
        torch.testing.assert_close(
            decoder.output, torch.cat(tuple(decoder.expected[i] for i in range(3))), rtol=0, atol=0
        )
        torch.testing.assert_close(norm.scale, original_scale, rtol=0, atol=0)
        return decoder

    def installed_stream_plan():
        task = TokenwiseReplayTask.from_implementation(execution.implementation(TOKENWISE_REPLAY_IMPLEMENTATION))
        return task.region.execution.stream_plan

    tuning = TokenwiseTuning(execution, TOKENWISE_REPLAY_IMPLEMENTATION, proposals, stream_lanes=2)
    try:
        assert tuning.count > 2
        results, plans = [], []
        for index in range(tuning.count):
            tuning.candidate(index)
            tuning.record_rows(rows.rows)
            plans.append(installed_stream_plan())
            results.append(run())
        for result in results[1:]:
            torch.testing.assert_close(result.output, results[0].output, rtol=0, atol=0)
            torch.testing.assert_close(result.cache, results[0].cache, rtol=0, atol=0)
        # Controlled scores exercise installation of a parallel candidate, not a throughput claim.
        winner = next(index for index, plan in enumerate(plans) if plan is not None)
        scores = tuple((100 if index == winner else 200,) for index in range(tuning.count))
        assert tuning.finish(scores).selected == winner
        assert installed_stream_plan() == plans[winner]
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        result = run()
        torch.testing.assert_close(result.output, results[0].output, rtol=0, atol=0)
        torch.testing.assert_close(result.cache, results[0].cache, rtol=0, atol=0)
    finally:
        tuning.abort()
        owner.synchronize()
        execution.retire((identity,))
