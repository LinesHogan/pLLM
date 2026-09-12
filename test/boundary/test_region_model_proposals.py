"""Model schedule proposals survive the real command, tuning and owner paths."""

from dataclasses import replace
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from pllm.authoring import BackendCompileFacts, TensorStepInput
from pllm.backends.vllm.replay import VllmTokenwiseReplay
from pllm.backends.vllm.runtime import BackendCommand, VllmSession, WorkloadModelRunner
from pllm.contracts.backend_attachment import AttachmentGraphMode, AttachmentTopology, BackendPlacementProbe, CausalModelDomain, SignedBackendAttachment
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.region_program import Block, BorrowedResource, Constant, Program, Region, ScalarType, TensorType, Value
from pllm.contracts.region_composition import RegionFunctionBody, inline_region
from pllm.contracts.region_imports import CompiledTensorImport, ExportedTensorImport
from pllm.contracts.region_proposal import REGION_STREAMS_CAPABILITY, RegionScheduleProposal
from pllm.contracts.region_selection import RegionSelection
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import AnchorId
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.paged_access import PagedLayer, PagedResourceView, bind_paged_access
from pllm.execution.region_resources import HostResourceBinding, PagedResourceBinding, ReadonlyResourceBinding
from pllm.execution.region_execution import RegionInstructionExecution
from pllm.execution.decoder_region import decoder_normalize_function
from pllm.execution.exported_cell import ExportedCell, NumericalTransformWarning
from pllm.execution.exported_resources import borrow_exported_inputs
from pllm.execution.region_storage import import_region_functions
from pllm.execution.backend_attachment import compile_backend_attachment
from pllm.execution.replay import TOKENWISE_REPLAY_IMPLEMENTATION, TokenwiseReplayPrimitive, TokenwiseReplayTask
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView
from pllm.execution.tokenwise_region import compile_tokenwise_region
from pllm.worker import GPUWorker
from pllm.workloads.recirculation import RecirculationWorkload


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_worker_tunes_and_installs_parallel_model_proposal_with_original_owner(monkeypatch):
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
    body = RegionFunctionBody(decoder_normalize_function(tensor, 3), Program(Region(Block(
        (Value("row", ScalarType.INDEX), h, r),
        (Constant((), (layer,), 3), *inlined.operations),
        (inlined.results[0].name,))), borrowed.program.functions, (context, weights)))
    body.validate()
    imports = import_region_functions(borrowed.program, local_sources)
    capabilities = (REGION_STREAMS_CAPABILITY, "model/tokenwise_replay@1")
    probe = BackendPlacementProbe("test.model.proposals@1", "a" * 64, "b" * 64, "c" * 64,
        AttachmentTopology.IN_PROCESS, AttachmentGraphMode.EAGER, StateDeviceKind.CUDA, 0, 1, 1,
        TensorDType.FLOAT32, 4, 4, (), capabilities, causal_domain=CausalModelDomain(1, 3, 4),
        region_imports=imports, region_bodies=(body,))
    facts = BackendCompileFacts(probe, RuntimeFacts(device).timeline(capabilities), (), 1, 0, 0, layer_count=3)
    workload = RecirculationWorkload()
    # The driver receives only the signed-source descriptors. It has neither
    # OriginalNorm's class nor its graph/parameter objects, and never owns CUDA.
    script = """
import sys
from unittest.mock import patch
import torch
from pllm.authoring import BackendCompileFacts
from pllm.contracts.backend_attachment import BackendPlacementProbe
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_execution import RegionInstructionExecution
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.workloads.recirculation import RecirculationWorkload
probe = decode_boundary(sys.stdin.read(), BackendPlacementProbe)
facts = BackendCompileFacts(probe, RuntimeFacts(torch.device('cuda:0')).timeline(probe.capabilities), (), 1, 0, 0, layer_count=3)
assert not torch.cuda.is_initialized()
def forbidden(*args, **kwargs):
    raise AssertionError('driver must not bind an executor, allocate pages, or import model graphs')
with patch.object(RegionInstructionExecution, '__init__', forbidden), patch.object(PageBundleArena, '__init__', forbidden), patch.object(ExportedCell, 'from_export', forbidden):
    planned = RecirculationWorkload().plan_backend(facts)
assert not torch.cuda.is_initialized()
print(encode_boundary(planned))
"""
    planned = subprocess.run([sys.executable, "-c", script], input=encode_boundary(probe),
        capture_output=True, text=True, check=True, timeout=60)
    attachment = decode_boundary(planned.stdout, SignedBackendAttachment)
    with patch.object(RegionInstructionExecution, "__init__", side_effect=AssertionError("cold source must not bind an executor")):
        deployment = workload.bind_backend(facts, attachment)
    assert decode_boundary(encode_boundary(probe), BackendPlacementProbe) == probe
    exported = next(item for item in imports if isinstance(item, ExportedTensorImport))
    for invalid, message in (
        (replace(probe, schema_version=1), "probe schema"),
        (replace(probe, region_imports=(*imports, imports[0])), "distinct imports"),
        (replace(probe, region_imports=(CompiledTensorImport(exported.access, "test", "f" * 64, ()),)), "numerical selection"),
        (replace(probe, region_imports=()), "original tensor graph import"),
    ):
        with pytest.raises(ValueError, match=message):
            compile_backend_attachment(replace(deployment.attachment.inputs, probe=invalid))
    execution = deployment.execution
    task = TokenwiseReplayTask.from_implementation(execution.implementation(TOKENWISE_REPLAY_IMPLEMENTATION))
    assert graph.digest not in task.region.functions
    with pytest.raises(ValueError, match="complete original tensor imports"):
        _ = task.region.execution
    with pytest.raises(ValueError, match="exactly the unbound"):
        task.region.bind_functions({})
    with pytest.raises(ValueError, match="differs from its source function"):
        task.region.bind_functions({graph.digest: replace(graph, digest="0" * 64)})
    execution.rebind_implementation(replace(task, region=task.region.bind_functions({graph.digest: graph})))
    source = compile_tokenwise_region(tensor, 4, probe.causal_domain, workload.config.effect(3), stream_lanes=2,
        source_imports=imports, function_bodies=(body,))
    parallel = next(proposal for proposal in source.candidate_proposals
        if proposal.stream_plan is not None and proposal.stream_plan.pipeline_depth == 2)
    assert parallel.stream_plan.host_fences
    proposals = (RegionScheduleProposal(source.payload), parallel)
    budget = next(f.workspace_bytes for f in execution.arena.plan.timeline.task_resources)
    assert parallel.peak_bytes[0].bytes > parallel.payload.devices[0].peak_bytes_upper_bound
    with pytest.raises(ValueError, match="stream capability"):
        source.with_catalog(proposals, budget)
    with pytest.raises(ValueError, match="payload reservation"):
        source.with_catalog(proposals, parallel.peak_bytes[0].bytes - 1, stream_lanes=2)
    forged = replace(parallel.stream_plan, signature="0" * 64)
    with pytest.raises(ValueError, match="stream plan differs"):
        source.with_catalog((proposals[0], replace(parallel, stream_plan=forged)), budget, stream_lanes=2)

    # Only model loading is omitted. Dispatch, session checks, codec, local
    # validation, measurement binding and final AER installation are production.
    backend = VllmTokenwiseReplay.__new__(VllmTokenwiseReplay)
    backend._auxiliary_streams = ()
    session = VllmSession.__new__(VllmSession)
    session._deployment, session._replay = deployment, backend
    session._step = session._auxiliary = session._tuning = None
    session._identities, session._pending_identities = {}, {}
    session._next_step_id = 0
    runner = WorkloadModelRunner.__new__(WorkloadModelRunner)
    runner._workload_session = session
    worker = GPUWorker.__new__(GPUWorker)
    worker.model_runner = runner

    def command(operation, payload=""):
        return worker.workload_command(operation.value, payload)

    def forbidden(*args, **kwargs):
        raise AssertionError("worker verification must not repeat driver search")
    from pllm import _core
    for name in ("explore_region_schedules", "explore_region_streams", "explore_region_windows", "explore_region_tiling"):
        monkeypatch.setattr(_core, name, forbidden)
    public_trace = execution.trace.snapshot()
    rng = torch.random.get_rng_state().clone()
    count = decode_boundary(command(BackendCommand.TUNE_BEGIN, encode_boundary(proposals)), int)
    assert count > 2
    candidates = session._tuning._candidates
    compiled_index = next(index for index, candidate in enumerate(candidates)
        if candidate.stream_plan is not None and any(isinstance(item, CompiledTensorImport) for item in candidate.imports))
    assert candidates[compiled_index].stream_plan != parallel.stream_plan
    owner = torch.cuda.Stream(device=device)
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
            return backend.borrow_streams(context, lanes)

        def bind_kv(self, resource, context):
            table = torch.zeros((1, 1), device=device, dtype=torch.int32)
            slots = torch.arange(3, device=device, dtype=torch.int64)
            layers = tuple(PagedLayer(str(i), self.cache[:, i].unsqueeze(1), table, slots) for i in range(3))
            plan = bind_paged_access(self.rows, layers)
            return PagedResourceBinding(PagedResourceView(resource, plan, tuple((i, str(i)) for i in range(3))), context)

        def bind_weights(self, resource, context):
            return ReadonlyResourceBinding(ReadonlyTensorView(resource,
                tuple(ReadonlyTensorCell.bind((i,), (norm.scale,) if i == 3 else ()) for i in range(4))), context)

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
            return torch.tensor([row.token_offset], device=device), inputs[row.packed_row_index:row.packed_row_index + 1]

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
        backend._invocation = invocation
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
        assert not decoder.saved and not invocation.context.page_publications
        # The unaudited model callback must stay on the original stream even
        # while eligible pure statements use a parallel physical plan.
        assert {stream for _, _, stream in decoder.calls} == {owner.cuda_stream}
        torch.testing.assert_close(decoder.output, torch.cat(tuple(decoder.expected[i] for i in range(3))), rtol=0, atol=0)
        torch.testing.assert_close(norm.scale, original_scale, rtol=0, atol=0)
        return decoder

    try:
        results = []
        for index in range(count):
            command(BackendCommand.TUNE_CANDIDATE, encode_boundary(index))
            session._tuning.record_rows(rows.rows)
            with patch.object(RegionInstructionExecution, "__init__",
                              side_effect=AssertionError("trial must bind before measured execution")):
                results.append(run())
        measured = session._tuning._measurements[1]
        command(BackendCommand.TUNE_CANDIDATE, encode_boundary(1))
        assert session._tuning._measurements[1] is measured
        for result in results[1:]:
            torch.testing.assert_close(result.output, results[0].output, rtol=0, atol=0)
            torch.testing.assert_close(result.cache, results[0].cache, rtol=0, atol=0)
        # Controlled scores exercise installation, not a throughput claim.
        scores = tuple((100 if index == compiled_index else 200,) for index in range(count))
        with pytest.warns(NumericalTransformWarning):
            selected = decode_boundary(command(BackendCommand.TUNE_FINISH, encode_boundary(scores)), RegionSelection)
        assert selected.selected == compiled_index
        installed = TokenwiseReplayTask.from_implementation(execution.implementation(TOKENWISE_REPLAY_IMPLEMENTATION))
        assert installed.region.execution.stream_plan == candidates[compiled_index].stream_plan
        assert installed.region.execution.request.source_imports == source.imports
        assert selected.candidates[compiled_index].changed_functions
        assert execution.trace.snapshot() == public_trace
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        result = run()
        torch.testing.assert_close(result.output, results[0].output, rtol=0, atol=0)
        torch.testing.assert_close(result.cache, results[0].cache, rtol=0, atol=0)
    finally:
        command(BackendCommand.TUNE_ABORT)
        owner.synchronize()
        execution.retire((identity,))
    assert execution.diagnostics().fallback_count == 0
