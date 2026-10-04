"""Real-frame math plus CUDA stories through the actual LLM session entry point."""

from contextlib import ExitStack
from dataclasses import replace
from functools import partial
from math import cos, pi, tanh
import multiprocessing as mp
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import pllm
from examples.video_session import VideoSession, consume, video_frames
from pllm.config import ExecutionOptions, ModelConfig
from pllm.contracts.packed_rows import PackedPhaseRowBatch, PackedSamplerRowBatch
from pllm.contracts.state_ref import LineageId, StateRef, VersionId
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.ptts.page_borrow import BorrowedStateRoot
from pllm.execution.ptts.page_bundle import BorrowedPage, PageBundleArena, PageFieldTransfer, PageSlot
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.scheduler import compile_state_schedule, validate_plan_bundle
from pllm.workloads.video import VideoWorkload, frame_features
from test.backend_facts import backend_facts


def frames(count=4):
    path = Path(os.environ.get("PLLM_VIDEO_FIXTURE", "/home/user/TLM/data/mllm_demo_data/1.mp4"))
    if not path.is_file():
        pytest.skip("set PLLM_VIDEO_FIXTURE to a real video")
    result = list(video_frames(path, 2, count))
    assert len(result) == count
    return result


def test_video_real_frame_math_and_signed_composition():
    first, second = frames(2)
    values = frame_features(second, first, 2)
    stationary = frame_features(first, first, 2)
    assert torch.count_nonzero(stationary[1]) == torch.count_nonzero(stationary[2]) == 0
    facts = replace(backend_facts(4, (), RuntimeFacts(torch.device("cpu")).timeline(("transition/full_root@1",)),
        PageBundleProviderFacts("video.cpu@1", 3)), layer_count=1)
    workload = VideoWorkload(grid_size=2)
    facts = workload.specialize(facts, ExecutionOptions())
    deployment = workload.compile_backend(facts)
    execution, plan = deployment.execution, deployment.attachment.inputs.plan
    assert execution.arena.owned_resources == ("inject",)
    binding, = deployment.attachment.inputs.request.activation_bindings
    assert binding.operator is pllm.ActivationPatchOperator.ADD
    layout, = (item for item in plan.tensor_layout_decision.bindings if item.resource == "features")
    ref = StateRef(layout.layout_id, LineageId(1), VersionId(1))
    with ExitStack() as lifetime:
        pages = []
        for index, value in enumerate(values):
            peer, receiver = mp.Pipe()
            lifetime.callback(peer.close)
            lifetime.callback(receiver.close)
            # Received-page fixture only; actual receive_field is CUDA-only and tested below.
            descriptor = PageFieldTransfer(plan.schedule_decision.signature, 1, ref, "features", LayoutFieldId(index), PageSlot(0, 0), 0)
            pages.append(BorrowedPage(descriptor, value, receiver))
        execution.admit(())
        execution.stage_readonly((BorrowedStateRoot(plan, "features", ref, tuple(pages)),))
        rows = PackedPhaseRowBatch(1, 0, 0, ())
        result = execution.step(pllm.authoring.TensorStepInput(rows, PackedSamplerRowBatch(rows, ())))
        signal = torch.cat((values[0].flatten() / 63, values[1].flatten().sqrt(),
            values[2].flatten() / 2 ** 0.5, values[3].flatten() / 3)).tolist()
        expected = torch.tensor([workload.gain * tanh(sum(cos(pi * (k + 1) * (j + .5) / len(signal)) * x
            for j, x in enumerate(signal)) / len(signal) ** .5) for k in range(4)])
        torch.testing.assert_close(execution.snapshot(result.publication.after_refs[0])[0], expected, atol=1e-6, rtol=1e-6)
        execution.retire(())


def observed_worker(probe, start, close, control, channels, workload, config):
    original_make, original_read, original_close = pllm.make_llm, BorrowedStateRoot.tensors, BorrowedPage.close
    seen = set()
    fenced = False
    def make(*args, **kwargs):
        llm = original_make(*args, **kwargs)
        assert llm.deployment.execution.arena.owned_resources == ("inject",)
        assert len(llm.deployment.execution.arena.allocation_ranges) == 1
        layer = llm.runner.model.model.layers[workload.layer]
        before = []
        layer.mlp.register_forward_hook(lambda module, inputs, output: before.append(output.detach().clone()))
        def check(module, inputs, output):
            execution = llm.deployment.execution
            delta, = execution.snapshot(execution.directory.latest_refs(())[0], torch.cuda.current_stream())
            expected = before.pop()
            original = expected.clone()
            selected = torch.tensor(sorted({0, len(expected) - 1}), device=expected.device)
            expected.index_add_(0, selected, delta.to(expected).expand(len(selected), -1))
            torch.testing.assert_close(output[0], expected, rtol=0, atol=0)
            assert torch.count_nonzero(expected - original) > 0
            probe.send_bytes(b"patched")
        layer.register_forward_hook(check)
        return llm
    def read(root, pin):
        values = original_read(root, pin)
        for value, page in zip(values, root.pages, strict=True):
            storage = page.descriptor.cuda_storage
            assert storage is not None and value.data_ptr() == page.tensor.data_ptr()
            assert value.storage_offset() == page.descriptor.storage_offset
            assert value.untyped_storage().nbytes() == storage.size_bytes  # A field clone cannot satisfy the pool extent.
        if root.ref not in seen:
            seen.add(root.ref)
            probe.send_bytes(encode_boundary(tuple(page.descriptor for page in root.pages)).encode())
            assert start.wait(120), "test did not release the consumer"
        return values
    def finish(page):
        nonlocal fenced
        if not fenced:
            fenced = True
            event = torch.cuda.Event()
            with torch.cuda.stream(page.stream):
                torch.cuda._sleep(100_000_000)
                event.record(page.stream)
            assert not event.query()
            probe.send_bytes(b"before-close")
            assert close.wait(120)
            original_close(page)
            assert event.query()
            probe.send_bytes(b"closed")
        else:
            original_close(page)
    try:
        with patch.object(pllm, "make_llm", make), patch.object(BorrowedStateRoot, "tensors", read), patch.object(BorrowedPage, "close", finish):
            consume(control, channels, workload, config)
    finally:
        probe.close()


def fault_worker(mode, control, channels, workload, config):
    receive = PageBundleArena.receive_field
    def drift(plan, channel, **kwargs):
        graph = replace(plan.semantic_graph, execution_parameters_digest="d" * 64)
        state_decision, state, schedule, physical = compile_state_schedule(graph, plan.timeline, plan.state_request)
        different = replace(plan, semantic_graph=graph, state_decision=state_decision, state_physical=state,
            schedule_decision=schedule, schedule_physical=physical)
        validate_plan_bundle(different)  # A distinct valid plan, not a corrupted signature string.
        return receive(different, channel, **kwargs)
    def fail(page):
        if mode == "exit":
            os._exit(17)  # Imported pages and outstanding acknowledgements disappear together.
        page.stream.synchronize()
        page._tensor = None
        page._connection.send_bytes(b"another export")
    target, method, replacement = (PageBundleArena, "receive_field", staticmethod(drift)) if mode == "drift" else (BorrowedPage, "close", fail)
    with patch.object(target, method, replacement):
        consume(control, channels, workload, config)


def model_config():
    return ModelConfig(os.environ.get("PLLM_VIDEO_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        max_model_len=64, max_num_seqs=1, max_num_batched_tokens=64, gpu_memory_utilization=0.25)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real LLM/CUDA IPC requires a GPU")
def test_video_session_zero_copy_backpressure_and_ack_lifetime():
    source = frames()
    context = mp.get_context("spawn")
    probe, child_probe = context.Pipe()
    start, close = context.Event(), context.Event()
    try:
        with VideoSession(VideoWorkload(grid_size=2), model_config(),
            worker=partial(observed_worker, child_probe, start, close)) as session:
            child_probe.close()
            session.submit(source[0])
            assert probe.poll(120)
            received = decode_boundary(probe.recv_bytes().decode(), tuple[PageFieldTransfer, ...])
            first = session.pending[0]
            assert received == first.fields  # Same GPU UUID/handle/offset, all fields of v1.
            session.submit(source[1])
            session.submit(source[2])
            assert not session.arena.release_complete(first.release)
            assert session.arena.root(first.ref).pages == tuple(field.page for field in first.fields)
            with pytest.raises(RuntimeError, match="live owners"):
                session.arena.prepare_transition((session.current,), session.stream)
            start.set()
            assert probe.poll(120) and probe.recv_bytes() == b"patched"
            assert probe.poll(120) and probe.recv_bytes() == b"before-close"
            start.clear()
            assert not session.arena.release_complete(first.release)
            close.set()
            assert probe.poll(120) and probe.recv_bytes() == b"closed"
            assert session.drain().ref == first.ref
            session.submit(source[3])
            reused = session.pending[-1].fields[0].page
            assert reused.slot_id == first.fields[0].page.slot_id and reused.generation == first.fields[0].page.generation + 1
            start.set()
            results = [session.drain() for _ in range(3)]
            assert results[-1].updates == 4 and all(result.fallback == 0 for result in results)
        assert session.arena.active_lineages == session.arena.pending_pin_releases == 0
    finally:
        start.set()
        close.set()
        probe.close()
        child_probe.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real LLM/CUDA IPC requires a GPU")
@pytest.mark.parametrize("mode", ("bad_ack", "exit", "drift"))
def test_video_session_failure_retains_owner_without_fallback(mode):
    # The group's trailing CudaIPCTypes warning comes from deliberate process death; the isolated success path has none.
    frame, = frames(1)
    with VideoSession(VideoWorkload(grid_size=2), model_config(), worker=partial(fault_worker, mode)) as session:
        with pytest.raises((EOFError, OSError, RuntimeError)):
            session.submit(frame)
            session.drain()
        assert session.failed and session.arena.root(session.current).ref == session.current
        assert session.arena.usage().pinned_versions > 0
        with pytest.raises(RuntimeError, match="fail-stopped"):
            session.submit(frame)
    assert not session.process.is_alive()
