"""Video frame features injected into a real vLLM child through zero-copy CUDA pages.

The CPU test checks the frame-feature math against a closed-form oracle. The GPU
tests run the example ``VideoSession`` with an instrumented worker: injected layer
outputs equal the captured output plus the published delta, KV caches stay intact,
pages are reused only after the child's ACK, held versions replay, and a dying child
fail-stops the session while it keeps the page owner.
"""

import multiprocessing as mp
import os
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
from math import cos, pi, tanh
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
from pllm.execution.ptts.page_bundle import BorrowedPage, PageFieldTransfer, PageSlot
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import RuntimeFacts
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
    timeline = RuntimeFacts(torch.device("cpu")).timeline(("transition/full_root@1",))
    facts = backend_facts(4, (), timeline, PageBundleProviderFacts("video.cpu@1", 3), layer_count=1)
    workload = VideoWorkload(grid_size=2)
    facts = workload.specialize(facts, ExecutionOptions())
    deployment = workload.compile_backend(facts)
    execution, plan = deployment.execution, deployment.attachment.inputs.plan
    (layout,) = (item for item in plan.tensor_layout_decision.bindings if item.resource == "features")
    ref = StateRef(layout.layout_id, LineageId(1), VersionId(1))
    with ExitStack() as lifetime:
        pages = []
        for index, value in enumerate(values):
            peer, receiver = mp.Pipe()
            lifetime.callback(peer.close)
            lifetime.callback(receiver.close)
            # Received-page fixture only; actual receive_field is CUDA-only and tested below.
            descriptor = PageFieldTransfer(
                plan.schedule_decision.signature, 1, ref, "features", LayoutFieldId(index), PageSlot(0, 0), 0
            )
            pages.append(BorrowedPage(descriptor, value, receiver))
        execution.admit(())
        execution.stage_readonly((BorrowedStateRoot(plan, "features", ref, tuple(pages)),))
        rows = PackedPhaseRowBatch(1, 0, 0, ())
        result = execution.step(pllm.authoring.TensorStepInput(rows, PackedSamplerRowBatch(rows, ())))
        signal = torch.cat(
            (
                values[0].flatten() / 63,
                values[1].flatten().sqrt(),
                values[2].flatten() / 2**0.5,
                values[3].flatten() / 3,
            )
        ).tolist()
        expected = torch.tensor(
            [
                workload.gain
                * tanh(
                    sum(cos(pi * (k + 1) * (j + 0.5) / len(signal)) * x for j, x in enumerate(signal))
                    / len(signal) ** 0.5
                )
                for k in range(4)
            ]
        )
        torch.testing.assert_close(
            execution.snapshot(result.publication.after_refs[0])[0], expected, atol=1e-6, rtol=1e-6
        )
        execution.retire(())


def observed_worker(probe, start, close, control, channels, workload, config):
    original_make, original_read, original_close = pllm.make_llm, BorrowedStateRoot.tensors, BorrowedPage.close
    seen = set()
    fenced = False
    steps, identities, phases = [], set(), []

    def make(*args, **kwargs):
        llm = original_make(*args, **kwargs)
        layer = llm.runner.model.model.layers[workload.layer]
        session = llm.runner.require_session()
        before = []
        history, current_views = {}, {}

        def views(module, rows):
            from vllm.forward_context import get_forward_context

            from pllm.backends.vllm.region_attention import attention_metadata

            _, cache, slots = attention_metadata(get_forward_context(), module.self_attn.attn)
            return tuple(
                (
                    row.token_offset,
                    cache[
                        int(slots[row.packed_row_index]) // cache.shape[2],
                        :,
                        int(slots[row.packed_row_index]) % cache.shape[2],
                        :,
                    ],
                )
                for row in rows
            )

        def capture(module, inputs, output):
            rows = session._step.rows.row_batch.rows
            before.append((output.detach().clone(), tuple(value.clone() for _, value in views(layer, rows))))

        layer.mlp.register_forward_hook(capture)

        def check(module, inputs, output):
            execution = llm.deployment.execution
            step = session._step
            (ref,) = step.invocation.context.publication.after_refs
            (delta,) = execution.snapshot(ref, torch.cuda.current_stream())
            expected, prior_kv = before.pop()
            original = expected.clone()
            selected = torch.tensor(sorted({0, len(expected) - 1}), device=expected.device)
            expected.index_add_(0, selected, delta.to(expected).expand(len(selected), -1))
            torch.testing.assert_close(output[0], expected, rtol=0, atol=0)
            assert torch.count_nonzero(expected - original) > 0
            rows = step.rows.row_batch.rows
            for (_, actual), prior in zip(views(layer, rows), prior_kv, strict=True):
                torch.testing.assert_close(
                    actual, prior, rtol=0, atol=0
                )  # Layer-output injection cannot rewrite this layer's KV.
            for ordinal, decoder in enumerate(llm.runner.model.model.layers):
                current_views.update({(ordinal, position): value for position, value in views(decoder, rows)})
            identities.update(row.identity for row in rows)
            steps.append((step.rows.step_id, int(ref.version_id)))
            phases.append(tuple(row.phase.value for row in rows))
            probe.send_bytes(b"patched")

        layer.register_forward_hook(check)
        complete = session.complete_forward

        def completed():
            for view, saved in history.values():
                torch.testing.assert_close(view, saved, rtol=0, atol=0)  # Previously cached positions stay unchanged.
            history.update({key: (view, view.clone()) for key, view in current_views.items()})
            current_views.clear()
            complete()

        session.complete_forward = completed
        return llm

    def read(root, pin):
        values = original_read(root, pin)
        for value, page in zip(values, root.pages, strict=True):
            storage = page.descriptor.cuda_storage
            assert storage is not None and value.data_ptr() == page.tensor.data_ptr()
            assert value.storage_offset() == page.descriptor.storage_offset
            assert (
                value.untyped_storage().nbytes() == storage.size_bytes
            )  # A field clone cannot satisfy the pool extent.
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
        with (
            patch.object(pllm, "make_llm", make),
            patch.object(BorrowedStateRoot, "tensors", read),
            patch.object(BorrowedPage, "close", finish),
        ):
            consume(control, channels, workload, config)
        assert len(identities) == 1 and steps == [(i, i + 1) for i in range(4)]
        assert all(phase == "prefill" for phase in phases[0])
        assert phases[1:] == [("decode",)] * 3
        probe.send_bytes(b"one-request-four-steps-kv-ok")
    finally:
        probe.close()


def preempted_worker(probe, control, channels, workload, config):
    from pllm.backends.vllm.runtime import VllmSession

    original = VllmSession.before_execute

    def preempt(session, output):
        if session.next_step_id == 1:
            assert session.diagnostics().update_count == 1
            output = replace(output, preempted_req_ids=set(session._identities))
        try:
            return original(session, output)
        except RuntimeError as error:
            probe.send_bytes(str(error).encode())
            raise

    try:
        with patch.object(VllmSession, "before_execute", preempt):
            consume(control, channels, workload, config)
    finally:
        probe.close()


def composed_worker(probe, control, channels, workload, config, *, versions=(1, 2, 3)):
    from pllm.backends.vllm.replay import VllmTokenwiseReplay
    from pllm.execution.exported_cell import ExportedCell

    original_make, original_invoke = pllm.make_llm, ExportedCell.invoke
    original_normalize, original_finish = VllmTokenwiseReplay.normalize, VllmTokenwiseReplay.finish
    holder, normals, history = {}, {}, {}
    calls, completed = [], []

    def make(*args, **kwargs):
        llm = original_make(*args, **kwargs)
        session = llm.runner.require_session()
        holder.update(session=session, primitive=session._replay, execution=llm.deployment.execution)
        return llm

    def invoke(cell, args):
        result = original_invoke(cell, args)
        if holder and cell.inputs == ("hidden", "delta") and cell.outputs == ("patched",):
            hidden, delta = args
            step = holder["session"]._step
            context = step.invocation.context
            if context.held_input is None:
                (ref,) = context.publication.after_refs
                pin = context.borrow_published((ref,))
            else:  # a held step replays the pinned version without publishing
                (ref,) = context.pin.refs
                pin = context.pin
            # The replay reads the pinned version's page in place.
            assert delta.data_ptr() == context.arena.borrow_field(pin, ref, LayoutFieldId(0)).data_ptr()
            torch.testing.assert_close(result[0], hidden + delta.to(hidden.dtype), rtol=0, atol=0)
            calls.append((step.rows.step_id, int(ref.version_id)))
        return result

    def normalize(self, row, hidden, residual):
        result = original_normalize(self, row, hidden, residual)
        if holder and self is holder["primitive"]:
            normals[row] = result.clone()
        return result

    def finish(self, outputs):
        if holder and self is holder["primitive"]:
            for actual, row in zip(outputs, self.rows, strict=True):
                torch.testing.assert_close(
                    actual, normals.pop(row), rtol=0, atol=0
                )  # Sampling receives normal output, not replay output.
            for view, saved in history.values():
                torch.testing.assert_close(view, saved, rtol=0, atol=0)
            for row in self.rows:
                for ordinal, layer in enumerate(self.model.layers):
                    block, offset = self._access.slot(layer.self_attn.attn.layer_name, row)
                    view = layer.self_attn.attn.kv_cache[block, :, offset, :]
                    history[ordinal, row.token_offset] = view, view.clone()
            step = holder["session"]._step
            seen = {ref for step_id, ref in calls if step_id == step.rows.step_id}
            assert seen == {versions[step.rows.step_id]}
            completed.append(tuple(row.identity for row in self.rows))
        return original_finish(self, outputs)

    try:
        with (
            patch.object(pllm, "make_llm", make),
            patch.object(ExportedCell, "invoke", invoke),
            patch.object(VllmTokenwiseReplay, "normalize", normalize),
            patch.object(VllmTokenwiseReplay, "finish", finish),
        ):
            consume(control, channels, workload, config)
        assert len(completed) == len(versions) and len({identity for rows in completed for identity in rows}) == 1
        assert (
            len([step for step, _ in calls if step == 0]) > 2
        )  # Real prefill suffix replay also applied the same injection.
        # The llm handle closes with consume; the detached execution keeps its counters.
        diagnostics = holder["execution"].diagnostics()
        assert diagnostics.update_count == len(set(versions))
        assert diagnostics.hold_count == len(versions) - len(set(versions)) and diagnostics.fallback_count == 0
        probe.send_bytes(b"composed-pins-hidden-normal-output-kv-ok")
    finally:
        probe.close()


def model_config():
    return ModelConfig(
        os.environ.get("PLLM_VIDEO_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        max_model_len=64,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.25,
    )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="real LLM/CUDA IPC requires a GPU")
def test_video_session_zero_copy_backpressure_and_ack_lifetime():
    source = frames()
    context = mp.get_context("spawn")
    probe, child_probe = context.Pipe()
    start, close = context.Event(), context.Event()
    try:
        with VideoSession(
            VideoWorkload(grid_size=2),
            model_config(),
            steps=len(source),
            worker=partial(observed_worker, child_probe, start, close),
        ) as session:
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
            with pytest.raises(RuntimeError):  # v1 still has live owners
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
            assert (
                reused.slot_id == first.fields[0].page.slot_id
                and reused.generation == first.fields[0].page.generation + 1
            )
            start.set()
            results = [session.drain() for _ in range(3)]
            assert results[-1].updates == 4 and all(result.fallback == 0 for result in results)
            assert all(result.text == "" for result in results[:-1])
            for _ in range(3):
                assert probe.poll(120)
                decode_boundary(probe.recv_bytes().decode(), tuple[PageFieldTransfer, ...])
                assert probe.poll(120) and probe.recv_bytes() == b"patched"
        assert session.arena.active_lineages == session.arena.pending_pin_releases == 0
        assert probe.poll(120) and probe.recv_bytes() == b"one-request-four-steps-kv-ok"
    finally:
        start.set()
        close.set()
        probe.close()
        child_probe.close()
    # One composed session covers fresh versions, held versions and explicit
    # absent/end evidence through the same tokenwise replay worker.
    config = replace(model_config(), execution=ExecutionOptions(allow_numerical_transforms=False))
    workload = VideoWorkload(
        grid_size=2, layer=1, recurrence=pllm.Recurrence(source_layer=-1, destination_layer=0, alpha=0.6, beta=None)
    )
    probe, child_probe = context.Pipe()
    try:
        with VideoSession(
            workload, config, steps=5, worker=partial(composed_worker, child_probe, versions=(1, 1, 2, 2, 2))
        ) as held:
            child_probe.close()
            held.submit(source[0])
            held.missing()
            held.submit(source[1])
            held.missing(end=True)
            results = [held.drain() for _ in range(5)]
            assert [result.step_id for result in results] == list(range(5))
            assert [result.updates for result in results] == [1, 1, 2, 2, 2]
            assert [result.holds for result in results] == [0, 1, 1, 2, 3]
            assert [result.missing for result in results] == [None, "absent", None, "end", "end"]
            assert [result.ref is None for result in results] == [False, True, False, True, True]
            assert all(result.fallback == 0 for result in results)
        assert probe.poll(120) and probe.recv_bytes() == b"composed-pins-hidden-normal-output-kv-ok"
        assert held.arena.active_lineages == held.arena.pending_pin_releases == 0
    finally:
        probe.close()
        child_probe.close()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="real LLM/CUDA IPC requires a GPU")
def test_video_session_failure_retains_owner_without_fallback():
    # Bad ACKs, closed channels and descriptor drift are covered at the arena
    # boundary in test_page_bundle_cuda.py. Here one real child dies mid-request
    # (vLLM preemption is rejected after a real forward) and the supervising
    # session must fail-stop while retaining the original page owner.
    (frame,) = frames(1)
    context = mp.get_context("spawn")
    probe, child_probe = context.Pipe()
    try:
        with VideoSession(
            VideoWorkload(grid_size=2), model_config(), steps=2, worker=partial(preempted_worker, child_probe)
        ) as session:
            child_probe.close()
            with pytest.raises((EOFError, OSError, RuntimeError)):
                session.submit(frame)
                session.missing(end=True)  # An explicit End cannot mask a transport integrity failure.
                session.drain()
            assert probe.poll(120)
            assert b"preemption" in probe.recv_bytes()
            assert session.failed and session.arena.root(session.current).ref == session.current
            assert session.arena.usage().pinned_versions > 0
            with pytest.raises(RuntimeError):  # the session fail-stopped
                session.submit(frame)
        assert not session.process.is_alive()
    finally:
        probe.close()
        child_probe.close()
