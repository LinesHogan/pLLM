"""Real video through the public feed: numerical output, CUDA aliases, replay and shutdown."""

import multiprocessing as mp
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from math import cos, pi, tanh
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import pllm
from examples.video_session import video_frames
from pllm.config import ExecutionOptions, ModelConfig
from pllm.contracts.packed_rows import PackedPhaseRowBatch, PackedSamplerRowBatch
from pllm.contracts.state_ref import LineageId, StateRef, VersionId
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.ptts.page_borrow import BorrowedStateRoot
from pllm.execution.ptts.page_bundle import BorrowedPage, PageFieldTransfer, PageSlot
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import model_timeline
from pllm.workloads.recirculation import RecirculationWorkload
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
    timeline = model_timeline(torch.device("cpu"), ("transition/full_root@1",))
    facts = backend_facts(4, (), timeline, PageBundleProviderFacts("video.cpu@1", 3), layer_count=1)
    workload = VideoWorkload(grid_size=2)
    facts = workload.specialize(facts, ExecutionOptions())
    deployment = workload.compile_backend(facts)
    execution, plan = deployment.execution, deployment.attachment.plan
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
                execution.arena.plan_digest, 1, ref, "features", LayoutFieldId(index), PageSlot(0, 0), 0
            )
            pages.append(BorrowedPage(descriptor, value, receiver))
        execution.admit(())
        execution.stage_readonly((BorrowedStateRoot(execution.arena, "features", ref, tuple(pages)),))
        rows = PackedPhaseRowBatch(0, 0, ())
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


@contextmanager
def observe_model(workload):
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
            execution = session._deployment.execution
            step = session._step
            (ref,) = step.context.publication.after_refs
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
            original_close(page)
            assert event.query()
        else:
            original_close(page)

    try:
        with (
            patch.object(pllm, "make_llm", make),
            patch.object(BorrowedStateRoot, "tensors", read),
            patch.object(BorrowedPage, "close", finish),
        ):
            yield
        assert len(identities) == 1 and steps == [(i, i + 1) for i in range(4)]
        assert all(phase == "prefill" for phase in phases[0])
        assert phases[1:] == [("decode",)] * 3
    finally:
        assert len(seen) == 4


@contextmanager
def observe_replay(*, versions):
    from vllm.forward_context import get_forward_context

    from pllm.backends.vllm.region_attention import attention_paged_access
    from pllm.backends.vllm.replay import RouteWave, VllmTokenwiseReplay

    original_make, original_layer = pllm.make_llm, RouteWave.layer
    original_run = VllmTokenwiseReplay.run
    original_capture, original_replay = RouteWave.capture, torch.cuda.CUDAGraph.replay
    holder, history, recordings = {}, {}, {}
    capturing = None
    calls, completed = [], []

    def make(*args, **kwargs):
        llm = original_make(*args, **kwargs)
        session = llm.runner.require_session()
        holder.update(session=session, primitive=session._replay, execution=session._deployment.execution)
        return llm

    def capture(item):
        nonlocal capturing
        capturing = {"patches": [], "normal": None}
        original_capture(item)
        recordings[item.graph] = item, capturing
        capturing = None

    def layer(wave, ordinal, hidden, residual, kv="commit"):
        base = []

        def native(module, args, result):
            if torch.cuda.is_current_stream_capturing():
                base.append(result[0].clone())

        hook = wave.replay.model.layers[ordinal].register_forward_hook(native) if ordinal == 1 else None
        try:
            result = original_layer(wave, ordinal, hidden, residual, kv)
        finally:
            if hook is not None:
                hook.remove()
        if capturing is not None and torch.cuda.is_current_stream_capturing():
            if ordinal == 1 and base and 1 in wave.fields:
                capturing["patches"].append((base[0], wave.fields[1][0].clone(), result[0].clone()))
            if ordinal == len(wave.replay.model.layers) - 1 and capturing["normal"] is None:
                capturing["normal"] = wave.replay.model.norm(result[0] + result[1]).clone()
        return result

    def replay(graph):
        original_replay(graph)
        if graph not in recordings:
            return
        wave, observed = recordings[graph]
        torch.testing.assert_close(wave.output, observed["normal"], rtol=0, atol=0)
        if 1 not in wave.fields:
            return
        step = holder["session"]._step
        context = step.context
        if context.held_input is None:
            (ref,) = context.publication.after_refs
            pin = context.borrow_published((ref,))
        else:
            (ref,) = context.pin.refs
            pin = context.pin
        field = context.arena.borrow_field(pin, ref, LayoutFieldId(0))
        staged = wave.fields[1][0]
        assert staged.data_ptr() != field.data_ptr()
        torch.testing.assert_close(staged, field.expand_as(staged), rtol=0, atol=0)
        for hidden, delta, patched in observed["patches"]:
            torch.testing.assert_close(delta, field.expand_as(delta), rtol=0, atol=0)
            torch.testing.assert_close(patched, hidden + delta.to(hidden.dtype), rtol=0, atol=0)
            calls.append((step.rows.step_id, int(ref.version_id)))

    def run(self, *args):
        result = original_run(self, *args)
        if holder and self is holder["primitive"]:
            for view, saved in history.values():
                torch.testing.assert_close(view, saved, rtol=0, atol=0)
            access = attention_paged_access(get_forward_context(), self.attentions, self.rows)
            for row in self.rows:
                for ordinal, layer in enumerate(self.model.layers):
                    block, offset = map(int, access.cell(layer.self_attn.attn.layer_name, row))
                    view = layer.self_attn.attn.kv_cache[block, :, offset, :]
                    history[ordinal, row.token_offset] = view, view.clone()
            step = holder["session"]._step
            seen = {ref for step_id, ref in calls if step_id == step.rows.step_id}
            assert seen == {versions[step.rows.step_id]}
            completed.append(tuple(row.identity for row in self.rows))
        return result

    with (
        patch.object(pllm, "make_llm", make),
        patch.object(RouteWave, "layer", layer),
        patch.object(RouteWave, "capture", capture),
        patch.object(torch.cuda.CUDAGraph, "replay", replay),
        patch.object(VllmTokenwiseReplay, "run", run),
    ):
        yield
    assert len(completed) == len(versions) and len({identity for rows in completed for identity in rows}) == 1
    assert len([step for step, _ in calls if step == 0]) > 2
    diagnostics = holder["execution"].diagnostics()
    assert diagnostics.update_count == len(set(versions))
    assert diagnostics.hold_count == len(versions) - len(set(versions))


def model_config():
    return ModelConfig(
        os.environ.get("PLLM_VIDEO_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        max_model_len=64,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.25,
    )


def produce_frames(feed, source, events):
    previous = source[0].cuda()
    try:
        for kind, frame in zip(events, source, strict=True):
            if kind == "missing":
                feed.missing()
                continue
            current = frame.cuda()
            values = frame_features(current, previous, 2)
            feed.publish(
                **dict(zip(("qp", "residual_energy", "motion_magnitude", "segmentation"), values, strict=True))
            )
            previous = current
            for value in values:
                value.zero_()  # publish returning permits immediate mutation of its source tensors.
        feed.end()
    finally:
        feed.close()


def run_video(workload, events, steps, config=None):
    from vllm import SamplingParams

    process = None
    try:
        with pllm.make_llm(config=config or model_config(), workload=workload) as llm:
            process = mp.get_context("spawn").Process(
                target=produce_frames,
                args=(llm.feed("features"), frames(len(events)), events),
            )
            process.start()
            outputs = llm.generate(
                ["Describe the scene in this video frame:"],
                SamplingParams(max_tokens=steps, temperature=0.0, ignore_eos=True),
            )
            assert len(outputs[0].outputs[0].token_ids) == steps
            return llm.diagnostics(), outputs[0].outputs[0].text
    finally:
        if process is not None:
            process.join(120)
            if process.is_alive():
                process.kill()
                process.join()
                pytest.fail("feed owner did not finish after consumer teardown")
            assert process.exitcode == 0


@dataclass(frozen=True)
class RecirculatingVideo(VideoWorkload, RecirculationWorkload):
    """Compose the video Activation and an ordinary user-authored layer route."""


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="real LLM/CUDA IPC requires a GPU")
def test_video_session_zero_copy_backpressure_and_ack_lifetime():
    workload = VideoWorkload(grid_size=2)
    with observe_model(workload):
        diagnostics, _ = run_video(workload, ("publish",) * 4, 4)
    assert diagnostics.update_count == 4
    config = replace(model_config(), execution=ExecutionOptions(allow_numerical_transforms=False))
    workload = RecirculatingVideo(grid_size=2, layer=1, alpha=0.6)
    with observe_replay(versions=(1, 1, 2, 2, 2)):
        diagnostics, _ = run_video(workload, ("publish", "missing", "publish"), 5, config)
    assert diagnostics.update_count == 2 and diagnostics.hold_count == 3


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="real LLM/CUDA IPC requires a GPU")
def test_video_session_failure_retains_owner_without_fallback():
    from pllm.backends.vllm.session import VllmSession

    prepare_input = VllmSession.prepare_input

    def recompute_history(session, rows):
        if session.next_step_id == 1:
            assert session.diagnostics().update_count == 1
            session._token_frontiers = dict.fromkeys(session._token_frontiers, 1 << 30)
        return prepare_input(session, rows)

    with patch.object(VllmSession, "prepare_input", recompute_history), pytest.raises(RuntimeError, match="preempt"):
        run_video(VideoWorkload(grid_size=2), ("publish", "publish"), 2)
