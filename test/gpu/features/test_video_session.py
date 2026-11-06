"""Real video through the public feed: numerical output, CUDA aliases, replay and shutdown."""

import multiprocessing as mp
import os
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass, replace
from unittest.mock import patch

import pytest
import torch

import pllm
from pllm.config import ExecutionOptions, ModelConfig
from pllm.contracts.semantic_graph import StateResource
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.ptts.page_borrow import BorrowedPage, BorrowedStateRoot
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.state_feed import _ConsumerClosed
from pllm.workloads.recirculation import RecirculationWorkload
from pllm.workloads.video import VideoWorkload, frame_features
from test.workloads.test_video import frames


@contextmanager
def observe_model(workload):
    from vllm.forward_context import get_forward_context

    from pllm.backends.vllm.model_sites import ModelSites, _Boundary

    original_make, original_read, original_close = pllm.make_llm, BorrowedStateRoot.tensors, BorrowedPage.close
    original_install, original_emit = ModelSites.__init__, _Boundary.emit
    seen, recordings, caches, history = set(), {}, {}, {}
    fenced = False
    steps, identities, phases = [], set(), []
    model = None

    def install(sites, decoder, *args, **kwargs):
        nonlocal model
        model = decoder.model.model
        original_install(sites, decoder, *args, **kwargs)

    def cells(cache, slots, count):
        indices = slots[:count].clamp_min(0)
        return cache[indices // cache.shape[2], :, indices % cache.shape[2]]

    def emit(boundary, inputs, outputs):
        if boundary.before != ("mlp", workload.layer):
            return original_emit(boundary, inputs, outputs)
        assert model is not None
        context = get_forward_context()
        if context.slot_mapping is None:
            return original_emit(boundary, inputs, outputs)
        assert isinstance(context.slot_mapping, dict)
        for ordinal, layer in enumerate(model.layers):
            attention = layer.self_attn.attn
            caches[ordinal] = attention.kv_cache, context.slot_mapping[attention.layer_name]
        branch, stream = (value.clone() for value in inputs)
        cache, slots = caches[workload.layer]
        prior_kv = cells(cache, slots, len(branch)).clone()
        original_emit(boundary, inputs, outputs)
        recordings[len(branch)] = (
            branch,
            stream,
            outputs[1].clone(),
            prior_kv,
            cells(cache, slots, len(branch)).clone(),
            dict(caches),
        )

    def make(*args, **kwargs):
        llm = original_make(*args, **kwargs)
        session = llm.runner.require_session()
        complete = session.complete_forward

        def completed():
            step = session._step
            rows = step.rows.row_batch.rows
            branch, stream, actual, prior_kv, after_kv, access = recordings[
                min(n for n in recordings if n >= len(rows))
            ]
            (ref,) = step.context.publication.after_refs
            (delta,) = session._deployment.execution.snapshot(ref, torch.cuda.current_stream())
            expected = branch[: len(rows)].clone()
            selected = torch.tensor(sorted({0, len(rows) - 1}), device=expected.device)
            expected.index_add_(0, selected, delta.to(expected).expand(len(selected), -1))
            torch.testing.assert_close(actual[: len(rows)], stream[: len(rows)] + expected, rtol=0, atol=0)
            assert torch.count_nonzero(expected - branch[: len(rows)]) > 0
            torch.testing.assert_close(after_kv[: len(rows)], prior_kv[: len(rows)], rtol=0, atol=0)
            for view, saved in history.values():
                torch.testing.assert_close(view, saved, rtol=0, atol=0)
            for ordinal, (cache, slots) in access.items():
                for row in rows:
                    slot = int(slots[row.packed_row_index])
                    view = cache[slot // cache.shape[2], :, slot % cache.shape[2]]
                    history[ordinal, row.token_offset] = view, view.clone()
            identities.update(row.identity for row in rows)
            steps.append((step.rows.step_id, int(ref.version_id)))
            phases.append(tuple(row.phase.value for row in rows))
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
            patch.object(ModelSites, "__init__", install),
            patch.object(_Boundary, "emit", emit),
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

    from pllm.backends.vllm.activation import VllmActivationPatch
    from pllm.backends.vllm.custom_forward import ProgramWave, VllmCustomForward
    from pllm.backends.vllm.custom_forward_attention import attention_paged_access

    original_make, original_layer, original_patch = pllm.make_llm, ProgramWave.layer, VllmActivationPatch.patch
    original_run = VllmCustomForward.run
    original_capture, original_replay = ProgramWave.capture, torch.cuda.CUDAGraph.replay
    holder, history, recordings = {}, {}, {}
    capturing = None
    calls, completed = [], []

    def make(*args, **kwargs):
        llm = original_make(*args, **kwargs)
        session = llm.runner.require_session()
        holder.update(session=session, primitive=session._custom_forward, execution=session._deployment.execution)
        return llm

    def capture(item):
        nonlocal capturing
        capturing = {"patches": [], "normal": None}
        original_capture(item)
        recordings[item.graph] = item, capturing
        capturing = None

    def layer(wave, ordinal, hidden, residual, kv="commit"):
        result = original_layer(wave, ordinal, hidden, residual, kv)
        if capturing is not None and torch.cuda.is_current_stream_capturing():
            if ordinal == len(wave.custom_forward.model.layers) - 1 and capturing["normal"] is None:
                capturing["normal"] = wave.custom_forward.model.norm(result[0] + result[1]).clone()
        return result

    def patching(activation, resource, layer, rows):
        """Record a captured wave's MLP output of the patched layer before and after the patch."""
        apply = original_patch(activation, resource, layer, rows)

        def recorded(hidden):
            base = hidden.clone() if capturing is not None and torch.cuda.is_current_stream_capturing() else None
            result = apply(hidden)
            if base is not None:
                capturing["patches"].append((base, result.clone()))
            return result

        return recorded

    def replay(graph):
        original_replay(graph)
        if graph not in recordings:
            return
        wave, observed = recordings[graph]
        torch.testing.assert_close(wave.output, observed["normal"], rtol=0, atol=0)
        if not any(wave.signature[1]):
            return
        step = holder["session"]._step
        context = step.context
        if context.held_input is None:
            (ref,) = context.publication.after_refs
            pin = context.borrow_published(StateRefs.of((ref,)))
        else:
            (ref,) = context.pin.refs
            pin = context.pin
        field = context.arena.borrow_field(pin, ref, LayoutFieldId(0))
        # Captured waves gather the invocation's exact PTTS slot, without per-row field copies.
        patch = holder["session"]._activation._patches[StateResource("inject"), 1]
        slots = patch.table[1].index_select(0, wave.indices[0])
        staged = patch.pools[0].index_select(0, slots)
        torch.testing.assert_close(staged, field.expand_as(staged), rtol=0, atol=0)
        for hidden, patched in observed["patches"]:
            torch.testing.assert_close(patched, hidden + field.expand_as(hidden).to(hidden.dtype), rtol=0, atol=0)
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
        patch.object(ProgramWave, "layer", layer),
        patch.object(VllmActivationPatch, "patch", patching),
        patch.object(ProgramWave, "capture", capture),
        patch.object(torch.cuda.CUDAGraph, "replay", replay),
        patch.object(VllmCustomForward, "run", run),
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


def produce_frames(feed, source, events, teardown):
    # Preemption deliberately closes the consumer while this producer still owns pages.
    with suppress(BrokenPipeError, _ConsumerClosed) if teardown is not None else nullcontext():
        previous = source[0].cuda()
        try:
            for index, (kind, frame) in enumerate(zip(events, source, strict=True)):
                if index == 1 and teardown is not None:
                    assert teardown.wait(120), "consumer did not finish teardown"
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


def run_video(workload, events, steps, config=None, *, consumer_fails=False):
    from vllm import SamplingParams

    context = mp.get_context("spawn")
    teardown = context.Event() if consumer_fails else None
    process = None
    try:
        with pllm.make_llm(config=config or model_config(), workload=workload) as llm:
            process = context.Process(
                target=produce_frames,
                args=(llm.feed("features"), frames(len(events)), events, teardown),
            )
            process.start()
            outputs = llm.generate(
                ["Describe the scene in this video frame:"],
                SamplingParams(max_tokens=steps, temperature=0.0, ignore_eos=True),
            )
            assert len(outputs[0].outputs[0].token_ids) == steps
            return llm.diagnostics(), outputs[0].outputs[0].text
    finally:
        if teardown is not None:
            teardown.set()
        if process is not None:
            process.join(120)
            if process.is_alive():
                process.kill()
                process.join()
                pytest.fail("feed owner did not finish after consumer teardown")
            assert process.exitcode == 0


@dataclass(frozen=True)
class RecirculatingVideo(VideoWorkload, RecirculationWorkload):
    """Compose the video ActivationPatch and an ordinary user-authored custom forward."""


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

    before_execute = VllmSession.before_execute

    def preempt(session, scheduler_output):
        if session.next_step_id == 1:
            assert session.diagnostics().update_count == 1
            scheduler_output.preempted_req_ids = set(session._identities)
        return before_execute(session, scheduler_output)

    with patch.object(VllmSession, "before_execute", preempt), pytest.raises(RuntimeError, match="preempt"):
        run_video(VideoWorkload(grid_size=2), ("publish", "publish"), 2, consumer_fails=True)
