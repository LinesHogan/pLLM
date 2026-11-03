"""Real video through the public feed: numerical output, CUDA aliases, replay and shutdown."""

import multiprocessing as mp
import os
from contextlib import contextmanager
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
from pllm.workloads.recirculation import RecirculationWorkload
from pllm.workloads.video import VideoWorkload, frame_features
from test.workloads.test_video import frames


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

            from pllm.backends.vllm.custom_forward_attention import attention_metadata

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
            torch.testing.assert_close(output[1], inputs[1] + expected, rtol=0, atol=0)
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

        # The patched MLP output joins the residual stream in the next norm; its second output is that stream.
        layers = llm.runner.model.model.layers
        following = layers[workload.layer + 1].input_layernorm if workload.layer + 1 < len(layers) else None
        (following or llm.runner.model.model.norm).register_forward_hook(check)
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

    def patching(activation, resource, layer, rows, packed):
        """Record a captured wave's MLP output of the patched layer before and after the patch."""
        apply = original_patch(activation, resource, layer, rows, packed)

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
        # Each lane's fields are staged into stable per-row buffers the captured wave gathers from.
        (buffer,) = holder["session"]._activation._row_fields[StateResource("inject"), 1]
        staged = buffer.index_select(0, wave.indices[0])
        assert buffer.data_ptr() != field.data_ptr()
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
        run_video(VideoWorkload(grid_size=2), ("publish", "publish"), 2)
