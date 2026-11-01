"""The CUDA page arena: copy-on-write versions, exact pins, and reclamation gated on reader events.

The second half shares a pinned page with a spawned process over CUDA IPC; the page
is reclaimed only after the reader's ACK proves its queued work finished.
"""

import multiprocessing as mp
from dataclasses import replace

import pytest
import torch
import triton
import triton.language as tl

from pllm.config import ExecutionOptions
from pllm.contracts.codec import decode_boundary, encode_boundary
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenSite
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding
from pllm.contracts.semantic_graph import AnchorSymbol, SchedulerTimeline, TimelineAnchor
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.ptts.page_borrow import receive_field
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.ptts.state_refs import StateRefs
from pllm.kernels.adamw import AdamWConfig, AdamWFields, apply_adamw
from pllm.testing import backend_facts
from pllm.workloads.esamp_request import ESampRequestConfig, ESampRequestWorkload


def _plan():
    bindings = tuple(
        PackedHiddenResourceBinding(
            str(layer), HiddenLayerKind.INDEX, HiddenSite.BLOCK_OUTPUT, f"model.layers[{layer}]"
        )
        for layer in (0, -1)
    )
    anchors = ("request.prepare", "model.execute", "sampler.pre_draw", "token.commit", "next.prepare")
    timeline = SchedulerTimeline(
        tuple(TimelineAnchor(AnchorSymbol(name), index) for index, name in enumerate(anchors)),
        ("compute/cuda@1", "state/dense_page_cuda@1"),
    )
    provider = PageBundleProviderFacts("test.page_bundle@1", 6, StateDeviceKind.CUDA, 0)
    facts = backend_facts(4, bindings, timeline, provider)
    workload = ESampRequestWorkload(ESampRequestConfig(rank=2))
    return workload.plan_backend(*workload.specialize(facts, ExecutionOptions())).plan


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_ptts_waits_for_exact_fence_before_reclaim() -> None:
    arena = PageBundleArena(_plan())
    device = torch.device("cuda", 0)
    writer, reader = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    write = arena.prepare_admission(("distiller",) * 2, writer)
    arena.abort(write)
    assert arena.free_pages == 6
    write = arena.prepare_admission(("distiller",) * 2, writer)
    roots = arena.writable_roots(write)
    with torch.cuda.stream(writer):
        initial = tuple(
            torch.arange(root[0].numel(), device=device, dtype=torch.float32) / 10 + index
            for index, root in enumerate(roots)
        )
        for root, value in zip(roots, initial, strict=True):
            root[0].copy_(value)
    admitted = arena.publish(write)
    pin = arena.pin_exact(admitted.after_refs, reader)
    borrowed = arena.borrow_field(pin, admitted.after_refs[0], LayoutFieldId(0))
    original = arena.snapshot(pin, admitted.after_refs[0])
    update = arena.prepare_transition(admitted.after_refs, writer)
    fields = AdamWFields(*(LayoutFieldId(index) for index in range(5)))
    with torch.cuda.stream(writer):
        gradients = tuple(
            torch.full_like(value, gradient) for value, gradient in zip(initial, (0.25, -0.5), strict=True)
        )
        apply_adamw(arena.writable_roots(update), fields, AdamWConfig(0.01, (0.9, 0.999), 1e-8, 0.0), gradients)
    publication = arena.publish(update)
    updated = arena.pin_exact(publication.after_refs, reader)
    result = arena.snapshot(updated, publication.after_refs[0])
    # Old roots must remain live while their original consumer stream still runs.
    with torch.cuda.stream(reader):
        torch.cuda._sleep(100_000_000)
        consumed = borrowed.clone()
    release = arena.release_pin(pin)
    assert arena.pending_pin_releases == 0 and arena.free_pages == 4
    with pytest.raises(AssertionError):  # a released pin admits no new reads
        arena.snapshot(pin, admitted.after_refs[0])
    pressure = arena.publish(arena.prepare_transition(publication.after_refs, writer))
    # Reusing the released reader's pages does not wait on the host: the writer's stream waits for the reader's
    # fence on the GPU, so the reader still sees the old value (checked below).
    update = arena.prepare_transition(pressure.after_refs, writer)
    arena.wait_release(release)
    arena.abort(update)
    assert arena.free_pages == 2
    torch.testing.assert_close(original[0], initial[0])
    torch.testing.assert_close(consumed, initial[0])
    torch.testing.assert_close(result[0], initial[0] - 0.01)
    assert result[2].item() == 1
    arena.retire(pressure.after_refs)
    replacement = arena.publish(arena.prepare_admission(("distiller",) * 2, writer))
    torch.testing.assert_close(arena.snapshot(updated, publication.after_refs[0])[0], result[0])
    arena.wait_release(arena.release_pin(updated))
    arena.retire(replacement.after_refs)
    arena.collect_completed(wait=True)
    assert arena.free_pages == 6 and arena.active_lineages == 0
    _cuda_ipc_lifecycle()


def _cuda_ipc_worker(control, read_channel, wire, outcome):
    try:
        arena = PageBundleArena(decode_boundary(wire, PlanBundle), _owned_resources=())
        stream = torch.cuda.Stream(device=torch.device("cuda", 0))
        with torch.cuda.stream(stream):
            read = receive_field(arena.plan, arena.plan_digest, read_channel, stream=stream)
            storage = read.descriptor.cuda_storage
            assert storage is not None and read.tensor.is_cuda
            with pytest.raises(ValueError):  # a descriptor for another GPU cannot be mapped
                replace(storage, device_uuid="different GPU").map(
                    storage.size_bytes,
                    read.tensor.dtype,
                    read.descriptor.storage_offset,
                    tuple(read.tensor.shape),
                    stream,
                )
            control.send_bytes(b"prepared")
            assert control.recv_bytes() == b"run"
            expected = torch.arange(read.tensor.numel(), device=stream.device, dtype=read.tensor.dtype)
            torch.testing.assert_close(read.tensor, expected)
            torch.cuda._sleep(100_000_000)
            consumed = read.tensor + 17
            done = torch.cuda.Event()
            done.record(stream)
            assert not done.query()
            control.send_bytes(b"queued")
            assert control.recv_bytes() == b"close"
            if outcome == "complete":
                read.close()
                assert done.query()
                torch.testing.assert_close(consumed, expected + 17)
            elif outcome == "bad_ack":
                read_channel.send_bytes(b"another export")
            else:
                read_channel.close()
            control.send_bytes(b"completed")
        assert control.recv_bytes() == b"exit"
    finally:
        control.close()
        read_channel.close()


def _cuda_ipc_lifecycle():
    context = mp.get_context("spawn")
    plan = _plan()
    for outcome in ("complete", "bad_ack", "eof"):
        arena = PageBundleArena(plan)
        stream = torch.cuda.Stream(device=arena.device)
        initial = arena.prepare_admission(("distiller",), stream)
        with torch.cuda.stream(stream):
            field = arena.write_field(initial, 0, LayoutFieldId(0))
            expected = torch.arange(field.numel(), device=arena.device, dtype=field.dtype)
            field.copy_(expected)
            del field
        initial = arena.publish(initial)
        (ref,) = initial.after_refs
        pin = arena.pin_exact(StateRefs.of((ref,)), stream)
        control, child_control = context.Pipe()
        read_channel, child_read = context.Pipe()
        process = context.Process(
            target=_cuda_ipc_worker, args=(child_control, child_read, encode_boundary(plan), outcome)
        )
        process.start()
        child_control.close()
        child_read.close()
        try:
            descriptor = arena.send_field(pin, ref, LayoutFieldId(0), read_channel, process.pid)
            assert descriptor.ref == ref and descriptor.cuda_storage is not None
            assert control.poll(20) and control.recv_bytes() == b"prepared"
            sender, receiver = context.Pipe()
            try:  # a descriptor for a different compiled plan is rejected
                sender.send_bytes(encode_boundary(replace(descriptor, plan_digest="b" * 64)).encode())
                with pytest.raises(ValueError):
                    receive_field(arena.plan, arena.plan_digest, receiver, stream=stream)
            finally:
                sender.close()
                receiver.close()
            # A local publication must not overwrite the remote reader's exact old version.
            write = arena.prepare_transition(StateRefs.of((ref,)), stream, fields=((LayoutFieldId(0),),))
            with torch.cuda.stream(stream):
                arena.write_field(write, 0, LayoutFieldId(0)).add_(99)
            publication = arena.publish(write)
            released = arena.release_pin(pin)
            arena.retire(publication.after_refs)
            assert not arena.release_complete(released)
            control.send_bytes(b"run")
            assert control.poll(20) and control.recv_bytes() == b"queued"
            assert not arena.release_complete(released)
            control.send_bytes(b"close")
            assert control.poll(20) and control.recv_bytes() == b"completed"
            if outcome == "complete":
                arena.wait_release(released)
                arena.collect_completed(wait=True)
                assert arena.active_lineages == arena.pending_pin_releases == 0 and arena.free_pages == 6
            else:
                with pytest.raises(RuntimeError):  # no completion proof: the page stays owned
                    arena.wait_release(released)
                assert arena.root(ref).ref == ref and arena.pending_pin_releases == 1
            control.send_bytes(b"exit")
            process.join(10)
            assert process.exitcode == 0
        finally:
            control.close()
            read_channel.close()
            if process.is_alive():
                process.terminate()
                process.join(10)
            process.close()


@triton.jit
def _wait_for_gate(gate, expired, budget):
    """Hold the launching stream until the host sets the pinned `gate`, or give up after `budget` reads."""
    spins = 0
    while (tl.load(gate, volatile=True) == 0) & (spins < budget):
        spins += 1
    tl.store(expired, spins == budget)


def test_bounded_staging_and_diagnostics_wait_for_completion_without_host_waits():
    """Busy host storage backpressures admission; a device snapshot never exposes partial counters."""
    from unittest.mock import patch

    from pllm.execution.admission import WindowBackpressure
    from pllm.execution.cuda_stream import DeviceCounters, PinnedStaging

    stream = torch.cuda.Stream()
    staging = PinnedStaging(capacity=8, slots=1)
    destination = torch.empty(8, dtype=torch.int64, device="cuda")
    counters = DeviceCounters(destination.device)
    increments = torch.tensor([3, 17], device="cuda")
    # The copy and the counter update stay queued behind the gate until the checks below have run. Their kernels
    # are loaded first: a lazy CUDA module load while the gate kernel spins would block the host until it ends.
    gate = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    expired = torch.zeros(1, dtype=torch.bool, device="cuda")
    DeviceCounters(destination.device).add(increments, stream)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        host = staging.take(8)
        address = host.data_ptr()
        host.fill_(23)
        _wait_for_gate[(1,)](gate, expired, 5_000_000)  # each read of host memory takes ~2 us: ~10 s
        staging.copy_to(destination)
        counters.add(increments, stream)
    try:
        with (
            patch.object(torch.cuda.Event, "synchronize", side_effect=AssertionError("host wait")),
            patch.object(torch.cuda.Stream, "synchronize", side_effect=AssertionError("host wait")),
        ):
            with pytest.raises(WindowBackpressure):
                staging.take(8)
            assert counters.read() == (0, 0)
    finally:
        gate[0] = 1
        stream.synchronize()
        assert not expired.item(), "the gate expired: a kernel launched behind it was not loaded beforehand"
    torch.testing.assert_close(destination, torch.full_like(destination, 23))
    assert staging.take(8).data_ptr() == address
    assert counters.read() == (3, 17)
