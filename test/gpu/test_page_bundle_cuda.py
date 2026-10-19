"""Evidence 13: actual CUDA PageBundle COW and exact event-gated reclamation."""

import multiprocessing as mp
from dataclasses import replace

import pytest
import torch

from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenSite
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding
from pllm.contracts.semantic_graph import AnchorSymbol, SchedulerTimeline, TimelineAnchor
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts, compile_page_bundle_plan
from pllm.kernels.adamw import AdamWConfig, AdamWFields, apply_adamw
from pllm.workloads.esamp_request import ESampRequestConfig, ESampRequestWorkload
from test.backend_facts import backend_facts


def _plan(shared=False, rank=2, memory_bytes=10000):
    bindings = tuple(
        PackedHiddenResourceBinding(
            str(layer), HiddenLayerKind.INDEX, HiddenSite.BLOCK_OUTPUT, f"model.layers[{layer}]"
        )
        for layer in (0, -1)
    )
    facts = backend_facts(
        4,
        bindings,
        SchedulerTimeline(
            tuple(
                TimelineAnchor(AnchorSymbol(name), index)
                for index, name in enumerate(("model.execute", "sampler.pre_draw", "token.commit", "next.prepare"))
            ),
            ("compute/cuda@1", "state/dense_page_cuda@1"),
        ),
        PageBundleProviderFacts("test.page_bundle@1", 6, StateDeviceKind.CUDA, 0, shared_memory=shared),
        memory_bytes=memory_bytes,
    )
    declaration = ESampRequestWorkload(ESampRequestConfig(rank=rank)).declaration(facts)
    return compile_page_bundle_plan(
        (declaration.graph, declaration.layouts),
        facts.timeline,
        facts.providers,
        facts.max_admitted_lineages,
        facts.state_memory_budget_bytes,
        facts.device,
    )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_ptts_waits_for_exact_fence_before_reclaim() -> None:
    arena = PageBundleArena(_plan())
    device = torch.device("cuda", 0)
    writer, reader = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    write = arena.prepare_admission(("request_distiller",) * 2, writer)
    arena.abort(write)
    assert arena.free_pages == 6
    write = arena.prepare_admission(("request_distiller",) * 2, writer)
    roots = arena.writable_roots(write)
    with torch.cuda.stream(writer):
        initial = tuple(
            torch.arange(root[0].numel(), device=device, dtype=torch.float32) / 10 + index
            for index, root in enumerate(roots)
        )
        for root, value in zip(roots, initial, strict=True):
            root[0].copy_(value)
    admitted = arena.publish(write)
    arena.settle(admitted)
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
    arena.settle(publication)
    updated = arena.pin_exact(publication.after_refs, reader)
    result = arena.snapshot(updated, publication.after_refs[0])
    # Old roots must remain live while their original consumer stream still runs.
    with torch.cuda.stream(reader):
        torch.cuda._sleep(100_000_000)
        consumed = borrowed.clone()
    release = arena.release_pin(pin)
    assert arena.pending_pin_releases == 1
    assert arena.free_pages == 2
    usage = arena.usage()
    assert usage.live_field_pages == usage.retiring_field_pages == usage.free_field_pages == 10
    assert usage.private_versions == 0 and usage.pending_receipts >= 1
    with pytest.raises(RuntimeError, match="no new reads"):
        arena.snapshot(pin, admitted.after_refs[0])
    with pytest.raises(RuntimeError, match="issued"):
        arena.wait_release(replace(release))
    addresses = arena.pool_addresses
    pressure = arena.publish(arena.prepare_transition(publication.after_refs, writer))
    arena.settle(pressure)
    # All six slots are owned until the handed-back reader fence completes.
    update = arena.prepare_transition(pressure.after_refs, writer)
    assert arena.release_complete(release)
    arena.abort(update)
    assert arena.pool_addresses == addresses and arena.free_pages == 2
    torch.testing.assert_close(original[0], initial[0])
    torch.testing.assert_close(consumed, initial[0])
    torch.testing.assert_close(result[0], initial[0] - 0.01)
    assert result[2].item() == 1
    arena.retire(pressure.after_refs)
    replacement = arena.publish(arena.prepare_admission(("request_distiller",) * 2, writer))
    arena.settle(replacement)
    torch.testing.assert_close(arena.snapshot(updated, publication.after_refs[0])[0], result[0])
    arena.wait_release(arena.release_pin(updated))
    arena.retire(replacement.after_refs)
    arena.collect_completed(wait=True)
    assert arena.free_pages == 6 and arena.active_lineages == 0
    _cuda_ipc_lifecycle()


def _cuda_ipc_worker(control, read_channel, wire, outcome):
    try:
        plan = decode_boundary(wire, PlanBundle)
        stream = torch.cuda.Stream(device=torch.device("cuda", 0))
        with torch.cuda.stream(stream):
            read = PageBundleArena.receive_field(plan, read_channel, stream=stream)
            storage = read.descriptor.cuda_storage
            assert storage is not None and read.tensor.is_cuda
            for invalid in (
                replace(storage, device_uuid="different GPU"),
                replace(storage, size_bytes=storage.size_bytes + 4),
                replace(storage, handle=""),
            ):
                with pytest.raises(ValueError):
                    invalid.map(
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
    plan = _plan(shared=True)
    for outcome in ("complete", "bad_ack", "eof"):
        arena = PageBundleArena(plan)
        stream = torch.cuda.Stream(device=arena.device)
        initial = arena.prepare_admission(("request_distiller",), stream)
        with torch.cuda.stream(stream):
            field = arena.write_field(initial, 0, LayoutFieldId(0))
            expected = torch.arange(field.numel(), device=arena.device, dtype=field.dtype)
            field.copy_(expected)
            del field
        initial = arena.publish(initial)
        arena.settle(initial)
        (ref,) = initial.after_refs
        pin = arena.pin_exact((ref,), stream)
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
            assert descriptor.ref is ref and descriptor.cuda_storage is not None
            assert control.poll(20) and control.recv_bytes() == b"prepared"
            for invalid in (
                replace(descriptor, owner_id=0),
                replace(descriptor, storage_offset=10**9),
                replace(descriptor, resource="unsigned"),
                replace(descriptor, compilation_signature="b" * 64),
            ):
                sender, receiver = context.Pipe()
                try:
                    sender.send_bytes(encode_boundary(invalid).encode())
                    with pytest.raises(ValueError):
                        PageBundleArena.receive_field(plan, receiver, stream=stream)
                finally:
                    sender.close()
                    receiver.close()
            # A local publication must not overwrite the remote reader's exact old version.
            write = arena.prepare_transition((ref,), stream, fields=((LayoutFieldId(0),),))
            with torch.cuda.stream(stream):
                arena.write_field(write, 0, LayoutFieldId(0)).add_(99)
            publication = arena.publish(write)
            arena.settle(publication)
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
                with pytest.raises(RuntimeError, match="completion proof"):
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
