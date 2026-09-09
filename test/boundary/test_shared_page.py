"""Evidence 16: real shared pages, exact versions and remote completion retention."""

from dataclasses import replace
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing as mp
from multiprocessing.connection import Connection
import os
from unittest.mock import patch

import pytest
import torch

from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.authoring import CausalTrainBatch, StateTensorUpdate
from pllm.execution.shared_tensor import CpuFileStorage, SharedTensor
from pllm.execution.ptts import page_bundle
from pllm.execution.ptts.page_bundle import PageAccess, PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts, compile_page_bundle_plan
from pllm.execution.host_runtime import HostRuntimeFacts, CompletionCapture
from pllm.contracts.feedback import Completion, CompletionBatch, FinishReason
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.semantic_graph import (
    AnchorSymbol, ExecutionKind, ImplementationId, SemanticEffect, SemanticGraph, SemanticState,
    SemanticTask, StateEffectKind, StateResource, StateScope, TaskImplementationSpec, TaskSymbol, TimingConstraint, WorkloadId,
)
from pllm.contracts.tensor_layout import (
    LayoutFieldId, MutationGranularity, StateTensorLayout, TensorDType, TensorFieldSpec, TensorLayoutRequest, TensorLayoutSpec,
)


def _plan(max_admitted_lineages: int = 1) -> PlanBundle:
    resource = StateResource("input")
    graph = SemanticGraph(WorkloadId("shared_page"), (SemanticTask(TaskSymbol("read"),
        TimingConstraint(AnchorSymbol("host.input.ready"), AnchorSymbol("host.next")), (), (),
        (TaskImplementationSpec(ImplementationId("test.read@1"), ExecutionKind.NATIVE_TASK),),
        (SemanticEffect(resource, StateEffectKind.READ),)),), "a" * 64,
        states=(SemanticState(resource, StateScope.INSTANCE, ()),))
    layout = TensorLayoutRequest((StateTensorLayout(resource, TensorLayoutSpec("test.shared@1", (
        TensorFieldSpec(LayoutFieldId(0), (2, 3), TensorDType.FLOAT64, 16, MutationGranularity.WHOLE_TENSOR),
    ))),))
    return compile_page_bundle_plan((graph, layout), HostRuntimeFacts(torch.device("cpu")).timeline(),
        (PageBundleProviderFacts("test.shared@1", 3 * max_admitted_lineages, shared_memory=True),), max_admitted_lineages, 65536, torch.device("cpu"))


def _reader(control: Connection, data: Connection, wire: str, mode: str) -> None:
    try:
        plan = decode_boundary(wire, PlanBundle)
        inodes = []
        native_map = page_bundle.map_cpu_storage

        def inspect_mapping(fd, size, dtype, offset, shape):
            inodes.append(os.fstat(fd).st_ino)
            return native_map(fd, size, dtype, offset, shape)

        with patch.object(page_bundle, "map_cpu_storage", inspect_mapping):
            borrowed = PageBundleArena.receive_field(plan, data)
        control.send_bytes(json.dumps(inodes).encode())
        assert control.recv_bytes() == b"read"
        torch.testing.assert_close(borrowed.tensor, torch.arange(6, dtype=torch.float64).reshape(2, 3))
        if mode == "bad_ack":
            data.send_bytes(b"another export")
        elif mode == "eof":
            data.close()
        else:
            borrowed.close()
            with pytest.raises(RuntimeError, match="closed"):
                borrowed.tensor
        control.send_bytes(b"read-finished")
        assert control.recv_bytes() == b"exit"
    finally:
        data.close()
        control.close()


def test_closed_mp_codec_round_trips_registered_payload_and_shared_tensors() -> None:
    with ExitStack() as lifetime:
        source = torch.arange(6, dtype=torch.float32).reshape(2, 3).T
        descriptor = SharedTensor.export(source, lifetime)
        for value, target in ((CausalTrainBatch(descriptor, descriptor), CausalTrainBatch[SharedTensor]),
                              (StateTensorUpdate((descriptor,)), StateTensorUpdate[SharedTensor])):
            assert decode_boundary(encode_boundary(value), target) == value
        mapped = descriptor.map(os.getpid())
        storage = descriptor.storage
        assert isinstance(storage, CpuFileStorage)
        torch.testing.assert_close(mapped, source)
        for bad in (replace(descriptor, process_id=0), replace(descriptor, storage=replace(storage, inode=0)),
                    replace(descriptor, field=replace(descriptor.field, shape=(6, 2)))):
            with pytest.raises(ValueError):
                bad.map(os.getpid())
        with pytest.raises(OSError):
            replace(descriptor, storage=replace(storage, descriptor=2**30)).map(os.getpid())
    # The mapping, not an FD or remote version ledger, owns this non-recycled output.
    mapped.fill_(7)
    assert torch.all(mapped == 7) and not torch.all(source == 7)
    identities = (RequestIdentity("a", 9, 0), RequestIdentity("b", 7, 0), RequestIdentity("c", 11, 0))
    capture = CompletionCapture(identities, mapped)
    with pytest.raises(RuntimeError, match="missing"):
        capture.require_complete()
    with pytest.raises(ValueError, match="sampling rows"):
        capture.write(identities, torch.ones(3, 1))
    capture.write((identities[2], identities[0]), torch.tensor([[2., 2.], [0., 0.]]))
    capture.write((identities[1], identities[0]), torch.tensor([[1., 1.], [3., 3.]]))
    batch = CompletionBatch(tuple(Completion(identity, 0, "feedback", (1,), "1", FinishReason.LENGTH)
        for identity in identities))
    assert decode_boundary(encode_boundary(batch), CompletionBatch) == batch
    for field, malformed in (("output_index", True), ("token_ids", [False]),
                              ("identity", "not-an-identity"), ("finish_reason", "unknown")):
        wire = json.loads(encode_boundary(batch))
        wire["entries"][0][field] = malformed
        with pytest.raises(ValueError, match="boundary"):
            decode_boundary(json.dumps(wire), CompletionBatch)
    completed = capture.finish(batch)
    assert completed.hidden is mapped and completed.completions is batch
    torch.testing.assert_close(mapped, torch.tensor([[3., 3.], [1., 1.], [2., 2.]]))
    plan = _plan()
    context = mp.get_context("spawn")
    for mode in ("complete", "bad_ack", "eof"):
        arena = PageBundleArena(plan)
        initial_capacity = arena.free_pages
        write = arena.prepare_admission(("input",))
        arena.write_field(write, 0, LayoutFieldId(0)).copy_(torch.arange(6).reshape(2, 3))
        publication = arena.publish(write)
        arena.settle(publication)
        ref, = publication.after_refs
        pin = arena.pin_exact((ref,))
        readers = []
        try:
            for _ in range(2 if mode == "complete" else 1):
                parent_control, child_control = context.Pipe()
                parent_data, child_data = context.Pipe()
                process = context.Process(target=_reader, args=(child_control, child_data, encode_boundary(plan), mode))
                process.start()
                child_control.close()
                child_data.close()
                readers.append((process, parent_control, parent_data))
                with pytest.raises(RuntimeError, match="issued"):
                    arena.send_field(replace(pin), ref, LayoutFieldId(0), parent_data, process.pid)
                descriptor = arena.send_field(pin, ref, LayoutFieldId(0), parent_data, process.pid)
                assert descriptor.ref is ref and descriptor.page is arena.root(ref).pages[0]
                assert parent_control.poll(20)
                # OS inode equality demonstrates one backing allocation, not a cloned tensor.
                inode, = json.loads(parent_control.recv_bytes())
                assert inode == os.fstat(arena._pools["input"].shared_files[LayoutFieldId(0)].fileno()).st_ino
            update = arena.prepare_transition((ref,))
            arena.write_field(update, 0, LayoutFieldId(0)).fill_(99)
            changed = arena.publish(update)
            arena.settle(changed)
            release = arena.release_pin(pin)
            held = arena.pin_exact(changed.after_refs)
            changed = arena.publish(arena.prepare_transition(changed.after_refs))
            arena.settle(changed)
            # Pressure must reject, not wait on remote readers awaiting our command.
            with pytest.raises(RuntimeError, match="live owners"):
                arena.prepare_transition(changed.after_refs)
            arena.wait_release(arena.release_pin(held))
            arena.retire(changed.after_refs)
            assert release.pin is pin and not arena.release_complete(release)
            assert arena.active_lineages == arena.pending_pin_releases == 1
            for index, (_, control, _) in enumerate(readers):
                control.send_bytes(b"read")
                assert control.poll(20) and control.recv_bytes() == b"read-finished"
                if mode == "complete":
                    assert arena.release_complete(release) == (index == len(readers) - 1)
                else:
                    with pytest.raises(RuntimeError, match="completion proof"):
                        arena.release_complete(release)
                    assert arena.pending_pin_releases == 1
                    assert arena.root(ref).ref is ref
            for invalid in (replace(descriptor, owner_id=0), replace(descriptor, access=PageAccess.WRITE),
                replace(descriptor, storage_offset=10**9), replace(descriptor, resource="unsigned"),
                replace(descriptor, compilation_signature="b" * 64)):
                sender, receiver = context.Pipe()
                try:
                    sender.send_bytes(encode_boundary(invalid).encode())
                    with pytest.raises(ValueError):
                        PageBundleArena.receive_field(plan, receiver)
                finally:
                    sender.close()
                    receiver.close()
            if mode == "complete":
                arena.wait_release(release)
                assert arena.active_lineages == arena.pending_pin_releases == 0
                assert arena.free_pages == initial_capacity
            for process, control, _ in readers:
                control.send_bytes(b"exit")
                process.join(10)
                assert process.exitcode == 0
        finally:
            for process, control, data in readers:
                control.close()
                data.close()
                if process.is_alive():
                    process.terminate()
                    process.join(10)
                process.close()


def _writer(control: Connection, data: Connection, wire: str, value: int, bad_ack: bool) -> None:
    try:
        plan = decode_boundary(wire, PlanBundle)
        borrowed = PageBundleArena.receive_field(plan, data, access=PageAccess.WRITE)
        control.send_bytes(b"prepared")
        assert control.recv_bytes() == b"write"
        borrowed.tensor.fill_(value)
        if bad_ack:
            data.send_bytes(b"wrong operation")
        else:
            borrowed.close()
        control.send_bytes(b"finished")
        assert control.recv_bytes() == b"exit"
    finally:
        data.close()
        control.close()


def test_mp_transaction_commit_abort_and_quorum_failure_are_atomic() -> None:
    plan = _plan(2)
    context = mp.get_context("spawn")
    for outcome in ("commit", "abort", "bad_ack"):
        arena = PageBundleArena(plan)
        admission = arena.prepare_admission(("input", "input"))
        initial = arena.publish(admission)
        arena.settle(initial)
        before = initial.after_refs
        original_pin = arena.pin_exact(before)
        write = arena.prepare_transition(before)
        workers = []
        abort_pool = ThreadPoolExecutor(max_workers=1)
        try:
            for index in range(2):
                control, child_control = context.Pipe()
                data, child_data = context.Pipe()
                process = context.Process(target=_writer, args=(child_control, child_data,
                    encode_boundary(plan), 17 + index, outcome == "bad_ack" and index == 1))
                process.start()
                child_control.close()
                child_data.close()
                workers.append((process, control, data))
                with pytest.raises(RuntimeError, match="issued"):
                    arena.send_write_field(replace(write), index, LayoutFieldId(0), data, process.pid)
                descriptor = arena.send_write_field(write, index, LayoutFieldId(0), data, process.pid)
                assert descriptor.ref is write.after_refs[index]
                assert descriptor.owner_id == write.operation_id
                assert control.poll(20) and control.recv_bytes() == b"prepared"
                with pytest.raises(RuntimeError, match="exclusively"):
                    arena.write_field(write, index, LayoutFieldId(0))
                with pytest.raises(RuntimeError, match="already granted"):
                    arena.send_write_field(write, index, LayoutFieldId(0), data, process.pid)
            with pytest.raises(RuntimeError, match="quorum"):
                arena.publish(write)
            abort = abort_pool.submit(arena.abort, write) if outcome == "abort" else None
            for index, (_, control, _) in enumerate(workers):
                control.send_bytes(b"write")
                assert control.poll(20) and control.recv_bytes() == b"finished"
                if abort is not None:
                    if index == 0:
                        assert not abort.done()
                    continue
                if index == 0:
                    with pytest.raises(RuntimeError, match="quorum"):
                        arena.publish(write)
                assert tuple(arena.latest(ref.lineage_id) for ref in before) == before
                for ref in before:
                    torch.testing.assert_close(arena.borrow_field(original_pin, ref, LayoutFieldId(0)),
                        torch.zeros(2, 3, dtype=torch.float64))
            if outcome == "bad_ack":
                for operation in (arena.publish, arena.abort):
                    with pytest.raises(RuntimeError, match="completion proof"):
                        operation(write)
                assert tuple(arena.latest(ref.lineage_id) for ref in before) == before
                with pytest.raises(RuntimeError, match="pending write"):
                    arena.prepare_transition(before)
            else:
                if outcome == "commit":
                    publication = arena.publish(write)
                    assert publication.before_refs is write.before_refs
                    assert publication.after_refs is write.after_refs
                    arena.settle(publication)
                    latest = publication.after_refs
                    pin = arena.pin_exact(latest)
                    for index, ref in enumerate(latest):
                        torch.testing.assert_close(arena.borrow_field(pin, ref, LayoutFieldId(0)),
                            torch.full((2, 3), 17 + index, dtype=torch.float64))
                    arena.wait_release(arena.release_pin(pin))
                else:
                    assert abort is not None
                    abort.result(timeout=10)
                    latest = before
                    assert tuple(arena.latest(ref.lineage_id) for ref in before) == before
                    for ref in before:
                        torch.testing.assert_close(arena.borrow_field(original_pin, ref, LayoutFieldId(0)),
                            torch.zeros(2, 3, dtype=torch.float64))
                for operation in (arena.publish, arena.abort):
                    with pytest.raises(RuntimeError, match="consumed"):
                        operation(write)
                arena.wait_release(arena.release_pin(original_pin))
                arena.retire(latest)
                assert arena.active_lineages == arena.pending_pin_releases == 0
                assert arena.free_pages == 6
            for process, control, _ in workers:
                control.send_bytes(b"exit")
                process.join(10)
                assert process.exitcode == 0
        finally:
            for process, control, data in workers:
                control.close()
                data.close()
                if process.is_alive():
                    process.terminate()
                    process.join(10)
                process.close()
            abort_pool.shutdown(wait=True)
