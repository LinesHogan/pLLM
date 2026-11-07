"""CUDA feed selection, source lifetime, backpressure and terminal grant retention."""

import multiprocessing as mp
import pickle
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch

import pllm
from pllm.contracts.packed_rows import PackedPhaseRowBatch, PackedSamplerRowBatch
from pllm.execution.ptts.page_borrow import BorrowedPage
from pllm.execution.state_feed import FeedReceiver, ReadonlyStepEnd, receive_field
from pllm.execution.step_input import StepInput
from test.workloads.test_state_feed import FeedWorkload, facts_for

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA IPC requires a GPU")]


def producer(feed, probe):
    arena = None
    try:
        pages = []
        for value in (1, 2, 3):
            tensor = torch.full((2,), float(value), device="cuda")
            feed.publish(value=tensor)
            with torch.cuda.stream(torch.cuda.Stream()):
                tensor.zero_()
            arena = feed.arena
            pages.append(arena.root(feed.current_ref).pages)
        with pytest.raises(RuntimeError, match="before publishing"):
            pickle.dumps(feed)
        probe.send(("ready", pages))
        feed.publish(value=torch.full((2,), 4.0, device="cuda"))
        probe.send(("fourth", arena.root(feed.current_ref).pages))
        feed.end()
    finally:
        feed.close()
    assert arena.active_states == arena.pending_pin_releases == 0
    probe.send(("closed", None))
    probe.close()


def setup_feed(workload):
    execution = workload.compile_backend(facts_for(torch.device("cuda", 0)), pllm.ExecutionOptions()).execution
    receiver = FeedReceiver(execution.arena, execution.feeds[0])
    execution.admit((), torch.cuda.current_stream())
    return execution, receiver


def step_rows(step):
    rows = PackedPhaseRowBatch.from_rows(step, 0, ())
    return PackedSamplerRowBatch.from_links(rows, ())


def join_producer(process, receiver, execution):
    execution.discard_borrows()
    receiver.close()
    process.join(30)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("producer did not wake after consumer teardown")
    assert process.exitcode == 0


@pytest.mark.parametrize("order,versions", [("ordered", (1, 2, 3, 4)), ("latest", (3, 4))])
def test_feed_selects_complete_versions_and_reclaims_only_after_completion(order, versions):
    execution, receiver = setup_feed(FeedWorkload(order))
    context = mp.get_context("spawn")
    probe, child = context.Pipe()
    feed = receiver.take("source")
    assert feed.arena is None
    process = context.Process(target=producer, args=(feed, child))
    process.start()
    assert feed.moved and feed.arena is None
    child.close()
    try:
        assert probe.poll(30)
        kind, pages = probe.recv()
        assert kind == "ready"
        assert not probe.poll(0.1), "fourth publication must block while all three slots are in flight"
        stream = torch.cuda.current_stream()
        fourth = []
        close = BorrowedPage.close

        def close_after_fourth(page):
            # Version 2's read ACK waits until version 4 is placed, so only version 1's page can be free then;
            # otherwise the producer may see both ACKs (LATEST closes 1 and 2 together) and take version 2's page.
            if page.descriptor.ref.version_id == 2 and not fourth:
                assert probe.poll(30), "the producer did not publish version 4 after version 1's ACK"
                fourth.append(probe.recv())
            close(page)

        with patch.object(BorrowedPage, "close", close_after_fourth):
            for step, version in enumerate(versions):
                rows = step_rows(step)
                incoming = receiver(rows, stream)
                (root,) = incoming.roots
                assert root.ref.version_id == version
                assert root.pages[0].descriptor.page == pages[version - 1][0] if version <= 3 else True
                tensor = root.pages[0].tensor
                torch.testing.assert_close(tensor, torch.full_like(tensor, float(version)))
                assert tensor.storage_offset() == root.pages[0].descriptor.storage_offset
                assert tensor.untyped_storage().nbytes() == root.pages[0].descriptor.cuda_storage.size_bytes
                del tensor
                execution.stage_readonly(incoming.roots)
                event = torch.cuda.Event()
                torch.cuda._sleep(100_000_000)
                event.record(stream)
                receipt = execution.step(StepInput(rows.row_batch, rows), stream)
                assert event.query(), "read ACK must follow CUDA completion"
                torch.testing.assert_close(
                    execution.snapshot(receipt.publication.after_refs[0], stream)[0],
                    torch.full((2,), float(version * 2), device="cuda"),
                )
        assert isinstance(receiver(step_rows(len(versions)), stream), ReadonlyStepEnd)
        ((kind, reused),) = fourth
        assert kind == "fourth" and reused[0].slot_id == pages[0][0].slot_id
        assert reused[0].generation == pages[0][0].generation + 1
        assert probe.poll(30) and probe.recv()[0] == "closed"
        execution.retire((), wait=True)
    finally:
        join_producer(process, receiver, execution)
        probe.close()


@dataclass(frozen=True)
class TwoFieldFeed(FeedWorkload):
    @property
    def source(self) -> pllm.InstanceState:
        return pllm.InstanceState({"value": (2,), "extra": (2,)}, feed=self.order)


def retained_producer(feed, probe):
    try:
        tensor = torch.ones(2, device="cuda")
        feed.publish(value=tensor, extra=tensor)
        arena = feed.arena
        feed.end()
        probe.send("retained")
        assert arena.usage().pinned_versions > 0
    finally:
        feed.close()
    probe.send("closed")
    probe.close()


def test_feed_partial_receive_is_fail_stopped_and_owner_waits_for_teardown():
    execution, receiver = setup_feed(TwoFieldFeed())
    context = mp.get_context("spawn")
    probe, child = context.Pipe()
    process = context.Process(target=retained_producer, args=(receiver.take("source"), child))
    process.start()
    child.close()
    count = 0

    def partial_receive(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError("injected descriptor failure")
        return receive_field(*args, **kwargs)

    try:
        assert probe.poll(30) and probe.recv() == "retained"
        with (
            patch("pllm.execution.state_feed.receive_field", partial_receive),
            pytest.raises(ValueError, match="descriptor"),
        ):
            receiver(step_rows(0), torch.cuda.current_stream())
        assert len(receiver.partial) == 1 and receiver.failed
        assert not probe.poll(0.1), "owner cannot close before the partial import is discarded"
        with pytest.raises(RuntimeError, match="failed feed"):
            receiver(step_rows(1), torch.cuda.current_stream())
    finally:
        join_producer(process, receiver, execution)
        assert probe.poll(30) and probe.recv() == "closed"
        probe.close()


def test_publish_rejects_invalid_user_fields_without_allocating_pages():
    execution, receiver = setup_feed(FeedWorkload())
    feed = receiver.take("source")
    try:
        for values in (
            {},
            {"other": torch.ones(2, device="cuda")},
            {"value": torch.ones(3, device="cuda")},
            {"value": torch.ones(2, dtype=torch.float64, device="cuda")},
            {"value": torch.ones(2)},
        ):
            with pytest.raises(ValueError):
                feed.publish(**values)
            assert feed.arena is None and not feed.failed
        feed.end()
        assert isinstance(receiver(step_rows(0), torch.cuda.current_stream()), ReadonlyStepEnd)
        feed.close()
    finally:
        execution.retire((), wait=True)
        receiver.close()


def bad_ack_producer(feed, probe):
    try:
        for _ in range(3):
            feed.publish(value=torch.ones(2, device="cuda"))
        probe.send("ready")
        with pytest.raises(RuntimeError, match="completion proof"):
            feed.publish(value=torch.ones(2, device="cuda"))
        assert feed.failed and feed.arena.pending_releases
        assert feed.arena.root(feed.current_ref).ref == feed.current_ref
        with pytest.raises(RuntimeError, match="fail-stopped"):
            feed.publish(value=torch.ones(2, device="cuda"))
        probe.send("failed-with-owner")
    finally:
        feed.close()
    probe.send("closed")
    probe.close()


def test_bad_ack_retains_failed_owner_and_cancels_queued_publications():
    execution, receiver = setup_feed(FeedWorkload())
    context = mp.get_context("spawn")
    probe, child = context.Pipe()
    process = context.Process(target=bad_ack_producer, args=(receiver.take("source"), child))
    process.start()
    child.close()
    try:
        assert probe.poll(30) and probe.recv() == "ready"
        incoming = receiver(step_rows(0), torch.cuda.current_stream())
        incoming.roots[0].pages[0].connection.send_bytes(b"wrong export")
        assert probe.poll(30) and probe.recv() == "failed-with-owner"
        # The cancellation can race this test's probe; wait for its actual control record.
        from multiprocessing.connection import wait

        wait((receiver.control,))
        with pytest.raises(RuntimeError, match="cancelled"):
            receiver(step_rows(1), torch.cuda.current_stream())
        assert receiver.failed and not probe.poll(0.1)
    finally:
        join_producer(process, receiver, execution)
        assert probe.poll(30) and probe.recv() == "closed"
        probe.close()


def test_unused_feed_closes_without_starting_a_producer():
    execution, receiver = setup_feed(FeedWorkload())
    writer = receiver.writer
    assert writer.arena is None
    receiver.close()
    receiver.close()
    assert writer.closed and writer.arena is None
    execution.retire((), wait=True)


def disconnecting_consumer(probe):
    execution, receiver = setup_feed(FeedWorkload())
    probe.send(receiver.take("source"))
    receiver(step_rows(0), torch.cuda.current_stream())
    probe.send("borrowed")
    assert probe.recv() == "disconnect"
    receiver.control.close()  # EOF alone must not allow the producer to destroy its arena.
    probe.send("disconnected")
    assert probe.recv() == "exit"
    receiver.current.discard()
    execution.retire((), wait=True)
    probe.close()


def test_control_eof_retains_owner_until_the_consumer_process_exits():
    from concurrent.futures import ThreadPoolExecutor

    context = mp.get_context("spawn")
    probe, child = context.Pipe()
    process = context.Process(target=disconnecting_consumer, args=(child,))
    process.start()
    child.close()
    assert probe.poll(30)
    feed = probe.recv()
    feed.publish(value=torch.ones(2, device="cuda"))
    arena = feed.arena
    assert probe.poll(30) and probe.recv() == "borrowed"
    probe.send("disconnect")
    assert probe.poll(30) and probe.recv() == "disconnected"
    with ThreadPoolExecutor(max_workers=1) as executor:
        closing = executor.submit(feed.close)
        try:
            from concurrent.futures import TimeoutError

            with pytest.raises(TimeoutError):
                closing.result(timeout=0.1)
            assert feed.arena is arena and arena.usage().pinned_versions > 0
        finally:
            probe.send("exit")
            process.join(30)
            if process.is_alive():
                process.kill()
                process.join()
        with pytest.raises(OSError):
            closing.result(timeout=30)
    assert process.exitcode == 0 and feed.arena is None
    probe.close()
