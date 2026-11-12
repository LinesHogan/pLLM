"""Feed handles and step receivers reuse PTTS pages for one CUDA producer and consumer."""

from __future__ import annotations

import ctypes
import os
from collections import deque
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing import get_context
from multiprocessing.connection import Connection, wait
from multiprocessing.resource_sharer import DupFd

import torch

from pllm.contracts.codec import decode_boundary, encode_boundary
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.packed_rows import PackedSamplerRowBatch
from pllm.contracts.state_feed import FEED_SLOTS, FeedBinding, FeedOrder
from pllm.contracts.state_ref import StateRef
from pllm.execution.ptts.page_borrow import BorrowedPage, BorrowedStateRoot, receive_field
from pllm.execution.ptts.page_bundle import PageBundleArena, PageRelease
from pllm.execution.ptts.state_refs import StateRefs


@dataclass(frozen=True, slots=True)
class ReadonlyStepInput:
    """The feed versions one engine step received, handed to the session that consumes them."""

    step_id: int
    roots: tuple[BorrowedStateRoot, ...]


@dataclass(frozen=True, slots=True)
class ReadonlyStepAbsent:
    step_id: int


@dataclass(frozen=True, slots=True)
class ReadonlyStepEnd:
    step_id: int


Channels = tuple[tuple[Connection, ...], ...]


class FeedEvent(StrEnum):
    """What one control record tells the consumer."""

    PUBLISH = "publish"
    MISSING = "missing"
    END = "end"
    CANCEL = "cancel"


@dataclass(frozen=True)
class FeedRecord:
    """One complete root or explicit producer lifecycle event."""

    kind: FeedEvent
    slot: int = -1
    ref: StateRef | None = None


def select_publications(records: deque[FeedRecord], order: FeedOrder) -> tuple[FeedRecord, ...]:
    """Select a bounded prefix; missing/end remain barriers in either order."""
    selected = [records.popleft()]
    if order is FeedOrder.LATEST and selected[0].kind is FeedEvent.PUBLISH:
        while records and records[0].kind is FeedEvent.PUBLISH:
            selected.append(records.popleft())
    return tuple(selected)


_PIDFD_OPEN = 434  # x86-64 Linux syscall number


def _process_exit(pid: int) -> Connection:
    # The bundled Python/libc omit pidfd_open; Linux still provides the lifetime fence.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    descriptor = int(libc.syscall(ctypes.c_long(_PIDFD_OPEN), ctypes.c_int(pid), ctypes.c_uint(0)))
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "state feeds need Linux process-exit descriptors")
    return Connection(descriptor)


def _restore_feed(
    plan: PlanBundle,
    binding: FeedBinding,
    control: DupFd,
    channels: tuple[tuple[DupFd, ...], ...],
    peer: int,
    device: torch.device,
) -> Feed:
    return Feed(
        plan,
        binding,
        Connection(control.detach()),
        tuple(tuple(Connection(channel.detach()) for channel in row) for row in channels),
        peer,
        device,
    )


class ConsumerClosed(RuntimeError):
    """The consumer has proved that no further GPU reads can occur."""


class Feed:
    """Publish complete tensor fields into a declared state, with bounded backpressure."""

    def __init__(
        self,
        plan: PlanBundle,
        binding: FeedBinding,
        control: Connection,
        channels: Channels,
        peer: int,
        device: torch.device,
    ) -> None:
        self._plan, self._binding, self._control, self._channels, self._peer = plan, binding, control, channels, peer
        (self._layout,) = (
            entry.layout for entry in plan.tensor_layout_request.layouts if entry.resource == binding.resource
        )
        self._fields = dict(zip(binding.fields, self._layout.fields, strict=True))
        self._device = device
        self._peer_exit = _process_exit(peer)
        self.arena: PageBundleArena | None = None
        self.current_ref: StateRef | None = None
        self._pending: dict[int, PageRelease] = {}
        self._credits = 0
        self.failed = self._ended = self.closed = self.moved = self._quiescent = False

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[PlanBundle, FeedBinding, DupFd, tuple[tuple[DupFd, ...], ...], int, torch.device], Feed],
        tuple[PlanBundle, FeedBinding, DupFd, tuple[tuple[DupFd, ...], ...], int, torch.device],
    ]:
        if self.arena is not None or self._credits or self.moved or self.closed:
            raise RuntimeError("transfer the feed once, before publishing")
        transfer = (
            self._plan,
            self._binding,
            DupFd(self._control.fileno()),
            tuple(tuple(DupFd(channel.fileno()) for channel in row) for row in self._channels),
            self._peer,
            self._device,
        )
        self.moved = True
        self._disconnect()
        return _restore_feed, transfer

    def _disconnect(self) -> None:
        self._peer_exit.close()
        self._control.close()
        for row in self._channels:
            for channel in row:
                channel.close()

    @contextmanager
    def _live(self) -> Generator[None]:
        if self.failed or self.closed or self.moved or self._ended:
            raise RuntimeError(
                "this feed is closed, handed to another process, ended, or stopped after an earlier error; "
                "start a new session"
            )
        try:
            yield
        except BaseException:
            self.failed = True
            raise

    def _signal(self) -> None:
        signal = self._control.recv_bytes()
        if signal == b"credit":
            self._credits -= 1
        else:
            assert signal == b"closed"
            self._quiescent = True

    def _wait(self, channels: tuple[Connection, ...] = ()) -> None:
        if self._control in wait((self._control, *channels)):
            self._signal()
            if self._quiescent:
                raise ConsumerClosed("feed consumer has closed")

    def _send(self, record: FeedRecord) -> None:
        while self._credits == len(self._channels):
            self._wait()
        self._control.send_bytes(encode_boundary(record).encode())
        self._credits += 1

    def _release(self, slot: int) -> None:
        receipt = self._pending.get(slot)
        if receipt is None:
            return
        assert self.arena is not None
        while not self.arena.release_complete(receipt):
            channels = tuple(read.connection for read in receipt.remote_reads if not read.query())
            if not channels:
                self.arena.wait_release(receipt)
                break
            self._wait(channels)
        del self._pending[slot]

    def _start(self, stream: torch.cuda.Stream) -> PageBundleArena:
        if self.arena is None:
            self.arena = PageBundleArena(self._plan, device=self._device, owned_resources=(self._binding.resource,))
            publication = self.arena.publish(self.arena.prepare_admission((self._binding.resource,), stream))
            (self.current_ref,) = publication.after_refs
        return self.arena

    def publish(self, **fields: torch.Tensor) -> None:
        """Copy all declared fields, then publish one immutable version; block on capacity."""
        # A mismatched call is rejected before _live, so it does not fail-stop the feed.
        if fields.keys() != self._fields.keys() or any(
            fields[name].layout != torch.strided
            or tuple(fields[name].shape) != field.shape
            or fields[name].dtype != field.dtype.torch
            or fields[name].device != self._device
            for name, field in self._fields.items()
        ):
            raise ValueError(
                f"{self._binding.resource}: publish strided tensors on {self._device} matching "
                f"{ {name: (field.shape, str(field.dtype)) for name, field in self._fields.items()} }"
            )
        with self._live():
            stream = torch.cuda.current_stream(self._device)
            arena = self._start(stream)
            assert self.current_ref is not None
            slot = self.current_ref.version_id % len(self._channels)
            self._release(slot)
            write = arena.prepare_transition(StateRefs.of((self.current_ref,)), stream, copy_fields=())
            for name, field in self._fields.items():
                arena.write_field(write, 0, field.field_id).copy_(fields[name])
            publication = arena.publish(write)
            (self.current_ref,) = publication.after_refs
            pin = arena.pin_exact(publication.after_refs, stream)
            try:
                for field, channel in zip(self._layout.fields, self._channels[slot], strict=True):
                    arena.send_field(pin, self.current_ref, field.field_id, channel, self._peer)
            finally:
                self._pending[slot] = arena.release_pin(pin)
            stream.synchronize()  # Returning transfers no source-tensor lifetime obligation to the caller.
            self._send(FeedRecord(FeedEvent.PUBLISH, slot, self.current_ref))

    def missing(self) -> None:
        """Submit one explicit absent-input step using the ActivationPatch on_missing policy."""
        with self._live():
            self._send(FeedRecord(FeedEvent.MISSING))

    def end(self) -> None:
        """End the feed normally; following steps use the declared missing policy."""
        if self._ended and not self.failed:
            return
        with self._live():
            self._send(FeedRecord(FeedEvent.END))
            self._ended = True

    def _await_shutdown(self) -> None:
        # A broken ACK is not proof of GPU quiescence. Keep the arena until teardown or process death.
        try:
            while not self._quiescent:
                wait((self._control,))
                self._signal()
        except (EOFError, OSError, RuntimeError):
            wait((self._peer_exit,))
        self._quiescent = True

    def close(self) -> None:
        """Release finished pages; cancellation retains owners until consumer teardown."""
        if self.closed or self.moved:
            return
        try:
            if not self._ended or self.failed:
                self._control.send_bytes(encode_boundary(FeedRecord(FeedEvent.CANCEL)).encode())
                self._await_shutdown()
            else:
                for slot in tuple(self._pending):
                    self._release(slot)
                while self._credits and not self._quiescent:
                    wait((self._control,))
                    self._signal()
                if self.arena is not None:
                    assert self.current_ref is not None
                    self.arena.retire(StateRefs.of((self.current_ref,)))
                    self.arena.collect_completed(wait=True)
        except ConsumerClosed:
            pass
        except (EOFError, OSError, RuntimeError):
            self.failed = True
            try:
                self._control.send_bytes(encode_boundary(FeedRecord(FeedEvent.CANCEL)).encode())
            except (EOFError, OSError):
                pass
            self._await_shutdown()
            raise
        finally:
            self.arena = None
            self.closed = True
            self._disconnect()


class FeedReceiver:
    """Install the framework's step input and retain partial receives until teardown."""

    def __init__(self, arena: PageBundleArena, binding: FeedBinding) -> None:
        self.arena, self.binding = arena, binding
        (layout,) = (e.layout for e in arena.plan.tensor_layout_request.layouts if e.resource == binding.resource)
        context = get_context("spawn")
        self.control, writer = context.Pipe()
        pipes = tuple(tuple(context.Pipe() for _ in layout.fields) for _ in range(FEED_SLOTS))
        self.channels = tuple(tuple(pair[0] for pair in row) for row in pipes)
        self.writer: Feed | None = Feed(
            arena.plan,
            binding,
            writer,
            tuple(tuple(pair[1] for pair in row) for row in pipes),
            os.getpid(),
            arena.device,
        )
        self.records: deque[FeedRecord] = deque()
        self.partial: list[BorrowedPage] = []
        self.current: BorrowedStateRoot | None = None
        self.ended = self.failed = self.closed = False

    def take(self, name: str) -> Feed:
        """Transfer the only writer before model requests begin."""
        if name != self.binding.resource or self.writer is None:
            raise ValueError(f"request the feed for {self.binding.resource} exactly once")
        writer, self.writer = self.writer, None
        return writer

    def _snapshot(self) -> None:
        if not self.records:
            self.records.append(decode_boundary(self.control.recv_bytes().decode(), FeedRecord))
        while (
            len(self.records) <= len(self.channels)
            and self.records[-1].kind is not FeedEvent.END
            and self.control.poll()
        ):
            self.records.append(decode_boundary(self.control.recv_bytes().decode(), FeedRecord))
        if any(record.kind is FeedEvent.CANCEL for record in self.records):
            raise RuntimeError("feed producer cancelled or sent an invalid record")

    def _receive(self, record: FeedRecord, stream: torch.cuda.Stream | None) -> BorrowedStateRoot:
        assert record.ref is not None and 0 <= record.slot < len(self.channels) and not self.partial
        for channel in self.channels[record.slot]:
            self.partial.append(receive_field(self.arena.plan, self.arena.plan_digest, channel, stream=stream))
        root = BorrowedStateRoot(self.arena, self.binding.resource, record.ref, tuple(self.partial))
        self.partial.clear()
        self.current = root
        return root

    def __call__(
        self, rows: PackedSamplerRowBatch, stream: torch.cuda.Stream | None
    ) -> ReadonlyStepInput | ReadonlyStepAbsent | ReadonlyStepEnd:
        if self.failed or self.closed or self.writer is not None:
            raise RuntimeError("obtain the feed writer before generate; a failed feed cannot resume")
        try:
            if self.ended:
                return ReadonlyStepEnd(rows.step_id)
            self._snapshot()
            selected = select_publications(self.records, self.binding.order)
            for record in selected[:-1]:
                self._receive(record, stream).close()
                self.control.send_bytes(b"credit")
            record = selected[-1]
            if record.kind is FeedEvent.PUBLISH:
                result = ReadonlyStepInput(rows.step_id, (self._receive(record, stream),))
            elif record.kind is FeedEvent.MISSING:
                result = ReadonlyStepAbsent(rows.step_id)
            else:
                self.ended = True
                result = ReadonlyStepEnd(rows.step_id)
            self.control.send_bytes(b"credit")
            return result
        except BaseException:
            self.failed = True
            raise

    def close(self) -> None:
        """Acknowledge teardown only after the session has discarded every borrowed view."""
        if self.closed:
            return
        if self.current is not None:
            self.current.discard()
        for page in self.partial:
            page.discard()
        self.partial.clear()
        try:
            self.control.send_bytes(b"closed")
        except (BrokenPipeError, EOFError):
            pass
        if self.writer is not None:
            self.writer.close()
        self.closed = True
        self.control.close()
        for row in self.channels:
            for channel in row:
                channel.close()
