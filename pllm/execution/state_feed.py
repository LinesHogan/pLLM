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

from pllm.contracts.compilation import PlanBundle
from pllm.contracts.packed_rows import PackedSamplerRowBatch
from pllm.contracts.state_feed import FeedBinding, FeedOrder
from pllm.contracts.state_ref import StateRef
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.ptts.page_borrow import BorrowedStateRoot
from pllm.execution.ptts.page_bundle import BorrowedPage, PageBundleArena, PageRelease
from pllm.execution.shared_tensor import tensor_dtype


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
) -> Feed:
    return Feed(
        plan,
        binding,
        Connection(control.detach()),
        tuple(tuple(Connection(channel.detach()) for channel in row) for row in channels),
        peer,
    )


class _ConsumerClosed(RuntimeError):
    """The consumer has proved that no further GPU reads can occur."""


class Feed:
    """Publish complete tensor fields into a declared state, with bounded backpressure."""

    def __init__(
        self, plan: PlanBundle, binding: FeedBinding, control: Connection, channels: Channels, peer: int
    ) -> None:
        self._plan, self._binding, self._control, self._channels, self._peer = plan, binding, control, channels, peer
        (self._layout,) = (
            entry.layout for entry in plan.tensor_layout_request.layouts if entry.resource == binding.resource
        )
        (provider,) = (entry for entry in plan.states.plans if entry.resource == binding.resource)
        self._fields = dict(zip(binding.fields, self._layout.fields, strict=True))
        self._device = torch.device("cuda", provider.device_ordinal)
        self._peer_exit = _process_exit(peer)
        self._arena: PageBundleArena | None = None
        self._current: StateRef | None = None
        self._pending: dict[int, PageRelease] = {}
        self._credits = 0
        self._failed = self._ended = self._closed = self._moved = self._quiescent = False

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[PlanBundle, FeedBinding, DupFd, tuple[tuple[DupFd, ...], ...], int], Feed],
        tuple[PlanBundle, FeedBinding, DupFd, tuple[tuple[DupFd, ...], ...], int],
    ]:
        if self._arena is not None or self._credits or self._moved or self._closed:
            raise RuntimeError("transfer the feed once, before publishing")
        transfer = (
            self._plan,
            self._binding,
            DupFd(self._control.fileno()),
            tuple(tuple(DupFd(channel.fileno()) for channel in row) for row in self._channels),
            self._peer,
        )
        self._moved = True
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
        if self._failed or self._closed or self._moved or self._ended:
            raise RuntimeError("feed is closed, transferred, ended or fail-stopped; start a new session")
        try:
            yield
        except BaseException:
            self._failed = True
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
                raise _ConsumerClosed("feed consumer has closed")

    def _send(self, record: FeedRecord) -> None:
        while self._credits == len(self._channels):
            self._wait()
        self._control.send_bytes(encode_boundary(record).encode())
        self._credits += 1

    def _release(self, slot: int) -> None:
        receipt = self._pending.get(slot)
        if receipt is None:
            return
        assert self._arena is not None
        while not self._arena.release_complete(receipt):
            channels = tuple(read.connection for read in receipt.remote_reads if not read.query())
            if not channels:
                self._arena.wait_release(receipt)
                break
            self._wait(channels)
        del self._pending[slot]

    def _start(self, stream: torch.cuda.Stream) -> PageBundleArena:
        if self._arena is None:
            self._arena = PageBundleArena(self._plan, _owned_resources=(self._binding.resource,))
            publication = self._arena.publish(self._arena.prepare_admission((self._binding.resource,), stream))
            self._arena.settle(publication)
            (self._current,) = publication.after_refs
        return self._arena

    def publish(self, **fields: torch.Tensor) -> None:
        """Copy all declared fields, then publish one immutable version; block on capacity."""
        if (
            self._failed
            or self._closed
            or self._moved
            or self._ended
            or fields.keys() != self._fields.keys()
            or any(
                fields[name].layout != torch.strided
                or tuple(fields[name].shape) != field.shape
                or fields[name].dtype != tensor_dtype(field.dtype)
                or fields[name].device != self._device
                for name, field in self._fields.items()
            )
        ):
            with self._live():
                pass
            raise ValueError(
                f"{self._binding.resource}: publish strided tensors on {self._device} matching "
                f"{ {name: (field.shape, str(field.dtype)) for name, field in self._fields.items()} }"
            )
        with self._live():
            stream = torch.cuda.current_stream(self._device)
            arena = self._start(stream)
            assert self._current is not None
            slot = self._current.version_id % len(self._channels)
            self._release(slot)
            write = arena.prepare_transition((self._current,), stream, copy_fields=())
            for name, field in self._fields.items():
                arena.write_field(write, 0, field.field_id).copy_(fields[name])
            publication = arena.publish(write)
            arena.settle(publication)
            (self._current,) = publication.after_refs
            pin = arena.pin_exact((self._current,), stream)
            try:
                for field, channel in zip(self._layout.fields, self._channels[slot], strict=True):
                    arena.send_field(pin, self._current, field.field_id, channel, self._peer)
            finally:
                self._pending[slot] = arena.release_pin(pin)
            stream.synchronize()  # Returning transfers no source-tensor lifetime obligation to the caller.
            self._send(FeedRecord(FeedEvent.PUBLISH, slot, self._current))

    def missing(self) -> None:
        """Submit one explicit absent-input step using the Activation missing policy."""
        with self._live():
            self._send(FeedRecord(FeedEvent.MISSING))

    def end(self) -> None:
        """End the feed normally; following steps use the declared missing policy."""
        if self._ended and not self._failed:
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
        """Release settled pages; cancellation retains owners until consumer teardown."""
        if self._closed or self._moved:
            return
        try:
            if not self._ended or self._failed:
                self._control.send_bytes(encode_boundary(FeedRecord(FeedEvent.CANCEL)).encode())
                self._await_shutdown()
            else:
                for slot in tuple(self._pending):
                    self._release(slot)
                while self._credits and not self._quiescent:
                    wait((self._control,))
                    self._signal()
                if self._arena is not None:
                    assert self._current is not None
                    self._arena.retire((self._current,))
                    self._arena.collect_completed(wait=True)
        except _ConsumerClosed:
            pass
        except (EOFError, OSError, RuntimeError):
            self._failed = True
            try:
                self._control.send_bytes(encode_boundary(FeedRecord(FeedEvent.CANCEL)).encode())
            except (EOFError, OSError):
                pass
            self._await_shutdown()
            raise
        finally:
            self._arena = None
            self._closed = True
            self._disconnect()


class FeedReceiver:
    """Install the framework's step input and retain partial receives until teardown."""

    def __init__(self, arena: PageBundleArena, binding: FeedBinding) -> None:
        self.arena, self.binding = arena, binding
        (provider,) = (p for p in arena.plan.states.plans if p.resource == binding.resource)
        (layout,) = (e.layout for e in arena.plan.tensor_layout_request.layouts if e.resource == binding.resource)
        context = get_context("spawn")
        self.control, writer = context.Pipe()
        pipes = tuple(tuple(context.Pipe() for _ in layout.fields) for _ in range(provider.page_capacity))
        self.channels = tuple(tuple(pair[0] for pair in row) for row in pipes)
        self.writer: Feed | None = Feed(
            arena.plan, binding, writer, tuple(tuple(pair[1] for pair in row) for row in pipes), os.getpid()
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

    def _receive(self, record: FeedRecord, stream: torch.cuda.Stream) -> BorrowedStateRoot:
        assert record.ref is not None and 0 <= record.slot < len(self.channels) and not self.partial
        for channel in self.channels[record.slot]:
            self.partial.append(self.arena.receive_field(channel, stream=stream))
        root = BorrowedStateRoot(self.arena, self.binding.resource, record.ref, tuple(self.partial))
        self.partial.clear()
        self.current = root
        return root

    def __call__(
        self, rows: PackedSamplerRowBatch, stream: torch.cuda.Stream
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
