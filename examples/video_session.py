"""A bounded real-video producer and a spawned, in-process vLLM consumer.

Run: .venv/bin/python -m examples.video_session /path/to/video.mp4 --chunks 8
Requires ffmpeg, CUDA and the configured local/cached Qwen2 checkpoint.
"""

import argparse
import multiprocessing as mp
import subprocess
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

import torch

import pllm
from pllm.authoring import TensorStepInput
from pllm.config import ModelConfig
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.packed_rows import PackedSamplerRowBatch
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.ptts.page_borrow import BorrowedStateRoot
from pllm.execution.ptts.page_bundle import BorrowedPage, PageBundleArena, PageFieldTransfer, PageRelease
from pllm.execution.scheduler import validate_plan_bundle
from pllm.execution.state_execution import StateExecution
from pllm.execution.tensor_binding import TensorOutput
from pllm.execution.tensor_frame import TensorFrame
from pllm.workloads.video import VideoWorkload, frame_features

if TYPE_CHECKING:
    from pllm.backends.vllm.runtime import ReadonlyStepAbsent, ReadonlyStepEnd, ReadonlyStepInput

Channels = tuple[tuple[Connection, ...], ...]


@dataclass(frozen=True)
class Chunk:
    slot: int
    ref: StateRef
    step_id: int


@dataclass(frozen=True)
class VideoRun:
    steps: int


@dataclass(frozen=True)
class InputGap:
    step_id: int
    end: bool


@dataclass(frozen=True)
class VideoInput:
    value: Chunk | InputGap


@dataclass(frozen=True)
class VideoResult:
    ref: StateRef | None
    text: str
    updates: int
    fallback: int
    step_id: int
    holds: int
    missing: str | None


@dataclass(frozen=True)
class Pending:
    ref: StateRef
    release: PageRelease
    fields: tuple[PageFieldTransfer, ...]


class _StepReceiver:
    """A finite explicit next-step protocol; partial receives remain owned here."""

    def __init__(
        self,
        control: Connection,
        channels: Channels,
        execution: StateExecution[TensorStepInput, TensorFrame, TensorOutput],
    ) -> None:
        self.control, self.channels, self.execution = control, channels, execution
        self.pending: Chunk | InputGap | None = None
        self.previous: tuple[int, StateRef | None, str | None] | None = None
        self.ended_for: tuple[object, ...] = ()
        self.partial: list[BorrowedPage] = []

    def report(self, text: str = "") -> None:
        if self.previous is not None:
            diagnostics = self.execution.diagnostics()
            step_id, ref, missing = self.previous
            self.control.send_bytes(
                encode_boundary(
                    VideoResult(
                        ref,
                        text,
                        diagnostics.update_count,
                        diagnostics.fallback_count,
                        step_id,
                        diagnostics.hold_count,
                        missing,
                    )
                ).encode()
            )
            self.previous = None

    def __call__(
        self, rows: PackedSamplerRowBatch, stream: torch.cuda.Stream
    ) -> "ReadonlyStepInput | ReadonlyStepAbsent | ReadonlyStepEnd":
        from pllm.backends.vllm.runtime import ReadonlyStepAbsent, ReadonlyStepEnd, ReadonlyStepInput

        # The preceding step has settled before this next prepare boundary.
        self.report()
        chunk = self.pending
        self.pending = None
        identities = tuple(rows.identities)
        if self.ended_for and identities != self.ended_for:
            self.ended_for = ()  # A new request must supply a fresh first version.
        if chunk is None:
            chunk = (
                InputGap(rows.step_id, True)
                if self.ended_for
                else decode_boundary(self.control.recv_bytes().decode(), VideoInput).value
            )
        if self.partial or chunk.step_id != rows.step_id:
            raise ValueError("video input differs from the exact step or has partial received grants")
        if isinstance(chunk, InputGap):
            self.previous = rows.step_id, None, "end" if chunk.end else "absent"
            if chunk.end:
                self.ended_for = identities
            outcome = ReadonlyStepEnd if chunk.end else ReadonlyStepAbsent
            return outcome(rows.plan_epoch, rows.step_id)
        if self.ended_for:
            raise ValueError("video received a new grant after the current request's End")
        if not 0 <= chunk.slot < len(self.channels) or self.partial:
            raise ValueError("video step requires its original complete grant slot")
        plan = self.execution.arena.plan
        for channel in self.channels[chunk.slot]:
            self.partial.append(PageBundleArena.receive_field(plan, channel, stream=stream))
        root = BorrowedStateRoot(plan, "features", chunk.ref, tuple(self.partial))
        self.partial.clear()
        self.previous = chunk.step_id, chunk.ref, None
        return ReadonlyStepInput(plan.schedule_physical.plan_epoch, chunk.step_id, (root,))


def consume(control: Connection, channels: Channels, workload: VideoWorkload, config: ModelConfig) -> None:
    from vllm import SamplingParams

    try:
        with pllm.make_llm(config=config, workload=workload) as llm:
            execution = llm.deployment.execution
            plan = execution.arena.plan
            control.send_bytes(encode_boundary(plan).encode())
            run = decode_boundary(control.recv_bytes().decode(), VideoRun)
            if run.steps < 1:
                raise ValueError("video generation requires a positive finite step count")
            receiver = _StepReceiver(control, channels, execution)
            llm.runner.require_session().bind_readonly_input(receiver)
            while (payload := control.recv_bytes()) != b"stop":
                receiver.pending = decode_boundary(payload.decode(), VideoInput).value
                outputs = llm.generate(
                    ["Describe the scene in this video frame:"],
                    SamplingParams(max_tokens=run.steps, temperature=0.0, ignore_eos=True),
                )
                if len(outputs[0].outputs[0].token_ids) != run.steps:
                    raise RuntimeError("generation did not consume all explicitly scheduled video steps")
                receiver.report(outputs[0].outputs[0].text)
    finally:
        control.close()
        for row in channels:
            for channel in row:
                channel.close()


class VideoSession:
    """Example-local supervision. Worker replacement is for process-level test harnesses."""

    def __init__(
        self,
        workload: VideoWorkload,
        config: ModelConfig,
        *,
        timeout: float = 120,
        steps: int = 1,
        worker: Callable[[Connection, Channels, VideoWorkload, ModelConfig], None] = consume,
    ) -> None:
        if config.multiprocessing or timeout <= 0 or steps < 1:
            raise ValueError("the video example owns spawn; its vLLM child must be in-process")
        context = mp.get_context("spawn")
        self.control, child_control = context.Pipe()
        pipes = tuple(tuple(context.Pipe() for _ in range(4)) for _ in range(workload.page_capacity))
        self.channels = tuple(tuple(pair[0] for pair in row) for row in pipes)
        child_channels = tuple(tuple(pair[1] for pair in row) for row in pipes)
        self.process = context.Process(target=worker, args=(child_control, child_channels, workload, config))
        self.timeout, self.workload = timeout, workload
        self.steps = steps
        self.pending: deque[Pending] = deque()
        self._reports: deque[tuple[int, StateRef | None, str | None]] = deque()
        self.failed = False
        self.stream: torch.cuda.Stream | None = None
        self._previous: torch.Tensor | None = None
        self._sequence = 0
        self.process.start()
        child_control.close()
        for row in child_channels:
            for channel in row:
                channel.close()
        try:
            self.plan = decode_boundary(self._receive().decode(), PlanBundle)
            validate_plan_bundle(self.plan)
            (provider,) = (p for p in self.plan.state_physical.representations if p.resource == "features")
            if provider.page_capacity != len(self.channels) or "state/shared_cuda@1" not in provider.capabilities:
                raise ValueError("video transport requires its signed CUDA pool capacity")
            (layout,) = (
                entry.layout for entry in self.plan.tensor_layout_request.layouts if entry.resource == "features"
            )
            if (
                tuple((f.shape, f.dtype) for f in layout.fields)
                != (((workload.grid_size,) * 2, TensorDType.FLOAT32),) * 4
            ):
                raise ValueError("video producer grid differs from the signed feature fields")
            self.arena = PageBundleArena(self.plan, _owned_resources=("features",))
            self.stream = torch.cuda.Stream(device=self.arena.device)
            admission = self.arena.publish(self.arena.prepare_admission(("features",), self.stream))
            self.arena.settle(admission)
            (self.current,) = admission.after_refs
            self.control.send_bytes(encode_boundary(VideoRun(steps)).encode())
        except BaseException:
            self.failed = True
            self.close()
            raise

    def _receive(self) -> bytes:
        if not self.control.poll(self.timeout):
            raise TimeoutError("video worker deadline expired; original page owners remain retained")
        return self.control.recv_bytes(16 * 1024 * 1024)

    def submit(self, frame: torch.Tensor) -> VideoResult | None:
        if self.failed:
            raise RuntimeError("video session is fail-stopped")
        assert self.stream is not None
        try:
            # Bound even completed-but-uncollected records by signed capacity (GPU backpressure test).
            completed = None
            while len(self.pending) == len(self.channels):
                completed = self.drain()
            write = self.arena.prepare_transition((self.current,), self.stream, copy_fields=())
            try:
                with torch.cuda.stream(self.stream):
                    current = frame.to(self.arena.device)
                    values = frame_features(
                        current, current if self._previous is None else self._previous, self.workload.grid_size
                    )
                    for index, value in enumerate(values):
                        self.arena.write_field(write, 0, LayoutFieldId(index)).copy_(value)
                    self._previous = current
                publication = self.arena.publish(write)
            except BaseException:
                self.arena.abort(write)
                raise
            self.arena.settle(publication)
            (self.current,) = publication.after_refs
            pin = self.arena.pin_exact((self.current,), self.stream)
            slot = self._sequence % len(self.channels)
            self.control.send_bytes(encode_boundary(VideoInput(Chunk(slot, self.current, self._sequence))).encode())
            assert self.process.pid is not None
            fields = tuple(
                self.arena.send_field(pin, self.current, LayoutFieldId(index), channel, self.process.pid)
                for index, channel in enumerate(self.channels[slot])
            )
            self.pending.append(Pending(self.current, self.arena.release_pin(pin), fields))
            self._reports.append((self._sequence, self.current, None))
            self._sequence += 1
            return completed
        except BaseException:
            self.failed = True
            raise

    @property
    def pending_steps(self) -> int:
        return len(self._reports)

    def missing(self, *, end: bool = False) -> None:
        """Explicit next-step evidence, never inferred from a timeout or an empty queue."""
        if self.failed:
            raise RuntimeError("video session is fail-stopped")
        try:
            self.control.send_bytes(encode_boundary(VideoInput(InputGap(self._sequence, end))).encode())
            count = self.steps - self._sequence % self.steps if end else 1
            for _ in range(count):
                self._reports.append((self._sequence, None, "end" if end else "absent"))
                self._sequence += 1
        except BaseException:
            self.failed = True
            raise

    def drain(self) -> VideoResult:
        if self.failed or not self._reports:
            raise RuntimeError("video session has no consumable pending chunk")
        try:
            result = decode_boundary(self._receive().decode(), VideoResult)
            if (result.step_id, result.ref, result.missing) != self._reports[0] or result.fallback:
                raise RuntimeError("video worker did not consume the original exact chunk without fallback")
            if result.ref is not None:
                pending = self.pending[0]
                if result.ref != pending.ref:
                    raise RuntimeError("video report differs from its outstanding grant")
                self.arena.wait_release(pending.release)
                self.pending.popleft()
            self._reports.popleft()
            return result
        except BaseException:
            self.failed = True
            raise

    def close(self) -> None:
        try:
            if not self.failed:
                if self._sequence % self.steps:
                    self.failed = True
                    raise RuntimeError("video ended before all explicitly scheduled decode inputs were supplied")
                while self._reports:
                    self.drain()
                self.control.send_bytes(b"stop")
                self.process.join(self.timeout)
                if self.process.exitcode != 0:
                    raise RuntimeError("video worker did not shut down cleanly")
                self.arena.retire((self.current,))
                self.arena.collect_completed(wait=True)
        finally:
            # Join every importer before the original arena may be discarded; fault tests check retention.
            if self.process.is_alive():
                self.process.kill()
            self.process.join()
            if self.stream is not None:
                self.stream.synchronize()
            self.control.close()
            for row in self.channels:
                for channel in row:
                    channel.close()

    def __enter__(self) -> "VideoSession":
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.failed |= error is not None
        self.close()


def video_frames(path: Path, grid_size: int, limit: int) -> Iterator[torch.Tensor]:
    side = grid_size * 8
    with subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"scale={side}:{side}",
            "-frames:v",
            str(limit),
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
    ) as decoder:
        assert decoder.stdout is not None
        try:
            while data := decoder.stdout.read(side * side):
                if len(data) != side * side:
                    raise ValueError("truncated decoded video frame")
                yield torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(side, side)
            if decoder.wait() != 0:
                raise RuntimeError("ffmpeg could not decode the requested video")
        finally:
            if decoder.poll() is None:
                decoder.kill()
                decoder.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()
    workload = VideoWorkload()
    with VideoSession(
        workload,
        ModelConfig(
            args.model, max_model_len=64, max_num_seqs=1, max_num_batched_tokens=64, gpu_memory_utilization=0.25
        ),
        steps=args.chunks,
    ) as session:
        supplied = 0
        for frame in video_frames(args.video, workload.grid_size, args.chunks):
            result = session.submit(frame)
            supplied += 1
            if result is not None:
                print(result)
        if supplied < args.chunks:
            session.missing(end=True)  # Normal iterator exhaustion, never a decoder/transport exception.
        while session.pending_steps:
            print(session.drain())


if __name__ == "__main__":
    main()
