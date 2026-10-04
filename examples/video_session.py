"""A bounded real-video producer and a spawned, in-process vLLM consumer.

Run: .venv/bin/python -m examples.video_session /path/to/video.mp4 --chunks 8
Requires ffmpeg, CUDA and the configured local/cached Qwen2 checkpoint.
"""

import argparse
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
import subprocess
from types import TracebackType

import torch
import pllm
from pllm.config import ModelConfig
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.ptts.page_bundle import PageBundleArena, PageFieldTransfer, PageRelease
from pllm.execution.ptts.page_borrow import BorrowedStateRoot
from pllm.execution.scheduler import validate_plan_bundle
from pllm.workloads.video import VideoWorkload, frame_features

Channels = tuple[tuple[Connection, ...], ...]


@dataclass(frozen=True)
class Chunk:
    slot: int
    ref: StateRef


@dataclass(frozen=True)
class VideoResult:
    ref: StateRef
    text: str
    updates: int
    fallback: int


@dataclass(frozen=True)
class Pending:
    ref: StateRef
    release: PageRelease
    fields: tuple[PageFieldTransfer, ...]


def consume(control: Connection, channels: Channels, workload: VideoWorkload, config: ModelConfig) -> None:
    from vllm import SamplingParams
    try:
        with pllm.make_llm(config=config, workload=workload) as llm:
            execution = llm.deployment.execution
            plan = execution.arena.plan
            control.send_bytes(encode_boundary(plan).encode())
            while (payload := control.recv_bytes()) != b"stop":
                chunk = decode_boundary(payload.decode(), Chunk)
                stream = torch.cuda.current_stream(execution.device)
                with torch.cuda.stream(stream):
                    pages = tuple(PageBundleArena.receive_field(plan, channel, stream=stream) for channel in channels[chunk.slot])
                    root = BorrowedStateRoot(plan, "features", chunk.ref, pages)
                    execution.stage_readonly((root,))
                    outputs = llm.generate(["Describe the scene in this video frame:"],
                        SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True))
                diagnostics = execution.diagnostics()
                control.send_bytes(encode_boundary(VideoResult(chunk.ref, outputs[0].outputs[0].text,
                    diagnostics.update_count, diagnostics.fallback_count)).encode())
    finally:
        control.close()
        for row in channels:
            for channel in row:
                channel.close()


class VideoSession:
    """Example-local supervision. Worker replacement is for process-level test harnesses."""

    def __init__(self, workload: VideoWorkload, config: ModelConfig, *, timeout: float = 120,
                 worker: Callable[[Connection, Channels, VideoWorkload, ModelConfig], None] = consume) -> None:
        if config.multiprocessing or timeout <= 0:
            raise ValueError("the video example owns spawn; its vLLM child must be in-process")
        context = mp.get_context("spawn")
        self.control, child_control = context.Pipe()
        pipes = tuple(tuple(context.Pipe() for _ in range(4)) for _ in range(workload.page_capacity))
        self.channels = tuple(tuple(pair[0] for pair in row) for row in pipes)
        child_channels = tuple(tuple(pair[1] for pair in row) for row in pipes)
        self.process = context.Process(target=worker, args=(child_control, child_channels, workload, config))
        self.timeout, self.workload = timeout, workload
        self.pending: deque[Pending] = deque()
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
            provider, = (p for p in self.plan.state_physical.representations if p.resource == "features")
            if provider.page_capacity != len(self.channels) or "state/shared_cuda@1" not in provider.capabilities:
                raise ValueError("video transport requires its signed CUDA pool capacity")
            layout, = (entry.layout for entry in self.plan.tensor_layout_request.layouts if entry.resource == "features")
            if tuple((f.shape, f.dtype) for f in layout.fields) != (((workload.grid_size,) * 2, TensorDType.FLOAT32),) * 4:
                raise ValueError("video producer grid differs from the signed feature fields")
            self.arena = PageBundleArena(self.plan, _owned_resources=("features",))
            self.stream = torch.cuda.Stream(device=self.arena.device)
            admission = self.arena.publish(self.arena.prepare_admission(("features",), self.stream))
            self.arena.settle(admission)
            self.current, = admission.after_refs
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
            completed = self.drain() if len(self.pending) == len(self.channels) else None
            write = self.arena.prepare_transition((self.current,), self.stream, copy_fields=())
            try:
                with torch.cuda.stream(self.stream):
                    current = frame.to(self.arena.device)
                    values = frame_features(current, current if self._previous is None else self._previous, self.workload.grid_size)
                    for index, value in enumerate(values):
                        self.arena.write_field(write, 0, LayoutFieldId(index)).copy_(value)
                    self._previous = current
                publication = self.arena.publish(write)
            except BaseException:
                self.arena.abort(write)
                raise
            self.arena.settle(publication)
            self.current, = publication.after_refs
            pin = self.arena.pin_exact((self.current,), self.stream)
            slot = self._sequence % len(self.channels)
            self.control.send_bytes(encode_boundary(Chunk(slot, self.current)).encode())
            assert self.process.pid is not None
            fields = tuple(self.arena.send_field(pin, self.current, LayoutFieldId(index), channel, self.process.pid)
                for index, channel in enumerate(self.channels[slot]))
            self.pending.append(Pending(self.current, self.arena.release_pin(pin), fields))
            self._sequence += 1
            return completed
        except BaseException:
            self.failed = True
            raise

    def drain(self) -> VideoResult:
        if self.failed or not self.pending:
            raise RuntimeError("video session has no consumable pending chunk")
        try:
            result = decode_boundary(self._receive().decode(), VideoResult)
            pending = self.pending[0]
            if result.ref != pending.ref or result.fallback:
                raise RuntimeError("video worker did not consume the original exact chunk without fallback")
            self.arena.wait_release(pending.release)
            self.pending.popleft()
            return result
        except BaseException:
            self.failed = True
            raise

    def close(self) -> None:
        try:
            if not self.failed:
                while self.pending:
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

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None) -> None:
        self.failed |= error is not None
        self.close()


def video_frames(path: Path, grid_size: int, limit: int) -> Iterator[torch.Tensor]:
    side = grid_size * 8
    with subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={side}:{side}",
        "-frames:v", str(limit), "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"], stdout=subprocess.PIPE) as decoder:
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
    with VideoSession(workload, ModelConfig(args.model, max_model_len=64, max_num_seqs=1,
        max_num_batched_tokens=64, gpu_memory_utilization=0.25)) as session:
        for frame in video_frames(args.video, workload.grid_size, args.chunks):
            result = session.submit(frame)
            if result is not None:
                print(result)
        while session.pending:
            print(session.drain())


if __name__ == "__main__":
    main()
