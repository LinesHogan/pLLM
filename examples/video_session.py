"""Decode real video in a spawned producer while vLLM consumes its state feed.

Run: python -m examples.video_session /path/to/video.mp4 --chunks 8
"""

import argparse
import multiprocessing as mp
import subprocess
from collections.abc import Iterator
from multiprocessing.process import BaseProcess
from pathlib import Path

import torch
from vllm import SamplingParams

import pllm
from pllm.workloads.video import FEATURES, VideoWorkload, frame_features


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


def produce(feed: pllm.state.Feed, path: Path, grid: int, chunks: int, device: torch.device) -> None:
    """Publish the same frame features and mark normal iterator exhaustion explicitly."""
    previous: torch.Tensor | None = None
    try:
        for frame in video_frames(path, grid, chunks):
            current = frame.to(device)
            fields = frame_features(current, current if previous is None else previous, grid)
            feed.publish(**dict(zip(FEATURES, fields, strict=True)))
            previous = current
        feed.end()
    finally:
        feed.close()


def main() -> None:
    """Own the model and join the producer after the model releases its imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()
    if args.chunks < 1:
        parser.error("--chunks must be positive")
    workload = VideoWorkload()
    worker: BaseProcess | None = None
    try:
        with pllm.make_llm(
            config=pllm.ModelConfig(
                args.model,
                max_model_len=64,
                max_num_seqs=1,
                max_num_batched_tokens=64,
                gpu_memory_utilization=0.25,
                cuda_graphs=False,  # the activation patch runs outside CUDA graphs
            ),
            workload=workload,
        ) as llm:
            worker = mp.get_context("spawn").Process(
                target=produce,
                args=(llm.feed("features"), args.video, workload.grid_size, args.chunks, llm.device),
            )
            worker.start()
            outputs = llm.generate(
                ["Describe the scene in this video frame:"],
                SamplingParams(max_tokens=args.chunks, temperature=0.0, ignore_eos=True),
            )
            if len(outputs[0].outputs[0].token_ids) != args.chunks:
                raise RuntimeError("generation did not consume the requested number of steps")
            print(outputs[0].outputs[0].text)
    finally:
        if worker is not None and worker.pid is not None:
            worker.join()
    if worker is not None and worker.exitcode != 0:
        raise RuntimeError("video producer failed; inspect its exception")


if __name__ == "__main__":
    main()
