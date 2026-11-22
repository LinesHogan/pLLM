"""Print what vLLM's MRv2 model runner hands the attention backend, step by step, for a mixed batch.

The first section of doc/dev/vllm-backend-internals.md explains the fields of `InputBatch`,
`CommonAttentionMetadata` and the backend's own metadata from a real run. Those are vLLM internals that
may change in any release: after a vLLM upgrade, rerun this probe and compare its output with the tables.

The probe starts Qwen2.5-0.5B-Instruct with a 64-token step budget and submits three requests: a 5-token
prompt sampled with n=2, a 2-token prompt, and a 150-token prompt that is prefilled in chunks while the
others decode. On every tensor-parallel rank it wraps the runner's `model_state.prepare_attn` and the
attention metadata builder, and appends each step's fields to a per-rank file. It then prints rank 0's
steps and whether every other rank saw exactly the same fields.

Run (needs one CUDA GPU, or two with --tp 2):

    python tools/probe_attention_metadata.py --backend FLASH_ATTN
    python tools/probe_attention_metadata.py --backend FLASHINFER --tp 2
"""

import argparse
import dataclasses
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from vllm import LLM, SamplingParams

Field = str | int | float | bool | None | dict[str, "Field"]
INPUT_BATCH_FIELDS = (
    "req_ids",
    "num_reqs",
    "num_reqs_after_padding",
    "num_tokens",
    "num_tokens_after_padding",
    "num_scheduled_tokens",
    "num_computed_tokens_np",
    "query_start_loc_np",
    "seq_lens",
    "positions",
    "idx_mapping_np",
    "logits_indices",
    "is_prefilling_np",
)


def describe(value: object) -> Field:
    """A JSON-friendly view of one field: arrays show shape, dtype, device and their first 40 values."""
    if isinstance(value, torch.Tensor):
        dtype = str(value.dtype).removeprefix("torch.")
        return f"Tensor{tuple(value.shape)} {dtype} {value.device.type} {preview(value.detach().cpu().numpy())}"
    if isinstance(value, np.ndarray):
        array: NDArray[Any] = value  # pyright: ignore[reportUnknownVariableType]
        return f"ndarray{array.shape} {array.dtype} {preview(array)}"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: describe(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, (int, float, bool, str, type(None))):
        return value
    return repr(value)[:200]


def preview(array: NDArray[Any]) -> str:
    return f"{array.flatten()[:40].tolist()}{' ...' if array.size > 40 else ''}"


class StepRecorder:
    """Lives in one worker; writes one JSON line per model step to that rank's file."""

    def __init__(self, runner: Any, path: Path) -> None:
        self.path = path
        self.prepare_attn = runner.model_state.prepare_attn
        self.builder = runner.attn_groups[0][0].get_metadata_builder()
        self.build = self.builder.build
        self.common: Field = None

    def record_build(self, common_prefix_len: int, common_attn_metadata: Any, **kwargs: Any) -> Any:
        # Parameter names match vLLM's builders: some releases pass these two by keyword.
        self.common = describe(common_attn_metadata)
        return self.build(common_prefix_len, common_attn_metadata, **kwargs)

    def record_prepare_attn(self, input_batch: Any, cudagraph_mode: Any, *args: Any, **kwargs: Any) -> Any:
        # prepare_attn calls the builder, so self.common already belongs to this step.
        attn_metadata: dict[str, Any] = self.prepare_attn(input_batch, cudagraph_mode, *args, **kwargs)
        metadata = next(iter(attn_metadata.values()))
        step = {
            "cudagraph_mode": str(cudagraph_mode),
            "input_batch": {name: describe(getattr(input_batch, name)) for name in INPUT_BATCH_FIELDS},
            "common": self.common,
            "builder": type(self.builder).__name__,
            "metadata_type": type(metadata).__name__,
            "metadata": describe(metadata),
        }
        with self.path.open("a") as file:
            file.write(json.dumps(step) + "\n")
        return attn_metadata


def install(worker: Any, directory: str) -> list[int]:
    """Runs inside each worker after startup, so warmup and graph capture are not recorded."""
    runner = worker.model_runner
    recorder = StepRecorder(runner, Path(directory) / f"rank{worker.rank}.jsonl")
    runner.model_state.prepare_attn = recorder.record_prepare_attn
    recorder.builder.build = recorder.record_build
    return list(runner.kv_caches[0].shape)


def print_steps(lines: list[str]) -> None:
    for number, line in enumerate(lines, 1):
        step = json.loads(line)
        print(f"\nstep {number}  cudagraph={step['cudagraph_mode']}  {step['builder']} -> {step['metadata_type']}")
        for group in ("input_batch", "common", "metadata"):
            for name, value in step[group].items():
                print(f"  {group}.{name} = {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="FLASH_ATTN", help="vLLM attention backend, e.g. FLASHINFER")
    parser.add_argument("--tp", type=int, default=1, help="tensor-parallel size")
    options = parser.parse_args()
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    # An in-process engine core sees all three requests before its first step, so the steps are reproducible.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # The hooks travel to the worker processes as pickled functions.
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    llm = LLM(
        "Qwen/Qwen2.5-0.5B-Instruct",
        revision=None,
        dtype="bfloat16",
        tensor_parallel_size=options.tp,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        distributed_executor_backend="mp",
        worker_cls="auto",
        enforce_eager=False,
        max_model_len=512,
        max_num_seqs=8,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.3,
        seed=0,
        trust_remote_code=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        attention_config={"backend": options.backend},
    )
    with tempfile.TemporaryDirectory() as directory:
        kv_cache_shapes = llm.collective_rpc(install, args=(directory,))
        long = " ".join(["The quick brown fox jumps over the lazy dog."] * 15)
        outputs = llm.generate(
            ["Hello, my name is", "Paris is", long],
            [SamplingParams(n=2, max_tokens=5, seed=1), SamplingParams(max_tokens=5), SamplingParams(max_tokens=2)],
            use_tqdm=False,
        )
        ranks = [(Path(directory) / f"rank{rank}.jsonl").read_text().splitlines() for rank in range(options.tp)]
    print(f"vLLM {version('vllm')}  backend {options.backend}  tp {options.tp}")
    for output in outputs:
        print(f"request {output.request_id}: {len(output.prompt_token_ids or [])} prompt tokens")
    for rank, shape in enumerate(kv_cache_shapes):
        print(f"rank {rank}: kv cache layer 0 shape {shape}")
    print_steps(ranks[0])
    for rank, lines in enumerate(ranks[1:], 1):
        print(f"\nrank {rank}: {'same fields as rank 0' if lines == ranks[0] else 'DIFFERENT fields:'}")
        if lines != ranks[0]:
            print_steps(lines)


if __name__ == "__main__":
    main()
