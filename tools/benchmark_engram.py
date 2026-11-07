"""Measure Engram-lite decode cost with explicit table placement and the shared throughput options."""

import argparse
import json
from dataclasses import dataclass, replace
from statistics import median
from time import perf_counter
from typing import Literal
from unittest.mock import patch

import torch
from vllm import SamplingParams

import pllm
from examples.engram_lite import EngramLite
from pllm.benchmark import BenchmarkOptions
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedSamplerRowBatch


@dataclass(frozen=True)
class _Identity(pllm.Workload):
    """Vanilla arithmetic at the same residual site, using the same source-operator graph mode."""

    @pllm.custom_residual(site="attention", layers=(2,))
    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


def measure(workload: pllm.Workload, options: BenchmarkOptions) -> float:
    """Count generated tokens after prefill; the wall interval includes device work and scheduling."""
    config = replace(
        options.model_config(),
        execution=pllm.ExecutionOptions(
            cuda_graph_pool_budget_bytes=options.cuda_graph_pool_budget_bytes,
            state_page_budget_bytes=options.state_page_budget_bytes,
            execution_peak_budget_bytes=options.execution_peak_budget_bytes,
        ),
    )
    sampling = SamplingParams(
        n=options.sampling_n, max_tokens=options.tokens, ignore_eos=True, temperature=0.8, top_p=0.95, top_k=-1, seed=7
    )
    prompts = ["Continue the sequence: 1, 2, 3,"] * options.batch
    rates: list[float] = []
    with pllm.make_llm(config, workload=workload) as handle:
        for _ in range(options.warmup):
            handle.generate(prompts, sampling)
        session = handle.runner.require_session()
        prepare = session.prepare_input
        started: float | None = None

        def timed_prepare(rows: PackedSamplerRowBatch) -> None:
            nonlocal started
            if (
                started is None
                and rows.row_batch.rows
                and all(row.phase is HiddenPhase.DECODE for row in rows.row_batch.rows)
            ):
                started = perf_counter()
            prepare(rows)

        with patch.object(session, "prepare_input", timed_prepare):
            for _ in range(options.rounds):
                started = None
                outputs = handle.generate(prompts, sampling)
                finished = perf_counter()
                assert started is not None, "use --tokens >= 2 to measure decode"
                tokens = sum(len(completion.token_ids) - 1 for output in outputs for completion in output.outputs)
                rates.append(tokens / (finished - started))
    print(
        json.dumps(
            {
                "workload": type(workload).__name__,
                "decode_tokens_per_second": median(rates),
                "round_rates": rates,
                "batch": options.batch,
                "sampling_n": options.sampling_n,
            }
        )
    )
    return median(rates)


def main() -> None:
    """Compare a declared placement with identity residual arithmetic at batch 1 and 8 in separate runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="offline-fitted weights from python -m examples.engram_lite")
    parser.add_argument("--buckets", type=int, default=65536)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--placement", choices=("host", "device"), default="device")
    options, args = BenchmarkOptions.parse(parser)
    placement: Literal["host", "device"] = "host" if args.placement == "host" else "device"
    baseline = measure(_Identity(), options)
    rate = measure(EngramLite(args.path, args.buckets, args.width, placement), options)
    print(
        json.dumps(
            {
                "placement": placement,
                "decode_vs_identity": rate / baseline,
                "decode_throughput_cost_percent": 100 * (1 - rate / baseline),
            }
        )
    )


if __name__ == "__main__":
    main()
