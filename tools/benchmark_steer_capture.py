"""Measure TPOT and throughput of ActivationPatch steering and async CPU export against plain vLLM.

Owned by the benchmark suite; run directly (`python -m tools.benchmark_steer_capture`) to compare single-stream or
batched decode TPOT when steering residuals and streaming captured representations to CPU.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from statistics import median
from threading import Condition
from time import perf_counter
from typing import TYPE_CHECKING, Literal

import torch
from vllm import SamplingParams

import pllm
from pllm.benchmark import BenchmarkOptions, throughput

if TYPE_CHECKING:
    from pllm.backends.vllm.runtime import VllmLlm

ArmName = Literal[
    "vanilla",
    "steer",
    "capture_sae",
    "capture_full",
    "steer_capture_sae",
    "steer_capture_full",
]
ARMS: tuple[ArmName, ...] = (
    "vanilla",
    "steer",
    "capture_sae",
    "capture_full",
    "steer_capture_sae",
    "steer_capture_full",
)


@dataclass
class _ExportCounter:
    """Thread-safe counter tracking rows materialized on CPU by `on_export`."""

    condition: Condition
    rows: int = 0

    def reset(self) -> None:
        with self.condition:
            self.rows = 0

    def record(self, event: pllm.io.TraceEvent) -> None:
        count = int(event.tensor().shape[0])
        with self.condition:
            self.rows += count
            self.condition.notify_all()

    def wait(self, target: int) -> int:
        with self.condition:
            if not self.condition.wait_for(lambda: self.rows >= target, timeout=30.0):
                raise TimeoutError(f"timed out waiting for {target} exported CPU rows (received {self.rows})")
            return self.rows


def _fill_steer(fields: Mapping[str, torch.Tensor], _generator: torch.Generator) -> None:
    fields["vector"].fill_(0.01)


def _build_workload(
    arm: ArmName, layer: int, max_rows: int, features: int, hidden_cols: int, counter: _ExportCounter
) -> pllm.Workload | None:
    """Instantiate the workload corresponding to `arm`."""
    if arm == "vanilla":
        return None
    if arm == "steer":

        class SteerOnly(pllm.Workload):
            steer = pllm.InstanceState(
                {"vector": pllm.Field(lambda width, _: (width,))},
                init=pllm.Computed(_fill_steer),
            )
            patch = pllm.model.ActivationPatch("steer", layers=(layer,), operator="add", decode=True)

        return SteerOnly()

    def make_encoder(dims: pllm.Dimensions) -> torch.nn.Module:
        return torch.nn.Linear(dims.hidden_size, features, bias=False)

    if arm in ("capture_sae", "steer_capture_sae"):
        with_steer = arm == "steer_capture_sae"

        class SteerCaptureSAE(pllm.Workload):
            if with_steer:
                steer = pllm.InstanceState(
                    {"vector": pllm.Field(lambda width, _: (width,))},
                    init=pllm.Computed(_fill_steer),
                )
                patch = pllm.model.ActivationPatch("steer", layers=(layer,), operator="add", decode=True)
            encoder = pllm.InstanceState(init=pllm.FromModule(make_encoder))

            @pllm.export(name="sae", max_rows=max_rows, max_cols=features, overflow="block")
            def capture(
                self,
                h: torch.Tensor = pllm.residual(layer),
                enc: torch.nn.Linear = pllm.module("encoder"),
            ) -> torch.Tensor:
                return torch.relu(enc(h))

            def on_export(self, event: pllm.io.TraceEvent) -> None:
                counter.record(event)

        return SteerCaptureSAE()

    with_steer = arm == "steer_capture_full"

    class SteerCaptureFull(pllm.Workload):
        if with_steer:
            steer = pllm.InstanceState(
                {"vector": pllm.Field(lambda width, _: (width,))},
                init=pllm.Computed(_fill_steer),
            )
            patch = pllm.model.ActivationPatch("steer", layers=(layer,), operator="add", decode=True)

        @pllm.export(name="hidden_full", max_rows=max_rows, max_cols=hidden_cols, overflow="block")
        def capture(self, h: torch.Tensor = pllm.residual(layer)) -> torch.Tensor:
            return h

        def on_export(self, event: pllm.io.TraceEvent) -> None:
            counter.record(event)

    return SteerCaptureFull()


def measure_arm(options: BenchmarkOptions, arm: ArmName, layer: int, features: int, hidden_cols: int) -> None:
    """Measure throughput and pure decode TPOT for one workload arm."""
    assert options.tokens >= 2, "TPOT measurement requires --tokens >= 2"
    rows = options.batch * options.sampling_n
    counter = _ExportCounter(Condition())
    workload = _build_workload(arm, layer, rows, features, hidden_cols, counter)
    has_export = "capture" in arm
    config = replace(
        options.model_config(),
        execution=pllm.ExecutionOptions(
            cuda_graph_pool_budget_bytes=options.cuda_graph_pool_budget_bytes,
            state_page_budget_bytes=options.state_page_budget_bytes,
            execution_peak_budget_bytes=options.execution_peak_budget_bytes,
            guidance_capacity=options.guidance_capacity,
        ),
    )
    prompts = [f"Explain the mathematical idea numbered {i}:" for i in range(options.batch)]
    sampling_full = SamplingParams(
        n=options.sampling_n, max_tokens=options.tokens, temperature=0.8, top_p=0.95, ignore_eos=True, seed=7
    )
    sampling_prefill = SamplingParams(
        n=options.sampling_n, max_tokens=1, temperature=0.8, top_p=0.95, ignore_eos=True, seed=7
    )

    with pllm.make_llm(config, workload=workload) as llm:
        tok_s = throughput(llm, prompts, sampling_full, warmup=options.warmup, rounds=options.rounds)
        llm.generate(prompts, sampling_prefill, use_tqdm=False)
        torch.cuda.synchronize()
        prefill_secs = _time_rounds(llm, prompts, sampling_prefill, options.rounds, counter, 0)
        expected = rows * (options.tokens - 1) if has_export else 0
        full_secs = _time_rounds(llm, prompts, sampling_full, options.rounds, counter, expected)
        dropped = llm.trace_snapshot().export_dropped if workload is not None else 0

    tpot_ms = (median(full_secs) - median(prefill_secs)) / (options.tokens - 1) * 1000.0
    print(
        json.dumps(
            {
                "arm": arm,
                "model": options.model,
                "rows": rows,
                "tokens_per_second": round(tok_s, 1),
                "tpot_ms": round(tpot_ms, 2),
                "export_dropped": dropped,
            }
        ),
        flush=True,
    )


def _time_rounds(
    llm: VllmLlm,
    prompts: list[str],
    sampling: SamplingParams,
    rounds: int,
    counter: _ExportCounter,
    expected_rows: int,
) -> list[float]:
    elapsed: list[float] = []
    for _ in range(rounds):
        counter.reset()
        started = perf_counter()
        llm.generate(prompts, sampling, use_tqdm=False)
        if expected_rows > 0:
            counter.wait(expected_rows)
        torch.cuda.synchronize()
        elapsed.append(perf_counter() - started)
    return elapsed


def main() -> None:
    """Parse CLI flags and run the requested steering/capture TPOT arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, default="steer_capture_full")
    parser.add_argument("--layer", type=int, default=14, help="residual layer index to steer and/or tap")
    parser.add_argument("--features", type=int, default=64, help="projected feature width for SAE/probe arms")
    parser.add_argument("--hidden-cols", type=int, default=3584, help="columns exported by full-hidden arms")
    options, args = BenchmarkOptions.parse(parser)
    measure_arm(options, args.arm, args.layer, args.features, args.hidden_cols)


if __name__ == "__main__":
    main()
