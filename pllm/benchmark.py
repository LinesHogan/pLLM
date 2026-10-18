"""Equal-work measurements through the public workload execution path."""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from time import perf_counter, process_time
from typing import TYPE_CHECKING, TypeVar

import torch
from vllm import SamplingParams

from pllm import make_llm
from pllm.authoring import AuxiliaryInput, BackendWorkload
from pllm.config import ExecutionOptions, ModelConfig
from pllm.execution.codec import cuda_profile_totals
from pllm.execution.ptts.page_bundle import PageBundleUsage

if TYPE_CHECKING:
    from pllm.backends.vllm.runtime import VllmLlm

D = TypeVar("D")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    batch: int = 8
    tokens: int = 256
    rounds: int = 3
    warmup: int = 1
    gpu_memory_utilization: float = 0.25
    profile: Path | None = None
    capture_memory_bytes: int = 256 * 1024**2
    state_memory_bytes: int = 256 * 1024**2
    memory_budget_bytes: int | None = None

    @classmethod
    def parse(cls, parser: argparse.ArgumentParser | None = None) -> BenchmarkOptions:
        parser = argparse.ArgumentParser(description=__doc__) if parser is None else parser
        parser.add_argument("--model", default=cls().model)
        parser.add_argument("--batch", type=int, default=8)
        parser.add_argument("--tokens", type=int, default=256)
        parser.add_argument("--rounds", type=int, default=3)
        parser.add_argument("--warmup", type=int, default=1)
        parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
        parser.add_argument(
            "--profile", type=Path, help="directory for a separate replay of the measured rounds per arm"
        )
        parser.add_argument(
            "--capture-memory-mib", type=int, default=256, help="explicit resident capacity for each captured task"
        )
        parser.add_argument(
            "--state-memory-mib",
            type=int,
            default=256,
            help="fixed signed state capacity for timing and profile reconstruction",
        )
        parser.add_argument(
            "--execution-memory-mib",
            type=int,
            help="original AER declared-memory limit, excluding undeclared model/KV/library costs",
        )
        args = parser.parse_args()
        return cls(
            args.model,
            args.batch,
            args.tokens,
            args.rounds,
            args.warmup,
            args.gpu_memory_utilization,
            args.profile,
            args.capture_memory_mib * 1024**2,
            args.state_memory_mib * 1024**2,
            None if args.execution_memory_mib is None else args.execution_memory_mib * 1024**2,
        )

    def __post_init__(self) -> None:
        if (
            min(self.batch, self.tokens, self.rounds, self.warmup, self.capture_memory_bytes, self.state_memory_bytes)
            < 1
        ):
            raise ValueError("positive batch, tokens, rounds, warmup and capture/state capacities required")

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            self.model,
            max_model_len=max(512, self.tokens + 64),
            max_num_seqs=self.batch,
            max_num_batched_tokens=max(512, self.batch * 64),
            gpu_memory_utilization=self.gpu_memory_utilization,
            seed=7,
        )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    implementation: str
    captured: bool
    cold_seconds: float
    round_seconds: tuple[float, ...]
    output_tokens: int
    loss_count: int
    update_count: int
    trained_rows: int
    peak_allocated_bytes: int
    baseline_allocated_bytes: int
    schedule_signature: str
    explicit_math: bool
    task_implementations: tuple[tuple[str, str], ...]
    planned_peak_memory_bytes: int
    round_process_cpu_seconds: tuple[float, ...]
    state_usage: PageBundleUsage

    @property
    def effective_work(self) -> tuple[int, int, int, int]:
        return self.output_tokens, self.loss_count, self.update_count, self.trained_rows


def compare(
    workload: BackendWorkload[D, R],
    options: BenchmarkOptions,
    *,
    prepare: Callable[[VllmLlm[D, R]], AuxiliaryInput[torch.Tensor]] | None = None,
    training_rows: int = 1,
    explicit_math: bool = False,
    capture: bool = False,
    require_training: bool = False,
    prompt_indices: tuple[int, ...] | None = None,
    alternatives: tuple[tuple[str, BackendWorkload[D, R], bool], ...] = (),
) -> tuple[BenchmarkResult, ...]:
    """Measure ordinary model workloads; optimized arms and nonzero training are opt-in."""
    indices = tuple(range(options.batch)) if prompt_indices is None else prompt_indices
    if len(indices) != options.batch or any(index < 0 for index in indices):
        raise ValueError("benchmark prompt indices must cover the original request batch")
    prompts = [f"Explain the mathematical idea numbered {index}:" for index in indices]
    sampling = SamplingParams(
        max_tokens=options.tokens, temperature=0.8, top_p=0.95, top_k=-1, ignore_eos=True, seed=7, n=1
    )
    reports: list[BenchmarkResult] = []
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "vllm": version("vllm"),
                "cuda": torch.version.cuda,
                "workload": type(workload).__name__,
                "options": asdict(options),
                "sampling_n": 1,
                "prompt_indices": indices,
            },
            default=str,
        )
    )
    cases = (
        ("tensor", workload, False, False),
        *((("tensor_explicit", workload, False, True),) if explicit_math else ()),
        *((("tensor_capture", workload, True, False),) if capture else ()),
        *((name, implementation, captured, False) for name, implementation, captured in alternatives),
    )
    if len({name for name, _, _, _ in cases}) != len(cases):
        raise ValueError("benchmark implementation labels must be distinct")
    if options.profile is not None:
        options.profile.mkdir(parents=True, exist_ok=True)
        for name, _, _, _ in cases:
            path = options.profile / f"{name}.json"
            if path.exists():
                raise FileExistsError(f"profile output already exists: {path}")
    # CUPTI can affect later launches even after the profiling context exits.
    runs = tuple(
        (directory, *case)
        for directory in ((None, options.profile) if options.profile is not None else (None,))
        for case in cases
    )
    for directory, name, implementation, captured, explicit in runs:
        config = replace(
            options.model_config(),
            execution=ExecutionOptions(
                captured,
                training_rows,
                capture_memory_bytes=options.capture_memory_bytes,
                explicit_math=explicit,
                state_memory_bytes=options.state_memory_bytes,
                memory_budget_bytes=options.memory_budget_bytes,
            ),
        )
        started = perf_counter()
        with make_llm(config, workload=implementation) as handle:
            cold = perf_counter() - started
            auxiliary = None if prepare is None else prepare(handle)
            for _ in range(options.warmup):
                handle.generate(prompts, sampling, auxiliary=auxiliary, prompt_indices=indices)
            before = handle.diagnostics()
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            allocation_before = torch.cuda.memory_stats()

            def generate_rounds() -> tuple[tuple[float, ...], tuple[float, ...], int]:
                elapsed: list[float] = []
                cpu_elapsed: list[float] = []
                tokens = 0
                for _ in range(options.rounds):
                    cpu_started = process_time()
                    started = perf_counter()
                    outputs = handle.generate(prompts, sampling, auxiliary=auxiliary, prompt_indices=indices)  # noqa: F821 -- runs before the del below
                    torch.cuda.synchronize()
                    elapsed.append(perf_counter() - started)
                    cpu_elapsed.append(process_time() - cpu_started)
                    tokens += sum(len(completion.token_ids) for output in outputs for completion in output.outputs)
                return tuple(elapsed), tuple(cpu_elapsed), tokens

            if directory is None:
                elapsed, cpu_elapsed, tokens = generate_rounds()
            else:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    profile_memory=True,
                ) as profiler:
                    elapsed, cpu_elapsed, tokens = generate_rounds()
                profiler.export_chrome_trace(str(directory / f"{name}.json"))
            after = handle.diagnostics()
            report = BenchmarkResult(
                name,
                captured,
                cold,
                elapsed,
                tokens,
                after.loss_count - before.loss_count,
                after.update_count - before.update_count,
                after.trained_rows - before.trained_rows,
                torch.cuda.max_memory_allocated(),
                baseline,
                handle.attachment.physical.schedule_signature,
                explicit,
                tuple(
                    (
                        str(handle.attachment.inputs.plan.semantic_graph.tasks[task.task_id].symbol),
                        str(task.implementation_id),
                    )
                    for task in handle.attachment.inputs.plan.schedule_physical.tasks
                ),
                handle.attachment.inputs.plan.schedule_decision.peak_memory_bytes,
                cpu_elapsed,
                after.state_usage,
            )
            if (
                tokens != options.batch * options.tokens * options.rounds
                or require_training
                and (report.loss_count <= 0 or report.update_count <= 0 or report.trained_rows <= 0)
                or after.inference_pins
                or after.pending_state_releases
                or after.fallback_count
            ):
                raise RuntimeError("benchmark failed effective-work or cleanup contract")
            if directory is None:
                reports.append(report)
                print(
                    json.dumps(
                        {
                            **asdict(report),
                            "output_tokens_per_second": tokens / sum(elapsed),
                            "peak_extra_allocated_bytes": report.peak_allocated_bytes - baseline,
                        }
                    )
                )
            else:
                measured = next(item for item in reports if item.implementation == name)
                if (
                    report.effective_work != measured.effective_work
                    or report.schedule_signature != measured.schedule_signature
                ):
                    raise RuntimeError(
                        f"{name}: profile reconstruction changed plan/work: "
                        f"measured={measured.schedule_signature}/{measured.effective_work}, "
                        f"profile={report.schedule_signature}/{report.effective_work}"
                    )
                path = directory / f"{name}.json"
                activity = cuda_profile_totals(path)
                allocation_after = torch.cuda.memory_stats()
                if activity.kernel_count == 0:
                    raise RuntimeError("profile generation recorded no CUDA activity")
                print(
                    json.dumps(
                        {
                            "implementation": name,
                            "profile": str(path),
                            **asdict(activity),
                            "profile_rounds": options.rounds,
                            "profile_effective_work": report.effective_work,
                            "profile_round_process_cpu_seconds": cpu_elapsed,
                            "state_usage": asdict(report.state_usage),
                            "profile_baseline_allocated_bytes": baseline,
                            "profile_peak_extra_allocated_bytes": report.peak_allocated_bytes - baseline,
                            "profile_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                            "profile_allocator_allocated_bytes": allocation_after["allocated_bytes.all.allocated"]
                            - allocation_before["allocated_bytes.all.allocated"],
                            "profile_allocator_freed_bytes": allocation_after["allocated_bytes.all.freed"]
                            - allocation_before["allocated_bytes.all.freed"],
                        }
                    )
                )
        del handle, auxiliary
        gc.collect()
        torch.cuda.empty_cache()
    if len({item.effective_work for item in reports}) != 1:
        raise RuntimeError("comparison arms executed different effective work")
    return tuple(reports)
