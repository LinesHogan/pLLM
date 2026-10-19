"""Equal-work measurements through the public workload execution path."""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from statistics import median
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
    sampling_n: int = 16  # completions per prompt, as in the aligned throughput settings
    cuda_graphs: bool = True  # --eager switches every arm and its vanilla reference to eager execution

    @classmethod
    def parse(cls, parser: argparse.ArgumentParser | None = None) -> tuple[BenchmarkOptions, argparse.Namespace]:
        """Build options from CLI arguments; the namespace carries any arguments the caller added."""
        parser = argparse.ArgumentParser(description=__doc__) if parser is None else parser
        parser.add_argument("--model", default=cls().model)
        parser.add_argument("--batch", type=int, default=8)
        parser.add_argument("--tokens", type=int, default=256)
        parser.add_argument("--rounds", type=int, default=3)
        parser.add_argument("--warmup", type=int, default=1)
        parser.add_argument("--sampling-n", type=int, default=16)
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
        parser.add_argument(
            "--eager",
            action="store_true",
            help="run every arm without CUDA graphs and add an eager vanilla reference",
        )
        args = parser.parse_args()
        options = cls(
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
            sampling_n=args.sampling_n,
            cuda_graphs=not args.eager,
        )
        return options, args

    def __post_init__(self) -> None:
        if (
            min(
                self.batch,
                self.tokens,
                self.rounds,
                self.warmup,
                self.sampling_n,
                self.capture_memory_bytes,
                self.state_memory_bytes,
            )
            < 1
        ):
            raise ValueError("positive batch, tokens, rounds, warmup and capture/state capacities required")

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            self.model,
            max_model_len=max(512, self.tokens + 64),
            max_num_seqs=self.batch * self.sampling_n,
            max_num_batched_tokens=max(512, self.batch * self.sampling_n * 64),
            gpu_memory_utilization=self.gpu_memory_utilization,
            seed=7,
            cuda_graphs=self.cuda_graphs,
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


@dataclass(frozen=True, slots=True)
class _ProfileFootprint:
    """CUDA allocator deltas and reserved peak observed while one arm's measured rounds ran."""

    allocator_allocated_bytes: int
    allocator_freed_bytes: int
    peak_reserved_bytes: int


def _round_rates(round_tokens: tuple[int, ...], round_seconds: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(tokens / seconds for tokens, seconds in zip(round_tokens, round_seconds, strict=True))


def _vanilla(options: BenchmarkOptions, prompts: list[str], sampling: SamplingParams, cuda_graphs: bool) -> float:
    """Plain vLLM output tokens per second for the same prompts and sampling, as the median round rate."""
    llm = make_llm(replace(options.model_config(), cuda_graphs=cuda_graphs))
    for _ in range(options.warmup):
        llm.generate(prompts, sampling, use_tqdm=False)
    torch.cuda.synchronize()
    round_seconds: list[float] = []
    round_tokens: list[int] = []
    for _ in range(options.rounds):
        started = perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        round_seconds.append(perf_counter() - started)
        round_tokens.append(sum(len(completion.token_ids) for output in outputs for completion in output.outputs))
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    rates = _round_rates(tuple(round_tokens), tuple(round_seconds))
    print(
        json.dumps(
            {
                "implementation": f"vanilla_{'graph' if cuda_graphs else 'eager'}",
                "output_tokens_per_second": median(rates),
                "round_rates": rates,
            }
        )
    )
    return median(rates)


def _generate_rounds(
    handle: VllmLlm[D, R],
    prompts: list[str],
    sampling: SamplingParams,
    auxiliary: AuxiliaryInput[torch.Tensor] | None,
    indices: tuple[int, ...],
    rounds: int,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[int, ...]]:
    """Time each round separately so arms report per-round rates, not one pooled average."""
    elapsed: list[float] = []
    cpu_elapsed: list[float] = []
    round_tokens: list[int] = []
    for _ in range(rounds):
        cpu_started = process_time()
        started = perf_counter()
        outputs = handle.generate(prompts, sampling, auxiliary=auxiliary, prompt_indices=indices)
        torch.cuda.synchronize()
        elapsed.append(perf_counter() - started)
        cpu_elapsed.append(process_time() - cpu_started)
        round_tokens.append(sum(len(completion.token_ids) for output in outputs for completion in output.outputs))
    return tuple(elapsed), tuple(cpu_elapsed), tuple(round_tokens)


def _measure_arm(
    options: BenchmarkOptions,
    directory: Path | None,
    name: str,
    implementation: BackendWorkload[D, R],
    captured: bool,
    explicit: bool,
    prepare: Callable[[VllmLlm[D, R]], AuxiliaryInput[torch.Tensor]] | None,
    training_rows: int,
    require_training: bool,
    prompts: list[str],
    sampling: SamplingParams,
    indices: tuple[int, ...],
) -> tuple[BenchmarkResult, tuple[int, ...], _ProfileFootprint]:
    """Run one arm end to end and check the effective-work and cleanup contract."""
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
        if directory is None:
            elapsed, cpu_elapsed, round_tokens = _generate_rounds(
                handle, prompts, sampling, auxiliary, indices, options.rounds
            )
        else:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                profile_memory=True,
            ) as profiler:
                elapsed, cpu_elapsed, round_tokens = _generate_rounds(
                    handle, prompts, sampling, auxiliary, indices, options.rounds
                )
            profiler.export_chrome_trace(str(directory / f"{name}.json"))
        after = handle.diagnostics()
        report = BenchmarkResult(
            name,
            captured,
            cold,
            elapsed,
            sum(round_tokens),
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
            report.output_tokens != options.batch * options.sampling_n * options.tokens * options.rounds
            or require_training
            and (report.loss_count <= 0 or report.update_count <= 0 or report.trained_rows <= 0)
            or after.inference_pins
            or after.pending_state_releases
            or after.fallback_count
        ):
            raise RuntimeError("benchmark failed effective-work or cleanup contract")
        allocation_after = torch.cuda.memory_stats()
        footprint = _ProfileFootprint(
            allocation_after["allocated_bytes.all.allocated"] - allocation_before["allocated_bytes.all.allocated"],
            allocation_after["allocated_bytes.all.freed"] - allocation_before["allocated_bytes.all.freed"],
            torch.cuda.max_memory_reserved(),
        )
    return report, round_tokens, footprint


def _print_arm(report: BenchmarkResult, round_tokens: tuple[int, ...], vanilla: float) -> None:
    """Print a measured arm as one JSON line; the headline rate is the median round rate."""
    rates = _round_rates(round_tokens, report.round_seconds)
    rate = median(rates)
    print(
        json.dumps(
            {
                **asdict(report),
                "output_tokens_per_second": rate,
                "round_rates": rates,
                "vs_vanilla": rate / vanilla,
                "peak_extra_allocated_bytes": report.peak_allocated_bytes - report.baseline_allocated_bytes,
            }
        )
    )


def _print_profile(
    options: BenchmarkOptions,
    reports: list[BenchmarkResult],
    directory: Path,
    report: BenchmarkResult,
    footprint: _ProfileFootprint,
) -> None:
    """Check the profiled replay against the measured arm and print the profile summary."""
    measured = next(item for item in reports if item.implementation == report.implementation)
    if report.effective_work != measured.effective_work or report.schedule_signature != measured.schedule_signature:
        raise RuntimeError(
            f"{report.implementation}: profile reconstruction changed plan/work: "
            f"measured={measured.schedule_signature}/{measured.effective_work}, "
            f"profile={report.schedule_signature}/{report.effective_work}"
        )
    path = directory / f"{report.implementation}.json"
    activity = cuda_profile_totals(path)
    if activity.kernel_count == 0:
        raise RuntimeError("profile generation recorded no CUDA activity")
    print(
        json.dumps(
            {
                "implementation": report.implementation,
                "profile": str(path),
                **asdict(activity),
                "profile_rounds": options.rounds,
                "profile_effective_work": report.effective_work,
                "profile_round_process_cpu_seconds": report.round_process_cpu_seconds,
                "state_usage": asdict(report.state_usage),
                "profile_baseline_allocated_bytes": report.baseline_allocated_bytes,
                "profile_peak_extra_allocated_bytes": report.peak_allocated_bytes - report.baseline_allocated_bytes,
                "profile_peak_reserved_bytes": footprint.peak_reserved_bytes,
                "profile_allocator_allocated_bytes": footprint.allocator_allocated_bytes,
                "profile_allocator_freed_bytes": footprint.allocator_freed_bytes,
            }
        )
    )


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
    # Workload generate takes n=1, so each prompt is repeated sampling_n times for every arm,
    # vanilla included; without prefix caching vLLM prefills each completion of n>1 anyway.
    indices = tuple(index for index in indices for _ in range(options.sampling_n))
    prompts = [f"Explain the mathematical idea numbered {index}:" for index in indices]
    sampling = SamplingParams(
        max_tokens=options.tokens,
        temperature=0.8,
        top_p=0.95,
        top_k=-1,
        ignore_eos=True,
        seed=7,
        n=1,
    )
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "vllm": version("vllm"),
                "cuda": torch.version.cuda,
                "workload": type(workload).__name__,
                "options": asdict(options),
                "sampling_n": options.sampling_n,
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
    # The gate compares every arm against vanilla vLLM in the same CUDA-graph mode; the eager
    # reference arm only runs when the workloads themselves run eager.
    vanilla = _vanilla(options, prompts, sampling, cuda_graphs=True)
    if not options.cuda_graphs:
        vanilla = _vanilla(options, prompts, sampling, cuda_graphs=False)
    # CUPTI can affect later launches even after the profiling context exits.
    runs = tuple(
        (directory, *case)
        for directory in ((None, options.profile) if options.profile is not None else (None,))
        for case in cases
    )
    reports: list[BenchmarkResult] = []
    for directory, name, implementation, captured, explicit in runs:
        report, round_tokens, footprint = _measure_arm(
            options,
            directory,
            name,
            implementation,
            captured,
            explicit,
            prepare,
            training_rows,
            require_training,
            prompts,
            sampling,
            indices,
        )
        if directory is None:
            reports.append(report)
            _print_arm(report, round_tokens, vanilla)
        else:
            _print_profile(options, reports, directory, report, footprint)
        gc.collect()
        torch.cuda.empty_cache()
    if len({item.effective_work for item in reports}) != 1:
        raise RuntimeError("comparison arms executed different effective work")
    return tuple(reports)
