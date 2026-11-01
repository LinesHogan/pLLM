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
from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar

import torch
from vllm import SamplingParams

from pllm import make_llm
from pllm.authoring import BackendWorkload, HostBatch
from pllm.config import ExecutionOptions, ModelConfig
from pllm.contracts.codec import boundary_digest
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
            help="run every arm and the vanilla reference without CUDA graphs",
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
    cold_seconds: float
    round_seconds: tuple[float, ...]
    output_tokens: int
    loss_count: int
    update_count: int
    trained_rows: int
    peak_allocated_bytes: int
    baseline_allocated_bytes: int
    plan_digest: str
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


@dataclass(frozen=True, slots=True)
class _Requests:
    """The prompts and sampling every arm generates, vanilla included."""

    prompts: list[str]
    sampling: SamplingParams
    indices: tuple[int, ...]

    def rates(self, round_seconds: tuple[float, ...]) -> tuple[float, ...]:
        """Output tokens per second of each round; every round generates all tokens (ignore_eos)."""
        tokens = len(self.prompts) * self.sampling.n * (self.sampling.max_tokens or 0)
        return tuple(tokens / seconds for seconds in round_seconds)


@dataclass(frozen=True, slots=True)
class _Arm(Generic[D, R]):
    name: str
    workload: BackendWorkload[D, R]


@dataclass(frozen=True, slots=True)
class _Setup(Generic[D, R]):
    """What every workload arm of one comparison shares."""

    options: BenchmarkOptions
    requests: _Requests
    prepare: Callable[[VllmLlm[D, R]], HostBatch[torch.Tensor]] | None
    max_training_rows: int
    require_training: bool


def _vanilla(options: BenchmarkOptions, requests: _Requests) -> float:
    """Plain vLLM's median round rate for the same prompts and sampling, in the arms' CUDA-graph mode."""
    llm = make_llm(options.model_config())
    for _ in range(options.warmup):
        llm.generate(requests.prompts, requests.sampling, use_tqdm=False)
    torch.cuda.synchronize()
    round_seconds: list[float] = []
    for _ in range(options.rounds):
        started = perf_counter()
        llm.generate(requests.prompts, requests.sampling, use_tqdm=False)
        torch.cuda.synchronize()
        round_seconds.append(perf_counter() - started)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    rates = requests.rates(tuple(round_seconds))
    mode = "graph" if options.cuda_graphs else "eager"
    print(
        json.dumps(
            {"implementation": f"vanilla_{mode}", "output_tokens_per_second": median(rates), "round_rates": rates}
        )
    )
    return median(rates)


def _generate_rounds(
    handle: VllmLlm[D, R], requests: _Requests, host_batch: HostBatch[torch.Tensor] | None, rounds: int
) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    """Wall and process CPU seconds of each round, and the tokens generated over all rounds."""
    elapsed: list[float] = []
    cpu_elapsed: list[float] = []
    tokens = 0
    for _ in range(rounds):
        cpu_started = process_time()
        started = perf_counter()
        outputs = handle.generate(
            requests.prompts, requests.sampling, batch=host_batch, prompt_indices=requests.indices
        )
        torch.cuda.synchronize()
        elapsed.append(perf_counter() - started)
        cpu_elapsed.append(process_time() - cpu_started)
        tokens += sum(len(completion.token_ids) for output in outputs for completion in output.outputs)
    return tuple(elapsed), tuple(cpu_elapsed), tokens


def _measure_arm(
    setup: _Setup[D, R], arm: _Arm[D, R], directory: Path | None
) -> tuple[BenchmarkResult, _ProfileFootprint]:
    """Run one arm end to end and check the effective-work and cleanup contract."""
    options, requests = setup.options, setup.requests
    config = replace(
        options.model_config(),
        execution=ExecutionOptions(
            setup.max_training_rows,
            capture_memory_bytes=options.capture_memory_bytes,
            state_memory_bytes=options.state_memory_bytes,
            memory_budget_bytes=options.memory_budget_bytes,
        ),
    )
    started = perf_counter()
    with make_llm(config, workload=arm.workload) as handle:
        cold = perf_counter() - started
        host_batch = None if setup.prepare is None else setup.prepare(handle)
        for _ in range(options.warmup):
            handle.generate(requests.prompts, requests.sampling, batch=host_batch, prompt_indices=requests.indices)
        before = handle.diagnostics()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        allocation_before = torch.cuda.memory_stats()
        if directory is None:
            elapsed, cpu_elapsed, tokens = _generate_rounds(handle, requests, host_batch, options.rounds)
        else:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                profile_memory=True,
            ) as profiler:
                elapsed, cpu_elapsed, tokens = _generate_rounds(handle, requests, host_batch, options.rounds)
            profiler.export_chrome_trace(str(directory / f"{arm.name}.json"))
        after = handle.diagnostics()
        plan = handle.attachment.plan
        report = BenchmarkResult(
            arm.name,
            cold,
            elapsed,
            tokens,
            after.loss_count - before.loss_count,
            after.update_count - before.update_count,
            after.trained_rows - before.trained_rows,
            torch.cuda.max_memory_allocated(),
            baseline,
            boundary_digest(plan),
            tuple(
                (str(plan.semantic_graph.tasks[task.task_id].symbol), str(task.implementation_id))
                for task in plan.schedule.tasks
            ),
            plan.schedule.peak_memory_bytes,
            cpu_elapsed,
            after.state_usage,
        )
        assert (
            report.output_tokens == options.batch * options.sampling_n * options.tokens * options.rounds
            and (not setup.require_training or min(report.loss_count, report.update_count, report.trained_rows) > 0)
            and not (after.inference_pins or after.pending_state_releases)
        )
        allocation_after = torch.cuda.memory_stats()
        footprint = _ProfileFootprint(
            allocation_after["allocated_bytes.all.allocated"] - allocation_before["allocated_bytes.all.allocated"],
            allocation_after["allocated_bytes.all.freed"] - allocation_before["allocated_bytes.all.freed"],
            torch.cuda.max_memory_reserved(),
        )
    return report, footprint


def _print_arm(report: BenchmarkResult, requests: _Requests, vanilla: float) -> None:
    """Print a measured arm as one JSON line; the headline rate is the median round rate."""
    rates = requests.rates(report.round_seconds)
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


@dataclass(frozen=True, slots=True)
class CudaProfileTotals:
    kernel_count: int
    kernel_microseconds: float
    cuda_api_microseconds: float
    cuda_enqueue_api_microseconds: float
    cuda_wait_api_microseconds: float
    gpu_activity_microseconds: float
    memcpy_bytes: int
    cuda_launch_api_count: int
    cuda_graph_launch_api_count: int


class _TraceArgs(TypedDict):
    bytes: int


class _TraceEvent(TypedDict):
    ph: str
    cat: str
    name: str
    pid: int | str
    tid: int | str
    ts: float
    dur: float
    args: _TraceArgs


def _span_totals(spans: dict[tuple[str, str], list[tuple[float, float]]]) -> tuple[int, float]:
    count = 0
    total = 0.0
    for intervals in spans.values():
        end = float("-inf")
        for start, stop in sorted(intervals):
            count += start >= end
            total += max(0.0, stop - max(start, end))
            end = max(end, stop)
    return count, total


def cuda_profile_totals(path: Path) -> CudaProfileTotals:
    """Decode Kineto activity, merging nested API spans per thread and GPU spans per device.

    Enqueue API time is not total Python submission time and calls may block internally.
    Launch counts merge nested runtime/driver calls on each host thread; graph
    launches are a subset, not the number of device kernels executed by a graph.
    Kernel duration sums may overlap; GPU activity includes memcpy/memset without overlap.
    Memcpy bytes exclude kernel-internal loads/stores and all allocator figures.
    Foreign trace metadata is deliberately not a closed PLLM wire contract.
    """
    events: list[_TraceEvent] = json.loads(path.read_text())["traceEvents"]
    kernels, kernel_us, transfer = 0, 0.0, 0
    spans: dict[str, dict[tuple[str, str], list[tuple[float, float]]]] = {
        kind: {} for kind in ("api", "enqueue", "wait", "gpu", "launch", "graph")
    }
    for event in events:
        category = event.get("cat")
        if event.get("ph") != "X" or category not in (
            "kernel",
            "cuda_runtime",
            "cuda_driver",
            "gpu_memcpy",
            "gpu_memset",
        ):
            continue
        duration = float(event["dur"])
        start = float(event["ts"])
        device_event = category in ("kernel", "gpu_memcpy", "gpu_memset")
        key = (str(event["pid"]), "device" if device_event else str(event["tid"]))
        interval = (start, start + duration)
        spans["gpu" if device_event else "api"].setdefault(key, []).append(interval)
        if category == "kernel":
            kernels += 1
            kernel_us += duration
        elif category == "gpu_memcpy":
            transfer += int(event["args"]["bytes"])
        elif not device_event:
            name = str(event["name"]).lower()
            if "synchronize" in name:
                spans["wait"].setdefault(key, []).append(interval)
            elif "launch" in name or "async" in name and ("memcpy" in name or "memset" in name):
                spans["enqueue"].setdefault(key, []).append(interval)
                if "launch" in name:
                    spans["launch"].setdefault(key, []).append(interval)
                    if "graphlaunch" in name:
                        spans["graph"].setdefault(key, []).append(interval)
    totals = {key: _span_totals(value) for key, value in spans.items()}
    return CudaProfileTotals(
        kernels,
        kernel_us,
        totals["api"][1],
        totals["enqueue"][1],
        totals["wait"][1],
        totals["gpu"][1],
        transfer,
        totals["launch"][0],
        totals["graph"][0],
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
    assert report.effective_work == measured.effective_work and report.plan_digest == measured.plan_digest
    path = directory / f"{report.implementation}.json"
    activity = cuda_profile_totals(path)
    assert activity.kernel_count > 0
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
    prepare: Callable[[VllmLlm[D, R]], HostBatch[torch.Tensor]] | None = None,
    max_training_rows: int = 1,
    require_training: bool = False,
    prompt_indices: tuple[int, ...] | None = None,
    alternatives: tuple[tuple[str, BackendWorkload[D, R]], ...] = (),
) -> tuple[BenchmarkResult, ...]:
    """Measure ordinary model workloads; optimized arms and nonzero training are opt-in."""
    indices = tuple(range(options.batch)) if prompt_indices is None else prompt_indices
    # Each prompt's sampling_n completions are independent samples with their own sample index (and request state).
    prompts = [f"Explain the mathematical idea numbered {index}:" for index in indices]
    sampling = SamplingParams(
        max_tokens=options.tokens,
        temperature=0.8,
        top_p=0.95,
        top_k=-1,
        ignore_eos=True,
        seed=7,
        n=options.sampling_n,
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
    arms = (
        _Arm("tensor", workload),
        *(_Arm(name, implementation) for name, implementation in alternatives),
    )
    if options.profile is not None:
        options.profile.mkdir(parents=True, exist_ok=True)
    requests = _Requests(prompts, sampling, indices)
    setup = _Setup(options, requests, prepare, max_training_rows, require_training)
    vanilla = _vanilla(options, requests)
    reports: list[BenchmarkResult] = []
    # CUPTI can affect later launches even after the profiling context exits, so profiles run last.
    for directory in (None,) if options.profile is None else (None, options.profile):
        for arm in arms:
            report, footprint = _measure_arm(setup, arm, directory)
            if directory is None:
                reports.append(report)
                _print_arm(report, requests, vanilla)
            else:
                _print_profile(options, reports, directory, report, footprint)
            gc.collect()
            torch.cuda.empty_cache()
    assert len({item.effective_work for item in reports}) == 1
    return tuple(reports)
