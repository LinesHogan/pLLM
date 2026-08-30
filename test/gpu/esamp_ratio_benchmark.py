#!/usr/bin/env python3
"""README-ratio benchmark: ESamp min-p active path vs raw vLLM baseline.

Reproduces the README's "aligned min-p ESamp benchmark" shape on the new
architecture: CUDA graphs ON, FlashInfer sampler ON, bf16, prefix caching
disabled for fairness, batch=8, n=16, min-p sampling, 128 new tokens.

Usage (one process per arm):
    python test/gpu/esamp_ratio_benchmark.py run --mode raw   --runner mrv2 --out raw.json
    python test/gpu/esamp_ratio_benchmark.py run --mode capture_only --runner mrv2 --out capture.json
    python test/gpu/esamp_ratio_benchmark.py run --mode tllm  --runner mrv2 --out tllm.json
    python test/gpu/esamp_ratio_benchmark.py capture-overhead raw.json capture.json
    python test/gpu/esamp_ratio_benchmark.py ratio raw.json tllm.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

PROMPTS = [
    "Give me a short introduction to large language models.",
    "Explain the theory of relativity to a high school student.",
    "Write a short story about a robot learning to paint.",
    "Describe the process of photosynthesis step by step.",
    "What are the main causes of climate change?",
    "Explain how a neural network learns from data.",
    "Summarize the history of the Roman Empire.",
    "How does the human immune system fight viruses?",
]

BATCH = 8
N_SAMPLES = 16
ACTIVE_SEQUENCES = BATCH * N_SAMPLES
MAX_NEW_TOKENS = 128
MAX_NUM_BATCHED_TOKENS = 4096
MIN_FULL_DECODE_RUN = MAX_NEW_TOKENS - 1
REQUIRED_RATIO = 0.98
STRETCH_RATIO = 0.995


def _output_token_count(outputs: Sequence[Any]) -> int:
    return sum(len(output.token_ids) for request in outputs for output in request.outputs)


def _workload_fingerprint(args: argparse.Namespace, prompts: Sequence[str]) -> dict[str, Any]:
    prompt_payload = json.dumps(list(prompts), ensure_ascii=True, separators=(",", ":"))
    return {
        "schema_version": 4,
        "model": str(args.model),
        "runner": str(args.runner),
        "mp": str(args.mp),
        "dtype": "bfloat16",
        "batch_size": len(prompts),
        "sampling_n": N_SAMPLES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 1.0,
        "min_p": 0.05,
        "ignore_eos": True,
        "max_model_len": 1024,
        "max_num_seqs": BATCH * N_SAMPLES,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_cudagraph_capture_size": ACTIVE_SEQUENCES,
        "enable_prefix_caching": False,
        "enforce_eager": False,
        "engine_seed": 1234,
        "warmup_rounds": int(args.warmup_rounds),
        "esamp_training_mode": "shared_first_per_prompt",
        "request_mapping": True,
        "dispatch_post_generate": False,
        "prompt_sha256": hashlib.sha256(prompt_payload.encode("ascii")).hexdigest(),
    }


def _summarize_rounds(durations: Sequence[float], token_counts: Sequence[int]) -> dict[str, Any]:
    if not durations or len(durations) != len(token_counts):
        raise ValueError("round durations and token counts must be non-empty and aligned")
    if any(float(duration) <= 0 for duration in durations):
        raise ValueError("round durations must be positive")
    throughputs = [int(tokens) / float(duration) for duration, tokens in zip(durations, token_counts)]
    elapsed = float(sum(float(duration) for duration in durations))
    output_tokens = int(sum(int(tokens) for tokens in token_counts))
    return {
        "elapsed_s": elapsed,
        "output_tokens": output_tokens,
        "out_tok_per_s": output_tokens / elapsed,
        "round_durations_s": [float(duration) for duration in durations],
        "round_output_tokens": [int(tokens) for tokens in token_counts],
        "round_out_tok_per_s": throughputs,
        "median_round_out_tok_per_s": float(statistics.median(throughputs)),
    }


class _SchedulerTraceCollector:
    """Process-local trace of the real vLLM scheduler output.

    Formal ratio runs use ``mp=0``, so the scheduler and this collector share
    one process. The wrapper is restored before the benchmark exits.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._trace: list[dict[str, int]] = []
        self._remaining_prompt_tokens: dict[str, int] = {}
        self._scheduler_type: type[Any] | None = None
        self._original_schedule: Any = None

    def install(self) -> None:
        from vllm.v1.core.sched.scheduler import Scheduler

        original = Scheduler.schedule
        collector = self

        def _traced_schedule(scheduler: Any, *args: Any, **kwargs: Any) -> Any:
            output = original(scheduler, *args, **kwargs)
            collector._record(output)
            return output

        Scheduler.schedule = _traced_schedule  # type: ignore[method-assign]
        self._scheduler_type = Scheduler
        self._original_schedule = original

    def restore(self) -> None:
        if self._scheduler_type is not None and self._original_schedule is not None:
            self._scheduler_type.schedule = self._original_schedule
        self._scheduler_type = None
        self._original_schedule = None

    def clear(self) -> None:
        with self._lock:
            self._trace.clear()
            self._remaining_prompt_tokens.clear()

    def snapshot(self) -> list[dict[str, int]]:
        with self._lock:
            return [dict(row) for row in self._trace]

    def _record(self, output: Any) -> None:
        with self._lock:
            for request in getattr(output, "scheduled_new_reqs", ()):
                request_id = str(getattr(request, "req_id"))
                token_ids = getattr(request, "prompt_token_ids", None)
                if token_ids is None:
                    token_ids = getattr(request, "prefill_token_ids", None)
                prompt_tokens = len(token_ids or ())
                computed_tokens = int(getattr(request, "num_computed_tokens", 0) or 0)
                self._remaining_prompt_tokens[request_id] = max(
                    0,
                    prompt_tokens - computed_tokens,
                )

            context_requests = 0
            context_tokens = 0
            generation_requests = 0
            generation_tokens = 0
            for request_id, scheduled_tokens in getattr(
                output,
                "num_scheduled_tokens",
                {},
            ).items():
                scheduled = int(scheduled_tokens)
                remaining = int(self._remaining_prompt_tokens.get(str(request_id), 0))
                context = min(scheduled, remaining)
                generation = scheduled - context
                if context > 0:
                    context_requests += 1
                    context_tokens += context
                if generation > 0:
                    generation_requests += 1
                    generation_tokens += generation
                self._remaining_prompt_tokens[str(request_id)] = remaining - context

            self._trace.append(
                {
                    "num_ctx_requests": context_requests,
                    "num_ctx_tokens": context_tokens,
                    "num_generation_requests": generation_requests,
                    "num_generation_tokens": generation_tokens,
                    "num_preemptions": len(
                        getattr(output, "preempted_req_ids", None) or ()
                    ),
                }
            )
            for request_id in getattr(output, "finished_req_ids", ()):
                self._remaining_prompt_tokens.pop(str(request_id), None)


def _summarize_scheduler_trace(
    trace: Sequence[Mapping[str, Any]],
    *,
    max_num_seqs: int,
    max_num_scheduled_tokens: int,
) -> dict[str, Any]:
    histogram: dict[str, int] = {}
    longest_full_decode_run = 0
    current_full_decode_run = 0
    max_generation_requests = 0
    total_preemptions = 0
    token_request_mismatch_count = 0
    for row in trace:
        ctx_requests = int(row.get("num_ctx_requests", 0))
        ctx_tokens = int(row.get("num_ctx_tokens", 0))
        generation_requests = int(row.get("num_generation_requests", 0))
        generation_tokens = int(row.get("num_generation_tokens", 0))
        total_preemptions += int(row.get("num_preemptions", 0))
        max_generation_requests = max(max_generation_requests, generation_requests)
        if generation_requests != generation_tokens:
            token_request_mismatch_count += 1
        shape = (
            f"ctx_reqs={ctx_requests},ctx_tokens={ctx_tokens},"
            f"gen_reqs={generation_requests},gen_tokens={generation_tokens}"
        )
        histogram[shape] = histogram.get(shape, 0) + 1
        if (
            ctx_requests == 0
            and generation_requests == int(max_num_seqs)
            and generation_tokens == int(max_num_seqs)
        ):
            current_full_decode_run += 1
            longest_full_decode_run = max(
                longest_full_decode_run,
                current_full_decode_run,
            )
        else:
            current_full_decode_run = 0
    return {
        "iteration_count": len(trace),
        "max_num_seqs": int(max_num_seqs),
        "max_num_scheduled_tokens": int(max_num_scheduled_tokens),
        "max_generation_requests": max_generation_requests,
        "longest_full_decode_run": longest_full_decode_run,
        "total_preemptions": total_preemptions,
        "token_request_mismatch_count": token_request_mismatch_count,
        "iteration_shape_histogram": histogram,
    }


def _validate_scheduler_comparable(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    raw_trace = baseline.get("scheduler_diagnostics")
    candidate_trace = candidate.get("scheduler_diagnostics")
    if not isinstance(raw_trace, Mapping) or not isinstance(candidate_trace, Mapping):
        workload = baseline.get("workload", {})
        mp_mode = str(workload.get("mp", "0")) if isinstance(workload, Mapping) else "0"
        if mp_mode == "1" and raw_trace is None and candidate_trace is None:
            return
        raise ValueError("formal mp=0 ratio requires scheduler diagnostics")
    if raw_trace.get("iteration_shape_histogram") != candidate_trace.get(
        "iteration_shape_histogram"
    ):
        raise ValueError("scheduler iteration shapes differ")
    for label, diagnostics in (("raw", raw_trace), ("candidate", candidate_trace)):
        if int(diagnostics.get("longest_full_decode_run", 0)) < MIN_FULL_DECODE_RUN:
            raise ValueError(f"{label} arm did not sustain the batch-full decode plateau")
        if int(diagnostics.get("total_preemptions", 0)) != 0:
            raise ValueError(f"{label} arm observed scheduler preemption")
        if int(diagnostics.get("token_request_mismatch_count", 0)) != 0:
            raise ValueError(f"{label} arm mixed non-unit work into decode iterations")


def _validate_comparable(raw: Mapping[str, Any], tllm: Mapping[str, Any]) -> None:
    candidate_mode = tllm.get("mode")
    if raw.get("mode") != "raw" or candidate_mode not in ("tllm", "native"):
        raise ValueError("ratio inputs must be ordered as raw then tllm/native")
    if raw.get("workload") != tllm.get("workload"):
        raise ValueError("raw and tllm workload fingerprints differ")
    _validate_scheduler_comparable(raw, tllm)
    if not bool(tllm.get("training_enabled", False)):
        raise ValueError("formal tllm ratio requires training_enabled")
    if not bool(tllm.get("intervention_enabled", False)):
        raise ValueError("formal tllm ratio requires intervention_enabled")
    loss_count = int(tllm.get("esamp_loss_count", 0))
    if loss_count <= 0:
        raise ValueError("tllm benchmark did not observe training: esamp_loss_count must be > 0")
    expected_loss_count = int(tllm.get("expected_esamp_loss_count", 0))
    minimum_loss_count = int(tllm.get("minimum_esamp_loss_count", 0))
    if (
        expected_loss_count < minimum_loss_count
        or minimum_loss_count <= 0
        or loss_count != expected_loss_count
    ):
        raise ValueError(
            "tllm benchmark training density is incomplete: "
            f"esamp_loss_count={loss_count} expected={expected_loss_count} "
            f"minimum={minimum_loss_count}"
        )
    candidate_sample_count = int(tllm.get("candidate_sample_count", 0))
    if candidate_sample_count <= 0:
        raise ValueError(
            "tllm benchmark did not exercise sampler intervention: "
            "candidate_sample_count must be > 0"
        )
    if int(tllm.get("candidate_row_count", 0)) < (
        candidate_sample_count * ACTIVE_SEQUENCES
    ):
        raise ValueError(
            "tllm benchmark candidate row density is incomplete: "
            f"rows={int(tllm.get('candidate_row_count', 0))} "
            f"samples={candidate_sample_count} active_sequences={ACTIVE_SEQUENCES}"
        )
    if int(tllm.get("candidate_kernel_triton_count", 0)) <= 0:
        raise ValueError("tllm benchmark did not exercise the Triton candidate kernel")
    if int(tllm.get("candidate_kernel_fallback_count", 0)) > 0:
        raise ValueError("tllm benchmark used a candidate-kernel fallback")
    if candidate_mode == "native":
        workgraph = tllm.get("native_workgraph", {})
        if (
            int(workgraph.get("execute_count", 0)) <= 0
            or int(workgraph.get("eager_count", 0)) <= 0
            or int(workgraph.get("replay_count", 0)) <= 0
            or int(workgraph.get("fallback_count", 0)) != 0
        ):
            raise ValueError(
                "native benchmark did not remain on its signed mixed eager/replay path"
            )
    else:
        placement = tllm.get("placement", {})
        required_positive = (
            "moved_dispatch_count",
            "bubble_drain_count",
            "fusion_execution_count",
            "fusion_replay_count",
            "capture_resident_stage_count",
            "capture_resident_stage_install_count",
            "step_lease_retire_count",
        )
        missing = [
            name
            for name in required_positive
            if int(placement.get(name, 0)) <= 0
        ]
        if missing:
            raise ValueError(
                "tllm benchmark did not exercise compiled DAG/fusion placement: "
                + ", ".join(missing)
            )
        forbidden_positive = (
            "deadline_flush_count",
            "fusion_fallback_count",
            "fusion_eager_fallback_count",
        )
        fallback = [
            name
            for name in forbidden_positive
            if int(placement.get(name, 0)) > 0
        ]
        if fallback:
            raise ValueError(
                "tllm benchmark used a compiler/fusion fallback during measured rounds: "
                + ", ".join(fallback)
            )


def _capture_overhead(
    raw: Mapping[str, Any],
    capture_only: Mapping[str, Any],
) -> dict[str, float]:
    """Validate and quantify the fixed capture-resident graph tax."""
    if raw.get("mode") != "raw" or capture_only.get("mode") != "capture_only":
        raise ValueError("capture diagnostic inputs must be ordered raw then capture_only")
    if raw.get("workload") != capture_only.get("workload"):
        raise ValueError("raw and capture-only workload fingerprints differ")
    _validate_scheduler_comparable(raw, capture_only)
    if bool(capture_only.get("training_enabled", True)):
        raise ValueError("capture-only diagnostic unexpectedly enabled training")
    if bool(capture_only.get("intervention_enabled", True)):
        raise ValueError("capture-only diagnostic unexpectedly enabled intervention")
    if int(capture_only.get("esamp_loss_count", 0)) != 0:
        raise ValueError("capture-only diagnostic unexpectedly observed training")
    if int(capture_only.get("candidate_sample_count", 0)) != 0:
        raise ValueError("capture-only diagnostic unexpectedly sampled candidates")
    placement = capture_only.get("placement", {})
    for name in (
        "capture_resident_stage_count",
        "capture_resident_stage_install_count",
    ):
        if int(placement.get(name, 0)) <= 0:
            raise ValueError(
                f"capture-only diagnostic did not install the graph producer: {name}"
            )
    raw_throughput = float(raw.get("out_tok_per_s", 0.0))
    capture_throughput = float(capture_only.get("out_tok_per_s", 0.0))
    if raw_throughput <= 0.0 or capture_throughput <= 0.0:
        raise ValueError("capture diagnostic throughput must be positive")
    return {
        "capture_ratio": capture_throughput / raw_throughput,
        "raw_us_per_step": ACTIVE_SEQUENCES / raw_throughput * 1e6,
        "capture_us_per_step": ACTIVE_SEQUENCES / capture_throughput * 1e6,
        "capture_fixed_tax_us_per_step": (
            ACTIVE_SEQUENCES / capture_throughput
            - ACTIVE_SEQUENCES / raw_throughput
        )
        * 1e6,
    }


def _paired_acceptance(
    raw_runs: Sequence[Mapping[str, Any]],
    tllm_runs: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    if len(raw_runs) != len(tllm_runs) or len(raw_runs) < 5:
        raise ValueError("paired acceptance requires at least five aligned raw/tllm blocks")
    ratios: list[float] = []
    for raw, tllm in zip(raw_runs, tllm_runs):
        _validate_comparable(raw, tllm)
        ratios.append(float(tllm["out_tok_per_s"]) / float(raw["out_tok_per_s"]))
    logs = [math.log(ratio) for ratio in ratios]
    mean_log = statistics.fmean(logs)
    # Two-sided 95% Student-t critical values. Acceptance intentionally
    # requires >=5 blocks; n>=30 uses the normal limit.
    t95 = {
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
    }.get(len(logs) - 1, 1.960)
    stderr = statistics.stdev(logs) / math.sqrt(len(logs))
    return {
        "median_ratio": float(statistics.median(ratios)),
        "geomean_ratio": float(math.exp(mean_log)),
        "ci95_lower": float(math.exp(mean_log - t95 * stderr)),
        "ci95_upper": float(math.exp(mean_log + t95 * stderr)),
    }


def run(args: argparse.Namespace) -> None:
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "mrv2" else "0"
    # Pin the process topology BEFORE any tllm/vllm import so compared pairs
    # differ only in the tllm consumer, not in mp mode.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = args.mp

    scheduler_trace = _SchedulerTraceCollector() if args.mp == "0" else None
    if scheduler_trace is not None:
        scheduler_trace.install()

    from vllm import SamplingParams

    # No per-request seed: a seeded SamplingParams gives every request its own
    # torch.Generator, and the exact-candidate fast path (correctly) refuses
    # to bypass generator-based RNG. The engine-level seed still keeps the
    # global RNG reproducible.
    params = SamplingParams(
        temperature=1.0,
        min_p=0.05,
        n=N_SAMPLES,
        max_tokens=MAX_NEW_TOKENS,
        ignore_eos=True,
    )
    prompts = PROMPTS[:BATCH]

    native_runtime = None
    if args.mode == "native":
        os.environ["TLLM_ENABLE_EXACT_CANDIDATE_SAMPLING"] = "1"
        os.environ["TLLM_ENABLE_TRITON_MINP_KEEP_MASK"] = "1"
        from tllm.consumers.esamp import ESampConfig, ESampWorkload
        from tllm.runtime import residual_runtime as core
        from tllm.runtime.runtime import TllmRuntime

        workload = ESampWorkload(
            ESampConfig(
                enabled=True,
                intervention=not args.no_intervention,
                distiller_hidden_dim=128,
                distiller_lr=1e-3,
                distiller_beta=0.5,
                training_row_selection="first_per_prompt",
            )
        )
        native_runtime = TllmRuntime(
            workloads=(workload,),
            workload_memory_budget_bytes=256 << 20,
        )
        llm = native_runtime.make_llm(
            model_name=args.model,
            dtype="bfloat16",
            gpu_memory_utilization=float(args.gpu_mem),
            max_model_len=1024,
            max_num_seqs=BATCH * N_SAMPLES,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_cudagraph_capture_size=ACTIVE_SEQUENCES,
            enable_prefix_caching=False,
            enforce_eager=False,
            seed=1234,
        )

        def generate_once():
            return native_runtime.generate_requests(
                llm,
                prompts,
                params,
                prompt_indices=tuple(range(len(prompts))),
            )

    elif args.mode != "raw":
        # Optimized ESamp path: exact candidate sampling + triton min-p
        # keep-mask kernel (README's "triton kernel" arm).
        os.environ["TLLM_ENABLE_EXACT_CANDIDATE_SAMPLING"] = "1"
        os.environ["TLLM_ENABLE_TRITON_MINP_KEEP_MASK"] = "1"
        from tllm.runtime import residual_runtime as core
        from tllm.workflows import esamp_support as esamp_workflow_support

        graph_scratch_rows = max(64, BATCH * N_SAMPLES)
        esamp_workflow_support.configure_esamp_runtime(
            graph_scratch_rows=graph_scratch_rows,
            enable_esamp_training=True,
            per_request_models=False,
            per_request_model_bank=False,
            training_row_selection="first_per_prompt",
            model_bank_slots=BATCH,
            model_bank_train_cudagraph=True,
            enable_distiller_intervention=(
                args.mode == "tllm" and not args.no_intervention
            ),
            distiller_beta=0.5,
            distiller_sampler_backend="post_filter_exact",
        )
        llm = core.make_llm(
            model_name=args.model,
            dtype="bfloat16",
            gpu_memory_utilization=float(args.gpu_mem),
            max_model_len=1024,
            max_num_seqs=BATCH * N_SAMPLES,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_cudagraph_capture_size=ACTIVE_SEQUENCES,
            enable_prefix_caching=False,
            enforce_eager=False,
            seed=1234,
        )
        if args.mode == "capture_only":
            # Keep the already-installed capture-resident layer wrappers in
            # vLLM's graph while removing the ESamp flow from dispatch. This
            # isolates the two fixed-shape hidden gathers and tLLM's no-op
            # runtime hooks from training, delivery, and sampler intervention.
            core.set_esamp_training_enabled(False)
            core.synchronize_esamp()

        def generate_once():
            return esamp_workflow_support.run_generate_with_request_mapping(
                llm,
                prompts,
                params,
                request_prompt_indices=tuple(range(len(prompts))),
                use_tqdm=False,
                dispatch_post_generate=False,
            )
    else:
        from vllm import LLM

        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            gpu_memory_utilization=float(args.gpu_mem),
            max_model_len=1024,
            max_num_seqs=BATCH * N_SAMPLES,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_cudagraph_capture_size=ACTIVE_SEQUENCES,
            enable_prefix_caching=False,
            enforce_eager=False,
            seed=1234,
        )

        def generate_once():
            return llm.generate(prompts, params, use_tqdm=False)

    round_durations: list[float] = []
    round_token_counts: list[int] = []
    tllm_metrics: dict[str, Any] = {}
    warmup_placement = None
    try:
        # Warm every lazy pipeline/CUDA-graph slot before measuring. Both arms
        # use the same count so the comparison is a steady-state ratio.
        for _ in range(max(1, int(args.warmup_rounds))):
            generate_once()
        if args.mode != "raw":
            core.read_and_reset_distiller_timing_stats()
            core.read_and_reset_esamp_stats()
            core.read_and_reset_path_hotspot_stats()
            warmup_placement = core.read_and_reset_placement_stats()
        if scheduler_trace is not None:
            scheduler_trace.clear()
        rounds = max(1, int(args.rounds))
        for _ in range(rounds):
            t0 = time.perf_counter()
            outputs = generate_once()
            round_durations.append(time.perf_counter() - t0)
            round_token_counts.append(_output_token_count(outputs))
        if args.mode != "raw":
            # Read child-owned stats while the EngineCore process is alive.
            stats = core.read_and_reset_esamp_stats()
            timing = core.read_and_reset_distiller_timing_stats()
            hotspots = core.read_and_reset_path_hotspot_stats()
            placement = core.read_and_reset_placement_stats()
            if warmup_placement is not None:
                placement.capture_resident_stage_count = max(
                    int(placement.capture_resident_stage_count),
                    int(warmup_placement.capture_resident_stage_count),
                )
                placement.capture_resident_stage_install_count += int(
                    warmup_placement.capture_resident_stage_install_count
                )
            graph = (
                None
                if args.mode == "native"
                else core.read_graph_debug_stats(mode="shared")
            )
            loss_count = int(getattr(stats, "loss_count", 0))
            training_enabled = args.mode in ("tllm", "native")
            if training_enabled and loss_count <= 0:
                raise RuntimeError(
                    "ESamp benchmark completed without observed training; "
                    "refusing a throughput result with esamp_loss_count == 0"
                )
            tllm_metrics = {
                "training_enabled": training_enabled,
                "intervention_enabled": bool(
                    training_enabled and not args.no_intervention
                ),
                "esamp_loss_count": loss_count,
                "expected_esamp_loss_count": (
                    max(1, int(args.rounds)) * max(0, MAX_NEW_TOKENS - 1)
                    if args.mode == "native"
                    else int(placement.moved_dispatch_count)
                ),
                "minimum_esamp_loss_count": (
                    max(1, int(args.rounds)) * max(0, MAX_NEW_TOKENS - 1)
                    if training_enabled
                    else 0
                ),
                "candidate_sample_count": int(timing.candidate_sample_count),
                "candidate_row_count": int(timing.candidate_row_count),
                "candidate_token_count": int(timing.candidate_token_count),
                "candidate_kernel_triton_count": int(
                    timing.candidate_kernel_triton_count
                ),
                "candidate_kernel_torch_count": int(
                    timing.candidate_kernel_torch_count
                ),
                "candidate_kernel_fallback_count": int(
                    timing.candidate_kernel_fallback_count
                ),
                "training_graph": (
                    {}
                    if graph is None
                    else {
                        "capture_state": str(graph.capture_state),
                        "replay_attempt_count": int(graph.replay_attempt_count),
                        "replay_hit_count": int(graph.replay_hit_count),
                        "kernel_fallback_count": int(graph.kernel_fallback_count),
                    }
                ),
                "distiller_timing": {
                    key: getattr(timing, key)
                    for key in (
                        "precompute_ms_total",
                        "precompute_count",
                        "wait_ms_total",
                        "wait_count",
                        "fallback_ms_total",
                        "fallback_count",
                        "port_publish_attempt_count",
                        "port_publish_hit_count",
                    )
                },
                "path_hotspots_ms": dict(hotspots.cpu_ms_total),
                "placement": {
                    key: getattr(placement, key)
                    for key in (
                        "pinned_dispatch_count",
                        "moved_dispatch_count",
                        "enqueued_count",
                        "bubble_drain_count",
                        "deadline_flush_count",
                        "max_queue_depth",
                        "fusion_eligible_count",
                        "fusion_fallback_count",
                        "fusion_execution_count",
                        "fusion_replay_count",
                        "fusion_eager_fallback_count",
                        "producer_stage_dispatch_count",
                        "capture_resident_stage_count",
                        "capture_resident_stage_install_count",
                        "capture_resident_stage_dispatch_count",
                        "step_lease_retire_count",
                    )
                },
            }
            if args.mode == "native":
                from tllm.runtime.mp.control import ControlPlane

                (native_stats,) = ControlPlane(llm).pull_stats(
                    "workgraph", sync=False, timeout=60.0
                )
                workgraph = native_stats.workgraph
                tllm_metrics["native_workgraph"] = {
                    "execute_count": workgraph.execute_count,
                    "eager_count": workgraph.eager_count,
                    "replay_count": workgraph.replay_count,
                    "fallback_count": workgraph.fallback_count,
                }
                execution = core.RUNTIME.workload_execution
                if execution is not None:
                    cost_report = execution.region_cost_report(block=True)
                    tllm_metrics["native_region_costs"] = {
                        key: {
                            "region_id": reading.region_id,
                            "phase": reading.phase,
                            "stream": reading.stream,
                            "samples": reading.samples,
                            "ewma_ns": reading.ewma_ns,
                            "last_ns": reading.last_ns,
                        }
                        for key, reading in cost_report.readings.items()
                    }
    finally:
        from tllm.util.tools import shutdown_llm_instance

        try:
            if native_runtime is not None:
                native_runtime.clear_consumers()
            shutdown_llm_instance(llm, cooldown_s=1.0)
        finally:
            if scheduler_trace is not None:
                scheduler_trace.restore()

    summary = _summarize_rounds(round_durations, round_token_counts)
    payload = {
        "mode": args.mode,
        "runner": args.runner,
        "mp": args.mp,
        "model": args.model,
        "rounds": len(round_durations),
        "warmup_rounds": max(1, int(args.warmup_rounds)),
        "workload": _workload_fingerprint(args, prompts),
        **summary,
    }
    if scheduler_trace is not None:
        payload["scheduler_diagnostics"] = _summarize_scheduler_trace(
            scheduler_trace.snapshot(),
            max_num_seqs=ACTIVE_SEQUENCES,
            max_num_scheduled_tokens=MAX_NUM_BATCHED_TOKENS,
        )
    if args.mode != "raw":
        payload.update(tllm_metrics)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(json.dumps(payload))


def ratio(args: argparse.Namespace) -> None:
    with open(args.a, encoding="utf-8") as fh:
        a = json.load(fh)
    with open(args.b, encoding="utf-8") as fh:
        b = json.load(fh)
    _validate_comparable(a, b)
    r = b["out_tok_per_s"] / a["out_tok_per_s"]
    print(f"baseline({a['mode']}/{a['runner']}/mp{a.get('mp', '?')}): {a['out_tok_per_s']:.3f} tok/s")
    print(f"esamp   ({b['mode']}/{b['runner']}/mp{b.get('mp', '?')}): {b['out_tok_per_s']:.3f} tok/s")
    print(
        f"ratio: {r:.4f}  (required: {args.min_ratio:.4f}; "
        f"stretch: {STRETCH_RATIO:.4f})"
    )
    if r < float(args.min_ratio):
        print("FAILED required ratio")
        raise SystemExit(1)


def capture_overhead(args: argparse.Namespace) -> None:
    with open(args.raw, encoding="utf-8") as handle:
        raw = json.load(handle)
    with open(args.capture_only, encoding="utf-8") as handle:
        capture_only = json.load(handle)
    print(json.dumps(_capture_overhead(raw, capture_only), sort_keys=True))


def acceptance(args: argparse.Namespace) -> None:
    def _load_many(paths: Sequence[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in paths:
            with open(path, encoding="utf-8") as handle:
                rows.append(dict(json.load(handle)))
        return rows

    result = _paired_acceptance(_load_many(args.raw), _load_many(args.tllm))
    print(json.dumps(result, sort_keys=True))
    threshold = float(args.min_ratio)
    if any(
        result[key] < threshold
        for key in ("median_ratio", "geomean_ratio", "ci95_lower")
    ):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument(
        "--mode",
        choices=["raw", "capture_only", "tllm", "native"],
        required=True,
    )
    p_run.add_argument("--runner", choices=["mrv1", "mrv2"], default="mrv2")
    p_run.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p_run.add_argument("--gpu-mem", default="0.4")
    p_run.add_argument("--no-intervention", action="store_true")
    p_run.add_argument("--rounds", type=int, default=3)
    p_run.add_argument("--warmup-rounds", type=int, default=3)
    p_run.add_argument("--mp", choices=["0", "1"], default="0", help="VLLM_ENABLE_V1_MULTIPROCESSING")
    p_run.add_argument("--out", required=True)
    p_run.set_defaults(func=run)
    p_ratio = sub.add_parser("ratio")
    p_ratio.add_argument("a")
    p_ratio.add_argument("b")
    p_ratio.add_argument("--min-ratio", type=float, default=REQUIRED_RATIO)
    p_ratio.set_defaults(func=ratio)
    p_capture = sub.add_parser("capture-overhead")
    p_capture.add_argument("raw")
    p_capture.add_argument("capture_only")
    p_capture.set_defaults(func=capture_overhead)
    p_accept = sub.add_parser("acceptance")
    p_accept.add_argument("--raw", nargs="+", required=True)
    p_accept.add_argument("--tllm", nargs="+", required=True)
    p_accept.add_argument("--min-ratio", type=float, default=REQUIRED_RATIO)
    p_accept.set_defaults(func=acceptance)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
