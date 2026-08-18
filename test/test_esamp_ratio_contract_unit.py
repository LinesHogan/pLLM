#!/usr/bin/env python3
"""CPU-only contracts for the aligned ESamp ratio harness."""

from __future__ import annotations

import unittest

from test.gpu.esamp_ratio_benchmark import (
    ACTIVE_SEQUENCES,
    MIN_FULL_DECODE_RUN,
    REQUIRED_RATIO,
    STRETCH_RATIO,
    _capture_overhead,
    _paired_acceptance,
    _summarize_scheduler_trace,
    _summarize_rounds,
    _validate_comparable,
)


def _scheduler_diagnostics() -> dict[str, object]:
    trace = [
        {
            "num_ctx_requests": 128,
            "num_ctx_tokens": 1312,
            "num_generation_requests": 0,
            "num_generation_tokens": 0,
            "num_preemptions": 0,
        },
        *[
            {
                "num_ctx_requests": 0,
                "num_ctx_tokens": 0,
                "num_generation_requests": ACTIVE_SEQUENCES,
                "num_generation_tokens": ACTIVE_SEQUENCES,
                "num_preemptions": 0,
            }
            for _ in range(MIN_FULL_DECODE_RUN)
        ],
    ]
    return _summarize_scheduler_trace(
        trace,
        max_num_seqs=ACTIVE_SEQUENCES,
        max_num_scheduled_tokens=4096,
    )


def _run_payload(*, mode: str, throughput: float, loss_count: int = 1) -> dict[str, object]:
    return {
        "mode": mode,
        "workload": {"schema_version": 3, "model": "m", "batch_size": 8},
        "out_tok_per_s": float(throughput),
        "scheduler_diagnostics": _scheduler_diagnostics(),
        "training_enabled": mode == "tllm",
        "intervention_enabled": mode == "tllm",
        "esamp_loss_count": int(loss_count),
        "expected_esamp_loss_count": 1,
        "minimum_esamp_loss_count": 1,
        "candidate_sample_count": 1,
        "candidate_row_count": ACTIVE_SEQUENCES,
        "candidate_token_count": ACTIVE_SEQUENCES,
        "candidate_kernel_triton_count": 1,
        "candidate_kernel_fallback_count": 0,
        "distiller_timing": {
            "port_publish_hit_count": 1,
        },
        "placement": {
            "moved_dispatch_count": 1,
            "bubble_drain_count": 1,
            "fusion_execution_count": 1,
            "fusion_replay_count": 1,
            "capture_resident_stage_count": 1,
            "capture_resident_stage_install_count": 1,
            "step_lease_retire_count": 1,
        },
    }


class ESampRatioContractUnitTest(unittest.TestCase):
    def test_user_required_ratio_target_is_explicit(self) -> None:
        self.assertEqual(REQUIRED_RATIO, 0.98)
        self.assertEqual(STRETCH_RATIO, 0.995)

    def test_capture_only_diagnostic_reports_fixed_step_tax(self) -> None:
        raw = _run_payload(mode="raw", throughput=20_000.0)
        capture_only = _run_payload(
            mode="capture_only",
            throughput=19_800.0,
            loss_count=0,
        )
        capture_only["training_enabled"] = False
        capture_only["intervention_enabled"] = False
        capture_only["candidate_sample_count"] = 0
        capture_only["candidate_row_count"] = 0
        capture_only["candidate_token_count"] = 0
        capture_only["candidate_kernel_triton_count"] = 0
        capture_only["placement"] = {
            "capture_resident_stage_count": 1,
            "capture_resident_stage_install_count": 2,
        }

        result = _capture_overhead(raw, capture_only)

        self.assertAlmostEqual(result["capture_ratio"], 0.99)
        self.assertAlmostEqual(
            result["capture_fixed_tax_us_per_step"],
            (ACTIVE_SEQUENCES / 19_800.0 - ACTIVE_SEQUENCES / 20_000.0) * 1e6,
        )
        with self.assertRaisesRegex(ValueError, "ordered as raw then tllm"):
            _validate_comparable(raw, capture_only)

    def test_scheduler_trace_requires_a_full_128_row_decode_plateau(self) -> None:
        diagnostics = _scheduler_diagnostics()

        self.assertEqual(diagnostics["max_generation_requests"], ACTIVE_SEQUENCES)
        self.assertEqual(diagnostics["longest_full_decode_run"], MIN_FULL_DECODE_RUN)
        self.assertEqual(diagnostics["total_preemptions"], 0)
        self.assertEqual(diagnostics["token_request_mismatch_count"], 0)

    def test_round_summary_uses_all_elapsed_time_and_tokens(self) -> None:
        summary = _summarize_rounds([2.0, 4.0], [100, 100])

        self.assertEqual(summary["elapsed_s"], 6.0)
        self.assertEqual(summary["output_tokens"], 200)
        self.assertAlmostEqual(float(summary["out_tok_per_s"]), 200.0 / 6.0)
        self.assertEqual(summary["round_durations_s"], [2.0, 4.0])

    def test_comparison_rejects_workload_drift_and_missing_training(self) -> None:
        raw = _run_payload(mode="raw", throughput=100.0)
        no_training = _run_payload(mode="tllm", throughput=100.0, loss_count=0)
        with self.assertRaisesRegex(ValueError, "loss_count"):
            _validate_comparable(raw, no_training)

        sparse_training = _run_payload(mode="tllm", throughput=100.0, loss_count=1)
        sparse_training["expected_esamp_loss_count"] = 100
        with self.assertRaisesRegex(ValueError, "training density"):
            _validate_comparable(raw, sparse_training)

        drifted = _run_payload(mode="tllm", throughput=100.0)
        drifted["workload"] = {"schema_version": 1, "model": "other"}
        with self.assertRaisesRegex(ValueError, "fingerprints differ"):
            _validate_comparable(raw, drifted)

        no_capture = _run_payload(mode="tllm", throughput=100.0)
        no_capture["placement"] = {}
        with self.assertRaisesRegex(ValueError, "compiled DAG/fusion placement"):
            _validate_comparable(raw, no_capture)

        no_candidates = _run_payload(mode="tllm", throughput=100.0)
        no_candidates["candidate_sample_count"] = 0
        with self.assertRaisesRegex(ValueError, "sampler intervention"):
            _validate_comparable(raw, no_candidates)

        sparse_candidates = _run_payload(mode="tllm", throughput=100.0)
        sparse_candidates["candidate_row_count"] = 1
        with self.assertRaisesRegex(ValueError, "candidate row density"):
            _validate_comparable(raw, sparse_candidates)

        schedule_drift = _run_payload(mode="tllm", throughput=100.0)
        diagnostics = dict(schedule_drift["scheduler_diagnostics"])  # type: ignore[arg-type]
        diagnostics["iteration_shape_histogram"] = {"drift": 1}
        schedule_drift["scheduler_diagnostics"] = diagnostics
        with self.assertRaisesRegex(ValueError, "scheduler iteration shapes differ"):
            _validate_comparable(raw, schedule_drift)

        not_full = _run_payload(mode="tllm", throughput=100.0)
        diagnostics = dict(not_full["scheduler_diagnostics"])  # type: ignore[arg-type]
        diagnostics["longest_full_decode_run"] = MIN_FULL_DECODE_RUN - 1
        not_full["scheduler_diagnostics"] = diagnostics
        with self.assertRaisesRegex(ValueError, "batch-full decode plateau"):
            _validate_comparable(raw, not_full)

    def test_mp1_comparison_does_not_require_process_local_scheduler_trace(self) -> None:
        raw = _run_payload(mode="raw", throughput=100.0)
        tllm = _run_payload(mode="tllm", throughput=99.0)
        for payload in (raw, tllm):
            payload["workload"] = {
                "schema_version": 4,
                "model": "m",
                "batch_size": 8,
                "mp": "1",
            }
            payload.pop("scheduler_diagnostics")

        _validate_comparable(raw, tllm)

    def test_paired_acceptance_reports_ci_from_five_blocks(self) -> None:
        raw = [_run_payload(mode="raw", throughput=100.0) for _ in range(5)]
        tllm = [
            _run_payload(mode="tllm", throughput=value)
            for value in (100.0, 100.1, 99.9, 100.0, 100.05)
        ]

        result = _paired_acceptance(raw, tllm)

        self.assertGreater(result["median_ratio"], 0.995)
        self.assertGreater(result["geomean_ratio"], 0.995)
        self.assertGreater(result["ci95_lower"], 0.995)


if __name__ == "__main__":
    unittest.main()
