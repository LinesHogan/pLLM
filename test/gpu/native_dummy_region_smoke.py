#!/usr/bin/env python3
"""GPU correctness smoke for the native eager-only Dummy ExecutionRegion.

This deliberately exercises ``DummyWorkload`` through ``TllmRuntime``.  It
does not register or instantiate the legacy ``DummyConsumer``. CPU-export
counters are read through the typed worker status RPC, so the assertion is
about the reconstructed active resource instance rather than the parent's
original workload object.
"""

from __future__ import annotations

import argparse
import os
import sys


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _require_native_eager_region(runtime: object) -> tuple[str, str]:
    """Validate the Rust-issued Region contract before generating tokens."""

    from tllm.schedule import RegionImplementationKind

    compilation = getattr(runtime, "workload_compilation", None)
    if compilation is None:
        raise AssertionError("native Dummy workload did not produce a compilation")
    regions = compilation.decision.execution_regions
    if len(regions) != 1:
        raise AssertionError(f"expected one Dummy ExecutionRegion, got {regions!r}")
    region = regions[0]
    if region.implementation_kind is not RegionImplementationKind.EXTERN:
        raise AssertionError(f"Dummy Region must be extern, got {region!r}")
    if len(region.tasks) != 1 or len(region.implementation_ids) != 1:
        raise AssertionError(f"Dummy extern Region must be singleton, got {region!r}")
    if region.compile_safe or region.capture_safe:
        raise AssertionError(f"Dummy extern Region must remain eager-only, got {region!r}")
    if "eager_only" not in region.required_capabilities:
        raise AssertionError(f"Dummy Region lost eager_only capability: {region!r}")
    if len(region.artifact_key) != 64:
        raise AssertionError(f"Dummy Region has invalid artifact key: {region!r}")
    return region.region_id, region.artifact_key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), required=True)
    parser.add_argument("--gpu-mem", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    args = parser.parse_args()

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "mrv2" else "0"
    # Export statistics are process-local native resources.  Do not let an
    # EngineCore child make this test look green without observable exports.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import SamplingParams

    from tllm.consumers.dummy.config import DummyConsumerConfig
    from tllm.consumers.dummy.workload import DummyWorkload
    from tllm.runtime import residual_runtime as core
    from tllm.runtime.mp.control import ControlPlane
    from tllm.runtime.runtime import TllmRuntime
    from tllm.util.tools import shutdown_llm_instance

    workload = DummyWorkload(
        DummyConsumerConfig(
            enable_async=False,
            export_every_n_steps=1,
            export_max_rows=1,
            export_max_cols=16,
        )
    )
    runtime = TllmRuntime(workloads=(workload,))
    if runtime.consumers:
        raise AssertionError("native Dummy smoke unexpectedly installed a legacy consumer")

    llm = runtime.make_llm(
        model_name=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=float(args.gpu_mem),
        max_model_len=64,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=0,
        max_num_seqs=2,
        max_num_batched_tokens=16,
    )
    try:
        region_id, artifact_key = _require_native_eager_region(runtime)
        execution = core.RUNTIME.workload_execution
        if execution is None:
            raise AssertionError("native Dummy decision was not bound to an executor")
        expected_report = ((region_id, artifact_key),)
        store = core.RUNTIME.workload_execution_store
        installed = (execution,) if store is None else (store.current, store.staged)
        if not any(
            candidate is not None and candidate.artifact_report == expected_report
            for candidate in installed
        ):
            raise AssertionError(
                "native artifact report does not match the signed decision: "
                f"installed={tuple(candidate.artifact_report for candidate in installed if candidate is not None)!r}, "
                f"expected={expected_report!r}"
            )
        if store is not None and store.staged is not None and not store.staged_activated:
            raise AssertionError("coordinated Dummy epoch was staged but not activated")

        outputs = llm.generate(
            [
                {"prompt_token_ids": [10, 11, 12]},
                {"prompt_token_ids": [20, 21, 22, 23]},
            ],
            SamplingParams(
                max_tokens=int(args.max_new_tokens),
                temperature=0.0,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )
        token_counts = [len(output.outputs[0].token_ids) for output in outputs]
        if token_counts != [int(args.max_new_tokens)] * 2:
            raise AssertionError(f"generation did not complete: {token_counts}")

        execution = core.RUNTIME.workload_execution
        if execution is None or execution.artifact_report != expected_report:
            raise AssertionError(
                "first user step did not promote the activated Dummy epoch: "
                f"actual={None if execution is None else execution.artifact_report!r}, "
                f"expected={expected_report!r}"
        )
        execution.synchronize()
        statuses = ControlPlane(llm).workload_status()
        if len(statuses) != 1:
            raise AssertionError(f"expected one Dummy worker status, got {statuses!r}")
        diagnostics = statuses[0].get("diagnostics")
        if not isinstance(diagnostics, dict):
            raise AssertionError(f"Dummy worker diagnostics are invalid: {statuses!r}")
        export_stats = diagnostics.get("dummy")
        if not isinstance(export_stats, dict):
            raise AssertionError(
                "active Dummy worker did not report namespaced resource diagnostics: "
                f"{statuses!r}"
            )
        physical_stats = execution.read_workgraph_runtime_stats()
        if physical_stats.execute_count <= 0 or physical_stats.eager_count <= 0:
            raise AssertionError(f"native eager Region did not execute: {physical_stats}")
        if physical_stats.replay_count != 0 or physical_stats.fallback_count != 0:
            raise AssertionError(
                "eager-only Dummy Region replayed or fell back: "
                f"{physical_stats}"
            )
        if (
            export_stats["consumed_batches"] <= 0
            or export_stats["consumed_rows"] <= 0
            or export_stats["processed_batches"] <= 0
        ):
            raise AssertionError(
                "native Dummy CPU export did not run: "
                f"export={export_stats}, physical={physical_stats}, "
                f"step={core.RUNTIME.event_step_id}, "
                f"decode_count={core.RUNTIME.decode_count}, "
                f"bindings={core.RUNTIME.residual_bindings!r}, "
                f"tap_keys={tuple(core.RUNTIME.tap_decode_hidden)}"
            )

        print(
            "OK: native Dummy ExecutionRegion smoke passed "
            f"(runner={args.runner}, tokens={token_counts}, "
            f"region_id={region_id}, artifact_key={artifact_key}, "
            f"physical_stats={physical_stats}, export_stats={dict(export_stats)})"
        )
        return 0
    finally:
        shutdown_llm_instance(llm, cooldown_s=1.0)
        core.publish_workload_compilation(None)


if __name__ == "__main__":
    sys.exit(main())
