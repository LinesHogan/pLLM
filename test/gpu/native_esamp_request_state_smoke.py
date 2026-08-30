#!/usr/bin/env python3
"""GPU smoke for eager native ESamp request-scoped state.

One invocation covers one of the four required topology cells:

``MRv1/MRv2 x in-process/EngineCore MP``.

The workload is intentionally small but not synthetic: two public parent
requests use ``SamplingParams(n=2)`` so vLLM creates parallel child samples.
Each child emits three tokens: the initial sampled token can come from
prefill, leaving exactly two decode updates. After the public call returns,
the closed worker-status DTO proves both live slots and request-id mappings
were retired while retaining a bounded immutable retirement receipt.

This is an eager correctness smoke.  It deliberately rejects any CUDA-graph
eligibility for the request-state Regions rather than accepting a graph
fallback as evidence of success.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.outputs import RequestOutput

    from tllm.runtime.mp.control import (
        WorkerPingStatus,
        WorkerRequestStateRetirementStatus,
        WorkerWorkloadStatus,
    )
    from tllm.schedule import WorkloadCompilation


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_PARENTS = 2
_PARALLEL_SAMPLES = 2
_GENERATED_TOKENS = 3
_DECODE_UPDATES = 2
_REQUEST_STATE_RESOURCE = "request_distiller"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), required=True)
    parser.add_argument(
        "--engine-core-mp",
        action="store_true",
        help="run through the EngineCore child rather than the in-process worker",
    )
    parser.add_argument("--model-name", default=MODEL)
    parser.add_argument("--gpu-mem", type=float, default=0.35)
    return parser.parse_args()


def _signed_eager_contract(
    compilation: "WorkloadCompilation",
) -> tuple[tuple[str, str], ...]:
    """Assert the Rust-signed eager contract, not merely its Python intent."""

    from tllm.schedule import (
        RegionCaptureMode,
        RegionHostSubmission,
        RegionStreamClass,
    )
    from tllm.workgraph.physical import CapturePolicy

    if compilation.timeline.graph_mode != "eager":
        raise AssertionError(
            f"request-state smoke requires eager timeline: {compilation.timeline!r}"
        )
    physical = compilation.physical
    if physical is None:
        raise AssertionError("native request-state compilation has no physical plan")
    regions = compilation.decision.execution_regions
    tasks = tuple(task for region in regions for task in region.tasks)
    expected = ("request_predict", "request_loss", "request_train")
    if set(tasks) != set(expected) or len(tasks) != len(expected):
        raise AssertionError(f"unexpected signed request-state tasks: {tasks!r}")
    for region in regions:
        if region.stream is not RegionStreamClass.DEFAULT:
            raise AssertionError(f"request-state Region is not default-stream: {region!r}")
        if region.host_submission is not RegionHostSubmission.INLINE:
            raise AssertionError(
                f"request-state Region is not inline eager work: {region!r}"
            )
        if region.compile_safe or region.capture_safe:
            raise AssertionError(
                "request-state Region unexpectedly claims compile/capture safety: "
                f"{region!r}"
            )
        if region.capture_mode is not RegionCaptureMode.DISABLED:
            raise AssertionError(
                "request-state CUDA Graph non-eligibility was not explicit: "
                f"{region!r}"
            )
    if any(node.capture_policy is not CapturePolicy.DISABLED for node in physical.nodes):
        raise AssertionError(
            "physical request-state plan contains a capture-eligible node: "
            f"{physical.nodes!r}"
        )
    return tuple((region.region_id, region.artifact_key) for region in regions)


def _assert_outputs(outputs: Sequence["RequestOutput"]) -> None:
    if len(outputs) != _PARENTS:
        raise AssertionError(f"expected {_PARENTS} parent outputs, got {outputs!r}")
    for output in outputs:
        children = tuple(output.outputs)
        if len(children) != _PARALLEL_SAMPLES:
            raise AssertionError(
                "SamplingParams(n=2) did not expose exactly two child samples: "
                f"{output!r}"
            )
        lengths = tuple(len(child.token_ids) for child in children)
        if lengths != (_GENERATED_TOKENS,) * _PARALLEL_SAMPLES:
            raise AssertionError(
                "parallel samples did not emit the required token count for two "
                "decode updates: "
                f"lengths={lengths!r} output={output!r}"
            )


def _status_artifacts(
    status: "WorkerWorkloadStatus",
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (entry["region_id"], entry["artifact_key"])
        for entry in status["artifacts"]
    )


def _status_diagnostics(status: "WorkerWorkloadStatus") -> dict[str, int | float]:
    try:
        return status["diagnostics"]["esamp"]
    except KeyError as error:
        raise AssertionError(f"worker ESamp diagnostics are absent: {status!r}") from error


def _typed_retirement_audit(
    status: "WorkerWorkloadStatus",
) -> tuple["WorkerRequestStateRetirementStatus", ...]:
    """Normalize post-cleanup typed receipts without worker-local inspection."""

    from tllm.contracts.request_identity import RequestKeyField

    entries = tuple(status["request_state_retirements"])
    for entry in entries:
        key = entry.key
        slot = entry.slot
        if (
            entry.resource != _REQUEST_STATE_RESOURCE
            or key.fields != (RequestKeyField.REQUEST_ID,)
            or not key.request_id
            or key.prompt_idx != -1
            or key.sample_idx != -1
            or slot.slot_id < 0
            or slot.generation < 1
            or entry.before_version < 0
            or entry.after_version < 0
            or entry.before_version != entry.after_version
        ):
            raise AssertionError(
                "worker request-state retirement violates the native parent-key "
                f"contract: {entry!r}"
            )
        if entry.after_version != _DECODE_UPDATES:
            raise AssertionError(
                "each parent slot must retire at exactly its final decode version: "
                f"{entry!r}"
            )
    return tuple(
        sorted(
            entries,
            key=lambda entry: (
                entry.key.request_id,
                entry.slot.slot_id,
            ),
        )
    )


def _assert_post_finish_statuses(
    statuses: Sequence["WorkerWorkloadStatus"],
    *,
    signature: str,
    plan_epoch: int,
    artifacts: tuple[tuple[str, str], ...],
) -> tuple["WorkerRequestStateRetirementStatus", ...]:
    if not statuses:
        raise AssertionError(f"post-cleanup workload status is empty: {statuses!r}")
    canonical: tuple["WorkerRequestStateRetirementStatus", ...] | None = None
    expected_loss_updates = _PARENTS * _DECODE_UPDATES
    for status in statuses:
        if status["signature"] != signature or status["plan_epoch"] != plan_epoch:
            raise AssertionError(f"worker changed signed plan identity: {status!r}")
        if _status_artifacts(status) != artifacts:
            raise AssertionError(f"worker artifacts changed after cleanup: {status!r}")
        if status["request_state_versions"]:
            raise AssertionError(
                "finished native request state remained resident after cleanup: "
                f"{status['request_state_versions']!r}"
            )
        diagnostics = _status_diagnostics(status)
        for name in ("loss_count", "update_count"):
            if diagnostics[name] != expected_loss_updates:
                raise AssertionError(
                    "native request ESamp did not transition once per parent per "
                    f"decode step: diagnostics={diagnostics!r}"
                )
        retirements = _typed_retirement_audit(status)
        if len(retirements) != _PARENTS:
            raise AssertionError(
                "parallel children did not retire exactly two parent slots: "
                f"retirements={retirements!r}"
            )
        if len({item.key.request_id for item in retirements}) != _PARENTS:
            raise AssertionError(
                f"retired parent request-state keys are not distinct: {retirements!r}"
            )
        if len({item.slot.slot_id for item in retirements}) != _PARENTS:
            raise AssertionError(
                f"retired parent request-state slots are not distinct: {retirements!r}"
            )
        if canonical is None:
            canonical = retirements
        elif canonical != retirements:
            raise AssertionError(
                "worker request-state retirement evidence diverged: "
                f"expected={canonical!r} actual={retirements!r}"
            )
    assert canonical is not None
    return canonical


def _mapping_sizes(replies: Sequence["WorkerPingStatus"]) -> tuple[int, ...]:
    return tuple(reply["mapping_size"] for reply in replies)


def _retirement_json(
    retirements: Sequence["WorkerRequestStateRetirementStatus"],
) -> list[dict[str, str | int]]:
    return [
        {
            "resource": entry.resource,
            "request_id": entry.key.request_id,
            "slot_id": entry.slot.slot_id,
            "generation": entry.slot.generation,
            "before_version": entry.before_version,
            "after_version": entry.after_version,
        }
        for entry in retirements
    ]


def main() -> int:
    args = _parse_args()
    if not 0.0 < float(args.gpu_mem) <= 1.0:
        raise ValueError("--gpu-mem must be in (0, 1]")

    # These topology choices are read by vLLM while it constructs the engine.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "mrv2" else "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1" if args.engine_core_mp else "0"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from tllm.consumers.esamp.workload import ESampConfig, ESampWorkload
    from tllm.runtime import residual_runtime as core
    from tllm.runtime.mp.control import ControlPlane
    from tllm.runtime.runtime import TllmRuntime
    from tllm.util.tools import shutdown_llm_instance

    workload = ESampWorkload(
        ESampConfig(
            enabled=True,
            per_request_models=True,
            intervention=False,
            distiller_hidden_dim=8,
            distiller_lr=1e-3,
        )
    )
    runtime = TllmRuntime(
        workloads=(workload,), workload_memory_budget_bytes=64 << 20
    )
    if runtime.consumers:
        raise AssertionError("native request ESamp installed a legacy/program adapter")

    llm: "LLM | None" = None
    try:
        llm = runtime.make_llm(
            model_name=str(args.model_name),
            dtype="bfloat16",
            gpu_memory_utilization=float(args.gpu_mem),
            max_model_len=64,
            enable_prefix_caching=False,
            enforce_eager=True,
            seed=1234,
            max_num_seqs=_PARENTS * _PARALLEL_SAMPLES,
            max_num_batched_tokens=16,
        )
        plane = ControlPlane.try_bind(llm)
        if plane is None:
            raise AssertionError("native request ESamp smoke has no worker control plane")
        params = SamplingParams(
            n=_PARALLEL_SAMPLES,
            temperature=0.8,
            top_p=0.95,
            top_k=32,
            seed=7,
            max_tokens=_GENERATED_TOKENS,
            ignore_eos=True,
        )
        prompts = tuple(
            TokensPrompt(prompt_token_ids=[10 + index, 20 + index, 30 + index])
            for index in range(_PARENTS)
        )
        # Do not expand prompts manually: n=2 must exercise vLLM's parent /
        # child grammar and its EngineCore suffix aliases where applicable.
        outputs = runtime.generate_requests(
            llm,
            prompts,
            params,
            prompt_indices=tuple(range(_PARENTS)),
        )
        _assert_outputs(outputs)
        # In-process vLLM cold-compiles native Regions at the first forward;
        # EngineCore MP has already coordinated its epoch during make_llm.
        compilation = runtime.workload_compilation
        if compilation is None:
            raise AssertionError("native request ESamp plan was not compiled")
        artifacts = _signed_eager_contract(compilation)
        signature = compilation.decision.signature
        physical = compilation.physical
        assert physical is not None
        plan_epoch = physical.plan_epoch
        if runtime.consumers:
            raise AssertionError("native request ESamp silently activated a legacy consumer")
        # The generic cleanup service has retired live slots by this point;
        # the closed worker status exposes their immutable final receipt.
        post_finish_statuses = plane.workload_status(timeout=60.0)
        retirement_audit = _assert_post_finish_statuses(
            post_finish_statuses,
            signature=signature,
            plan_epoch=plan_epoch,
            artifacts=artifacts,
        )
        training = core.read_and_reset_esamp_stats(sync=True)
        if training.loss_count != _PARENTS * _DECODE_UPDATES:
            raise AssertionError(
                "parent did not report all request-state losses: "
                f"{training!r}"
            )
        physical_stats = core.read_workgraph_runtime_stats()
        if (
            physical_stats.execute_count <= 0
            or physical_stats.eager_count <= 0
            or physical_stats.fallback_count != 0
            or physical_stats.replay_count != 0
        ):
            raise AssertionError(
                "request-state eager execution used a fallback or graph replay: "
                f"{physical_stats!r}"
            )
        post_mapping_sizes = _mapping_sizes(plane.ping(timeout=60.0))
        if any(post_mapping_sizes):
            raise AssertionError(
                "finished native request mappings remained live: "
                f"{post_mapping_sizes!r}"
            )
        print(
            json.dumps(
                {
                    "runner": args.runner,
                    "engine_core_mp": bool(args.engine_core_mp),
                    "graph_mode": compilation.timeline.graph_mode,
                    "signature": signature,
                    "plan_epoch": plan_epoch,
                    "signed_tasks": [
                        task
                        for region in compilation.decision.execution_regions
                        for task in region.tasks
                    ],
                    "parents": _PARENTS,
                    "parallel_samples": _PARALLEL_SAMPLES,
                    "generated_tokens": _GENERATED_TOKENS,
                    "decode_updates": _DECODE_UPDATES,
                    "loss_count": training.loss_count,
                    "update_count": _PARENTS * _DECODE_UPDATES,
                    "request_state_retirements": _retirement_json(retirement_audit),
                    "execute_count": physical_stats.execute_count,
                    "eager_count": physical_stats.eager_count,
                    "fallback_count": physical_stats.fallback_count,
                    "replay_count": physical_stats.replay_count,
                    "post_finish_mapping_sizes": post_mapping_sizes,
                    "post_finish_request_state_versions": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        runtime.clear_consumers()
        if llm is not None:
            shutdown_llm_instance(llm, cooldown_s=1.0)
        core.clear_dispatch_consumers()


if __name__ == "__main__":
    sys.exit(main())
