#!/usr/bin/env python3
"""Native In-Place-TTT update -> dense row overlay -> vLLM eager smoke."""

from __future__ import annotations

import argparse
import json
import os

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
INTERMEDIATE_SIZE = 4864


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), default="mrv2")
    parser.add_argument("--gpu-mem", type=float, default=0.35)
    parser.add_argument("--engine-core-mp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _args()
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = (
        "1" if args.runner == "mrv2" else "0"
    )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = (
        "1" if args.engine_core_mp else "0"
    )

    import torch
    from vllm import SamplingParams

    from tllm.consumers.inplace_ttt import (
        InPlaceTTTConfig,
        InPlaceTTTKey,
        InPlaceTTTMetadata,
        InPlaceTTTPhase,
        InPlaceTTTWorkload,
    )
    from tllm.contracts.request_identity import RequestIdentity
    from tllm.runtime import residual_runtime as core
    from tllm.runtime.mp.control import ControlPlane
    from tllm.runtime.runtime import TllmRuntime
    from tllm.runtime.vllm_patch.capture_runner import (
        run_generate_with_request_mapping,
    )
    from tllm.util.tools import shutdown_llm_instance

    workload = InPlaceTTTWorkload(
        InPlaceTTTConfig(
            layers=(0,),
            chunk_size=1,
            learning_rate=1e-3,
            update_prefill=True,
            update_decode=False,
        )
    )
    runtime = TllmRuntime(
        workloads=(workload,), workload_memory_budget_bytes=512 << 20
    )
    llm = runtime.make_llm(
        model_name=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=64,
        max_num_seqs=1,
        max_num_batched_tokens=16,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=1234,
    )
    try:
        compilation = runtime.workload_compilation
        if compilation is None:
            raise AssertionError("native In-Place-TTT workload was not compiled")
        family = compilation.timeline.runner_family
        if family != args.runner:
            raise AssertionError(f"requested {args.runner}, got {family}")
        spec = workload.specialization
        base_weight = torch.zeros(
            (spec.hidden_size, INTERMEDIATE_SIZE),
            device=spec.device,
            dtype=spec.dtype,
        )
        request = RequestIdentity("0", 0, 0)
        runtime.stage_pair(
            InPlaceTTTWorkload.batch_tensors,
            {
                "hidden": torch.ones(
                    (1, base_weight.shape[1]),
                    device=base_weight.device,
                    dtype=base_weight.dtype,
                ),
                "target": torch.ones(
                    (1, base_weight.shape[0]),
                    device=base_weight.device,
                    dtype=base_weight.dtype,
                ),
                "base_weight": base_weight,
            },
            InPlaceTTTWorkload.batch_metadata,
            InPlaceTTTMetadata(
                InPlaceTTTKey(request, 0), InPlaceTTTPhase.PREFILL
            ),
        )
        params = SamplingParams(
            temperature=0.0, max_tokens=3, ignore_eos=True
        )
        prompts = ({"prompt_token_ids": [1, 2, 3]},)
        first = run_generate_with_request_mapping(
            runtime=core.RUNTIME,
            llm=llm,
            prompts=prompts,
            params=params,
            request_prompt_indices=[0],
            request_sample_indices=[0],
            use_tqdm=False,
        )
        if str(first[0].request_id) != request.request_id:
            raise AssertionError(
                f"expected request id {request.request_id}, got {first[0].request_id}"
            )
        publication = runtime.drain(InPlaceTTTWorkload.publications)[0]
        if publication.version is None or publication.update_count <= 0:
            raise AssertionError(
                f"native In-Place-TTT did not publish: {publication!r}"
            )
        target = publication.version.lineage.channel.target
        if target is None:
            raise AssertionError("dense publication has no typed target")
        (status,) = ControlPlane(llm).workload_status()
        metrics = status["diagnostics"]["inplace_ttt"]
        applied_rows = metrics["dense_apply_rows"]
        if applied_rows <= 0:
            raise AssertionError(
                "published dense version was not applied to any vLLM row"
            )
        cleanup = runtime.drain(InPlaceTTTWorkload.cleanup)
        if (
            len(cleanup) != 1
            or cleanup[0].cleared_slots != 1
            or cleanup[0].released_versions != 1
        ):
            raise AssertionError(
                "signed request cleanup did not run at generate end: "
                f"cleanup={cleanup!r} metrics={metrics!r}"
            )
        if metrics["update_count"] <= 0 or metrics["trained_rows"] <= 0:
            raise AssertionError(
                "native In-Place-TTT update counters did not advance"
            )
        if metrics["fast_weight_slots"] or metrics["overlay_bytes"]:
            raise AssertionError(
                f"finished request retained workload state: {metrics!r}"
            )

        second = run_generate_with_request_mapping(
            runtime=core.RUNTIME,
            llm=llm,
            prompts=prompts,
            params=params,
            request_prompt_indices=[1],
            request_sample_indices=[0],
            use_tqdm=False,
        )
        if (
            first[0].outputs[0].token_ids
            == second[0].outputs[0].token_ids
        ):
            raise AssertionError(
                "dense overlay request matched the base-weight token sequence"
            )
        print(
            json.dumps(
                {
                    "runner": family,
                    "engine_core_mp": args.engine_core_mp,
                    "version": publication.version.version,
                    "update_count": metrics["update_count"],
                    "trained_rows": metrics["trained_rows"],
                    "dense_apply_rows": applied_rows,
                    "released_versions": cleanup[0].released_versions,
                    "tokens": first[0].outputs[0].token_ids,
                    "base_tokens": second[0].outputs[0].token_ids,
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        runtime.clear_consumers()
        shutdown_llm_instance(llm, cooldown_s=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
