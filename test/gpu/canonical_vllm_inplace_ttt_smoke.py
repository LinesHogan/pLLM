"""One real MRv2 boundary proof for canonical dense-fast-weight overlays."""

from __future__ import annotations

from vllm import SamplingParams

from tllm.canonical.workloads.inplace_ttt import (
    InPlaceTTTConfig,
    InPlaceTTTWorkload,
)
from tllm.canonical_runtime.parameter_overlay_llm import (
    create_canonical_parameter_overlay_llm,
)
from tllm.canonical_runtime.runtime import CanonicalVllmConfig, CanonicalVllmDType
from tllm.contracts.parameter_overlay import (
    ParameterModuleKind,
    ParameterRole,
    ParameterTarget,
)


def main() -> int:
    workload = InPlaceTTTWorkload(
        InPlaceTTTConfig(
            (
                ParameterTarget(
                    0,
                    ParameterModuleKind.MLP,
                    ParameterRole.DOWN_PROJECTION_WEIGHT,
                ),
            ),
            chunk_size=1,
            learning_rate=1e-4,
            update_prefill=True,
            update_decode=False,
        )
    )
    handle = create_canonical_parameter_overlay_llm(
        workload,
        CanonicalVllmConfig(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            dtype=CanonicalVllmDType.BFLOAT16,
            max_model_len=64,
            max_num_seqs=2,
            max_num_batched_tokens=64,
            gpu_memory_utilization=0.25,
            seed=7,
        ),
    )
    try:
        outputs = handle.generate(
            ["The sky is", "Water freezes at"],
            SamplingParams(
                max_tokens=2,
                temperature=0.0,
                n=1,
                ignore_eos=True,
            ),
        )
        diagnostics = handle.deployment.execution.diagnostics()
        if (
            any(not output.outputs[0].text for output in outputs)
            or diagnostics.update_count < 1
            or diagnostics.trained_rows < 1
            or handle.runner.overlay_apply_count < 2
            or handle.runner.overlay_updated_apply_count < 1
            or diagnostics.active_lineages != 0
            or diagnostics.pending_state_releases != 0
            or diagnostics.fallback_count != 0
        ):
            raise RuntimeError(
                "canonical MRv2 parameter-overlay lifecycle evidence is incomplete"
            )
        print(
            {
                "text": tuple(output.outputs[0].text for output in outputs),
                "update_count": diagnostics.update_count,
                "trained_rows": diagnostics.trained_rows,
                "dense_apply_rows": handle.runner.overlay_apply_count,
                "updated_dense_apply_rows": (
                    handle.runner.overlay_updated_apply_count
                ),
                "active_lineages": diagnostics.active_lineages,
                "fallback_count": diagnostics.fallback_count,
                "attachment": handle.deployment.attachment.physical.bundle_signature[
                    :12
                ],
            }
        )
    finally:
        handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
