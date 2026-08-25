#!/usr/bin/env python3
"""GPU smoke for request-keyed ModuleWeightView overlays on vLLM 0.27."""

from __future__ import annotations

import argparse
import os
import sys


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SOURCE_PATH = "model.model.layers[0].input_layernorm"
TARGET_PATH = "model.model.layers[-1].input_layernorm"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), required=True)
    parser.add_argument("--gpu-mem", type=float, default=0.4)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "mrv2" else "0"
    # Keep state and wrapper handles inspectable by this verification process.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import tllm
    from tllm.consumers.base import BaseConsumer
    from tllm.contracts.request_identity import resolve_request_identity
    from tllm.ports.base import ConsumerFlow
    from tllm.ports.module_weight_view import ModuleWeightView
    from tllm.runtime import residual_runtime as core
    from tllm.runtime.fast_weight.state import FastWeightKey, FastWeightState
    from tllm.util.tools import shutdown_llm_instance

    class OverlayProbe(BaseConsumer[object]):
        config = object()
        consumer_id = "test.module_weight_overlay_smoke"

        def flows(self):
            return (
                ConsumerFlow(
                    reads=(),
                    writes=(
                        ModuleWeightView.write(
                            layer=-1,
                            module="mlp",
                            phase="prefill",
                            mode="wrapper",
                        ),
                    ),
                    window="same_step",
                ),
            )

    core.configure_runtime(
        graph_scratch_rows=8,
        tap_layer_paths=(SOURCE_PATH, TARGET_PATH),
        source_layer_path=SOURCE_PATH,
        target_layer_path=TARGET_PATH,
        fast_weight_enabled=True,
        fast_weight_layers=(-1,),
        fast_weight_chunk_size=2,
        fast_weight_lr=1e-7,
        fast_weight_mode="exact",
        fast_weight_key_mode="request_sample",
        fast_weight_max_slots=8,
    )
    core.register_dispatch_consumer(OverlayProbe())
    llm = tllm.make_llm(
        model_name=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=float(args.gpu_mem),
        max_model_len=128,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=1234,
        max_num_seqs=2,
    )
    try:
        from vllm import SamplingParams

        from tllm.runtime.vllm_patch.capture_runner import run_generate_with_request_mapping

        params = SamplingParams(
            temperature=0.0,
            max_tokens=int(args.max_new_tokens),
            ignore_eos=True,
        )
        outputs = run_generate_with_request_mapping(
            runtime=core.RUNTIME,
            llm=llm,
            prompts=(
                "Explain why typed interfaces matter in one sentence.",
                "Name two uses for test-time model adaptation.",
            ),
            params=params,
            use_tqdm=False,
        )

        token_counts = [len(output.outputs[0].token_ids) for output in outputs]
        if token_counts != [int(args.max_new_tokens), int(args.max_new_tokens)]:
            print(f"FAIL: generation did not complete: token_counts={token_counts}")
            return 1

        handles = core.RUNTIME.fast_weight_handles
        if len(handles) != 1:
            print(f"FAIL: expected one installed overlay, got {tuple(handles)}")
            return 1
        handle_path, handle = next(iter(handles.items()))
        try:
            layer = int(handle_path.split(".")[2])
        except (IndexError, ValueError):
            print(f"FAIL: cannot resolve installed layer from {handle_path!r}")
            return 1

        state = core.RUNTIME.fast_weight_state
        if not isinstance(state, FastWeightState):
            print("FAIL: fast-weight state was not initialized")
            return 1
        request_stats = {}
        for output in outputs:
            request_id = str(output.request_id)
            matching_keys = [
                key
                for key in state.keys()
                if key.layer == layer
                and (
                    identity := resolve_request_identity(
                        key.request_id,
                        prompt_indices=core.RUNTIME.reqid_to_promptidx,
                        sample_indices=core.RUNTIME.reqid_to_sampleidx,
                    )
                )
                is not None
                and identity.request_id == request_id
                and identity.sample_idx == 0
            ]
            if len(matching_keys) != 1:
                print(
                    f"FAIL: request {request_id!r} maps to {matching_keys}; "
                    f"actual_keys={state.keys()}"
                )
                return 1
            key: FastWeightKey = matching_keys[0]
            slot = state.get(key)
            stats = state.stats(key)
            request_stats[request_id] = stats
            if slot is None:
                print(
                    f"FAIL: request {request_id!r} has no independent state slot; "
                    f"actual_keys={state.keys()}"
                )
                return 1
            if stats.update_count <= 0 or stats.apply_count <= 0:
                print(f"FAIL: request {request_id!r} did not update and apply: {stats}")
                return 1

        module = handle.module
        wrapped_forward = module.forward
        original_forward = handle.original_forward
        core.set_fast_weight_enabled(False)
        if core.RUNTIME.fast_weight_handles or core.RUNTIME.fast_weight_binding is not None:
            print("FAIL: disable left overlay lifecycle state installed")
            return 1
        if module.forward is not original_forward or module.forward is wrapped_forward:
            print("FAIL: disable did not restore the original module forward")
            return 1

        print(
            f"OK: ModuleWeightView overlay smoke passed (runner={args.runner}, "
            f"tokens={token_counts}, layer={layer}, request_stats={request_stats})"
        )
        return 0
    finally:
        core.set_fast_weight_enabled(False)
        shutdown_llm_instance(llm, cooldown_s=1.0)


if __name__ == "__main__":
    sys.exit(main())
