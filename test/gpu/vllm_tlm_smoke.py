"""TLM on a real vLLM model: one training batch publishes q/v LoRA factors used by every later token.

The fused QKV projection of every layer is checked against the layer's own kernel (dense or quantized)
plus dense PyTorch linear arithmetic with the published factors. --model takes a Hub id or a local directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from huggingface_hub import snapshot_download
from torch.nn import functional as F
from vllm import SamplingParams

import pllm
from pllm.authoring import CausalTrainBatch
from pllm.config import ModelConfig
from pllm.contracts.packed_rows import PackedRows
from pllm.workloads.tlm import TLMConfig, TLMMode, TLMWorkload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--mode", choices=tuple(TLMMode), default=TLMMode.ONLINE)
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    model_id = (
        args.model
        if Path(args.model).is_dir()
        else snapshot_download(
            args.model,
            local_files_only=True,
            allow_patterns=("*.json", "*.safetensors", "*.txt", "*.model"),
        )
    )
    workload = TLMWorkload(
        TLMConfig(TLMMode(args.mode), pllm.AdamW(0.01, (0.9, 0.999), 1e-8, 0.01), threshold=0.0), rank=2, alpha=4
    )
    handle = pllm.make_llm(workload=workload, config=ModelConfig(model_id, "bfloat16", 64, 2, 64, 0.5, seed=7))
    execution = handle.runner.require_session().deployment.execution
    tokens = torch.tensor([[1, 2, 3, 4]], device=device)
    batch = CausalTrainBatch(tokens, tokens)
    hooks = []
    checks = [0]

    try:
        (initial,) = execution.directory.select(PackedRows.from_rows(())).refs
        outputs = handle.generate(
            ["The sky is", "Water freezes at"],
            SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True),
            batch=batch,
        )
        assert all(output.outputs[0].token_ids for output in outputs)
        diagnostics = execution.diagnostics()
        assert diagnostics.update_count == diagnostics.loss_count == 1
        assert diagnostics.active_states == 1 and diagnostics.inference_pins == 0
        assert diagnostics.pending_state_releases == 0
        assert (handle.runner.require_session().updated_apply_count == 0) == (args.mode == TLMMode.ONLINE)
        (current,) = execution.directory.select(PackedRows.from_rows(())).refs
        assert current.version_id == initial.version_id + 1
        fields = execution.snapshot(current, torch.cuda.current_stream(device))
        factors = tuple(
            tuple(view.read(fields[view.field]) for view in binding.fields)
            for binding in handle.attachment.spec.overlay_bindings
        )
        assert any(torch.count_nonzero(b) for a, b in factors)

        # Real vLLM fused QKV, independently evaluated with dense PyTorch linear
        # arithmetic. Generation below has no training input, so this exact root
        # must remain selected throughout every request and every token.
        for layer_index, layer in enumerate(handle.runner.model.model.layers):
            q_size, kv_size = layer.self_attn.q_size, layer.self_attn.kv_size
            factor = tuple(
                value.to(layer.self_attn.qkv_proj.weight.dtype)
                for value in (*factors[2 * layer_index], *factors[2 * layer_index + 1])
            )

            def check(module, args, output, factor=factor, q_size=q_size, kv_size=kv_size):
                (hidden,) = args
                actual, _ = output
                expected = module.quant_method.apply(module, hidden, module.bias)
                expected[:, :q_size] += F.linear(F.linear(hidden, factor[0]), factor[1]) * (
                    workload.alpha / workload.rank
                )
                expected[:, q_size + kv_size :] += F.linear(F.linear(hidden, factor[2]), factor[3]) * (
                    workload.alpha / workload.rank
                )
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
                checks[0] += 1

            hooks.append(layer.self_attn.qkv_proj.register_forward_hook(check))
        handle.generate(
            ["The sky is", "Water freezes at"], SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True)
        )
        diagnostics = execution.diagnostics()
        assert checks[0] >= 2 * len(handle.runner.model.model.layers)
        assert tuple(execution.directory.select(PackedRows.from_rows(())).refs) == (current,)
        assert diagnostics.update_count == 1 and diagnostics.trained_rows > 0
        assert handle.runner.require_session().updated_apply_count > 0
        assert diagnostics.inference_pins == diagnostics.pending_state_releases == 0
        print(f"TLM {args.model} {args.mode}: updates={diagnostics.update_count} qkv_oracle_checks={checks[0]}")
    finally:
        for hook in hooks:
            hook.remove()
        handle.close()
    assert execution.diagnostics().active_states == execution.diagnostics().inference_pins == 0


if __name__ == "__main__":
    main()
