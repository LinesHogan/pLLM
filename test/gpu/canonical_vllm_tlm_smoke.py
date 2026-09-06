"""Evidence 33: actual fused q/v arithmetic, global publication and admission pins."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.nn import functional as F
from vllm import SamplingParams

import tllm
from tllm.canonical.backend_program import CausalTrainBatch
from tllm.canonical.kernels.adamw import AdamWConfig
from tllm.canonical.workloads.tlm import HfQvConfig, HfQvModel, TLMConfig, TLMMode, TLMWorkload
from tllm.canonical_vllm.runtime import CanonicalVllmConfig, CanonicalVllmDType
from tllm.contracts.request_state import IndexedRequestIdentityBatch
from tllm.contracts.tensor_layout import TensorDType


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=tuple(TLMMode), default=TLMMode.ONLINE)
    mode = TLMMode(parser.parse_args().mode)
    device = torch.device("cuda", 0)
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model = HfQvModel.load(HfQvConfig(model_id, "main", True, False, 2, 4, 0.0,
        TensorDType.BFLOAT16), device)
    workload = TLMWorkload(TLMConfig(mode, AdamWConfig(0.01, (0.9, 0.999), 1e-8, 0.01),
        threshold=0.0), model)
    handle = tllm.make_llm(workloads=(workload,), canonical_config=CanonicalVllmConfig(model_id,
        CanonicalVllmDType.BFLOAT16, 64, 2, 64, 0.25, seed=7))
    execution = handle.deployment.execution
    tokens = torch.tensor([[1, 2, 3, 4]], device=device)
    batch = CausalTrainBatch(tokens, tokens)
    hooks = []
    checks = [0]
    try:
        initial, = execution.directory.select(IndexedRequestIdentityBatch(())).refs
        outputs = handle.generate(["The sky is", "Water freezes at"],
            SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True), training=batch)
        assert all(output.outputs[0].token_ids for output in outputs)
        diagnostics = execution.diagnostics()
        assert diagnostics.update_count == diagnostics.loss_count == 1
        assert diagnostics.active_lineages == 1 and diagnostics.inference_pins == 0
        assert diagnostics.pending_state_releases == diagnostics.fallback_count == 0
        assert (handle.runner.overlay_updated_apply_count == 0) == (mode is TLMMode.ONLINE)
        current, = execution.directory.select(IndexedRequestIdentityBatch(())).refs
        assert current.version_id == initial.version_id + 1
        fields = execution.snapshot(current, torch.cuda.current_stream(device))
        assert any(torch.count_nonzero(fields[index * 10 + 5]) for index in range(len(model.targets)))

        # Real vLLM fused QKV, independently evaluated with dense PyTorch linear
        # arithmetic. Generation below has no training input, so this exact root
        # must remain selected throughout every request and every token.
        for layer_index, layer in enumerate(handle.runner.model.model.layers):
            q_size, kv_size = layer.self_attn.q_size, layer.self_attn.kv_size
            factor = tuple(fields[20 * layer_index + offset] for offset in (0, 5, 10, 15))
            def check(module, args, output, factor=factor, q_size=q_size, kv_size=kv_size):
                hidden, = args
                actual, _ = output
                expected = F.linear(hidden, module.weight, module.bias)
                expected[:, :q_size] += F.linear(F.linear(hidden, factor[0]), factor[1]) * model.scale
                expected[:, q_size + kv_size:] += F.linear(F.linear(hidden, factor[2]), factor[3]) * model.scale
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
                checks[0] += 1
            hooks.append(layer.self_attn.qkv_proj.register_forward_hook(check))
        handle.generate(["The sky is", "Water freezes at"],
            SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True))
        diagnostics = execution.diagnostics()
        assert checks[0] >= 2 * len(handle.runner.model.model.layers)
        assert execution.directory.select(IndexedRequestIdentityBatch(())).refs == (current,)
        assert diagnostics.update_count == 1 and diagnostics.trained_rows > 0
        assert handle.runner.overlay_updated_apply_count > 0
        assert diagnostics.inference_pins == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
        print(dict(mode=mode, updates=diagnostics.update_count, loss_count=diagnostics.loss_count,
            qkv_oracle_checks=checks[0], apply_rows=handle.runner.overlay_apply_count,
            updated_apply_rows=handle.runner.overlay_updated_apply_count, fallback_count=diagnostics.fallback_count))
    finally:
        for hook in hooks:
            hook.remove()
        handle.close()
    assert execution.diagnostics().active_lineages == execution.diagnostics().inference_pins == 0


if __name__ == "__main__":
    main()
