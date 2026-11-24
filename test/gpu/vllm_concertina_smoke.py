"""End-to-end GPU smoke, contract check and HuggingFace alignment test for Concertina on pLLM/vLLM."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm import testing
from pllm.workloads.concertina import ConcertinaConfig, ConcertinaWorkload, prepare_concertina_checkpoint

DEFAULT_PROMPTS = [
    "The capital of France is",
    "Explain why folding adjacent transformer layers reduces memory:",
    "2 + 3 * 4 =",
]


def _hf_reference(
    model_dir: str, prompts: list[str], max_new_tokens: int
) -> tuple[list[str], list[torch.Tensor], float]:
    """Run official HF ConcertinaForCausalLM with trust_remote_code=True on cuda:0."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.bfloat16)
        .to("cuda:0")
        .eval()
    )
    outputs: list[str] = []
    first_logits: list[torch.Tensor] = []
    total_tokens = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
            first = model(**encoded).logits[0, -1].float().cpu()
            first_logits.append(first)
            generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
            new_tokens = generated[0, encoded["input_ids"].shape[1] :]
            total_tokens += int(new_tokens.numel())
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    elapsed = time.perf_counter() - t0
    del model
    torch.cuda.empty_cache()
    return outputs, first_logits, total_tokens / max(elapsed, 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Local HF Concertina directory or repo id.")
    parser.add_argument("--prepared", required=True, help="Output directory for prepared pLLM checkpoint.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--check-on-gpu", action="store_true", help="Run pllm.testing.check_on_gpu contract suite.")
    args = parser.parse_args()

    prepared_dir = Path(args.prepared)
    adapter_file = prepared_dir / "pllm_adapters" / "concertina_adapters.safetensors"
    has_weights = (prepared_dir / "model.safetensors").exists() or (
        prepared_dir / "model.safetensors.index.json"
    ).is_file()
    if (prepared_dir / "config.json").is_file() and adapter_file.is_file() and has_weights:
        config = ConcertinaConfig.from_checkpoint(str(prepared_dir))
    else:
        _, config = prepare_concertina_checkpoint(args.source, str(prepared_dir))

    hf_texts: list[str] = []
    hf_logits: list[torch.Tensor] = []
    if not args.skip_hf:
        hf_texts, hf_logits, hf_tps = _hf_reference(args.source, DEFAULT_PROMPTS, args.max_tokens)
        print(f"HF Concertina bs=1 throughput: {hf_tps:.1f} tok/s")
        for i, text in enumerate(hf_texts):
            print(f"HF[{i}]: {text!r}")

    workload = ConcertinaWorkload(config)
    model_cfg = pllm.ModelConfig(
        str(prepared_dir),
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=max(512, args.max_model_len),
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        enable_prefix_caching=False,
    )
    report = pllm.explain(workload, model_cfg)
    assert report.eager_reason == "", f"expected CUDA graphs, got eager_reason={report.eager_reason!r}"
    assert report.budget_errors == (), f"unexpected budget errors: {report.budget_errors}"

    if args.check_on_gpu:
        check_cfg = pllm.ModelConfig(
            str(prepared_dir),
            max_model_len=512,
            max_num_seqs=8,
            max_num_batched_tokens=512,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype="bfloat16",
        )
        check_result = testing.check_on_gpu(workload, check_cfg)
        print("check_on_gpu passed:", check_result.passed)

    with pllm.make_llm(model_cfg, workload=workload) as llm:
        diag = llm.diagnostics()
        print(f"pLLM eager_reason={diag.eager_reason!r}")
        assert diag.eager_reason == "", f"expected CUDA graphs on GPU, got {diag.eager_reason!r}"

        # Verify zero-copy linear weight tying across all folded groups when running in-process.
        runner = llm._runner
        if runner is not None:
            layers = runner.decoder.model.model.layers
            for group in config.groups:
                leader = layers[group[0]]
                for follower_idx in group[1:]:
                    follower = layers[follower_idx]
                    assert follower.self_attn.qkv_proj.weight.data_ptr() == leader.self_attn.qkv_proj.weight.data_ptr()
                    assert follower.self_attn.o_proj.weight.data_ptr() == leader.self_attn.o_proj.weight.data_ptr()
                    assert follower.mlp.gate_up_proj.weight.data_ptr() == leader.mlp.gate_up_proj.weight.data_ptr()
                    assert follower.mlp.down_proj.weight.data_ptr() == leader.mlp.down_proj.weight.data_ptr()
            unique_bytes = sum(
                p.numel() * p.element_size()
                for p in {p.data_ptr(): p for p in runner.decoder.model.parameters()}.values()
            )
            print(f"pLLM unique base model params: {unique_bytes / (1 << 30):.2f} GiB")

        params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=8)
        pllm_texts: list[str] = []
        for idx, prompt in enumerate(DEFAULT_PROMPTS):
            out = llm.generate([prompt], params, use_tqdm=False)
            seq = out[0].outputs[0]
            text = seq.text
            pllm_texts.append(text)
            print(f"pLLM[{idx}]: {text!r}")
            if hf_logits:
                hf_lp = torch.log_softmax(hf_logits[idx], dim=-1)
                top8_hf = torch.topk(hf_lp, 8)
                first_lp = seq.logprobs[0] if seq.logprobs else {}
                pllm_top1 = seq.token_ids[0]
                diffs = [
                    abs(float(first_lp[int(tid)].logprob) - float(val))
                    for tid, val in zip(top8_hf.indices.tolist(), top8_hf.values.tolist(), strict=True)
                    if int(tid) in first_lp
                ]
                max_diff = max(diffs) if diffs else float("inf")
                print(
                    f"Logprobs[{idx}]: top1_match={int(top8_hf.indices[0]) == pllm_top1} "
                    f"matched={len(diffs)}/8 top8_max_diff={max_diff:.4f}"
                )
                assert int(top8_hf.indices[0]) == pllm_top1, (
                    f"Prompt {idx} top-1 token mismatch: HF={int(top8_hf.indices[0])} vs pLLM={pllm_top1}"
                )
                assert len(diffs) >= 6 and max_diff <= 0.15, (
                    f"Prompt {idx} top-8 logprob diff {max_diff:.4f} (matched={len(diffs)}/8) exceeds 0.15 BF16 bound"
                )

        # Verify concurrent batch isolation on deterministic continuation prompts (bs=16 matches bs=1).
        batch_params = SamplingParams(temperature=0.0, max_tokens=16)
        det_prompts = ["Continue the sequence: 1, 2, 3,", "Continue the alphabet: A, B, C,"]
        det_single = [llm.generate([p], batch_params, use_tqdm=False)[0].outputs[0].text for p in det_prompts]
        batch_prompts = [det_prompts[i % len(det_prompts)] for i in range(16)]
        batch_out = llm.generate(batch_prompts, batch_params, use_tqdm=False)
        for i, result in enumerate(batch_out):
            expected = det_single[i % len(det_prompts)]
            actual = result.outputs[0].text
            assert actual == expected, f"Batch item {i} diverged from single-request output"

        # Throughput benchmark at bs=1, 16, 64 (64 output tokens, ignore_eos=True).
        bench_prompt = "Explain the architecture of modern transformer language models in clear technical terms."
        bench_sp = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)
        for bs in (1, 16, 64):
            prompts = [f"{bench_prompt} Variant {i}." for i in range(bs)]
            t0 = time.perf_counter()
            outs = llm.generate(prompts, bench_sp, use_tqdm=False)
            dt = time.perf_counter() - t0
            toks = sum(len(o.outputs[0].token_ids) for o in outs)
            print(f"CONCERTINA_BENCH bs={bs} toks={toks} dt={dt:.3f}s tok_per_s={toks / dt:.1f}")

    print("CONCERTINA_GPU_SMOKE_PASS")


if __name__ == "__main__":
    main()
