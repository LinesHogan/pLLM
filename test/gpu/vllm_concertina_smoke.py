"""GPU end-to-end verification for ConcertinaWorkload against Hugging Face ConcertinaForCausalLM."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm.workloads.concertina import ConcertinaWorkload, prepare_concertina_checkpoint

DEFAULT_PROMPTS = (
    "The capital of France is",
    "In linear algebra, the singular value decomposition of a matrix",
    "Question: If a train travels at 60 miles per hour for 2.5 hours, how far does it travel?\nAnswer:",
)


def run_hf_reference(model_dir: str, prompts: tuple[str, ...], max_new_tokens: int) -> tuple[list[str], float, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.bfloat16)
        .to("cuda:0")
        .eval()
    )
    weight_gb = torch.cuda.memory_allocated() / (1024**3)
    outputs: list[str] = []
    # Warmup
    warm = tokenizer(prompts[0], return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        _ = model.generate(**warm, max_new_tokens=4, do_sample=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total_tokens = 0
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0, inputs["input_ids"].shape[1] :]
        total_tokens += int(gen.numel())
        outputs.append(tokenizer.decode(gen, skip_special_tokens=True))
    torch.cuda.synchronize()
    tps = total_tokens / max(time.perf_counter() - t0, 1e-6)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs, weight_gb, tps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF repo id or local Concertina checkpoint directory")
    parser.add_argument("--prepared-dir", required=True, help="Temporary directory for prepared native checkpoint")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--skip-hf", action="store_true")
    args = parser.parse_args()

    prepared_dir, cfg = prepare_concertina_checkpoint(args.model, args.prepared_dir)
    print(f"Prepared Concertina checkpoint at {prepared_dir}: {cfg}")
    assert Path(prepared_dir, "model.safetensors").is_symlink()

    hf_outputs: list[str] | None = None
    if not args.skip_hf:
        hf_outputs, hf_gb, hf_tps = run_hf_reference(args.model, DEFAULT_PROMPTS, args.max_tokens)
        print(f"[HF]   weight_mem={hf_gb:.2f} GiB  bs=1_tps={hf_tps:.1f} tok/s")
        for idx, text in enumerate(hf_outputs):
            print(f"[HF   #{idx}] {text!r}")

    workload = ConcertinaWorkload(cfg)
    model_cfg = pllm.ModelConfig(
        prepared_dir,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max(512, args.max_model_len),
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    report = pllm.explain(workload, model_cfg)
    assert report.eager_reason == "", f"expected CUDA graphs, got: {report.eager_reason}"
    assert report.budget_errors == (), f"unexpected budget errors: {report.budget_errors}"
    print(f"[pLLM] explain OK: eager_reason={report.eager_reason!r}")

    greedy = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    with pllm.make_llm(model_cfg, workload=workload) as llm:
        diag = llm.diagnostics()
        print(f"[pLLM] diagnostics: eager_reason={diag.eager_reason!r}")
        assert diag.eager_reason == "", f"expected CUDA graphs on GPU, got: {diag.eager_reason}"

        # Verify linear weights are physically tied across every group while norms stay independent.
        model = llm.runner.decoder.model.model
        for group in cfg.groups:
            leader = model.layers[group[0]]
            for follower_idx in group[1:]:
                follower = model.layers[follower_idx]
                assert follower.self_attn.qkv_proj.weight.data_ptr() == leader.self_attn.qkv_proj.weight.data_ptr()
                assert follower.self_attn.o_proj.weight.data_ptr() == leader.self_attn.o_proj.weight.data_ptr()
                assert follower.mlp.gate_up_proj.weight.data_ptr() == leader.mlp.gate_up_proj.weight.data_ptr()
                assert follower.mlp.down_proj.weight.data_ptr() == leader.mlp.down_proj.weight.data_ptr()
                assert follower.input_layernorm.weight.data_ptr() != leader.input_layernorm.weight.data_ptr()
                assert (
                    follower.post_attention_layernorm.weight.data_ptr()
                    != leader.post_attention_layernorm.weight.data_ptr()
                )
        unique_param_bytes = sum(p.numel() * p.element_size() for p in llm.runner.decoder.model.parameters())
        print(
            f"[pLLM] verified {len(cfg.groups)} tied layer groups; unique base model params={unique_param_bytes / (1024**3):.2f} GiB"
        )

        # Warmup + timed bs=1 and batched runs
        _ = llm.generate([DEFAULT_PROMPTS[0]], SamplingParams(max_tokens=4, temperature=0.0))
        pllm_outputs: list[str] = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total_tokens = 0
        for prompt in DEFAULT_PROMPTS:
            (out,) = llm.generate([prompt], greedy)
            pllm_outputs.append(out.outputs[0].text)
            total_tokens += len(out.outputs[0].token_ids)
        torch.cuda.synchronize()
        pllm_bs1_tps = total_tokens / max(time.perf_counter() - t0, 1e-6)
        print(f"[pLLM] bs=1_tps={pllm_bs1_tps:.1f} tok/s")
        for idx, text in enumerate(pllm_outputs):
            print(f"[pLLM #{idx}] {text!r}")

        if hf_outputs is not None:
            for idx, (hf_text, pllm_text) in enumerate(zip(hf_outputs, pllm_outputs, strict=True)):
                match = hf_text == pllm_text
                print(f"[MATCH #{idx}] exact={match}")

        # Concurrent batch test (16 requests)
        batch_prompts = [DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)] for i in range(args.max_num_seqs)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch_outs = llm.generate(batch_prompts, greedy)
        torch.cuda.synchronize()
        batch_tokens = sum(len(o.outputs[0].token_ids) for o in batch_outs)
        batch_tps = batch_tokens / max(time.perf_counter() - t0, 1e-6)
        print(f"[pLLM] bs={args.max_num_seqs} throughput={batch_tps:.1f} tok/s ({batch_tokens} tokens)")
        for i in range(args.max_num_seqs):
            assert batch_outs[i].outputs[0].text == pllm_outputs[i % len(DEFAULT_PROMPTS)], (
                f"batch isolation mismatch at index {i}"
            )
        print("CONCERTINA_GPU_SMOKE_PASS")


if __name__ == "__main__":
    main()
