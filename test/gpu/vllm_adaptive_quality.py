"""State-saturation measurements and an independent serial FP32 Hugging Face exit oracle.

Run `native` first, then `oracle` on the same JSON artifact. The oracle consumes the
native token prefixes, so exit disagreement measures arithmetic instead of text drift.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

import pllm


@dataclass(frozen=True)
class Saturation(pllm.Workload):
    threshold: float = 0.95

    @pllm.custom_forward(tokens="decode")
    def forward(self, layers):
        h = layers.embeddings
        active = torch.ones_like(h[:, :1], dtype=torch.bool)
        bounds = (0, layers.layer_count // 2, 3 * layers.layer_count // 4, layers.layer_count)
        for start, stop in zip(bounds, bounds[1:]):
            after = layers(start, stop, hidden=h, active=active)[stop - 1]
            similarity = F.cosine_similarity(after.float(), h.float(), dim=-1).unsqueeze(-1)
            active = active & (similarity <= self.threshold)
            h = after
        return h


def native(args):
    from vllm import SamplingParams

    from pllm.backends.vllm.adaptive_depth import AdaptiveSlice
    from pllm.backends.vllm.custom_forward import ProgramWave

    prompts = [
        f"Answer briefly. Example {i + 1}. " + question
        for i, question in enumerate(
            [
                "What is the capital of France?",
                "What is 17 times 23?",
                "Why is the sky blue?",
                "Name three mammals.",
                "Translate hello into French.",
                "Explain photosynthesis.",
                "Write a Python factorial function.",
                "What is gravity?",
                "What is 144 divided by 12?",
                "Name a prime between 20 and 30.",
                "Describe an ocean.",
                "How do birds fly?",
                "What causes rain?",
                "Write a sentence about autumn.",
                "Who wrote Hamlet?",
                "What is the square root of 81?",
                "Explain a binary search.",
                "Name the planets closest to the sun.",
                "What is a triangle?",
                "How does ice melt?",
                "What is 8 cubed?",
                "Why do leaves change color?",
                "Describe a healthy breakfast.",
                "What is a computer algorithm?",
                "Complete the sequence 2, 4, 8, 16.",
                "What is a noun?",
                "Explain tides.",
                "Name two renewable energy sources.",
                "Write a short greeting.",
                "What is 19 plus 28?",
                "What is an atom?",
                "How do magnets work?",
            ]
        )
    ]
    params = SamplingParams(max_tokens=args.tokens, temperature=0, ignore_eos=True)
    config = pllm.ModelConfig(
        args.model,
        max_num_seqs=8,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.75,
        execution=pllm.ExecutionOptions(capture_memory_bytes=2048 << 20),
    )
    result = {"model": args.model, "tokens": args.tokens, "prompts": prompts, "runs": []}
    call, capture, replay = AdaptiveSlice.__call__, ProgramWave.capture, torch.cuda.CUDAGraph.replay
    for threshold in (2.0, *args.thresholds):
        buffers, graphs, observations = {}, {}, []

        def called(selection, step, hidden, active):
            if torch.cuda.is_current_stream_capturing():
                buffers.setdefault(selection.wave, {})[step.start] = (step.stop, active)
            return call(selection, step, hidden, active)

        def captured(wave):
            capture(wave)
            graphs[wave.graph] = wave

        def replayed(graph):
            replay(graph)
            if graph in graphs:
                wave = graphs[graph]
                rows = tuple(wave.custom_forward.rows[index] for index in wave.indices[0].tolist())
                for start, (stop, mask) in buffers[wave].items():
                    observations.append(
                        (
                            tuple((row.identity.prompt_index, row.token_offset - row.prompt_length) for row in rows),
                            start,
                            stop,
                            mask.clone(),
                        )
                    )

        handle = pllm.make_llm(config, workload=Saturation(threshold))
        try:
            with (
                patch.object(AdaptiveSlice, "__call__", called),
                patch.object(ProgramWave, "capture", captured),
                patch.object(torch.cuda.CUDAGraph, "replay", replayed),
            ):
                outputs = handle.generate(prompts, params)
            depth = {}
            skipped = total = 0
            for rows, start, stop, mask in observations:
                for (prompt_index, token), active in zip(rows, mask.cpu().flatten().tolist(), strict=True):
                    key = (prompt_index, token)
                    depth[key] = max(depth.get(key, 0), stop if active else 0)
                    skipped += (stop - start) * (not active)
                    total += stop - start
            run = {
                "threshold": threshold,
                "skipped": skipped,
                "layer_rows": total,
                "requests": [
                    {
                        "prompt": output.prompt_token_ids,
                        "tokens": list(output.outputs[0].token_ids),
                        "depths": [depth[(prompt_index, token)] for token in range(args.tokens - 1)],
                    }
                    for prompt_index, output in enumerate(outputs)
                ],
            }
            result["runs"].append(run)
            baseline = result["runs"][0]["requests"]
            agree = sum(
                a == b
                for request, full in zip(run["requests"], baseline, strict=True)
                for a, b in zip(request["tokens"], full["tokens"], strict=True)
            )
            run["greedy_matches"], run["greedy_total"] = agree, len(prompts) * args.tokens
            print(
                f"threshold={threshold} skipped={skipped}/{total} greedy_agreement={agree}/{len(prompts) * args.tokens}",
                flush=True,
            )
            Path(args.output).write_text(json.dumps(result, indent=2))
        finally:
            handle.close()
        graphs.clear()
        buffers.clear()
        observations.clear()
        del handle
        import gc

        gc.collect()
        torch.cuda.empty_cache()


def oracle(args):
    from transformers import AutoModelForCausalLM, DynamicCache
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    artifact = json.loads(Path(args.output).read_text())
    model = AutoModelForCausalLM.from_pretrained(
        artifact["model"],
        dtype=torch.float32,
        device_map="auto",
        max_memory={0: "19GiB", "cpu": "24GiB"},
        attn_implementation="eager",
    )
    model.eval()
    layers = model.model.layers
    bounds = (len(layers) // 2, 3 * len(layers) // 4, len(layers))
    with torch.inference_mode():
        for run in artifact["runs"][1:]:
            correct = total = 0
            for request in run["requests"]:
                cache = DynamicCache(config=model.config)
                prompt = torch.tensor([request["prompt"]], device="cuda")
                model(prompt, past_key_values=cache, use_cache=True)
                for token, expected_depth in enumerate(request["depths"]):
                    position = len(request["prompt"]) + token
                    ids = torch.tensor([[request["tokens"][token]]], device="cuda")
                    hidden = model.model.embed_tokens(ids)
                    pos = torch.tensor([[position]], device=hidden.device)
                    cos, sin = model.model.rotary_emb(hidden, pos)
                    active, depth, before = True, 0, hidden.clone()
                    for ordinal, layer in enumerate(layers):
                        if active:
                            hidden = layer(
                                hidden, position_embeddings=(cos, sin), past_key_values=cache, use_cache=True
                            )
                            depth = ordinal + 1
                        else:
                            normal = layer.input_layernorm(hidden)
                            attn = layer.self_attn
                            q = attn.q_proj(normal).view(1, 1, -1, attn.head_dim).transpose(1, 2)
                            k = attn.k_proj(normal).view(1, 1, -1, attn.head_dim).transpose(1, 2)
                            v = attn.v_proj(normal).view(1, 1, -1, attn.head_dim).transpose(1, 2)
                            _, k = apply_rotary_pos_emb(q, k, cos.to(q.device), sin.to(q.device))
                            cache.update(k, v, ordinal)
                        if ordinal + 1 in bounds:
                            active = (
                                active
                                and float(F.cosine_similarity(hidden.to(before.device), before, dim=-1))
                                <= run["threshold"]
                            )
                            before = hidden.clone()
                    correct += depth == expected_depth
                    total += 1
            run["fp32_exit_matches"] = correct
            run["fp32_exit_total"] = total
            print(
                f"threshold={run['threshold']} fp32_exit_matches={correct}/{total} agreement={correct / total:.6f}",
                flush=True,
            )
            Path(args.output).write_text(json.dumps(artifact, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("native", "oracle"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--thresholds", nargs=5, type=float, default=(0.8, 0.9, 0.95, 0.99, 0.995))
    parser.add_argument("--output", default="/tmp/cap1m1-quality.json")
    args = parser.parse_args()
    (native if args.stage == "native" else oracle)(args)


if __name__ == "__main__":
    main()
