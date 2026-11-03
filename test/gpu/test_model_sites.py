"""Model sites on Qwen2.5-0.5B against independent FP32 Hugging Face oracles: residual readouts and layer logits
under CUDA graphs, and DoLa through pllm.layer_logits."""

import math
from dataclasses import dataclass, field
from functools import cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm.workloads.dola import DoLaWorkload

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = ("The capital of France is", "Water boils at a temperature of", "def fibonacci(n):")
LAYERS = (4, 12, 20, -1)
# Fixed vocabulary columns compared exactly; the logsumexp column covers the whole vocabulary.
IDS = tuple(range(0, 151936, 4751))
WIDTH = 64
CONFIG = pllm.ModelConfig(
    MODEL, max_model_len=128, max_num_seqs=4, max_num_batched_tokens=128, gpu_memory_utilization=0.3
)


@dataclass(frozen=True)
class Lens(pllm.Workload):
    """Export layer logits of sampling rows and residual streams of decode rows."""

    events: list[pllm.io.TraceEvent] = field(default_factory=lambda: [], compare=False)

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        self.events.append(event)

    @pllm.export(name="lens", max_rows=4, max_cols=len(LAYERS) * (1 + len(IDS)))
    def lens(self, logits: torch.Tensor = pllm.layer_logits(LAYERS)) -> torch.Tensor:
        return torch.cat((logits.logsumexp(-1), logits[:, :, IDS].flatten(1)), 1)

    @pllm.export(name="stream", max_rows=4, max_cols=2 * WIDTH)
    def stream(
        self,
        after: torch.Tensor = pllm.residual(12),
        middle: torch.Tensor = pllm.residual(12, after="attention"),
    ) -> torch.Tensor:
        return torch.cat((after[:, :WIDTH], middle[:, :WIDTH]), 1)


@cache
def _reference() -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()


@torch.no_grad()
def _hidden(tokens: list[int]) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    """FP32 streams after each layer (the last entry normalized), logits and the stream between layer 12's
    attention and MLP, which Hugging Face's post-attention norm reads."""
    model, middle = _reference(), []
    hook = model.model.layers[12].post_attention_layernorm.register_forward_pre_hook(
        lambda module, args: middle.append(args[0][0])
    )
    output = model(torch.tensor([tokens], device="cuda"), output_hidden_states=True)
    hook.remove()
    return tuple(value[0] for value in output.hidden_states), output.logits[0], middle[0]


@torch.no_grad()
def _layer_logits(tokens: list[int], layers: tuple[int, ...], rows: list[int]) -> torch.Tensor:
    """FP32 logit lens [rows, layers, vocab]; the last layer is the model's logits."""
    model = _reference()
    hidden, logits, _ = _hidden(tokens)
    count = len(hidden) - 1
    return torch.stack(
        [
            logits[rows]
            if layer % count == count - 1
            else model.lm_head(model.model.norm(hidden[layer % count + 1][rows]))
            for layer in layers
        ],
        1,
    )


def _exports(events: list[pllm.io.TraceEvent]) -> dict[tuple[str, int], torch.Tensor]:
    rows: dict[tuple[str, int], list[torch.Tensor]] = {}
    for event in sorted(events, key=lambda event: event.sequence):
        assert event.tensor_name is not None
        for identity, row in zip(event.identities, event.tensor(), strict=True):
            rows.setdefault((event.tensor_name, identity.prompt_index), []).append(row)
    return {key: torch.stack(value) for key, value in rows.items()}


def test_layer_logits_and_residuals_match_fp32_oracle():
    workload, tokenizer = Lens(), AutoTokenizer.from_pretrained(MODEL)
    with pllm.make_llm(CONFIG, workload=workload) as handle:
        assert handle.diagnostics().eager_reason == ""  # readouts run under vLLM's CUDA graphs
        outputs = handle.generate(list(PROMPTS), SamplingParams(max_tokens=5, temperature=0.0, ignore_eos=True))
    exports = _exports(workload.events)
    lens_error, stream_error = torch.zeros(len(LAYERS)), 0.0
    for index, (prompt, output) in enumerate(zip(PROMPTS, outputs, strict=True)):
        prompt_ids, generated = tokenizer(prompt).input_ids, list(output.outputs[0].token_ids)
        rows = list(range(len(prompt_ids) - 1, len(prompt_ids) + len(generated) - 1))
        logits = _layer_logits(prompt_ids + generated, LAYERS, rows)
        expected = torch.cat((logits.logsumexp(-1).unsqueeze(-1), logits[:, :, IDS]), -1).cpu()
        lens = exports["lens", index]
        actual = torch.cat(
            (lens[:, : len(LAYERS)].unsqueeze(-1), lens[:, len(LAYERS) :].view(len(rows), len(LAYERS), -1)), -1
        )
        lens_error = torch.maximum(lens_error, (actual - expected).abs().amax((0, 2)))
        # Residual markers still select decode only; the lens also includes the prompt end.
        rows = rows[1:]
        hidden, _, middle = _hidden(prompt_ids + generated)
        streams = torch.cat((hidden[13][rows, :WIDTH], middle[rows, :WIDTH]), 1).cpu()
        stream_error = max(
            stream_error, ((exports["stream", index] - streams).abs() / streams.abs().clamp_min(1)).max().item()
        )
    print(
        f"lens max-abs error vs FP32 per layer {dict(zip(LAYERS, lens_error.tolist()))}; stream error {stream_error:.4f}"
    )
    # Earlier layers' logits deviate from FP32 truth no more than vLLM's own bf16 final logits do.
    assert (lens_error[:-1] <= lens_error[-1]).all()
    assert stream_error < 0.1


def _dola_scores(logits: torch.Tensor, plausibility: float) -> torch.Tensor:
    """Independent DoLa: per row, contrast the final layer with the candidate of largest Jensen-Shannon divergence."""
    scores = []
    for row in logits:
        final, early = row[-1].log_softmax(-1), row[:-1].log_softmax(-1)
        divergences = []
        for candidate in early:
            mixture = torch.logsumexp(torch.stack((final, candidate)), 0) - math.log(2)
            divergences.append(
                (final.exp() * (final - mixture)).sum() + (candidate.exp() * (candidate - mixture)).sum()
            )
        chosen = early[int(torch.stack(divergences).argmax())]
        keep = final.exp() >= plausibility * final.exp().max()
        scores.append(torch.where(keep, final - chosen, -torch.inf))
    return torch.stack(scores)


def test_dola_greedy_tokens_follow_fp32_oracle():
    workload, tokenizer = DoLaWorkload(), AutoTokenizer.from_pretrained(MODEL)
    with pllm.make_llm(CONFIG, workload=workload) as handle:
        outputs = handle.generate(list(PROMPTS), SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))
    exact, worst = 0, 0.0
    for prompt, output in zip(PROMPTS, outputs, strict=True):
        prompt_ids, generated = tokenizer(prompt).input_ids, list(output.outputs[0].token_ids)
        # Every sampled token is contrasted: the one after the prompt and each decode token.
        rows = list(range(len(prompt_ids) - 1, len(prompt_ids) + len(generated) - 1))
        scores = _dola_scores(_layer_logits(prompt_ids + generated, workload.lens, rows), workload.plausibility)
        chosen = scores[torch.arange(len(rows)), torch.tensor(generated, device=scores.device)]
        exact += int((scores.argmax(-1) == torch.tensor(generated, device=scores.device)).sum())
        worst = max(worst, (scores.max(-1).values - chosen).max().item())
    print(f"DoLa: {exact}/{len(PROMPTS) * 8} greedy tokens equal the FP32 oracle's; worst score gap {worst:.4f}")
    # bf16 may flip a near-tie or the premature layer the divergence picks, never choose an implausible token.
    assert math.isfinite(worst) and exact >= len(PROMPTS) * 8 - 2
