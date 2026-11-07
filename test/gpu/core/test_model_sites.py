"""Model sites on Qwen2.5-0.5B against independent FP32 Hugging Face oracles: residual readouts and layer logits
under CUDA graphs, and DoLa through pllm.layer_logits."""

import copy
import math
from dataclasses import dataclass, field, replace
from functools import cache

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm.workloads.dola import DoLaWorkload

pytestmark = pytest.mark.gpu


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


# Contrast pairs for a steering vector (CAA); longer than a 16-token step, so every prompt is prefilled in chunks.
PAIRS = (
    (
        "I love this movie, the story was wonderful and the actors were brilliant",
        "I hate this movie, the story was awful and the actors were terrible",
    ),
    (
        "What a fantastic day, everything went perfectly and everyone was happy",
        "What a miserable day, everything went wrong and everyone was upset",
    ),
    (
        "The food was delicious and the staff were kind and welcoming to us all",
        "The food was disgusting and the staff were rude and dismissive to us all",
    ),
)
CHUNKED = replace(CONFIG, max_num_batched_tokens=16, enable_chunked_prefill=True)


@dataclass(frozen=True)
class Residuals(pllm.Workload):
    """Export the residual stream after layer 12 for every computed token, the activations CAA averages."""

    events: list[pllm.io.TraceEvent] = field(default_factory=lambda: [], compare=False)

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        self.events.append(event)

    @pllm.export(name="residual", max_rows=16, max_cols=896, overflow="block")
    def residual(self, h: torch.Tensor = pllm.residual(12, tokens="all")) -> torch.Tensor:
        return h


def _relative(value: torch.Tensor, truth: torch.Tensor) -> float:
    """The largest per-row relative L2 error."""
    return ((value - truth).norm(dim=-1) / truth.norm(dim=-1)).max().item()


def _steering_vector(streams: list[torch.Tensor]) -> torch.Tensor:
    """CAA: the mean over pairs of (positive - negative) prompt activations, each averaged over its tokens."""
    means = torch.stack([stream.mean(0) for stream in streams])
    return (means[0::2] - means[1::2]).mean(0)


def test_all_token_residuals_give_the_fp32_oracle_steering_vector():
    workload, tokenizer = Residuals(), AutoTokenizer.from_pretrained(MODEL)
    prompts = [text for pair in PAIRS for text in pair]
    with pllm.make_llm(CHUNKED, workload=workload) as handle:
        assert handle.diagnostics().eager_reason == ""
        handle.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0))
    rows: dict[tuple[int, int], torch.Tensor] = {}
    for event in workload.events:
        assert event.dropped_rows == 0
        for identity, position, row in zip(event.identities, event.positions, event.tensor(), strict=True):
            assert (identity.prompt_index, position) not in rows  # every token once, across chunks
            rows[identity.prompt_index, position] = row
    rounded_model = copy.deepcopy(_reference()).to(torch.bfloat16)
    exported, truth, rounded = [], [], []
    for index, prompt in enumerate(prompts):
        ids = tokenizer(prompt).input_ids
        exported.append(torch.stack([rows.pop((index, position)) for position in range(len(ids))]))
        truth.append(_hidden(ids)[0][13].cpu())
        with torch.no_grad():
            output = rounded_model(torch.tensor([ids], device="cuda"), output_hidden_states=True)
        rounded.append(output.hidden_states[13][0].float().cpu())
    # Async scheduling may compute one decode token past max_tokens; prompt tokens are all accounted for above.
    assert all(position >= len(tokenizer(prompts[index]).input_ids) for index, position in rows)
    errors = {
        name: (
            max(_relative(value, oracle) for value, oracle in zip(streams, truth, strict=True)),
            _relative(_steering_vector(streams), _steering_vector(truth)),
        )
        for name, streams in (("pllm", exported), ("HF bf16", rounded))
    }
    print(f"relative error vs FP32 (worst token, steering vector): {errors}; {len(workload.events)} export events")
    # bf16 rounding is the reference: pllm's error is within twice that of the same model run by Hugging Face.
    assert all(ours <= 2 * theirs for ours, theirs in zip(errors["pllm"], errors["HF bf16"], strict=True))


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


@pytest.mark.parametrize("bits,group", ((4, -1), (4, 32), (8, -1), (8, 32)))
def test_marlin_inverse_matches_dense_quantized_weight(bits, group):
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import get_weight_perm, marlin_weights
    from vllm.model_executor.layers.quantization.utils.quant_utils import quantize_weights
    from vllm.scalar_type import scalar_types

    from pllm.backends.vllm.marlin import dequantize

    generator = torch.Generator(device="cuda").manual_seed(71)
    weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
    quant_type = scalar_types.uint4b8 if bits == 4 else scalar_types.uint8b128
    dense, quantized, scales, _ = quantize_weights(weight, quant_type, group)
    packed = marlin_weights(quantized, *weight.shape, bits, get_weight_perm(bits))
    scales = marlin_permute_scales(scales, *weight.shape, group)
    actual = dequantize(packed, scales, None, tuple(weight.shape), quant_type, group)
    torch.testing.assert_close(actual, dense.T, rtol=torch.finfo(weight.dtype).eps, atol=0)
    # A later read reconstructs from the quantized storage instead of retaining a dense cache.
    scales.mul_(2)
    torch.testing.assert_close(dequantize(packed, scales, None, tuple(weight.shape), quant_type, group), 2 * dense.T)
