"""C2 acceptance on Qwen2.5-0.5B: one workload composes @pllm.custom_forward (a loop over the last layers),
@pllm.custom_residual (LAuReL residual updates) and @pllm.train_in_forward (In-Place TTT chunk arithmetic on the
looped layers' down projections), checked token by token against an independent FP32 torch oracle."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cache

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm.model import Projection
from test.gpu.features.test_custom_residual import BF16_NOISE, logprob_deviation, oracle_logprobs, vanilla_tokens

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = ("The capital of France is", "Water boils at a temperature of", "def fibonacci(n):")
LOOPED = (22, 23)
CONFIG = pllm.ModelConfig(
    MODEL,
    max_model_len=128,
    max_num_seqs=4,
    max_num_batched_tokens=128,
    gpu_memory_utilization=0.3,
    execution=pllm.ExecutionOptions(cuda_graph_pool_budget_bytes=512 << 20),
)


@dataclass(frozen=True)
class LoopedTTT(pllm.Workload):
    """Re-run the last two layers after a test-time-training update of their down projections, with LAuReL residual
    updates at every sublayer of every layer visit.

    The update is In-Place TTT's chunk arithmetic: targets are the MLP inputs, convolved per chunk with a depthwise
    kernel and projected; W_down += lr * values^T inputs once a chunk completes, and rows read the incoming W.
    """

    chunk: int = 1
    learning_rate: float = 1e-5
    scale: float = 0.05

    @property
    def targets(self) -> tuple[Projection, ...]:
        return tuple(Projection("down_proj", layer) for layer in LOOPED)

    @property
    def target_parameters(self) -> pllm.InstanceState:
        return pllm.InstanceState(
            {
                "conv": pllm.Field(lambda d, _: (d, 1, 5)),
                "proj": pllm.Field(lambda d, _: (d, d)),
            },
            layers=LOOPED,
            init=pllm.Computed(self.initial_targets),
        )

    def initial_targets(self, fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
        values = target_values(fields["conv_22"].device, fields["conv_22"].dtype)
        for name, value in values.items():
            fields[name].copy_(value)

    @property
    def fast_weights(self) -> pllm.RequestState:
        return pllm.RequestState(
            {"delta": pllm.Field(lambda inputs, outputs: (outputs, inputs))},
            key_fields=("request_id", "prompt_index", "sample_index"),
        )

    @property
    def laurel(self) -> pllm.InstanceState:
        return pllm.InstanceState(
            {"alpha": lambda m: (2 * m.layer_count,), "beta": lambda m: (2 * m.layer_count,)},
            init=pllm.Computed(self.initial_gains),
            dtype=torch.float32,
        )

    def initial_gains(self, fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
        alpha, beta = laurel_gains(fields["alpha"].device, self.scale)
        fields["alpha"].copy_(alpha)
        fields["beta"].copy_(beta)

    @pllm.custom_forward(tokens="all")
    def settle(self, layers: pllm.model.Layers, alpha=pllm.parameter("laurel", "alpha")) -> torch.Tensor:
        taps = layers()
        layers.apply_updates()
        hidden = layers(start=LOOPED[0], stop=LOOPED[-1] + 1, hidden=taps[LOOPED[0] - 1])[-1]
        return torch.lerp(taps[-1], hidden, alpha[0].sigmoid().to(hidden.dtype))

    @pllm.train_in_forward(
        visibility="exact",
        state="fast_weights",
        targets=lambda self: self.targets,
        weight="delta",
        chunk=lambda self: self.chunk,
        tokens="all",
    )
    def step(
        self,
        delta: torch.Tensor = pllm.parameter("fast_weights", "delta"),
        inputs: torch.Tensor = pllm.operand("input"),
        targets: torch.Tensor = pllm.operand("target"),
        conv: torch.Tensor = pllm.parameter("target_parameters", "conv"),
        proj: torch.Tensor = pllm.parameter("target_parameters", "proj"),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = F.conv1d(targets.T.unsqueeze(0), conv, padding=2, groups=targets.shape[1]).squeeze(0).T @ proj
        update = (values.T @ inputs) * self.learning_rate
        return delta + update.float(), inputs @ delta.to(inputs.dtype).T

    @pllm.custom_residual(site="both")
    def residual(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        s: int = pllm.depth(),
        alpha: torch.Tensor = pllm.parameter("laurel", "alpha"),
        beta: torch.Tensor = pllm.parameter("laurel", "beta"),
    ) -> torch.Tensor:
        return (alpha[s] * y.float() + beta[s] * x.float()).to(x.dtype)


def target_values(device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """A near-identity depthwise kernel and projection per looped layer, from a fixed seed."""
    generator = torch.Generator(device="cpu").manual_seed(11)
    values: dict[str, torch.Tensor] = {}
    for layer in LOOPED:
        conv = torch.randn(896, 1, 5, generator=generator) * 0.05
        conv[:, 0, 2] += 1
        values[f"conv_{layer}"] = conv.to(device, dtype)
        values[f"proj_{layer}"] = (torch.eye(896) + torch.randn(896, 896, generator=generator) * 0.01).to(device, dtype)
    return values


def laurel_gains(device: torch.device, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(13)
    alpha = 1 + torch.randn(48, generator=generator) * scale
    beta = 1 + torch.randn(48, generator=generator) * scale / 4
    return alpha.to(device), beta.to(device)


@cache
def _reference() -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()


def _rms(norm: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    return norm.weight * value * torch.rsqrt(value.square().mean(-1, keepdim=True) + norm.variance_epsilon)


@torch.no_grad()
def _oracle(tokens: list[int], workload: LoopedTTT) -> torch.Tensor:
    """FP32 log-probabilities, one token at a time: every position runs the program (a full pass committing K/V,
    the TTT boundary, a re-run of the looped layers committing their K/V again), with LAuReL at every sublayer of
    every visit and the down projections of the looped layers reading their request's fast weights."""
    model = _reference()
    body, config = model.model, model.config
    heads, kv_heads, width = config.num_attention_heads, config.num_key_value_heads, config.hidden_size
    head_dim = width // heads
    alpha, beta = laurel_gains(torch.device("cuda"), workload.scale)
    targets = {name: value.float() for name, value in target_values(torch.device("cuda"), torch.bfloat16).items()}
    keys = torch.zeros(len(body.layers), len(tokens), kv_heads, head_dim, device="cuda")
    values = torch.zeros_like(keys)
    delta = {layer: torch.zeros(width, config.intermediate_size, device="cuda") for layer in LOOPED}
    pending: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {layer: [] for layer in LOOPED}
    logprobs = []

    def visit(index: int, stream: torch.Tensor, position: int, operands: list[tuple[int, torch.Tensor, torch.Tensor]]):
        layer = body.layers[index]
        attention = layer.self_attn
        normed = _rms(layer.input_layernorm, stream)
        q = attention.q_proj(normed).view(heads, head_dim)
        k = attention.k_proj(normed).view(kv_heads, head_dim)
        v = attention.v_proj(normed).view(kv_heads, head_dim)
        cos, sin = body.rotary_emb(stream[None, None], torch.tensor([[position]], device="cuda"))
        cos, sin = cos[0, 0], sin[0, 0]

        def rotate(value: torch.Tensor) -> torch.Tensor:
            first, second = value.chunk(2, -1)
            return value * cos + torch.cat((-second, first), -1) * sin

        keys[index, position], values[index, position] = rotate(k), v
        group = heads // kv_heads
        past_keys = keys[index, : position + 1].repeat_interleave(group, 1)
        past_values = values[index, : position + 1].repeat_interleave(group, 1)
        scores = torch.einsum("hd,phd->hp", rotate(q), past_keys) / head_dim**0.5
        attended = torch.einsum("hp,phd->hd", scores.softmax(-1), past_values).reshape(-1)
        stream = alpha[2 * index] * attention.o_proj(attended) + beta[2 * index] * stream
        normed = _rms(layer.post_attention_layernorm, stream)
        inputs = F.silu(layer.mlp.gate_proj(normed)) * layer.mlp.up_proj(normed)
        output = layer.mlp.down_proj(inputs)
        if index in delta:
            output = output + delta[index] @ inputs
            operands.append((index, inputs, normed))
        return alpha[2 * index + 1] * output + beta[2 * index + 1] * stream

    for position, token in enumerate(tokens):
        stream, operands, after = body.embed_tokens(torch.tensor(token, device="cuda")), [], {}
        for index in range(len(body.layers)):
            stream = visit(index, stream, position, operands)
            after[index] = stream
        for index, inputs, normed in operands:  # the TTT boundary trains on the full pass's operands
            pending[index].append((inputs, normed))
            if len(pending[index]) == workload.chunk:
                rows, chunk_targets = (torch.stack(column) for column in zip(*pending[index], strict=True))
                convolved = (
                    F.conv1d(chunk_targets.T.unsqueeze(0), targets[f"conv_{index}"], padding=2, groups=width)
                    .squeeze(0)
                    .T
                )
                delta[index] += workload.learning_rate * (convolved @ targets[f"proj_{index}"]).T @ rows
                pending[index] = []
        stream = after[LOOPED[0] - 1]
        for index in LOOPED:  # the re-run reads the updated fast weights; its operands are not trained
            stream = visit(index, stream, position, [])
        stream = torch.lerp(after[len(body.layers) - 1], stream, alpha[0].sigmoid())
        logprobs.append(model.lm_head(_rms(body.norm, stream)).log_softmax(-1))
    return torch.stack(logprobs)


def test_custom_forward_custom_residual_and_ttt_compose_on_qwen() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    baseline, _ = logprob_deviation(vanilla_tokens(), oracle_logprobs)
    sampling = SamplingParams(max_tokens=10, temperature=0.0, ignore_eos=True, logprobs=4)
    for chunk in (1, 4):
        workload = LoopedTTT(chunk=chunk)
        with pllm.make_llm(CONFIG, workload=workload) as handle:
            outputs = handle.generate(list(PROMPTS), sampling)
        worst, misses, composed, trained = 0.0, 0, 0.0, 0.0
        for prompt, output in zip(PROMPTS, outputs, strict=True):
            prompt_ids, generated = tokenizer(prompt).input_ids, list(output.outputs[0].token_ids)
            sequence, rows = prompt_ids + generated, slice(len(prompt_ids) - 1, -1)
            expected = _oracle(sequence, workload)[rows]
            index = torch.arange(len(generated)), torch.tensor(generated, device="cuda")
            chosen = expected[index]
            actual = [
                step[token].logprob for token, step in zip(generated, output.outputs[0].logprobs or [], strict=True)
            ]
            worst = max(worst, (chosen.cpu() - torch.tensor(actual)).abs().max().item())
            misses += int((expected.argmax(-1).cpu() != torch.tensor(generated)).sum())
            composed = max(composed, (chosen - oracle_logprobs(sequence)[rows][index]).abs().max().item())
            untrained = _oracle(sequence, replace(workload, learning_rate=0.0))[rows][index]
            trained = max(trained, (chosen - untrained).abs().max().item())
        print(
            f"chunk={chunk}: max logprob gap to the FP32 oracle {worst:.4f} (plain bf16 vLLM {baseline:.4f}), "
            f"{misses} non-argmax tokens; logprobs move by up to {composed:.4f} against the plain model and "
            f"{trained:.4f} against the same program without the TTT update"
        )
        assert min(composed, trained) > 5 * max(baseline, 0.01), "every effect must matter beyond bf16 noise"
        assert worst <= max(2 * baseline, BF16_NOISE) and misses <= 2
