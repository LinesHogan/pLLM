"""@pllm.custom_residual on Qwen2.5-0.5B: identity rewrites reproduce the model bitwise, and LAuReL, value residual
and static hyper-connections follow an independent FP32 Hugging Face oracle as closely as plain bf16 vLLM does."""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import cache
from typing import Literal

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

import pllm
from pllm.testing import check_on_gpu
from test.workloads.test_gpu_contracts import fixed_kv_cache as fixed_kv_cache

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = ("The capital of France is", "Water boils at a temperature of", "def fibonacci(n):")
CONFIG = pllm.ModelConfig(
    MODEL,
    max_model_len=128,
    max_num_seqs=4,
    max_num_batched_tokens=128,
    # These tiny schedules need little KV; leave room for the resident FP32 oracle on a 24 GiB GPU.
    gpu_memory_utilization=0.2,
    cuda_graphs=False,
    execution=pllm.ExecutionOptions(cuda_graph_pool_budget_bytes=256 << 20),
)
# The largest per-token logprob gap of bf16 vLLM to an FP32 oracle over these prompts: one near-tie dominates it, so it
# varies by GPU (0.24 on an RTX 4090, 0.06 on an A100) while a wrong residual update moves logprobs by more than 1.
BF16_NOISE = 0.5
SAMPLING = SamplingParams(max_tokens=12, temperature=0.0, ignore_eos=True, logprobs=4)


@dataclass(frozen=True)
class Observe(pllm.Workload):
    """Leaves the forward alone: the reference the rewrites run beside, on the same pllm runner."""

    @pllm.export(max_cols=1)
    def mean(self, h: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
        return h.mean(-1, keepdim=True)


@dataclass(frozen=True)
class Laurel(pllm.Workload):
    """LAuReL-RW+LR (Menghani et al., 2024) at every sublayer: alpha y + beta x + (x A^T) B^T; scale=0 is the model."""

    rank: int = 4
    scale: float = 0.0

    @property
    def laurel(self) -> pllm.InstanceState:
        rank = self.rank
        return pllm.InstanceState(
            {
                "alpha": lambda m: (2 * m.layer_count,),
                "beta": lambda m: (2 * m.layer_count,),
                "a": lambda m: (2 * m.layer_count, rank, m.hidden_size),
                "b": lambda m: (2 * m.layer_count, m.hidden_size, rank),
            },
            init=pllm.Computed(self.initial),
            dtype=torch.float32,
        )

    def initial(self, fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
        generator = torch.Generator(device=fields["a"].device).manual_seed(3)
        fields["alpha"].fill_(1).add_(
            torch.randn(fields["alpha"].shape, generator=generator, device=fields["alpha"].device) * self.scale
        )
        fields["beta"].fill_(1)
        fields["a"].normal_(0, 0.02, generator=generator)
        fields["b"].normal_(0, 0.02, generator=generator).mul_(self.scale)

    @pllm.custom_residual(site="both")
    def update(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        s: int = pllm.depth(),
        alpha: torch.Tensor = pllm.parameter("laurel", "alpha"),
        beta: torch.Tensor = pllm.parameter("laurel", "beta"),
        a: torch.Tensor = pllm.parameter("laurel", "a"),
        b: torch.Tensor = pllm.parameter("laurel", "b"),
    ) -> torch.Tensor:
        x32 = x.float()
        return (alpha[s] * y.float() + beta[s] * x32 + (x32 @ a[s].T) @ b[s].T).to(x.dtype)


@dataclass(frozen=True)
class ValueResidual(pllm.Workload):
    """ResFormer value residual: later layers mix in layer 0's values, V_l + lam_l (V_0 - V_l); lam=0 is the model."""

    mixing: float = 0.0

    @property
    def gates(self) -> pllm.InstanceState:
        return pllm.InstanceState(
            {"lam": lambda m: (m.layer_count,)}, init=pllm.Computed(self.initial), dtype=torch.float32
        )

    def initial(self, fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
        fields["lam"].fill_(self.mixing)

    @property
    def later(self) -> tuple[int, ...]:
        return tuple(range(1, 24))

    @pllm.custom_residual(site="value", layers=lambda self: self.later)
    def mix(
        self,
        v: torch.Tensor,
        first: torch.Tensor = pllm.value(0),
        layer: int = pllm.depth(),
        lam: torch.Tensor = pllm.parameter("gates", "lam"),
    ) -> torch.Tensor:
        return (v.float() + lam[layer] * (first.float() - v.float())).to(v.dtype)


@dataclass(frozen=True)
class StaticHyperConnections(pllm.Workload):
    """Static hyper-connections (Zhu et al., 2024) with n residual streams; identity routing reproduces the model."""

    n: int = 4
    perturb: float = 0.0

    @property
    def routing(self) -> pllm.InstanceState:
        n = self.n
        return pllm.InstanceState(
            {
                "read": lambda m: (2 * m.layer_count, n),
                "mix": lambda m: (2 * m.layer_count, n, n),
                "write": lambda m: (2 * m.layer_count, n),
            },
            init=pllm.Computed(self.initial),
            dtype=torch.float32,
        )

    def initial(self, fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
        device = fields["read"].device
        generator = torch.Generator(device=device).manual_seed(5)
        fields["read"].fill_(1 / self.n)
        fields["write"].fill_(1)
        fields["mix"].copy_(torch.eye(self.n, device=device).expand_as(fields["mix"]))
        noise = torch.randn(fields["mix"].shape, generator=generator, device=device) * self.perturb
        fields["mix"].add_(noise - noise.mean(-1, keepdim=True))

    def choose(
        self, x: torch.Tensor, s: int = pllm.depth(), read: torch.Tensor = pllm.parameter("routing", "read")
    ) -> torch.Tensor:
        return torch.einsum("n,rnd->rd", read[s], x.float()).to(x.dtype)

    @pllm.custom_residual(site="both", read="choose", streams=lambda self: self.n)
    def update(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        s: int = pllm.depth(),
        mix: torch.Tensor = pllm.parameter("routing", "mix"),
        write: torch.Tensor = pllm.parameter("routing", "write"),
    ) -> torch.Tensor:
        return (torch.einsum("mn,rnd->rmd", mix[s], x.float()) + write[s][:, None] * y.float()[:, None]).to(x.dtype)

    @pllm.custom_residual(site="final")
    def collapse(self, x: torch.Tensor) -> torch.Tensor:
        return x.float().mean(1).to(x.dtype)


@dataclass(frozen=True)
class FastWeightMemory(pllm.Workload):
    """A fast-weight memory beside one layer's MLP (TTT-Linear / Titans style): every row reads M k for its normalized
    stream k, and once a chunk completes M takes one gradient step on |M k - y|^2 over the chunk; gate=0 is the model."""

    layer: int = 12
    chunk: int = 16
    lr: float = 0.5
    gate: float = 0.25
    shared: bool = False

    @property
    def memory(self) -> pllm.RequestState:
        return pllm.RequestState(
            {"M": lambda m: (m.hidden_size, m.hidden_size)},
            key_fields=("prompt_index",) if self.shared else ("request_id",),
        )

    @property
    def layers_(self) -> tuple[int, ...]:
        return (self.layer,)

    @pllm.custom_residual(site="mlp", layers=lambda self: self.layers_, state="memory", chunk=lambda self: self.chunk)
    def remember(
        self, x: torch.Tensor, y: torch.Tensor, M: torch.Tensor = pllm.parameter("memory", "M")
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = torch.nn.functional.normalize(x.float(), dim=-1)
        read = k @ M.T
        return (x.float() + y.float() + self.gate * read).to(x.dtype), M - self.lr / len(x) * (read - y.float()).T @ k


def generate_tokens(workload: pllm.Workload, config: pllm.ModelConfig = CONFIG) -> list[tuple[list[int], list[float]]]:
    """Greedy tokens and their logprobs."""
    with pllm.make_llm(config, workload=workload) as handle:
        outputs = handle.generate(list(PROMPTS), SAMPLING)
    return [
        (
            list(output.outputs[0].token_ids),
            [
                step[token].logprob
                for token, step in zip(output.outputs[0].token_ids, output.outputs[0].logprobs or [], strict=True)
            ],
        )
        for output in outputs
    ]


@cache
def vanilla_tokens() -> list[tuple[list[int], list[float]]]:
    """Generate the baseline tokens and log probabilities without a residual rewrite."""
    return generate_tokens(Observe())


@pytest.mark.parametrize(
    "workload", [Laurel(), ValueResidual(), StaticHyperConnections(), FastWeightMemory(gate=0.0)], ids=type
)
@pytest.mark.parametrize("graphs", (False, "auto"))
def test_identity_rewrites_reproduce_the_model_bitwise(workload: pllm.Workload, graphs: bool | Literal["auto"]) -> None:
    config = replace(CONFIG, cuda_graphs=graphs)
    assert generate_tokens(workload, config) == vanilla_tokens()


def test_site_compile_failure_names_method_and_allows_opt_out() -> None:
    """Fullgraph failures happen during engine binding; the numerical opt-out runs the authored body."""

    @dataclass(frozen=True)
    class Untraceable(pllm.Workload):
        @pllm.custom_residual(site="final")
        def collapse(self, x: torch.Tensor) -> torch.Tensor:
            memoryview(b"\0")[0]
            return x

    with pytest.raises(
        ValueError, match=r"(?s)model-site method `collapse`.*full graph.*memoryview.*allow_numerical_transforms=False"
    ):
        with pllm.make_llm(CONFIG, workload=Untraceable()):
            pass
    eager = replace(CONFIG, execution=replace(CONFIG.execution, allow_numerical_transforms=False))
    assert generate_tokens(Untraceable(), eager) == vanilla_tokens()


@cache
def _reference() -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()


Rewrite = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


@torch.no_grad()
def oracle_logprobs(
    tokens: list[int],
    *,
    attention: Rewrite | None = None,
    mlp: Rewrite | None = None,
    value: Callable[[torch.Tensor, dict[int, torch.Tensor], int], torch.Tensor] | None = None,
    read: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    final: Callable[[torch.Tensor], torch.Tensor] | None = None,
    streams: int = 1,
) -> torch.Tensor:
    """FP32 log-probabilities of a token sequence through Hugging Face's Qwen2 modules with the rewrites applied in
    their own arithmetic: the stream after each sublayer, value rows and branch inputs, written from the papers."""
    model = _reference()
    body = model.model
    hidden = body.embed_tokens(torch.tensor(tokens, device="cuda"))
    rotary = body.rotary_emb(hidden[None], torch.arange(len(tokens), device="cuda")[None])
    stream = hidden if streams == 1 else hidden[:, None].expand(-1, streams, -1)
    values: dict[int, torch.Tensor] = {}
    for index, layer in enumerate(body.layers):
        attention_module = layer.self_attn
        projection = attention_module.v_proj

        def rewritten(x: torch.Tensor, layer: int = index, projection: torch.nn.Module = projection) -> torch.Tensor:
            rows = torch.nn.functional.linear(x, projection.weight, projection.bias)
            shaped = rows.view(*rows.shape[:-1], -1, attention_module.head_dim)
            if value is not None:
                shaped = value(shaped, values, layer)
            values[layer] = shaped
            return shaped.reshape(rows.shape)

        projection.forward = rewritten
        branch = stream if read is None else read(stream, 2 * index)
        output, _ = attention_module(
            layer.input_layernorm(branch)[None], position_embeddings=rotary, attention_mask=None
        )
        del projection.forward
        stream = stream + output[0] if attention is None else attention(stream, output[0], index)
        branch = stream if read is None else read(stream, 2 * index + 1)
        output = layer.mlp(layer.post_attention_layernorm(branch))
        stream = stream + output if mlp is None else mlp(stream, output, index)
    if final is not None:
        stream = final(stream)
    return model.lm_head(body.norm(stream)).log_softmax(-1)


def logprob_deviation(
    runs: list[tuple[list[int], list[float]]], oracle: Callable[[list[int]], torch.Tensor]
) -> tuple[float, int]:
    """The largest logprob gap to the oracle over the greedy tokens, and how many tokens are not its argmax."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    worst, misses = 0.0, 0
    for prompt, (tokens, logprobs) in zip(PROMPTS, runs, strict=True):
        prompt_ids = tokenizer(prompt).input_ids
        expected = oracle(prompt_ids + tokens)[len(prompt_ids) - 1 : -1]
        chosen = expected[torch.arange(len(tokens)), torch.tensor(tokens, device="cuda")]
        worst = max(worst, (chosen.cpu() - torch.tensor(logprobs)).abs().max().item())
        misses += int((expected.argmax(-1).cpu() != torch.tensor(tokens)).sum())
    return worst, misses


def test_rewrites_follow_fp32_oracle() -> None:
    baseline, _ = logprob_deviation(vanilla_tokens(), oracle_logprobs)
    laurel = Laurel(scale=0.05)
    workloads: dict[str, tuple[pllm.Workload, Callable[[list[int]], torch.Tensor]]] = {}
    # The same initial values, built with each workload's own initializer, drive the oracle.
    count, width = 24, 896
    fields = {
        "alpha": torch.empty(2 * count, device="cuda"),
        "beta": torch.empty(2 * count, device="cuda"),
        "a": torch.empty(2 * count, 4, width, device="cuda"),
        "b": torch.empty(2 * count, width, 4, device="cuda"),
    }
    laurel.initial(fields, torch.Generator())

    def laurel_add(x: torch.Tensor, y: torch.Tensor, s: int) -> torch.Tensor:
        return fields["alpha"][s] * y + fields["beta"][s] * x + (x @ fields["a"][s].T) @ fields["b"][s].T

    workloads["laurel"] = (
        laurel,
        lambda tokens: oracle_logprobs(
            tokens,
            attention=lambda x, y, layer: laurel_add(x, y, 2 * layer),
            mlp=lambda x, y, layer: laurel_add(x, y, 2 * layer + 1),
        ),
    )
    workloads["value"] = (
        ValueResidual(mixing=0.3),
        lambda tokens: oracle_logprobs(
            tokens, value=lambda v, values, layer: v if layer == 0 else v + 0.3 * (values[0] - v)
        ),
    )
    hc = StaticHyperConnections(perturb=0.05)
    routing = {
        "read": torch.empty(2 * count, 4, device="cuda"),
        "mix": torch.empty(2 * count, 4, 4, device="cuda"),
        "write": torch.empty(2 * count, 4, device="cuda"),
    }
    hc.initial(routing, torch.Generator())

    def hc_update(x: torch.Tensor, y: torch.Tensor, s: int) -> torch.Tensor:
        return torch.einsum("mn,rnd->rmd", routing["mix"][s], x) + routing["write"][s][:, None] * y[:, None]

    workloads["hyper-connections"] = (
        hc,
        lambda tokens: oracle_logprobs(
            tokens,
            attention=lambda x, y, layer: hc_update(x, y, 2 * layer),
            mlp=lambda x, y, layer: hc_update(x, y, 2 * layer + 1),
            read=lambda x, s: torch.einsum("n,rnd->rd", routing["read"][s], x),
            final=lambda x: x.mean(1),
            streams=4,
        ),
    )
    for name, (workload, oracle) in workloads.items():
        eager = generate_tokens(workload)
        captured = generate_tokens(workload, replace(CONFIG, cuda_graphs=True))
        assert captured == eager
        deviation, misses = logprob_deviation(captured, oracle)
        print(
            f"{name}: max logprob gap to FP32 {deviation:.4f} (plain bf16 vLLM {baseline:.4f}), {misses} non-argmax tokens"
        )
        assert deviation <= max(2 * baseline, BF16_NOISE) and misses <= 2
    assert math.isfinite(baseline)


def test_memory_sublayer_follows_fp32_chunk_oracle() -> None:
    baseline, _ = logprob_deviation(vanilla_tokens(), oracle_logprobs)
    for chunk in (1, 16):
        workload = FastWeightMemory(chunk=chunk)

        def memory(x: torch.Tensor, y: torch.Tensor, layer: int, workload: FastWeightMemory = workload) -> torch.Tensor:
            """Rows read the memory their chunk started with; M steps on |M k - y|^2 after each complete chunk."""
            if layer != workload.layer:
                return x + y
            keys, reads, state = (
                torch.nn.functional.normalize(x, dim=-1),
                [],
                torch.zeros(len(x[0]), len(x[0]), device=x.device),
            )
            for start in range(0, len(x), workload.chunk):
                k, v = keys[start : start + workload.chunk], y[start : start + workload.chunk]
                reads.append(k @ state.T)
                if len(k) == workload.chunk:
                    state = state - workload.lr / workload.chunk * (k @ state.T - v).T @ k
            return x + y + workload.gate * torch.cat(reads)

        eager = generate_tokens(workload)
        captured = generate_tokens(workload, replace(CONFIG, cuda_graphs=True))
        for (tokens, logprobs), (expected_tokens, expected_logprobs) in zip(captured, eager, strict=True):
            assert tokens == expected_tokens
            torch.testing.assert_close(torch.tensor(logprobs), torch.tensor(expected_logprobs), rtol=0, atol=BF16_NOISE)
        deviation, misses = logprob_deviation(captured, lambda tokens: oracle_logprobs(tokens, mlp=memory))
        print(
            f"memory chunk={chunk}: max logprob gap to FP32 {deviation:.4f} (plain bf16 vLLM {baseline:.4f}), {misses} non-argmax tokens"
        )
        assert deviation <= max(2 * baseline, BF16_NOISE) and misses <= 2


@pytest.mark.parametrize(
    "workload",
    (Laurel(), ValueResidual(mixing=0.3), StaticHyperConnections(perturb=0.05), FastWeightMemory()),
    ids=type,
)
def test_captured_rewrites_have_no_host_syncs(workload: pllm.Workload) -> None:
    result = check_on_gpu(workload, replace(CONFIG, cuda_graphs=True))
    assert "host syncs: eight steady decode steps" in result.passed


@pytest.mark.parametrize("chunk,shared", ((1, False), (4, False), (16, False), (4, True)))
def test_memory_capture_replays_mixed_spans_and_private_slots(chunk: int, shared: bool) -> None:
    """One graph follows the serial chunk recurrence while PTTS recycles private write slots."""
    from pllm.contracts.hidden_locator import HiddenPhase
    from pllm.contracts.request_identity import RequestIdentity
    from pllm.contracts.state_representation import StateDeviceKind
    from pllm.contracts.tensor_layout import TensorDType
    from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
    from pllm.execution.runtime_facts import model_timeline
    from pllm.execution.site_memory import SiteMemorySchedule
    from pllm.testing import backend_facts, compile_execution, step_input

    device = torch.device("cuda", 0)
    stream = torch.cuda.current_stream()
    workload = FastWeightMemory(layer=0, chunk=chunk, shared=shared)
    facts = backend_facts(
        8,
        timeline=model_timeline(device, ("transition/full_root@1",)),
        provider=PageBundleProviderFacts("testing.cuda@1", 32, StateDeviceKind.CUDA),
        max_lineages=3,
        max_rows=16,
        memory_bytes=1 << 24,
        dtype=TensorDType.BFLOAT16,
        layer_count=1,
    )
    execution = compile_execution(workload, facts, stream)
    identities = tuple(RequestIdentity(str(i), i // 2 if shared else i, 0) for i in range(3))
    execution.admit(identities, stream)
    (rewrite,) = execution.math.model_sites
    schedule = SiteMemorySchedule(rewrite, execution.arena, 3, 16)
    names = ("carry", "_pending_rows_x", "_pending_rows_y", "_pending_length")
    expected = [dict(zip(names, (pool[0].clone().zero_() for pool in schedule.pools), strict=True)) for _ in identities]
    if shared:
        expected[1] = expected[0]
    x, y = (torch.zeros(16, 8, device=device, dtype=torch.bfloat16) for _ in range(2))
    try:
        schedule(x, y, ())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = schedule(x, y, ())
        frontiers = [0, 0, 0]
        for step, counts in enumerate(((5, 1, 0), (1, 0, 7), (1, 1, 1), (0, 9, 1), (13, 1, 1))):
            packed = tuple(identity for identity, count in zip(identities, counts, strict=True) for _ in range(count))
            value = step_input(packed, step_id=step, sampler_order=())
            offsets = [0, 0, 0]
            rows = []
            for row in value.rows.rows:
                request = int(row.identity.request_id)
                rows.append(
                    replace(
                        row,
                        request_ordinal=request,
                        token_offset=frontiers[request] + offsets[request],
                        phase=HiddenPhase.PREFILL if counts[request] > 1 else HiddenPhase.DECODE,
                    )
                )
                offsets[request] += 1
            row_batch = type(value.rows).from_rows(value.rows.step_id, value.rows.total_rows, tuple(rows))
            value = replace(value, rows=row_batch, sampler_rows=replace(value.sampler_rows, row_batch=row_batch))
            context = execution.start_step(value, stream)
            execution.advance_to(context, "model.execute")
            schedule.prepare(value.rows, context)
            x.normal_()
            y.normal_()
            mode = torch.cuda.get_sync_debug_mode()
            try:
                torch.cuda.set_sync_debug_mode("error")
                graph.replay()
            finally:
                torch.cuda.set_sync_debug_mode(mode)
            result = x + y
            start = 0
            for request, count in enumerate(counts):
                if count:
                    result[start : start + count] = schedule.update(
                        **expected[request], x=x[start : start + count], y=y[start : start + count]
                    ).output
                    start += count
                frontiers[request] += count
            torch.testing.assert_close(
                actual[: sum(counts)],
                result[: sum(counts)],
                rtol=0,
                atol=0,
                msg=lambda message: f"chunk={chunk} step={step}: {message}",
            )
            execution.advance_to(context, "next.prepare")
            execution.finish_step(context, stream)
            for request, identity in enumerate(identities):
                (ref,) = execution.latest((identity,))
                root = execution.snapshot(ref, stream)
                for view, name in zip(schedule.views, names, strict=True):
                    torch.testing.assert_close(
                        view.read(root[view.field]), expected[request][name], rtol=1e-5, atol=1e-6
                    )
    finally:
        execution.close()


def _check_request_token_history(config: pllm.ModelConfig) -> None:
    from unittest.mock import patch

    from test.workloads.test_contracts import TokenHistory

    with pllm.make_llm(config, workload=TokenHistory()) as llm:
        session = llm.runner.require_session()
        prepare = session.prepare_input
        seen = []

        def record(rows):
            seen.extend(
                zip(rows.row_batch.rows, session.input_tokens[: rows.row_batch.total_rows].cpu().tolist(), strict=True)
            )
            buffer = session.input_tokens
            session.input_tokens = buffer.clone()
            prepare(rows)
            # Inspect private buffers on their producer lane, before a model consumer waits on its event.
            with torch.cuda.stream(session.deployment.execution.side_stream):
                for value in session.model_sites.result_values.values():
                    torch.testing.assert_close(
                        value[: rows.row_batch.total_rows], session.input_tokens[: rows.row_batch.total_rows]
                    )
            session.input_tokens = buffer

        with patch.object(session, "prepare_input", record):
            outputs = llm.generate(list(PROMPTS), SAMPLING)
        histories = [list(output.prompt_token_ids) + list(output.outputs[0].token_ids) for output in outputs]
        for row, ids in seen:
            tokens = histories[row.identity.prompt_index]
            assert ids == [tokens[row.token_offset - j] if row.token_offset >= j else -1 for j in range(3)]
        assert any(row.token_offset > 0 and row.phase.value == "prefill" for row, _ in seen)
        assert any(row.phase.value == "decode" for row, _ in seen)
        assert session.token_capture.graph is not None


def test_input_tokens_match_request_history() -> None:
    from test.workloads.test_contracts import TokenHistory

    config = replace(CONFIG, cuda_graphs=True, max_num_batched_tokens=8, enable_chunked_prefill=True)
    # Drop the inspected session and its bound callback before loading another engine.
    _check_request_token_history(config)
    assert (
        "host syncs: eight steady decode steps"
        in check_on_gpu(TokenHistory(), replace(config, max_num_batched_tokens=128)).passed
    )
