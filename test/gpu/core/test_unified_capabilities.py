"""End-to-end GPU tests for unified sync/async controls, CUDA-graph adapters, feeds, plain completions and MLA."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM
from vllm import SamplingParams
from vllm.outputs import RequestOutput

import pllm

pytestmark = pytest.mark.gpu

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass(frozen=True)
class _BatchControlWorkload(pllm.Workload):
    state = pllm.InstanceState({"w": (4,)}, optimizer=pllm.AdamW(lr=1e-2))

    @pllm.add_logits
    def bias(
        self,
        logits: torch.Tensor = pllm.layer_logits((-1,)),
        w: torch.Tensor = pllm.parameter("state", "w"),
    ) -> torch.Tensor:
        return logits[:, 0] * 0.0 + w[:1].to(logits.dtype)

    @pllm.train(state="state")
    def train_rows(
        self,
        hidden: torch.Tensor = pllm.batch("hidden"),
        targets: torch.Tensor = pllm.batch("targets"),
        weights: torch.Tensor = pllm.batch("weights"),
        w: torch.Tensor = pllm.parameter("state", "w"),
    ) -> torch.Tensor:
        pred = (hidden[:, :4].float() * w).sum(dim=-1)
        return ((pred - targets.float()).square() * weights.float()).mean()


@dataclass(frozen=True)
class _RouteImportWorkload(pllm.Workload):
    bias = pllm.RequestState({"v": (1,)}, key_fields=("route",), init=pllm.Imported())

    @pllm.add_logits
    def delta(
        self,
        logits: torch.Tensor = pllm.layer_logits((-1,)),
        v: torch.Tensor = pllm.parameter("bias", "v"),
    ) -> torch.Tensor:
        return logits[:, 0] * 0.0 + v.to(logits.dtype)


@dataclass(frozen=True)
class _FrozenGraphAdapter(pllm.Workload):
    adapter = pllm.InstanceState(
        {
            "a": pllm.Field(lambda inputs, outputs: (2, inputs)),
            "b": pllm.Field(lambda inputs, outputs: (outputs, 2)),
        },
        init=pllm.Computed(lambda f, g: (f["a_0"].normal_(0, 0.01, generator=g), f["b_0"].zero_())),
    )

    @property
    def down(self) -> pllm.model.Adapter:
        return pllm.model.Adapter("adapter", (pllm.model.Projection("down_proj", 0),), operator="low_rank")


@dataclass(frozen=True)
class _FedPatchWorkload(pllm.Workload):
    fed = pllm.InstanceState({"v": (1,)}, dtype=torch.float32, init=pllm.Fed("latest"))
    inject = pllm.InstanceState({"delta": pllm.Field(lambda inputs, outputs: (inputs,))})

    @property
    def patch(self) -> pllm.model.ActivationPatch:
        return pllm.model.ActivationPatch(
            "inject",
            (0,),
            first_tokens=1,
            last_tokens=0,
            operator="add",
            decode=True,
            refresh="step",
            on_missing="hold",
        )

    @pllm.write(state="inject", fields=("delta",))
    def copy_fed(
        self,
        v: torch.Tensor = pllm.parameter("fed", "v"),
        template: torch.Tensor = pllm.parameter("inject", "delta"),
    ) -> tuple[torch.Tensor, ...]:
        return (template * 0.0 + v.to(template.dtype),)


@dataclass(frozen=True)
class _MLATopKWorkload(pllm.Workload):
    @pllm.custom_attention(layers=(0,))
    def compress_select(self, q: torch.Tensor, kv: pllm.execution.attention.PagedKV) -> torch.Tensor:
        full_out, lse = kv.attend(q)
        positions = torch.arange(2, device=q.device)[None, :].expand(len(q), -1)
        selected_out, selected_lse = kv.attend(q, topk=positions)
        return full_out + (selected_out - selected_out.detach()) + (lse + selected_lse).sum() * 0.0

    @pllm.export(max_rows=4, max_cols=4)
    def export_lse(self, lse: torch.Tensor = pllm.attention(0, "lse")) -> torch.Tensor:
        return lse


def _tiny_mla_checkpoint(root: Path) -> Path:
    vocab = {"<unk>": 0, "<bos>": 1, "<eos>": 2, **{f"tok{i}": i for i in range(3, 64)}}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": "<bos>", "eos_token": "<eos>", "unk_token": "<unk>"})
    )
    cfg = DeepseekV2Config(
        architectures=["DeepseekV2ForCausalLM"],
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        q_lora_rank=None,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=2,
        vocab_size=64,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
    )
    cfg.save_pretrained(root)
    torch.manual_seed(0)
    model = DeepseekV2ForCausalLM(cfg).to(torch.bfloat16)
    save_file(dict(model.state_dict()), str(root / "model.safetensors"))
    return root


def _base_config(
    *,
    enable_prefix_caching: bool = False,
    cuda_graphs: Literal["auto"] | bool = "auto",
    execution: pllm.ExecutionOptions | None = None,
) -> pllm.ModelConfig:
    return pllm.ModelConfig(
        MODEL,
        gpu_memory_utilization=0.25,
        max_model_len=192,
        max_num_seqs=4,
        enable_prefix_caching=enable_prefix_caching,
        cuda_graphs=cuda_graphs,
        execution=execution or pllm.ExecutionOptions(),
    )


def test_async_live_train_batch_completions_and_route_import() -> None:
    async def run() -> None:
        config = _base_config(execution=pllm.ExecutionOptions(max_host_training_rows=8, publish_lag=0))
        async with pllm.make_async_llm(config, workload=_BatchControlWorkload()) as llm:
            params = SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True)
            stream_task = asyncio.create_task(llm.generate("Explain paging in two sentences:", params))
            await asyncio.sleep(0)
            batch = pllm.host.WeightedRowBatch(
                torch.ones(4, 896, dtype=torch.bfloat16),
                torch.ones(4, dtype=torch.bfloat16),
                torch.ones(4, dtype=torch.bfloat16),
            )
            step = await llm.train_batch(batch)
            assert isinstance(step, pllm.host.TrainStep)
            assert step.loss is not None and step.loss.isfinite()
            assert "w" in step.parameters
            out = await stream_task
            assert len(out.outputs[0].token_ids) == 16
            completions = await llm.generate_completions(
                ["Hello world", "Next prompt"],
                SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True),
                group_id="g1",
            )
            assert len(completions.completions.entries) == 2
            assert completions.hidden.shape == (2, 896)

        async with pllm.make_async_llm(_base_config(), workload=_RouteImportWorkload()) as route_llm:
            pre = pllm.io.StateImport((torch.tensor([[0.5]], dtype=torch.float32),))
            await route_llm.import_state(pre, route="tenant-a")
            turn1 = await route_llm.generate(
                "Count one two",
                SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True),
                route="tenant-a",
            )
            assert len(turn1.outputs[0].token_ids) == 4
            per_req = pllm.io.StateImport((torch.tensor([[1.0]], dtype=torch.float32),))
            turn2 = await route_llm.generate(
                "Count one two",
                SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True),
                route="tenant-b",
                state_import=per_req,
            )
            assert len(turn2.outputs[0].token_ids) == 4

    asyncio.run(run())


def test_sync_cuda_graph_adapter_prefix_cache_and_plain_completions() -> None:
    config = _base_config(cuda_graphs=True, enable_prefix_caching=True)
    report = pllm.explain(_FrozenGraphAdapter(), config)
    assert not report.eager_reason
    assert next(member for member in report.members if member.effect == "Adapter").eager_reason is None
    with pllm.make_llm(config, workload=_FrozenGraphAdapter()) as handle:
        assert handle.diagnostics().eager_reason == ""
        out = handle.generate(
            ["Prefix caching and graphs work cleanly."], SamplingParams(max_tokens=6, temperature=0.0)
        )
        assert len(out[0].outputs[0].token_ids) > 0

    plain_config = _base_config(cuda_graphs=True)
    with pllm.make_llm(plain_config) as plain:
        completions = plain.generate_completions(
            ["Hello from plain engine"],
            SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True),
            group_id="plain",
        )
        assert len(completions.completions.entries) == 1
        assert completions.hidden.shape == (1, 896)
        raw = plain.require_llm()
        raw.enqueue(["Raw enqueue still works"], SamplingParams(max_tokens=4, temperature=0.0), use_tqdm=False)
        outputs = raw.wait_for_completion(RequestOutput, use_tqdm=False)
        assert len(outputs) == 1 and len(outputs[0].outputs[0].token_ids) > 0

    async def run_async_plain() -> None:
        async with pllm.make_async_llm(plain_config) as async_plain:
            async_comp = await async_plain.generate_completions(
                ["Async plain completion"],
                SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True),
                group_id="async-plain",
            )
            assert len(async_comp.completions.entries) == 1
            assert async_comp.hidden.shape == (1, 896)

    asyncio.run(run_async_plain())


def test_async_multiprocess_feed() -> None:
    async def run() -> None:
        async with pllm.make_async_llm(_base_config(), workload=_FedPatchWorkload()) as llm:
            with await llm.feed("fed") as feed:
                with feed.publish() as fields:
                    fields["v"].fill_(0.25)
                feed.end()
                out = await llm.generate(
                    "One two three",
                    SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True),
                )
                assert len(out.outputs[0].token_ids) == 4
            diag = await llm.diagnostics()
            for _ in range(40):
                if diag.inference_pins == 0 and diag.pending_state_releases == 0:
                    break
                await asyncio.sleep(0.05)
                diag = await llm.diagnostics()
            assert diag.inference_pins == 0
            assert diag.pending_state_releases == 0

    asyncio.run(run())


def test_gpu_mla_custom_attention(tmp_path: Path) -> None:
    checkpoint = _tiny_mla_checkpoint(tmp_path)
    config = pllm.ModelConfig(
        str(checkpoint),
        gpu_memory_utilization=0.25,
        max_model_len=64,
        max_num_seqs=4,
    )
    workload = _MLATopKWorkload()
    report = pllm.explain(workload, config)
    assert report.model.kv_lora_rank == 512
    with pllm.make_llm(config, workload=workload) as handle:
        outputs = handle.generate(
            ["tok3 tok4 tok5 tok6", "tok7 tok8 tok9 tok10"],
            SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True),
        )
        assert all(len(item.outputs[0].token_ids) == 4 for item in outputs)
