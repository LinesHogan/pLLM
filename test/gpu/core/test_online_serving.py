"""End-to-end GPU tests for continuous-batching `make_async_llm` and OpenAI `build_openai_app`."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
import pytest
import torch
from torch.nn import functional as F
from vllm import SamplingParams

import pllm
from pllm.backends.vllm.serving import AsyncVllmLlm, build_openai_app
from pllm.execution.state_execution import StateDiagnostics
from pllm.workloads.esamp import ESampConfig, ESampWorkload

pytestmark = pytest.mark.gpu

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass(frozen=True)
class RouteSessionWorkload(pllm.Workload):
    """Per-route learnable logit bias trained continuously across multi-turn HTTP requests."""

    session = pllm.RequestState({"bias": (1,)}, key_fields=("route",), optimizer=pllm.AdamW(lr=0.05))

    @pllm.add_logits
    @pllm.differentiate(parameters=True)
    def delta(
        self,
        logits: torch.Tensor = pllm.layer_logits((-1,)),
        bias: torch.Tensor = pllm.parameter("session", "bias"),
    ) -> torch.Tensor:
        return logits[:, 0] * 0.0 + bias.to(logits.dtype)

    @pllm.train(state="session", reduction="sum")
    @pllm.differentiate(inputs=("delta",))
    def loss(
        self,
        base: torch.Tensor = pllm.base_logits(),
        delta: torch.Tensor = pllm.result("delta"),
        tokens: torch.Tensor = pllm.sampled_tokens(),
    ) -> torch.Tensor:
        return F.cross_entropy((base + delta).float(), tokens, reduction="none")


async def _poll_until_drained(
    llm: AsyncVllmLlm,
    *,
    expect_active: int | None = 0,
    attempts: int = 40,
    delay_s: float = 0.05,
) -> StateDiagnostics:
    diag = await llm.diagnostics()
    for _ in range(attempts):
        drained = diag.inference_pins == 0 and diag.pending_state_releases == 0
        if expect_active is not None:
            drained = drained and diag.active_states == expect_active
        if drained:
            return diag
        await asyncio.sleep(delay_s)
        diag = await llm.diagnostics()
    return diag


def _online_config() -> pllm.ModelConfig:
    return pllm.ModelConfig(
        MODEL,
        gpu_memory_utilization=0.25,
        max_model_len=128,
        max_num_seqs=8,
        multiprocessing=True,
    )


def test_async_llm_concurrent_stream_cancel_and_drain() -> None:
    async def run() -> None:
        workload = ESampWorkload(ESampConfig(rank=4, intervention_beta=0.2))
        async with pllm.make_async_llm(_online_config(), workload=workload) as llm:
            assert (await llm.diagnostics()).eager_reason == ""

            first, second = await asyncio.gather(
                llm.generate("Explain KV paging briefly:", SamplingParams(max_tokens=8, temperature=0.8)),
                llm.generate("Count from one to five:", SamplingParams(max_tokens=8, temperature=0.8, n=2)),
            )
            assert len(first.outputs) == 1 and len(first.outputs[0].token_ids) > 0
            assert len(second.outputs) == 2

            stream = llm.stream("Write a long story about stars:", SamplingParams(max_tokens=48, temperature=0.8))
            try:
                await anext(stream)
                await anext(stream)
            finally:
                await stream.aclose()

            follow_up = await llm.generate("Hello:", SamplingParams(max_tokens=6, temperature=0.8))
            assert len(follow_up.outputs[0].token_ids) > 0

            diag = await _poll_until_drained(llm, expect_active=0)
            assert diag.active_states == 0
            assert diag.inference_pins == 0
            assert diag.pending_state_releases == 0
            assert diag.update_count > 0

    asyncio.run(run())


def test_openai_app_and_route_session_persistence() -> None:
    async def run() -> None:
        async with pllm.make_async_llm(_online_config(), workload=RouteSessionWorkload()) as llm:
            app, stack = await build_openai_app(llm, served_model_name="qwen-route")
            async with (
                stack,
                httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
            ):
                models_resp = await client.get("/v1/models")
                assert models_resp.status_code == 200
                assert any(item["id"] == "qwen-route" for item in models_resp.json()["data"])

                missing_route = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "qwen-route",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_completion_tokens": 4,
                    },
                )
                assert missing_route.status_code >= 400

                turn1 = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "qwen-route",
                        "messages": [{"role": "user", "content": "Hi there"}],
                        "max_completion_tokens": 6,
                        "stream": False,
                        "vllm_xargs": {"pllm_route": "session-a"},
                    },
                )
                assert turn1.status_code == 200
                assert turn1.json()["choices"][0]["message"]["content"]
                diag1 = await _poll_until_drained(llm, expect_active=None)
                assert diag1.inference_pins == 0
                assert diag1.active_states == 1
                assert diag1.update_count > 0

                turn2 = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "qwen-route",
                        "messages": [{"role": "user", "content": "Tell me more"}],
                        "max_completion_tokens": 6,
                        "stream": True,
                        "vllm_xargs": {"pllm_route": "session-a"},
                    },
                )
                assert turn2.status_code == 200
                assert "data: [DONE]" in turn2.text
                diag2 = await _poll_until_drained(llm, expect_active=None)
                assert diag2.inference_pins == 0
                assert diag2.active_states == 1
                assert diag2.update_count > diag1.update_count

                other_turn = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "qwen-route",
                        "messages": [{"role": "user", "content": "Separate user"}],
                        "max_completion_tokens": 6,
                        "stream": False,
                        "vllm_xargs": {"pllm_route": "session-b"},
                    },
                )
                assert other_turn.status_code == 200
                diag3 = await _poll_until_drained(llm, expect_active=None)
                assert diag3.inference_pins == 0
                assert diag3.active_states == 2

    asyncio.run(run())
