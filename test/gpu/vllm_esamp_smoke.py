"""One real MRv2 boundary proof for canonical ESamp request execution."""

from __future__ import annotations

import gc
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from vllm import SamplingParams

import pllm
from pllm.authoring import BackendCompileFacts, BackendRequirements
from pllm.backends.vllm.mrv1 import Mrv1Runner
from pllm.backends.vllm.runtime import VllmLlm, WorkloadModelRunner, probe_backend
from pllm.config import ExecutionOptions, ModelConfig, VllmDType
from pllm.contracts.state_ref import StateRef
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.workloads.esamp_request import (
    ESampRequestConfig,
    ESampRequestWorkload,
)


def checked_probe(
    requirements: BackendRequirements, runner: WorkloadModelRunner | Mrv1Runner, *, publish_lag: int = 1
) -> BackendCompileFacts:
    facts = probe_backend(requirements, runner, publish_lag=publish_lag)
    timeline = facts.timeline
    for index in range(len(timeline.anchors)):
        try:
            probe_backend(
                requirements,
                runner,
                replace(
                    timeline,
                    reserved_bytes=tuple(size - int(i == index) for i, size in enumerate(timeline.reserved_bytes)),
                ),
                publish_lag=publish_lag,
            )
        except ValueError as error:
            assert "input/sampler buffers" in str(error)
        else:
            raise AssertionError("worker accepted an underfunded input/sampler reservation")
    assert probe_backend(requirements, runner, timeline, publish_lag=publish_lag).timeline is timeline
    return facts


def lag_equivalence() -> None:
    """Training-only generation has identical tokens, counters and every final optimizer field."""
    states: list[tuple[torch.Tensor, ...]] = []
    retire = PageBundleArena.retire

    def snapshot(arena: PageBundleArena, refs: Sequence[StateRef]) -> None:
        stream = torch.cuda.current_stream(arena.device)
        pin = arena.pin_exact(refs, stream)
        states.extend(arena.snapshot(pin, ref) for ref in refs)
        arena.release_pin(pin)
        retire(arena, refs)

    expected: tuple[tuple[int, ...], int, int] | None = None
    expected_states: list[tuple[torch.Tensor, ...]] = []
    for lag in (0, 1):
        states.clear()
        config = ModelConfig(
            "Qwen/Qwen2.5-0.5B-Instruct",
            max_model_len=64,
            max_num_seqs=2,
            max_num_batched_tokens=64,
            gpu_memory_utilization=0.25,
            seed=7,
            cuda_graphs=True,
            execution=ExecutionOptions(capture=True, publish_lag=lag, capture_memory_bytes=16 * 1024**2),
        )
        with pllm.make_llm(config, workload=ESampRequestWorkload(ESampRequestConfig(rank=2))) as llm:
            with patch.object(PageBundleArena, "retire", snapshot):
                outputs = llm.generate(
                    ["The sky is", "The sea is"],
                    SamplingParams(max_tokens=16, temperature=0.8, seed=7, ignore_eos=True),
                )
            diagnostics = llm.diagnostics()
            observed = (
                tuple(token for output in outputs for token in output.outputs[0].token_ids),
                diagnostics.loss_count,
                diagnostics.update_count,
            )
            if expected is None:
                expected, expected_states = observed, list(states)
            else:
                assert observed == expected
                assert len(states) == len(expected_states) == 2
                for actual, wanted in zip(states, expected_states, strict=True):
                    for field, reference in zip(actual, wanted, strict=True):
                        torch.testing.assert_close(field, reference, rtol=0, atol=0)
        gc.collect()
        torch.cuda.empty_cache()
    print("publish_lag=0/1: identical beta=0 tokens, loss/update counts and all final state fields")


def main() -> int:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    plain = pllm.make_llm(
        ModelConfig(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            dtype=VllmDType.BFLOAT16,
            gpu_memory_utilization=0.25,
            max_model_len=64,
            enable_prefix_caching=False,
            seed=7,
            max_num_seqs=1,
            max_num_batched_tokens=64,
        )
    )
    try:
        outputs = plain.generate(
            ["The sky is"],
            SamplingParams(max_tokens=3, temperature=0.0),
            use_tqdm=False,
        )
        if not outputs[0].outputs[0].text:
            raise RuntimeError("plain public inference produced no text")
    finally:
        plain.llm_engine.engine_core.shutdown()
        del plain
        gc.collect()
        torch.cuda.empty_cache()
    workload = ESampRequestWorkload(
        ESampRequestConfig(
            rank=2,
            source_layer=0,
            target_layer=-1,
            intervention_beta=0.2,
        )
    )
    with patch("pllm.backends.vllm.runtime.probe_backend", side_effect=checked_probe):
        routed = pllm.make_llm(
            workload=workload,
            config=ModelConfig(
                model="Qwen/Qwen2.5-0.5B-Instruct",
                dtype=VllmDType.BFLOAT16,
                max_model_len=64,
                max_num_seqs=1,
                max_num_batched_tokens=64,
                gpu_memory_utilization=0.4,
                seed=7,
                execution=ExecutionOptions(capture=True, capture_memory_bytes=16 * 1024**2),
            ),
        )
    match routed:
        case VllmLlm() as handle:
            pass
        case _:
            raise RuntimeError("public make_llm did not select canonical runtime")
    try:
        plan = handle.attachment.inputs.plan
        probe = handle.attachment.inputs.probe
        owned_hidden_bytes = 2 * len(probe.hidden_resources) * probe.max_packed_rows * probe.hidden_size * 4
        for anchor, size in zip(plan.timeline.anchors, plan.timeline.reserved_bytes, strict=True):
            assert (
                size > owned_hidden_bytes
                if anchor.symbol in ("sampler.pre_draw", "sampler.complete")
                else size == owned_hidden_bytes
            )
        assert plan.schedule_decision.peak_memory_bytes >= plan.state_physical.reserved_bytes + owned_hidden_bytes
        outputs = handle.generate(["The sky is"], SamplingParams(max_tokens=3, temperature=0.0, n=1, ignore_eos=True))
        execution = handle.deployment.execution
        assert sum("cuda_graph" in task.implementation_id for task in execution.plan.tasks) == 3
        diagnostics = execution.diagnostics()
        if (
            not outputs[0].outputs[0].text
            or diagnostics.loss_count != 2
            or diagnostics.update_count != 2
            or diagnostics.completed_sampler_pins != 2
            or diagnostics.active_lineages != 0
            or diagnostics.pending_state_releases != 0
            or execution.plan.fallback_count != 0
        ):
            raise RuntimeError("canonical ESamp lifecycle evidence is incomplete")
        print(
            {
                "text": outputs[0].outputs[0].text,
                "runner": handle.runner.runner_kind.value,
                "loss_count": diagnostics.loss_count,
                "update_count": diagnostics.update_count,
                "active_lineages": diagnostics.active_lineages,
                "fallback_count": execution.plan.fallback_count,
                "attachment": (handle.deployment.attachment.physical.bundle_signature[:12]),
            }
        )
    finally:
        handle.close()
    del handle, routed
    gc.collect()
    torch.cuda.empty_cache()
    lag_equivalence()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
