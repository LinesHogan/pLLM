"""ESamp on a real vLLM MRv2 model: plain inference, captured ESamp, and publish-lag equivalence.

Training-only ESamp (beta 0) must give identical tokens, counters and final optimizer
state with publish_lag 0, 1 and 2.
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from vllm import SamplingParams

import pllm
from pllm.backends.vllm.runtime import VllmLlm
from pllm.config import ExecutionOptions, ModelConfig
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.state_refs import StateRefs
from pllm.workloads.esamp import ESampConfig, ESampWorkload


def lag_equivalence() -> None:
    """Training-only generation has identical tokens, counters and every final optimizer field."""
    states: list[tuple[torch.Tensor, ...]] = []
    retire = PageBundleArena.retire

    def snapshot(arena: PageBundleArena, refs: StateRefs) -> None:
        stream = torch.cuda.current_stream(arena.device)
        pin = arena.pin_exact(refs, stream)
        states.extend(arena.snapshot(pin, ref) for ref in refs)
        arena.release_pin(pin)
        retire(arena, refs)

    expected: tuple[tuple[int, ...], int, int] | None = None
    expected_states: list[tuple[torch.Tensor, ...]] = []
    for lag in (0, 1, 2):
        states.clear()
        config = ModelConfig(
            "Qwen/Qwen2.5-0.5B-Instruct",
            max_model_len=64,
            max_num_seqs=2,
            max_num_batched_tokens=64,
            gpu_memory_utilization=0.25,
            seed=7,
            cuda_graphs=True,
            execution=ExecutionOptions(publish_lag=lag, capture_memory_bytes=16 * 1024**2),
        )
        with pllm.make_llm(config, workload=ESampWorkload(ESampConfig(rank=2))) as llm:
            assert any(task.lane == "side" for task in llm.attachment.plan.schedule.tasks) == bool(lag)
            # Requests retire inside generate; read each final version just before it is retired.
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
    print("publish_lag=0/1/2: identical beta=0 tokens, loss/update counts and all final state fields")


def main() -> int:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    plain = pllm.make_llm(
        ModelConfig(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            dtype="bfloat16",
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
        plain.close()
        del plain
        gc.collect()
        torch.cuda.empty_cache()
    workload = ESampWorkload(ESampConfig(rank=2, source_layer=0, target_layer=-1, intervention_beta=0.2))
    routed = pllm.make_llm(
        workload=workload,
        config=ModelConfig(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            dtype="bfloat16",
            max_model_len=64,
            max_num_seqs=1,
            max_num_batched_tokens=64,
            gpu_memory_utilization=0.4,
            seed=7,
            execution=ExecutionOptions(capture_memory_bytes=16 * 1024**2),
        ),
    )
    match routed:
        case VllmLlm() as handle:
            pass
        case _:
            raise RuntimeError("public make_llm did not select canonical runtime")
    try:
        outputs = handle.generate(["The sky is"], SamplingParams(max_tokens=3, temperature=0.0, n=1, ignore_eos=True))
        plan = handle.attachment.plan.schedule
        assert sum("cuda_graph" in task.implementation_id for task in plan.tasks) == 3
        diagnostics = handle.diagnostics()
        if (
            not outputs[0].outputs[0].text
            or diagnostics.loss_count != 2
            or diagnostics.update_count != 2
            or diagnostics.completed_sampler_pins != 2
            or diagnostics.active_states != 0
            or diagnostics.pending_state_releases != 0
        ):
            raise RuntimeError("canonical ESamp lifecycle evidence is incomplete")
        print(f"captured ESamp: text={outputs[0].outputs[0].text!r} loss/update={diagnostics.loss_count}")
    finally:
        handle.close()
    del handle, routed
    gc.collect()
    torch.cuda.empty_cache()
    lag_equivalence()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
