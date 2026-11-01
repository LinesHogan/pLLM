"""J1/J2 on two GPUs: TP token parity, transport, admission and replicated state determinism.

Run explicitly with pytest; all new multi-GPU evidence lives in this file. Worker
callbacks observe final state through vLLM's existing callable collective_rpc.
"""

from dataclasses import replace
from hashlib import sha256

import pytest
import torch
from vllm import RequestOutput, SamplingParams

import pllm
from pllm.backends.vllm import compat
from pllm.config import ExecutionOptions, ModelConfig
from pllm.contracts.feedback import RetryAction
from pllm.execution.ptts.state_refs import StateRefs
from pllm.workloads.dummy import DummyConfig, DummyWorkload
from pllm.workloads.esamp import ESampConfig, ESampWorkload
from pllm.workloads.prefill_adapters import PrefillAdaptersWorkload
from pllm.workloads.tlm import TLMConfig, TLMWorkload
from pllm.workloads.ttrl import TTRLConfig, TTRLWorkload, ttrl_rewards
from pllm.workloads.video import VideoWorkload

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2 or compat.VLLM_027, reason="needs two GPUs and vLLM 0.32")
PROMPTS = [
    "The sky is",
    "Water freezes at",
    "The sea is",
    "One plus two is",
    "Paris is",
    "A tree has",
    "The moon is",
    "Hello",
]
_FINAL = []
_EXPORTS = []


class ExportingDummy(DummyWorkload):
    def on_export(self, event):
        from test.gpu.vllm_tp_smoke import _EXPORTS

        _EXPORTS.append(event.invocation_id)


def _exports(worker):
    from test.gpu.vllm_tp_smoke import _EXPORTS

    worker.workload_runner.require_session().trace.drain()
    return _EXPORTS


def config(graphs=True, lag=1):
    return ModelConfig(
        "Qwen/Qwen2.5-0.5B-Instruct",
        tensor_parallel_size=2,
        max_model_len=128,
        max_num_seqs=8,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.35,
        seed=7,
        cuda_graphs=graphs,
        execution=ExecutionOptions(publish_lag=lag, capture_memory_bytes=32 << 20),
    )


def _digest(arena, refs):
    pin = arena.pin_exact(refs, torch.cuda.current_stream(arena.device))
    try:
        return [
            tuple(sha256(field.cpu().numpy().tobytes()).hexdigest() for field in arena.snapshot(pin, ref))
            for ref in refs
        ]
    finally:
        arena.wait_release(arena.release_pin(pin))


def _record_retirement(worker):
    from test.gpu.vllm_tp_smoke import _FINAL, _digest

    _FINAL.clear()
    arena = worker.workload_runner.require_session()._deployment.execution.arena
    retire = arena.retire

    def record(refs):
        _FINAL.extend(_digest(arena, refs))
        return retire(refs)

    arena.retire = record


def _final(worker):
    from test.gpu.vllm_tp_smoke import _FINAL

    return _FINAL


def _head(worker):
    execution = worker.workload_runner.require_session()._deployment.execution
    refs = execution.directory.latest_refs(())
    return _digest(execution.arena, StateRefs.of(refs))


def tokens(outputs):
    return tuple(tuple(output.outputs[0].token_ids) for output in outputs)


@pytest.mark.parametrize("graphs", [False, True])
def test_j1_tokens(graphs):
    settings = config(graphs)
    sampling = [SamplingParams(max_tokens=64, temperature=t, seed=7, ignore_eos=True) for t in (0.0, 0.8)]
    plain = pllm.make_llm(replace(settings, execution=ExecutionOptions()))
    try:
        expected = []
        for params in sampling:
            # Match pllm's all-prompts admission barrier; MP EngineCore otherwise runs during enqueue.
            plain.sleep(level=0, mode="keep")
            plain.enqueue(PROMPTS, params, use_tqdm=False)
            plain.wake_up(tags=["scheduling"])
            expected.append(tokens(plain.wait_for_completion(RequestOutput, use_tqdm=False)))
    finally:
        plain.llm_engine.engine_core.shutdown()
    for workload in (
        ExportingDummy(DummyConfig(export_every_n_steps=1)),
        ESampWorkload(ESampConfig(rank=4)),
    ):
        with pllm.make_llm(settings, workload=workload) as handle:
            actual = [tokens(handle.generate(PROMPTS, params)) for params in sampling]
            differences = [
                (arm, row, next((i, a, b) for i, (a, b) in enumerate(zip(got, want)) if a != b))
                for arm, (got_arm, want_arm) in enumerate(zip(actual, expected))
                for row, (got, want) in enumerate(zip(got_arm, want_arm))
                if got != want
            ]
            assert not differences, (workload.name, differences)
            counters = handle.diagnostics()
            assert counters.active_states == counters.inference_pins == 0
            assert len(handle.trace_snapshots()) == 2
            if isinstance(workload, ExportingDummy):
                exports = handle._require_open().collective_rpc(_exports)
                assert len(exports[0]) == len(set(exports[0])) == 126  # Two runs of 63 decode steps.
                assert exports[1] == []
            print(
                f"J1 graphs={graphs} workload={workload.name}: greedy/seeded 8x64 bitwise parity; {counters}",
                flush=True,
            )


def test_j1_admission():
    for name in ("pipeline_parallel_size", "data_parallel_size"):
        with pytest.raises(ValueError, match=name + "=1"):
            pllm.make_llm(replace(config(), **{name: 2}), workload=DummyWorkload())
    for workload, message in (
        (PrefillAdaptersWorkload(), "parameter overlays"),
        (TLMWorkload(TLMConfig()), "causal_model"),
        (VideoWorkload(), "state feeds"),
    ):
        with pytest.raises(ValueError, match=message + ".*tensor_parallel_size=1"):
            pllm.make_llm(config(), workload=workload)
    with pytest.raises(ValueError, match='runner="mrv2"'):
        pllm.make_llm(replace(config(), runner="mrv1"), workload=DummyWorkload())


def test_j1_ttrl(monkeypatch):
    from pllm.authoring import WeightedRowBatch

    # Only the test digest callback crosses EngineCore via pickle; production commands remain JSON.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    settings = replace(config(), multiprocessing=True, execution=ExecutionOptions(max_training_rows=8, publish_lag=0))
    with pllm.make_llm(settings, workload=TTRLWorkload(TTRLConfig(rank=4))) as handle:

        def feedback(completed):
            reward = ttrl_rewards(completed, TTRLConfig(rank=4))
            # Known targets ensure the transport/training check does not depend on boxed-answer quality.
            batch = WeightedRowBatch(completed.hidden, torch.ones(8), torch.ones(8))
            handle.generate(PROMPTS, SamplingParams(max_tokens=2, temperature=0), batch=batch)
            return replace(
                reward,
                retry_controls=tuple(
                    replace(control, action=RetryAction.RETRY, retry_after_update=False)
                    for control in reward.retry_controls
                ),
            )

        rounds = list(
            handle.generate_feedback(
                PROMPTS,
                SamplingParams(max_tokens=4, temperature=0),
                feedback=feedback,
                feedback_group="tp",
                max_rounds=2,
            )
        )
        assert len(rounds) == 2
        assert all(torch.isfinite(batch.hidden).all() and batch.hidden.abs().sum() > 0 for batch, _ in rounds)
        assert handle.diagnostics().update_count == 2
        digests = handle._require_open().collective_rpc(_head)
        assert digests[0] == digests[1]
        print(f"J1 TTRL: two feedback rounds, two updates, equal head fields {digests[0]}", flush=True)


@pytest.mark.parametrize("lag", [0, 1])
def test_j2(lag):
    expected = None
    for run in range(2):
        with pllm.make_llm(
            config(lag=lag), workload=ESampWorkload(ESampConfig(rank=4, intervention_beta=0.2))
        ) as handle:
            llm = handle._require_open()
            llm.collective_rpc(_record_retirement)
            output = handle.generate(PROMPTS, SamplingParams(max_tokens=64, temperature=0.8, seed=7, ignore_eos=True))
            counters = handle.diagnostics()
            digests = llm.collective_rpc(_final)
            assert len(digests[0]) == 8 and all(len(fields) > 1 for fields in digests[0])
            assert digests[0] == digests[1], ("rank divergence", lag, run, digests)
            observed = (tokens(output), digests[0], counters.loss_count, counters.update_count)
            if expected is not None:
                assert observed == expected, ("run divergence", lag, observed, expected)
            expected = observed
            print(
                f"J2 lag={lag} run={run}: rank-equal final fields {digests[0]}; updates={counters.update_count}",
                flush=True,
            )
