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
from pllm.workloads.ttrl import TTRLConfig, TTRLWorkload, ttrl_vote
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
    plain_handle = pllm.make_llm(replace(settings, execution=ExecutionOptions()))
    plain = plain_handle._require_open()
    try:
        expected = []
        for params in sampling:
            # Match pllm's all-prompts admission barrier; MP EngineCore otherwise runs during enqueue.
            plain.sleep(level=0, mode="keep")
            plain.enqueue(PROMPTS, params, use_tqdm=False)
            plain.wake_up(tags=["scheduling"])
            expected.append(tokens(plain.wait_for_completion(RequestOutput, use_tqdm=False)))
    finally:
        plain_handle.close()
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
            assert not differences, (type(workload).__name__, differences)
            counters = handle.diagnostics()
            assert counters.active_states == counters.inference_pins == 0
            assert len(handle.trace_snapshots()) == 2
            if isinstance(workload, ExportingDummy):
                exports = handle._require_open().collective_rpc(_exports)
                assert len(exports[0]) == len(set(exports[0])) == 126  # Two runs of 63 decode steps.
                assert exports[1] == []
            print(
                f"J1 graphs={graphs} workload={type(workload).__name__}: greedy/seeded 8x64 bitwise parity; {counters}",
                flush=True,
            )


def test_j1_admission():
    for name in ("pipeline_parallel_size", "data_parallel_size"):
        with pytest.raises(ValueError, match=name + "=1"):
            pllm.make_llm(replace(config(), **{name: 2}), workload=DummyWorkload())
    for workload, message, milestone in (
        (TLMWorkload(TLMConfig()), "causal_model", "M5"),
        (VideoWorkload(), "state feeds", "E1"),
    ):
        with pytest.raises(ValueError, match=message + ".*tensor_parallel_size=1.*" + milestone):
            pllm.make_llm(config(), workload=workload)
    with pytest.raises(ValueError, match="not shard-local.*other ranks"):
        pllm.explain(PrefillAdaptersWorkload(), config())
    with pytest.raises(ValueError, match='runner="mrv2"'):
        pllm.make_llm(replace(config(), runner="mrv1"), workload=DummyWorkload())


def test_j1_ttrl(monkeypatch):
    from pllm.authoring import WeightedRowBatch

    # Only the test digest callback crosses EngineCore via pickle; production commands remain JSON.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    settings = replace(config(), multiprocessing=True, execution=ExecutionOptions(max_training_rows=8, publish_lag=0))
    with pllm.make_llm(settings, workload=TTRLWorkload(TTRLConfig(rank=4))) as handle:

        def feedback(completed):
            reward = ttrl_vote(completed, TTRLConfig(rank=4))
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
                group_id="tp",
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


class Steering(pllm.Workload):
    """Activation steering (CAA): add an imported vector to layer 12's MLP output on decode rows."""

    steer: pllm.InstanceState = pllm.InstanceState({"vector": pllm.Field(lambda width, _: (width,))}, imported=True)
    patch: pllm.model.ActivationPatch = pllm.model.ActivationPatch("steer", layers=(12,), operator="add", decode=True)


class Observe(pllm.Workload):
    @pllm.export(max_cols=1)
    def mean(self, h: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
        return h.mean(-1, keepdim=True)


def _greedy(settings, workload, batch=None):
    sampling = SamplingParams(max_tokens=24, temperature=0.0, ignore_eos=True, logprobs=0)
    with pllm.make_llm(settings, workload=workload) as llm:
        outputs = llm.generate(PROMPTS, sampling, batch=batch, use_tqdm=False)
    return [
        (
            tuple(o.outputs[0].token_ids),
            tuple(s[t].logprob for t, s in zip(o.outputs[0].token_ids, o.outputs[0].logprobs)),
        )
        for o in outputs
    ]


def test_activation_patch_steers_replicated_streams():
    """ActivationPatch runs in ModelSites' norm hooks after the MLP all-reduce: a zero vector is the model bitwise, and
    a steering vector moves TP=2 exactly as far as TP=1 (an edit applied per rank before the reduce would double)."""
    settings = config(graphs=False)
    vector = torch.randn(896, generator=torch.Generator().manual_seed(3)) * 4
    assert _greedy(settings, Steering(), pllm.io.StateImport((torch.zeros(896),))) == _greedy(settings, Observe())
    steered = _greedy(settings, Steering(), pllm.io.StateImport((vector,)))
    single = _greedy(replace(settings, tensor_parallel_size=1), Steering(), pllm.io.StateImport((vector,)))
    assert steered != _greedy(settings, Observe())
    agreeing, gap = 0, 0.0
    for (tokens, logprobs), (reference, expected) in zip(steered, single, strict=True):
        for token, other, value, target in zip(tokens, reference, logprobs, expected):
            if token != other:
                break
            agreeing, gap = agreeing + 1, max(gap, abs(value - target))
    print(f"TP=2 steering vs TP=1: {agreeing}/{24 * len(PROMPTS)} leading tokens agree, logprob gap {gap:.4f}")
    # TP=1 and TP=2 reduce in different orders (bf16 logprobs differ by ~0.2); a doubled edit diverges the tokens.
    assert agreeing >= 0.75 * 24 * len(PROMPTS) and gap < 0.5


class EarlyExit(pllm.Workload):
    """Exercise empty, full and compact active slices with replicated hidden decisions."""

    @pllm.custom_forward(tokens="decode")
    def forward(self, layers):
        middle = layers.layer_count // 2
        hidden = layers(stop=middle)[middle - 1]
        active = (hidden.float().square().mean(-1, keepdim=True) > 0) & (layers.position % 4 < 2)
        return layers(middle, hidden=hidden, active=active)[-1]

    @pllm.export(max_cols=1)
    def readout(self, hidden=pllm.residual(22)):
        return hidden.float().mean(-1, keepdim=True)


_ROUTE = []
_SAMPLES = []


def _record_routes(worker, forced=()):
    from vllm.forward_context import get_forward_context

    from test.gpu.vllm_tp_smoke import _ROUTE, _SAMPLES

    _ROUTE.clear()
    _SAMPLES.clear()
    runner = worker.workload_runner
    primitive = runner.require_session()._custom_forward
    forward = primitive.model.forward
    sample = runner.sampler._sample_random

    def record(*args, **kwargs):
        output = forward(*args, **kwargs)
        context = get_forward_context()
        pages = tuple(
            layer.kv_cache.view(-1, output.shape[-1])[
                context.slot_mapping[layer.layer_name] // primitive.allocation
            ].cpu()
            for layer in primitive.position_layers
        )
        _ROUTE.append((output.cpu(), primitive.observations.cpu(), pages))
        return output

    def sampled(*args, **kwargs):
        result = sample(*args, **kwargs)
        natural, logits = result
        target = forced[len(_SAMPLES)].to(natural.device) if forced else natural
        chosen = logits.gather(1, target.reshape(-1, 1)).flatten()
        regret = logits.max(-1).values - chosen
        _SAMPLES.append((natural.cpu(), chosen.cpu(), regret.cpu()))
        _ROUTE.append(natural.cpu())
        # A shared token history distinguishes TP rounding from an autoregressive cascade.
        natural.copy_(target)
        return result

    primitive.model.forward = record
    runner.sampler._sample_random = sampled


def _route_digest(worker):
    from test.gpu.vllm_tp_smoke import _ROUTE

    def digest(value):
        if isinstance(value, torch.Tensor):
            return sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
        return tuple(digest(item) for item in value)

    return digest(_ROUTE)


def _route_samples(worker):
    from test.gpu.vllm_tp_smoke import _SAMPLES

    return _SAMPLES


def _native_steps(worker, enabled):
    from test.gpu.native_replay import NativeSteps

    if enabled:
        NativeSteps.begin_native_steps(worker)
    else:
        NativeSteps.end_native_steps(worker)


_ROUTE_PROMPTS = ["Continue the alphabet: A, B, C,", "One plus one is", "Water is"]


def _route_generation(settings, workload, forced, native):
    with pllm.make_llm(settings, workload=workload) as handle:
        llm = handle._require_open()
        if native:
            llm.collective_rpc(_native_steps, args=(True,))
        try:
            llm.collective_rpc(_record_routes, args=(forced,))
            output = handle.generate(
                _ROUTE_PROMPTS, SamplingParams(max_tokens=16, temperature=0, ignore_eos=True), use_tqdm=False
            )
            digests = llm.collective_rpc(_route_digest)
            assert digests[0] and all(item == digests[0] for item in digests)
            samples = llm.collective_rpc(_route_samples)[0]
            handle.diagnostics()
            return tokens(output), digests[0], samples
        finally:
            if native:
                llm.collective_rpc(_native_steps, args=(False,))


@pytest.mark.parametrize("graphs", [False, "auto"])
@pytest.mark.parametrize("route", ["step_loop", "recirculation", "early_exit"])
def test_j5_routes(route, graphs):
    """Compare TP choices on shared histories, with strict rank/run/native equality.

    BF16 GEMMs/reduction order change across TP sizes. Allow 0.5 logit for
    recirculation/exit and 1.0 for the repeatedly fed-back StepLoop carry over
    sixteen steps. Teacher forcing avoids changes to later token inputs. The
    independent native StepLoop oracle must match every carry, norm and KV bit.
    """
    from pllm.workloads.recirculation import RecirculationWorkload
    from pllm.workloads.step_loop import StepLoopWorkload

    workloads = {"step_loop": StepLoopWorkload(), "recirculation": RecirculationWorkload(), "early_exit": EarlyExit()}
    settings = replace(config(graphs=graphs), execution=ExecutionOptions(capture_memory_bytes=256 << 20))
    reference = repeated = None
    forced = reference_scores = ()
    tolerance = 1.0 if route == "step_loop" else 0.5
    for tp in (1, 2, 2):
        observed, digest, samples = _route_generation(
            replace(settings, tensor_parallel_size=tp), workloads[route], forced, route == "step_loop"
        )
        if tp == 1:
            reference = observed
            forced = tuple(sample[0] for sample in samples)
            reference_scores = tuple(sample[1] for sample in samples)
            continue
        assert observed == reference
        regret = max(float(sample[2].max()) for sample in samples)
        agreeing = sum(int((sample[0] == target).sum()) for sample, target in zip(samples, forced, strict=True))
        score_error = max(
            float((sample[1] - score).abs().max()) for sample, score in zip(samples, reference_scores, strict=True)
        )
        assert regret <= tolerance and score_error <= tolerance, (route, graphs, regret, score_error)
        if repeated is not None:
            assert (observed, digest) == repeated
        repeated = observed, digest
        print(
            f"J5 {route} graphs={graphs}: {agreeing}/{16 * len(_ROUTE_PROMPTS)} natural TP=1/2 tokens agree; "
            f"max greedy regret={regret:.6f}, selected-logit error={score_error:.6f} (bound {tolerance}); "
            "ranks/runs bitwise equal",
            flush=True,
        )


_M4_FIELDS = []
_M4_SYNC = None
_M4_RETIRE = None


def _m4_instrument(worker):
    from types import SimpleNamespace

    from pllm.testing_gpu import _SyncCheck
    from test.gpu import vllm_tp_smoke as test

    runner = worker.workload_runner
    session = runner.require_session()
    execution = session._deployment.execution
    sync = _SyncCheck(SimpleNamespace(runner=runner))
    test._M4_SYNC = sync
    execution.execute_anchor = sync.anchor
    execution.execute_part = sync.part
    session.prepare_input = sync.input
    session._forward_overlay.prepare = sync.overlay
    test._M4_FIELDS.clear()
    retire = execution.arena.retire
    test._M4_RETIRE = retire

    def record(refs):
        if not test._M4_FIELDS:
            selected = refs[:1]
            pin = execution.arena.pin_exact(selected, torch.cuda.current_stream())
            try:
                test._M4_FIELDS.extend(
                    tuple(field.float().cpu().numpy().tobytes() for field in execution.arena.snapshot(pin, ref))
                    for ref in selected
                )
            finally:
                execution.arena.wait_release(execution.arena.release_pin(pin))
        return retire(refs)

    execution.arena.retire = record


def _m4_fields(worker):
    from test.gpu.vllm_tp_smoke import _M4_FIELDS, _M4_RETIRE, _M4_SYNC

    assert len(_M4_SYNC.input_checked) >= 8
    session = worker.workload_runner.require_session()
    execution = session._deployment.execution
    execution.arena.retire = _M4_RETIRE
    execution.execute_anchor = _M4_SYNC.execute
    execution.execute_part = _M4_SYNC.execute_part
    session.prepare_input = _M4_SYNC.prepare_input
    session._forward_overlay.prepare = _M4_SYNC.prepare_overlay
    return _M4_FIELDS


def _m4_rounding_unit(values):
    scale = values.abs().max().bfloat16()
    return (torch.nextafter(scale, torch.full_like(scale, float("inf"))).float() - scale.float()).item()


def _m4_projection_reference(kind, fields):
    from pllm.contracts.parameter_overlay import ParameterOverlayOperator
    from pllm.execution.parameter_projection import parameter_correction

    operator = (
        ParameterOverlayOperator.LOW_RANK_ADD if kind.startswith("low_rank") else ParameterOverlayOperator.DENSE_ADD
    )
    inputs, outputs = fields[0].shape[-1], fields[-1].shape[0]
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(8, inputs, generator=generator).bfloat16().cuda()
    weight = (0.02 * torch.randn(outputs, inputs, generator=generator)).bfloat16().cuda()
    fields = tuple(value.cuda() for value in fields)
    base_truth = hidden.double() @ weight.double().T
    truth = base_truth + parameter_correction(hidden.double(), tuple(value.double() for value in fields), operator)
    dtype = torch.float32 if kind.endswith("train") else torch.bfloat16
    partials, bases = [], []
    for x, w, a in zip(hidden.chunk(2, -1), weight.chunk(2, -1), fields[0].chunk(2, -1)):
        x, w, a = x.contiguous(), w.contiguous(), a.contiguous()
        base = x @ w.T
        weights = tuple(
            value.to(dtype).unsqueeze(0).expand(len(x), *value.shape).contiguous() for value in (a, *fields[1:])
        )
        delta = parameter_correction(x.to(dtype)[:, None], weights, operator, batched=True)[:, 0]
        bases.append(base.float())
        partials.append(base.float() + delta.float())
    native_error = sum(bases).bfloat16().double() - base_truth
    error = sum(partials).bfloat16().double() - truth
    native_rms = native_error.square().mean().sqrt().item()
    rms = error.square().mean().sqrt().item()
    assert rms <= 1.05 * native_rms
    assert error.abs().max() <= native_error.abs().max() + _m4_rounding_unit(truth)
    print(f"M4 {kind} FP64 projection: native RMS={native_rms:.6g}, adapter RMS={rms:.6g}", flush=True)


def _m4_scores(output):
    return torch.tensor(
        [
            [step[token].logprob for token, step in zip(row.outputs[0].token_ids, row.outputs[0].logprobs)]
            for row in output
        ]
    )


def _m4_params():
    # Raw logprobs precede this mask, retaining the unmodified model distribution.
    return SamplingParams(max_tokens=16, temperature=0, ignore_eos=True, logprobs=1, allowed_token_ids=[42])


@pytest.fixture(scope="module")
def m4_reference():
    scores = []
    for tp in (1, 2):
        settings = replace(config(), tensor_parallel_size=tp, multiprocessing=True)
        with pllm.make_llm(settings) as handle:
            plain = handle._require_open()
            plain.sleep(level=0, mode="keep")
            plain.enqueue(PROMPTS, _m4_params(), use_tqdm=False)
            plain.wake_up(tags=["scheduling"])
            scores.append(_m4_scores(plain.wait_for_completion(RequestOutput, use_tqdm=False)))
    error = (scores[1] - scores[0]).abs()
    print(f"M4 plain vLLM TP logprob max_abs={error.max():.6g}, RMS={error.square().mean().sqrt():.6g}", flush=True)
    return error


@pytest.mark.parametrize("kind", ["dense", "low_rank", "dense_train", "low_rank_train"])
def test_m4_down(kind, monkeypatch, m4_reference):
    from time import perf_counter

    from test.tp_overlays import DenseDownTraining, DownAdapter, LowRankDownTraining

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    workload = {
        "dense": DownAdapter(),
        "low_rank": DownAdapter("low_rank"),
        "dense_train": DenseDownTraining(),
        "low_rank_train": LowRankDownTraining(),
    }[kind]
    observed = []
    rates = []
    params = _m4_params()
    for tp in (1, 2):
        settings = replace(
            config(),
            tensor_parallel_size=tp,
            multiprocessing=True,
            execution=ExecutionOptions(capture_memory_bytes=256 << 20),
        )
        report = pllm.explain(workload, settings)
        with pllm.make_llm(settings, workload=workload) as handle:
            llm = handle._require_open()
            llm.collective_rpc(_m4_instrument)
            output = handle.generate(PROMPTS, params)
            fields = llm.collective_rpc(_m4_fields)
            # Field snapshots are explicit diagnostics, outside the measured generation.
            started = perf_counter()
            handle.generate(PROMPTS, SamplingParams(max_tokens=64, temperature=0, ignore_eos=True))
            rates.append(8 * 64 / (perf_counter() - started))
            diagnostics = handle.diagnostics()
            assert diagnostics.active_states == 0
            if kind.endswith("train"):
                assert diagnostics.update_count > 0
        roots = []
        for row in range(1):
            values = []
            for index, spec in enumerate(report.states[0].fields):
                parts = [
                    torch.frombuffer(bytearray(rank[row][index]), dtype=torch.float32).reshape(spec.shape)
                    for rank in fields
                ]
                value = torch.cat(parts, dim=-1) if index == 0 else parts[0]
                if index:
                    assert all(torch.equal(parts[0], part) for part in parts[1:])
                values.append(value)
            roots.append(values)
        observed.append((tokens(output), roots, _m4_scores(output)))
    assert observed[0][0] == observed[1][0]
    # Use native TP drift plus one model-precision ULP; the FP64 oracle checks the projection arithmetic.
    torch.testing.assert_close(
        observed[1][2], observed[0][2], rtol=0, atol=m4_reference.max().item() + _m4_rounding_unit(observed[0][2])
    )
    _m4_projection_reference(kind, observed[0][1][0])
    for whole, sharded in zip(observed[0][1], observed[1][1]):
        for index, (value, reference) in enumerate(zip(sharded, whole)):
            torch.testing.assert_close(
                value, reference, rtol=0, atol=1e-6 if kind.endswith("train") and index == 0 else 0
            )
    print(
        f"M4 {kind}: TP1={rates[0]:.1f} TP2={rates[1]:.1f} tok/s; "
        f"logprob max_abs={(observed[1][2] - observed[0][2]).abs().max():.6g}; "
        f"field max_abs={max((a - b).abs().max().item() for x, y in zip(observed[1][1], observed[0][1]) for a, b in zip(x, y)):.6g}; "
        "zero host syncs",
        flush=True,
    )
