"""CPU boundary checks for online request stamping, StepDriver admission/retirement, and rejection guards."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

import pllm
from pllm import testing
from pllm.backends.vllm.prefix_cache import CachePolicy, OnlineRequest, read_online
from pllm.backends.vllm.serving import AsyncVllmLlm, _WorkloadAsyncLLM
from pllm.contracts.host_batch import StateImport, WeightedRowBatch
from pllm.contracts.request_identity import RequestIdentity
from pllm.execution.step_driver import StepDriver
from pllm.workloads.lora import OutputLoRAConfig, OutputLoRAWorkload
from pllm.workloads.reft import ReFTConfig, ReFTWorkload
from pllm.workloads.tlm import TLMConfig, TLMWorkload
from pllm.workloads.video import VideoWorkload


@dataclass(frozen=True)
class _OnlineStepWorkload(pllm.Workload):
    state = pllm.RequestState({"v": (4,)}, optimizer=pllm.AdamW(lr=0.1))

    @pllm.export
    def read(
        self,
        hidden: torch.Tensor = pllm.final_hidden(),
        v: torch.Tensor = pllm.parameter("state", "v"),
    ) -> torch.Tensor:
        return hidden[:, :4] + v

    @pllm.train(state="state")
    def loss(
        self,
        hidden: torch.Tensor = pllm.final_hidden(),
        v: torch.Tensor = pllm.parameter("state", "v"),
    ) -> torch.Tensor:
        return (hidden[:, :4] - v).square().mean(dim=1)


def test_online_sampling_roundtrip_and_route_stripping() -> None:
    policy = CachePolicy(3, True, "isolate")
    identity = OnlineRequest(route="s1", sample_index=2, prompt_index=7)
    original = SamplingParams(max_tokens=16, extra_args={"pllm_route": "s1", "caller": 42})

    stamped = policy.online_sampling(original, identity)

    assert "pllm_route" not in (stamped.extra_args or {})
    assert stamped.extra_args is not None and stamped.extra_args["caller"] == 42
    assert original.extra_args == {"pllm_route": "s1", "caller": 42}
    assert CachePolicy.read(stamped) == policy
    assert read_online(stamped) == identity
    assert read_online(SamplingParams()) is None


def test_step_driver_staggered_online_admission_and_drain() -> None:
    binding = testing.final_hidden_binding(2)
    facts = testing.backend_facts(8, (binding,), max_lineages=4, memory_bytes=1 << 20, layer_count=2)
    compiled = _OnlineStepWorkload().compile_backend(facts)
    compiled.execution.admit_instances(None)
    driver = StepDriver(compiled.execution, compiled.attachment)
    try:
        req_a = RequestIdentity("req-a", 0, 0)
        req_b = RequestIdentity("req-b", 1, 0)
        req_cancel = RequestIdentity("req-cancel", 2, 0)

        driver.stage_online(req_a)
        assert driver.is_staged("req-a")
        with pytest.raises(ValueError, match="duplicate online request"):
            driver.stage_online(req_a)

        driver.schedule(finished=(), preempted=(), new=("req-a",))
        assert not driver.is_staged("req-a")
        assert "req-a" in driver.admitted

        step0 = testing.step_input((req_a,), {binding: torch.randn(1, 8)}, step_id=0)
        driver.begin(step0, None)
        driver.after_forward(step0, None)
        driver.sample(None, None)
        driver.commit(None)

        driver.stage_online(req_b)
        driver.stage_online(req_cancel)
        driver.schedule(finished=("req-cancel",), preempted=(), new=("req-b",))
        assert not driver.is_staged("req-cancel")
        assert "req-cancel" not in driver.admitted
        assert "req-a" in driver.admitted and "req-b" in driver.admitted

        step1 = testing.step_input((req_a, req_b), {binding: torch.randn(2, 8)}, step_id=1)
        driver.begin(step1, None)
        driver.after_forward(step1, None)
        driver.sample(None, None)
        driver.commit(None)

        driver.schedule(finished=("req-a",), preempted=(), new=())
        assert "req-a" not in driver.admitted and "req-b" in driver.admitted

        step2 = testing.step_input((req_b,), {binding: torch.randn(1, 8)}, step_id=2)
        driver.begin(step2, None)
        driver.after_forward(step2, None)
        driver.sample(None, None)
        driver.commit(None)

        driver.schedule(finished=("req-b",), preempted=(), new=())
        diag = driver.execution.diagnostics()
        assert diag.active_states == 0
        assert diag.inference_pins == 0
        assert diag.pending_state_releases == 0
    finally:
        driver.release()
        driver.close()


@dataclass(frozen=True)
class _BatchWorkload(pllm.Workload):
    state = pllm.InstanceState({"v": (4,)}, optimizer=pllm.AdamW(lr=0.1))

    @pllm.export
    def read(self, hidden: torch.Tensor = pllm.final_hidden()) -> torch.Tensor:
        return hidden

    @pllm.train(state="state")
    def loss(
        self,
        hidden=pllm.batch("hidden"),
        targets=pllm.batch("targets"),
        weights=pllm.batch("weights"),
        v=pllm.parameter("state", "v"),
    ):
        return (((hidden[:, :4] * v).sum(1) - targets).square() * weights).mean()


@pytest.mark.parametrize(
    "workload",
    (
        ReFTWorkload(ReFTConfig(layers=(0,), rank=2)),
        TLMWorkload(TLMConfig()),
        OutputLoRAWorkload(OutputLoRAConfig()),
        _BatchWorkload(),
        VideoWorkload(),
        None,
    ),
)
def test_async_admits_host_control_workloads(monkeypatch, workload) -> None:
    import os

    from pllm.backends.vllm import serving

    config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct")
    assert not config.uses_multiprocessing()
    assert config.uses_multiprocessing(async_engine=True)
    engine = object.__new__(_WorkloadAsyncLLM)

    def args(config, options, *, cuda_graphs):
        assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "1"
        assert options["worker_cls"] == "pllm.worker.GPUWorker"
        assert ("scheduler_cls" in options) == (workload is not None)
        return Mock()

    monkeypatch.setattr(serving, "_async_engine_args", args)
    monkeypatch.setattr(_WorkloadAsyncLLM, "from_vllm_config", Mock(return_value=engine))
    assert pllm.make_async_llm(config, workload=workload).engine is engine


def test_async_rejects_explicit_single_process() -> None:
    with pytest.raises(ValueError, match="multiprocessing=False"):
        pllm.make_async_llm(pllm.ModelConfig("unused", multiprocessing=False))


def test_train_batch_while_request_is_admitted() -> None:
    facts = testing.backend_facts(8, (testing.final_hidden_binding(2),), layer_count=2, memory_bytes=1 << 20)
    compiled = _BatchWorkload().compile_backend(facts)
    driver = StepDriver(compiled.execution, compiled.attachment)
    driver.execution.admit_instances()
    try:
        identity = RequestIdentity("in-flight", 0, 0)
        driver.stage_online(identity)
        driver.schedule((), (), (identity.request_id,))
        batch = WeightedRowBatch(torch.ones(2, 8), torch.ones(2), torch.ones(2))
        first = driver.train_batch(batch)
        second = driver.train_batch(batch)
        assert first.loss is not None and second.loss is not None and first.loss > second.loss
        assert first.parameters["v"].ne(second.parameters["v"]).all()
        assert driver.execution.latest(())[0].version_id == 2
        assert driver.execution.diagnostics().completed_sampler_pins == 0
    finally:
        driver.finish_call()
        driver.release()
        driver.close()


@dataclass(frozen=True)
class _ImportedRoute(pllm.Workload):
    state = pllm.RequestState({"v": (4,)}, key_fields=("route",), init=pllm.Imported())

    @pllm.export
    def read(self, hidden=pllm.final_hidden(), v=pllm.parameter("state", "v")):
        return hidden[:, :4] + v


def test_import_idle_live_and_dormant_route() -> None:
    facts = testing.backend_facts(8, (testing.final_hidden_binding(2),), layer_count=2, memory_bytes=1 << 20)
    compiled = _ImportedRoute().compile_backend(facts)
    driver = StepDriver(compiled.execution, compiled.attachment)
    update = StateImport((torch.arange(4.0).reshape(1, 4),))
    identity = RequestIdentity("reader", 0, 0, route="tenant")
    try:
        assert driver.import_state(update, route="tenant") == 1
        driver.stage_online_import(identity.request_id, update)
        driver.stage_online(identity)
        driver.schedule((), (), (identity.request_id,))
        assert driver.import_state(update, route="tenant") == 3
        (ref,) = driver.execution.latest((identity,))
        torch.testing.assert_close(driver.execution.snapshot(ref)[0], update.values[0][0])
        driver.schedule((identity.request_id,), (), ())
        assert driver.import_state(update, route="tenant") == 4
    finally:
        driver.finish_call()
        driver.release()
        driver.close()


def test_async_generate_uses_final_only_without_mutating_caller_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[SamplingParams | None] = []
    fake_output = Mock(finished=True)

    async def fake_stream(
        self: AsyncVllmLlm,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        *,
        request_id: str | None = None,
        route: str = "",
        state_import: StateImport[torch.Tensor] | None = None,
    ):
        seen.append(sampling_params)
        yield fake_output

    monkeypatch.setattr(AsyncVllmLlm, "stream", fake_stream)
    handle = object.__new__(AsyncVllmLlm)
    caller_params = SamplingParams(max_tokens=8, n=2)
    result = asyncio.run(handle.generate("hello", caller_params))
    assert result is fake_output
    assert caller_params.output_kind == RequestOutputKind.CUMULATIVE
    assert len(seen) == 1 and seen[0] is not None and seen[0] is not caller_params
    assert seen[0].output_kind == RequestOutputKind.FINAL_ONLY


@dataclass
class _FakeRequest:
    request_id: str
    sampling_params: SamplingParams


def test_bind_online_policy_stamps_parent_children_at_output_processor() -> None:
    recorded: list[tuple[str, CachePolicy, OnlineRequest | None]] = []

    def fake_register(
        request: _FakeRequest,
        prompt: str | None,
        parent_req: object | None = None,
        request_index: int = 0,
        queue: object | None = None,
    ) -> None:
        recorded.append(
            (request.request_id, CachePolicy.read(request.sampling_params), read_online(request.sampling_params))
        )

    engine = object.__new__(_WorkloadAsyncLLM)
    engine.output_processor = SimpleNamespace(request_states={}, add_request=fake_register)  # type: ignore[assignment]
    policy = CachePolicy(1, True, "isolate")
    engine.bind_online_policy(policy=policy, sample_limit=2, requires_route=False, has_route_state=False)

    parent = SimpleNamespace(request_id="parent-1", n=2)
    for idx in range(2):
        child = _FakeRequest(f"{idx}_parent-1", SamplingParams(max_tokens=4))
        engine.output_processor.add_request(child, "hi", parent, idx, None)  # type: ignore[arg-type]

    assert recorded == [
        ("0_parent-1", policy, OnlineRequest(route="", sample_index=0, prompt_index=0)),
        ("1_parent-1", policy, OnlineRequest(route="", sample_index=1, prompt_index=0)),
    ]

    oversized_parent = SimpleNamespace(request_id="parent-2", n=3)
    oversized_child = _FakeRequest("0_parent-2", SamplingParams(max_tokens=4))
    with pytest.raises(ValueError, match="sample index 2"):
        engine.output_processor.add_request(oversized_child, "hi", oversized_parent, 0, None)  # type: ignore[arg-type]


def test_weighted_ce_control_uses_model_vocabulary_and_returns_fields() -> None:
    from pllm.backends.vllm.control import BackendTrain, run_train_batch
    from pllm.contracts.codec import decode_boundary, encode_boundary
    from pllm.contracts.host_batch import WeightedCEBatch

    facts = testing.backend_facts(
        8, (testing.final_hidden_binding(2),), layer_count=2, vocab_size=32, memory_bytes=1 << 20
    )
    compiled = OutputLoRAWorkload(OutputLoRAConfig(rank=2)).compile_backend(facts)
    driver = StepDriver(compiled.execution, compiled.attachment)
    driver.execution.admit_instances()
    hidden = torch.ones(1, 8)
    labels = torch.tensor([5])
    try:
        with pytest.raises(ValueError, match="WeightedCEBatch"):
            driver.train_batch(WeightedCEBatch(hidden, torch.zeros(1, 31), labels, torch.ones(1)))
        batch = WeightedCEBatch(hidden, torch.zeros(1, 32), labels, torch.ones(1))
        command = decode_boundary(encode_boundary(BackendTrain.of(batch)), BackendTrain)
        step = run_train_batch(driver, command)
        assert step.loss is not None and step.loss.isfinite().all()
        assert step.parameters["a"].shape == (8, 2)
        assert step.parameters["b"].shape == (2, 32)
    finally:
        driver.release()
        driver.close()


def test_fanout_copies_fields_before_publication_and_discards_failed_edits() -> None:
    from contextlib import contextmanager

    from pllm.execution.state_feed import FanoutFeed

    published = []
    ranks = (Mock(), Mock())

    @contextmanager
    def writable():
        fields = {"v": torch.empty(3)}
        yield fields
        published.append(fields["v"].clone())

    for rank in ranks:
        rank.publish.side_effect = writable
    with FanoutFeed(ranks) as feed:
        with feed.publish() as fields:
            fields["v"].fill_(7)
        with pytest.raises(RuntimeError):
            with feed.publish():
                raise RuntimeError("cancel publication")
        feed.zero_()
        feed.missing()
    assert [v.tolist() for v in published] == [[7] * 3, [7] * 3, [0] * 3, [0] * 3]
    for rank in ranks:
        rank.missing.assert_called_once()
        rank.end.assert_called_once()
        rank.close.assert_called_once()
