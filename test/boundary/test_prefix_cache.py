"""CPU evidence for selective hashes, stock admission, and signed role sensitivity."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.request import Request

import pllm
from pllm.backends.vllm.prefix_cache import CachePolicy, RequestBlockHasher, WorkloadAsyncScheduler, WorkloadScheduler
from pllm.config import ExecutionOptions
from pllm.workloads.recirculation import RecirculationWorkload
from pllm.workloads.step_loop import StepLoop
from test.workloads.test_recirculation import FACTS

POLICY = CachePolicy("a" * 64, True, "isolate")
HASH = get_hash_fn_by_name("sha256")
init_none_hash(HASH)


def request(prompt, size=4, policy=POLICY, **kwargs):
    result = Request("test", [1] * prompt, policy.sampling(SamplingParams(max_tokens=32)), None, **kwargs)
    result._block_hasher = RequestBlockHasher(size, HASH, policy)
    result.update_block_hashes()
    return result


@pytest.mark.parametrize("size", (2, 4, 8))
@pytest.mark.parametrize("boundary", (0, 3, 4, 5, 8))
def test_boundaries_and_incremental_resume(size, boundary):
    donor, receiver = request(boundary, size), request(16, size)
    donor.append_output_token_ids([1] * (16 - boundary))
    for index, (left, right) in enumerate(zip(donor.block_hashes, receiver.block_hashes, strict=True)):
        assert (left == right) == ((index + 1) * size <= boundary)
    expected = list(donor.block_hashes)
    donor.block_hashes.clear()
    donor.update_block_hashes()
    assert donor.block_hashes == expected
    donor.update_block_hashes()
    assert donor.block_hashes == expected


@pytest.mark.parametrize("policy", (replace(POLICY, generated_kv="reuse"), replace(POLICY, role_sensitive=False)))
def test_reuse_and_stateless_routes(policy):
    donor, receiver = request(3, policy=policy), request(16, policy=policy)
    donor.append_output_token_ids([1] * 13)
    assert donor.block_hashes == receiver.block_hashes


@pytest.mark.parametrize("scheduler_type", (WorkloadScheduler, WorkloadAsyncScheduler))
@pytest.mark.parametrize("enabled", (True, False))
def test_admission_rehashes_before_connector(scheduler_type, enabled):
    scheduler = object.__new__(scheduler_type)
    scheduler.hash_block_size, scheduler.cache_config = (
        4,
        SimpleNamespace(prefix_caching_hash_algo="sha256", enable_prefix_caching=enabled),
    )
    # vLLM 0.32's add_request also reads spec_decode_metrics_level.
    scheduler.requests, scheduler.waiting, scheduler.log_stats, scheduler.spec_decode_metrics_level = (
        {},
        FCFSRequestQueue(),
        False,
        "none",
    )
    scheduler.connector = Mock() if enabled else None
    incoming = Request(
        "test",
        [1] * 8,
        POLICY.sampling(SamplingParams(max_tokens=8)),
        None,
        block_hasher=get_request_block_hasher(4, HASH) if enabled else None,
    )
    stock = list(incoming.block_hashes)

    def admitted(req):
        assert req.block_hashes != stock

    if enabled:
        scheduler.connector.on_new_request.side_effect = admitted
    scheduler.add_request(incoming)
    assert incoming.block_hashes == (request(8).block_hashes if enabled else [])
    assert scheduler.requests["test"] is incoming and scheduler.waiting.peek_request() is incoming
    if enabled:
        scheduler.connector.on_new_request.assert_called_once_with(incoming)


def test_namespace_caller_extras_and_envelope():
    params = SamplingParams(extra_args={"caller": 17})
    assert POLICY.sampling(params).extra_args["caller"] == 17 and params.extra_args == {"caller": 17}
    assert CachePolicy.read(POLICY.sampling(params)) == POLICY
    baseline = request(8).block_hashes
    assert request(8, cache_salt="caller").block_hashes != baseline
    assert request(8, policy=replace(POLICY, namespace="b" * 64)).block_hashes != baseline
    assert request(8, policy=replace(POLICY, generated_kv="reuse")).block_hashes != baseline
    with pytest.raises(ValueError, match="workload handle"):
        CachePolicy.read(SamplingParams())


class GeneratedRoute(pllm.Workload):
    @pllm.layer_route(tokens="all")
    def loop(self, layers):
        return layers(hidden=torch.where(layers.generated, layers.seed * 0.5, layers.seed))[-1]


class DecodeRoute(pllm.Workload):
    @pllm.layer_route(tokens="decode")
    def loop(self, layers):
        return layers(hidden=layers.seed * 0.5)[-1]


@pytest.mark.parametrize(
    "workload,sensitive",
    ((StepLoop(), True), (RecirculationWorkload(), False), (GeneratedRoute(), True), (DecodeRoute(), True)),
)
def test_signed_sensitivity(workload, sensitive):
    facts = replace(FACTS, probe=replace(FACTS.probe, position_states=workload.backend_requirements().position_states))
    attachment = workload.compile_backend(facts, ExecutionOptions()).attachment
    policy = CachePolicy.for_attachment(attachment, "isolate")
    assert policy.role_sensitive is sensitive
    assert CachePolicy.for_attachment(attachment, "isolate") == policy
    changed = replace(
        attachment,
        probe=replace(attachment.probe, position_states=(("history", 1.0),)),
    )
    assert CachePolicy.for_attachment(changed, "isolate").namespace != policy.namespace


@pytest.mark.parametrize("active", ("_step", "_identities", "_pending_identities"))
def test_active_replacement_rejected(active):
    from pllm.backends.vllm.session import VllmSession

    with pytest.raises(RuntimeError, match="finish or abort"):
        VllmSession.require_idle(
            SimpleNamespace(
                **{name: (True if name == active else None) for name in ("_step", "_identities", "_pending_identities")}
            )
        )
