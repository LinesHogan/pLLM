"""Request cache identity and admission hashing for vLLM's scheduler process."""

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Literal

from vllm import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import BlockHash, generate_block_hash_extra_keys, hash_block_tokens
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from pllm.contracts.backend_attachment import BackendAttachment
from pllm.contracts.codec import boundary_digest, decode_boundary, encode_boundary

POLICY_KEY = "pllm.prefix_cache"


@dataclass(frozen=True)
class CachePolicy:
    """Driver-owned cache identity, sent to the worker to reject mismatched requests."""

    namespace: str
    role_sensitive: bool
    generated_kv: Literal["reuse", "isolate"]

    @classmethod
    def for_attachment(cls, attachment: BackendAttachment, policy: Literal["reuse", "isolate"]) -> "CachePolicy":
        """Conservatively derive role dependence and stable mathematical identity from backend facts."""
        graph = attachment.plan.semantic_graph
        replay = None if graph.replay_task is None else graph.replay_task.replay
        sensitive = (
            bool(attachment.spec.activation_bindings or attachment.spec.overlay_bindings)
            # Reading `layers.generated` or PositionState adds the position resource to the route source.
            or (replay is not None and (replay.rows != "all" or replay.role_sensitive))
        )
        return cls(
            boundary_digest((attachment.probe.model_fingerprint, graph, attachment.probe.position_states)),
            sensitive,
            policy,
        )

    def sampling(self, params: SamplingParams) -> SamplingParams:
        """Copy caller options and carry this identity through EngineCore serialization."""
        result = params.clone()
        result.extra_args = {**(params.extra_args or {}), POLICY_KEY: encode_boundary(self)}
        return result

    @classmethod
    def read(cls, params: SamplingParams | None) -> "CachePolicy":
        """Read the identity at scheduler admission; raw engine calls carry none."""
        value = None if params is None else (params.extra_args or {}).get(POLICY_KEY)
        if not isinstance(value, str):
            raise ValueError("submit requests through the workload handle to supply the cache policy")
        return decode_boundary(value, cls)


@dataclass(frozen=True)
class RequestBlockHasher:
    """Incremental upstream hash chaining with a selective generated-boundary extra key."""

    block_size: int
    hash_function: Callable[[Hashable], bytes]
    policy: CachePolicy

    def __call__(self, request: Request) -> list[BlockHash]:
        start = len(request.block_hashes) * self.block_size
        cursor = -1 if start else 0
        parent = request.block_hashes[-1] if request.block_hashes else None
        hashes: list[BlockHash] = []
        for start in range(start, request.num_tokens - self.block_size + 1, self.block_size):
            end = start + self.block_size
            extras, cursor = generate_block_hash_extra_keys(request, start, end, cursor)
            keys = extras or ()
            if start == 0:
                keys += (("pllm.workload", self.policy.namespace, self.policy.generated_kv),)
            if self.policy.role_sensitive and self.policy.generated_kv == "isolate" and end > request.num_prompt_tokens:
                keys += (("pllm.generated-boundary", request.num_prompt_tokens),)
            parent = hash_block_tokens(self.hash_function, parent, request.all_token_ids[start:end], keys or None)
            hashes.append(parent)
        return hashes


class WorkloadScheduler(Scheduler):
    """Rehash before stock admission and connector callbacks, including EngineCore's initial hashes."""

    def add_request(self, request: Request) -> None:
        policy = CachePolicy.read(request.sampling_params)
        if self.cache_config.enable_prefix_caching or self.connector is not None:
            request._block_hasher = RequestBlockHasher(  # pyright: ignore[reportPrivateUsage]
                self.hash_block_size, get_hash_fn_by_name(self.cache_config.prefix_caching_hash_algo), policy
            )
            request.block_hashes.clear()
            request.update_block_hashes()
        super().add_request(request)


class WorkloadAsyncScheduler(WorkloadScheduler, AsyncScheduler):
    """Keep upstream asynchronous scheduling with the same admission boundary."""
