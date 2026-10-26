"""Request cache identity and admission hashing for vLLM's scheduler process."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from vllm import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import BlockHash, generate_block_hash_extra_keys, hash_block_tokens
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from pllm.contracts.backend_attachment import SignedBackendAttachment
from pllm.execution.codec import boundary_digest, decode_boundary, encode_boundary

POLICY_ENV = "PLLM_GENERATED_KV"
POLICY_KEY = "pllm.prefix_cache"


@dataclass(frozen=True)
class CachePolicy:
    """Versioned request envelope, reconstructed independently by the installed worker."""

    namespace: str
    role_sensitive: bool
    generated_kv: Literal["reuse", "isolate"]
    version: int = 1

    @classmethod
    def for_attachment(cls, attachment: SignedBackendAttachment, policy: Literal["reuse", "isolate"]) -> "CachePolicy":
        """Conservatively derive role dependence and stable mathematical identity from signed facts."""
        inputs = attachment.inputs
        graph = inputs.plan.semantic_graph
        sensitive = (
            bool(inputs.request.activation_bindings or inputs.request.overlay_bindings)
            or any(
                task.replay is not None and (task.replay.rows != "all" or task.replay.position_states)
                for task in graph.tasks
            )
            or any(
                resource.name == "position" for source in graph.source_regions for resource in source.program.resources
            )
        )
        return cls(
            boundary_digest((inputs.probe.model_fingerprint, graph, inputs.probe.position_states)), sensitive, policy
        )

    def sampling(self, params: SamplingParams) -> SamplingParams:
        """Copy caller options and carry this identity through EngineCore serialization."""
        result = params.clone()
        result.extra_args = {**(params.extra_args or {}), POLICY_KEY: encode_boundary(self)}
        return result

    @classmethod
    def read(cls, params: SamplingParams | None) -> "CachePolicy":
        """Validate the transport at scheduler admission, including missing raw-call metadata."""
        value = None if params is None else (params.extra_args or {}).get(POLICY_KEY)
        if not isinstance(value, str):
            raise ValueError("submit requests through the workload handle to supply the cache policy")
        try:
            result = decode_boundary(value, cls)
        except Exception as error:
            raise ValueError("supply a version 1 workload cache policy envelope") from error
        if result.version != 1:
            raise ValueError("supply a version 1 workload cache policy envelope")
        if len(result.namespace) != 64 or any(c not in "0123456789abcdef" for c in result.namespace):
            raise ValueError("cache namespace must be a mathematical SHA256 digest")
        return result


@dataclass(frozen=True)
class RequestBlockHasher:
    """Incremental upstream hash chaining with a selective generated-boundary extra key."""

    block_size: int
    hash_function: Callable[[object], bytes]
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
