"""Closed-window admission through vLLM's scheduler_cls extension.

This dedicated text-model policy uses the original Scheduler request queues,
KVCacheManager, output bookkeeping and retirement. It never changes the token
count after allocating a window, and it never manufactures Mamba alignment.
"""

from dataclasses import dataclass
from math import ceil
from time import monotonic

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from pllm.contracts.backend_attachment import SignedBackendAttachment
from pllm.execution.codec import decode_boundary

WINDOW_ARGUMENT = "pllm.closed_window"


@dataclass(frozen=True, slots=True)
class WindowAdmission:
    """Compact admission facts taken from the installed signed compilation."""

    compilation_signature: str
    chunk_size: int
    chunks: int


def model_scan_admission(attachment: SignedBackendAttachment) -> WindowAdmission:
    """Project only the compiler-selected admission facts into the engine request."""
    (source,) = attachment.inputs.plan.schedule_physical.source_schedules
    window = source.window
    assert window is not None
    return WindowAdmission(attachment.inputs.plan.schedule_decision.signature, window.request.quantum, window.chunks)


@dataclass(frozen=True, slots=True)
class WindowGrant:
    """The native allocator's complete blocks and token interval for one request."""

    request_id: str
    prefix: int
    tokens: int
    prompt_length: int
    blocks: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class WindowBatch:
    """Original allocation witnesses and unused budget for one engine step."""

    admission: WindowAdmission | None
    grants: tuple[WindowGrant, ...]
    block_size: int
    token_budget: int
    unused_tokens: int


@dataclass(kw_only=True)
class WindowSchedulerOutput(SchedulerOutput):
    """The ordinary runner output plus original-owner allocation witnesses."""

    window: WindowBatch


class ClosedWindowScheduler(Scheduler):
    """Fair complete-window admission for a dedicated non-preempting ModelScan engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            hash_block_size,
            mm_registry,
            include_finished_set,
            log_stats,
        )
        if (
            vllm_config.cache_config.enable_prefix_caching
            or vllm_config.speculative_config is not None
            or vllm_config.lora_config is not None
            or self.connector is not None
            or self.ec_connector is not None
            or self.is_encoder_decoder
            or not self.use_v2_model_runner
            or len(kv_cache_config.kv_cache_groups) != 1
            or self.scheduler_config.async_scheduling
        ):
            raise ValueError("closed windows need synchronous text admission, MRv2 and nonshared single-group KV")
        self._window: WindowAdmission | None = None
        self._round = 0

    def add_request(self, request: Request) -> None:
        """Accept only requests carrying the installed workload's window contract."""
        sampling = request.sampling_params
        if sampling is None or request.has_encoder_inputs or request.structured_output_request is not None:
            raise ValueError("a ModelScan engine accepts ordinary text sampling requests")
        extras = sampling.extra_args or {}
        encoded = extras.get(WINDOW_ARGUMENT)
        if not isinstance(encoded, str):
            raise ValueError("submit ModelScan requests through the installed workload handle")
        window = decode_boundary(encoded, WindowAdmission)
        if (
            window.chunk_size < 1
            or window.chunks < 1
            or window.chunk_size * window.chunks > self.max_num_scheduled_tokens
        ):
            raise ValueError("increase the engine token budget to fit the signed complete window")
        if self._window is not None and self._window != window:
            raise ValueError("a dedicated ModelScan engine cannot mix compiled window contracts")
        self._window = window
        if self._capacity(request) > self.kv_cache_manager.block_pool.get_num_free_blocks() + sum(
            len(self.kv_cache_manager.get_blocks(r.request_id).get_block_ids()[0]) for r in self.running
        ):
            raise ValueError("increase the KV budget to fit this request's prompt and maximum output")
        super().add_request(request)

    def _capacity(self, request: Request) -> int:
        return ceil(min(request.num_prompt_tokens + request.max_tokens, self.max_model_len) / self.block_size)

    def _headroom(self) -> int:
        # The allocator owns the blocks. This reserves future growth by refusing
        # new admissions, without allocating speculative tokens or a second ledger.
        return sum(
            self._capacity(r) - len(self.kv_cache_manager.get_blocks(r.request_id).get_block_ids()[0])
            for r in self.running
        )

    def _count(self, request: Request, budget: int) -> int:
        assert self._window is not None
        remaining = request.num_prompt_tokens - request.num_computed_tokens
        if remaining <= 0:
            return request.num_tokens - request.num_computed_tokens
        chunk = self._window.chunk_size
        if request.num_computed_tokens % chunk:
            raise ValueError("resume prefill only at a complete learning-chunk boundary")
        limit = min(chunk * self._window.chunks, budget)
        return remaining if remaining <= limit else limit - limit % chunk

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        """Reserve whole intervals before advancing the original scheduler's counters."""
        self.current_step += 1
        self.kv_cache_manager.new_step_starts()
        budget = self.max_num_scheduled_tokens if self._pause_state != PauseState.PAUSED_ALL else 0
        new: list[Request] = []
        running: list[Request] = []
        allocations: dict[str, KVCacheBlocks] = {}
        counts: dict[str, int] = {}
        grants: list[WindowGrant] = []
        candidates = list(self.running)
        # Rotation gives long windows progress even while short decode requests
        # could otherwise consume the first token of every engine budget.
        if candidates:
            first = self._round % len(candidates)
            candidates = candidates[first:] + candidates[:first]
        self._round += 1
        if self._round % 2 == 0:
            budget = self._admit_waiting(budget, new, allocations, counts, grants)
        for request in candidates:
            count = self._count(request, budget)
            if not 0 < count <= budget:
                continue
            blocks = self.kv_cache_manager.allocate_slots(request, count)
            if blocks is None:
                continue
            running.append(request)
            allocations[request.request_id] = blocks
            counts[request.request_id] = count
            budget -= count
            grants.append(self._grant(request, count))
        budget = self._admit_waiting(budget, new, allocations, counts, grants)
        cached = self._make_cached_request_data(running, [], counts, {}, allocations)
        window = self._window
        result = WindowSchedulerOutput(
            scheduled_new_reqs=[
                NewRequestData.from_request(r, allocations[r.request_id].get_block_ids(), r.all_token_ids.copy())
                for r in new
            ],
            scheduled_cached_reqs=cached,
            num_scheduled_tokens=counts,
            total_num_scheduled_tokens=sum(counts.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=[],
            preempted_req_ids=self.reset_preempted_req_ids,
            new_block_ids_to_zero=self._get_new_block_ids_to_zero(),
            window=WindowBatch(window, tuple(grants), self.block_size, self.max_num_scheduled_tokens, budget),
        )
        self._update_after_schedule(result)
        return result

    def _admit_waiting(
        self,
        budget: int,
        new: list[Request],
        allocations: dict[str, KVCacheBlocks],
        counts: dict[str, int],
        grants: list[WindowGrant],
    ) -> int:
        if self._pause_state != PauseState.UNPAUSED:
            return budget
        while self.waiting and budget and len(self.running) < self.max_num_running_reqs:
            request = self.waiting.peek_request()
            count = self._count(request, budget)
            if not 0 < count <= budget:
                break
            if self._capacity(request) + self._headroom() > self.kv_cache_manager.block_pool.get_num_free_blocks():
                break
            blocks = self.kv_cache_manager.allocate_slots(
                request, count, full_sequence_must_fit=True, has_scheduled_reqs=bool(self.running)
            )
            if blocks is None:
                break
            self.waiting.pop_request()
            request.status = RequestStatus.RUNNING
            self.running.append(request)
            new.append(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED, monotonic())
            allocations[request.request_id] = self.kv_cache_manager.get_blocks(request.request_id)
            counts[request.request_id] = count
            budget -= count
            grants.append(self._grant(request, count))
        return budget

    def _grant(self, request: Request, count: int) -> WindowGrant:
        (blocks,) = self.kv_cache_manager.get_blocks(request.request_id).get_block_ids()
        return WindowGrant(
            request.request_id, request.num_computed_tokens, count, request.num_prompt_tokens, tuple(blocks)
        )
