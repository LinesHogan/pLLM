"""Teacher-forced full-vocabulary diagnostics for workload authors via pllm.testing."""

from __future__ import annotations

# Like testing_gpu, diagnostics inspect the in-process engine and sampler.
# pyright: reportPrivateUsage=false
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import ceil, isfinite
from time import perf_counter
from typing import ParamSpec, TypeVar
from unittest.mock import patch

import torch
from vllm import SamplingParams
from vllm.v1.sample.sampler import Sampler as Mrv1Sampler
from vllm.v1.worker.gpu.sample.sampler import Sampler as Mrv2Sampler

import pllm
from pllm.backends.vllm.runtime import VllmLlm
from pllm.config import ModelConfig
from pllm.testing_gpu import _PROMPTS, _plain_model, _release_models
from pllm.usage import Workload

_Sample = ParamSpec("_Sample")
_DataT = TypeVar("_DataT")
_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True, slots=True)
class TokenDistance:
    """A distance at a zero-based prompt index and predicted token offset in its full sequence."""

    prompt_index: int
    token_offset: int
    value: float


@dataclass(frozen=True, slots=True)
class DistanceSummary:
    """Token-weighted mean, nearest-rank P99, maximum, and largest distances with their positions."""

    mean: float
    p99: float
    max: float
    worst: tuple[TokenDistance, ...]


@dataclass(frozen=True, slots=True)
class DistributionResult:
    """Exact diagnostic values and threshold failures; dataclasses.asdict is JSON-ready.

    KL units are nats. run_seconds includes generation and per-token diagnostics for both models;
    elapsed_seconds also includes loading, compilation, graph capture and cleanup.
    """

    kl_plain_to_workload: DistanceSummary
    kl_workload_to_plain: DistanceSummary
    tv: DistanceSummary
    token_count: int
    run_seconds: float
    elapsed_seconds: float
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Whether all supplied thresholds passed and no distance was NaN."""
        return not self.failures

    def __str__(self) -> str:
        metrics = (
            ("KL(plain || workload)", self.kl_plain_to_workload),
            ("KL(workload || plain)", self.kl_workload_to_plain),
            ("TV", self.tv),
        )
        return "\n".join(
            (
                f"{'PASS' if self.passed else 'FAIL'} distribution: {self.token_count} tokens",
                "distance                 mean        P99         max",
                *(f"{name:24} {metric.mean:.6g}  {metric.p99:.6g}  {metric.max:.6g}" for name, metric in metrics),
                *(
                    f"worst {name} (prompt:token=value): "
                    + ", ".join(f"{item.prompt_index}:{item.token_offset}={item.value:.6g}" for item in metric.worst)
                    for name, metric in metrics
                ),
                f"run {self.run_seconds:.3f}s; total {self.elapsed_seconds:.3f}s",
                *self.failures,
            )
        )


def _distances(log_plain: torch.Tensor, log_workload: torch.Tensor) -> torch.Tensor:
    """Reduce FP32 log probabilities over the entire vocabulary, preserving support mismatches."""
    p, q = log_plain.exp(), log_workload.exp()
    delta = log_plain - log_workload
    forward = torch.where(p == 0, 0, p * delta).sum(-1).clamp_min(0)
    reverse = torch.where(q == 0, 0, -q * delta).sum(-1).clamp_min(0)
    return torch.stack((forward, reverse, 0.5 * (p - q).abs().sum(-1)), dim=-1)


def _summaries(values: torch.Tensor, positions: list[tuple[int, int]], worst: int) -> tuple[DistanceSummary, ...]:
    """Reduce on device and transfer only final summaries and worst-position indices."""
    count = values.shape[0]
    largest, indices = values.topk(min(worst, count), dim=0)
    stats = torch.stack((values.mean(0), values.kthvalue(ceil(0.99 * count), dim=0).values, values.amax(0)))
    # Explicit diagnostic queries synchronize once, after all measured model steps have finished.
    packed = torch.cat((stats, largest, indices.float())).T.cpu()
    width = largest.shape[0]
    return tuple(
        DistanceSummary(
            float(row[0]),
            float(row[1]),
            float(row[2]),
            tuple(
                TokenDistance(*positions[int(index)], float(value))
                for value, index in zip(row[3 : 3 + width], row[3 + width :], strict=True)
            ),
        )
        for row in packed
    )


@dataclass
class _Observation:
    """Device-only recordings at the native sampler boundary, with optional teacher forcing."""

    temperature: float = 1
    greedy: bool = False
    reference: Sequence[torch.Tensor] = ()
    forced: torch.Tensor | None = None
    log_probs: list[torch.Tensor] = field(default_factory=list[torch.Tensor])
    distances: list[torch.Tensor] = field(default_factory=list[torch.Tensor])
    step: int = 0

    def record(self, sampled: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        assert logits.shape[0] == 1
        log_probs = (logits.float() * self.temperature).log_softmax(-1)
        if self.forced is None:
            self.log_probs.append(log_probs.detach())
            if self.greedy:
                sampled = logits.argmax(-1).reshape_as(sampled)
        else:
            self.distances.append(_distances(self.reference[self.step], log_probs))
            sampled = self.forced[self.step].expand_as(sampled)
        self.step += 1
        return sampled


def _record_sample(
    observation: _Observation,
    sample: Callable[_Sample, tuple[torch.Tensor, torch.Tensor | None]],
) -> Callable[_Sample, tuple[torch.Tensor, torch.Tensor | None]]:
    """Observe the native result without depending on a release's sample argument list."""

    def observed(*args: _Sample.args, **kwargs: _Sample.kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:
        sampled, processed = sample(*args, **kwargs)
        assert processed is not None
        return observation.record(sampled, processed), processed

    return observed


@contextmanager
def _observe_mrv2(sampler: Mrv2Sampler, observation: _Observation) -> Generator[None]:
    with patch.object(sampler, "sample", _record_sample(observation, sampler.sample)):
        yield


@contextmanager
def _observe_mrv1(sampler: Mrv1Sampler, observation: _Observation) -> Generator[None]:
    with (
        patch.object(sampler, "sample", _record_sample(observation, sampler.sample)),
        patch.object(sampler, "logprobs_mode", "processed_logits"),
        patch.object(sampler.topk_topp_sampler, "logprobs_mode", "processed_logits"),
    ):
        yield


@contextmanager
def _observe(llm: VllmLlm[_DataT, _ResultT], config: ModelConfig, observation: _Observation) -> Generator[None]:
    runner = llm._require_open().llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    sampler = runner.sampler
    if config.runner == "mrv2":
        assert isinstance(sampler, Mrv2Sampler)
        with _observe_mrv2(sampler, observation):
            yield
    else:
        assert isinstance(sampler, Mrv1Sampler)
        with _observe_mrv1(sampler, observation):
            yield


@dataclass(frozen=True)
class _Reference:
    prompt: str
    prompt_length: int
    continuation: tuple[int, ...]
    log_probs: tuple[torch.Tensor, ...]


def _plain_continuations(
    workload: Workload,
    config: ModelConfig,
    prompts: Sequence[str],
    tokens: int,
    temperature: float,
) -> tuple[tuple[_Reference, ...], float]:
    references: list[_Reference] = []
    seconds = 0.0
    with _plain_model(workload, config) as plain:
        tokenizer = plain._require_open().get_tokenizer()
        for prompt in prompts:
            prompt_length = len(tokenizer.encode(prompt))
            if prompt_length + tokens > config.max_model_len or prompt_length > config.max_num_batched_tokens:
                raise ValueError(
                    "check_distribution needs each prompt to fit max_num_batched_tokens and prompt + tokens <= max_model_len"
                )
            observation = _Observation(temperature=temperature or 1, greedy=temperature == 0)
            params = SamplingParams(temperature=temperature or 1, seed=17, max_tokens=tokens, ignore_eos=True)
            torch.cuda.synchronize()
            start = perf_counter()
            with _observe(plain, config, observation):
                output = plain.generate([prompt], params, use_tqdm=False)[0]
            torch.cuda.synchronize()
            seconds += perf_counter() - start
            continuation = tuple(output.outputs[0].token_ids)
            assert len(continuation) == observation.step == tokens
            references.append(_Reference(prompt, prompt_length, continuation, tuple(observation.log_probs)))
    return tuple(references), seconds


def _workload_distances(
    workload: Workload,
    config: ModelConfig,
    references: tuple[_Reference, ...],
) -> tuple[torch.Tensor, float]:
    distances: list[torch.Tensor] = []
    seconds = 0.0
    with pllm.make_llm(config, workload=workload) as handle:
        for index, reference in enumerate(references):
            forced = torch.tensor(reference.continuation, dtype=torch.int64, device="cuda")
            observation = _Observation(reference=reference.log_probs, forced=forced)
            params = SamplingParams(temperature=1, seed=17, max_tokens=len(reference.continuation), ignore_eos=True)
            torch.cuda.synchronize()
            start = perf_counter()
            with _observe(handle, config, observation):
                output = handle.generate([reference.prompt], params, use_tqdm=False, prompt_indices=(index,))[0]
            torch.cuda.synchronize()
            seconds += perf_counter() - start
            assert tuple(output.outputs[0].token_ids) == reference.continuation
            distances.extend(observation.distances)
    return torch.cat(distances), seconds


def _failures(
    summaries: tuple[DistanceSummary, ...],
    max_mean_kl: float | None,
    max_p99_kl: float | None,
) -> tuple[str, ...]:
    failures: list[str] = []
    for name, summary in zip(("KL(plain || workload)", "KL(workload || plain)", "TV"), summaries, strict=True):
        if any(value != value for value in (summary.mean, summary.p99, summary.max)):
            failures.append(f"{name} contains NaN; check the workload's logits")
        if name != "TV":
            for statistic, value, limit in (("mean", summary.mean, max_mean_kl), ("P99", summary.p99, max_p99_kl)):
                if limit is not None and value > limit:
                    failures.append(f"{name} {statistic} {value:.6g} exceeds {limit:.6g}")
    return tuple(failures)


def check_distribution(
    workload: Workload,
    config: ModelConfig,
    *,
    prompts: Sequence[str] | None = None,
    tokens: int = 16,
    temperature: float = 0,
    max_mean_kl: float | None = None,
    max_p99_kl: float | None = None,
    worst: int = 5,
) -> DistributionResult:
    """Measure how far a workload moves the plain model's output distribution on CUDA.

    A dense, cheap quality signal complements the bitwise neutral contract: a workload can look
    fine on average while breaking rare token positions. P99 exposes that tail; a mean alone can
    hide it. This measures movement, not task quality, and only covers the supplied sequences.

    The plain model generates `tokens` greedy (temperature=0) or seeded sampled continuation
    tokens for each prompt (check_on_gpu's two sequence prompts by default). The workload then
    teacher-forces those same tokens through vLLM's native sampler, including sampler feedback
    and state updates. Prompts are prefills; measured predictions start at the prompt end and
    cover the continuation, not every interior prompt position. Requests run one at a time so
    both engines have identical batch geometry. Each prompt must fit max_num_batched_tokens,
    and prompt + tokens must fit max_model_len. Both engines use the workload's checkpoint and
    graph/compile settings, including eager fallback. Run in a fresh process; multiprocessing
    and parallel workers are unsupported. Every engine is closed, including on failure.

    Both KL directions (nats) and TV are reduced over the FULL vocabulary on device in FP32,
    using torch log_softmax on native logits. Distances use temperature=1 with no top-k/top-p
    truncation; `temperature` only selects the plain continuation. The workload's configured
    bounded/dense guidance is preserved. Zero probability contributes zero to KL; support
    removal can give infinite KL. No diagnostic host reads/syncs occur inside model steps;
    synchronization happens at run boundaries and when fetching final summaries. Baseline
    log probabilities stay on device (about prompts * tokens * vocabulary * 4 bytes).

    Returns mean, nearest-rank P99, max and `worst` positions for each metric, plus timings.
    Positions are (prompt_index, token_offset), zero-based in prompt + continuation, naming
    the predicted token. max_mean_kl / max_p99_kl bound BOTH KL directions. With no thresholds
    this is a measurement; supplied thresholds set result.passed and result.failures without
    discarding the numbers or raising ContractError. NaN distances always fail.
    """
    if (
        config.multiprocessing
        or max(config.tensor_parallel_size, config.pipeline_parallel_size, config.data_parallel_size) > 1
    ):
        raise ValueError("check_distribution needs multiprocessing=False and all parallel sizes=1")
    texts = tuple(_PROMPTS if prompts is None else prompts)
    if not texts or any(not prompt for prompt in texts) or tokens < 1 or worst < 1:
        raise ValueError("check_distribution needs nonempty prompts, tokens >= 1 and worst >= 1")
    for name, value in (("temperature", temperature), ("max_mean_kl", max_mean_kl), ("max_p99_kl", max_p99_kl)):
        if value is not None and (not isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and >= 0")
    start = perf_counter()
    _release_models()
    try:
        references, plain_seconds = _plain_continuations(workload, config, texts, tokens, temperature)
        _release_models()
        values, workload_seconds = _workload_distances(workload, config, references)
        positions = [
            (index, reference.prompt_length + token)
            for index, reference in enumerate(references)
            for token in range(tokens)
        ]
        summaries = _summaries(values, positions, worst)
    finally:
        _release_models()
    forward, reverse, tv = summaries
    return DistributionResult(
        forward,
        reverse,
        tv,
        len(positions),
        plain_seconds + workload_seconds,
        perf_counter() - start,
        _failures(summaries, max_mean_kl, max_p99_kl),
    )
