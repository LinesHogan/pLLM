#!/usr/bin/env python3
"""Contracts for dependency-light test-time trainer handoff."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Literal

from tllm.contracts.test_time_lora import LoRAScope, TestTimeLoRAUpdate

ObjectiveKind = Literal["feedback_lm", "grpo", "supervised_lm", "sft", "reward", "custom"]
_OBJECTIVE_KINDS = frozenset(("feedback_lm", "grpo", "supervised_lm", "sft", "reward", "custom"))

__test__ = False


def _require_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    return value


def _require_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    return value


def _require_real(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} values must be real")
    return float(value)


def _metadata_proxy(metadata: Mapping[str, object], *, field_name: str) -> Mapping[str, object]:
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized: dict[str, object] = {}
    for key, value in metadata.items():
        normalized[_require_str(key, field_name=f"{field_name} key")] = value
    return MappingProxyType(normalized)


def _str_tuple(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        normalized.append(_require_str(value, field_name=field_name))
    return tuple(normalized)


def _ensure_unique(values: tuple[str, ...], *, field_name: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {field_name}")
        seen.add(value)


@dataclass(frozen=True)
class TestTimeTrainingExample:
    """One prompt/sample/output training record for a test-time objective."""

    request_id: str
    prompt_id: str
    sample_id: str
    output_id: str
    objective_kind: ObjectiveKind
    token_ids: tuple[int, ...]
    labels: tuple[int, ...] | None = None
    rewards: tuple[float, ...] | None = None
    logprobs: tuple[float, ...] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        request_id = _require_str(self.request_id, field_name="request_id")
        prompt_id = _require_str(self.prompt_id, field_name="prompt_id")
        sample_id = _require_str(self.sample_id, field_name="sample_id")
        output_id = _require_str(self.output_id, field_name="output_id")
        if self.objective_kind not in _OBJECTIVE_KINDS:
            raise ValueError(f"objective_kind must be one of {sorted(_OBJECTIVE_KINDS)}")
        token_ids: list[int] = []
        for token_id in self.token_ids:
            token_id = _require_int(token_id, field_name="token_ids must contain int")
            if token_id < 0:
                raise ValueError("token_ids must be non-negative")
            token_ids.append(token_id)
        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        labels = self._normalize_labels()
        rewards = self._normalize_reals(self.rewards, field_name="rewards")
        logprobs = self._normalize_reals(self.logprobs, field_name="logprobs")
        if logprobs is not None and len(logprobs) != len(token_ids):
            raise ValueError("logprobs length must match token_ids")
        metadata = _metadata_proxy(self.metadata, field_name="metadata")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "prompt_id", prompt_id)
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "output_id", output_id)
        object.__setattr__(self, "token_ids", tuple(token_ids))
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "logprobs", logprobs)
        object.__setattr__(self, "metadata", metadata)

    def _normalize_labels(self) -> tuple[int, ...] | None:
        if self.labels is None:
            return None
        labels: list[int] = []
        for label in self.labels:
            labels.append(_require_int(label, field_name="labels must contain int"))
        if len(labels) != len(self.token_ids):
            raise ValueError("labels length must match token_ids")
        return tuple(labels)

    @staticmethod
    def _normalize_reals(values: tuple[float, ...] | None, *, field_name: str) -> tuple[float, ...] | None:
        if values is None:
            return None
        normalized = tuple(_require_real(value, field_name=field_name) for value in values)
        if not normalized:
            raise ValueError(f"{field_name} must be non-empty when provided")
        return normalized

    @property
    def identity(self) -> tuple[str, str, str, str]:
        return (self.request_id, self.prompt_id, self.sample_id, self.output_id)


@dataclass(frozen=True)
class TestTimeTrainingBatch:
    """A strict collection of training examples for one trainer request."""

    examples: tuple[TestTimeTrainingExample, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("TestTimeTrainingBatch requires examples")
        examples: list[TestTimeTrainingExample] = []
        identities: set[tuple[str, str, str, str]] = set()
        for example in self.examples:
            if not isinstance(example, TestTimeTrainingExample):
                raise TypeError("examples must contain TestTimeTrainingExample")
            if example.identity in identities:
                raise ValueError("duplicate training example identity")
            identities.add(example.identity)
            examples.append(example)
        metadata = _metadata_proxy(self.metadata, field_name="metadata")
        object.__setattr__(self, "examples", tuple(examples))
        object.__setattr__(self, "metadata", metadata)

    @property
    def request_ids(self) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []
        for example in self.examples:
            if example.request_id not in seen:
                seen.add(example.request_id)
                ordered.append(example.request_id)
        return tuple(ordered)


@dataclass(frozen=True)
class TestTimeTrainerRequest:
    """Runtime-owned trainer request independent of external trainer stacks."""

    algorithm_id: str
    adapter_key: str
    target_modules: tuple[str, ...]
    rank: int
    scope: LoRAScope
    request_ids: tuple[str, ...]
    examples: TestTimeTrainingBatch
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        algorithm_id = _require_str(self.algorithm_id, field_name="algorithm_id")
        adapter_key = _require_str(self.adapter_key, field_name="adapter_key")
        target_modules = _str_tuple(self.target_modules, field_name="target_modules")
        if not target_modules:
            raise ValueError("target_modules must be non-empty")
        rank = _require_int(self.rank, field_name="rank")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if self.scope not in ("global", "request"):
            raise ValueError("scope must be 'global' or 'request'")
        request_ids = _str_tuple(self.request_ids, field_name="request_id")
        _ensure_unique(request_ids, field_name="request_id")
        if self.scope == "request" and not request_ids:
            raise ValueError("request-scoped trainer request requires request_ids")
        if self.scope == "global" and request_ids:
            raise ValueError("global trainer request must not include request_ids")
        if not isinstance(self.examples, TestTimeTrainingBatch):
            raise TypeError("examples must be TestTimeTrainingBatch")
        if self.scope == "request" and tuple(self.examples.request_ids) != tuple(request_ids):
            raise ValueError("examples request_ids must match trainer request_ids")
        metadata = _metadata_proxy(self.metadata, field_name="metadata")
        object.__setattr__(self, "algorithm_id", algorithm_id)
        object.__setattr__(self, "adapter_key", adapter_key)
        object.__setattr__(self, "target_modules", tuple(target_modules))
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "request_ids", tuple(request_ids))
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class TestTimeTrainerResult:
    """Trainer output containing staged LoRA updates and scalar metrics."""

    updates: tuple[TestTimeLoRAUpdate, ...] = field(default_factory=tuple)
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        updates: list[TestTimeLoRAUpdate] = []
        for update in self.updates:
            if not isinstance(update, TestTimeLoRAUpdate):
                raise TypeError("updates must contain TestTimeLoRAUpdate")
            updates.append(update)
        metrics: dict[str, float] = {}
        if not isinstance(self.metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        for name, value in self.metrics.items():
            metrics[_require_str(name, field_name="metrics key")] = _require_real(value, field_name="metrics")
        object.__setattr__(self, "updates", tuple(updates))
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
