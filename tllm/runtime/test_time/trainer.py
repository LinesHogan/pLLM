#!/usr/bin/env python3
"""Dependency-light bridge for test-time training integrations."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from typing import Protocol

from tllm.contracts.test_time_lora import TestTimeLoRAUpdate
from tllm.contracts.test_time_training import TestTimeTrainerRequest, TestTimeTrainerResult
from tllm.runtime.test_time.lora import TestTimeLoRAService

__test__ = False


class TestTimeTrainerBridge(Protocol):
    """Protocol implemented by in-process or optional external trainers."""

    def train(self, request: TestTimeTrainerRequest) -> TestTimeTrainerResult:
        ...


def stage_trainer_result_updates(
    request: TestTimeTrainerRequest,
    result: TestTimeTrainerResult,
    lora_service: TestTimeLoRAService,
) -> int:
    """Stage all LoRA updates returned by a trainer result."""

    if not isinstance(request, TestTimeTrainerRequest):
        raise TypeError("request must be TestTimeTrainerRequest")
    if not isinstance(result, TestTimeTrainerResult):
        raise TypeError("result must be TestTimeTrainerResult")
    if not isinstance(lora_service, TestTimeLoRAService):
        raise TypeError("lora_service must be TestTimeLoRAService")
    for update in result.updates:
        _validate_update_matches_request(update, request)
    return lora_service.stage_updates(result.updates)


def _validate_update_matches_request(update: TestTimeLoRAUpdate, request: TestTimeTrainerRequest) -> None:
    if update.adapter_key != request.adapter_key:
        raise ValueError("trainer result update adapter_key does not match request")
    if update.scope != request.scope:
        raise ValueError("trainer result update scope does not match request")
    if update.rank != request.rank:
        raise ValueError("trainer result update rank does not match request")
    if tuple(update.request_ids) != tuple(request.request_ids):
        raise ValueError("trainer result update request_ids must match request")
    allowed_targets = set(request.target_modules)
    update_targets = set(update.target_modules)
    if not update_targets.issubset(allowed_targets):
        raise ValueError("trainer result update target_modules must be within request target_modules")


class InProcessTestTimeTrainerBridge:
    """Call a local trainer function and optionally stage returned LoRA updates."""

    def __init__(
        self,
        trainer: Callable[[TestTimeTrainerRequest], TestTimeTrainerResult],
        *,
        lora_service: TestTimeLoRAService | None = None,
        stage_updates: bool = True,
    ) -> None:
        if not callable(trainer):
            raise TypeError("trainer must be callable")
        if lora_service is not None and not isinstance(lora_service, TestTimeLoRAService):
            raise TypeError("lora_service must be TestTimeLoRAService")
        if not isinstance(stage_updates, bool):
            raise TypeError("stage_updates must be bool")
        self._trainer = trainer
        self._lora_service = lora_service
        self._stage_updates = stage_updates

    def train(self, request: TestTimeTrainerRequest) -> TestTimeTrainerResult:
        if not isinstance(request, TestTimeTrainerRequest):
            raise TypeError("request must be TestTimeTrainerRequest")
        result = self._trainer(request)
        if not isinstance(result, TestTimeTrainerResult):
            raise TypeError("trainer callable must return TestTimeTrainerResult")
        if self._stage_updates and self._lora_service is not None:
            stage_trainer_result_updates(request, result, self._lora_service)
        return result


class FakeTestTimeTrainerBridge:
    """Deterministic trainer bridge for unit tests and consumer smoke tests."""

    def __init__(
        self,
        *,
        result: TestTimeTrainerResult | None = None,
        results: Iterable[TestTimeTrainerResult] | None = None,
        lora_service: TestTimeLoRAService | None = None,
        stage_updates: bool = True,
    ) -> None:
        if result is not None and results is not None:
            raise ValueError("provide either result or results, not both")
        if lora_service is not None and not isinstance(lora_service, TestTimeLoRAService):
            raise TypeError("lora_service must be TestTimeLoRAService")
        if not isinstance(stage_updates, bool):
            raise TypeError("stage_updates must be bool")
        if results is None:
            results = (result if result is not None else TestTimeTrainerResult(),)
        self._results = deque(results)
        for trainer_result in self._results:
            if not isinstance(trainer_result, TestTimeTrainerResult):
                raise TypeError("results must contain TestTimeTrainerResult")
        self._lora_service = lora_service
        self._stage_updates = stage_updates
        self._requests: list[TestTimeTrainerRequest] = []

    @property
    def requests(self) -> tuple[TestTimeTrainerRequest, ...]:
        return tuple(self._requests)

    def train(self, request: TestTimeTrainerRequest) -> TestTimeTrainerResult:
        if not isinstance(request, TestTimeTrainerRequest):
            raise TypeError("request must be TestTimeTrainerRequest")
        if not self._results:
            raise RuntimeError("FakeTestTimeTrainerBridge has no remaining results")
        self._requests.append(request)
        result = self._results.popleft()
        if self._stage_updates and self._lora_service is not None:
            stage_trainer_result_updates(request, result, self._lora_service)
        return result
