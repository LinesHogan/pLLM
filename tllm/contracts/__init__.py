#!/usr/bin/env python3
"""Producer-consumer data contracts."""

from tllm.contracts.hidden_batch import HiddenBatch
from tllm.contracts.gpu_stage import DeviceTensorLease
from tllm.contracts.request_meta_view import RowBatchMeta
from tllm.contracts.runtime_context import RuntimeContext
from tllm.contracts.subscription import ConsumerSubscription
from tllm.contracts.test_time_lora import TestTimeLoRATarget, TestTimeLoRAUpdate, TestTimeLoRAWeights
from tllm.contracts.test_time_training import (
    ObjectiveKind,
    TestTimeTrainerRequest,
    TestTimeTrainerResult,
    TestTimeTrainingBatch,
    TestTimeTrainingExample,
)

__all__ = [
    "ConsumerSubscription",
    "DeviceTensorLease",
    "HiddenBatch",
    "ObjectiveKind",
    "RowBatchMeta",
    "RuntimeContext",
    "TestTimeLoRATarget",
    "TestTimeLoRAUpdate",
    "TestTimeLoRAWeights",
    "TestTimeTrainerRequest",
    "TestTimeTrainerResult",
    "TestTimeTrainingBatch",
    "TestTimeTrainingExample",
]
