#!/usr/bin/env python3
"""Typed consumer input views compiled from the compatibility PortBundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, cast, runtime_checkable

import torch

from tllm.contracts.gpu_stage import DeviceTensorLease
from tllm.contracts.port_bundle import PortBundle
from tllm.contracts.request_meta_view import RowBatchMeta
from tllm.runtime.generation_trace_runtime import GenerationTraceBatch
from tllm.runtime.reward_outcome_runtime import RewardOutcomeBatch
from tllm.runtime.retry_control_runtime import RetryControlBatch
from tllm.runtime.update_barrier_runtime import UpdateBarrierBatch

BatchT = TypeVar("BatchT")


@runtime_checkable
class AppendSink(Protocol, Generic[BatchT]):
    def append(self, batch: BatchT) -> int: ...


@dataclass(frozen=True, slots=True)
class ResidualPairInput:
    source: torch.Tensor
    target: torch.Tensor
    request_meta: RowBatchMeta | None
    uses_device_lease: bool


@dataclass(frozen=True, slots=True)
class PostGenerateInput:
    trace: GenerationTraceBatch
    reward_outcome: AppendSink[RewardOutcomeBatch]
    retry_control: AppendSink[RetryControlBatch] | None
    update_barrier: AppendSink[UpdateBarrierBatch] | None


def residual_pair_input(
    bundle: PortBundle,
) -> ResidualPairInput | None:
    lease = bundle.entries.get("device_lease")
    uses_device_lease = isinstance(lease, DeviceTensorLease)
    if uses_device_lease:
        source = lease.entries.get("source")
        target = lease.entries.get("target")
        active_rows = max(0, int(lease.active_rows))
        if isinstance(source, torch.Tensor) and active_rows < int(source.shape[0]):
            source = source[:active_rows]
        if isinstance(target, torch.Tensor) and active_rows < int(target.shape[0]):
            target = target[:active_rows]
    else:
        source = bundle.entries.get("source")
        target = bundle.entries.get("target")
    if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
        return None

    raw_meta = bundle.entries.get("request_meta")
    request_meta = raw_meta if isinstance(raw_meta, RowBatchMeta) else None
    return ResidualPairInput(
        source=source,
        target=target,
        request_meta=request_meta,
        uses_device_lease=uses_device_lease,
    )


def post_generate_input(bundle: PortBundle) -> PostGenerateInput:
    trace = bundle.entries.get("generation_trace")
    if not isinstance(trace, GenerationTraceBatch):
        raise TypeError("generation_trace entry must be GenerationTraceBatch")
    reward_sink = bundle.entries.get("reward_outcome")
    if not isinstance(reward_sink, AppendSink):
        raise TypeError("reward_outcome entry must provide append")
    retry_sink = bundle.entries.get("retry_control")
    if retry_sink is not None and not isinstance(retry_sink, AppendSink):
        raise TypeError("retry_control entry must provide append")
    barrier_sink = bundle.entries.get("update_barrier")
    if barrier_sink is not None and not isinstance(barrier_sink, AppendSink):
        raise TypeError("update_barrier entry must provide append")
    return PostGenerateInput(
        trace=trace,
        reward_outcome=cast(AppendSink[RewardOutcomeBatch], reward_sink),
        retry_control=cast(AppendSink[RetryControlBatch] | None, retry_sink),
        update_barrier=cast(AppendSink[UpdateBarrierBatch] | None, barrier_sink),
    )
