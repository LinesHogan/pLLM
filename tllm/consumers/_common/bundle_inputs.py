#!/usr/bin/env python3
"""Typed consumer input views compiled from the compatibility PortBundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, cast, runtime_checkable

import torch

from tllm.contracts.gpu_stage import DeviceTensorLease
from tllm.contracts.port_bundle import BundleKey, PortBundle
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


def provided_bundle(
    *,
    consumer_id: str,
    step_id: int,
    entries: dict[str, torch.Tensor],
) -> PortBundle:
    """Build a PortBundle carrying caller-supplied (``provided()``) tensors.

    Entry keys must equal the program's declared input names; every tensor is
    detached here so the program's grad mode starts from clean leaves.
    """
    rows = int(next(iter(entries.values())).shape[0])
    for name, value in entries.items():
        if int(value.shape[0]) != rows:
            raise RuntimeError(f"provided input {name!r} row count differs from the batch")
    return PortBundle(
        key=BundleKey(
            engine_step_id=int(step_id),
            phase="decode",
            request_id=consumer_id,
            sample_idx=0,
        ),
        entries={
            "device_lease": DeviceTensorLease(
                entries={name: value.detach() for name, value in entries.items()},
                active_rows=rows,
            ),
            "request_meta": RowBatchMeta(
                request_ids=tuple(f"{consumer_id}-{i}" for i in range(rows)),
                prompt_idxs=tuple(0 for _ in range(rows)),
                sample_idxs=tuple(0 for _ in range(rows)),
                phase="decode",
                engine_step_id=int(step_id),
            ),
        },
    )
