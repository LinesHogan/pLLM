"""Asynchronous CPU export workload."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

import torch

from tllm.consumers.dummy.stream_runtime import DummyStreamRuntime
from tllm.consumers.dummy.transfer import clone_hidden_to_cpu
from tllm.consumers.dummy.worker import DummyCpuWorker
from tllm.kit.workload import Workload
from tllm.workload import (
    Before,
    NumericConstraint,
    WorkloadConfig,
    config_field,
    extern,
    hidden,
    step_id,
    task,
    workload,
)


class DummyConfig(WorkloadConfig):
    enabled: bool = config_field(default=True)
    enable_async: bool = config_field(default=True)
    export_to_cpu: bool = config_field(default=True)
    max_queue_size: int = config_field(
        default=4096, constraint=NumericConstraint.POSITIVE
    )
    noise_std: float = config_field(
        default=1e-3, constraint=NumericConstraint.NONNEGATIVE
    )
    export_every_n_steps: int = config_field(
        default=256, constraint=NumericConstraint.POSITIVE
    )
    export_max_rows: int = config_field(
        default=1, constraint=NumericConstraint.NONNEGATIVE
    )
    export_max_cols: int = config_field(
        default=16, constraint=NumericConstraint.NONNEGATIVE
    )
    layer_path: str = config_field(default="model.layers.0")


class DummyExporter:
    def __init__(self, config: DummyConfig) -> None:
        self.config = config
        self.stream = DummyStreamRuntime(enable_async=config.enable_async)
        self.worker = DummyCpuWorker()
        self.consumed_batches = 0
        self.consumed_rows = 0
        self.dropped_batches = 0

    def export(self, hidden_rows: torch.Tensor, engine_step: int) -> None:
        if engine_step % self.config.export_every_n_steps != 0:
            return
        self.consumed_batches += 1
        self.consumed_rows += hidden_rows.shape[0]
        if not self.config.export_to_cpu:
            return
        row_limit = self.config.export_max_rows or hidden_rows.shape[0]
        column_limit = self.config.export_max_cols or hidden_rows.shape[1]
        rows = hidden_rows[:row_limit, :column_limit]
        if self.config.enable_async:
            if self.worker.pending() >= self.config.max_queue_size:
                self.dropped_batches += 1
                return

            self.stream.run(lambda: self.worker.enqueue(clone_hidden_to_cpu(rows)))
            return
        self.worker.enqueue(clone_hidden_to_cpu(rows))
        self.worker.drain(limit=1, noise_std=self.config.noise_std)

    def synchronize(self) -> None:
        self.stream.synchronize()
        self.worker.drain(limit=0, noise_std=self.config.noise_std)

    def diagnostics(self) -> Mapping[str, float]:
        values = {
            "consumed_batches": float(self.consumed_batches),
            "consumed_rows": float(self.consumed_rows),
            "dropped_batches": float(self.dropped_batches),
        }
        values.update(self.worker.stats())
        return MappingProxyType(values)


@workload(id="dummy", config=DummyConfig, enabled=DummyConfig.enabled)
class DummyWorkload(Workload[DummyConfig]):
    def __init__(self, config: DummyConfig = DummyConfig()) -> None:
        self.exporter = DummyExporter(config)
        super().__init__(config)
        self._resources = replace(
            self._resources,
            diagnostics_reader=self.exporter.diagnostics,
            synchronizer=self.exporter.synchronize,
        )

    @task(
        inputs={
            "hidden_rows": hidden(
                0,
                "block_output",
                "decode",
                path=DummyConfig.layer_path,
            ),
            "step_id": step_id(),
        },
        timing=Before("step.end"),
        implementation=extern("dummy:cpu_export", capabilities=("eager_only",)),
    )
    def export_hidden(self, hidden_rows: torch.Tensor, step_id: int) -> None:
        self.exporter.export(hidden_rows, step_id)


__all__ = ["DummyConfig", "DummyExporter", "DummyWorkload"]
