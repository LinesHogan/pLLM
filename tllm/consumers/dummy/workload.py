"""Native ExecutionRegion declaration for the asynchronous Dummy CPU export."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import torch

from tllm.consumers.dummy.config import DummyConsumerConfig
from tllm.consumers.dummy.stream_runtime import DummyStreamRuntime
from tllm.consumers.dummy.transfer import clone_hidden_to_cpu
from tllm.consumers.dummy.worker import DummyCpuWorker
from tllm.kit import RegionAlgorithm
from tllm.ports.residual_stream import ResidualLocator
from tllm.workload import (
    Before,
    extern,
    hidden,
    step_id,
    task,
    workload,
)


class DummyNativeResources:
    """Process-local CPU worker and stream used by the typed extern Region."""

    def __init__(self, config: DummyConsumerConfig) -> None:
        self._config = config
        self._stream_runtime = DummyStreamRuntime(enable_async=bool(config.enable_async))
        self._worker = DummyCpuWorker()
        self._consumed_batches = 0
        self._consumed_rows = 0
        self._dropped_batches = 0

    def export_hidden(self, hidden_rows: torch.Tensor, *, step_id: int) -> None:
        """Stage one Region input without creating a dispatch-side consumer."""

        if int(step_id) % max(1, int(self._config.export_every_n_steps)) != 0:
            return
        self._consumed_batches += 1
        self._consumed_rows += int(hidden_rows.shape[0]) if hidden_rows.ndim else 0
        if not self._config.export_to_cpu:
            return
        export_rows = hidden_rows
        max_rows = int(self._config.export_max_rows)
        max_cols = int(self._config.export_max_cols)
        if max_rows > 0 and export_rows.ndim >= 1:
            export_rows = export_rows[:max_rows]
        if max_cols > 0 and export_rows.ndim >= 2:
            export_rows = export_rows[:, :max_cols]
        if self._config.enable_async:
            if self._worker.pending() >= int(self._config.max_queue_size):
                self._dropped_batches += 1
                return

            def submit() -> None:
                self._worker.enqueue(clone_hidden_to_cpu(export_rows))

            self._stream_runtime.run(submit)
            return
        self._worker.enqueue(clone_hidden_to_cpu(export_rows))
        self._worker.drain(limit=1, noise_std=self._config.noise_std)

    def synchronize(self) -> None:
        self._stream_runtime.synchronize()
        self._worker.drain(limit=0, noise_std=self._config.noise_std)

    def read_stats(self) -> Mapping[str, float]:
        result = {
            "consumed_batches": float(self._consumed_batches),
            "consumed_rows": float(self._consumed_rows),
            "dropped_batches": float(self._dropped_batches),
        }
        result.update(self._worker.stats())
        return MappingProxyType(result)


@workload(id="dummy")
class DummyWorkload(RegionAlgorithm[DummyConsumerConfig]):
    """A complete native singleton extern Region, with no legacy authority."""

    algorithm = "dummy-cpu-export-region-v1"
    config_type = DummyConsumerConfig

    def __init__(self, config: DummyConsumerConfig | None = None) -> None:
        super().__init__(config if config is not None else DummyConsumerConfig())

    def validate_config(self, config: DummyConsumerConfig) -> None:
        """Reject legacy-only scheduling semantics before native compilation."""
        if int(config.feedback_interval) > 0:
            raise ValueError(
                "native Dummy Region does not support feedback_interval > 0; "
                "use the direct legacy DummyConsumer for per-step feedback draining"
            )
        if config.capture_policy != "prefer":
            raise ValueError(
                "native Dummy Region does not support non-default capture_policy; "
                "its typed extern Region is explicitly eager-only"
            )

    def make_resources(self) -> DummyNativeResources:
        return DummyNativeResources(self.config)

    @property
    def resources(self) -> DummyNativeResources:
        return self._resources  # type: ignore[return-value]

    def diagnostics_reader(self) -> object:
        return self.resources.read_stats

    def read_stats(self) -> Mapping[str, float]:
        """Host-facing export counters (same pipe as the diagnostics reader)."""
        return self.resources.read_stats()

    def synchronize(self) -> None:
        self.resources.synchronize()

    def residual_paths(self) -> Mapping[object, str]:
        path = self.config.layer_filter or "model.layers.0"
        return {ResidualLocator(0, "block_output", "decode"): path}

    @task(
        inputs={
            "hidden_rows": hidden(0, "block_output", "decode"),
            "step_id": step_id(),
        },
        timing=Before("step.end"),
        implementation=extern("dummy:cpu_export", capabilities=("eager_only",)),
    )
    def export_hidden(self, hidden_rows: torch.Tensor, step_id: int) -> None:
        self.resources.export_hidden(hidden_rows, step_id=step_id)
