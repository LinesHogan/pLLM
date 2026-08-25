"""Native ExecutionRegion declaration for the asynchronous Dummy CPU export."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import torch

from tllm.common.dataclass_json import dataclass_from_dict, dataclass_to_jsonable
from tllm.consumers.dummy.config import DummyConsumerConfig
from tllm.consumers.dummy.stream_runtime import DummyStreamRuntime
from tllm.consumers.dummy.transfer import clone_hidden_to_cpu
from tllm.consumers.dummy.worker import DummyCpuWorker
from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime.workload_plugins import (
    NativeWorkload,
    NativeWorkloadResources,
    WorkloadWorkerSpec,
)
from tllm.workload import (
    Before,
    ExternImplementation,
    hidden,
    step_id,
    task,
    workload,
)


def _hidden_path(config: DummyConsumerConfig) -> str:
    """Resolve the one explicit native tap path from the declared config."""

    return "model.layers.0" if config.layer_filter is None else config.layer_filter


def _validate_native_config(config: DummyConsumerConfig) -> None:
    """Reject legacy-only scheduling semantics before native compilation."""

    if not isinstance(config, DummyConsumerConfig):
        raise TypeError("config must be DummyConsumerConfig")
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


def _specialization_key(config: DummyConsumerConfig) -> Mapping[str, str]:
    """Immutable identity for the CPU-export behavior bound to this Region."""

    return MappingProxyType(
        {
            "algorithm": "dummy-cpu-export-region-v1",
            "async": str(bool(config.enable_async)),
            "export": str(bool(config.export_to_cpu)),
            "stride": str(int(config.export_every_n_steps)),
            "max_queue_size": str(int(config.max_queue_size)),
            "max_rows": str(int(config.export_max_rows)),
            "max_cols": str(int(config.export_max_cols)),
            "noise_std": float(config.noise_std).hex(),
            "feedback_interval": str(int(config.feedback_interval)),
            "hidden_path": _hidden_path(config),
        }
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
class DummyWorkload(NativeWorkload):
    """A complete native singleton extern Region, with no legacy authority."""

    def __init__(self, config: DummyConsumerConfig | None = None) -> None:
        self.config = config or DummyConsumerConfig()
        _validate_native_config(self.config)
        self._resources = DummyNativeResources(self.config)
        locator = ResidualLocator(layer=0, site="block_output", phase="decode")
        self._native_resources = NativeWorkloadResources(
            workload=self,
            diagnostics_reader=self._resources.read_stats,
            synchronizer=self._resources.synchronize,
            specialization_key=_specialization_key(self.config),
            residual_paths={locator: _hidden_path(self.config)},
            worker_spec=WorkloadWorkerSpec(
                factory="tllm.consumers.dummy.workload:reconstruct_dummy_workload",
                config=_config_to_wire(self.config),
            ),
        )

    @property
    def resources(self) -> DummyNativeResources:
        return self._resources

    def native_workload_resources(self) -> NativeWorkloadResources:
        return self._native_resources

    def read_stats(self) -> Mapping[str, float]:
        return self._resources.read_stats()

    @task(
        inputs={
            "hidden_rows": hidden(0, "block_output", "decode"),
            "step_id": step_id(),
        },
        timing=Before("step.end"),
        implementation=ExternImplementation(
            namespace="dummy",
            name="cpu_export",
            input_types=("Tensor", "int"),
            output_types=("None",),
            capabilities=("eager_only",),
        ),
    )
    def export_hidden(self, hidden_rows: torch.Tensor, step_id: int) -> None:
        self._resources.export_hidden(hidden_rows, step_id=step_id)


def _config_to_wire(config: DummyConsumerConfig) -> Mapping[str, object]:
    raw = dataclass_to_jsonable(config)
    if not isinstance(raw, dict):
        raise TypeError("Dummy config did not encode as a JSON object")
    return raw


def reconstruct_dummy_workload(raw: Mapping[str, object]) -> NativeWorkloadResources:
    config = dataclass_from_dict(DummyConsumerConfig, dict(raw))
    workload_instance = DummyWorkload(config)
    return workload_instance.native_workload_resources()


__all__ = [
    "DummyNativeResources",
    "DummyWorkload",
    "reconstruct_dummy_workload",
]
