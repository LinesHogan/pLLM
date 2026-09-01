"""vLLM worker entrypoint that constructs the clean-room MRv2 runner."""

from __future__ import annotations

from tllm.canonical_runtime.runner import CanonicalGPUModelRunner

from vllm.v1.worker.gpu_worker import Worker


class CanonicalGPUWorker(Worker):
    """Single-process GPU worker with a canonical runner before model load."""

    def init_device(self) -> None:
        super().init_device()
        if not self.use_v2_model_runner:
            raise RuntimeError("canonical vLLM requires the MRv2 model runner")
        self.model_runner = CanonicalGPUModelRunner(
            self.vllm_config,
            self.device,
        )


__all__ = ["CanonicalGPUWorker"]
