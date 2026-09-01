"""vLLM worker entrypoint for the canonical parameter-overlay runner."""

from __future__ import annotations

from tllm.canonical_runtime.parameter_overlay_runner import (
    CanonicalParameterOverlayGPUModelRunner,
)

from vllm.v1.worker.gpu_worker import Worker


class CanonicalParameterOverlayGPUWorker(Worker):
    def init_device(self) -> None:
        super().init_device()
        if not self.use_v2_model_runner:
            raise RuntimeError("canonical parameter overlay requires MRv2")
        self.model_runner = CanonicalParameterOverlayGPUModelRunner(
            self.vllm_config,
            self.device,
        )


__all__ = ["CanonicalParameterOverlayGPUWorker"]
