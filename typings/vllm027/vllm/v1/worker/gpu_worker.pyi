"""Strict tLLM view of the vLLM 0.27.1 MRv2 worker construction seam."""

import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu_model_runner import GPUModelRunner as Mrv1GPUModelRunner
from vllm.v1.worker.worker_base import WorkerBase


class Worker(WorkerBase):
    vllm_config: VllmConfig
    device: torch.device
    use_v2_model_runner: bool
    model_runner: GPUModelRunner | Mrv1GPUModelRunner

    def init_device(self) -> None: ...
