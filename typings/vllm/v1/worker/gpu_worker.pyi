"""Strict pllm view of the vLLM worker's runner construction seam.

0.32.0 builds the runner through _make_model_runner; on 0.27.1 pllm.backends.vllm.compat.Worker adds that factory.
"""

import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.worker_base import WorkerBase


class Worker(WorkerBase):
    vllm_config: VllmConfig
    device: torch.device
    model_runner: GPUModelRunner

    def init_device(self) -> None: ...
    def _make_model_runner(self) -> GPUModelRunner: ...
