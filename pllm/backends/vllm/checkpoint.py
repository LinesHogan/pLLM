"""Register vLLM's ordinary loader with exact exclusions for declared checkpoint state."""

import os
from collections.abc import Generator
from dataclasses import replace

import torch
from vllm.config import LoadConfig, ModelConfig
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from pllm.contracts.codec import decode_boundary
from pllm.execution.checkpoint import checkpoint_directory, checkpoint_files

CHECKPOINT_ENV = "PLLM_CHECKPOINT_NAMES"


@register_model_loader("pllm")
class CheckpointLoader(DefaultModelLoader):
    """Load every base tensor normally, omitting only names the workload owns."""

    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(replace(load_config, load_format="auto"))
        self.excluded = frozenset(decode_boundary(os.environ[CHECKPOINT_ENV], tuple[str, ...]))

    def get_all_weights(
        self, model_config: ModelConfig, model: torch.nn.Module
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Retain vLLM's sharding, quantization and name mapping for all other tensors."""
        super().download_model(model_config)
        checkpoint = checkpoint_directory(model_config.model, model_config.revision)
        checkpoint_files(checkpoint, tuple(sorted(self.excluded)))
        for name, tensor in super().get_all_weights(model_config, model):
            if name not in self.excluded:
                yield name, tensor
