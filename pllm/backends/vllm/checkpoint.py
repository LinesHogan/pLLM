"""Register vLLM's ordinary loader with exact exclusions for declared checkpoint state."""

from collections.abc import Generator
from dataclasses import replace
from typing import Any

import torch
from cattrs import structure
from vllm.config import LoadConfig, ModelConfig
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from pllm.execution.checkpoint import checkpoint_directory, checkpoint_files


def load_options(names: tuple[str, ...]) -> dict[str, str | dict[str, tuple[str, ...]]]:
    """Carry workload exclusions through vLLM's serialized loader configuration."""
    return {"load_format": "pllm", "model_loader_extra_config": {"checkpoint_names": names}} if names else {}


@register_model_loader("pllm")
class CheckpointLoader(DefaultModelLoader):
    """Load every base tensor normally, omitting only names the workload owns."""

    def __init__(self, load_config: LoadConfig) -> None:
        options: dict[str, Any] = dict(load_config.model_loader_extra_config)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        self.excluded = frozenset(structure(options.pop("checkpoint_names"), tuple[str, ...]))
        super().__init__(replace(load_config, load_format="auto", model_loader_extra_config=options))

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
