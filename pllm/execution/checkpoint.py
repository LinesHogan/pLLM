"""Read declared frozen state from safetensors checkpoints for state initialization."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from cattrs import structure


@dataclass(frozen=True)
class _Index:
    weight_map: dict[str, str]


def checkpoint_directory(model: str, revision: str | None = None) -> str:
    """Resolve the local snapshot that the model loader has already downloaded."""
    from huggingface_hub import snapshot_download  # pyright: ignore[reportUnknownVariableType]

    return model if Path(model).is_dir() else snapshot_download(model, revision=revision, local_files_only=True)


def checkpoint_files(checkpoint: str, names: tuple[str, ...]) -> tuple[Path, ...]:
    """Validate declared names before vLLM can reject other, undeclared checkpoint tensors."""
    from safetensors import safe_open

    directory = Path(checkpoint)
    index = directory / "model.safetensors.index.json"
    shards = structure(json.loads(index.read_text()), _Index).weight_map if index.is_file() else {}
    paths = tuple(directory / shards.get(name, "model.safetensors") for name in names)
    for name, path in zip(names, paths, strict=True):
        missing = f"tensor {name!r} is missing from checkpoint {checkpoint!r}; use a checkpoint containing it"
        if not checkpoint or not path.is_file():
            raise ValueError(missing)
        tensors = safe_open(path, framework="pt", device="cpu")
        with tensors:
            if name not in tensors.keys():
                raise ValueError(missing)
    return paths


def initialize_checkpoint(
    checkpoint: str, names: tuple[str, ...], root: tuple[torch.Tensor, ...], generator: torch.Generator
) -> None:
    """Copy the named tensors, rejecting mismatched shapes at the initialization boundary."""
    from safetensors import safe_open

    for name, destination, path in zip(names, root, checkpoint_files(checkpoint, names), strict=True):
        tensors = safe_open(path, framework="pt", device="cpu")
        with tensors:
            value = tensors.get_tensor(name)
        if value.shape != destination.shape:
            raise ValueError(
                f"tensor {name!r} in checkpoint {checkpoint!r} has shape {tuple(value.shape)}; adjust the declared field shape to match it"
            )
        destination.copy_(value)
