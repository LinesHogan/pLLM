"""Read declared frozen state from safetensors checkpoints for state initialization."""

import json
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path

import torch
from cattrs import structure


@dataclass(frozen=True)
class _Index:
    weight_map: dict[str, str]


def checkpoint_directory(model: str, revision: str | None = None) -> str:
    """Resolve the local snapshot that the model loader has already downloaded."""
    from huggingface_hub import snapshot_download  # pyright: ignore[reportUnknownVariableType]

    # Only the files read here need to be cached: the loader downloads weights and configs, not README.md.
    return (
        model
        if Path(model).is_dir()
        else snapshot_download(
            model, revision=revision, local_files_only=True, allow_patterns=["*.safetensors", "*.json"]
        )
    )


def checkpoint_files(checkpoint: str, names: tuple[str, ...], *, state: str = "") -> tuple[Path, ...]:
    """Validate declared names before vLLM can reject other, undeclared checkpoint tensors."""
    from safetensors import safe_open

    directory = Path(checkpoint)
    index = directory / "model.safetensors.index.json"
    shards = structure(json.loads(index.read_text()), _Index).weight_map if index.is_file() else {}
    paths = tuple(
        directory if directory.is_file() else directory / shards.get(name, "model.safetensors") for name in names
    )
    keys_by_path: dict[Path, frozenset[str]] = {}
    ordered_keys_by_path: dict[Path, tuple[str, ...]] = {}
    for name, path in zip(names, paths, strict=True):
        missing = f"tensor {name!r} is missing from checkpoint {checkpoint!r}; use a checkpoint containing it"
        if not checkpoint or not path.is_file():
            raise ValueError(missing)
        keys = keys_by_path.get(path)
        if keys is None:
            with safe_open(path, framework="pt", device="cpu") as tensors:
                ordered = tuple(tensors.keys())
            keys = frozenset(ordered)
            keys_by_path[path] = keys
            ordered_keys_by_path[path] = ordered
        if name not in keys:
            raise ValueError(
                f"{checkpoint} has no tensor {name!r} for state {state!r}; it holds {ordered_keys_by_path[path]}. "
                "Name the fields like the saved tensors, or save the tensors under the field names"
                if state
                else missing
            )
    return paths


def initialize_checkpoint(
    checkpoint: str,
    names: tuple[str, ...],
    root: tuple[torch.Tensor, ...],
    generator: torch.Generator,
    *,
    state: str = "",
) -> None:
    """Copy the named tensors, rejecting mismatched shapes at the initialization boundary."""
    values = checkpoint_tensors(checkpoint, names, state=state)
    for name, destination in zip(names, root, strict=True):
        value = values[name]
        if value.shape != destination.shape:
            raise ValueError(
                f"tensor {name!r} in checkpoint {checkpoint!r} has shape {tuple(value.shape)}; adjust the declared field shape to match it"
            )
        destination.copy_(value)


def checkpoint_tensors(checkpoint: str, names: tuple[str, ...], *, state: str = "") -> dict[str, torch.Tensor]:
    """Read named tensors from one safetensors file or a sharded model snapshot."""
    from safetensors import safe_open

    paths = checkpoint_files(checkpoint, names, state=state)
    values: dict[str, torch.Tensor] = {}
    with ExitStack() as stack:
        handles = {
            path: stack.enter_context(safe_open(path, framework="pt", device="cpu")) for path in dict.fromkeys(paths)
        }
        for name, path in zip(names, paths, strict=True):
            values[name] = handles[path].get_tensor(name)
    return values


@dataclass(frozen=True)
class PretrainedFile:
    """A state weight file and the identity its declaration signed; used again when binding."""

    path: str
    state: str
    sha256: str

    @classmethod
    def read(cls, path: str, state: str) -> "PretrainedFile":
        """Identify a safetensors file at the authoring boundary, before an engine starts."""
        if not Path(path).is_file():
            raise ValueError(
                f"pllm.Pretrained({path!r}) for state {state!r} does not exist; "
                "save it with safetensors.torch.save_file before make_llm"
            )
        if Path(path).suffix != ".safetensors":
            raise ValueError("pllm.Pretrained needs a .safetensors file; save it with safetensors.torch.save_file")
        with open(path, "rb") as stream:
            return cls(path, state, file_digest(stream, "sha256").hexdigest())

    def verify(self) -> None:
        """Reject rewritten weights rather than run values absent from the workload identity."""
        if self != self.read(self.path, self.state):
            raise ValueError(f"{self.path} changed after make_llm hashed it; do not rewrite it while the engine starts")

    @property
    def source(self) -> str:
        """The exact weight provenance displayed by explain."""
        return f"pretrained {self.path} sha256 {self.sha256} ({Path(self.path).stat().st_size} bytes)"

    def initialize(
        self, names: tuple[str, ...], fields: Mapping[str, torch.Tensor], generator: torch.Generator
    ) -> None:
        """Seed logical field views using the same loader as base checkpoint states."""
        self.verify()
        initialize_checkpoint(self.path, names, tuple(fields[name] for name in names), generator, state=self.state)

    def module_values(self) -> dict[str, torch.Tensor]:
        """Read the complete state dict so strict module loading also rejects unexpected names."""
        from safetensors import safe_open

        self.verify()
        with safe_open(self.path, framework="pt", device="cpu") as tensors:
            names = tuple(tensors.keys())
        return checkpoint_tensors(self.path, names, state=self.state)
