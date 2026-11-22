"""Host updates, feed transfers and completion rows exchanged between vLLM clients and workers."""

from __future__ import annotations

import io
import pickle
from base64 import b64decode, b64encode
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import prod
from traceback import format_exception
from typing import TypeVar

import torch
from vllm import SamplingParams

from pllm.contracts.host_batch import HostBatch, StateImport, map_host_batch
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.shared_tensor import SharedTensor
from pllm.execution.state_feed import FanoutFeed, Feed, decode_feed
from pllm.execution.step_driver import StepDriver, TrainStep

T = TypeVar("T")

CONTROL_KEY = "pllm.control_request"


def control_key(params: SamplingParams | None) -> str:
    """Read the staged host input key assigned to a child before EngineCore submission."""
    value = (params.extra_args or {}).get(CONTROL_KEY) if params is not None else None
    return value if isinstance(value, str) else ""


def reported_call(call: Callable[[], T]) -> str:
    """Return a result or a typed exception with its worker traceback to the client."""
    result: T | Exception
    try:
        result = call()
    except Exception as error:
        error.add_note("EngineCore worker traceback:\n" + "".join(format_exception(error)))
        result = error
    return b64encode(pickle.dumps(result)).decode("ascii")


def reported_result(reply: str, expected: type[T]) -> T:
    """Unwrap a trusted worker reply, preserving the type of a reported failure."""
    result = pickle.loads(b64decode(reply))
    if isinstance(result, Exception):
        raise result
    assert isinstance(result, expected)
    return result


@dataclass(frozen=True, slots=True)
class CpuTensor:
    """Contiguous CPU tensor bytes with the dtype and shape needed to reconstruct them."""

    dtype: TensorDType
    shape: tuple[int, ...]
    data: str

    @classmethod
    def of(cls, tensor: torch.Tensor) -> CpuTensor:
        spec = SharedTensor.describe(tensor)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return cls(spec.dtype, spec.shape, b64encode(raw).decode("ascii"))

    def tensor(self) -> torch.Tensor:
        """Reconstruct writable CPU storage from the wire representation."""
        raw = bytearray(b64decode(self.data, validate=True))
        if any(size < 0 for size in self.shape) or len(raw) != prod(self.shape) * self.dtype.itemsize:
            raise ValueError("control tensor byte count differs from its declared shape")
        if not raw:
            return torch.empty(self.shape, dtype=self.dtype.torch)
        return torch.frombuffer(raw, dtype=self.dtype.torch).reshape(self.shape)


@dataclass(frozen=True, slots=True)
class BackendImport:
    """Host field values and the optional route whose version they replace."""

    values: tuple[CpuTensor, ...]
    route: str = ""

    @classmethod
    def of(cls, update: StateImport[torch.Tensor], route: str = "") -> BackendImport:
        return cls(tuple(CpuTensor.of(tensor) for tensor in update.values), route)

    def update(self) -> StateImport[torch.Tensor]:
        """Restore an import ready for validation on the worker."""
        return StateImport(tuple(value.tensor() for value in self.values))


@dataclass(frozen=True, slots=True)
class BackendOnlineImport:
    """An import shared by these samples, or a cancellation of their staged input."""

    request_ids: tuple[str, ...]
    update: BackendImport | None


@dataclass(frozen=True, slots=True)
class BackendTrain:
    """A training batch with CPU-backed fields for transport to every rank."""

    batch: HostBatch[CpuTensor]

    @classmethod
    def of(cls, batch: HostBatch[torch.Tensor]) -> BackendTrain:
        encoded = map_host_batch(batch, CpuTensor.of)
        assert encoded is not None
        return cls(encoded)


@dataclass(frozen=True, slots=True)
class BackendCompletions:
    """Capture targets and prompt lengths. Online targets learn their prompt length at admission."""

    requests: tuple[tuple[str, int], ...]
    discard: bool = False


@dataclass(slots=True)
class CompletionRows:
    """The last sampled position of one request, updated only once its prompt is complete."""

    prompt_length: int
    request_id: str
    offset: int = -1
    hidden: torch.Tensor | None = None
    sampled_token: torch.Tensor | None = None

    def record(self, offset: int, hidden: torch.Tensor, token: torch.Tensor) -> None:
        if offset >= max(self.offset, self.prompt_length - 1):
            self.offset = offset
            self.hidden = hidden.detach().clone()
            self.sampled_token = token.detach().clone()

    @staticmethod
    def encode(rows: Sequence[CompletionRows]) -> bytes:
        """Copy final hidden and sampled token rows off device for the requesting client."""
        hidden: list[torch.Tensor] = []
        tokens: list[torch.Tensor] = []
        for row in rows:
            if row.hidden is None or row.sampled_token is None:
                raise ValueError("completion capture is missing a sampled row")
            hidden.append(row.hidden)
            tokens.append(row.sampled_token)
        buffer = io.BytesIO()
        torch.save((torch.stack(hidden).float().cpu(), torch.stack(tokens).cpu()), buffer)
        return buffer.getvalue()


def completion_hidden(replies: Sequence[str]) -> torch.Tensor:
    """Read the output rank's completion rows after checking every worker's reply."""
    payload, *_ = [reported_result(reply, bytes) for reply in replies]
    hidden, _tokens = torch.load(io.BytesIO(payload), weights_only=True)
    assert isinstance(hidden, torch.Tensor)
    return hidden


def run_import_state(driver: StepDriver, command: BackendImport) -> int:
    """Apply a worker import and acknowledge the published version."""
    return driver.import_state(command.update(), route=command.route)


def run_train_batch(driver: StepDriver, command: BackendTrain) -> TrainStep:
    """Return the worker's loss and parameters in CPU storage."""
    batch = map_host_batch(command.batch, CpuTensor.tensor)
    assert batch is not None
    result = driver.train_batch(batch)
    return TrainStep(
        None if result.loss is None else result.loss.detach().cpu(),
        {name: value.detach().cpu() for name, value in result.parameters.items()},
    )


def import_versions(replies: Sequence[str]) -> None:
    """Require every rank to acknowledge the same published version."""
    versions = [reported_result(reply, int) for reply in replies]
    assert len(set(versions)) == 1, "TP ranks published different import versions"


def train_result(replies: Sequence[str]) -> TrainStep:
    """Read the first rank's training result after checking worker errors."""
    result, *_ = (reported_result(reply, TrainStep) for reply in replies)
    return result


def unwrap_feeds(replies: Sequence[str]) -> Feed | FanoutFeed:
    """Claim each rank's producer endpoint exactly once."""
    feeds = tuple(decode_feed(reported_result(reply, str)) for reply in replies)
    return feeds[0] if len(feeds) == 1 else FanoutFeed(feeds)
