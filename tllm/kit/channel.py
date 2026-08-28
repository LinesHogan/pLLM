"""Process-local host channels shared with reconstructed worker instances.

The engine never executes the host's workload object: real runs drive an
instance reconstructed from the signed worker spec, because callables
cannot cross the JSON control plane (this holds even in-process).  State
the host pushes between steps — staged training rows, published
intervention weights, publish targets — must therefore live in a
process-local registry that both the host object and the executing
reconstruction can reach.  This is the same seam as
``residual_runtime.RUNTIME``.

Scope honesty: under mp=1 each worker process has its own (empty)
registry, so channels are in-process transport only.  A signed
control-plane staging transport is the follow-up that closes the MP gap;
until then MP workers observe zero-row steps through the audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

P = TypeVar("P")


@dataclass(slots=True)
class HostChannelBox:
    """Single-slot mailbox; each consumer stores its own typed payload."""

    value: object | None = None


_CHANNELS: dict[str, HostChannelBox] = {}


def host_channel(key: str) -> HostChannelBox:
    """Return the process-local channel box for ``key`` (created on demand)."""
    if not isinstance(key, str) or not key:
        raise TypeError("host channel key must be a non-empty str")
    box = _CHANNELS.get(key)
    if box is None:
        box = HostChannelBox()
        _CHANNELS[key] = box
    return box


def typed_host_channel(key: str, payload_type: type[P]) -> P:
    """Return the channel payload for ``key``, created as ``payload_type()``.

    Owns the create-on-demand + type-check dance once so workloads never
    hand-write it around :func:`host_channel`.
    """
    box = host_channel(key)
    if box.value is None:
        box.value = payload_type()
    payload = box.value
    if not isinstance(payload, payload_type):
        raise TypeError(f"host channel {key!r} payload type mismatch")
    return payload


def clear_host_channels() -> None:
    """Drop every channel (test isolation hook; not for production paths)."""
    _CHANNELS.clear()


__all__ = ["HostChannelBox", "clear_host_channels", "host_channel", "typed_host_channel"]
