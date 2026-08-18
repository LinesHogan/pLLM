#!/usr/bin/env python3
"""Developer-only timing-mobility cross-check support."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import StrEnum

import torch

from tllm.contracts.port_bundle import PortBundle


DEBUG_TIMING_MODE_ENV = "TLLM_DEBUG_TIMING_MODE"


class TimingDebugMode(StrEnum):
    OFF = "off"
    CROSS_CHECK = "cross_check"


@dataclass(frozen=True, slots=True)
class PayloadFingerprint:
    kind: str
    metadata: str
    data: bytes


@dataclass(frozen=True, slots=True)
class BundleFingerprint:
    version: int
    entries: tuple[tuple[str, PayloadFingerprint], ...]


def timing_debug_mode() -> TimingDebugMode:
    raw = os.environ.get(DEBUG_TIMING_MODE_ENV, "").strip().lower()
    if not raw:
        return TimingDebugMode.OFF
    try:
        return TimingDebugMode(raw)
    except ValueError as exc:
        raise ValueError(
            f"{DEBUG_TIMING_MODE_ENV} must be one of {[mode.value for mode in TimingDebugMode]}, got {raw!r}"
        ) from exc


def _payload_fingerprint(value: object) -> PayloadFingerprint:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous()
        raw = tensor.view(torch.uint8).cpu().numpy().tobytes()
        return PayloadFingerprint(
            kind="tensor",
            metadata=f"dtype={tensor.dtype};shape={tuple(tensor.shape)}",
            data=raw,
        )
    if is_dataclass(value) and not isinstance(value, type):
        return _payload_fingerprint(asdict(value))
    if isinstance(value, Mapping):
        items = tuple(
            (str(key), _payload_fingerprint(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
        return PayloadFingerprint(kind="mapping", metadata=repr(items), data=b"")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = tuple(_payload_fingerprint(item) for item in value)
        return PayloadFingerprint(kind="sequence", metadata=repr(items), data=b"")
    return PayloadFingerprint(kind=type(value).__qualname__, metadata=repr(value), data=b"")


def fingerprint_bundle(bundle: PortBundle) -> BundleFingerprint:
    return BundleFingerprint(
        version=int(bundle.version),
        entries=tuple(
            (str(name), _payload_fingerprint(payload))
            for name, payload in sorted(bundle.entries.items(), key=lambda pair: str(pair[0]))
        ),
    )


def assert_bundle_matches(expected: BundleFingerprint, actual: PortBundle, *, flow_id: str) -> None:
    observed = fingerprint_bundle(actual)
    if observed != expected:
        raise RuntimeError(
            f"timing cross-check failed for flow {flow_id!r}: declared-position "
            f"(port, version, bytes) differs from scheduled-position input"
        )
