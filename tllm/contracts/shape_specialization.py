"""Strict JSON contracts for cold Region shape specialization.

The request crosses the coordinator/worker boundary.  It therefore contains
only stable strings and integers, never tensors or framework runtime objects.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast


_REQUEST_VERSION = 1
_REQUEST_FIELDS = frozenset(
    {
        "version",
        "base_cache_key",
        "base_signature",
        "observed_plan_epoch",
        "region_id",
        "artifact_key",
        "inputs",
    }
)
_INPUT_FIELDS = frozenset({"binding_id", "shape", "dtype", "device"})


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty str")
    return value


def _sha256(value: object, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive int")
    return value


def _non_negative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"{label} must be a non-negative int")
    return value


def _version(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value != _REQUEST_VERSION:
        raise ValueError(f"unsupported Region shape specialization version: {value!r}")
    return value


def _json_object(pairs: list[tuple[object, object]]) -> dict[object, object]:
    result: dict[object, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _normalize_json(payload: str) -> str:
    """Delegate closed request semantics to the Rust control-plane core."""

    from tllm import _core

    return _core.normalize_region_shape_specialization_request(payload)


def _strict_mapping(value: object, label: str, fields: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    payload = cast(Mapping[object, object], value)
    if any(not isinstance(key, str) for key in payload):
        raise TypeError(f"{label} fields must be strings")
    observed = frozenset(cast(Mapping[str, object], payload))
    if observed != fields:
        missing = sorted(fields - observed)
        unexpected = sorted(observed - fields)
        raise ValueError(
            f"{label} has invalid fields: missing={missing}, unexpected={unexpected}"
        )
    return cast(Mapping[str, object], payload)


@dataclass(frozen=True, slots=True)
class RegionShapeSpecializationInput:
    """One signed Region input shape observed by a worker."""

    binding_id: str
    shape: tuple[int, ...]
    dtype: str
    device: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding_id", _text(self.binding_id, "binding_id"))
        raw_shape = cast(object, self.shape)
        if not isinstance(raw_shape, tuple):
            raise TypeError("shape must be tuple[int, ...]")
        shape = tuple(
            _non_negative_int(dimension, "shape dimension")
            for dimension in cast(tuple[object, ...], raw_shape)
        )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", _text(self.dtype, "dtype"))
        object.__setattr__(self, "device", _text(self.device, "device"))

    def to_wire(self) -> dict[str, object]:
        return {
            "binding_id": self.binding_id,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
        }

    @classmethod
    def from_wire(cls, payload: object) -> "RegionShapeSpecializationInput":
        fields = _strict_mapping(payload, "RegionShapeSpecializationInput", _INPUT_FIELDS)
        raw_shape = fields["shape"]
        if not isinstance(raw_shape, list):
            raise TypeError("RegionShapeSpecializationInput.shape must be an array")
        shape_items = cast(list[object], raw_shape)
        return cls(
            binding_id=_text(fields["binding_id"], "RegionShapeSpecializationInput.binding_id"),
            shape=cast(tuple[int, ...], tuple(shape_items)),
            dtype=_text(fields["dtype"], "RegionShapeSpecializationInput.dtype"),
            device=_text(fields["device"], "RegionShapeSpecializationInput.device"),
        )


@dataclass(frozen=True, slots=True)
class RegionShapeSpecializationRequest:
    """Canonical, JSON-only request for a new immutable Region epoch."""

    base_cache_key: str
    base_signature: str
    observed_plan_epoch: int
    region_id: str
    artifact_key: str
    inputs: tuple[RegionShapeSpecializationInput, ...]
    version: int = _REQUEST_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _version(cast(object, self.version)))
        object.__setattr__(self, "base_cache_key", _sha256(self.base_cache_key, "base_cache_key"))
        object.__setattr__(self, "base_signature", _sha256(self.base_signature, "base_signature"))
        object.__setattr__(
            self,
            "observed_plan_epoch",
            _positive_int(self.observed_plan_epoch, "observed_plan_epoch"),
        )
        object.__setattr__(self, "region_id", _text(self.region_id, "region_id"))
        object.__setattr__(self, "artifact_key", _sha256(self.artifact_key, "artifact_key"))
        raw_inputs = cast(object, self.inputs)
        if not isinstance(raw_inputs, tuple) or not raw_inputs:
            raise TypeError("inputs must be a non-empty tuple[RegionShapeSpecializationInput, ...]")
        inputs = cast(tuple[object, ...], raw_inputs)
        if any(not isinstance(item, RegionShapeSpecializationInput) for item in inputs):
            raise TypeError("inputs must contain RegionShapeSpecializationInput values")
        normalized_inputs = cast(tuple[RegionShapeSpecializationInput, ...], inputs)
        object.__setattr__(self, "inputs", normalized_inputs)
        binding_ids = tuple(item.binding_id for item in normalized_inputs)
        if binding_ids != tuple(sorted(binding_ids)) or len(binding_ids) != len(set(binding_ids)):
            raise ValueError("inputs must be uniquely ordered by binding_id")

    def to_wire(self) -> dict[str, object]:
        return {
            "version": self.version,
            "base_cache_key": self.base_cache_key,
            "base_signature": self.base_signature,
            "observed_plan_epoch": self.observed_plan_epoch,
            "region_id": self.region_id,
            "artifact_key": self.artifact_key,
            "inputs": [item.to_wire() for item in self.inputs],
        }

    def to_json(self) -> str:
        payload = json.dumps(
            self.to_wire(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return _normalize_json(payload)

    @property
    def dedup_key(self) -> str:
        return hashlib.sha256(self.to_json().encode("ascii")).hexdigest()

    @classmethod
    def from_wire(cls, payload: object) -> "RegionShapeSpecializationRequest":
        fields = _strict_mapping(payload, "RegionShapeSpecializationRequest", _REQUEST_FIELDS)
        raw_inputs = fields["inputs"]
        if not isinstance(raw_inputs, list):
            raise TypeError("RegionShapeSpecializationRequest.inputs must be an array")
        return cls(
            version=_version(fields["version"]),
            base_cache_key=_sha256(fields["base_cache_key"], "base_cache_key"),
            base_signature=_sha256(fields["base_signature"], "base_signature"),
            observed_plan_epoch=_positive_int(fields["observed_plan_epoch"], "observed_plan_epoch"),
            region_id=_text(fields["region_id"], "region_id"),
            artifact_key=_sha256(fields["artifact_key"], "artifact_key"),
            inputs=tuple(
                RegionShapeSpecializationInput.from_wire(item)
                for item in cast(list[object], raw_inputs)
            ),
        )

    @classmethod
    def from_json(cls, payload: object) -> "RegionShapeSpecializationRequest":
        if not isinstance(payload, str):
            raise TypeError("RegionShapeSpecializationRequest JSON must be str")
        try:
            # Keep duplicate detection at the JSON wire boundary. The Rust
            # core owns all closed field, scalar, and ordering semantics.
            json.loads(payload, object_pairs_hook=_json_object)
            decoded = json.loads(_normalize_json(payload), object_pairs_hook=_json_object)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid Region shape specialization JSON") from exc
        return cls.from_wire(decoded)


__all__ = ["RegionShapeSpecializationInput", "RegionShapeSpecializationRequest"]
