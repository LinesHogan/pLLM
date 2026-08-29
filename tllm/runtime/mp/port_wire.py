"""Safe local-MP transport for typed workload port payloads."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
from multiprocessing.shared_memory import SharedMemory

import torch
from safetensors.torch import load, save

from tllm.common.dataclass_json import dataclass_from_dict, dataclass_to_jsonable


@dataclass(slots=True)
class SharedPortPayload:
    wire: dict[str, object]
    memory: SharedMemory

    def release(self) -> None:
        try:
            self.memory.close()
        finally:
            try:
                self.memory.unlink()
            except FileNotFoundError:
                pass


@dataclass(frozen=True, slots=True)
class InlinePortPayload:
    wire: dict[str, object]

    def release(self) -> None:
        pass


EncodedPortPayload = SharedPortPayload | InlinePortPayload


def _shared_tensor_wire(tensors: Mapping[str, torch.Tensor], kind: str) -> EncodedPortPayload:
    devices = {name: str(tensor.device) for name, tensor in tensors.items()}
    payload = save(
        {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}
    )
    memory = SharedMemory(create=True, size=len(payload))
    encoded = SharedPortPayload(
        wire={
            "kind": kind,
            "shared_memory": memory.name,
            "size": len(payload),
            "devices": devices,
        },
        memory=memory,
    )
    buffer = memory.buf
    try:
        if buffer is None:
            raise RuntimeError("shared-memory port payload has no writable buffer")
        buffer[: len(payload)] = payload
    except BaseException:
        try:
            if buffer is not None:
                buffer.release()
        finally:
            encoded.release()
        raise
    return encoded


def encode_port_payload(value: object) -> EncodedPortPayload:
    if isinstance(value, torch.Tensor):
        return _shared_tensor_wire({"value": value}, "tensor")
    if isinstance(value, Mapping) and all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in value.items()
    ):
        return _shared_tensor_wire(value, "tensor_map")
    if is_dataclass(value) and not isinstance(value, type):
        value_type = type(value)
        return InlinePortPayload(
            wire={
                "kind": "dataclass",
                "type": f"{value_type.__module__}:{value_type.__qualname__}",
                "json": json.dumps(
                    dataclass_to_jsonable(value),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    raise TypeError(f"unsupported staged payload type {type(value).__name__}")


def decode_port_payload(wire: Mapping[str, object]) -> object:
    kind = wire["kind"]
    if not isinstance(kind, str):
        raise TypeError("staged payload kind must be a string")
    if kind in ("tensor", "tensor_map"):
        memory_name = wire["shared_memory"]
        size = wire["size"]
        raw_devices = wire["devices"]
        if (
            not isinstance(memory_name, str)
            or not isinstance(size, int)
            or type(raw_devices) is not dict
        ):
            raise TypeError("shared staged payload metadata is invalid")
        devices: dict[str, torch.device] = {}
        for name, device_wire in raw_devices.items():
            if type(name) is not str or type(device_wire) is not str:
                raise TypeError("shared staged payload devices are invalid")
            device = torch.device(device_wire)
            if str(device) != device_wire:
                raise ValueError("shared staged payload device is not canonical")
            devices[name] = device
        memory = SharedMemory(name=memory_name)
        try:
            buffer = memory.buf
            if buffer is None:
                raise RuntimeError("shared staged payload has no readable buffer")
            cpu_tensors = load(bytes(buffer[:size]))
        finally:
            memory.close()
        if devices.keys() != cpu_tensors.keys():
            raise ValueError("shared staged payload device keys do not match tensors")
        tensors = {
            name: tensor.to(devices[name]) for name, tensor in cpu_tensors.items()
        }
        return tensors["value"] if kind == "tensor" else tensors
    if kind == "dataclass":
        type_path = wire["type"]
        payload_json = wire["json"]
        if not isinstance(type_path, str) or not isinstance(payload_json, str):
            raise TypeError("dataclass staged payload metadata is invalid")
        module_name, qualname = type_path.split(":", 1)
        value_type: object = importlib.import_module(module_name)
        for part in qualname.split("."):
            value_type = getattr(value_type, part)
        payload = json.loads(payload_json)
        return dataclass_from_dict(value_type, payload)  # type: ignore[arg-type]
    raise ValueError(f"unsupported staged payload kind {kind!r}")


__all__ = ["EncodedPortPayload", "decode_port_payload", "encode_port_payload"]
