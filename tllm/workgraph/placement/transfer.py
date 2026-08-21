#!/usr/bin/env python3
"""Typed transfer requests, moves, and device topology for WorkGraph placement."""

from __future__ import annotations
from dataclasses import dataclass
from typing import cast
from tllm.workgraph.placement.vocab import (
    CopyEngine,
    DeviceKind,
    MemoryKind,
    MoveKind,
    Placement,
    Residency,
    ValueKey,
    require_enum,
    require_instance,
    require_int,
    require_tuple_of,
)


@dataclass(frozen=True, slots=True)
class TransferRequest:
    value: ValueKey
    source: Residency
    destination: Residency
    byte_count: int = 0

    def __post_init__(self) -> None:
        value = require_instance(self.value, ValueKey, "TransferRequest.value")
        source = require_instance(self.source, Residency, "TransferRequest.source")
        destination = require_instance(
            self.destination, Residency, "TransferRequest.destination"
        )
        if source.value != value or destination.value != value:
            raise ValueError("transfer residencies must reference the requested value")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(
            self,
            "byte_count",
            require_int(self.byte_count, "TransferRequest.byte_count"),
        )
@dataclass(frozen=True, slots=True)
class Move:
    value: ValueKey
    source: Residency
    destination: Residency
    kind: MoveKind
    byte_count: int = 0
    asynchronous: bool = True

    def __post_init__(self) -> None:
        value = require_instance(self.value, ValueKey, "Move.value")
        source = require_instance(self.source, Residency, "Move.source")
        destination = require_instance(self.destination, Residency, "Move.destination")
        kind = require_enum(cast(object, self.kind), MoveKind, "Move.kind")
        if source.value != value or destination.value != value:
            raise ValueError("Move residencies must reference Move.value")
        if source.placement == destination.placement:
            raise ValueError("Move source and destination must differ")
        source_kind = source.placement.kind
        destination_kind = destination.placement.kind
        expected = (
            MoveKind.H2D
            if source_kind == DeviceKind.CPU and destination_kind == DeviceKind.CUDA
            else MoveKind.D2H
            if source_kind == DeviceKind.CUDA and destination_kind == DeviceKind.CPU
            else MoveKind.P2P
            if source_kind == DeviceKind.CUDA
            and destination_kind == DeviceKind.CUDA
            else None
        )
        if kind != expected:
            raise ValueError("Move kind does not match its typed placements")
        if not isinstance(cast(object, self.asynchronous), bool):
            raise TypeError("Move.asynchronous must be bool")
        if not self.asynchronous:
            raise ValueError("Move must be asynchronous")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "byte_count",
            require_int(self.byte_count, "Move.byte_count"),
        )

    @property
    def engine(self) -> CopyEngine:
        return CopyEngine(self.kind.value)
@dataclass(frozen=True, slots=True)
class DeviceTopology:
    peer_access: frozenset[tuple[int, int]]
    cuda_devices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        peers = cast(object, self.peer_access)
        if not isinstance(peers, frozenset):
            raise TypeError("DeviceTopology.peer_access must be frozenset")
        normalized: set[tuple[int, int]] = set()
        for edge in cast(frozenset[object], peers):
            if not isinstance(edge, tuple):
                raise TypeError("peer access entries must be (source, destination)")
            pair = cast(tuple[object, ...], edge)
            if len(pair) != 2:
                raise TypeError("peer access entries must be (source, destination)")
            source = require_int(pair[0], "peer source")
            destination = require_int(pair[1], "peer destination")
            if source == destination:
                raise ValueError("peer access cannot contain a self edge")
            normalized.add((source, destination))
        devices = tuple(
            require_int(device, "DeviceTopology.cuda_devices")
            for device in self.cuda_devices
        )
        if len(set(devices)) != len(devices):
            raise ValueError("DeviceTopology.cuda_devices must be unique")
        if not devices and normalized:
            devices = tuple(sorted({index for edge in normalized for index in edge}))
        if any(
            source not in devices or destination not in devices
            for source, destination in normalized
        ):
            raise ValueError("peer access references an undeclared CUDA device")
        object.__setattr__(self, "peer_access", frozenset(normalized))
        object.__setattr__(self, "cuda_devices", tuple(sorted(devices)))

    def supports_peer(self, source: Placement, destination: Placement) -> bool:
        if source.kind != DeviceKind.CUDA or destination.kind != DeviceKind.CUDA:
            return False
        if source.index is None or destination.index is None:
            return False
        return (source.index, destination.index) in self.peer_access

    def contains(self, placement: Placement) -> bool:
        placement = require_instance(placement, Placement, "DeviceTopology.placement")
        if placement.kind == DeviceKind.CPU:
            return True
        assert placement.index is not None
        return placement.index in self.cuda_devices
def compile_moves(
    requests: tuple[TransferRequest, ...],
    *,
    topology: DeviceTopology,
) -> tuple[Move, ...]:
    requests = require_tuple_of(requests, TransferRequest, "compile_moves.requests")
    topology = require_instance(topology, DeviceTopology, "compile_moves.topology")
    moves: list[Move] = []
    for request in requests:
        source = request.source
        destination = request.destination
        if not topology.contains(source.placement) or not topology.contains(
            destination.placement
        ):
            raise ValueError("transfer references a placement outside DeviceTopology")
        if source.placement == destination.placement:
            continue
        if (
            source.placement.kind == DeviceKind.CUDA
            and destination.placement.kind == DeviceKind.CUDA
        ):
            if topology.supports_peer(source.placement, destination.placement):
                moves.append(
                    Move(
                        request.value,
                        source,
                        destination,
                        MoveKind.P2P,
                        request.byte_count,
                    )
                )
                continue
            host = Residency(
                request.value,
                Placement(DeviceKind.CPU),
                MemoryKind.PINNED_HOST,
            )
            moves.extend(
                (
                    Move(
                        request.value,
                        source,
                        host,
                        MoveKind.D2H,
                        request.byte_count,
                    ),
                    Move(
                        request.value,
                        host,
                        destination,
                        MoveKind.H2D,
                        request.byte_count,
                    ),
                )
            )
            continue
        kind = (
            MoveKind.H2D
            if source.placement.kind == DeviceKind.CPU
            else MoveKind.D2H
        )
        moves.append(
            Move(
                request.value,
                source,
                destination,
                kind,
                request.byte_count,
            )
        )
    return tuple(moves)
