#!/usr/bin/env python3
"""Producer-side port declaration."""

from __future__ import annotations

from dataclasses import dataclass

from tllm.ports.base import PortKind
from tllm.ports.port_id import PortId


@dataclass(frozen=True)
class PortSpec:
    """A port a producer offers.

    ``payload_type`` is a descriptive string (not an imported class) so
    producer modules stay cheap to import.
    """

    id: PortId
    kind: PortKind
    payload_type: str = ""
    description: str = ""
    writable: bool = False
    write_modes: tuple[str, ...] = ()
