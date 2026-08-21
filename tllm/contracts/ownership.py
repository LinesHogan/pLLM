#!/usr/bin/env python3
"""Buffer ownership vocabulary shared by port contracts and GPU leases."""

from __future__ import annotations

from enum import StrEnum


class BufferOwnership(StrEnum):
    BORROWED = "borrowed"
    RUNTIME_LEASE = "runtime_lease"
