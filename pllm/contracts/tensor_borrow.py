"""Invocation-scoped indexed tensor storage owned by an external provider."""

from dataclasses import dataclass
from enum import StrEnum


class AccessMode(StrEnum):
    READ = "read"
    WRITE = "write"


@dataclass(frozen=True, slots=True)
class BorrowedResource:
    """External indexed storage requiring an invocation-scoped alias proof.

    The original backend owns its storage and recovery. A runtime binding must
    prove writable-cell separation and enforce every imported access interval.
    """

    name: str
    domain: tuple[int, ...]
