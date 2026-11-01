#!/usr/bin/env python3
"""Stable request identity resolved at the vLLM/private-API boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RequestKeyField(StrEnum):
    """Closed fields that can identify native request-scoped state."""

    REQUEST_ID = "request_id"
    PROMPT_INDEX = "prompt_index"
    SAMPLE_INDEX = "sample_index"
    ROUTE = "route"


def ordered_key_fields(
    fields: tuple[RequestKeyField, ...],
) -> tuple[RequestKeyField, ...]:
    """Return one exact canonical request-key declaration or reject it."""

    canonical = tuple(
        field
        for field in (
            RequestKeyField.REQUEST_ID,
            RequestKeyField.PROMPT_INDEX,
            RequestKeyField.SAMPLE_INDEX,
            RequestKeyField.ROUTE,
        )
        if field in fields
    )
    if not fields or fields != canonical:
        raise ValueError("request key fields must be closed, sorted, and unique")
    return fields


@dataclass(frozen=True, slots=True)
class RequestIdentity:
    request_id: str
    prompt_index: int
    sample_index: int
    # User-computed routing value (regex, discriminator, tenant id); the framework
    # signs the key declaration and never interprets the value.
    route: str = ""
