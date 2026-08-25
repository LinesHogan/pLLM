#!/usr/bin/env python3
"""Composite request identity and auxiliary state share ProgramStateBank."""

from __future__ import annotations

import pytest
import torch

from tllm.program import StateSpec
from tllm.program.state import ProgramStateBank, group_rows_by_request_key
from tllm.workgraph.frontend import (
    RequestKeyField,
    StateScope,
    request_sample_key,
    request_state_key,
)


def test_composite_request_sample_groups_preserve_first_appearance() -> None:
    groups = group_rows_by_request_key(
        request_ids=("a", "b", "a", "a"),
        prompt_idxs=(0, 1, 0, 0),
        sample_idxs=(1, 0, 1, 2),
        key_spec=request_sample_key(),
    )
    assert groups == (
        (("a", 1), (0, 2)),
        (("b", 0), (1,)),
        (("a", 2), (3,)),
    )


def test_state_spec_accepts_typed_composite_key() -> None:
    spec = StateSpec(
        scope=StateScope.REQUEST,
        key=request_state_key(
            RequestKeyField.REQUEST_ID,
            RequestKeyField.SAMPLE_IDX,
        ),
    )
    assert spec.key == request_sample_key()


def test_aux_buffers_are_versioned_limited_and_clearable() -> None:
    bank = ProgramStateBank(
        "fast",
        StateScope.REQUEST,
        {},
        key_spec=request_sample_key(),
        max_slots=1,
        max_bytes=16,
    )
    key = ("request-a", 0)
    slot = bank.slot_for_key(key)
    assert bank.publish_buffers(slot, {"tail": torch.ones(2)}) == 1
    assert bank.estimated_bytes == 8
    assert bank.keys() == (key,)
    with pytest.raises(RuntimeError, match="slot limit"):
        bank.slot_for_key(("request-b", 0))
    with pytest.raises(RuntimeError, match="byte limit"):
        bank.publish_buffers(slot, {"tail": torch.ones(8)})
    bank.clear(key)
    assert bank.keys() == ()


def test_empty_request_identity_fails_pg180() -> None:
    with pytest.raises(RuntimeError, match="PG180"):
        group_rows_by_request_key(
            request_ids=("",),
            prompt_idxs=(0,),
            sample_idxs=(0,),
            key_spec=request_sample_key(),
        )
