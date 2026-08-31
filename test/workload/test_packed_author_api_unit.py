"""Nominal authoring contracts for signed packed workload inputs."""

from __future__ import annotations

import pytest
import torch

from tllm.contracts.hidden_locator import HiddenPhase
from tllm.contracts.packed_rows import PackedPhaseRowBatch
from tllm.runtime.request_trainable_state import (
    RequestStateSelectionBatch,
)
from tllm.schedule import HiddenLayerKind
from tllm.workload import (
    AdamWSpec,
    PackedHiddenInput,
    PackedRequestRowsInput,
    RequestKeyField,
    State,
    StateSelectionInput,
    WorkloadConfig,
    declare_workload,
    definition_for_workload_class,
    packed_hidden,
    packed_request_rows,
    packed_selection,
    read,
    request_key,
    task,
    workload,
)
from tllm.workload.ir import ValueSourceKind


def test_packed_author_inputs_lower_nominal_same_task_relations() -> None:
    rows = packed_request_rows()

    @workload(id="author.packed")
    class PackedAuthor:
        weights = State.request_trainable(
            key=request_key(RequestKeyField.REQUEST_ID),
            optimizer=AdamWSpec.fixed(1e-3),
        )
        selection = packed_selection(
            weights.selection(), rows=rows, phase=HiddenPhase.DECODE
        )

        @task(
            inputs={
                "rows": rows,
                "source": packed_hidden(
                    0,
                    rows=rows,
                    path="model.layers.0",
                ),
                "target": packed_hidden(
                    -1,
                    rows=rows,
                    path="model.layers.7",
                ),
                "selection": selection,
            },
            effects=(read(selection),),
        )
        def train(
            self,
            rows: PackedPhaseRowBatch,
            source: torch.Tensor,
            target: torch.Tensor,
            selection: RequestStateSelectionBatch,
        ) -> None:
            _ = rows, source, target, selection

    declaration = declare_workload(PackedAuthor())
    task_ir = declaration.ir.tasks[0]
    assert declaration.ir.schema_version == 4
    assert tuple(item.kind for item in task_ir.inputs) == (
        ValueSourceKind.PACKED_REQUEST_ROWS,
        ValueSourceKind.PACKED_HIDDEN,
        ValueSourceKind.PACKED_HIDDEN,
        ValueSourceKind.STATE_SELECTION,
    )
    assert tuple(item.packed_rows_port for item in task_ir.inputs) == (
        "",
        "rows",
        "rows",
        "rows",
    )
    assert task_ir.inputs[1].layer == "0"
    assert task_ir.inputs[1].layer_kind == HiddenLayerKind.INDEX.value
    assert task_ir.inputs[1].site.value == "block_output"
    assert task_ir.inputs[1].phase.value == ""
    assert task_ir.inputs[-1].phase is HiddenPhase.DECODE
    assert task_ir.effects[0].selection_port == "selection"

    definition = definition_for_workload_class(PackedAuthor)
    assert definition is not None
    assert definition.hidden_inputs == ()
    assert tuple(item.layer for item in definition.packed_hidden_inputs) == (
        "0",
        "-1",
    )
    assert tuple(
        item.path.resolve(WorkloadConfig())
        for item in definition.packed_hidden_inputs
    ) == ("model.layers.0", "model.layers.7")


def test_packed_author_descriptors_are_nominal_public_types() -> None:
    rows: PackedRequestRowsInput = packed_request_rows()
    hidden: PackedHiddenInput = packed_hidden(0, rows=rows)
    state = State[torch.nn.Linear].request_trainable(
        key=request_key(RequestKeyField.REQUEST_ID),
        optimizer=AdamWSpec.fixed(1e-3),
    )
    selection: StateSelectionInput[torch.nn.Linear] = packed_selection(
        state.selection(), rows=rows, phase=HiddenPhase.DECODE
    )

    assert hidden.rows is rows
    assert selection.packed_rows is rows


def test_packed_author_rejects_missing_foreign_and_duplicate_rows() -> None:
    rows = packed_request_rows()
    foreign = packed_request_rows()

    with pytest.raises(ValueError, match="same-task packed request rows"):
        @task(inputs={"source": packed_hidden(0, rows=rows)})
        def missing(source: torch.Tensor) -> None:
            _ = source

    with pytest.raises(ValueError, match="same-task packed request rows"):
        @task(
            inputs={
                "rows": rows,
                "source": packed_hidden(0, rows=foreign),
            }
        )
        def foreign_rows(
            rows: PackedPhaseRowBatch, source: torch.Tensor
        ) -> None:
            _ = rows, source

    left = packed_request_rows()
    right = packed_request_rows()
    with pytest.raises(ValueError, match="one packed request rows"):
        @task(inputs={"left": left, "right": right})
        def duplicate(
            left: PackedPhaseRowBatch,
            right: PackedPhaseRowBatch,
        ) -> None:
            _ = left, right

    shared = packed_request_rows()

    @task(inputs={"rows": shared})
    def first(rows: PackedPhaseRowBatch) -> None:
        _ = rows

    @task(inputs={"rows": shared})
    def second(rows: PackedPhaseRowBatch) -> None:
        _ = rows


def test_packed_selection_effect_requires_exact_related_selection() -> None:
    rows = packed_request_rows()
    state = State[torch.nn.Linear].request_trainable(
        key=request_key(RequestKeyField.REQUEST_ID),
        optimizer=AdamWSpec.fixed(1e-3),
    )
    ordinary = state.selection()
    packed = packed_selection(
        ordinary, rows=rows, phase=HiddenPhase.DECODE
    )
    other_rows = packed_request_rows()

    with pytest.raises(ValueError, match="different packed request rows"):
        packed_selection(
            packed, rows=other_rows, phase=HiddenPhase.DECODE
        )

    with pytest.raises(ValueError, match="exact selection input"):
        @task(
            inputs={"rows": rows, "selection": packed},
            effects=(read(ordinary),),
        )
        def wrong_effect(
            rows: PackedPhaseRowBatch,
            selection: RequestStateSelectionBatch,
        ) -> None:
            _ = rows, selection

    foreign_selection = packed_selection(
        ordinary, rows=other_rows, phase=HiddenPhase.DECODE
    )
    with pytest.raises(ValueError, match="same-task packed request rows"):
        @task(
            inputs={"rows": rows, "selection": foreign_selection},
            effects=(read(foreign_selection),),
        )
        def wrong_rows(
            rows: PackedPhaseRowBatch,
            selection: RequestStateSelectionBatch,
        ) -> None:
            _ = rows, selection


def test_packed_hidden_rejects_non_block_output() -> None:
    rows = packed_request_rows()
    with pytest.raises(ValueError, match="block_output"):
        packed_hidden(0, site="post", rows=rows)
