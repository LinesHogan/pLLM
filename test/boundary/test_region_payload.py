"""An atomic resource can contain heterogeneous data without splitting effects."""

from dataclasses import replace

import pytest
import torch

from pllm.contracts.region_program import (
    AffineIndex,
    Call,
    Coordinate,
    InvalidProgram,
    Program,
    Publish,
    Resource,
    ScalarType,
    Scan,
    Snapshot,
    Value,
    verify,
)
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_compilation import validate_region_program
from test.boundary.test_region_control import bind
from test.boundary.test_region_pages import owner_fixture
from test.boundary.test_region_program import region
from test.region_reference import interpret

X = ScalarType.INDEX


def payload_program(cell, rounds):
    types = cell.declaration.inputs

    def values(prefix):
        return tuple(Value(f"{prefix}{i}", type_) for i, type_ in enumerate(types))

    old, new, initial, final = (values(prefix) for prefix in ("old", "new", "initial", "final"))
    body = region(
        (Value("i", X),),
        (
            Coordinate((), (Value("zero", X),), AffineIndex(0)),
            Snapshot(("zero",), (*old,), "layers"),
            Call(tuple(v.name for v in old), new, cell.digest),
            Publish((*tuple(v.name for v in new), "zero"), (), "layers"),
        ),
        (),
    )
    return Program(
        region(
            (),
            (
                Coordinate((), (Value("zero", X),), AffineIndex(0)),
                Snapshot(("zero",), (*initial,), "layers"),
                Scan((), (), rounds, body),
                Snapshot(("zero",), (*final,), "layers"),
            ),
            (*tuple(v.name for v in (*initial, *final)),),
        ),
        (cell.declaration,),
        (Resource("layers", types, (2,)),),
    )


def configure_payload(execution, task, rounds):
    specs = execution.arena.plan.tensor_layout_request.layouts[0].layout.fields
    device = execution.arena.device
    examples = tuple(
        torch.zeros(s.shape, dtype=torch.int64 if i == 3 else torch.float32, device=device) for i, s in enumerate(specs)
    )

    def update(w, column, row, step):
        column = column + 1
        row = row + 2
        return w + column + row.T, column, row, step + 1

    cell = ExportedCell.capture(
        update, ("w", "column", "row", "step"), ("next_w", "next_column", "next_row", "next_step"), examples
    )
    program = payload_program(cell, rounds)
    assert decode_boundary(encode_boundary(program), Program) == program
    verify(program)
    validate_region_program(program)
    task.program, task.region = program, bind(program, {cell.digest: cell})
    task.fields = tuple(LayoutFieldId(i) for i in range(len(specs)))
    return cell


@pytest.mark.parametrize("cuda", [False, True])
def test_heterogeneous_fields_and_counter_publish_atomically(cuda):
    if cuda and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    stream = torch.cuda.Stream() if cuda else None
    execution, task, rows, refs = owner_fixture(rounds=1, stream=stream)
    cell = configure_payload(execution, task, 5)
    final = execution.step(rows, stream)
    if stream is not None:
        torch.cuda.current_stream().wait_stream(stream)
    # Independent closed form: each step adds 3*k to every weight element.
    initial, actual = task.result.values[:4], task.result.values[4:]
    for tensor, number in zip(initial, (1, 0, 0, 0), strict=True):
        torch.testing.assert_close(tensor, torch.full_like(tensor, number), rtol=0, atol=0)
    for tensor, number in zip(actual, (46, 5, 10, 5), strict=True):
        torch.testing.assert_close(tensor, torch.full_like(tensor, number), rtol=0, atol=0)
    assert actual[3].dtype is torch.int64 and actual[3].shape == ()
    assert tuple(ref.version_id for ref in final) == (5, 0)
    assert len(task.context.page_publications) == 5  # One commit per payload, not per field.
    assert all(len(receipt.after_refs) == 1 for receipt in task.context.page_publications)
    if not cuda:
        from pllm.execution.region_values import Bank

        oracle_fields = tuple(
            Bank(
                (
                    torch.full(s.shape, 1 if i == 0 else 0, dtype=s.dtype),
                    torch.full(s.shape, 2 if i == 0 else 0, dtype=s.dtype),
                )
            )
            for i, s in enumerate(cell.input_signatures)
        )
        expected = interpret(
            task.program, {}, resources={"layers": oracle_fields}, functions={cell.digest: lambda args: cell(*args)}
        )
        for got, want in zip(task.result.values, expected.values, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
    execution.retire(tuple(row.identity for row in rows))
    execution.arena.collect_completed(wait=True)
    assert execution.arena.active_lineages == 0
    assert execution.arena.usage().pending_receipts == 0


def test_partial_payload_copy_failure_does_not_publish_some_fields(monkeypatch):
    execution, task, rows, refs = owner_fixture(rounds=1)
    configure_payload(execution, task, 3)
    original = execution.arena.write_field
    copied = []

    def fail(token, root, field):
        if field == 2:
            raise RuntimeError("injected third-field failure")
        copied.append(field)
        return original(token, root, field)

    monkeypatch.setattr(execution.arena, "write_field", fail)
    invocation = execution.start_step(rows)
    with pytest.raises(RuntimeError, match="third-field"):
        execution.execute_anchor(invocation, execution.plan.tasks[0].anchor_id)
    assert copied == [0, 1]
    assert not invocation.context.page_publications and not invocation.context.page_writes
    assert tuple(execution.arena.latest(ref.lineage_id) for ref in refs) == refs
    pin = execution.arena.pin_exact(refs)
    for value, expected in zip(execution.arena.snapshot(pin, refs[0]), (1, 0, 0, 0), strict=True):
        torch.testing.assert_close(value, torch.full_like(value, expected), rtol=0, atol=0)
    execution.arena.wait_release(execution.arena.release_pin(pin))
    execution.retire(tuple(row.identity for row in rows))
    assert execution.arena.active_lineages == 0


def test_payload_signature_rejects_missing_fields_and_wrong_types():
    execution, task, rows, _ = owner_fixture(rounds=1)
    configure_payload(execution, task, 2)
    program = task.program
    scan = program.entry.block.operations[2]
    publish = scan.body.block.operations[-1]
    for altered in (
        replace(publish, inputs=publish.inputs[1:]),
        replace(publish, inputs=(publish.inputs[3], *publish.inputs[1:])),
    ):
        changed_body = replace(
            scan.body, block=replace(scan.body.block, operations=(*scan.body.block.operations[:-1], altered))
        )
        changed_scan = replace(scan, body=changed_body)
        changed = replace(
            program,
            entry=replace(
                program.entry,
                block=replace(
                    program.entry.block,
                    operations=(*program.entry.block.operations[:2], changed_scan, program.entry.block.operations[3]),
                ),
            ),
        )
        with pytest.raises(ValueError, match="signature"):
            validate_region_program(changed)
    changed = replace(program, resources=(Resource("layers", (), (2,)),))
    with pytest.raises(ValueError):
        validate_region_program(changed)
    execution.retire(tuple(row.identity for row in rows))


def test_reference_effect_adapter_updates_multiple_payload_fields_together():
    from pllm.contracts.region_program import Function

    i = ScalarType.INT
    program = Program(
        region(
            (),
            (
                Call((), (), "advance"),
                Snapshot((), (Value("a", i), Value("b", i)), "pair"),
            ),
            ("a", "b"),
        ),
        (Function("advance", (), (), ("pair",)),),
        (Resource("pair", (i, i)),),
    )
    result = interpret(
        program, {}, resources={"pair": (3, 5)}, functions={"advance": lambda args: (args[0] + 1, args[1] + 2)}
    )
    assert result.values == (4, 7)
    assert result.resources == {"pair": (4, 7)}
    assert [event.kind for event in result.effects] == ["advance", "snapshot"]


def test_frontend_rejects_untyped_payload_members_before_lowering():
    entry = region((), (), ())
    for payload in (("int",), ScalarType.INT):
        with pytest.raises(InvalidProgram, match="payload|value type"):
            verify(Program(entry, resources=(Resource("state", payload),)))
