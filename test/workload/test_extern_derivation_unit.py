"""extern() type-derivation golden: annotation-derived signed types must
reproduce every existing explicit extern declaration byte-for-byte, so
migrating consumers to extern() cannot drift any signed artifact.

Scope: consumers migrated to the kit surface (dummy, esamp, ttrl, reft,
override).  Recirculation is deliberately excluded — E4 rewrites it onto
the new surface (its legacy () output convention would derive ("None",)).
"""

from __future__ import annotations

import inspect
import typing
import unittest

import torch

from tllm.workload import (
    definition_for_workload_class,
    result,
    task,
    task_declaration,
)
from tllm.workload.frontend import ExternImplementation, _annotation_type_name, extern


def _extern_tasks(cls: type) -> list[tuple[str, object, object]]:
    definition = definition_for_workload_class(cls)
    assert definition is not None
    return [
        (entry.name, entry.executor, entry.declaration)
        for entry in definition.tasks
        if entry.declaration.implementations and entry.generated_after is None
    ]


class ExternDerivationGoldenTest(unittest.TestCase):
    def test_derived_types_match_every_existing_declaration(self) -> None:
        from tllm.consumers.dummy.workload import DummyWorkload
        from tllm.consumers.esamp.workload import ESampWorkload
        from tllm.consumers.override.workload import OverRIDEWorkload
        from tllm.consumers.reft.workload import ReftWorkload
        from tllm.consumers.ttrl.workload import TTRLWorkload

        checked = 0
        for cls in (
            DummyWorkload,
            ESampWorkload,
            OverRIDEWorkload,
            ReftWorkload,
            TTRLWorkload,
        ):
            for task_name, member, decl in _extern_tasks(cls):
                hints = typing.get_type_hints(member)
                parameters = [
                    name
                    for name in inspect.signature(member).parameters
                    if name not in ("self", "cls")
                ]
                derived_inputs = tuple(
                    _annotation_type_name(hints[name], f"{task_name} input {name!r}")
                    for name, _ in decl.inputs
                )
                derived_outputs = (
                    _annotation_type_name(hints["return"], f"{task_name} return"),
                )
                self.assertEqual(
                    tuple(name for name, _ in decl.inputs),
                    tuple(parameters),
                    f"{cls.__name__}.{task_name}: extern parameters must all be inputs",
                )
                for impl in decl.implementations:
                    self.assertEqual(
                        impl.input_types,
                        derived_inputs,
                        f"{cls.__name__}.{task_name} {impl.key} input_types drift",
                    )
                    self.assertEqual(
                        impl.output_types,
                        derived_outputs,
                        f"{cls.__name__}.{task_name} {impl.key} output_types drift",
                    )
                    checked += 1
        # Generated optimizer/publish transitions are definition-owned entries,
        # not class methods. This test covers only author-declared extern tasks.
        self.assertGreaterEqual(checked, 3)


class ExternConstructorTest(unittest.TestCase):
    def test_extern_key_parsing(self) -> None:
        impl = extern("tllm.esamp:adamw_transition@1")
        self.assertEqual(impl.namespace, "tllm.esamp")
        self.assertEqual(impl.name, "adamw_transition")
        self.assertEqual(impl.version, "1")
        self.assertTrue(impl.derive_types)
        self.assertEqual(extern("a:b@2").version, "2")
        self.assertEqual(extern("a:b").version, "1")

    def test_extern_key_validation(self) -> None:
        for bad in ("", "no-namespace", ":name@1", "ns:@1"):
            with self.assertRaises((TypeError, ValueError), msg=bad):
                extern(bad)
        with self.assertRaises(TypeError):
            extern(1)  # type: ignore[arg-type]

    def test_extern_rejects_explicit_types(self) -> None:
        with self.assertRaises(ValueError):
            ExternImplementation(
                namespace="a", name="b", input_types=("int",), derive_types=True
            )

    def test_task_derives_types_into_declaration(self) -> None:
        @task(
            inputs={"loss": result("train", "loss")},
            implementation=extern("tllm.test:derived@1"),
        )
        def update(loss: torch.Tensor) -> None:
            del loss

        decl = task_declaration(update)
        (impl,) = decl.implementations
        self.assertFalse(impl.derive_types)
        self.assertEqual(impl.input_types, ("Tensor",))
        self.assertEqual(impl.output_types, ("None",))
        self.assertEqual(impl.key, "tllm.test:derived@1")
        # The derived implementation lowers to IR cleanly.
        ir = impl.to_ir()
        self.assertEqual(ir.input_types, ("Tensor",))

    def test_underived_implementation_cannot_lower(self) -> None:
        with self.assertRaises(RuntimeError):
            extern("a:b").to_ir()

    def test_extern_requires_all_parameters_declared(self) -> None:
        with self.assertRaises(TypeError):
            @task(
                inputs={"loss": result("other", "loss")},
                implementation=extern("tllm.test:undeclared@1"),
            )
            def update(loss: torch.Tensor, extra: int) -> None:
                del loss, extra


if __name__ == "__main__":
    unittest.main()
