"""Focused contracts for compiler-specific import-layer bridges."""

from tools.check_import_layers import RULES, _collect, _is_allowed_target


def test_workload_and_schedule_only_allow_their_declared_bridge() -> None:
    assert RULES["schedule"] == frozenset()
    assert RULES["workload"] == frozenset()
    assert _is_allowed_target(
        "workload",
        "tllm.workload.lowering",
        "schedule",
        "tllm.schedule.contracts",
    )
    assert _is_allowed_target(
        "schedule",
        "tllm.schedule.coordinator",
        "workload",
        "tllm.workload.lowering",
    )
    assert not _is_allowed_target(
        "workload",
        "tllm.workload.frontend",
        "schedule",
        "tllm.schedule.compiler",
    )
    assert not _is_allowed_target(
        "schedule",
        "tllm.schedule.compiler",
        "runtime",
        "tllm.runtime.workload_execution",
    )


def test_runtime_can_consume_but_not_become_compiler_authority() -> None:
    assert _is_allowed_target(
        "runtime",
        "tllm.runtime.workload_execution",
        "schedule",
        "tllm.schedule.contracts",
    )
    assert not _is_allowed_target(
        "workload",
        "tllm.workload.region_compiler",
        "runtime",
        "tllm.runtime.workload_execution",
    )


def test_compiler_packages_have_no_unapproved_cross_package_imports() -> None:
    violations = _collect()
    assert not [
        violation
        for violation in violations
        if violation.importer in {"schedule", "workload"}
    ]
