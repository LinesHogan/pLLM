"""Regression tests for the generic Region authority gate."""

from __future__ import annotations

from pathlib import Path

from tools.check_region_authority import (
    _line_violations,
    _native_binding_violations,
    _recurrence_hot_path_violations,
    collect,
)


def _rules(source: str) -> tuple[str, ...]:
    return tuple(
        violation.rule
        for violation in _line_violations(Path("tllm/runtime/workload_execution.py"), source)
    )


def test_current_region_authority_surface_is_clean() -> None:
    assert collect() == []


def test_gate_rejects_consumer_and_physical_redecision_leaks() -> None:
    rules = _rules(
        "from tllm.consumers.esamp import Consumer\n"
        "if region.compile_safe:\n"
        "    pass\n"
        "policy = options.compile_policy\n"
        "for task in region.tasks:\n"
        "    pass\n"
        "adapter = declaration.instance.legacy_consumer\n"
    )
    assert rules == (
        "consumer authority leaked into Region core",
        "physical path may not re-decide compile safety",
        "physical path may not read raw compile policy",
        "physical path may not inspect Region task membership",
        "native Region path may not inspect legacy adapter ownership",
    )


def test_gate_rejects_executor_grammar_and_eager_retry() -> None:
    rules = _rules(
        'namespace = implementation_id.split(":", 1)\n'
        "try:\n"
        "    torch.compile(region_callable)\n"
        "except RuntimeError:\n"
        "    eager(region_callable)\n"
    )
    assert "physical path may not parse executor strings" in rules
    assert "physical path may not silently retry with eager" in rules


def test_gate_rejects_materialization_refcount_and_native_binding_redecision() -> None:
    assert "physical path may not derive materialization refcounts" in _rules(
        "uses = len(binding.consumer_regions)\n"
    )
    rules = tuple(
        violation.rule
        for violation in _native_binding_violations(
            Path("tllm/runtime/mp/worker_extension.py"),
            "if region.compile_safe:\n"
            "    pass\n"
            "uses = len(binding.consumer_regions)\n",
        )
    )
    assert rules == (
        "native binding may not re-decide compile safety",
        "native binding may not derive materialization refcounts",
    )


def test_gate_rejects_sampler_policy_inference_from_workload_config() -> None:
    rules = tuple(
        violation.rule
        for violation in _line_violations(
            Path("tllm/backends/vllm/sampler_mrv1.py"),
            'backend = getattr(provider.config, "sampler_backend")\n',
        )
    )
    assert rules == (
        "sampler bridge may not infer policy from workload config",
    )


def test_rust_test_fixture_is_not_treated_as_production_authority() -> None:
    assert _line_violations(
        Path("rust/tllm_core/src/schedule.rs"),
        "fn production() {}\n#[cfg(test)]\nfn esamp_fixture() {}\n",
    ) == []


def test_gate_rejects_recurrence_compile_and_bundle_parse_on_hot_paths() -> None:
    path = Path("tllm/consumers/recirculation/vllm_region_workload.py")
    violations = _recurrence_hot_path_violations(
        path,
        "def _stage_typed_recirculation(context):\n"
        "    graph = build_recirculation_graph(context)\n"
        "    artifact = _core.compile_typed_recurrence_plan(graph)\n"
        "    return json.loads(artifact)\n",
    )
    assert tuple(violation.rule for violation in violations) == (
        "native recurrence hot path may not compile or rebuild an artifact",
        "native recurrence hot path may not compile or rebuild an artifact",
        "native recurrence hot path may not parse an artifact bundle",
    )


def test_gate_allows_explicit_cold_recurrence_compilation() -> None:
    path = Path("tllm/consumers/recirculation/vllm_region_workload.py")
    assert _recurrence_hot_path_violations(
        path,
        "def compile_cold_artifacts(spec):\n"
        "    return _core.compile_typed_recurrence_stage_artifact_json(spec)\n",
    ) == []


def test_gate_rejects_python_reconstruction_of_observed_stage_facts() -> None:
    path = Path("tllm/consumers/recirculation/vllm_region_workload.py")
    violations = _recurrence_hot_path_violations(
        path,
        "def _stage_typed_recirculation(row_plan, artifact):\n"
        "    if any(phase != 'prefill' for phase in row_plan.phases):\n"
        "        raise RuntimeError\n"
        "    facts = _bound_stage_facts(row_count=2)\n"
        "    return _core.bind_typed_recurrence_stage(artifact, facts)\n",
    )
    assert {violation.rule for violation in violations} == {
        "native recurrence hot path may not infer admission from row phases",
        "native recurrence hot path may not compile or rebuild an artifact",
    }
