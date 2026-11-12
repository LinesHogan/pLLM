"""Workload identity admits standard names and literal lookups while rejecting opaque namespaces."""

import builtins
import collections.abc
import sys
import typing

import pytest

import pllm
from pllm.backends.vllm.prefix_cache import CachePolicy
from pllm.contracts.codec import boundary_digest
from pllm.execution.freeze import fingerprint
from pllm.testing_contracts import contract_facts
from test.workloads.kernel_workloads import Residual, add_kernel


@pytest.mark.parametrize(
    "expression",
    (
        'getattr(x, "dtype")',
        'getattr(x, "dtype", x.default)',
        'getattr(x if x else name, "dtype")',
        'getattr(x.method(1, 2), "dtype")',
        'getattr(x, "dtype", len(x.default))',
        'getattr(x, "dtype", 1 if x else 2)',
        'hasattr(x, "dtype")',
        "vars(x)",
        "x.ops",
        '__import__("math")',
        '__import__("typing")',
        '__import__("collections.abc")',
        '__import__("builtins")',
        '[getattr(item, "dtype") for item in x]',
    ),
)
def test_literal_lookups_have_stable_identity(expression):
    function = eval(f"lambda x, name: {expression}")
    assert fingerprint(function) == fingerprint(eval(f"lambda x, name: {expression}"))
    if '"dtype"' in expression:
        changed = expression.replace('"dtype"', '"shape"')
        assert fingerprint(function) != fingerprint(eval(f"lambda x, name: {changed}"))


@pytest.mark.parametrize(
    "expression,replacement",
    (
        ("getattr(x, name)", 'getattr(obj, "field")'),
        ('getattr(x, "dtype" + name)', 'getattr(obj, "field")'),
        ('getattr(x, name, "dtype")', 'getattr(obj, "field")'),
        ("getattr(x.method(1, 2), name)", 'getattr(obj, "field")'),
        ("hasattr(x, name)", 'hasattr(obj, "field")'),
        ("vars()", "vars(obj)"),
        ("globals()", "config fields"),
        ("locals()", "local variables directly"),
        ('eval("x")', "Python helper function"),
        ('exec("x = 1")', "Python helper function"),
        ("__import__(name)", "import torch, math or pllm at module scope"),
        ('__import__("os")', "import torch, math or pllm at module scope"),
        ("[getattr(item, name) for item in x]", 'getattr(obj, "field")'),
    ),
)
def test_hidden_dependencies_explain_the_replacement(expression, replacement):
    with pytest.raises(ValueError) as error:
        fingerprint(eval(f"lambda x, name: {expression}"))
    assert replacement in str(error.value)


@pytest.mark.parametrize(
    "dependency",
    (typing.Mapping, typing.Any, typing.cast, collections.abc.Mapping, list, len),
)
@pytest.mark.parametrize("body", ("return dependency", "def inner(value: dependency): return value\n return inner"))
def test_standard_names_in_runtime_and_nested_annotations(dependency, body):
    assert fingerprint(dependency) == boundary_digest(
        dependency.__module__, dependency.__qualname__, sys.version_info[:3]
    )
    namespace = {"dependency": dependency}
    exec("def outer():\n " + body, namespace)
    identity = fingerprint(namespace["outer"])
    assert identity == fingerprint(namespace["outer"])
    namespace["dependency"] = tuple
    assert identity != fingerprint(namespace["outer"])


@pytest.mark.parametrize("module,member", ((typing, "Mapping"), (collections.abc, "Mapping"), (builtins, "list")))
def test_standard_module_names(module, member):
    assert fingerprint(module) == boundary_digest(module.__name__, sys.version_info[:3])
    assert fingerprint(eval(f"lambda: dependency.{member}", {"dependency": module}))


def test_opaque_namespace_explains_config_boundary():
    with pytest.raises(ValueError, match="sys:.*compute other values outside the workload"):
        fingerprint(eval("lambda: namespace.version", {"namespace": sys}))


def test_triton_source_changes_workload_and_cache_identity():
    config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct")
    workload = Residual()
    _, facts = contract_facts(workload, config)
    before = workload.build(facts).graph.execution_parameters_digest
    attachment = workload.compile_backend(facts).attachment
    cache = CachePolicy.for_attachment(attachment, "reuse").namespace
    source = add_kernel.src
    try:
        add_kernel._unsafe_update_src(source.replace("value + branch", "value - branch"))
        assert workload.build(facts).graph.execution_parameters_digest != before
        assert CachePolicy.for_attachment(workload.compile_backend(facts).attachment, "reuse").namespace != cache
    finally:
        add_kernel._unsafe_update_src(source)
