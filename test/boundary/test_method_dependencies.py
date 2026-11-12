"""Workload methods call custom ops through their op objects and keep state out of setattr."""

import pytest
import torch

from pllm.execution.method_dependencies import check_dependencies
from test.workloads import kernel_workloads  # noqa: F401  (registers kernel_test::add)


@pytest.mark.parametrize(
    "expression,replacement",
    (
        ("setattr(x, name, 1)", "return the value instead"),
        ('setattr(x, "dtype", name)', "declare pllm.RequestState/pllm.InstanceState"),
        ('lambda: setattr(x, "dtype", name)', "return a state update"),
    ),
)
def test_setattr_explains_the_replacement(expression, replacement):
    with pytest.raises(ValueError) as error:
        check_dependencies(eval(f"lambda x, name: {expression}"))
    assert replacement in str(error.value)


@pytest.mark.parametrize(
    "expression",
    (
        "torch.ops.mylab.add(x)",
        'getattr(torch, "ops").mylab.add(x)',
        "torch.ops.mylab",
        "op(x)",
        "namespace.add(x)",
    ),
)
def test_dispatcher_lookup_needs_the_op_object(expression):
    namespace = {"torch": torch, "op": torch.ops.kernel_test.add, "namespace": torch.ops.kernel_test}
    with pytest.raises(ValueError, match="call the op object.*receive the op"):
        check_dependencies(eval(f"lambda x: {expression}", namespace))
