"""Numerical transformation notices for selected execution implementations."""

import warnings

from pllm.execution.numerics import NumericalTransformWarning, warn_numerical_transforms


def test_numerical_warning_requires_selected_changes_and_describes_opt_out():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_numerical_transforms("test-no-change", ())
        assert not caught
        warn_numerical_transforms("test-selected-fusion", ("reduction fusion",))
        warn_numerical_transforms("test-selected-fusion", ("reduction fusion",))
    assert len(caught) == 1
    assert caught[0].category is NumericalTransformWarning
    assert "reduction fusion" in str(caught[0].message)
    assert "allow_numerical_transforms=False" in str(caught[0].message)
