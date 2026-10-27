"""Warn once per prepared tensor implementation about numerical transformations."""

import warnings
from threading import Lock

_warned: set[str] = set()
_warning_lock = Lock()


class NumericalTransformWarning(UserWarning):
    """The selected plan may change floating-point rounding, not algorithm semantics."""


def warn_numerical_transforms(signature: str, transformations: tuple[str, ...]) -> None:
    if not transformations:
        return
    with _warning_lock:
        if signature in _warned:
            return
        _warned.add(signature)
    warnings.warn(
        "pLLM selected "
        + ", ".join(transformations)
        + "; these transformations may change floating-point results, including rounding and overflow. "
        "Set ExecutionOptions(allow_numerical_transforms=False) to retain the original numerical path.",
        NumericalTransformWarning,
        stacklevel=3,
    )
