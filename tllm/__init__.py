"""Public tLLM API and the sole workload routing boundary."""

from __future__ import annotations


def make_llm(*args: object, **kwargs: object) -> object:
    workloads = kwargs.pop("workloads", None)
    if workloads is not None:
        if not isinstance(workloads, (tuple, list)):
            raise TypeError("make_llm.workloads must be a tuple or list")
        workload_items = tuple(workloads)
        if not workload_items:
            raise TypeError("canonical make_llm requires one workload")
        from tllm.canonical.workloads.base import CanonicalWorkload

        if not all(
            isinstance(workload, CanonicalWorkload) for workload in workload_items
        ):
            raise TypeError("legacy make_llm(workloads=...) is retired")
        return _make_canonical_llm(args, kwargs, workload_items)
    if any(name.startswith("workload_") for name in kwargs):
        raise TypeError("legacy workload compiler options are retired")
    from tllm.runtime import residual_runtime

    return residual_runtime.make_llm(*args, **kwargs)


def _make_canonical_llm(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    workloads: tuple[object, ...],
) -> object:
    """The sole dynamic coexistence boundary; canonical internals stay closed."""

    from tllm.canonical_vllm.runtime import (
        CanonicalVllmConfig,
        create_canonical_vllm_llm,
    )
    if len(workloads) != 1:
        raise TypeError("canonical vLLM currently supports one workload")
    workload = workloads[0]
    if args or set(kwargs) != {"canonical_config"}:
        raise TypeError(
            "canonical make_llm requires only canonical_config=CanonicalVllmConfig(...)"
        )
    config = kwargs["canonical_config"]
    if type(config) is not CanonicalVllmConfig:
        raise TypeError("canonical_config must be an exact CanonicalVllmConfig")
    return create_canonical_vllm_llm(workload, config)


def __getattr__(name: str) -> object:
    raise AttributeError(f"module 'tllm' has no attribute {name!r}")


__all__ = [
    "make_llm",
]
