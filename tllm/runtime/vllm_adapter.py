"""Quarantined dynamic adapter for the external vLLM object graph."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, cast
import weakref


class VllmHandle(Protocol):
    def generate(self, *args: Any, **kwargs: Any) -> Any: ...


_runner_owners: "weakref.WeakKeyDictionary[Any, object]" = weakref.WeakKeyDictionary()


def bind_runner_owner(runner: object, owner: object) -> None:
    _runner_owners[cast(Any, runner)] = owner


def create_llm(args: tuple[object, ...], kwargs: dict[str, object]) -> VllmHandle:
    from tllm.runtime import residual_runtime

    factory = cast(Any, residual_runtime.make_llm)
    return cast(VllmHandle, factory(*args, **kwargs))


def bind_inprocess_runner(llm: object, owner: object) -> None:
    dynamic = cast(Any, llm)
    try:
        driver = dynamic.llm_engine.engine_core.model_executor.driver_worker
        runner = driver.model_runner
    except AttributeError:
        return
    bind_runner_owner(runner, owner)


def install_generate_poll(
    llm: object,
    owner: object,
    before_generate: Callable[[], None],
) -> None:
    dynamic = cast(Any, llm)
    installed_owner: object | None = getattr(
        dynamic, "_tllm_shape_specialization_poll_owner", None
    )
    if installed_owner is owner:
        return
    if installed_owner is not None:
        raise RuntimeError(
            "LLM already has a shape-specialization poll owned by another runtime"
        )
    generate: Any = getattr(dynamic, "generate", None)
    if generate is None:
        return

    def generate_after_shape_poll(*args: Any, **kwargs: Any) -> Any:
        before_generate()
        return generate(*args, **kwargs)

    dynamic._tllm_shape_specialization_poll_owner = owner
    dynamic.generate = generate_after_shape_poll


__all__ = [
    "VllmHandle",
    "bind_inprocess_runner",
    "bind_runner_owner",
    "create_llm",
    "install_generate_poll",
]
